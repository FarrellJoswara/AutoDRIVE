"""In-process TelemetryBus + WebSocket fan-out with latest-state fleet delivery."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set

from fastapi import WebSocket


@dataclass
class TelemetryBus:
    """Publish/subscribe spine — in-memory now; Redis-swappable later."""

    last_metrics: Optional[Dict[str, Any]] = None
    last_fleet: Optional[Dict[str, Any]] = None
    last_train_phase: Optional[Dict[str, Any]] = None
    last_evaluator_live: Optional[Dict[str, Any]] = None
    last_replay_fleet: Optional[Dict[str, Any]] = None
    last_replay_status: Optional[Dict[str, Any]] = None
    metrics_ring: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=600))
    # Official training streams are isolated by run. The legacy scalar fields
    # remain as a compatibility view for non-training diagnostics.
    run_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_fleets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    run_evaluators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _clients: Set["WSClient"] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Thread-safe publish (API handlers + TrainJob watcher thread)."""
        event = {"type": event_type, "payload": payload}
        with self._lock:
            if event_type == "telemetry":
                kind = payload.get("kind", "metrics")
                if kind == "fleet":
                    self.last_fleet = payload
                    self._store_run_payload(self.run_fleets, payload)
                elif kind == "phase":
                    self.last_train_phase = payload
                    self._store_run_payload(self.run_phases, payload)
                else:
                    self.last_metrics = payload
                    self.metrics_ring.append(payload)
                    self._store_run_payload(self.run_metrics, payload)
            elif event_type == "replay_telemetry" and payload.get("kind") == "fleet":
                self.last_replay_fleet = payload
            elif event_type == "replay_status":
                self.last_replay_status = payload
            elif event_type == "train_phase":
                self.last_train_phase = payload
                self._store_run_payload(self.run_phases, payload)
            elif event_type == "evaluator_live":
                self.last_evaluator_live = payload
                self._store_run_payload(self.run_evaluators, payload)
            clients = list(self._clients)
            loop = self._loop

        def _fanout() -> None:
            for client in clients:
                client.enqueue(event)

        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(_fanout)
                return
            except RuntimeError:
                pass
        _fanout()

    @staticmethod
    def _store_run_payload(
        destination: Dict[str, Dict[str, Any]], payload: Dict[str, Any]
    ) -> None:
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            destination[run_id] = payload

    def run_snapshot(self, run_id: str) -> Dict[str, Any]:
        """Get latest official training telemetry without crossing run streams."""
        with self._lock:
            return {
                "run_id": run_id,
                "last_metrics": self.run_metrics.get(run_id),
                "last_fleet": self.run_fleets.get(run_id),
                "last_train_phase": self.run_phases.get(run_id),
                "last_evaluator_live": self.run_evaluators.get(run_id),
            }

    def clear_run(self, run_id: str) -> None:
        """Clear only one run's in-memory latest values."""
        with self._lock:
            self.run_metrics.pop(run_id, None)
            self.run_fleets.pop(run_id, None)
            self.run_phases.pop(run_id, None)
            self.run_evaluators.pop(run_id, None)

    def publish_status(self, payload: Dict[str, Any]) -> None:
        self.publish("status", payload)

    def publish_train_phase(self, payload: Dict[str, Any]) -> None:
        self.publish("train_phase", payload)

    def publish_evaluator_live(self, payload: Dict[str, Any]) -> None:
        self.publish("evaluator_live", payload)

    def publish_replay_status(self, payload: Dict[str, Any]) -> None:
        self.publish("replay_status", payload)

    def clear_train_phase(self) -> None:
        with self._lock:
            self.last_train_phase = None

    def clear_train_telemetry(self) -> None:
        """Drop the prior run's live values before a new run starts."""
        with self._lock:
            self.last_metrics = None
            self.last_fleet = None
            self.last_train_phase = None
            self.last_evaluator_live = None
            self.metrics_ring.clear()
            self.run_metrics.clear()
            self.run_fleets.clear()
            self.run_phases.clear()
            self.run_evaluators.clear()

    def attach(self, client: "WSClient") -> None:
        with self._lock:
            self._clients.add(client)

    def detach(self, client: "WSClient") -> None:
        with self._lock:
            self._clients.discard(client)

    def snapshot_for_client(self) -> List[Dict[str, Any]]:
        """Events to send a newly connected client for context."""
        out: List[Dict[str, Any]] = []
        with self._lock:
            if self.last_metrics is not None:
                out.append({"type": "telemetry", "payload": self.last_metrics})
            if self.last_fleet is not None:
                out.append({"type": "telemetry", "payload": self.last_fleet})
            if self.last_train_phase is not None:
                out.append({"type": "train_phase", "payload": self.last_train_phase})
            if self.last_evaluator_live is not None:
                out.append({"type": "evaluator_live", "payload": self.last_evaluator_live})
            if self.last_replay_fleet is not None:
                out.append({"type": "replay_telemetry", "payload": self.last_replay_fleet})
            if self.last_replay_status is not None:
                out.append({"type": "replay_status", "payload": self.last_replay_status})
            # Reconnects get the latest frame for every active run; the client
            # selects by run_id and never treats these as one shared fleet.
            out.extend({"type": "telemetry", "payload": p} for p in self.run_metrics.values())
            out.extend({"type": "telemetry", "payload": p} for p in self.run_fleets.values())
            out.extend({"type": "train_phase", "payload": p} for p in self.run_phases.values())
            out.extend({"type": "evaluator_live", "payload": p} for p in self.run_evaluators.values())
        return out


class WSClient:
    """One browser WebSocket with latest-state fleet delivery.

    Fleet frames are complete snapshots, so only the newest pending snapshot
    for each stream is useful. Other events stay ordered in the event queue.
    """

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._queue: Deque[Dict[str, Any]] = deque()
        self._latest_fleet: Dict[str, Dict[str, Any]] = {}
        self._wake = asyncio.Event()
        self._alive = True
        self.fleet_frames_coalesced = 0

    def enqueue(self, event: Dict[str, Any]) -> None:
        if not self._alive:
            return
        event_type = event.get("type", "")
        payload = event.get("payload")
        is_fleet = (
            event_type in {"telemetry", "replay_telemetry"}
            and isinstance(payload, dict)
            and payload.get("kind") == "fleet"
        )
        if is_fleet:
            if event_type in self._latest_fleet:
                self.fleet_frames_coalesced += 1
            self._latest_fleet[event_type] = event
        else:
            self._queue.append(event)
        self._wake.set()

    async def sender(self) -> None:
        try:
            while self._alive:
                await self._wake.wait()
                while self._alive:
                    if self._queue:
                        event = self._queue.popleft()
                    elif self._latest_fleet:
                        _, event = self._latest_fleet.popitem()
                    else:
                        self._wake.clear()
                        break
                    await self.ws.send_text(json.dumps(event, default=str))
        except Exception:
            self._alive = False
        finally:
            self._alive = False

    def close(self) -> None:
        self._alive = False
        self._wake.set()
