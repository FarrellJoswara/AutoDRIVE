"""In-process TelemetryBus + WebSocket fan-out (drop-oldest per client)."""

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
    metrics_ring: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=600))
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
                else:
                    self.last_metrics = payload
                    self.metrics_ring.append(payload)
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

    def publish_status(self, payload: Dict[str, Any]) -> None:
        self.publish("status", payload)

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
        return out


class WSClient:
    """One browser WebSocket with a bounded send queue (drop-oldest)."""

    def __init__(self, ws: WebSocket, maxsize: int = 8) -> None:
        self.ws = ws
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._alive = True

    def enqueue(self, event: Dict[str, Any]) -> None:
        if not self._alive:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Client too slow — mark for disconnect
                self._alive = False

    async def sender(self) -> None:
        try:
            while self._alive:
                event = await self._queue.get()
                await self.ws.send_text(json.dumps(event, default=str))
        except Exception:
            self._alive = False
        finally:
            self._alive = False

    def close(self) -> None:
        self._alive = False
