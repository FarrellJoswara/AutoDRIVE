"""Low-overhead periodic trajectory clips for comparing policy runs.

The recorder stores data, rather than simulator screenshots.  Samples are
queued to a daemon writer so a slow disk cannot delay the control loop.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrajectoryRecorder:
    """Write periodic JSONL clips from fleet telemetry.

    A clip starts immediately, lasts ``duration_s``, and the next clip starts
    after ``interval_s``.  Set interval or duration to zero to disable it.
    """

    def __init__(
        self,
        directory: Path,
        *,
        interval_s: float = 300.0,
        duration_s: float = 20.0,
        max_queue: int = 128,
    ) -> None:
        self.directory = Path(directory)
        self.interval_s = max(0.0, float(interval_s))
        self.duration_s = max(0.0, float(duration_s))
        self._next_start = 0.0
        self._clip_started = 0.0
        self._clip_index = 0
        self._path: Optional[Path] = None
        self._queue: queue.Queue = queue.Queue(maxsize=max(8, int(max_queue)))
        self._stop = object()
        self._thread = threading.Thread(target=self._write, name="trajectory-recorder", daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0.0 and self.duration_s > 0.0

    def submit(self, fleet: Dict[str, Any]) -> bool:
        """Queue one compact fleet sample; return whether it was accepted."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if self._path is None:
            if now < self._next_start:
                return False
            self._clip_index += 1
            self._clip_started = now
            self._path = self.directory / f"clip-{self._clip_index:06d}.jsonl"
            header = {
                "kind": "trajectory_clip",
                "format": 1,
                "run_id": fleet.get("run_id"),
                "runtime": fleet.get("runtime", "custom"),
                "clip": self._clip_index,
                "started_at": fleet.get("ts"),
            }
            self._put((self._path, header))
        if now - self._clip_started >= self.duration_s:
            self._next_start = now + max(0.0, self.interval_s - self.duration_s)
            self._path = None
            return False
        sample = {
            "kind": "sample",
            "ts": fleet.get("ts"),
            "step": fleet.get("step"),
            "episode": fleet.get("episode"),
            "cars": [
                {
                    key: car.get(key)
                    for key in (
                        "env_id", "pose", "yaw", "speed", "collision", "reset",
                        "lap_count", "lap_times_s", "last_lap_time_s",
                        "best_lap_time_s", "lap_elapsed_s",
                    )
                    if key in car
                }
                for car in fleet.get("cars", [])
            ],
        }
        self._put((self._path, sample))
        return True

    def _put(self, item: tuple[Path, Dict[str, Any]]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # A missed visualization sample is harmless; control must remain
            # responsive and the queue must never grow without bound.
            pass

    def close(self) -> None:
        try:
            self._queue.put(self._stop, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)

    def _write(self) -> None:
        handles: Dict[Path, Any] = {}
        try:
            while True:
                item = self._queue.get()
                if item is self._stop:
                    return
                path, payload = item
                handle = handles.get(path)
                if handle is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = path.open("a", encoding="utf-8")
                    handles[path] = handle
                handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
                handle.flush()
        except Exception:
            # Visualization evidence is best effort and must not affect a run.
            return
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
