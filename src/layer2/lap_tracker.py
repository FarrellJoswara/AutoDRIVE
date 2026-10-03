"""Lap timing driven by Layer 2's validated, forward-only route progress."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional
from pathlib import Path

from .route_progress import RouteProgressTracker


class LapTracker:
    """Count complete centerline circuits from the episode's starting gate.

    The gate is the full-width corridor line at the projected reset pose. A lap
    completes when the route frontier advances one complete closed-route
    length beyond that starting projection. This keeps map geometry and
    crossing decisions out of the UI and avoids adding simulator-side events.
    """

    def __init__(
        self,
        route: Optional[RouteProgressTracker],
        gate_progress_m: Optional[float] = None,
    ) -> None:
        self.route = route
        self.supported = bool(route is not None and route.closed)
        self.length_m = float(route.length_m) if self.supported and route else 0.0
        self.gate_progress_m = gate_progress_m
        self.reset()

    def reset(self, progress: Optional[Dict[str, Any]] = None, now: float = 0.0) -> None:
        self.lap_count = 0
        self.last_lap_time_s: Optional[float] = None
        self.best_lap_time_s: Optional[float] = None
        self._start_progress: Optional[float] = None
        self._next_lap_progress: Optional[float] = None
        self._lap_started_at: Optional[float] = None
        self._lap_frontier_start_m: Optional[float] = None
        self._attempt_started_at: Optional[float] = None
        self._timing_started = False
        self._previous_progress: Optional[float] = None
        self._previous_time: Optional[float] = None
        self.lap_gate = None
        if not self.supported or progress is None:
            return

        start_progress = float(progress["progress_m"])
        self._start_progress = start_progress
        gate_s = start_progress
        if self.gate_progress_m is not None:
            configured = float(self.gate_progress_m)
            if math.isfinite(configured) and 0.0 <= configured < self.length_m:
                gate_s = configured
        distance_to_gate = (gate_s - (start_progress % self.length_m)) % self.length_m
        self._next_lap_progress = start_progress + distance_to_gate
        if distance_to_gate <= 0.05:
            self._next_lap_progress = start_progress + self.length_m
            self._lap_started_at = float(now)
            self._lap_frontier_start_m = start_progress
            self._attempt_started_at = float(now)
            self._timing_started = True
        self._previous_progress = start_progress
        self._previous_time = float(now)
        if self.route is not None:
            self.lap_gate = self.route.gate_line_at(gate_s)

    def update(self, progress_m: Optional[float], now: float) -> Dict[str, Any]:
        completed_frontier_speed = None
        completed_attempt_elapsed_s = None
        completed_attempt_average_frontier_speed_mps = None
        if self.supported and progress_m is not None and self._next_lap_progress is not None:
            progress = float(progress_m)
            stamp = float(now)
            previous_progress = self._previous_progress
            previous_time = self._previous_time

            # Interpolate the crossing time between telemetry samples, rather
            # than assigning the entire sample interval to the lap.
            while progress >= self._next_lap_progress:
                crossing_progress = self._next_lap_progress
                crossing_time = stamp
                if (
                    previous_progress is not None
                    and previous_time is not None
                    and progress > previous_progress
                ):
                    fraction = (crossing_progress - previous_progress) / (
                        progress - previous_progress
                    )
                    crossing_time = previous_time + max(0.0, min(1.0, fraction)) * (
                        stamp - previous_time
                    )

                if not self._timing_started:
                    # First crossing establishes the start time; only the next
                    # crossing completes a full lap.
                    self._timing_started = True
                    self._lap_started_at = crossing_time
                    self._lap_frontier_start_m = crossing_progress
                    self._attempt_started_at = crossing_time
                else:
                    lap_started_at = (
                        crossing_time
                        if self._lap_started_at is None
                        else self._lap_started_at
                    )
                    lap_time = max(0.0, crossing_time - lap_started_at)
                    self.lap_count += 1
                    self.last_lap_time_s = lap_time
                    frontier_start = (
                        crossing_progress - self.length_m
                        if self._lap_frontier_start_m is None
                        else self._lap_frontier_start_m
                    )
                    frontier_distance = max(0.0, crossing_progress - frontier_start)
                    completed_frontier_speed = (
                        frontier_distance / lap_time if lap_time > 0.0 else 0.0
                    )
                    self._lap_frontier_start_m = crossing_progress
                    if self._attempt_started_at is not None:
                        completed_attempt_elapsed_s = max(
                            0.0, crossing_time - self._attempt_started_at
                        )
                        completed_distance = self.lap_count * self.length_m
                        completed_attempt_average_frontier_speed_mps = (
                            completed_distance / completed_attempt_elapsed_s
                            if completed_attempt_elapsed_s > 0.0 else 0.0
                        )
                    if self.best_lap_time_s is None or lap_time < self.best_lap_time_s:
                        self.best_lap_time_s = lap_time
                    self._lap_started_at = crossing_time
                self._next_lap_progress += self.length_m

            self._previous_progress = progress
            self._previous_time = stamp
        result = self.sample(float(now))
        result["completed_lap_frontier_speed_mps"] = completed_frontier_speed
        result["completed_attempt_elapsed_s"] = completed_attempt_elapsed_s
        result["completed_attempt_average_frontier_speed_mps"] = (
            completed_attempt_average_frontier_speed_mps
        )
        return result

    def sample(self, now: float) -> Dict[str, Any]:
        elapsed = None
        if self.supported and self._timing_started and self._lap_started_at is not None:
            elapsed = max(0.0, float(now) - self._lap_started_at)
        return {
            "lap_supported": self.supported,
            "lap_count": int(self.lap_count),
            "last_lap_time_s": self.last_lap_time_s,
            "best_lap_time_s": self.best_lap_time_s,
            "lap_elapsed_s": elapsed,
            "lap_gate": self.lap_gate,
        }


def map_lap_gate_config(
    map_id: str,
    *,
    route: Optional[RouteProgressTracker] = None,
    repository_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve a map's custom lap gate, falling back to its saved spawn pose."""
    from .route_progress import map_centerline_path

    path = map_centerline_path(map_id, repository_root)
    if path is None:
        return {"supported": False, "reason": "centerline_missing"}
    if route is None:
        try:
            route = RouteProgressTracker.from_csv(path)
        except (OSError, ValueError):
            return {"supported": False, "reason": "centerline_invalid"}
    if not route.closed:
        return {
            "supported": False,
            "reason": "centerline_open",
            "route_length_m": float(route.length_m),
        }

    meta_path = path.parent / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}

    spawn = meta.get("spawn")
    default_progress = None
    if isinstance(spawn, dict):
        try:
            default_progress = route.project_pose(float(spawn["x"]), float(spawn["z"]))
        except (KeyError, TypeError, ValueError):
            default_progress = None
    if default_progress is None:
        return {
            "supported": False,
            "reason": "spawn_not_on_centerline",
            "route_length_m": float(route.length_m),
        }

    centerline_meta = meta.get("centerline")
    centerline_built_at = (
        centerline_meta.get("built_at") if isinstance(centerline_meta, dict) else None
    )
    saved = meta.get("lap_gate")
    selected_progress = float(default_progress)
    customized = False
    if (
        isinstance(saved, dict)
        and centerline_built_at
        and saved.get("centerline_built_at") == centerline_built_at
    ):
        try:
            candidate = float(saved["progress_m"])
            if math.isfinite(candidate) and 0.0 <= candidate < route.length_m:
                selected_progress = candidate
                customized = True
        except (KeyError, TypeError, ValueError):
            pass

    return {
        "supported": True,
        "reason": None,
        "route_length_m": float(route.length_m),
        "progress_m": selected_progress,
        "default_progress_m": float(default_progress),
        "customized": customized,
        "centerline_built_at": centerline_built_at,
        "gate_line": route.gate_line_at(selected_progress),
    }
