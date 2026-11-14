"""Append compact, per-run fleet metrics for reward experiments."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_CAR_FIELDS = (
    "env_id",
    "reset",
    "speed",
    "speed_source",
    "v_long",
    "throttle_command",
    "steering_command",
    "collision",
    "collision_count",
    "frontier_line",
    "frontier_progress_m",
    "frontier_speed_mps",
    "current_progress_line",
    "current_progress_m",
    "signed_route_delta_m",
    "current_route_speed_mps",
    "route_projection_valid",
    "time_since_frontier_push_s",
    "reward_components",
    "neutral_fallback_count",
    "neutral_fallback_diagnostic",
    "command_expiry_diagnostic",
)


def _json_value(value: Any) -> Any:
    """Convert telemetry to JSON values without serializing sensor buffers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # Do not call arbitrary array conversion here; fleet samples can reference
    # large LiDAR/camera buffers, which are intentionally excluded.
    return None


class RewardExperimentMetricsRecorder:
    """Write one compact fleet snapshot per second to a run-local JSONL file."""

    def __init__(self, run_dir: Path, run_id: str, runtime: str) -> None:
        self.path = Path(run_dir) / "reward_experiment_metrics.jsonl"
        self.run_id = str(run_id)
        self.runtime = str(runtime)
        self._last_write = 0.0

    def record(self, fleet: Dict[str, Any]) -> bool:
        """Append a sanitized fleet sample; return whether a row was written."""
        now = time.monotonic()
        if now - self._last_write < 1.0:
            return False
        # Rate-limit even repeated filesystem failures; a broken mount must not
        # turn high-rate fleet callbacks into repeated disk work.
        self._last_write = now
        cars = fleet.get("cars")
        if not isinstance(cars, (list, tuple)):
            cars = []
        row: Dict[str, Any] = {
            "kind": "reward_experiment_metrics",
            "run_id": self.run_id,
            "runtime": self.runtime,
            "step": _json_value(fleet.get("step")),
            "episode": _json_value(fleet.get("episode")),
            "ts": datetime.now(timezone.utc).isoformat(),
            "cars": [],
        }
        for car in cars:
            if not isinstance(car, dict):
                continue
            entry = {key: _json_value(car.get(key)) for key in _CAR_FIELDS}
            entry["fault_fields_available"] = bool(
                car.get("_fault_fields_available", False)
            )
            row["cars"].append(entry)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
                stream.write("\n")
                stream.flush()
        except (OSError, TypeError, ValueError):
            return False
        return True
