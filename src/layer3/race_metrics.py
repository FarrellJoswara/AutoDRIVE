"""Crash-resilient append-only race metrics for official PPO training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


class RaceMetricsJournal:
    """Persist per-lap, per-episode, and post-checkpoint summaries as JSONL.

    Each complete event is flushed and fsynced before returning. Existing rows
    are scanned on resume to restore event deduplication, episode numbering,
    cumulative totals, rolling lap windows, and the current checkpoint interval.
    """

    SCHEMA_VERSION = 1

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.path = Path(run_dir) / "race_metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self._seen_event_ids: set[str] = set()
        self._episode_indices: dict[int, int] = {}
        self._race_lap_records: list[tuple[int, Optional[int], Optional[float]]] = []
        self._interval_lap_records: list[tuple[int, Optional[int], Optional[float]]] = []
        self._lap_event_count = 0
        self._valid_lap_count = 0
        self._episode_count = 0
        self._clean_episode_count = 0
        self._collision_episode_count = 0
        self._collision_total = 0
        self._replay_existing_rows()
        self._ensure_append_boundary()

    def _replay_existing_rows(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                event_id = row.get("event_id")
                if isinstance(event_id, str):
                    self._seen_event_ids.add(event_id)
                env_id = _int(row.get("env_id"), 0)
                episode_index = row.get("episode_index")
                if isinstance(episode_index, int) and not isinstance(episode_index, bool):
                    self._episode_indices[env_id] = max(
                        self._episode_indices.get(env_id, 0), episode_index + 1
                    )
                event_type = row.get("event_type")
                if event_type == "lap":
                    if row.get("phase") != "race":
                        continue
                    timing_valid = row.get("timing_valid") is True
                    lap_time = _number(row.get("lap_time_s")) if timing_valid else None
                    record = (episode_index if isinstance(episode_index, int) else -1,
                              row.get("lap_index") if isinstance(row.get("lap_index"), int) else None,
                              lap_time)
                    self._race_lap_records.append(record)
                    self._interval_lap_records.append(record)
                    self._lap_event_count += 1
                    self._valid_lap_count += int(lap_time is not None)
                elif event_type == "episode":
                    self._episode_count += 1
                    self._clean_episode_count += int(row.get("clean_10_lap_episode") is True)
                    collisions = max(0, _int(row.get("race_collisions"), 0))
                    self._collision_total += collisions
                    self._collision_episode_count += int(collisions > 0)
                elif event_type == "checkpoint":
                    self._interval_lap_records.clear()

    def _ensure_append_boundary(self) -> None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            has_newline = stream.read(1) == b"\n"
        if not has_newline:
            with self.path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def next_episode_index(self, env_id: int) -> int:
        return self._episode_indices.get(int(env_id), 0)

    def _append(self, event_id: str, payload: dict[str, Any]) -> bool:
        if event_id in self._seen_event_ids:
            return False
        row = {
            "schema_version": self.SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": payload["event_type"],
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        encoded = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._seen_event_ids.add(event_id)
        return True

    @staticmethod
    def _faults(info: dict[str, Any]) -> dict[str, Any]:
        return {
            "race_disqualified": bool(info.get("race_disqualified", False)),
            "race_lap_timing_missing_count": _int(
                info.get("race_lap_timing_missing_count"), 0
            ),
            "race_lap_timing_valid": info.get("race_lap_timing_valid"),
            "command_expiry_count": _int(info.get("command_expiry_count"), 0),
            "neutral_fallback_count": _int(info.get("neutral_fallback_count"), 0),
            "termination_reason": info.get("termination_reason"),
        }

    def record_lap(
        self,
        *,
        timestep: int,
        env_id: int,
        episode_index: int,
        lap_event: dict[str, Any],
        info: dict[str, Any],
    ) -> bool:
        official_lap_count = _int(lap_event.get("official_lap_count"), -1)
        phase = str(lap_event.get("phase", "race"))
        lap_index = lap_event.get("lap_index")
        valid = lap_event.get("timing_valid") is True
        lap_time = _number(lap_event.get("lap_time_s")) if valid else None
        valid = valid and lap_time is not None and lap_time > 0.0
        if not valid:
            lap_time = None
        event_id = (
            f"{self.run_id}:lap:{env_id}:{timestep}:{episode_index}:"
            f"{official_lap_count}:{phase}"
        )
        payload = {
            "event_type": "lap",
            "timestep": int(timestep),
            "env_id": int(env_id),
            "episode_index": int(episode_index),
            "lap_index": lap_index if isinstance(lap_index, int) else None,
            "official_lap_count": official_lap_count,
            "phase": phase,
            "lap_time_s": lap_time,
            "timing_valid": bool(valid),
            "timing_missing_reason": lap_event.get("timing_missing_reason"),
            "timing_update_count": lap_event.get("timing_update_count"),
            "collision_event": bool(info.get("collision_event", False)),
            "race_collisions": max(0, _int(info.get("race_collisions"), 0)),
            "raw_collision_count": max(0, _int(info.get("raw_collision_count"), 0)),
            "faults": self._faults(info),
        }
        appended = self._append(event_id, payload)
        if appended and phase == "race":
            record = (int(episode_index), lap_index if isinstance(lap_index, int) else None, lap_time)
            self._race_lap_records.append(record)
            self._interval_lap_records.append(record)
            self._lap_event_count += 1
            self._valid_lap_count += int(lap_time is not None)
        return appended

    def record_episode(
        self,
        *,
        timestep: int,
        env_id: int,
        episode_index: int,
        outcome: dict[str, Any],
        info: dict[str, Any],
    ) -> bool:
        times = [
            number for item in (outcome.get("race_lap_times_s") or [])
            if (number := _number(item)) is not None and number > 0.0
        ]
        episode_info = info.get("episode")
        collisions = max(0, _int(outcome.get("race_collisions"), 0))
        event_id = f"{self.run_id}:episode:{env_id}:{timestep}:{episode_index}"
        payload = {
            "event_type": "episode",
            "timestep": int(timestep),
            "env_id": int(env_id),
            "episode_index": int(episode_index),
            "clean_10_lap_episode": bool(outcome.get("clean_10_lap_episode", False)),
            "episode_won": bool(outcome.get("episode_won", False)),
            "termination_reason": outcome.get("termination_reason"),
            "race_laps_completed": _int(outcome.get("race_laps_completed"), 0),
            "race_laps_required": _int(outcome.get("race_laps_required"), 0),
            "race_lap_times_s": times,
            "race_lap_timing_valid": bool(outcome.get("race_lap_timing_valid", False)),
            "race_lap_timing_missing_count": _int(
                outcome.get("race_lap_timing_missing_count"), 0
            ),
            "race_collisions": collisions,
            "raw_collision_count": max(0, _int(outcome.get("raw_collision_count"), 0)),
            "race_time_s": _number(outcome.get("race_time_s")),
            "route_frontier_distance_m": _number(outcome.get("route_frontier_distance_m")),
            "route_frontier_target_m": _number(outcome.get("route_frontier_target_m")),
            "episode_return": _number(episode_info.get("r")) if isinstance(episode_info, dict) else None,
            "episode_length": _int(episode_info.get("l"), 0) if isinstance(episode_info, dict) else None,
            "episode_duration_s": _number(episode_info.get("t")) if isinstance(episode_info, dict) else None,
            "lap_time_mean_s": statistics.mean(times) if times else None,
            "lap_time_median_s": statistics.median(times) if times else None,
            "lap_time_min_s": min(times) if times else None,
        }
        appended = self._append(event_id, payload)
        if appended:
            self._episode_count += 1
            self._clean_episode_count += int(payload["clean_10_lap_episode"])
            self._collision_total += collisions
            self._collision_episode_count += int(collisions > 0)
            self._episode_indices[env_id] = episode_index + 1
        return appended

    @staticmethod
    def _summary(
        records: list[tuple[int, Optional[int], Optional[float]]]
    ) -> dict[str, Any]:
        valid = [value for _episode, _index, value in records if value is not None]
        return {
            "lap_sample_count": len(records),
            "valid_lap_count": len(valid),
            "invalid_lap_count": len(records) - len(valid),
            "lap_time_mean_s": statistics.mean(valid) if valid else None,
            "lap_time_median_s": statistics.median(valid) if valid else None,
            "lap_time_min_s": min(valid) if valid else None,
        }

    def record_checkpoint(
        self,
        *,
        timestep: int,
        optimizer_updates: int,
        checkpoint_path: Path,
    ) -> bool:
        try:
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        except OSError:
            digest = None
        checkpoint = str(checkpoint_path.resolve())
        event_id = f"{self.run_id}:checkpoint:{checkpoint}:{digest or 'unavailable'}"
        rolling_windows = [
            self._race_lap_records[index:index + 10]
            for index in range(max(0, len(self._race_lap_records) - 9))
        ]
        complete_windows = []
        for window in rolling_windows:
            if len(window) != 10:
                continue
            episode_ids = {record[0] for record in window}
            lap_indices = [record[1] for record in window]
            lap_times = [record[2] for record in window]
            if (
                len(episode_ids) == 1
                and all(index is not None for index in lap_indices)
                and lap_indices == list(range(lap_indices[0], lap_indices[0] + 10))
                and all(value is not None for value in lap_times)
            ):
                complete_windows.append([float(value) for value in lap_times])
        window_totals = [sum(window) for window in complete_windows]
        best_window = min(complete_windows, key=sum) if complete_windows else []
        payload = {
            "event_type": "checkpoint",
            "timestep": int(timestep),
            "optimizer_updates": int(optimizer_updates),
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": digest,
            "interval": self._summary(self._interval_lap_records),
            "cumulative": {
                "lap_event_count": self._lap_event_count,
                "valid_lap_count": self._valid_lap_count,
                "episode_count": self._episode_count,
                "clean_10_lap_episode_count": self._clean_episode_count,
                "collision_episode_count": self._collision_episode_count,
                "collision_total": self._collision_total,
            },
            "rolling_10_lap_window_count": len(window_totals),
            "rolling_10_lap_best_total_s": min(window_totals) if window_totals else None,
            "rolling_10_lap_best_mean_s": (min(window_totals) / 10.0) if window_totals else None,
            "rolling_10_lap_mean_total_s": statistics.mean(window_totals) if window_totals else None,
            "rolling_10_lap_mean_window_lap_s": (
                statistics.mean(window_totals) / 10.0 if window_totals else None
            ),
            "rolling_10_lap_best_window_s": best_window,
        }
        appended = self._append(event_id, payload)
        if appended:
            self._interval_lap_records.clear()
        return appended
