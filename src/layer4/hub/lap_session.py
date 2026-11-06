"""Durable per-run lap history assembled from authoritative fleet telemetry."""

from __future__ import annotations

import json
import copy
import hashlib
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any


class LapSessionStore:
    """Aggregate Layer 2 lap times across episode resets and persist completions."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._observed: dict[tuple[str, str], dict[str, Any]] = {}

    def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return self.snapshot(payload, None)
        cars = payload.get("cars")
        if not isinstance(cars, list):
            return self.snapshot(payload, None)
        changed = False
        with self._lock:
            session = self._load(run_id)
            for index, car in enumerate(cars):
                if not isinstance(car, dict):
                    continue
                car_id = str(car.get("env_id", index))
                key = (run_id, car_id)
                if car_id not in session["cars"]:
                    changed = True
                state = session["cars"].setdefault(car_id, {
                    "car_id": car_id, "laps": [], "best_lap_time_s": None,
                    "interrupted_attempts": [], "track_id": payload.get("map_id"),
                    "layout_id": payload.get("layout_id"), "direction": payload.get("direction"),
                })
                previous = self._observed.get(key, {
                    "episode": state.get("_last_episode"),
                    "array_len": state.get("_last_array_len", 0),
                    "elapsed": state.get("_last_elapsed_s"),
                })
                raw_times = car.get("lap_times_s")
                times = self._valid_times(raw_times)
                has_car_episode = "episode" in car
                episode = (car.get("episode") if has_car_episode else
                           (previous.get("episode") if previous.get("episode") is not None
                            else payload.get("episode")))
                # The Layer 2 tracker owns crossing validation and timestamps.
                # Its lap array resets per episode; append only newly completed entries.
                known = int(previous.get("array_len", 0))
                array_reset = len(times) < known
                episode_changed = (
                    has_car_episode and previous.get("episode") is not None
                    and episode != previous["episode"]
                ) or array_reset or (
                    bool(car.get("reset")) and known == 0 and not times
                    and previous.get("episode") is not None
                )
                if episode_changed:
                    if not has_car_episode and (array_reset or car.get("reset")):
                        episode = int(previous.get("episode") or 0) + 1
                    elapsed = self._finite(previous.get("elapsed"))
                    if elapsed is not None and elapsed > 0:
                        state["interrupted_attempts"].append({
                            "episode": previous.get("episode"),
                            "elapsed_s": elapsed,
                            "status": "interrupted",
                            "reason": "episode_reset",
                        })
                    known = 0
                completed = times[known:] if len(times) >= known else times
                for lap_index, lap_time in enumerate(completed, start=len(state["laps"]) + 1):
                    previous_best = state.get("best_lap_time_s")
                    state["laps"].append({
                        "lap_number": lap_index, "time_s": lap_time,
                        "episode": episode,
                        "completed_at": payload.get("ts"),
                        "delta_s": None if previous_best is None else lap_time - previous_best,
                        "status": "complete",
                    })
                    state["best_lap_time_s"] = lap_time if previous_best is None else min(previous_best, lap_time)
                    changed = True
                next_known = len(times)
                if next_known != previous.get("array_len") or episode_changed:
                    changed = True
                elapsed_now = self._finite(car.get("lap_elapsed_s"))
                elapsed_for_interrupt = (previous.get("elapsed") if car.get("reset") and elapsed_now == 0
                                         else elapsed_now)
                self._observed[key] = {"episode": episode, "array_len": next_known,
                                       "elapsed": elapsed_for_interrupt}
                state["current_lap_elapsed_s"] = self._finite(car.get("lap_elapsed_s"))
                state["last_lap_time_s"] = self._finite(car.get("last_lap_time_s"))
                state["lap_supported"] = bool(car.get("lap_supported", False))
                state["track_id"] = payload.get("map_id", state.get("track_id"))
                state["layout_id"] = payload.get("layout_id", state.get("layout_id"))
                state["direction"] = payload.get("direction", state.get("direction"))
                state["_last_episode"] = episode
                state["_last_array_len"] = next_known
                state["_last_elapsed_s"] = elapsed_for_interrupt
            if (changed or session.get("track_id") != payload.get("map_id")
                    or session.get("layout_id") != payload.get("layout_id")
                    or session.get("direction") != payload.get("direction")):
                session["track_id"] = payload.get("map_id", session.get("track_id"))
                session["layout_id"] = payload.get("layout_id", session.get("layout_id"))
                session["direction"] = payload.get("direction", session.get("direction"))
                session["updated_at"] = payload.get("ts")
                self._save(run_id, session)
            result = self._summary_snapshot(session, run_id)
        return result

    def snapshot(self, payload: dict[str, Any] | None, run_id: str | None) -> dict[str, Any]:
        if not run_id and isinstance(payload, dict):
            run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return {"session_id": None, "run_id": None, "state": "unavailable", "track_id": None, "direction": None, "cars": []}
        with self._lock:
            session = self._load(run_id)
            live = {str(car.get("env_id", i)): car for i, car in enumerate((payload or {}).get("cars", [])) if isinstance(car, dict)}
            result = self._public(session, run_id)
            for car in result["cars"]:
                current = live.get(car["car_id"])
                if current is not None:
                    car["current_lap_elapsed_s"] = self._finite(current.get("lap_elapsed_s"))
            return result

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public(self._load(run_id), run_id)

    def enrich_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add durable summaries to a read snapshot without observing/counting it."""
        result = dict(payload)
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not isinstance(result.get("cars"), list):
            return result
        copied_cars = [dict(car) if isinstance(car, dict) else car for car in result["cars"]]
        result["cars"] = copied_cars
        with self._lock:
            session = self._load(run_id)
            for index, car in enumerate(copied_cars):
                if not isinstance(car, dict):
                    continue
                state = session.get("cars", {}).get(str(car.get("env_id", index)))
                if state is not None:
                    car["lap_session"] = self._lap_summary(state, car)
        return result

    def enrich_status(self, status: dict[str, Any]) -> dict[str, Any]:
        """Enrich a run/replay status fleet without re-observing telemetry."""
        result = dict(status)
        fleet = result.get("last_fleet")
        if isinstance(fleet, dict):
            result["last_fleet"] = self.enrich_frame(fleet)
        return result

    @staticmethod
    def _summary_snapshot(session: dict[str, Any], run_id: str) -> dict[str, Any]:
        return {
            "session_id": run_id, "run_id": run_id,
            "track_id": session.get("track_id"), "layout_id": session.get("layout_id"),
            "direction": session.get("direction"),
            "cars": [{"car_id": car_id, "lap_session": LapSessionStore._lap_summary(state)}
                     for car_id, state in sorted(session.get("cars", {}).items())],
        }

    @staticmethod
    def _lap_summary(state: dict[str, Any], live_car: dict[str, Any] | None = None) -> dict[str, Any]:
        laps = state.get("laps", [])
        latest = laps[-1] if laps else None
        elapsed = (LapSessionStore._finite(live_car.get("lap_elapsed_s"))
                   if live_car is not None else state.get("current_lap_elapsed_s"))
        supported = (bool(live_car.get("lap_supported")) if live_car is not None
                     else bool(state.get("lap_supported")))
        return {
            "completed_laps": len(laps),
            "latest_lap_time_s": latest.get("time_s") if latest else None,
            "best_lap_time_s": state.get("best_lap_time_s"),
            "last_lap_delta_s": latest.get("delta_s") if latest else None,
            "current_lap_number": len(laps) + 1 if supported and elapsed is not None else None,
            "current_lap_elapsed_s": elapsed,
        }

    def history(self, run_id: str, env_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._load(run_id)
            selected = session.get("cars", {})
            if env_id is not None:
                selected = {str(env_id): selected.get(str(env_id), {
                    "laps": [], "interrupted_attempts": [], "best_lap_time_s": None
                })}
            laps = []
            previous_sessions = self._matching_previous_sessions(run_id, session, env_id)
            for car_key, car in sorted(selected.items()):
                car_laps = list(car.get("laps", []))
                laps.extend({**lap, "env_id": int(car_key) if str(car_key).isdigit() else car_key}
                            for lap in car_laps)
            return {
                "run_id": run_id,
                "env_id": int(env_id) if env_id is not None and env_id.isdigit() else env_id,
                "track_id": session.get("track_id"), "layout_id": session.get("layout_id"),
                "direction": session.get("direction"), "laps": laps,
                "previous_sessions": previous_sessions,
            }

    def _matching_previous_sessions(self, run_id: str, session: dict[str, Any],
                                    env_id: str | None) -> list[dict[str, Any]]:
        identity = (session.get("track_id"), session.get("layout_id"), session.get("direction"))
        if not all(identity):
            return []
        result = []
        for path in self.root.glob("*.json"):
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (not isinstance(prior, dict) or prior.get("run_id") == run_id
                    or (prior.get("track_id"), prior.get("layout_id"), prior.get("direction")) != identity):
                continue
            for prior_env, car in sorted((prior.get("cars") or {}).items()):
                if env_id is not None and str(prior_env) != str(env_id):
                    continue
                laps = copy.deepcopy(car.get("laps", []))
                if laps:
                    result.append({"run_id": prior["run_id"],
                                   "env_id": int(prior_env) if str(prior_env).isdigit() else prior_env,
                                   "track_id": prior.get("track_id"),
                                   "layout_id": prior.get("layout_id"),
                                   "direction": prior.get("direction"),
                                   "completed_at": prior.get("updated_at"), "laps": laps})
        return sorted(result, key=lambda row: (row["run_id"], str(row["env_id"])))

    def _load(self, run_id: str) -> dict[str, Any]:
        if run_id in self._sessions:
            return self._sessions[run_id]
        path = self._path(run_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {"session_id": run_id, "run_id": run_id, "state": "active", "track_id": None, "layout_id": None, "direction": None, "cars": {}, "updated_at": None}
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            value = {"session_id": run_id, "run_id": run_id, "state": "active", "track_id": None, "layout_id": None, "direction": None, "cars": {}, "updated_at": None}
        self._sessions[run_id] = value
        return value

    def _save(self, run_id: str, value: dict[str, Any]) -> None:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".lap-session-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def _path(self, run_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
        return self.root / f"{safe_id[:80]}-{digest}.json"

    @staticmethod
    def _valid_times(raw: Any) -> list[float]:
        if not isinstance(raw, list):
            return []
        result = []
        for value in raw:
            number = LapSessionStore._finite(value)
            if number is not None and number > 0:
                result.append(number)
        return result

    @staticmethod
    def _finite(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _public(session: dict[str, Any], run_id: str) -> dict[str, Any]:
        return {
            "session_id": run_id, "run_id": run_id,
            "state": session.get("state", "active"),
            "track_id": session.get("track_id"), "layout_id": session.get("layout_id"),
            "direction": session.get("direction"),
            "updated_at": session.get("updated_at"),
            "cars": [LapSessionStore._public_car(car)
                     for _, car in sorted(session.get("cars", {}).items())],
        }

    @staticmethod
    def _public_car(car: dict[str, Any]) -> dict[str, Any]:
        public = copy.deepcopy({key: value for key, value in car.items() if not key.startswith("_")})
        public["lap_session"] = LapSessionStore._lap_summary(car)
        return public
