"""Non-blocking hub telemetry publisher for SB3 training.

When HUB_URL is set, registers next to CheckpointCallback. The step loop only
enqueues; a daemon thread POSTs with timeouts + circuit breaker. Training never
blocks on a hung hub.
"""

from __future__ import annotations

import logging
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _min_pool_lidar(arr: np.ndarray, target_beams: int) -> List[float]:
    """Min-pool the complete LiDAR scan → target beams without dropping its tail."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = flat.shape[0]
    if n == 0 or target_beams <= 0:
        return []
    if target_beams >= n:
        return [round(float(x), 3) for x in flat.tolist()]
    # Split the entire fan, including its endpoint rays. Floor-based grouping
    # used to silently discard the final one (or more) measurements.
    pooled = np.asarray(
        [group.min() for group in np.array_split(flat, target_beams)],
        dtype=np.float32,
    )
    return [round(float(x), 3) for x in pooled.tolist()]


def _lap_fields(info: Dict[str, Any]) -> Dict[str, Any]:
    """Forward Layer 2 lap diagnostics without doing route math in the hub."""
    return {
        "lap_supported": bool(info.get("lap_supported", info.get("official_race", False))),
        "lap_count": int(info.get("race_laps_completed", info.get("lap_count", 0))),
        "lap_times_s": list(info.get("race_lap_times_s", info.get("lap_times_s", []))),
        "last_lap_time_s": info.get("race_last_lap_time_s", info.get("last_lap_time_s")),
        "best_lap_time_s": info.get("race_best_lap_time_s", info.get("best_lap_time_s")),
        "lap_elapsed_s": info.get("lap_elapsed_s"),
    }


class _Publisher:
    """Daemon thread: drop-oldest queue → HTTP POST with timeouts + breaker."""

    def __init__(
        self, hub_url: str, *, snapshot_path: Optional[Path] = None, maxsize: int = 4
    ) -> None:
        self.url = hub_url.rstrip("/") + "/telemetry"
        self.snapshot_path = snapshot_path
        self._last_snapshot_write = 0.0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = object()
        self._thread = threading.Thread(
            target=self._run, name="hub-telemetry-pub", daemon=True
        )
        self._failures = 0
        self._backoff_until = 0.0
        self._last_log = 0.0
        self._thread.start()

    def enqueue(self, payload: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def stop(self, join_timeout: float = 2.0) -> None:
        try:
            self._q.put_nowait(self._stop)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(self._stop)
            except queue.Full:
                pass
        self._thread.join(timeout=join_timeout)

    def _run(self) -> None:
        try:
            import requests
        except ImportError:
            logger.warning("hub_callback: requests not installed; telemetry disabled")
            return

        session = requests.Session()
        while True:
            item = self._q.get()
            if item is self._stop:
                break
            now = time.monotonic()
            if (
                self.snapshot_path is not None
                and item.get("kind") == "fleet"
                and now - self._last_snapshot_write >= 1.0
            ):
                try:
                    self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    self.snapshot_path.write_text(
                        json.dumps(item, separators=(",", ":")), encoding="utf-8"
                    )
                    self._last_snapshot_write = now
                except (OSError, TypeError, ValueError):
                    logger.warning("hub_callback: could not persist latest Watch frame")
            if self.snapshot_path is not None and item.get("kind") in {"metrics", "phase"}:
                target_name = (
                    "latest_training_phase.json"
                    if item.get("kind") == "phase"
                    else "latest_telemetry.json"
                )
                try:
                    self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    (self.snapshot_path.parent / target_name).write_text(
                        json.dumps(item, separators=(",", ":")), encoding="utf-8"
                    )
                except (OSError, TypeError, ValueError):
                    logger.warning("hub_callback: could not persist latest run telemetry")
            if now < self._backoff_until:
                continue
            try:
                target_url = item.pop("_endpoint", self.url)
                session.post(target_url, json=item, timeout=(0.5, 1.0))
                self._failures = 0
            except Exception:
                self._failures += 1
                if now - self._last_log > 60.0:
                    logger.warning(
                        "hub_callback: POST failed (failures=%s)", self._failures
                    )
                    self._last_log = now
                if self._failures >= 5:
                    delay = min(30.0, 1.0 * (2 ** min(self._failures - 5, 4)))
                    self._backoff_until = now + delay


class HubTelemetryCallback(BaseCallback):
    """Publish train metrics (step-based) and optional fleet sim-state (time-based)."""

    def __init__(
        self,
        *,
        hub_url: str,
        run_id: str,
        every_n: int = 200,
        fleet_hz: float = 15.0,
        lidar_beams: int = 120,
        lidar_max_envs: int = 4,
        runtime: str = "custom",
        run_dir: Optional[Path] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.hub_url = hub_url
        self.run_id = run_id
        self.every_n = max(1, int(every_n))
        self.fleet_hz = max(0.1, float(fleet_hz))
        self.lidar_beams = max(1, int(lidar_beams))
        self.lidar_max_envs = max(0, int(lidar_max_envs))
        self.runtime = str(runtime)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.rollout_size = 1
        self._pub: Optional[_Publisher] = None
        self._last_fleet_t = 0.0
        self._episode_count = 0
        self._ep_returns: Optional[np.ndarray] = None
        self._last_episode_laps: Dict[int, int] = {}
        self._completed_laps = 0
        self._clean_episode_wins = 0
        self._best_lap_time_s: Optional[float] = None
        self._best_10_lap_time_s: Optional[float] = None
        self._ten_lap_recent: Dict[int, List[float]] = {}
        self._ten_lap_seen_splits: Dict[int, int] = {}
        # Unity's collision count survives environment respawns. Track its
        # per-step deltas to report contacts for the current training episode.
        self._last_collision_counts: Dict[int, int] = {}
        self._episode_collision_counts: Dict[int, int] = {}
        # Pose / yaw history — refreshed every env step (not only fleet_hz).
        self._last_pose: Dict[int, tuple] = {}
        self._last_yaw: Dict[int, float] = {}
        self._latest_ppo_diagnostics: Dict[str, float] = {}
        self._latest_ppo_step: Optional[int] = None
        self._clock_source_counts: Dict[str, int] = {}
        self._clock_elapsed_by_env: Dict[int, float] = {}
        self._wall_elapsed_by_env: Dict[int, float] = {}
        self._clock_total_elapsed_by_env: Dict[int, float] = {}
        self._wall_total_elapsed_by_env: Dict[int, float] = {}
        self._clock_lap_count_by_env: Dict[int, int] = {}
        self._clock_lap_comparisons: List[Dict[str, float]] = []

    def _on_training_start(self) -> None:
        snapshot_path = self.run_dir / "watch_latest.json" if self.run_dir else None
        self._pub = _Publisher(self.hub_url, snapshot_path=snapshot_path)
        n = getattr(self.training_env, "num_envs", 1)
        self.rollout_size = max(1, int(n) * int(getattr(self.model, "n_steps", 1)))
        self._ep_returns = np.zeros(n, dtype=np.float64)
        self._last_episode_laps = {}
        self._completed_laps = 0
        self._clean_episode_wins = 0
        self._best_lap_time_s = None
        self._best_10_lap_time_s = None
        self._ten_lap_recent = {}
        self._ten_lap_seen_splits = {}
        self._last_pose = {}
        self._last_yaw = {}
        self._last_collision_counts = {}
        self._episode_collision_counts = {}
        self._clock_lap_count_by_env = {env_id: 0 for env_id in range(n)}
        self._publish_phase("rollout")

    def _publish_phase(self, phase: str) -> None:
        if self._pub is not None:
            self._pub.enqueue({
                "kind": "phase",
                "phase": phase,
                "step": int(self.num_timesteps),
                "run_id": self.run_id,
                "runtime": self.runtime,
                "rollout_size": self.rollout_size,
                "ts": datetime.now(timezone.utc).isoformat(),
                "_endpoint": self.hub_url.rstrip("/") + "/api/train/phase",
            })

    def publish_evaluator_live(self, payload: Dict[str, Any]) -> None:
        """Queue high-rate evaluator pose frames to the hub WebSocket fan-out."""
        if self._pub is not None:
            self._pub.enqueue({
                **payload,
                "_endpoint": self.hub_url.rstrip("/") + "/api/evaluator/live",
            })

    def _on_rollout_end(self) -> None:
        self._publish_phase("ppo_update")

    def _on_rollout_start(self) -> None:
        # SB3 writes train/* scalars after an update and before the next
        # rollout starts. Capture that completed update once, then include it
        # in regular telemetry so the UI can chart genuine PPO diagnostics.
        diagnostics = self._read_ppo_diagnostics()
        if diagnostics:
            self._latest_ppo_diagnostics = diagnostics
            self._latest_ppo_step = int(self.num_timesteps)
        self._publish_phase("rollout")

    def _read_ppo_diagnostics(self) -> Dict[str, float]:
        names = {
            "fps": "time/fps",
            "approx_kl": "train/approx_kl",
            "clip_fraction": "train/clip_fraction",
            "entropy_loss": "train/entropy_loss",
            "explained_variance": "train/explained_variance",
            "learning_rate": "train/learning_rate",
            "loss": "train/loss",
            "policy_gradient_loss": "train/policy_gradient_loss",
            "std": "train/std",
            "value_loss": "train/value_loss",
        }
        try:
            values = self.model.logger.name_to_value
        except Exception:
            return {}
        result: Dict[str, float] = {}
        for output_name, logger_name in names.items():
            try:
                # SB3 uses a defaultdict here; indexing a metric before it has
                # been recorded inserts a value without its exclusion metadata,
                # which makes the next logger.dump() fail its key alignment.
                raw_value = values.get(logger_name)
                if raw_value is None:
                    continue
                value = float(raw_value)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                result[output_name] = value
        return result

    def _on_step(self) -> bool:
        if self._pub is None:
            return True

        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is not None:
            for i, raw in enumerate(infos):
                info = raw if isinstance(raw, dict) else {}
                source = str(info.get("control_interval_source", "unknown"))
                self._clock_source_counts[source] = self._clock_source_counts.get(source, 0) + 1
                try:
                    elapsed = float(info.get("control_interval_s", 0.0))
                except (TypeError, ValueError):
                    elapsed = 0.0
                if np.isfinite(elapsed) and elapsed > 0:
                    self._clock_elapsed_by_env[i] = self._clock_elapsed_by_env.get(i, 0.0) + elapsed
                    self._clock_total_elapsed_by_env[i] = self._clock_total_elapsed_by_env.get(i, 0.0) + elapsed
                try:
                    wall_elapsed = float(info.get("control_interval_wall_s", 0.0))
                except (TypeError, ValueError):
                    wall_elapsed = 0.0
                if np.isfinite(wall_elapsed) and wall_elapsed > 0:
                    self._wall_elapsed_by_env[i] = self._wall_elapsed_by_env.get(i, 0.0) + wall_elapsed
                    self._wall_total_elapsed_by_env[i] = self._wall_total_elapsed_by_env.get(i, 0.0) + wall_elapsed
                current_laps = max(0, int(
                    info.get("race_laps_completed", info.get("lap_count", 0)) or 0
                ))
                previous_laps = self._last_episode_laps.get(i, 0)
                raw_lap_count = int(info.get("lap_count", 0) or 0)
                previous_raw_lap_count = self._clock_lap_count_by_env.get(i, raw_lap_count)
                if raw_lap_count > previous_raw_lap_count:
                    try:
                        official_lap_s = float(info.get("last_lap_time_s", 0.0))
                    except (TypeError, ValueError):
                        official_lap_s = 0.0
                    action_elapsed_s = self._clock_elapsed_by_env.get(i, 0.0)
                    wall_elapsed_s = self._wall_elapsed_by_env.get(i, 0.0)
                    if np.isfinite(official_lap_s) and official_lap_s > 0:
                        self._clock_lap_comparisons.append({
                            "env_id": int(i), "action_elapsed_s": action_elapsed_s,
                            "wall_elapsed_s": wall_elapsed_s,
                            "official_lap_time_s": official_lap_s,
                            "ratio": action_elapsed_s / official_lap_s,
                        })
                        self._clock_lap_comparisons = self._clock_lap_comparisons[-100:]
                    self._clock_elapsed_by_env[i] = 0.0
                    self._wall_elapsed_by_env[i] = 0.0
                self._clock_lap_count_by_env[i] = raw_lap_count
                self._completed_laps += max(0, current_laps - previous_laps)
                self._last_episode_laps[i] = current_laps
                lap_time = info.get("race_best_lap_time_s", info.get("best_lap_time_s"))
                if lap_time is not None:
                    try:
                        lap_time = float(lap_time)
                        if np.isfinite(lap_time) and lap_time > 0:
                            self._best_lap_time_s = (
                                lap_time if self._best_lap_time_s is None
                                else min(self._best_lap_time_s, lap_time)
                            )
                    except (TypeError, ValueError):
                        pass
                raw_lap_times = info.get("race_lap_times_s", info.get("lap_times_s"))
                if isinstance(raw_lap_times, (list, tuple)):
                    seen = self._ten_lap_seen_splits.get(i, 0)
                    if len(raw_lap_times) < seen:
                        seen = 0
                        self._ten_lap_recent[i] = []
                    recent = self._ten_lap_recent.setdefault(i, [])
                    for raw_time in raw_lap_times[seen:]:
                        try:
                            completed_lap_time = float(raw_time)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(completed_lap_time) or completed_lap_time <= 0:
                            continue
                        recent.append(completed_lap_time)
                        if len(recent) >= 10:
                            total = float(sum(recent[-10:]))
                            self._best_10_lap_time_s = (
                                total if self._best_10_lap_time_s is None
                                else min(self._best_10_lap_time_s, total)
                            )
                            del recent[:-10]
                    self._ten_lap_seen_splits[i] = len(raw_lap_times)
                if dones is not None and i < len(dones) and bool(dones[i]):
                    if bool(info.get("episode_won", False)):
                        self._clean_episode_wins += 1
                    self._last_episode_laps[i] = 0
                    self._ten_lap_recent[i] = []
                    self._ten_lap_seen_splits[i] = 0
                    self._clock_elapsed_by_env[i] = 0.0
                    self._clock_lap_count_by_env.pop(i, None)

        if self.num_timesteps % self.every_n == 0:
            rewards = self.locals.get("rewards")
            mean_r = 0.0
            if rewards is not None:
                mean_r = float(np.mean(rewards))
            loss = None
            try:
                logger_dict = self.model.logger.name_to_value
                if "train/loss" in logger_dict:
                    loss = float(logger_dict["train/loss"])
                elif "train/policy_gradient_loss" in logger_dict:
                    loss = float(logger_dict["train/policy_gradient_loss"])
            except Exception:
                loss = None

            payload = {
                "kind": "metrics",
                "runtime": self.runtime,
                "rollout_size": self.rollout_size,
                "step": int(self.num_timesteps),
                "reward": mean_r,
                "episode": int(self._episode_count),
                "loss": loss,
                "ppo_step": self._latest_ppo_step,
                "ppo": self._latest_ppo_diagnostics,
                "completed_laps": int(self._completed_laps),
                "clean_episode_wins": int(self._clean_episode_wins),
                "best_lap_time_s": self._best_lap_time_s,
                "best_10_lap_time_s": self._best_10_lap_time_s,
                "checkpoint": None,
                "run_id": self.run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            self._pub.enqueue(payload)

        if infos is not None and self._ep_returns is not None:
            rewards = self.locals.get("rewards")
            for i, info in enumerate(infos):
                if rewards is not None and i < len(rewards):
                    self._ep_returns[i] += float(rewards[i])
                if dones is not None and i < len(dones) and bool(dones[i]):
                    self._episode_count += 1
                    self._ep_returns[i] = 0.0

        # Keep pose/yaw cache fresh every env step (heading for Watch).
        if infos is not None:
            self._refresh_heading_cache(infos, dones)

        episode_collision_counts = self._refresh_collision_counts(infos, dones)

        now = time.monotonic()
        if now - self._last_fleet_t >= 1.0 / self.fleet_hz:
            self._last_fleet_t = now
            fleet = self._build_fleet_sample(infos, dones, episode_collision_counts)
            if fleet is not None:
                self._pub.enqueue(fleet)

        return True

    def _refresh_heading_cache(self, infos: Any, dones: Any) -> None:
        """Update pose/yaw every env step so fleet samples get real headings."""
        if infos is None:
            return
        for i, raw in enumerate(infos):
            if dones is not None and i < len(dones) and bool(dones[i]):
                self._last_pose.pop(i, None)
                self._last_yaw.pop(i, None)
                continue
            info = raw if isinstance(raw, dict) else {}
            pos = info.get("position")
            if pos is None or len(pos) < 3:
                continue
            pose = (float(pos[0]), float(pos[2]))
            yaw = float(info["yaw"]) if "yaw" in info else None
            speed = (
                float(info["watch_speed_mps"])
                if "watch_speed_mps" in info
                else float(info["true_speed"]) if "true_speed" in info else None
            )
            if yaw is None or abs(yaw) < 1e-4:
                prev = self._last_pose.get(i)
                if prev is not None:
                    dx = pose[0] - prev[0]
                    dz = pose[1] - prev[1]
                    # Per-physics-step motion can be millimetres.
                    if dx * dx + dz * dz > 1e-8 and (
                        speed is None or abs(speed) > 0.02
                    ):
                        yaw = float(np.arctan2(dx, dz))
                if (yaw is None or abs(yaw) < 1e-4) and i in self._last_yaw:
                    yaw = self._last_yaw[i]
            self._last_pose[i] = pose
            if yaw is not None and abs(yaw) >= 1e-4:
                self._last_yaw[i] = float(yaw)

    def _refresh_collision_counts(self, infos: Any, dones: Any) -> List[int]:
        """Accumulate Unity collision-count deltas per training episode."""
        if infos is None:
            return []
        counts: List[int] = []
        for i, raw in enumerate(infos):
            info = raw if isinstance(raw, dict) else {}
            current = max(0, int(
                info.get("race_collisions", info.get("collision_count", 0))
            ))
            previous = self._last_collision_counts.get(i)
            delta = 0 if previous is None or current < previous else current - previous
            self._last_collision_counts[i] = current
            total = self._episode_collision_counts.get(i, 0) + delta
            counts.append(total)
            if dones is not None and i < len(dones) and bool(dones[i]):
                self._episode_collision_counts[i] = 0
            else:
                self._episode_collision_counts[i] = total
        return counts

    def _build_fleet_sample(
        self,
        infos: Any,
        dones: Any,
        episode_collision_counts: List[int],
    ) -> Optional[Dict[str, Any]]:
        if infos is None:
            return None
        new_obs = self.locals.get("new_obs")
        lidar_batch = None
        if isinstance(new_obs, dict) and "lidar" in new_obs:
            lidar_batch = np.asarray(new_obs["lidar"])

        cars: List[Dict[str, Any]] = []
        n = len(infos)
        lap_gate = next(
            (
                raw["lap_gate"]
                for raw in infos
                if isinstance(raw, dict) and raw.get("lap_gate") is not None
            ),
            None,
        )
        for i in range(n):
            info = infos[i] if isinstance(infos[i], dict) else {}
            # Done-step hazard: infos=terminal pose, new_obs=post-reset.
            if dones is not None and i < len(dones) and bool(dones[i]):
                self._last_pose.pop(i, None)
                self._last_yaw.pop(i, None)
                cars.append(
                    {
                        "env_id": i,
                        "pose": None,
                        "yaw": None,
                        "collision": False,
                        "collision_count": episode_collision_counts[i]
                        if i < len(episode_collision_counts)
                        else 0,
                        "speed": None,
                        "episode_return": None,
                        "frontier_line": None,
                        "frontier_progress_m": None,
                        "current_progress_line": None,
                        "current_progress_m": None,
                        "signed_route_delta_m": None,
                        "current_route_speed_mps": None,
                        "route_projection_valid": False,
                        "reward_components": None,
                        "time_since_frontier_push_s": None,
                        "frontier_speed_mps": None,
                        **_lap_fields(info),
                        "lidar": [],
                        "reset": True,
                        "reset_reason": info.get("termination_reason")
                        or info.get("truncate_reason")
                        or "episode_end",
                    }
                )
                continue
            pos = info.get("position")
            pose = None
            if pos is not None and len(pos) >= 3:
                # info["position"] is Unity (x, y, z). Fleet/occupancy use [x, z].
                pose = [float(pos[0]), float(pos[2])]

            yaw = float(info["yaw"]) if "yaw" in info else None
            speed = (
                float(info["watch_speed_mps"])
                if "watch_speed_mps" in info
                else float(info["true_speed"]) if "true_speed" in info else None
            )
            # Prefer Layer-1 yaw. If still ~0, use per-step pose delta / sticky.
            # Old 5cm threshold at 15 Hz never fired at ~0.3 m/s → yaw stuck at 0.
            if pose is not None and (yaw is None or abs(yaw) < 1e-4):
                prev = self._last_pose.get(i)
                if prev is not None:
                    dx = pose[0] - prev[0]
                    dz = pose[1] - prev[1]
                    if dx * dx + dz * dz > 1e-6:
                        yaw = float(np.arctan2(dx, dz))
                if (yaw is None or abs(yaw) < 1e-4) and i in self._last_yaw:
                    yaw = self._last_yaw[i]
            if pose is not None:
                self._last_pose[i] = (pose[0], pose[1])
            if yaw is not None and abs(yaw) >= 1e-4:
                self._last_yaw[i] = float(yaw)

            car: Dict[str, Any] = {
                "env_id": i,
                "pose": pose,
                "yaw": yaw,
                # This is a per-step event flash. Keep the accumulated episode
                # contact count separate so old hits do not mark a respawned car.
                "collision": bool(info.get("collision_event", False)),
                "collision_count": episode_collision_counts[i]
                if i < len(episode_collision_counts)
                else 0,
                "speed": speed,
                "speed_source": (
                    "lidar_estimate" if "watch_speed_mps" in info
                    else "wheel_encoder" if self.runtime == "official"
                    else "simulator"
                ),
                "v_long": info.get("v_long"),
                "throttle_command": info.get("throttle_command"),
                "steering_command": info.get("steering_command"),
                "episode_return": (
                    float(self._ep_returns[i])
                    if self._ep_returns is not None and i < len(self._ep_returns)
                    else None
                ),
                "reset": False,
                "frontier_line": info.get("frontier_line"),
                "frontier_progress_m": info.get("frontier_progress_m"),
                "current_progress_line": info.get("current_progress_line"),
                "current_progress_m": info.get("current_progress_m"),
                "signed_route_delta_m": info.get("signed_route_delta_m"),
                "current_route_speed_mps": info.get("current_route_speed_mps"),
                "route_projection_valid": bool(info.get("route_projection_valid", False)),
                "reward_components": info.get("reward_components"),
                "time_since_frontier_push_s": info.get("time_since_frontier_push_s"),
                "frontier_speed_mps": info.get("frontier_speed_mps"),
                **_lap_fields(info),
            }
            if self.lidar_max_envs <= 0:
                include_lidar = False
            elif n > self.lidar_max_envs:
                include_lidar = i == 0
            else:
                include_lidar = i < self.lidar_max_envs

            if include_lidar and lidar_batch is not None and i < lidar_batch.shape[0]:
                car["lidar"] = _min_pool_lidar(lidar_batch[i], self.lidar_beams)
            cars.append(car)

        if not cars:
            return None
        return {
            "kind": "fleet",
            "runtime": self.runtime,
            "rollout_size": self.rollout_size,
            "step": int(self.num_timesteps),
            "episode": int(self._episode_count),
            "run_id": self.run_id,
            "best_10_lap_time_s": self._best_10_lap_time_s,
            "ts": datetime.now(timezone.utc).isoformat(),
            # RoboRacer planar LiDAR: 0.06 … 10.0 m
            "lidar_range_min": 0.06,
            "lidar_range_max": 10.0,
            "lap_gate": lap_gate,
            "cars": cars,
        }

    def _on_training_end(self) -> None:
        if self.run_dir is not None:
            try:
                (self.run_dir / "timing_diagnostics.json").write_text(
                    json.dumps({
                        "clock_sources": self._clock_source_counts,
                        "ros_action_elapsed_s_by_env": self._clock_total_elapsed_by_env,
                        "wall_action_elapsed_s_by_env": self._wall_total_elapsed_by_env,
                        "ros_to_wall_elapsed_ratio_by_env": {
                            str(env_id): self._clock_total_elapsed_by_env[env_id] / wall_s
                            for env_id, wall_s in self._wall_total_elapsed_by_env.items()
                            if wall_s > 0 and env_id in self._clock_total_elapsed_by_env
                        },
                        "completed_lap_clock_comparisons": self._clock_lap_comparisons,
                        "comparison_note": "Sum of action intervals between policy decisions compared with the official lap timer; boundary intervals and the first post-reset lap include reset timing differences.",
                    }, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not save clock diagnostics: %s", exc)
        if self._pub is not None:
            self._publish_phase("stopped")
            self._pub.stop(join_timeout=2.0)
            self._pub = None


def maybe_hub_callback(
    run_id: str, *, runtime: str = "custom", run_dir: Optional[Path] = None
) -> Optional[HubTelemetryCallback]:
    """Return a callback when HUB_URL is set; else None (CLI parity)."""
    hub = os.environ.get("HUB_URL", "").strip()
    if not hub:
        return None
    return HubTelemetryCallback(
        hub_url=hub,
        run_id=run_id,
        run_dir=run_dir,
        every_n=_env_int("AICAR_TELEMETRY_EVERY_N", 200),
        fleet_hz=_env_float("AICAR_FLEET_HZ", 15.0),
        lidar_beams=_env_int("AICAR_LIDAR_DISPLAY_BEAMS", 120),
        lidar_max_envs=_env_int("AICAR_TELEMETRY_LIDAR_MAX_ENVS", 4),
        runtime=runtime,
    )
