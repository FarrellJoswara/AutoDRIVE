"""Non-blocking hub telemetry publisher for SB3 training.

When HUB_URL is set, registers next to CheckpointCallback. The step loop only
enqueues; a daemon thread POSTs with timeouts + circuit breaker. Training never
blocks on a hung hub.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
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
    """Min-pool 1080 → target beams (groups of beam_count // target)."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = flat.shape[0]
    if n == 0 or target_beams <= 0:
        return []
    if target_beams >= n:
        return [round(float(x), 3) for x in flat.tolist()]
    group = n // target_beams
    usable = group * target_beams
    pooled = flat[:usable].reshape(target_beams, group).min(axis=1)
    return [round(float(x), 3) for x in pooled.tolist()]


def _lap_fields(info: Dict[str, Any]) -> Dict[str, Any]:
    """Forward Layer 2 lap diagnostics without doing route math in the hub."""
    return {
        "lap_supported": bool(info.get("lap_supported", False)),
        "lap_count": int(info.get("lap_count", 0)),
        "last_lap_time_s": info.get("last_lap_time_s"),
        "best_lap_time_s": info.get("best_lap_time_s"),
        "lap_elapsed_s": info.get("lap_elapsed_s"),
    }


class _Publisher:
    """Daemon thread: drop-oldest queue → HTTP POST with timeouts + breaker."""

    def __init__(self, hub_url: str, maxsize: int = 4) -> None:
        self.url = hub_url.rstrip("/") + "/telemetry"
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
            if now < self._backoff_until:
                continue
            try:
                session.post(self.url, json=item, timeout=(0.5, 1.0))
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
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.hub_url = hub_url
        self.run_id = run_id
        self.every_n = max(1, int(every_n))
        self.fleet_hz = max(0.1, float(fleet_hz))
        self.lidar_beams = max(1, int(lidar_beams))
        self.lidar_max_envs = max(0, int(lidar_max_envs))
        self._pub: Optional[_Publisher] = None
        self._last_fleet_t = 0.0
        self._episode_count = 0
        self._ep_returns: Optional[np.ndarray] = None
        # Unity's collision count survives environment respawns. Track its
        # per-step deltas to report contacts for the current training episode.
        self._last_collision_counts: Dict[int, int] = {}
        self._episode_collision_counts: Dict[int, int] = {}
        # Pose / yaw history — refreshed every env step (not only fleet_hz).
        self._last_pose: Dict[int, tuple] = {}
        self._last_yaw: Dict[int, float] = {}

    def _on_training_start(self) -> None:
        self._pub = _Publisher(self.hub_url)
        n = getattr(self.training_env, "num_envs", 1)
        self._ep_returns = np.zeros(n, dtype=np.float64)
        self._last_pose = {}
        self._last_yaw = {}
        self._last_collision_counts = {}
        self._episode_collision_counts = {}

    def _on_step(self) -> bool:
        if self._pub is None:
            return True

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
                "step": int(self.num_timesteps),
                "reward": mean_r,
                "episode": int(self._episode_count),
                "loss": loss,
                "checkpoint": None,
                "run_id": self.run_id,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            self._pub.enqueue(payload)

        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
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
            speed = float(info["true_speed"]) if "true_speed" in info else None
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
            current = max(0, int(info.get("collision_count", 0)))
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
                        "time_since_frontier_push_s": None,
                        "frontier_speed_mps": None,
                        **_lap_fields(info),
                        "lidar": [],
                        "reset": True,
                    }
                )
                continue
            pos = info.get("position")
            pose = None
            if pos is not None and len(pos) >= 3:
                # info["position"] is Unity (x, y, z). Fleet/occupancy use [x, z].
                pose = [float(pos[0]), float(pos[2])]

            yaw = float(info["yaw"]) if "yaw" in info else None
            speed = float(info["true_speed"]) if "true_speed" in info else None
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
                "collision": episode_collision_counts[i] > 0
                if i < len(episode_collision_counts)
                else False,
                "collision_count": episode_collision_counts[i]
                if i < len(episode_collision_counts)
                else 0,
                "speed": speed,
                "episode_return": (
                    float(self._ep_returns[i])
                    if self._ep_returns is not None and i < len(self._ep_returns)
                    else None
                ),
                "reset": False,
                "frontier_line": info.get("frontier_line"),
                "frontier_progress_m": info.get("frontier_progress_m"),
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
            "step": int(self.num_timesteps),
            "episode": int(self._episode_count),
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            # RoboRacer planar LiDAR: 0.06 … 10.0 m
            "lidar_range_min": 0.06,
            "lidar_range_max": 10.0,
            "lap_gate": lap_gate,
            "cars": cars,
        }

    def _on_training_end(self) -> None:
        if self._pub is not None:
            self._pub.stop(join_timeout=2.0)
            self._pub = None


def maybe_hub_callback(run_id: str) -> Optional[HubTelemetryCallback]:
    """Return a callback when HUB_URL is set; else None (CLI parity)."""
    hub = os.environ.get("HUB_URL", "").strip()
    if not hub:
        return None
    return HubTelemetryCallback(
        hub_url=hub,
        run_id=run_id,
        every_n=_env_int("AICAR_TELEMETRY_EVERY_N", 200),
        fleet_hz=_env_float("AICAR_FLEET_HZ", 15.0),
        lidar_beams=_env_int("AICAR_LIDAR_DISPLAY_BEAMS", 120),
        lidar_max_envs=_env_int("AICAR_TELEMETRY_LIDAR_MAX_ENVS", 4),
    )
