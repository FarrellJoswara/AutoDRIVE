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
                # Circuit breaker: after 5 consecutive failures, back off
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

    def _on_training_start(self) -> None:
        self._pub = _Publisher(self.hub_url)
        n = getattr(self.training_env, "num_envs", 1)
        self._ep_returns = np.zeros(n, dtype=np.float64)

    def _on_step(self) -> bool:
        if self._pub is None:
            return True

        # --- metrics (step cadence) ---
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

        # Track episode returns / counts from infos
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

        # --- fleet sim-state (time cadence) ---
        now = time.monotonic()
        if now - self._last_fleet_t >= 1.0 / self.fleet_hz:
            self._last_fleet_t = now
            fleet = self._build_fleet_sample(infos, dones)
            if fleet is not None:
                self._pub.enqueue(fleet)

        return True

    def _build_fleet_sample(
        self,
        infos: Any,
        dones: Any,
    ) -> Optional[Dict[str, Any]]:
        if infos is None:
            return None
        new_obs = self.locals.get("new_obs")
        lidar_batch = None
        if isinstance(new_obs, dict) and "lidar" in new_obs:
            lidar_batch = np.asarray(new_obs["lidar"])

        cars: List[Dict[str, Any]] = []
        n = len(infos)
        for i in range(n):
            # Done-step hazard: skip — infos=terminal, new_obs=post-reset
            if dones is not None and i < len(dones) and bool(dones[i]):
                continue
            info = infos[i] if isinstance(infos[i], dict) else {}
            pos = info.get("position")
            pose = None
            if pos is not None and len(pos) >= 3:
                pose = [float(pos[0]), float(pos[2])]  # Unity X–Z

            car: Dict[str, Any] = {
                "env_id": i,
                "pose": pose,
                "yaw": float(info["yaw"]) if "yaw" in info else None,
                "collision": bool(info.get("collision", False)),
                "speed": float(info["true_speed"]) if "true_speed" in info else None,
                "episode_return": (
                    float(self._ep_returns[i])
                    if self._ep_returns is not None and i < len(self._ep_returns)
                    else None
                ),
            }
            # Publish LiDAR for at most lidar_max_envs; above that, env 0 only
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
            # Display denorm for normalised LiDAR [0,1] → metres (Layer 1 defaults)
            "lidar_range_min": 0.05,
            "lidar_range_max": 30.0,
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
