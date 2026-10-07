"""Layer 2 Gymnasium adapter for the official RoboRacer ROS 2 race.

This local evaluator builds observations only from permitted sensor topics.
Restricted lap/collision counters are read by Layer 1's evaluation-only
monitor and affect reporting/termination only; they never enter the observation
or action path. Collision handling is left to the simulator checkpoint reset.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.layer1.ros2_racer import RacerRos2
from src.layer1.telemetry import TelemetrySnapshot
from .spaces import (
    NegativeThrottleMode,
    SteeringMode,
    make_action_space,
    make_observation_space,
    map_policy_throttle,
    map_throttle_action,
    map_steering_action,
    snapshot_to_obs,
)


class OfficialRaceEnv(gym.Env):
    """Official race flow: warm-up lap, then N timed laps, including collisions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        racer: Optional[RacerRos2] = None,
        warmup_laps: int = 1,
        race_laps: int = 10,
        timeout_s: float = 180.0,
        steering_action_scale: float = 1.0,
        straight_throttle_gain: float = 1.0,
        straight_throttle_steering_threshold: float = 0.15,
        throttle_mode: str = "bidirectional",
        negative_throttle_mode: NegativeThrottleMode = "allow",
        steering_mode: SteeringMode = "normal",
        vehicle_id: str = "roboracer_1",
    ) -> None:
        super().__init__()
        if warmup_laps < 0 or race_laps < 1:
            raise ValueError("warmup_laps must be >= 0 and race_laps must be >= 1")
        if not 0.0 <= steering_action_scale <= 1.0:
            raise ValueError("steering_action_scale must be in [0, 1]")
        if not 1.0 <= straight_throttle_gain <= 2.0:
            raise ValueError("straight_throttle_gain must be in [1, 2]")
        if not 0.0 <= straight_throttle_steering_threshold <= 1.0:
            raise ValueError("straight_throttle_steering_threshold must be in [0, 1]")

        self.racer = racer or RacerRos2(
            vehicle_id=vehicle_id,
            timeout_s=timeout_s,
            include_race_metrics=True,
        )
        self._owns_racer = racer is None
        self.warmup_laps = int(warmup_laps)
        self.race_laps = int(race_laps)
        self.steering_action_scale = float(steering_action_scale)
        self.straight_throttle_gain = float(straight_throttle_gain)
        self.straight_throttle_steering_threshold = float(straight_throttle_steering_threshold)
        # Validate once during environment construction.
        map_throttle_action(0.0, negative_throttle_mode)
        self.negative_throttle_mode = negative_throttle_mode
        map_policy_throttle(0.0, throttle_mode)
        self.throttle_mode = throttle_mode
        map_steering_action(0.0, steering_mode)
        self.steering_mode = steering_mode
        self.action_space = make_action_space()
        self.observation_space = make_observation_space()
        self._initial_lap_count = 0
        self._last_lap_count = 0
        self._last_collision_count = 0
        self._initial_collision_count = 0
        self._race_collision_baseline: Optional[int] = None
        self._warmup_collision_count = 0
        self._race_lap_times_s: list[float] = []
        self._race_laps_count = 0
        self._warmup_lap_times_s: list[float] = []
        self._episode_steps = 0
        self._last_snap = TelemetrySnapshot(lidar_valid=False)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        snap = self.racer.wait_until_ready()
        metrics = self.racer.wait_for_race_metrics()
        self._initial_lap_count = int(metrics.lap_count)
        self._last_lap_count = self._initial_lap_count
        self._last_collision_count = int(metrics.collision_count)
        self._initial_collision_count = self._last_collision_count
        self._race_collision_baseline = (
            self._last_collision_count if self.warmup_laps == 0 else None
        )
        self._warmup_collision_count = 0
        self._race_lap_times_s = []
        self._race_laps_count = 0
        self._warmup_lap_times_s = []
        self._episode_steps = 0
        self._last_snap = snap
        return snapshot_to_obs(snap, 0.0, 0.0), self._build_info(
            snap, metrics, collision_event=False
        )

    def step(self, action: np.ndarray):
        command = np.asarray(action, dtype=np.float32).reshape(2)
        throttle = (
            map_policy_throttle(command[0], self.throttle_mode)
            if self.throttle_mode == "forward_only"
            else map_throttle_action(command[0], self.negative_throttle_mode)
        )
        steering = map_steering_action(command[1], self.steering_mode)
        steering = float(steering * self.steering_action_scale)
        if throttle > 0 and abs(steering) < self.straight_throttle_steering_threshold:
            throttle = min(1.0, throttle * self.straight_throttle_gain)

        snap = self.racer.step(throttle, steering)
        self._episode_steps += 1
        metrics = self.racer.race_metrics
        collision_count = int(metrics.collision_count)
        collision_event = collision_count > self._last_collision_count
        current_lap_count = int(metrics.lap_count)
        completed_since_last = max(0, current_lap_count - self._last_lap_count)
        if completed_since_last:
            for offset in range(completed_since_last):
                relative_laps = self._last_lap_count - self._initial_lap_count + offset + 1
                lap_time = float(metrics.last_lap_time)
                if relative_laps <= self.warmup_laps:
                    if lap_time > 0:
                        self._warmup_lap_times_s.append(lap_time)
                    if relative_laps >= self.warmup_laps:
                        # Warm-up collisions do not count toward the race score.
                        self._race_collision_baseline = collision_count
                        self._warmup_collision_count = max(
                            0, collision_count - self._initial_collision_count
                        )
                else:
                    self._race_laps_count += 1
                    if lap_time > 0:
                        self._race_lap_times_s.append(lap_time)
            self._last_lap_count = current_lap_count

        terminated = self._race_laps_count >= self.race_laps
        self._last_collision_count = collision_count
        self._last_snap = snap
        info = self._build_info(snap, metrics, collision_event=collision_event)
        info.update({
            "throttle_command": throttle,
            "steering_command": steering,
            "step_duration_s": float(self.racer.last_step_duration_s),
            "official_lap_count": current_lap_count,
            "laps_since_start": max(0, current_lap_count - self._initial_lap_count),
            "warmup_lap_times_s": list(self._warmup_lap_times_s),
            "race_lap_times_s": list(self._race_lap_times_s),
            "lap_times_s": list(self._race_lap_times_s),
            "race_laps_completed": self._race_laps_count,
            "race_collisions": self._race_collisions(collision_count),
            "race_collision_baseline": self._race_collision_baseline,
            "warmup_collisions": self._warmup_collision_count,
            "raw_collision_count": collision_count,
            "race_time_s": sum(self._race_lap_times_s),
            "race_complete": terminated,
            "lidar_scan_rate_hz": float(snap.lidar_scan_rate),
            "control_interval_s": float(
                getattr(self.racer, "last_control_interval_s", 0.0)
            ),
        })
        observation = snapshot_to_obs(snap, throttle, steering)
        # The official scoring harness is deterministic evaluation, not PPO
        # training; scores come from official lap timing and collision counts.
        return observation, 0.0, terminated, False, info

    def _race_collisions(self, current_count: int) -> int:
        baseline = self._race_collision_baseline
        if baseline is None:
            return 0
        return max(0, int(current_count) - int(baseline))

    def _build_info(
        self,
        snap: TelemetrySnapshot,
        metrics: Any,
        *,
        collision_event: bool,
    ) -> Dict[str, Any]:
        return {
            "v_long": float(snap.v_long),
            "true_speed": float(snap.true_speed),
            "throttle": float(snap.throttle),
            "steering": float(snap.steering),
            "collision": False,
            "collision_event": bool(collision_event),
            "collision_count": int(metrics.collision_count),
            "lap_count": int(metrics.lap_count),
            "lap_time_s": float(metrics.lap_time),
            "last_lap_time_s": float(metrics.last_lap_time),
            "best_lap_time_s": float(metrics.best_lap_time),
            "lidar_range_min": float(snap.lidar_range_min),
            "lidar_range_max": float(snap.lidar_range_max),
            "lidar_beams": int(snap.lidar_ranges.size),
        }

    def close(self) -> None:
        if self._owns_racer:
            self.racer.kill()
