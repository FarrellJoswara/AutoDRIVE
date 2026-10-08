"""Layer 2 Gym adapter for official-race evaluation and training.

Policy observations/actions use the same sensor-only transform in both modes.
Training mode may use restricted race counters for reward/episode control and
the documented reset topic; neither capability is enabled for policy runtime.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.layer1.ros2_racer import OfficialRosTransportError, RacerRos2
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
    transform_policy_action,
    _normalize_lidar,
)
from .lidar_odometry import LidarOdometry, acceleration_consistent_lidar_speed


class OfficialRaceEnv(gym.Env):
    """Official race flow with optional training-only reset and score reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        racer: Optional[RacerRos2] = None,
        warmup_laps: int = 1,
        race_laps: int = 10,
        timeout_s: float = 180.0,
        frame_timeout_s: float = 5.0,
        steering_action_scale: float = 1.0,
        straight_throttle_gain: float = 1.0,
        straight_throttle_steering_threshold: float = 0.15,
        throttle_mode: str = "bidirectional",
        negative_throttle_mode: NegativeThrottleMode = "allow",
        steering_mode: SteeringMode = "normal",
        vehicle_id: str = "roboracer_1",
        observation_profile: str = "official_sensors",
        training_mode: bool = False,
        training_timeout_s: float = 600.0,
        training_lap_reward: float = 100.0,
        training_failure_penalty: float = 1000.0,
        training_time_cost_per_simulated_second: float = 1.0,
        training_collision_penalty_base: float = 10.0,
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

        self.training_mode = bool(training_mode)
        if observation_profile not in (
            "official_sensors", "official_sensors_history", "official_sensors_camera"
        ):
            raise ValueError("unsupported official race observation profile")
        self.observation_profile = observation_profile
        self.include_camera = observation_profile == "official_sensors_camera"
        if frame_timeout_s <= 0:
            raise ValueError("frame_timeout_s must be positive")
        # Race setup/metric waits may legitimately be longer. A policy step,
        # however, must never block for the full race-level timeout while the
        # simulator has stopped producing observations.
        self.frame_timeout_s = min(float(frame_timeout_s), float(timeout_s))
        self.training_timeout_s = max(0.0, float(training_timeout_s))
        self.training_lap_reward = max(0.0, float(training_lap_reward))
        self.training_failure_penalty = max(0.0, float(training_failure_penalty))
        self.training_time_cost_per_simulated_second = max(
            0.0, float(training_time_cost_per_simulated_second)
        )
        self.training_collision_penalty_base = max(
            0.0, float(training_collision_penalty_base)
        )
        self.racer = racer or RacerRos2(
            vehicle_id=vehicle_id,
            timeout_s=timeout_s,
            frame_timeout_s=self.frame_timeout_s,
            include_race_metrics=True,
            allow_training_reset=self.training_mode,
            require_camera=self.include_camera,
        )
        self._owns_racer = racer is None
        self._observation_builder = OfficialObservationBuilder(
            observation_profile=observation_profile
        )
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
        self.observation_space = make_observation_space(
            lidar_history_frames=4 if observation_profile == "official_sensors_history" else 1,
            include_camera=self.include_camera,
        )
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
        self._training_elapsed_s = 0.0
        self._last_snap = TelemetrySnapshot(lidar_valid=False)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if self.training_mode:
            reset_for_training = getattr(self.racer, "reset_simulation_for_training", None)
            if not callable(reset_for_training):
                raise RuntimeError("training mode requires a training-reset-capable official racer")
            snap = reset_for_training()
        else:
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
        self._training_elapsed_s = 0.0
        self._last_snap = snap
        initial_obs = self._observation_builder.reset(snap)
        info = self._build_info(snap, metrics, collision_event=False)
        info["watch_speed_mps"] = abs(
            float(self._observation_builder.last_forward_speed_mps)
        )
        return initial_obs, info

    def step(self, action: np.ndarray):
        throttle, steering = transform_policy_action(
            action,
            throttle_mode=self.throttle_mode,
            negative_throttle_mode=self.negative_throttle_mode,
            steering_mode=self.steering_mode,
            steering_action_scale=self.steering_action_scale,
            straight_throttle_gain=self.straight_throttle_gain,
            straight_throttle_steering_threshold=self.straight_throttle_steering_threshold,
        )

        race_was_active = self._race_collision_baseline is not None
        previous_race_laps = self._race_laps_count
        previous_lap_count = self._last_lap_count
        previous_race_collisions = self._race_collisions(self._last_collision_count)
        try:
            snap = self.racer.step(throttle, steering)
        except OfficialRosTransportError as exc:
            raise RuntimeError(
                "OfficialRaceEnv.step cannot continue because the official ROS "
                f"transport did not deliver a fresh observation: {exc}"
            ) from exc
        self._episode_steps += 1
        scan_interval_s = max(0.0, float(self.racer.last_step_duration_s))
        action_interval_s = max(
            0.0, float(getattr(self.racer, "last_control_interval_s", 0.0))
        )
        # Use elapsed time between policy actions. During PPO optimization the
        # official simulator continues to hold the previous actuator command;
        # counting only the final scan interval would undercharge race time and
        # distort the speed estimate at the next decision.
        step_duration_s = action_interval_s if action_interval_s > 0 else scan_interval_s
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

        race_collisions = self._race_collisions(collision_count)
        race_disqualified = race_collisions > 10
        terminated = self._race_laps_count >= self.race_laps or race_disqualified
        self._last_collision_count = collision_count
        self._last_snap = snap
        info = self._build_info(snap, metrics, collision_event=collision_event)
        info.update({
            "throttle_command": throttle,
            "steering_command": steering,
            "step_duration_s": step_duration_s,
            "official_lap_count": current_lap_count,
            "laps_since_start": max(0, current_lap_count - self._initial_lap_count),
            "warmup_lap_times_s": list(self._warmup_lap_times_s),
            "race_lap_times_s": list(self._race_lap_times_s),
            "race_last_lap_time_s": (
                self._race_lap_times_s[-1] if self._race_lap_times_s else None
            ),
            "race_best_lap_time_s": (
                min(self._race_lap_times_s) if self._race_lap_times_s else None
            ),
            "lap_times_s": list(self._race_lap_times_s),
            "race_laps_completed": self._race_laps_count,
            "watch_lap_count": self._race_laps_count,
            "race_collisions": race_collisions,
            "race_collision_baseline": self._race_collision_baseline,
            "warmup_collisions": self._warmup_collision_count,
            "raw_collision_count": collision_count,
            "race_time_s": sum(self._race_lap_times_s),
            "race_complete": self._race_laps_count >= self.race_laps,
            "race_disqualified": race_disqualified,
            "lidar_scan_rate_hz": float(snap.lidar_scan_rate),
            "control_interval_s": float(
                getattr(self.racer, "last_control_interval_s", 0.0)
            ),
            "control_interval_source": getattr(self.racer, "control_interval_source", "unknown"),
            "scan_interval_source": getattr(self.racer, "scan_interval_source", "unknown"),
            "control_interval_wall_s": float(
                getattr(self.racer, "last_control_wall_interval_s", 0.0)
            ),
        })
        observation = self._observation_builder.observe(
            snap,
            throttle,
            steering,
            elapsed_s=step_duration_s,
        )
        # The encoder-derived ``true_speed`` is useful for diagnostics but can
        # report wheel spin while the body is stopped at a wall. Watch displays
        # the same acceleration-guarded LiDAR estimate used by the policy.
        info["watch_speed_mps"] = abs(
            float(self._observation_builder.last_forward_speed_mps)
        )
        reward = 0.0
        truncated = False
        if self.training_mode:
            reward_components = {
                "race_time_cost": -self.training_time_cost_per_simulated_second * step_duration_s,
                "warmup_completion": 0.0,
                "lap_completion": 0.0,
                "collision_penalty": 0.0,
                "failure_penalty": 0.0,
            }
            self._training_elapsed_s += step_duration_s
            warmup_laps_completed = max(
                0,
                min(
                    self.warmup_laps,
                    current_lap_count - self._initial_lap_count,
                ),
            )
            warmup_laps_completed_previous = max(
                0,
                min(
                    self.warmup_laps,
                    previous_lap_count - self._initial_lap_count,
                ),
            )
            reward_components["warmup_completion"] = self.training_lap_reward * max(
                0, warmup_laps_completed - warmup_laps_completed_previous
            )
            if race_was_active:
                new_collisions = max(0, race_collisions - previous_race_collisions)
                reward_components["collision_penalty"] = -sum(
                    self.training_collision_penalty_base * collision_index
                    for collision_index in range(
                        previous_race_collisions + 1,
                        previous_race_collisions + new_collisions + 1,
                    )
                )
                new_laps = max(0, self._race_laps_count - previous_race_laps)
                reward_components["lap_completion"] = self.training_lap_reward * new_laps
                if race_disqualified:
                    reward_components["failure_penalty"] = -self.training_failure_penalty
            if (
                not terminated
                and self.training_timeout_s > 0
                and self._training_elapsed_s >= self.training_timeout_s
            ):
                truncated = True
                reward_components["failure_penalty"] = -self.training_failure_penalty
            reward = float(sum(reward_components.values()))
            info["training_reward_components"] = reward_components
            info["training_elapsed_s"] = self._training_elapsed_s
            if truncated:
                info["termination_reason"] = "training_watchdog_timeout"
            elif self._race_laps_count >= self.race_laps:
                info["termination_reason"] = "race_complete"
            elif race_disqualified:
                info["termination_reason"] = "disqualified"
        # The official scoring harness is deterministic evaluation, not PPO
        # training; scores come from official lap timing and collision counts.
        return observation, reward, terminated, truncated, info

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
            # Restricted state is monitoring-only: it can support training
            # reward/evaluation but is intentionally excluded from observation.
            "race_position": metrics.position,
            # Watch may render the restricted training/evaluation position;
            # this field is not read by the policy observation builder.
            "position": metrics.position,
            "yaw": float(snap.heading_yaw),
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
            "official_race": True,
            "lap_supported": True,
            "lidar_range_min": float(snap.lidar_range_min),
            "lidar_range_max": float(snap.lidar_range_max),
            "lidar_beams": int(snap.lidar_ranges.size),
        }

    def close(self) -> None:
        if self._owns_racer:
            self.racer.kill()


class OfficialObservationBuilder:
    """Canonical sensor-only observation transform for official training and race.

    Keep this stateful transform shared: a small difference in speed filtering,
    scan history, or update ordering changes the policy's effective dynamics.
    """

    def __init__(self, *, observation_profile: str = "official_sensors") -> None:
        if observation_profile not in (
            "official_sensors", "official_sensors_history", "official_sensors_camera"
        ):
            raise ValueError("unsupported official race observation profile")
        self.observation_profile = observation_profile
        self.include_camera = observation_profile == "official_sensors_camera"
        self._lidar_history: list[np.ndarray] = []
        self._lidar_speed_estimator = LidarOdometry()
        self._observation_heading_yaw: Optional[float] = None
        self._last_lidar_forward_speed_mps = 0.0
        self.last_forward_speed_mps = 0.0

    def reset(self, snap: TelemetrySnapshot) -> Dict[str, np.ndarray]:
        self._lidar_history = []
        self._lidar_speed_estimator.reset()
        self._observation_heading_yaw = float(snap.heading_yaw)
        self._last_lidar_forward_speed_mps = 0.0
        self.last_forward_speed_mps = 0.0
        if self.observation_profile == "official_sensors_history":
            scan = _normalize_lidar(snap)
            self._lidar_history = [scan.copy(), scan.copy(), scan.copy()]
        history = (
            self._lidar_history
            if self.observation_profile == "official_sensors_history"
            else None
        )
        return snapshot_to_obs(
            snap,
            0.0,
            0.0,
            forward_speed_mps=0.0,
            lateral_speed_mps=0.0,
            lidar_history=history,
            include_camera=self.include_camera,
        )

    def observe(
        self,
        snap: TelemetrySnapshot,
        prev_throttle: float,
        prev_steering: float,
        *,
        elapsed_s: float,
    ) -> Dict[str, np.ndarray]:
        previous_yaw = self._observation_heading_yaw
        yaw_delta = None
        if previous_yaw is not None:
            yaw_delta = float(
                np.arctan2(
                    np.sin(float(snap.heading_yaw) - previous_yaw),
                    np.cos(float(snap.heading_yaw) - previous_yaw),
                )
            )
        motion = self._lidar_speed_estimator.update(
            snap.lidar_ranges,
            elapsed_s,
            yaw_delta_rad=yaw_delta,
        )
        self._observation_heading_yaw = float(snap.heading_yaw)
        forward_speed = 0.0
        if motion.valid:
            forward_speed = acceleration_consistent_lidar_speed(
                motion.forward_m,
                self._last_lidar_forward_speed_mps,
                float(snap.linear_acceleration[0]),
                elapsed_s,
            )
        self._last_lidar_forward_speed_mps = forward_speed
        self.last_forward_speed_mps = forward_speed
        history = (
            self._lidar_history
            if self.observation_profile == "official_sensors_history"
            else None
        )
        observation = snapshot_to_obs(
            snap,
            prev_throttle,
            prev_steering,
            forward_speed_mps=forward_speed,
            lateral_speed_mps=0.0,
            lidar_history=history,
            include_camera=self.include_camera,
        )
        if history is not None:
            self._lidar_history.append(_normalize_lidar(snap))
            self._lidar_history = self._lidar_history[-3:]
        return observation
