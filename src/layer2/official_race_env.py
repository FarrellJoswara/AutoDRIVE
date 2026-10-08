"""Layer 2 Gym adapter for official-race evaluation and training.

Policy observations/actions use the same sensor-only transform in both modes.
Training mode may use restricted race counters for reward/episode control and
the documented reset topic; neither capability is enabled for policy runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
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
from .route_progress import RouteProgressTracker
from .rewards import RewardConfig, compute_reward_components


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
        training_time_cost_per_simulated_second: float = 5.0,
        training_collision_penalty_magnitude: float = 100.0,
        training_collision_reward_percent: float = 20.0,
        training_failure_penalty: float = 100.0,
        training_frontier_stagnation_s: float = 10.0,
        frontier_path: Optional[Path] = None,
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
        self.training_failure_penalty = max(0.0, float(training_failure_penalty))
        self.training_time_cost_per_simulated_second = max(
            0.0, float(training_time_cost_per_simulated_second)
        )
        self.training_collision_penalty_magnitude = max(0.0, float(training_collision_penalty_magnitude))
        self.training_collision_reward_percent = max(0.0, float(training_collision_reward_percent))
        self.training_frontier_stagnation_s = max(0.0, float(training_frontier_stagnation_s))
        self.frontier_path = Path(frontier_path) if frontier_path else None
        self._frontier_spawn_xy: Optional[Tuple[float, float]] = None
        if self.frontier_path is not None:
            metadata_path = self.frontier_path.with_name("official_frontier.json")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                spawn = metadata.get("spawn_xy") if isinstance(metadata, dict) else None
                if isinstance(spawn, (list, tuple)) and len(spawn) == 2:
                    self._frontier_spawn_xy = (float(spawn[0]), float(spawn[1]))
            except (OSError, ValueError, TypeError) as exc:
                if self.training_mode:
                    raise ValueError(f"official training frontier metadata is unavailable: {metadata_path}") from exc
            if self.training_mode and self._frontier_spawn_xy is None:
                raise ValueError(f"official training frontier metadata has no valid spawn_xy: {metadata_path}")
        self.route_progress: Optional[RouteProgressTracker] = (
            RouteProgressTracker.from_csv(self.frontier_path)
            if self.training_mode and self.frontier_path is not None else None
        )
        if self.training_mode and self.route_progress is None:
            raise ValueError("official training requires a validated frontier_path")
        self._positive_episode_return = 0.0
        self._episode_frontier_distance_m = 0.0
        self._frontier_last_push_s = 0.0
        self._last_training_steering = 0.0
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
        self._positive_episode_return = 0.0
        self._episode_frontier_distance_m = 0.0
        self._frontier_last_push_s = 0.0
        self._last_training_steering = 0.0
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
        self._positive_episode_return = 0.0
        self._episode_frontier_distance_m = 0.0
        self._frontier_last_push_s = 0.0
        self._last_training_steering = 0.0
        if self.route_progress is not None:
            position = metrics.position
            if position is None:
                raise RuntimeError("official training reset did not provide restricted IPS position")
            if self._frontier_spawn_xy is not None:
                spawn_error = float(np.linalg.norm(
                    np.asarray(position[:2], dtype=np.float64)
                    - np.asarray(self._frontier_spawn_xy, dtype=np.float64)
                ))
                if spawn_error > 0.75:
                    raise RuntimeError(
                        "official training reset position is not at the configured IPS spawn "
                        f"({spawn_error:.2f} m away); refusing to anchor the route frontier"
                    )
            self.route_progress.reset(float(position[0]), float(position[1]), now=0.0)
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
            self._training_elapsed_s += step_duration_s
            position = metrics.position
            progress = None
            if self.route_progress is not None and position is not None:
                progress = self.route_progress.update(
                    float(position[0]), float(position[1]), now=self._training_elapsed_s
                )
            frontier_advanced = max(0.0, float((progress or {}).get("advanced_m", 0.0)))
            self._episode_frontier_distance_m += frontier_advanced
            if frontier_advanced > 0:
                self._frontier_last_push_s = self._training_elapsed_s
            average_frontier_speed = (
                self._episode_frontier_distance_m / self._training_elapsed_s
                if self._training_elapsed_s > 0 else 0.0
            )
            frontier_stalled = (
                self.training_frontier_stagnation_s > 0
                and self._training_elapsed_s - self._frontier_last_push_s >= self.training_frontier_stagnation_s
            )
            collision_termination = bool(collision_event)
            terminated = collision_termination or frontier_stalled
            truncated = bool(
                not terminated and self.training_timeout_s > 0
                and self._training_elapsed_s >= self.training_timeout_s
            )
            reason = "collision" if collision_termination else "frontier_stagnation" if frontier_stalled else None
            episode_failure = bool(truncated)
            components = compute_reward_components(
                v_long=float(snap.v_long),
                step_duration_s=step_duration_s,
                frontier_advanced_m=frontier_advanced,
                frontier_average_speed_mps=average_frontier_speed,
                frontier_current_speed_mps=(frontier_advanced / step_duration_s if step_duration_s > 0 else 0.0),
                positive_episode_return=self._positive_episode_return,
                collision_event=collision_termination,
                episode_failure=episode_failure,
                slip_angle=float(snap.slip_angle),
                prev_steering=self._last_training_steering,
                steering=float(steering),
                cfg=RewardConfig(
                    time_penalty_per_second=self.training_time_cost_per_simulated_second,
                    collision_penalty_magnitude=self.training_collision_penalty_magnitude,
                    collision_reward_percent=self.training_collision_reward_percent,
                    episode_failure_penalty_magnitude=self.training_failure_penalty,
                    episode_failure_reward_percent=100.0,
                ),
            )
            reward = float(components["total"])
            self._positive_episode_return += max(0.0, float(components["route_progress"]))
            self._last_training_steering = float(steering)
            info["training_reward_components"] = components
            info["frontier_progress_m"] = (progress or {}).get("progress_m")
            info["frontier_advanced_m"] = frontier_advanced
            info["frontier_speed_mps"] = average_frontier_speed
            # Keep the names consumed by Layer 4's Watch payload in sync with
            # the official training diagnostics. These are monitoring-only;
            # policy observations remain sensor-only.
            info["frontier_line"] = (progress or {}).get("line")
            info["current_progress_line"] = (progress or {}).get("current_line")
            info["current_progress_m"] = (progress or {}).get("current_progress_m")
            info["signed_route_delta_m"] = (progress or {}).get("current_delta_m")
            info["current_route_speed_mps"] = (
                frontier_advanced / step_duration_s if step_duration_s > 0 else 0.0
            )
            info["route_projection_valid"] = bool(
                progress.get("current_projection_valid") if progress is not None else False
            )
            info["reward_components"] = components
            info["time_since_frontier_push_s"] = (
                max(0.0, self._training_elapsed_s - self._frontier_last_push_s)
                if progress is not None else None
            )
            info["training_positive_frontier_return"] = self._positive_episode_return
            info["training_frontier_distance_m"] = self._episode_frontier_distance_m
            info["training_frontier_projection_valid"] = (
                bool(progress.get("current_projection_valid")) if progress is not None else False
            )
            info["training_frontier_stagnated"] = frontier_stalled
            info["training_elapsed_s"] = self._training_elapsed_s
            info["termination_reason"] = reason or ("training_watchdog_timeout" if truncated else None)
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
