"""Low-speed centerline teacher used only to generate imitation examples.

The teacher reads privileged map geometry and simulator pose from ``info``.
Those values are never added to the observation passed to PPO or the saved
policy, which continues to consume Layer 2's camera, LiDAR and vehicle state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from src.layer2.route_progress import RouteProgressTracker


class CenterlineExpert:
    """A cautious pure-pursuit teacher for collecting drive demonstrations."""

    def __init__(
        self,
        route: RouteProgressTracker,
        *,
        max_speed_mps: float = 1.2,
        wheelbase_m: float = 0.26,
        max_steer_angle_rad: float = 0.45,
        max_lateral_accel_mps2: float = 0.8,
        lookahead_base_m: float = 1.5,
    ) -> None:
        self.route = route
        self.max_speed_mps = max(0.2, float(max_speed_mps))
        self.wheelbase_m = max(0.05, float(wheelbase_m))
        self.max_steer_angle_rad = max(0.05, float(max_steer_angle_rad))
        self.max_lateral_accel_mps2 = max(0.1, float(max_lateral_accel_mps2))
        self.lookahead_base_m = max(0.5, float(lookahead_base_m))
        self._speed_samples_by_env: Dict[int, list[float]] = {}
        self._last_progress_by_env: Dict[int, float] = {}
        self._last_position_by_env: Dict[int, np.ndarray] = {}

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _scan_sector_min(
        lidar: np.ndarray, angle_low_deg: float, angle_high_deg: float,
        range_min_m: float, range_max_m: float,
    ) -> float:
        """Return a robust nearby return for a sector in the canonical scan.

        Layer 2's 1081-beam scan spans -135..+135 degrees. In the official
        simulator's calibrated scan after Layer 1 canonicalization,
        positive-angle beams measure the vehicle's left wall and
        negative-angle beams its right wall.
        """
        values = np.asarray(lidar, dtype=np.float32).reshape(-1)
        if values.size != 1081 or not np.isfinite(values).all():
            return float(range_max_m)
        # Layer 1 reverses increasing-angle ROS scans into its canonical
        # left-to-right beam order.
        angles = np.linspace(135.0, -135.0, values.size)
        selected = values[(angles >= angle_low_deg) & (angles <= angle_high_deg)]
        if selected.size == 0:
            return float(range_max_m)
        # A low percentile ignores isolated invalid echoes while remaining
        # sensitive to a wall filling a meaningful part of the sector.
        normalized = float(np.percentile(selected, 8.0))
        return float(range_min_m + normalized * (range_max_m - range_min_m))

    def action(
        self,
        info: Dict[str, Any],
        lidar: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        *,
        env_index: int = 0,
    ) -> np.ndarray:
        """Return [throttle, steering] in [-1, 1] for an environment info row."""
        position = info.get("position")
        if position is None or len(position) < 2:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        # The official IPS topic is ROS XYZ and its track plane is XY. The
        # centerline is in that same XY frame, so ignore the vertical Z value.
        x, z = float(position[0]), float(position[1])
        if not np.isfinite([x, z, float(info.get("yaw", float("nan")))]).all():
            return np.asarray([0.0, 0.0], dtype=np.float32)
        # Layer 1 yaw is Unity Y-up yaw (zero points Unity +Z). The official
        # map transform is ROS (x, y) = (Unity z, -Unity x + offset), so the
        # route-bearing convention atan2(dx, dy) is yaw + pi/2.
        yaw = self._wrap_angle(float(info.get("yaw", 0.0)) + math.pi / 2.0)
        # Project measured pose directly. Reward frontier/progress may be
        # constrained or stale and must not select the teacher's target.
        route_s = self.route.project_pose(x, z, enforce_corridor=False)
        if route_s is None or not math.isfinite(float(route_s)):
            return np.asarray([0.0, 0.0], dtype=np.float32)
        previous_progress = self._last_progress_by_env.get(env_index)
        if previous_progress is not None and float(route_s) < previous_progress - 0.2:
            self._speed_samples_by_env.pop(env_index, None)
        self._last_progress_by_env[env_index] = float(route_s)

        # Encoder speed can spike while the chassis is barely moving, while
        # LiDAR odometry can under-report through repeated near-identical
        # scans. Estimate chassis speed from successive official IPS positions
        # and the simulator's measured action interval whenever available.
        position_now = np.asarray([x, z], dtype=np.float64)
        position_before = self._last_position_by_env.get(env_index)
        self._last_position_by_env[env_index] = position_now
        measured_speed = float("nan")
        try:
            interval = float(info.get("step_duration_s") or info.get("control_interval_s"))
        except (TypeError, ValueError):
            interval = float("nan")
        if (
            position_before is not None
            and np.isfinite(interval)
            and 0.005 <= interval <= 2.0
        ):
            displacement = float(np.linalg.norm(position_now - position_before))
            # A reset/teleport is not physical speed; discard unusually large
            # jumps and use sensor speed as a fallback for that sample.
            if displacement <= 1.0:
                measured_speed = displacement / interval
        if not np.isfinite(measured_speed):
            for key in ("watch_speed_mps", "true_speed"):
                try:
                    candidate = abs(float(info.get(key)))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(candidate):
                    measured_speed = candidate
                    break
        else:
            # IPS displacement is the primary body-speed estimate, but it can
            # lag one control interval. Use encoder speed as a conservative
            # upper bound so the throttle controller does not keep accelerating
            # through a bend while IPS catches up. Clamp wheel-spin outliers.
            try:
                encoder_speed = abs(float(info.get("true_speed")))
            except (TypeError, ValueError):
                encoder_speed = float("nan")
            if np.isfinite(encoder_speed):
                measured_speed = max(
                    measured_speed,
                    min(encoder_speed, self.max_speed_mps * 1.05),
                )
        if np.isfinite(measured_speed):
            speed_samples = self._speed_samples_by_env.setdefault(env_index, [])
            speed_samples.append(measured_speed)
            if len(speed_samples) > 5:
                del speed_samples[:-5]
            # The median treated three missed velocity samples as a true stop.
            # The upper quartile holds throttle off when recent samples show
            # the car is already moving, while still recovering after resets.
            speed = float(np.percentile(speed_samples, 75.0))
        else:
            speed = 0.0
        if not np.isfinite(measured_speed) and state is not None:
            state_values = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_values.size and np.isfinite(state_values[0]):
                speed = abs(float(state_values[0]) * 22.88)
        # Keep the preview short enough to follow a tight bend instead
        # of cutting toward the outer wall; let it grow slightly with speed.
        lookahead = float(np.clip(self.lookahead_base_m + 0.35 * speed, 0.9, 2.5))
        target, _ = self.route.point_and_tangent_at(float(route_s) + lookahead)
        # Pure pursuit in the map plane: steering is determined solely by
        # measured pose and the target on the route. Use actual chord length,
        # including lateral error, rather than the distance along the route.
        dx, dy = float(target[0]) - x, float(target[1]) - z
        distance_squared = dx * dx + dy * dy
        if distance_squared < 1e-8:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        lateral_right = dx * math.cos(yaw) - dy * math.sin(yaw)
        curvature = 2.0 * lateral_right / distance_squared
        steering_angle = math.atan(self.wheelbase_m * curvature)
        # Positive actuator steering decreases Unity yaw (left in map XY).
        steering = float(np.clip(
            -steering_angle / self.max_steer_angle_rad, -1.0, 1.0
        ))

        # LiDAR still limits throttle for front contact risk.
        # Lower the teacher's target speed before tighter bends. Curvature is
        # estimated from the route tangents on either side of the target.
        ahead = float(route_s) + lookahead
        _, tangent_before = self.route.point_and_tangent_at(ahead)
        _, tangent_after = self.route.point_and_tangent_at(ahead + 1.0)
        turn = self._wrap_angle(
            math.atan2(float(tangent_after[0]), float(tangent_after[1]))
            - math.atan2(float(tangent_before[0]), float(tangent_before[1]))
        )
        curvature_ahead = abs(turn)
        target_speed = min(
            self.max_speed_mps,
            math.sqrt(self.max_lateral_accel_mps2 / max(curvature_ahead, 0.08)),
        )
        speed_error = target_speed - speed
        # Negative AutoDRIVE throttle commands reverse, not braking. Let the
        # car coast whenever it is at/above target speed; positive throttle
        # closes the speed gap without teaching the policy to drive backward.
        # The official car coasts for a long distance after throttle is cut.
        # Keep teacher acceleration gentle so the speed target remains
        # achievable before a bend instead of relying on zero throttle to brake.
        # The official bridge accepts normalized throttle, and a 0.005 cap
        # left the teacher stationary for thousands of control steps. Keep a
        # low target speed but provide enough command to turn through a bend.
        throttle = float(np.clip(0.12 * speed_error, 0.0, 0.035))
        if lidar is not None:
            range_min = float(info.get("lidar_range_min", 0.06))
            range_max = float(info.get("lidar_range_max", 10.0))
            if (
                np.isfinite([range_min, range_max]).all()
                and range_min >= 0.0
                and range_max > range_min
            ):
                front_clearance = self._scan_sector_min(
                    lidar, -12.0, 12.0, range_min, range_max
                )
                # Stop for immediate contact risk, but retain a crawl at
                # ordinary corner clearances so the steering can carry the
                # car around the inside wall instead of pinning it in place.
                if front_clearance <= 0.18:
                    throttle = 0.0
                else:
                    throttle *= float(np.clip(
                        (front_clearance - 0.12) / 0.42, 0.35, 1.0
                    ))
        return np.asarray([throttle, steering], dtype=np.float32)

    def action_batch(self, infos: Any, observations: Optional[Dict[str, Any]] = None) -> np.ndarray:
        lidar_batch = None if observations is None else observations.get("lidar")
        state_batch = None if observations is None else observations.get("state")
        actions = []
        for index, row in enumerate(infos):
            lidar = None if lidar_batch is None else np.asarray(lidar_batch[index])
            state = None if state_batch is None else np.asarray(state_batch[index])
            actions.append(self.action(
                row if isinstance(row, dict) else {}, lidar, state,
                env_index=index,
            ))
        return np.stack(actions)
