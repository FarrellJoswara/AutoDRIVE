"""Low-speed centerline teacher used only to generate imitation examples.

The teacher reads privileged map geometry and simulator pose from ``info``.
Those values are never added to the observation passed to PPO or the saved
policy, which continues to consume only Layer 2's LiDAR and vehicle state.
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
    ) -> None:
        self.route = route
        self.max_speed_mps = max(0.5, float(max_speed_mps))
        self.wheelbase_m = max(0.05, float(wheelbase_m))
        self.max_steer_angle_rad = max(0.05, float(max_steer_angle_rad))
        self.max_lateral_accel_mps2 = max(0.1, float(max_lateral_accel_mps2))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def action(self, info: Dict[str, Any]) -> np.ndarray:
        """Return [throttle, steering] in [-1, 1] for an environment info row."""
        position = info.get("position")
        if position is None or len(position) < 3:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        x, z = float(position[0]), float(position[2])
        yaw = float(info.get("yaw", 0.0))
        route_s = info.get("current_progress_m")
        if route_s is None:
            route_s = self.route.project_pose(x, z)
        if route_s is None or not math.isfinite(float(route_s)):
            return np.asarray([0.0, 0.0], dtype=np.float32)

        speed = max(0.0, float(info.get("true_speed", abs(info.get("v_long", 0.0))) or 0.0))
        lookahead = float(np.clip(0.9 + 0.35 * speed, 0.9, 2.0))
        target, _ = self.route.point_and_tangent_at(float(route_s) + lookahead)
        target_bearing = math.atan2(float(target[0]) - x, float(target[1]) - z)
        alpha = self._wrap_angle(target_bearing - yaw)
        curvature = 2.0 * math.sin(alpha) / lookahead
        steering_angle = math.atan(self.wheelbase_m * curvature)
        # AutoDRIVE's V1 steering command has the opposite sign from positive
        # Unity yaw: a positive command decreases yaw. The Porto smoke trajectory
        # measured this directly (positive labels sent the car outside the right
        # wall on a left-yawing segment), so invert the bicycle-model command.
        steering = float(
            np.clip(-steering_angle / self.max_steer_angle_rad, -1.0, 1.0)
        )

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
        # Use total ground speed here, not body-longitudinal speed. Porto's
        # pilot showed large sideways velocity in the tight bend; v_long alone
        # then looked slow and kept applying throttle while the car was already
        # moving dangerously fast.
        speed_error = target_speed - speed
        throttle = float(np.clip(0.3 * speed_error, -0.55, 0.2))
        return np.asarray([throttle, steering], dtype=np.float32)

    def action_batch(self, infos: Any) -> np.ndarray:
        return np.stack([self.action(row if isinstance(row, dict) else {}) for row in infos])
