"""Small reactive LiDAR teacher for collecting safe driving examples."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class LidarCorridorTeacher:
    """Steer toward the center of the clearest forward LiDAR gap.

    This is intentionally a tiny reactive baseline: it uses no map, pose,
    route projection, or learned model. The same normalized Layer 2 LiDAR
    array that is saved for behavior cloning selects the steering direction.
    """

    def __init__(
        self,
        route: Any = None,  # accepted for compatibility with existing setup
        *,
        max_speed_mps: float = 0.3,
        forward_fov_deg: float = 85.0,
        minimum_gap_m: float = 0.9,
        bubble_radius_m: float = 0.22,
        steering_angle_deg: float = 45.0,
    ) -> None:
        del route
        self.max_speed_mps = max(0.1, float(max_speed_mps))
        self.forward_fov_deg = float(np.clip(forward_fov_deg, 30.0, 120.0))
        self.minimum_gap_m = max(0.2, float(minimum_gap_m))
        self.bubble_radius_m = max(0.05, float(bubble_radius_m))
        self.steering_angle_deg = max(10.0, float(steering_angle_deg))

    def action(
        self,
        info: Dict[str, Any],
        observation: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        observation = observation or {}
        lidar = np.asarray(observation.get("lidar", []), dtype=np.float32).reshape(-1)
        if lidar.size != 1081 or not np.isfinite(lidar).all():
            return np.asarray([0.0, 0.0], dtype=np.float32)

        range_min = float(info.get("lidar_range_min", 0.06))
        range_max = float(info.get("lidar_range_max", 10.0))
        if not np.isfinite([range_min, range_max]).all() or range_max <= range_min:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        ranges_m = range_min + np.clip(lidar, 0.0, 1.0) * (range_max - range_min)

        # The official normalized scan runs from -135 to +135 degrees, with
        # positive angles on the vehicle's right. Positive steering turns
        # left, so negate the target bearing to steer toward it.
        angles_deg = np.linspace(-135.0, 135.0, lidar.size)
        step_deg = float(angles_deg[1] - angles_deg[0])
        in_view = np.abs(angles_deg) <= self.forward_fov_deg
        candidates = np.flatnonzero(in_view & (ranges_m >= self.minimum_gap_m))
        if candidates.size == 0:
            return np.asarray([0.0, 0.0], dtype=np.float32)

        # Expand the nearest obstacle by the car's half-width. This keeps the
        # chosen gap away from walls without building a map or fitting a path.
        blocked = np.zeros(lidar.size, dtype=bool)
        nearest = int(np.argmin(np.where(in_view, ranges_m, np.inf)))
        nearest_range = float(ranges_m[nearest])
        # Do not create an arbitrary steering bias when the track is open and
        # all visible returns are well beyond the car's safety distance.
        if nearest_range < self.minimum_gap_m:
            bubble_half_deg = float(np.degrees(np.arcsin(min(
                0.99, self.bubble_radius_m / max(nearest_range, self.bubble_radius_m)
            ))))
            half_beams = max(1, int(np.ceil(bubble_half_deg / step_deg)))
            blocked[max(0, nearest - half_beams):min(lidar.size, nearest + half_beams + 1)] = True
        free = in_view & (ranges_m >= self.minimum_gap_m) & ~blocked

        # Select the widest contiguous clear opening, with ties biased toward
        # straight ahead. Its middle is a simple, stable target direction.
        indices = np.flatnonzero(free)
        if not indices.size:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        gap = max(groups, key=lambda group: (len(group), -abs(float(np.mean(angles_deg[group])))))
        target_angle = float(np.mean(angles_deg[gap]))
        steering = float(np.clip(
            -target_angle / self.steering_angle_deg, -1.0, 1.0
        ))

        # Encoder speed can include wheel spin; prefer Layer 2's guarded LiDAR
        # speed estimate when available. Negative throttle is reverse on this
        # simulator, so the teacher only drives forward or coasts.
        speed = info.get("watch_speed_mps", info.get("true_speed"))
        try:
            speed_mps = max(0.0, float(speed))
        except (TypeError, ValueError):
            speed_mps = 0.0
        throttle = float(np.clip(0.22 * (self.max_speed_mps - speed_mps), 0.0, 0.1))
        front = ranges_m[np.abs(angles_deg) <= 10.0]
        if front.size:
            front_clearance = float(np.min(front))
            if front_clearance <= 0.22:
                throttle = 0.0
            else:
                # Preserve a small crawl when a bend puts the inside wall
                # close to the front sector. Zero throttle here left the
                # steering saturated but the car unable to turn away.
                throttle *= float(np.clip((front_clearance - 0.12) / 0.55, 0.2, 1.0))
        return np.asarray([throttle, steering], dtype=np.float32)

    def action_batch(
        self,
        infos: Any,
        observations: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if observations is None or "lidar" not in observations:
            raise ValueError("LiDAR gap teacher requires Layer 2 LiDAR observations")
        lidar_batch = np.asarray(observations["lidar"])
        return np.stack([
            self.action(
                info if isinstance(info, dict) else {},
                {"lidar": lidar_batch[index]},
            )
            for index, info in enumerate(infos)
        ])
