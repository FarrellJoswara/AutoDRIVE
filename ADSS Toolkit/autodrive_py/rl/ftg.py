"""Follow-the-Gap planner (classical baseline)."""

from __future__ import annotations

import numpy as np

from .contracts import encode_action
from .observation import downsample_lidar
from .contracts import LIDAR_MAX_M, N_LIDAR_DEFAULT


class FollowTheGap:
    """
    Classic FTG on LiDAR ranges.
    Returns MultiDiscrete action indices compatible with contracts.
    """

    def __init__(
        self,
        n_lidar: int = N_LIDAR_DEFAULT,
        gap_threshold_m: float = 1.5,
        bubble_radius: int = 8,
        max_throttle: float = 1.0,
        min_throttle: float = 0.33,
        *,
        fov_rad: float = 4.7,
        steer_max_rad: float = 0.4189,
        forward_arc_rad: float = 1.75,
        brake_distance_m: float = 6.0,
    ):
        self.n_lidar = n_lidar
        self.gap_threshold_m = gap_threshold_m
        self.bubble_radius = bubble_radius
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle
        self.fov_rad = float(fov_rad)
        self.steer_max_rad = float(steer_max_rad)
        # The scan spans ~269°, so the largest gap is often *behind* the car.
        # Only beams within this half-angle of straight ahead may be targeted.
        self.forward_arc_rad = float(forward_arc_rad)
        self.brake_distance_m = float(brake_distance_m)

    def act(self, raw_ranges: np.ndarray) -> np.ndarray:
        # Work in meters
        lidar_norm = downsample_lidar(raw_ranges, n_lidar=self.n_lidar)
        ranges_full = lidar_norm * LIDAR_MAX_M

        angles = np.linspace(-self.fov_rad / 2.0, self.fov_rad / 2.0, self.n_lidar)
        window = np.flatnonzero(np.abs(angles) <= self.forward_arc_rad)
        if window.size == 0:
            window = np.arange(self.n_lidar)
        base = int(window[0])
        ranges = ranges_full[window].copy()
        win_angles = angles[window]
        n = int(ranges.size)

        # Safety bubble around closest point
        closest = int(np.argmin(ranges))
        lo = max(0, closest - self.bubble_radius)
        hi = min(n, closest + self.bubble_radius + 1)
        ranges[lo:hi] = 0.0

        free = ranges > self.gap_threshold_m
        # Find largest contiguous free gap
        best_len = 0
        best_start = 0
        best_end = 0
        i = 0
        while i < n:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < n and free[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start, best_end = i, j
            i = j

        front_min = self._front_clearance(ranges_full, angles)

        if best_len == 0:
            # emergency: steer away from closest obstacle, slow
            steer = -1.0 if win_angles[closest] > 0.0 else 1.0
            return encode_action(0.0, steer)

        # Aim at the deepest part of the gap. Open track saturates most beams at
        # max range, so argmax alone would lock onto the first tied beam and pull
        # the car to one edge — take the middle of the deepest plateau instead.
        gap = ranges[best_start:best_end]
        depth = float(np.max(gap))
        plateau = np.flatnonzero(gap >= depth - 1e-3)
        target = best_start + int(plateau[plateau.size // 2])

        # Beam angle → steering, so a target within one steering lock maps to
        # proportional command rather than being scaled by the whole 269° FOV.
        target_angle = float(win_angles[target])
        steer = target_angle / max(self.steer_max_rad, 1e-6)

        narrow = best_len < n * 0.15
        throttle = self.max_throttle
        if narrow or front_min < 1.5:
            throttle = self.min_throttle
        elif front_min < self.brake_distance_m:
            # Ease off as the wall ahead closes in.
            span = max(self.brake_distance_m - 1.5, 1e-6)
            frac = (front_min - 1.5) / span
            throttle = self.min_throttle + frac * (self.max_throttle - self.min_throttle)

        _ = base  # window offset kept for clarity when debugging beam indices
        return encode_action(float(throttle), float(np.clip(steer, -1.0, 1.0)))

    def _front_clearance(self, ranges_full: np.ndarray, angles: np.ndarray) -> float:
        """Nearest return in a narrow cone straight ahead (drives braking)."""
        cone = np.flatnonzero(np.abs(angles) <= 0.30)
        if cone.size == 0:
            return float(np.min(ranges_full))
        return float(np.min(ranges_full[cone]))
