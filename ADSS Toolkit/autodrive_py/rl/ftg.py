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
        max_throttle: float = 0.66,
        min_throttle: float = 0.33,
    ):
        self.n_lidar = n_lidar
        self.gap_threshold_m = gap_threshold_m
        self.bubble_radius = bubble_radius
        self.max_throttle = max_throttle
        self.min_throttle = min_throttle

    def act(self, raw_ranges: np.ndarray) -> np.ndarray:
        # Work in meters
        lidar_norm = downsample_lidar(raw_ranges, n_lidar=self.n_lidar)
        ranges = lidar_norm * LIDAR_MAX_M

        # Safety bubble around closest point
        closest = int(np.argmin(ranges))
        lo = max(0, closest - self.bubble_radius)
        hi = min(self.n_lidar, closest + self.bubble_radius + 1)
        ranges = ranges.copy()
        ranges[lo:hi] = 0.0

        free = ranges > self.gap_threshold_m
        # Find largest contiguous free gap
        best_len = 0
        best_start = 0
        best_end = 0
        i = 0
        while i < self.n_lidar:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < self.n_lidar and free[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start, best_end = i, j
            i = j

        if best_len == 0:
            # emergency: steer away from closest obstacle, slow
            mid = self.n_lidar // 2
            steer = -1.0 if closest > mid else 1.0
            return encode_action(0.0, steer)

        # Aim at deepest point in gap (max range)
        gap = ranges[best_start:best_end]
        target = best_start + int(np.argmax(gap))
        # Map beam index → steering in [-1, 1] (left→right)
        steer = (target / max(self.n_lidar - 1, 1)) * 2.0 - 1.0
        # Slow when gap narrow or close walls
        min_r = float(np.min(ranges[ranges > 0])) if np.any(ranges > 0) else 0.0
        if best_len < self.n_lidar * 0.15 or min_r < 1.0:
            throttle = self.min_throttle
        else:
            throttle = self.max_throttle
        return encode_action(throttle, float(np.clip(steer, -1.0, 1.0)))
