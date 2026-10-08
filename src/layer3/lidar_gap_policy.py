"""Deterministic LiDAR gap-following baseline for the official sensor path.

This controller is a Layer 3 diagnostic and demonstration source. It consumes
only canonical Layer 2 LiDAR and encoder-derived forward speed, and emits the
same normalized throttle/steering action as PPO.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LidarGapPolicy:
    """Follow the clearest forward LiDAR gap with speed limited by steering."""

    def __init__(
        self,
        *,
        max_range_m: float = 10.0,
        minimum_gap_m: float = 0.7,
        safety_radius_m: float = 0.32,
        max_steer_angle_rad: float = 0.75,
        cruise_speed_mps: float = 4.0,
        minimum_speed_mps: float = 1.5,
        speed_gain: float = 0.24,
    ) -> None:
        self.max_range_m = float(max_range_m)
        self.minimum_gap_m = float(minimum_gap_m)
        self.safety_radius_m = float(safety_radius_m)
        self.max_steer_angle_rad = float(max_steer_angle_rad)
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.minimum_speed_mps = float(minimum_speed_mps)
        self.speed_gain = float(speed_gain)
        self.last_debug: dict[str, float] = {}
        if min(
            self.max_range_m,
            self.minimum_gap_m,
            self.safety_radius_m,
            self.max_steer_angle_rad,
            self.cruise_speed_mps,
            self.minimum_speed_mps,
            self.speed_gain,
        ) <= 0:
            raise ValueError("gap policy parameters must be positive")

    @staticmethod
    def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
        padded = np.concatenate(([False], mask, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size == 0:
            return None
        lengths = ends - starts
        index = int(np.argmax(lengths))
        return int(starts[index]), int(ends[index])

    @classmethod
    def _widest_gap_center_index(
        cls, ranges: np.ndarray, minimum_gap_m: float
    ) -> int | None:
        """Return the center ray of the widest safe contiguous opening."""
        gap = cls._longest_true_run(np.asarray(ranges) >= float(minimum_gap_m))
        if gap is None:
            return None
        start, end = gap
        return (start + end - 1) // 2

    def action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        ranges = np.asarray(observation["lidar"], dtype=np.float32).reshape(-1)
        if ranges.size != 1081 or not np.isfinite(ranges).all():
            raise ValueError("gap policy requires 1081 finite canonical LiDAR values")
        ranges = np.clip(ranges, 0.0, 1.0) * self.max_range_m

        # Canonical Layer 2 beam order spans -135°..+135°. Restrict planning to
        # the forward 210° so rear returns cannot pull the target behind us.
        angles = np.deg2rad(np.linspace(-135.0, 135.0, ranges.size))
        forward = np.abs(angles) <= np.deg2rad(105.0)
        sector = ranges[forward].copy()
        sector_angles = angles[forward]

        # Smooth isolated range spikes, then clear a vehicle-sized bubble around
        # the nearest return before searching for contiguous safe space.
        smooth = np.convolve(sector, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        nearest = int(np.argmin(smooth))
        nearest_distance = float(smooth[nearest])
        bubble_half_angle = np.arcsin(
            np.clip(self.safety_radius_m / max(nearest_distance, self.safety_radius_m), 0.0, 1.0)
        )
        smooth[np.abs(sector_angles - sector_angles[nearest]) <= bubble_half_angle] = 0.0

        target_idx = self._widest_gap_center_index(smooth, self.minimum_gap_m)
        if target_idx is None:
            scores = smooth - 0.5 * np.abs(sector_angles)
            target_idx = int(np.argmax(scores))

        target_angle = float(sector_angles[target_idx])
        # Quantized beam centers can put a fully open, symmetric scene half a
        # beam off-axis. Avoid steering oscillation for that measurement noise.
        if abs(target_angle) < np.deg2rad(2.0):
            target_angle = 0.0
        # Positive LiDAR angle points left; the official actuator's positive
        # steering turns right, so negate the angle to match measured convention.
        steering = float(np.clip(-target_angle / self.max_steer_angle_rad, -1.0, 1.0))

        state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        if state.size < 1 or not np.isfinite(state[0]):
            raise ValueError("gap policy requires finite encoder-derived forward speed")
        speed_mps = max(0.0, float(state[0]) * 22.88)
        target_speed = max(
            self.minimum_speed_mps,
            self.cruise_speed_mps * (1.0 - 0.62 * abs(steering)),
        )
        # Layer 2's bidirectional action mapping is used in competition mode;
        # keeping throttle positive makes this baseline incapable of reversing.
        # Zero is a meaningful coast command in AutoDRIVE. A positive throttle
        # floor would keep adding power after target speed is exceeded and make
        # the nominal speed limit ineffective; negative throttle is reverse,
        # not a service brake, so the policy deliberately coasts instead.
        throttle = float(np.clip(self.speed_gain * (target_speed - speed_mps), 0.0, 0.85))
        self.last_debug = {
            "target_angle_rad": target_angle,
            "nearest_distance_m": nearest_distance,
            "forward_min_10deg_m": float(np.min(ranges[np.abs(angles) <= np.deg2rad(10.0)])),
            "forward_min_30deg_m": float(np.min(ranges[np.abs(angles) <= np.deg2rad(30.0)])),
            "left_min_30_90deg_m": float(np.min(ranges[(angles >= np.deg2rad(30.0)) & (angles <= np.deg2rad(90.0))])),
            "right_min_30_90deg_m": float(np.min(ranges[(angles <= -np.deg2rad(30.0)) & (angles >= -np.deg2rad(90.0))])),
            "speed_mps": speed_mps,
            "target_speed_mps": target_speed,
        }
        return np.asarray([throttle, steering], dtype=np.float32)

    def predict(self, observation: dict[str, np.ndarray], deterministic: bool = True) -> tuple[np.ndarray, None]:
        """SB3-compatible predict surface for the shared official evaluator."""
        del deterministic
        return self.action(observation), None


def load_gap_policy() -> LidarGapPolicy:
    """Construct the fixed diagnostic controller without loading a checkpoint."""
    return LidarGapPolicy()
