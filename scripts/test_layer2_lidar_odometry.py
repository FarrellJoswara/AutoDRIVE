"""Synthetic sensor-only validation for Layer 2 LiDAR odometry."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.layer2.lidar_odometry import LidarOdometry, estimate_scan_motion


def _room_scan(x: float, y: float, yaw: float, *, beams: int = 1081) -> np.ndarray:
    angles = np.linspace(-3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0, beams)
    ray_angles = angles + yaw
    dx = np.cos(ray_angles)
    dy = np.sin(ray_angles)
    candidates = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for wall_x in (-5.0, 8.0):
            t = (wall_x - x) / dx
            wall_y = y + t * dy
            candidates.append(np.where((t > 0.0) & (wall_y >= -4.0) & (wall_y <= 4.0), t, np.inf))
        for wall_y in (-4.0, 4.0):
            t = (wall_y - y) / dy
            wall_x = x + t * dx
            candidates.append(np.where((t > 0.0) & (wall_x >= -5.0) & (wall_x <= 8.0), t, np.inf))
    return np.minimum.reduce(candidates).astype(np.float32)


class LidarOdometryTests(unittest.TestCase):
    def test_recovers_forward_lateral_and_yaw_motion_from_scans(self) -> None:
        previous = _room_scan(0.0, 0.0, 0.0)
        current = _room_scan(0.10, 0.04, 0.02)

        estimate = estimate_scan_motion(previous, current)

        self.assertTrue(estimate.valid, estimate)
        self.assertAlmostEqual(estimate.forward_m, 0.10, delta=0.035)
        self.assertAlmostEqual(estimate.lateral_m, 0.04, delta=0.035)
        self.assertAlmostEqual(estimate.yaw_rad, 0.02, delta=0.012)

    def test_stateful_estimator_converts_scan_displacement_to_velocity(self) -> None:
        estimator = LidarOdometry()
        first = _room_scan(0.0, 0.0, 0.0)
        second = _room_scan(0.05, 0.0, 0.0)

        self.assertFalse(estimator.update(first, 0.025).valid)
        estimate = estimator.update(second, 0.025)

        self.assertTrue(estimate.valid, estimate)
        self.assertAlmostEqual(estimate.forward_m, 2.0, delta=0.5)
        self.assertAlmostEqual(estimate.lateral_m, 0.0, delta=0.2)

    def test_stateful_estimator_uses_imu_yaw_delta_as_scan_alignment_prior(self) -> None:
        estimator = LidarOdometry()
        first = _room_scan(0.0, 0.0, 0.0)
        second = _room_scan(0.10, 0.04, 0.02)

        self.assertFalse(estimator.update(first, 0.05).valid)
        estimate = estimator.update(second, 0.05, yaw_delta_rad=0.02)

        self.assertTrue(estimate.valid, estimate)
        self.assertAlmostEqual(estimate.forward_m, 2.0, delta=0.7)
        self.assertAlmostEqual(estimate.lateral_m, 0.8, delta=0.7)
        self.assertAlmostEqual(estimate.yaw_rad, 0.4, delta=0.02)

    def test_imu_yaw_prior_stabilizes_translation_fit(self) -> None:
        previous = _room_scan(0.0, 0.0, 0.0)
        current = _room_scan(0.10, 0.04, 0.02)

        estimate = estimate_scan_motion(previous, current, yaw_delta_rad=0.02)

        self.assertTrue(estimate.valid, estimate)
        self.assertAlmostEqual(estimate.forward_m, 0.10, delta=0.035)
        self.assertAlmostEqual(estimate.lateral_m, 0.04, delta=0.035)
        self.assertAlmostEqual(estimate.yaw_rad, 0.02, delta=1e-6)

    def test_reset_requires_a_new_baseline_scan(self) -> None:
        estimator = LidarOdometry()
        scan = _room_scan(0.0, 0.0, 0.0)
        estimator.update(scan, 0.025)
        estimator.reset()

        self.assertFalse(estimator.update(_room_scan(0.1, 0.0, 0.0), 0.025).valid)

    def test_rejects_invalid_scan_length(self) -> None:
        with self.assertRaises(ValueError):
            estimate_scan_motion(np.ones(64), np.ones(64))


if __name__ == "__main__":
    unittest.main()
