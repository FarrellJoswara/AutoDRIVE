"""Unit tests for the Layer 3 LiDAR gap-following diagnostic controller."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.layer3.lidar_gap_policy import LidarGapPolicy
from src.layer3.official_policy import load_policy


class LidarGapPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = LidarGapPolicy()

    @staticmethod
    def observation(ranges: np.ndarray, speed_mps: float = 0.0) -> dict[str, np.ndarray]:
        state = np.zeros(9, dtype=np.float32)
        state[0] = speed_mps / 22.88
        return {"lidar": ranges.astype(np.float32), "state": state}

    def test_open_track_prefers_straight_ahead(self) -> None:
        action = self.policy.action(self.observation(np.ones(1081, dtype=np.float32)))
        self.assertGreater(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, delta=0.02)
        self.assertAlmostEqual(self.policy.last_debug["target_angle_rad"], 0.0, delta=0.02)
        self.assertEqual(self.policy.last_debug["forward_min_10deg_m"], 10.0)

    def test_obstacle_left_routes_toward_clear_right_gap(self) -> None:
        ranges = np.ones(1081, dtype=np.float32)
        angles = np.linspace(-135.0, 135.0, 1081)
        ranges[np.abs(angles - 35.0) < 9.0] = 0.04
        action = self.policy.action(self.observation(ranges))
        self.assertGreater(float(action[1]), 0.0)

    def test_obstacle_right_routes_toward_clear_left_gap(self) -> None:
        ranges = np.ones(1081, dtype=np.float32)
        angles = np.linspace(-135.0, 135.0, 1081)
        ranges[np.abs(angles + 35.0) < 9.0] = 0.04
        action = self.policy.action(self.observation(ranges))
        self.assertLess(float(action[1]), 0.0)

    def test_throttle_is_forward_and_reduces_at_speed(self) -> None:
        ranges = np.ones(1081, dtype=np.float32)
        stopped = self.policy.action(self.observation(ranges, 0.0))
        fast = self.policy.action(self.observation(ranges, 6.0))
        self.assertGreater(float(stopped[0]), 0.0)
        self.assertEqual(float(fast[0]), 0.0)
        self.assertGreater(float(stopped[0]), float(fast[0]))

    def test_rejects_invalid_scan_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "1081"):
            self.policy.action(self.observation(np.ones(1080, dtype=np.float32)))

    def test_shared_official_loader_constructs_gap_policy_without_checkpoint(self) -> None:
        policy = load_policy(Path("unused.zip"), controller="lidar_gap")
        action, state = policy.predict(self.observation(np.ones(1081, dtype=np.float32)))
        self.assertIsNone(state)
        self.assertEqual(action.shape, (2,))


if __name__ == "__main__":
    unittest.main()
