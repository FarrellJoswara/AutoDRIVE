"""Tests for privileged centerline demonstration labels and route sampling."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.layer2.route_progress import RouteProgressTracker
from src.layer3.expert import CenterlineExpert
from src.layer3.behavior_cloning import collect_centerline_demonstrations


class CenterlineExpertTests(unittest.TestCase):
    def setUp(self) -> None:
        # Official ROS ground-plane route starts east (+X), turns north (+Y).
        points = np.asarray(
            [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        self.route = RouteProgressTracker(points)
        self.expert = CenterlineExpert(self.route)

    def test_route_sampling_wraps_across_closed_route(self) -> None:
        point, tangent = self.route.point_and_tangent_at(self.route.length_m + 1.0)
        np.testing.assert_allclose(point, [1.0, 0.0])
        np.testing.assert_allclose(tangent, [1.0, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0)

    def test_teacher_stays_straight_on_straight_route_and_throttles(self) -> None:
        action = self.expert.action({
            "position": (0.0, 0.0, 0.0),
            "yaw": 0.0,
            "current_progress_m": 0.0,
            "true_speed": 0.0,
            "v_long": 0.0,
        })
        self.assertGreater(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=5)

    def test_teacher_accepts_official_planar_pose(self) -> None:
        action = self.expert.action({
            "position": (0.0, 0.0),
            "yaw": 0.0,
            "current_progress_m": 0.0,
            "true_speed": 0.0,
        })
        self.assertGreater(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=5)

    def test_teacher_uses_ros_xy_and_converts_unity_yaw(self) -> None:
        # Unity yaw 0 points +Unity Z, which the official map maps to +ROS X.
        # The third ROS coordinate is height and must not enter route steering.
        action = self.expert.action({
            "position": (2.0, 0.0, 99.0),
            "yaw": 0.0,
            "current_progress_m": 2.0,
            "true_speed": 0.0,
        })
        self.assertGreater(float(action[0]), 0.0)
        self.assertAlmostEqual(float(action[1]), 0.0, places=5)

    def test_teacher_steers_toward_inside_of_next_turn(self) -> None:
        s = 4.5
        point, _ = self.route.point_and_tangent_at(s)
        action = self.expert.action({
            "position": (float(point[0]), float(point[1]), 0.0),
            "yaw": 0.0,
            "current_progress_m": s,
            "true_speed": 1.0,
            "v_long": 1.0,
        })
        # Target lies to the positive ROS-y side; steering must turn left.
        self.assertGreater(float(action[1]), 0.0)

    def test_missing_pose_returns_safe_neutral_action(self) -> None:
        np.testing.assert_allclose(self.expert.action({}), [0.0, 0.0])

    def test_black_camera_demonstrations_are_rejected(self) -> None:
        class BlackCameraEnv:
            num_envs = 1
            observations = {
                "lidar": np.zeros((1, 2, 8), dtype=np.float32),
                "state": np.zeros((1, 6), dtype=np.float32),
                "camera": np.zeros((1, 90, 160, 3), dtype=np.uint8),
            }

            def reset(self):
                return self.observations

            def env_method(self, name):
                return [{}]

            def step(self, actions):
                return self.observations, np.zeros(1), np.zeros(1, dtype=bool), [{}]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "all collected camera frames are black"):
                collect_centerline_demonstrations(
                    BlackCameraEnv(), self.expert, steps=2,
                    output_path=Path(directory) / "demonstrations.npz",
                )

    def test_teacher_coasts_on_total_speed_instead_of_commanding_reverse(self) -> None:
        point, _ = self.route.point_and_tangent_at(0.0)
        action = self.expert.action({
            "position": (float(point[0]), float(point[1]), 0.0),
            "yaw": 0.0,
            "current_progress_m": 0.0,
            "true_speed": 4.0,
            "v_long": 0.1,
        })
        self.assertEqual(float(action[0]), 0.0)
        self.assertGreaterEqual(float(action[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
