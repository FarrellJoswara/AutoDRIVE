"""Tests for privileged centerline demonstration labels and route sampling."""

from __future__ import annotations

import unittest

import numpy as np

from src.layer2.route_progress import RouteProgressTracker
from src.layer3.expert import CenterlineExpert


class CenterlineExpertTests(unittest.TestCase):
    def setUp(self) -> None:
        # Starts north (+Z), turns east (+X), then closes a four-sided route.
        points = np.asarray(
            [[0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        self.route = RouteProgressTracker(points)
        self.expert = CenterlineExpert(self.route)

    def test_route_sampling_wraps_across_closed_route(self) -> None:
        point, tangent = self.route.point_and_tangent_at(self.route.length_m + 1.0)
        np.testing.assert_allclose(point, [0.0, 1.0])
        np.testing.assert_allclose(tangent, [0.0, 1.0])
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

    def test_teacher_steers_toward_inside_of_next_turn(self) -> None:
        s = 4.5
        point, _ = self.route.point_and_tangent_at(s)
        action = self.expert.action({
            "position": (float(point[0]), 0.0, float(point[1])),
            "yaw": 0.0,
            "current_progress_m": s,
            "true_speed": 1.0,
            "v_long": 1.0,
        })
        # Target lies to the positive-yaw side, while simulator steering uses
        # the opposite sign from its positive-yaw rotation convention.
        self.assertLess(float(action[1]), 0.0)

    def test_missing_pose_returns_safe_neutral_action(self) -> None:
        np.testing.assert_allclose(self.expert.action({}), [0.0, 0.0])

    def test_teacher_brakes_on_total_speed_even_when_forward_speed_is_low(self) -> None:
        point, _ = self.route.point_and_tangent_at(0.0)
        action = self.expert.action({
            "position": (float(point[0]), 0.0, float(point[1])),
            "yaw": 0.0,
            "current_progress_m": 0.0,
            "true_speed": 4.0,
            "v_long": 0.1,
        })
        self.assertLess(float(action[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
