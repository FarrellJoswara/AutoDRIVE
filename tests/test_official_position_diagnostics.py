from __future__ import annotations

import math
import unittest

from src.layer3.official_evaluate import summarize_evaluator_positions


class OfficialPositionDiagnosticsTests(unittest.TestCase):
    def test_closed_route_reports_path_but_zero_net_displacement(self) -> None:
        result = summarize_evaluator_positions(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
             [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )

        self.assertEqual(result["source"], "restricted_ips_evaluator_only")
        self.assertEqual(result["positions_collected"], 5)
        self.assertAlmostEqual(result["path_length_m"], 4.0)
        self.assertAlmostEqual(result["net_displacement_m"], 0.0)
        self.assertAlmostEqual(result["max_distance_from_start_m"], math.sqrt(2.0))
        self.assertAlmostEqual(result["max_3d_displacement_from_start_m"], math.sqrt(2.0))
        self.assertEqual(result["path_efficiency"], 0.0)
        self.assertEqual(result["sampled_xy_m"][0], [0.0, 0.0])
        self.assertEqual(result["sampled_xy_m"][-1], [0.0, 0.0])
        self.assertEqual(result["sampled_xyz_m"][0], [0.0, 0.0, 0.0])
        self.assertEqual(result["sampled_xyz_m"][-1], [0.0, 0.0, 0.0])

    def test_trace_is_bounded_and_keeps_endpoints(self) -> None:
        positions = [[float(index), 0.0, 0.0] for index in range(1001)]

        result = summarize_evaluator_positions(positions, max_points=51)

        self.assertEqual(len(result["sampled_xy_m"]), 51)
        self.assertEqual(result["sampled_xy_m"][0], [0.0, 0.0])
        self.assertEqual(result["sampled_xy_m"][-1], [1000.0, 0.0])
        self.assertAlmostEqual(result["path_length_m"], 1000.0)
        self.assertAlmostEqual(result["path_efficiency"], 1.0)

    def test_altitude_is_not_counted_as_race_path(self) -> None:
        result = summarize_evaluator_positions(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 100.0]]
        )

        self.assertEqual(result["path_length_m"], 0.0)
        self.assertEqual(result["net_displacement_m"], 0.0)
        self.assertIsNone(result["path_efficiency"])
        self.assertEqual(result["vertical_min_m"], 0.0)
        self.assertEqual(result["vertical_max_m"], 100.0)
        self.assertEqual(result["vertical_span_m"], 100.0)
        self.assertAlmostEqual(result["max_3d_displacement_from_start_m"], 100.0)

    def test_empty_trace_is_explicit(self) -> None:
        result = summarize_evaluator_positions([])

        self.assertEqual(result["positions_collected"], 0)
        self.assertIsNone(result["path_length_m"])
        self.assertEqual(result["sampled_xy_m"], [])
        self.assertEqual(result["sampled_xyz_m"], [])

    def test_sampling_limit_must_hold_both_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            summarize_evaluator_positions([[0.0, 0.0]], max_points=1)


if __name__ == "__main__":
    unittest.main()
