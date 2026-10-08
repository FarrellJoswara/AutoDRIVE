import unittest

import numpy as np

from src.layer3.lidar_gap_policy import LidarGapPolicy


class LidarGapPolicyTests(unittest.TestCase):
    def test_targets_center_of_widest_safe_gap_not_local_farthest_ray(self):
        # The longest safe opening spans indices 1..5. Its local maximum is
        # deliberately off-center to guard against edge-seeking target flips.
        ranges = np.asarray([0.1, 2.0, 9.0, 4.0, 4.0, 4.0, 0.1])

        target = LidarGapPolicy._widest_gap_center_index(ranges, 1.0)

        self.assertEqual(target, 3)

    def test_no_safe_gap_returns_none(self):
        ranges = np.asarray([0.0, 0.4, 0.8, 0.2, 0.0])

        self.assertIsNone(LidarGapPolicy._widest_gap_center_index(ranges, 1.0))

    def test_equal_width_gaps_keep_deterministic_first_gap_tie_break(self):
        ranges = np.asarray([2.0, 2.0, 0.0, 2.0, 2.0])

        self.assertEqual(LidarGapPolicy._widest_gap_center_index(ranges, 1.0), 0)


if __name__ == "__main__":
    unittest.main()
