"""Regression checks for extracted centerlines against checked-in map grids."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.layer4.hub.centerline import (
    _extract_valid_closed_route,
    _route_is_drivable,
    _thin_binary,
)
from src.layer4.hub.meshgen.occupancy import load_occupancy
from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir


ROOT = Path(__file__).resolve().parents[1]
MAP_IDS = ("porto", "berlin", "icra2026_classic", "icra2026_master")


class CenterlineGenerationTests(unittest.TestCase):
    def test_centerline_is_closed_continuous_and_inside_each_map(self) -> None:
        for map_id in MAP_IDS:
            with self.subTest(map_id=map_id):
                occupancy_dir = _occupancy_dir(ROOT / "simulator" / "maps" / map_id)
                grid = load_occupancy(occupancy_dir, _find_yaml(occupancy_dir))
                clearance = cv2.distanceTransform(
                    grid.free.astype(np.uint8), cv2.DIST_L2, 5
                )
                route = _extract_valid_closed_route(
                    grid, _thin_binary(grid.free.astype(np.uint8)), 0.25
                )

                segments = np.linalg.norm(np.diff(route, axis=0), axis=1)
                self.assertGreater(len(route), 40)
                self.assertLess(np.linalg.norm(route[0] - route[-1]), 1e-6)
                self.assertLessEqual(float(segments.max()), 0.30)
                self.assertGreater(float(segments.sum()), 10.0)
                self.assertTrue(
                    _route_is_drivable(
                        grid, route, clearance, minimum_clearance_m=0.12
                    )
                )

    def test_checked_in_centerline_csvs_pass_wall_and_loop_validation(self) -> None:
        for map_id in MAP_IDS:
            with self.subTest(map_id=map_id):
                occupancy_dir = _occupancy_dir(ROOT / "simulator" / "maps" / map_id)
                grid = load_occupancy(occupancy_dir, _find_yaml(occupancy_dir))
                path = occupancy_dir / "centerline.csv"
                values = []
                with path.open("r", encoding="utf-8", newline="") as stream:
                    for row in csv.reader(
                        line for line in stream
                        if not line.lstrip().startswith("#")
                    ):
                        if len(row) >= 2:
                            values.append([float(value) for value in row[:2]])
                route = np.asarray(values, dtype=np.float64)
                clearance = cv2.distanceTransform(
                    grid.free.astype(np.uint8), cv2.DIST_L2, 5
                )
                segments = np.linalg.norm(np.diff(route, axis=0), axis=1)
                self.assertGreater(len(route), 40)
                self.assertLess(np.linalg.norm(route[0] - route[-1]), 1e-6)
                self.assertLessEqual(float(segments.max()), 0.30)
                self.assertTrue(
                    _route_is_drivable(
                        grid, route, clearance, minimum_clearance_m=0.12
                    )
                )


if __name__ == "__main__":
    unittest.main()
