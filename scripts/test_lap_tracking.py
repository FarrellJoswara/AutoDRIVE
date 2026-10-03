"""Unit and map-geometry smoke tests for centerline lap timing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.layer2.lap_tracker import LapTracker, map_lap_gate_config
from src.layer2.route_progress import RouteProgressTracker, map_centerline_path
from src.layer4.hub.lap_gate import get_map_lap_gate, save_map_lap_gate


ROOT = Path(__file__).resolve().parents[1]


class LapTrackerTests(unittest.TestCase):
    def test_reversing_back_and_forth_across_finish_gate_does_not_add_laps(self) -> None:
        route = RouteProgressTracker(
            np.asarray([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
        )
        start = route.reset(0.0, 0.0, now=0.0)
        laps = LapTracker(route)
        laps.reset(start, now=0.0)

        def pose_at(distance: float) -> tuple[float, float]:
            local = distance % route.length_m
            segment = min(
                max(int(np.searchsorted(route._cum, local, side="right") - 1), 0),
                len(route._lengths) - 1,
            )
            alpha = (local - route._cum[segment]) / max(route._lengths[segment], 1e-9)
            point = route.points[segment] + alpha * route._seg[segment]
            return float(point[0]), float(point[1])

        now = 0.0
        for distance in np.arange(0.5, route.length_m + 0.1, 0.5):
            now += 0.1
            x, z = pose_at(float(distance))
            progress = route.update(x, z, now=now)
            sample = laps.update(progress["progress_m"], now=now)
        self.assertEqual(sample["lap_count"], 1)

        # Repeatedly reverse over the physical gate and drive forward across it.
        # The route frontier stays near one lap; it must not count lap two.
        for _ in range(10):
            for distance in (route.length_m - 0.5, route.length_m + 0.5):
                now += 0.1
                x, z = pose_at(distance)
                progress = route.update(x, z, now=now)
                sample = laps.update(progress["progress_m"], now=now)
        self.assertEqual(sample["lap_count"], 1)
        self.assertLess(progress["progress_m"], 2 * route.length_m)

    def test_lap_is_counted_once_at_full_route_distance_with_interpolated_time(self) -> None:
        route = RouteProgressTracker(
            np.asarray([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
        )
        start = route.reset(0.0, 0.0, now=100.0)
        tracker = LapTracker(route)
        tracker.reset(start, now=100.0)

        tracker.update(route.length_m - 0.2, now=139.8)
        self.assertEqual(tracker.lap_count, 0)
        sample = tracker.update(route.length_m + 0.2, now=140.2)
        self.assertEqual(sample["lap_count"], 1)
        self.assertAlmostEqual(sample["last_lap_time_s"], 40.0, places=6)
        self.assertAlmostEqual(sample["best_lap_time_s"], 40.0, places=6)
        self.assertAlmostEqual(
            sample["completed_lap_frontier_speed_mps"], 1.0, places=6
        )
        self.assertAlmostEqual(sample["completed_attempt_elapsed_s"], 40.0, places=6)
        self.assertAlmostEqual(
            sample["completed_attempt_average_frontier_speed_mps"], 1.0, places=6
        )
        self.assertEqual(len(sample["lap_gate"]), 2)

        sample = tracker.update(2 * route.length_m + 0.2, now=180.2)
        self.assertEqual(sample["lap_count"], 2)
        self.assertAlmostEqual(sample["best_lap_time_s"], 40.0, places=6)
        self.assertAlmostEqual(sample["completed_attempt_elapsed_s"], 80.0, places=6)
        self.assertAlmostEqual(
            sample["completed_attempt_average_frontier_speed_mps"], 1.0, places=6
        )

    def test_open_centerline_does_not_publish_a_finish_gate_or_lap(self) -> None:
        route = RouteProgressTracker(np.asarray([[0, 0], [5, 0], [10, 0]], dtype=float))
        tracker = LapTracker(route)
        sample = tracker.update(5.0, now=10.0)
        self.assertFalse(sample["lap_supported"])
        self.assertEqual(sample["lap_count"], 0)
        self.assertIsNone(sample["lap_gate"])
        self.assertIsNone(sample["lap_elapsed_s"])

    def test_moved_gate_starts_timing_on_first_pass_then_counts_a_lap(self) -> None:
        route = RouteProgressTracker(
            np.asarray([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
        )
        start = route.reset(0.0, 0.0, now=100.0)
        tracker = LapTracker(route, gate_progress_m=15.0)
        tracker.reset(start, now=100.0)
        self.assertIsNone(tracker.sample(100.0)["lap_elapsed_s"])

        sample = tracker.update(15.0, now=110.0)
        self.assertEqual(sample["lap_count"], 0)
        self.assertEqual(sample["lap_elapsed_s"], 0.0)
        sample = tracker.update(55.0, now=150.0)
        self.assertEqual(sample["lap_count"], 1)
        self.assertAlmostEqual(sample["last_lap_time_s"], 40.0, places=6)

    def test_each_map_centerline_completes_one_synthetic_circuit(self) -> None:
        for map_id in ("porto", "berlin", "icra2026_classic", "icra2026_master"):
            with self.subTest(map_id=map_id):
                path = map_centerline_path(map_id, ROOT)
                self.assertIsNotNone(path)
                route = RouteProgressTracker.from_csv(path)
                self.assertTrue(route.closed)
                meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
                spawn = meta["spawn"]
                start = route.reset(spawn["x"], spawn["z"], now=1000.0)
                gate_config = map_lap_gate_config(map_id, route=route, repository_root=ROOT)
                self.assertTrue(gate_config["supported"])
                self.assertAlmostEqual(
                    gate_config["progress_m"], gate_config["default_progress_m"], places=5
                )
                tracker = LapTracker(route, gate_progress_m=gate_config["progress_m"])
                tracker.reset(start, now=1000.0)
                self.assertIsNotNone(tracker.lap_gate)

                start_s = start["progress_m"]
                for distance in np.arange(0.2, route.length_m + 0.1, 0.2):
                    local_s = (start_s + float(distance)) % route.length_m
                    segment = min(
                        max(int(np.searchsorted(route._cum, local_s, side="right") - 1), 0),
                        len(route._lengths) - 1,
                    )
                    segment_length = route._lengths[segment]
                    alpha = (local_s - route._cum[segment]) / max(segment_length, 1e-9)
                    point = route.points[segment] + alpha * route._seg[segment]
                    projected = route.update(float(point[0]), float(point[1]), now=1000.0 + distance)
                    tracker.update(projected["progress_m"], now=1000.0 + distance)

                self.assertEqual(tracker.lap_count, 1)
                self.assertGreater(tracker.last_lap_time_s, 0.0)
                self.assertGreater(tracker.best_lap_time_s, 0.0)

    def test_map_gate_can_be_saved_and_reset_without_changing_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            maps_root = Path(temp) / "simulator" / "maps"
            occ = maps_root / "oval" / "occupancy"
            occ.mkdir(parents=True)
            Image.new("L", (64, 64), color=255).save(occ / "map.png")
            (occ / "map.yaml").write_text(
                "image: map.png\nresolution: 0.1\norigin: [-2, -2, 0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
                encoding="utf-8",
            )
            (occ / "centerline.csv").write_text(
                "# x_m,y_m,w_right,w_left\n"
                "0,0,0.5,0.5\n1,0,0.5,0.5\n1,1,0.5,0.5\n"
                "0,1,0.5,0.5\n0,0,0.5,0.5\n",
                encoding="utf-8",
            )
            metadata = {
                "centerline": {"status": "ready", "closed": True, "built_at": "test-route"},
                "spawn": {"x": 0.5, "z": 0.0, "yaw": 1.5708},
            }
            (occ / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")

            default = get_map_lap_gate("oval", maps_root)
            self.assertTrue(default["supported"])
            self.assertFalse(default["customized"])
            self.assertAlmostEqual(default["progress_m"], default["default_progress_m"])
            moved = save_map_lap_gate("oval", maps_root, 2.5)
            self.assertTrue(moved["customized"])
            self.assertAlmostEqual(moved["progress_m"], 2.5)
            self.assertEqual(len(moved["preview"]["gate_line"]), 2)

            reset = save_map_lap_gate("oval", maps_root, None)
            self.assertFalse(reset["customized"])
            self.assertAlmostEqual(reset["progress_m"], reset["default_progress_m"])


if __name__ == "__main__":
    unittest.main()
