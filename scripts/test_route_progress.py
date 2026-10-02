"""Route-frontier geometry and per-map centerline smoke tests."""

from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from src.layer2.route_progress import RouteProgressTracker, map_centerline_path
from src.layer2.rewards import RewardConfig, compute_reward


ROOT = Path(__file__).resolve().parents[1]


class RouteProgressTests(unittest.TestCase):
    def test_frontier_never_moves_backwards_and_reports_push_timing(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101)))
        tracker = RouteProgressTracker(
            points, np.full(101, 1.0), np.full(101, 1.0), max_forward_m=2.0
        )
        tracker.reset(1.0, 0.0, now=10.0)
        pushed = tracker.update(2.0, 0.0, now=11.0)
        self.assertAlmostEqual(pushed["progress_m"], 2.0, places=3)
        self.assertAlmostEqual(pushed["time_since_push_s"], 0.0, places=6)
        self.assertAlmostEqual(pushed["speed_mps"], 1.0, places=2)
        backed_up = tracker.update(0.5, 0.0, now=12.0)
        self.assertAlmostEqual(backed_up["progress_m"], 2.0, places=3)
        self.assertAlmostEqual(backed_up["time_since_push_s"], 1.0, places=6)
        self.assertEqual(backed_up["speed_mps"], 0.0)

    def test_projection_window_rejects_large_shortcut_jump(self) -> None:
        points = np.column_stack((np.linspace(0.0, 20.0, 201), np.zeros(201)))
        tracker = RouteProgressTracker(points, max_forward_m=1.5)
        tracker.reset(1.0, 0.0, now=1.0)
        jumped = tracker.update(15.0, 0.0, now=2.0)
        self.assertAlmostEqual(jumped["progress_m"], 1.0, places=3)
        self.assertEqual(jumped["advanced_m"], 0.0)

    def test_projection_rejects_pose_outside_track_width(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 11), np.zeros(11)))
        tracker = RouteProgressTracker(
            points, np.full(11, 0.5), np.full(11, 0.5), max_forward_m=2.0
        )
        tracker.reset(1.0, 0.0, now=1.0)
        outside = tracker.update(2.0, 1.0, now=2.0)
        self.assertEqual(outside["progress_m"], 1.0)
        self.assertEqual(outside["advanced_m"], 0.0)

    def test_frontier_line_spans_both_track_sides(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 11), np.zeros(11)))
        tracker = RouteProgressTracker(
            points, np.full(11, 1.25), np.full(11, 1.75)
        )
        result = tracker.reset(5.0, 0.0, now=1.0)
        self.assertEqual(len(result["line"]), 2)
        a, b = np.asarray(result["line"])
        self.assertAlmostEqual(float(np.linalg.norm(b - a)), 3.0, places=5)
        self.assertAlmostEqual(float(a[0]), float(b[0]), places=5)

    def test_route_reward_uses_frontier_delta_instead_of_raw_speed(self) -> None:
        cfg = RewardConfig(forward_scale=1.0, route_progress_scale=10.0)
        no_progress = compute_reward(
            v_long=8.0,
            route_progress_delta_m=0.0,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        actual_progress = compute_reward(
            v_long=0.0,
            route_progress_delta_m=0.2,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        self.assertEqual(no_progress, 0.0)
        self.assertAlmostEqual(actual_progress, 2.0)

    def test_closed_route_wrap_increases_global_frontier(self) -> None:
        points = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
        tracker = RouteProgressTracker(points, max_forward_m=2.0)
        tracker.reset(0.0, 0.0, now=1.0)
        route = np.vstack((points,))
        for a, b in zip(route[:-1], route[1:]):
            distance = float(np.linalg.norm(b - a))
            count = int(distance)
            for k in range(1, count + 1):
                p = a + (b - a) * min(1.0, k / distance)
                wrapped = tracker.update(float(p[0]), float(p[1]), now=2.0 + k)
        self.assertGreater(wrapped["progress_m"], 39.0)

    def test_checked_in_maps_have_usable_route_geometry(self) -> None:
        for map_id in ("porto", "berlin", "icra2026_classic", "icra2026_master"):
            with self.subTest(map_id=map_id):
                path = map_centerline_path(map_id, ROOT)
                self.assertIsNotNone(path)
                tracker = RouteProgressTracker.from_csv(path)
                self.assertGreater(tracker.length_m, 10.0)
                meta_path = path.parent / "meta.json"
                spawn = json.loads(meta_path.read_text(encoding="utf-8"))["spawn"]
                tracker.reset(spawn["x"], spawn["z"], now=1.0)
                self.assertIsNotNone(tracker.sample(now=1.0)["line"])
                # The generated spawn heading is expected to point along the
                # ordered route; a short forward movement should push it.
                x = spawn["x"] + 0.3 * np.sin(spawn["yaw"])
                z = spawn["z"] + 0.3 * np.cos(spawn["yaw"])
                sample = tracker.update(float(x), float(z), now=1.1)
                self.assertGreater(sample["advanced_m"], 0.1)


if __name__ == "__main__":
    unittest.main()
