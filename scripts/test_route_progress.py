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

    def test_small_reverse_motion_within_projection_tolerance_keeps_frontier(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101)))
        tracker = RouteProgressTracker(points, backward_tolerance_m=0.5)
        tracker.reset(2.0, 0.0, now=1.0)
        tracker.update(3.0, 0.0, now=2.0)

        backed_up = tracker.update(2.8, 0.0, now=3.0)

        self.assertAlmostEqual(backed_up["progress_m"], 3.0, places=3)
        self.assertEqual(backed_up["advanced_m"], 0.0)

    def test_episode_reset_restarts_only_that_cars_frontier(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101)))
        car_a = RouteProgressTracker(points)
        car_b = RouteProgressTracker(points)
        car_a.reset(1.0, 0.0, now=1.0)
        car_b.reset(1.0, 0.0, now=1.0)
        car_a.update(3.0, 0.0, now=2.0)
        car_b.update(2.0, 0.0, now=2.0)

        restarted = car_a.reset(0.5, 0.0, now=3.0)

        self.assertAlmostEqual(restarted["progress_m"], 0.5, places=3)
        self.assertEqual(restarted["speed_mps"], 0.0)
        self.assertEqual(restarted["time_since_push_s"], 0.0)
        self.assertAlmostEqual(car_b.sample(now=3.0)["progress_m"], 2.0, places=3)

    def test_current_route_delta_is_signed_while_frontier_stays_monotonic(self) -> None:
        points = np.column_stack((np.linspace(0.0, 10.0, 101), np.zeros(101)))
        tracker = RouteProgressTracker(points)
        tracker.reset(2.0, 0.0, now=1.0)
        pushed = tracker.update(3.0, 0.0, now=2.0)
        backed_up = tracker.update(2.8, 0.0, now=3.0)
        recovered = tracker.update(2.9, 0.0, now=4.0)

        self.assertAlmostEqual(pushed["current_delta_m"], 1.0, places=3)
        self.assertAlmostEqual(backed_up["current_delta_m"], -0.2, places=3)
        self.assertAlmostEqual(recovered["current_delta_m"], 0.1, places=3)
        self.assertAlmostEqual(backed_up["progress_m"], 3.0, places=3)
        self.assertAlmostEqual(recovered["progress_m"], 3.0, places=3)
        self.assertEqual(recovered["advanced_m"], 0.0)
        self.assertAlmostEqual(recovered["current_progress_m"], 2.9, places=3)

    def test_current_route_delta_unwraps_forward_across_lap_start(self) -> None:
        points = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=float)
        tracker = RouteProgressTracker(points)
        length = tracker.length_m
        tracker.reset(0.0, 0.1, now=1.0)
        tracker.current_s = length - 0.1
        tracker.frontier_s = length - 0.1

        crossed = tracker.update(0.1, 0.0, now=2.0)

        self.assertGreater(crossed["current_delta_m"], 0.0)
        self.assertAlmostEqual(crossed["current_delta_m"], 0.2, places=3)
        self.assertGreater(crossed["progress_m"], length)

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

    def test_only_new_frontier_progress_earns_reward(self) -> None:
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
            frontier_advanced_m=0.2,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        self.assertEqual(no_progress, 0.0)
        self.assertAlmostEqual(actual_progress, 2.0)

        # Current route movement (including recovering previously visited
        # ground) and raw speed are not paid without a new frontier push.
        recovered_ground = compute_reward(
            v_long=8.0,
            route_progress_delta_m=0.2,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        self.assertEqual(recovered_ground, 0.0)

    def test_backward_motion_penalty_scales_with_reverse_distance(self) -> None:
        cfg = RewardConfig(backward_speed_penalty_scale=2.0, time_penalty_per_second=0.0)
        reward = compute_reward(
            v_long=-1.0,
            step_duration_s=0.5,
            route_progress_delta_m=0.0,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        self.assertAlmostEqual(reward, -0.9)

    def test_backward_deadband_and_forward_motion_are_unpenalized(self) -> None:
        cfg = RewardConfig(backward_speed_penalty_scale=2.0, time_penalty_per_second=0.0)
        common = dict(
            step_duration_s=0.5,
            route_progress_delta_m=0.0,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        self.assertEqual(compute_reward(v_long=-0.05, **common), 0.0)
        self.assertEqual(compute_reward(v_long=1.0, **common), 0.0)

    def test_frontier_reward_requires_new_advance_and_not_body_reverse(self) -> None:
        from src.layer2.rewards import compute_reward_components

        cfg = RewardConfig(
            route_progress_scale=10.0,
            backward_speed_penalty_scale=1.0,
            time_penalty_per_second=0.0,
        )
        common = dict(
            v_long=0.0,
            step_duration_s=0.025,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        forward = compute_reward_components(frontier_advanced_m=0.02, **common)
        no_new_record = compute_reward_components(frontier_advanced_m=0.0, **common)

        self.assertAlmostEqual(forward["route_progress"], 0.2)
        self.assertAlmostEqual(forward["total"], 0.2)
        self.assertEqual(no_new_record["route_progress"], 0.0)
        self.assertEqual(no_new_record["total"], 0.0)

    def test_route_progress_reward_is_withheld_while_car_moves_backward(self) -> None:
        from src.layer2.rewards import compute_reward_components

        components = compute_reward_components(
            v_long=-0.3,
            step_duration_s=0.025,
            frontier_advanced_m=0.03,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=RewardConfig(
                route_progress_scale=10.0,
                backward_speed_penalty_scale=1.0,
                backward_speed_deadband_mps=0.1,
                time_penalty_per_second=0.0,
            ),
        )

        self.assertEqual(components["route_progress"], 0.0)
        self.assertAlmostEqual(components["reverse_direction_gate"], -0.3)
        self.assertAlmostEqual(components["backward_motion"], -0.005)
        self.assertAlmostEqual(components["total"], -0.305)

    def test_reverse_direction_gate_respects_body_speed_deadband(self) -> None:
        from src.layer2.rewards import compute_reward_components

        components = compute_reward_components(
            v_long=-0.05,
            frontier_advanced_m=0.02,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=RewardConfig(route_progress_scale=10.0, time_penalty_per_second=0.0),
        )

        self.assertAlmostEqual(components["route_progress"], 0.2)
        self.assertEqual(components["reverse_direction_gate"], 0.0)
        self.assertEqual(components["backward_motion"], 0.0)

    def test_time_and_noncollision_failure_are_costs(self) -> None:
        from src.layer2.rewards import compute_reward_components

        components = compute_reward_components(
            v_long=0.0,
            step_duration_s=2.0,
            frontier_advanced_m=0.0,
            collision_event=False,
            episode_failure=True,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=RewardConfig(time_penalty_per_second=1.5, episode_failure_penalty=-80.0),
        )
        self.assertEqual(components["time_cost"], -3.0)
        self.assertEqual(components["episode_failure"], -80.0)
        self.assertEqual(components["total"], -83.0)

    def test_no_frontier_map_never_rewards_raw_speed(self) -> None:
        reward = compute_reward(
            v_long=12.0,
            step_duration_s=0.1,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=RewardConfig(forward_scale=100.0, time_penalty_per_second=0.0),
        )
        self.assertEqual(reward, 0.0)

    def test_reward_breakdown_collision_default_is_minus_one_hundred(self) -> None:
        from src.layer2.rewards import compute_reward_components

        components = compute_reward_components(
            v_long=0.0,
            route_progress_delta_m=0.0,
            collision_event=True,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=RewardConfig(),
        )
        self.assertEqual(components["collision"], -100.0)
        self.assertEqual(components["total"], -100.0)

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
