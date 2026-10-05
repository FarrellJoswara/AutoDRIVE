"""Tests for Layer 3 fleet telemetry semantics."""

from __future__ import annotations

import unittest
from collections import defaultdict
from io import StringIO
from types import SimpleNamespace

import numpy as np
from stable_baselines3.common.logger import HumanOutputFormat, Logger

from src.layer3.hub_callback import HubTelemetryCallback
from src.layer3.hub_callback import _min_pool_lidar


class LidarDownsampleTests(unittest.TestCase):
    def test_min_pool_preserves_the_final_endpoint_ray(self) -> None:
        scan = np.full(1081, 10.0, dtype=np.float32)
        scan[-1] = 0.06

        pooled = _min_pool_lidar(scan, 360)

        self.assertEqual(len(pooled), 360)
        self.assertAlmostEqual(pooled[-1], 0.06, places=3)


class PpoDiagnosticsLoggerTests(unittest.TestCase):
    def test_missing_metrics_do_not_corrupt_sb3_logger_bookkeeping(self) -> None:
        logger = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(StringIO())],
        )
        logger.record("rollout/ep_rew_mean", 1.0)
        callback = HubTelemetryCallback(
            hub_url="http://127.0.0.1:8090",
            run_id="test",
            lidar_max_envs=0,
        )
        callback.model = SimpleNamespace(logger=logger)

        self.assertEqual(callback._read_ppo_diagnostics(), {})
        self.assertEqual(set(logger.name_to_value), {"rollout/ep_rew_mean"})
        self.assertEqual(set(logger.name_to_value), set(logger.name_to_excluded))

        # This was the first PPO logging point that crashed in the failed run.
        logger.dump(step=1)

    def test_available_metrics_are_captured_without_mutating_logger(self) -> None:
        values = defaultdict(float, {"train/approx_kl": 0.125})
        logger = SimpleNamespace(name_to_value=values)
        callback = HubTelemetryCallback(
            hub_url="http://127.0.0.1:8090",
            run_id="test",
            lidar_max_envs=0,
        )
        callback.model = SimpleNamespace(logger=logger)

        self.assertEqual(callback._read_ppo_diagnostics(), {"approx_kl": 0.125})
        self.assertEqual(set(values), {"train/approx_kl"})


class CollisionMarkerTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.callback = HubTelemetryCallback(
            hub_url="http://127.0.0.1:8090",
            run_id="test",
            lidar_max_envs=0,
        )
        self.callback._last_pose = {}
        self.callback._last_yaw = {}

    def _sample(self, *, collision_count: int, collision_event: bool, done: bool = False):
        info = {
            "position": (4.1, 0.15, 1.0),
            "yaw": 1.2,
            "true_speed": 0.5,
            "v_long": -0.4,
            "throttle_command": -0.7,
            "steering_command": 0.25,
            "collision_count": collision_count,
            "collision_event": collision_event,
            "lap_supported": True,
            "lap_count": 0,
            "frontier_line": [[3.0, 0.0], [3.0, 1.0]],
            "frontier_progress_m": 3.0,
            "current_progress_line": [[2.8, 0.0], [2.8, 1.0]],
            "current_progress_m": 2.8,
            "signed_route_delta_m": -0.02,
            "current_route_speed_mps": -0.8,
            "route_projection_valid": True,
            "reward_components": {
                "route_progress": -0.2,
                "backward_motion": -0.01,
                "collision": 0.0,
                "total": -0.21,
            },
        }
        dones = [done]
        totals = self.callback._refresh_collision_counts([info], dones)
        self.callback.update_locals({"new_obs": {}})
        sample = self.callback._build_fleet_sample([info], dones, totals)
        return sample["cars"][0]

    def test_control_commands_and_signed_speed_are_transmitted(self) -> None:
        car = self._sample(collision_count=0, collision_event=False)

        self.assertEqual(car["v_long"], -0.4)
        self.assertEqual(car["throttle_command"], -0.7)
        self.assertEqual(car["steering_command"], 0.25)

    def test_current_route_position_and_reward_terms_are_transmitted(self) -> None:
        car = self._sample(collision_count=0, collision_event=False)

        self.assertEqual(car["frontier_progress_m"], 3.0)
        self.assertEqual(car["current_progress_m"], 2.8)
        self.assertEqual(car["signed_route_delta_m"], -0.02)
        self.assertEqual(car["current_route_speed_mps"], -0.8)
        self.assertTrue(car["route_projection_valid"])
        self.assertEqual(car["reward_components"]["route_progress"], -0.2)
        self.assertEqual(car["reward_components"]["total"], -0.21)

    def test_past_episode_contact_does_not_leave_x_on_car(self) -> None:
        self._sample(collision_count=0, collision_event=False)

        impact = self._sample(collision_count=1, collision_event=True)
        self.assertTrue(impact["collision"])
        self.assertEqual(impact["collision_count"], 1)

        after_impact = self._sample(collision_count=1, collision_event=False)
        self.assertFalse(after_impact["collision"])
        self.assertEqual(after_impact["collision_count"], 1)

    def test_contact_count_clears_for_new_episode(self) -> None:
        self._sample(collision_count=0, collision_event=False)
        self._sample(collision_count=1, collision_event=True)
        reset_frame = self._sample(
            collision_count=1, collision_event=True, done=True
        )
        self.assertTrue(reset_frame["reset"])
        self.assertFalse(reset_frame["collision"])

        spawned = self._sample(collision_count=1, collision_event=False)
        self.assertFalse(spawned["reset"])
        self.assertFalse(spawned["collision"])
        self.assertEqual(spawned["collision_count"], 0)


if __name__ == "__main__":
    unittest.main()
