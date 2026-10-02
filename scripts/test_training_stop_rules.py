"""Focused tests for run-level stopping and collision episode endings."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.layer2.autodrive_env import AutoDriveEnv
from src.layer2.rewards import RewardConfig
from src.layer3.train import RunStopCallback


class RunStopCallbackTests(unittest.TestCase):
    def test_lap_target_stops_when_any_env_reaches_target(self) -> None:
        callback = RunStopCallback(stop_after_laps=3)
        callback.update_locals({"infos": [{"lap_count": 2}, {"lap_count": 3}]})

        self.assertFalse(callback._on_step())
        self.assertEqual(callback.stop_reason, "lap_target")

    def test_lap_target_accumulates_across_car_episodes(self) -> None:
        callback = RunStopCallback(stop_after_laps=3)
        callback.update_locals(
            {"infos": [{"lap_count": 2}, {"lap_count": 0}], "dones": [True, False]}
        )
        self.assertTrue(callback._on_step())

        callback.update_locals(
            {"infos": [{"lap_count": 1}, {"lap_count": 0}], "dones": [False, False]}
        )
        self.assertFalse(callback._on_step())
        self.assertEqual(callback.stop_reason, "lap_target")

    def test_duration_limit_stops_training(self) -> None:
        callback = RunStopCallback(max_duration_seconds=1)
        callback._started_at = time.monotonic() - 2
        callback.update_locals({"infos": []})

        self.assertFalse(callback._on_step())
        self.assertEqual(callback.stop_reason, "max_duration")

    def test_zero_optional_limits_leave_run_active(self) -> None:
        callback = RunStopCallback()
        callback.update_locals({"infos": [{"lap_count": 50}]})

        self.assertTrue(callback._on_step())
        self.assertIsNone(callback.stop_reason)


class CollisionTerminationTests(unittest.TestCase):
    @staticmethod
    def _step_with_collision(snap, terminate_on_collision: bool):
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.frame_skip = 1
        env.racer = SimpleNamespace(step=lambda *_: snap)
        env._last_snap = None
        env._episode_steps = 0
        env._idle_steps = 0
        env._prev_steering = 0.0
        env._prev_throttle = 0.0
        env._prev_collision_count = 1
        env.stagnation_speed_threshold = 0.1
        env.stagnation_steps = 100
        env.max_episode_steps = 0
        env.route_progress = None
        env.frontier_stagnation_seconds = 0.0
        env.terminate_on_collision = terminate_on_collision
        env.lap_tracker = SimpleNamespace(
            update=lambda *_: None,
            sample=lambda *_: {"lap_supported": False},
        )
        env.reward_config = RewardConfig()
        env.lidar_beams = 1080

        with patch("src.layer2.autodrive_env.snapshot_to_obs", return_value={"state": np.zeros(1)}):
            return env.step(np.array([0.0, 0.0], dtype=np.float32))

    def test_collision_toggle_ends_only_the_enabled_episode(self) -> None:
        snap = SimpleNamespace(
            collision_count=2,
            timestamp=1.0,
            position=(0.0, 0.0, 0.0),
            v_long=1.0,
            true_speed=1.0,
            collision=True,
            heading_yaw=0.0,
            slip_angle=0.0,
            lidar=np.ones(1080),
        )

        _, _, terminated, truncated, info = self._step_with_collision(snap, True)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "collision")

        _, _, terminated, truncated, info = self._step_with_collision(snap, False)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn("termination_reason", info)


if __name__ == "__main__":
    unittest.main()
