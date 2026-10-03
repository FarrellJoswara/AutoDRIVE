"""Focused tests for run-level stopping and collision episode endings."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.layer1.racer import Racer
from src.layer2.autodrive_env import AutoDriveEnv
from src.layer2.rewards import RewardConfig
from src.layer3.envs import env_kwargs_from_args
from src.layer3.train import (
    RunStopCallback,
    SimulatorPauseCallback,
    build_arg_parser,
)
from src.layer4.settings import Settings


class RunStopCallbackTests(unittest.TestCase):
    def test_default_train_settings_disable_the_total_step_cap(self) -> None:
        settings = Settings()
        argv = settings.to_train_argv()

        self.assertEqual(settings.timesteps, 0)
        self.assertEqual(settings.max_episode_steps, 0)
        self.assertEqual(argv[argv.index("--timesteps") + 1], "0")
        self.assertIn("--plateau-min-timesteps", argv)
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.timesteps, 0)
        self.assertEqual(args.max_episode_steps, 0)
        self.assertEqual(settings.laps_per_episode, 10)
        self.assertEqual(args.laps_per_episode, 10)
        self.assertTrue(args.terminate_on_collision)
        self.assertEqual(args.forward_scale, 0.0)
        self.assertEqual(args.backward_speed_penalty_scale, 1.0)
        self.assertEqual(args.route_progress_scale, 10.0)
        self.assertEqual(args.collision_penalty, -100.0)

        configured = Settings(map_id="porto")
        parsed = build_arg_parser().parse_args(configured.to_train_argv())
        env_kwargs = env_kwargs_from_args(parsed)
        self.assertEqual(env_kwargs["laps_per_episode"], 10)
        self.assertEqual(env_kwargs["lap_time_reward_scale"], 1000.0)
        self.assertEqual(env_kwargs["forward_scale"], 0.0)
        self.assertEqual(env_kwargs["backward_speed_penalty_scale"], 1.0)
        self.assertEqual(env_kwargs["route_progress_scale"], 10.0)
        self.assertEqual(env_kwargs["collision_penalty"], -100.0)
        self.assertTrue(env_kwargs["terminate_on_collision"])

    @staticmethod
    def _plateau_step(callback, progress_m: float, timesteps: int) -> bool:
        callback.model = SimpleNamespace(num_timesteps=timesteps)
        callback.update_locals({
            "infos": [{"frontier_advanced_m": progress_m}],
            "rewards": [progress_m * 10.0],
            "dones": [False],
        })
        return callback._on_step()

    def test_progress_plateau_stops_after_patience_windows(self) -> None:
        callback = RunStopCallback(
            plateau_min_timesteps=0,
            plateau_window_timesteps=2,
            plateau_patience=2,
            plateau_min_improvement_pct=1.0,
        )

        self.assertTrue(self._plateau_step(callback, 1.0, 1))
        self.assertTrue(self._plateau_step(callback, 1.0, 2))  # baseline window
        self.assertTrue(self._plateau_step(callback, 0.99, 3))
        self.assertTrue(self._plateau_step(callback, 0.99, 4))  # first stale window
        self.assertTrue(self._plateau_step(callback, 0.99, 5))
        self.assertFalse(self._plateau_step(callback, 0.99, 6))
        self.assertEqual(callback.stop_reason, "progress_plateau")
        self.assertEqual(callback.plateau_summary["metric"], "frontier_m_per_env_step")

    def test_material_improvement_resets_plateau_patience(self) -> None:
        callback = RunStopCallback(
            plateau_min_timesteps=0,
            plateau_window_timesteps=2,
            plateau_patience=2,
            plateau_min_improvement_pct=1.0,
        )

        self.assertTrue(self._plateau_step(callback, 1.0, 1))
        self.assertTrue(self._plateau_step(callback, 1.0, 2))
        self.assertTrue(self._plateau_step(callback, 0.99, 3))
        self.assertTrue(self._plateau_step(callback, 0.99, 4))
        self.assertTrue(self._plateau_step(callback, 1.02, 5))  # material gain
        self.assertTrue(self._plateau_step(callback, 1.02, 6))
        self.assertTrue(self._plateau_step(callback, 1.02, 7))
        self.assertTrue(self._plateau_step(callback, 1.02, 8))
        self.assertEqual(callback.stop_reason, None)

    def test_plateau_warmup_blocks_early_stop(self) -> None:
        callback = RunStopCallback(
            plateau_min_timesteps=10,
            plateau_window_timesteps=2,
            plateau_patience=1,
            plateau_min_improvement_pct=1.0,
        )

        self.assertTrue(self._plateau_step(callback, 1.0, 1))
        self.assertTrue(self._plateau_step(callback, 1.0, 2))
        self.assertTrue(self._plateau_step(callback, 0.99, 3))
        self.assertTrue(self._plateau_step(callback, 0.99, 4))
        self.assertEqual(callback.stop_reason, None)

    def test_builtin_map_uses_reward_rate_when_frontier_data_is_absent(self) -> None:
        callback = RunStopCallback(
            plateau_min_timesteps=0,
            plateau_window_timesteps=2,
            plateau_patience=1,
            plateau_min_improvement_pct=1.0,
        )
        callback.model = SimpleNamespace(num_timesteps=1)
        callback.update_locals({"infos": [{}, {}], "rewards": [1.0, 1.0]})
        self.assertTrue(callback._on_step())
        self.assertEqual(callback.plateau_summary["metric"], "reward_per_env_step")

        callback.model.num_timesteps = 3
        callback.update_locals({"infos": [{}, {}], "rewards": [0.99, 0.99]})
        self.assertFalse(callback._on_step())
        self.assertEqual(callback.stop_reason, "progress_plateau")

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


class SimulatorPauseCallbackTests(unittest.TestCase):
    def test_simulator_is_held_only_during_policy_updates(self) -> None:
        class FakeVecEnv:
            def __init__(self):
                self.pauses = []

            def env_method(self, method, paused):
                self.pauses.append((method, paused))

        callback = SimulatorPauseCallback()
        fake_env = FakeVecEnv()
        callback.model = SimpleNamespace(get_env=lambda: fake_env)

        callback._on_rollout_start()
        callback._on_rollout_end()
        callback._on_rollout_start()

        self.assertEqual(
            fake_env.pauses,
            [
                ("set_simulation_paused", True),
                ("set_simulation_paused", False),
            ],
        )

    def test_simulator_clock_excludes_policy_update_pause(self) -> None:
        racer = Racer.__new__(Racer)
        racer._simulation_pause_started_at = None
        racer._simulation_paused_total_s = 0.0

        with patch("src.layer1.racer.time.monotonic", return_value=10.0):
            before_pause = racer.simulation_time()
        with patch("src.layer1.racer.time.monotonic", return_value=12.0):
            racer._mark_simulation_paused()
        with patch("src.layer1.racer.time.monotonic", return_value=112.0):
            during_pause = racer.simulation_time()
            racer._mark_simulation_resumed()
        with patch("src.layer1.racer.time.monotonic", return_value=115.0):
            after_resume = racer.simulation_time()

        self.assertAlmostEqual(during_pause, 12.0)
        self.assertAlmostEqual(after_resume - before_pause, 5.0)

    def test_layer_two_uses_pause_aware_simulator_clock(self) -> None:
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.racer = SimpleNamespace(simulation_time=lambda: 42.5)
        snap = SimpleNamespace(timestamp=9999.0)

        self.assertEqual(env._progress_time(snap), 42.5)


class CollisionTerminationTests(unittest.TestCase):
    @staticmethod
    def _step_with_collision(
        snap,
        terminate_on_collision: bool,
        previous_collision_flag: bool = False,
        pending_collision_event: bool = False,
        progress=None,
        frontier_stagnation_seconds: float = 0.0,
        lap_tracker=None,
        laps_per_episode: int = 0,
    ):
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.frame_skip = 1
        env.racer = SimpleNamespace(step=lambda *_: snap)
        env._last_snap = None
        env._episode_steps = 0
        env._idle_steps = 0
        env._prev_steering = 0.0
        env._prev_throttle = 0.0
        env._prev_collision_count = 1
        env._prev_collision_flag = previous_collision_flag
        env._pending_collision_event = pending_collision_event
        env.stagnation_speed_threshold = 0.1
        env.stagnation_steps = 100
        env.max_episode_steps = 0
        env.route_progress = (
            SimpleNamespace(update=lambda *_: progress) if progress is not None else None
        )
        if progress is not None:
            progress.setdefault("current_delta_m", progress.get("advanced_m", 0.0))
            progress.setdefault("current_progress_m", progress.get("progress_m", 0.0))
            progress.setdefault("current_line", progress.get("line"))
            progress.setdefault("current_projection_valid", True)
        env.frontier_stagnation_seconds = frontier_stagnation_seconds
        env.terminate_on_collision = terminate_on_collision
        env.lap_tracker = lap_tracker or SimpleNamespace(
            update=lambda *_: None,
            sample=lambda *_: {"lap_supported": False},
        )
        env.laps_per_episode = laps_per_episode
        env.reward_config = RewardConfig()
        env.lidar_beams = 1080

        with patch("src.layer2.autodrive_env.snapshot_to_obs", return_value={"state": np.zeros(1)}):
            return env.step(np.array([0.0, 0.0], dtype=np.float32))

    def test_collision_flag_fallback_uses_only_its_rising_edge(self) -> None:
        snap = SimpleNamespace(
            collision_count=1,
            timestamp=1.0,
            position=(0.0, 0.0, 0.0),
            v_long=1.0,
            true_speed=1.0,
            collision=True,
            heading_yaw=0.0,
            slip_angle=0.0,
            lidar=np.ones(1080),
        )

        _, _, terminated, _, info = self._step_with_collision(snap, True)
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "collision")

        # A sticky Bridge flag already present at reset must not kill the next
        # episode again when the counter is unchanged.
        _, _, terminated, _, info = self._step_with_collision(
            snap, True, previous_collision_flag=True
        )
        self.assertFalse(terminated)
        self.assertNotIn("termination_reason", info)

    def test_reset_preserves_collision_seen_during_settling_ticks(self) -> None:
        before_reset_settles = SimpleNamespace(
            collision_count=0,
            collision=False,
            timestamp=1.0,
            position=(0.0, 0.0, 0.0),
        )
        collision_during_settle = SimpleNamespace(
            collision_count=1,
            collision=True,
            timestamp=1.1,
            position=(0.0, 0.0, 0.0),
        )
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.frame_skip = 1
        env.racer = SimpleNamespace(
            reset=lambda: before_reset_settles,
            step=lambda *_: collision_during_settle,
        )
        env.route_progress = None
        env.lap_tracker = SimpleNamespace(reset=lambda *_: None)
        env.lidar_beams = 1080
        env._build_info = lambda *_args, **_kwargs: {}

        with patch(
            "src.layer2.autodrive_env.snapshot_to_obs",
            return_value={"state": np.zeros(1)},
        ):
            env.reset()

        self.assertTrue(env._pending_collision_event)
        self.assertEqual(env._prev_collision_count, 1)

    def test_reset_rejects_pose_outside_configured_spawn(self) -> None:
        stale_pose = SimpleNamespace(
            collision_count=0,
            collision=False,
            timestamp=1.0,
            position=(0.0, 0.0, 0.0),
        )
        tracker = SimpleNamespace(reset=lambda *_: self.fail("frontier was re-anchored"))
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.frame_skip = 1
        env.racer = SimpleNamespace(
            reset=lambda: stale_pose,
            step=lambda *_: stale_pose,
        )
        env.map_id = "porto"
        env._expected_spawn_xz = (4.175, 0.929)
        env.route_progress = tracker

        with self.assertRaisesRegex(RuntimeError, "Refusing to reset the frontier"):
            env.reset()

    def test_reset_anchors_frontier_to_configured_spawn_after_pose_validation(self) -> None:
        settled_pose = SimpleNamespace(
            collision_count=0,
            collision=False,
            timestamp=1.0,
            position=(4.097, 0.15, 1.069),
        )
        reset_calls = []
        progress_state = {"progress_m": 25.6105}
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.frame_skip = 1
        env.racer = SimpleNamespace(
            reset=lambda: settled_pose,
            step=lambda *_: settled_pose,
        )
        env.map_id = "porto"
        env._expected_spawn_xz = (4.175001, 0.928669)
        env.route_progress = SimpleNamespace(
            reset=lambda *args: reset_calls.append(args) or progress_state
        )
        env.lap_tracker = SimpleNamespace(reset=lambda *_: None)
        env.lidar_beams = 1080
        env._progress_time = lambda _snap: 1.0
        env._build_info = lambda *_args, **_kwargs: {}

        with patch(
            "src.layer2.autodrive_env.snapshot_to_obs",
            return_value={"state": np.zeros(1)},
        ):
            env.reset()

        self.assertEqual(reset_calls[0][:2], env._expected_spawn_xz)

    def test_collision_during_reset_settling_ends_the_new_episode(self) -> None:
        snap = SimpleNamespace(
            collision_count=1,
            timestamp=1.0,
            position=(0.0, 0.0, 0.0),
            v_long=1.0,
            true_speed=1.0,
            collision=False,
            heading_yaw=0.0,
            slip_angle=0.0,
            lidar=np.ones(1080),
        )

        _, _, terminated, _, info = self._step_with_collision(
            snap, True, pending_collision_event=True
        )
        self.assertTrue(terminated)
        self.assertEqual(info["termination_reason"], "collision")

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

    def test_frontier_stagnation_terminates_car_at_configured_threshold(self) -> None:
        snap = SimpleNamespace(
            collision_count=0,
            timestamp=6.0,
            position=(0.0, 0.0, 0.0),
            v_long=0.0,
            true_speed=0.0,
            collision=False,
            heading_yaw=0.0,
            slip_angle=0.0,
            lidar=np.ones(1080),
        )
        progress = {
            "line": ((0.0, 0.0), (0.0, 1.0)),
            "progress_m": 1.0,
            "advanced_m": 0.0,
            "time_since_push_s": 5.0,
            "speed_mps": 0.0,
        }

        _, _, terminated, truncated, info = self._step_with_collision(
            snap,
            terminate_on_collision=True,
            progress=progress,
            frontier_stagnation_seconds=5.0,
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "frontier_stagnation")

    def test_frontier_stagnation_does_not_terminate_before_threshold(self) -> None:
        snap = SimpleNamespace(
            collision_count=0,
            timestamp=5.9,
            position=(0.0, 0.0, 0.0),
            v_long=0.0,
            true_speed=0.0,
            collision=False,
            heading_yaw=0.0,
            slip_angle=0.0,
            lidar=np.ones(1080),
        )
        progress = {
            "line": ((0.0, 0.0), (0.0, 1.0)),
            "progress_m": 1.0,
            "advanced_m": 0.0,
            "time_since_push_s": 4.9,
            "speed_mps": 0.0,
        }

        _, _, terminated, truncated, info = self._step_with_collision(
            snap,
            terminate_on_collision=True,
            progress=progress,
            frontier_stagnation_seconds=5.0,
        )

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn("termination_reason", info)

    def test_tenth_lap_awards_one_bonus_for_average_speed_across_whole_run(self) -> None:
        class CompletingLapTracker:
            supported = True
            length_m = 120.0
            lap_count = 9

            def update(self, *_args):
                self.lap_count = 10
                return {
                    "completed_lap_frontier_speed_mps": 3.5,
                    "completed_attempt_elapsed_s": 400.0,
                    "completed_attempt_average_frontier_speed_mps": 3.0,
                }

            def sample(self, *_args):
                return {"lap_supported": True, "lap_count": self.lap_count}

        snap = SimpleNamespace(
            collision_count=0,
            timestamp=10.0,
            position=(1.0, 0.0, 1.0),
            v_long=2.0,
            true_speed=2.0,
            collision=False,
            heading_yaw=0.0,
            slip_angle=0.0,
            lidar=np.ones(1080),
        )
        progress = {
            "progress_m": 120.0,
            "advanced_m": 0.0,
            "time_since_push_s": 0.0,
            "line": [[0.0, 0.0], [1.0, 0.0]],
            "speed_mps": 0.0,
        }

        _, reward, terminated, truncated, info = self._step_with_collision(
            snap,
            False,
            progress=progress,
            lap_tracker=CompletingLapTracker(),
            laps_per_episode=10,
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "lap_target")
        self.assertTrue(info["episode_won"])
        self.assertEqual(info["lap_count"], 10)
        self.assertEqual(info["lap_reward"], 3000.0)
        self.assertGreaterEqual(reward, info["lap_reward"])

    def test_collision_on_target_lap_invalidates_clean_win_and_bonus(self) -> None:
        class CollidingLapTracker:
            supported = True
            length_m = 120.0
            lap_count = 9

            def update(self, *_args):
                self.lap_count = 10
                return {"completed_attempt_average_frontier_speed_mps": 3.0}

            def sample(self, *_args):
                return {"lap_supported": True, "lap_count": self.lap_count}

        snap = SimpleNamespace(
            collision_count=2, timestamp=10.0, position=(1.0, 0.0, 1.0),
            v_long=2.0, true_speed=2.0, collision=True, heading_yaw=0.0,
            slip_angle=0.0, lidar=np.ones(1080),
        )
        progress = {
            "progress_m": 120.0, "advanced_m": 0.0, "time_since_push_s": 0.0,
            "line": [[0.0, 0.0], [1.0, 0.0]], "speed_mps": 0.0,
        }
        _, reward, terminated, truncated, info = self._step_with_collision(
            snap, True, progress=progress, lap_tracker=CollidingLapTracker(),
            laps_per_episode=10,
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "collision")
        self.assertFalse(info["episode_won"])
        self.assertEqual(info["lap_reward"], 0.0)
        self.assertLess(reward, 0.0)


class LapPaceRewardTests(unittest.TestCase):
    def test_faster_clean_target_finish_receives_larger_reward(self) -> None:
        from src.layer2.rewards import compute_reward

        cfg = RewardConfig(lap_time_reward_scale=1000.0)
        common = dict(
            v_long=0.0,
            route_progress_delta_m=0.0,
            collision_event=False,
            slip_angle=0.0,
            prev_steering=0.0,
            steering=0.0,
            cfg=cfg,
        )
        slow = compute_reward(**common, clean_run_average_frontier_speed_mps=2.0)
        fast = compute_reward(**common, clean_run_average_frontier_speed_mps=4.0)

        self.assertEqual(slow, 2000.0)
        self.assertEqual(fast, 4000.0)

    def test_no_clean_target_finish_means_no_pace_reward(self) -> None:
        from src.layer2.rewards import compute_reward

        reward = compute_reward(
            v_long=20.0, route_progress_delta_m=0.0, collision_event=False,
            slip_angle=0.0, prev_steering=0.0, steering=0.0, cfg=RewardConfig(),
        )
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
