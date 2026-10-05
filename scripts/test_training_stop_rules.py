"""Focused tests for run-level stopping and collision episode endings."""

from __future__ import annotations

import time
import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.layer1.racer import Racer
from src.layer2.autodrive_env import AutoDriveEnv
from src.layer2.rewards import RewardConfig
from src.layer3.envs import env_kwargs_from_args
from src.layer3.train import (
    aggregate_evaluation_attempts,
    RunStopCallback,
    SimulatorPauseCallback,
    build_arg_parser,
)
from src.layer4.settings import Settings


class RunStopCallbackTests(unittest.TestCase):
    def test_repeated_evaluation_uses_median_not_luckiest_attempt(self) -> None:
        attempts = [
            {"frontier_speed_mps": 2.0, "total_reward": 20.0, "laps_observed": 0,
             "simulated_seconds": 10.0, "frontier_distance_m": 20.0,
             "reward_per_simulated_second": 2.0, "collisions": 1,
             "failed_episodes": 0, "lap_times_s": [], "best_10_lap_time_s": None,
             "termination_reasons": {"collision": 1}},
            {"frontier_speed_mps": 2.2, "total_reward": 22.0, "laps_observed": 1,
             "simulated_seconds": 10.0, "frontier_distance_m": 22.0,
             "reward_per_simulated_second": 2.2, "collisions": 0,
             "failed_episodes": 0, "lap_times_s": [10.0], "best_10_lap_time_s": None,
             "termination_reasons": {"stall": 1}},
            {"frontier_speed_mps": 20.0, "total_reward": 200.0, "laps_observed": 10,
             "simulated_seconds": 10.0, "frontier_distance_m": 200.0,
             "reward_per_simulated_second": 20.0, "collisions": 0,
             "failed_episodes": 0, "lap_times_s": [10.0] * 10, "best_10_lap_time_s": 100.0,
             "termination_reasons": {"lap_target": 1}},
        ]

        result = aggregate_evaluation_attempts(attempts, 50_000, "total_reward")

        self.assertEqual(result["selection_score"], 22.0)
        self.assertEqual(result["selection_score_mean"], 242 / 3)
        self.assertEqual(result["selection_score_best"], 200.0)
        self.assertEqual(result["successful_attempts"], 1)
        self.assertEqual(result["collision_rate"], 1 / 3)
        self.assertEqual(result["collisions"], 1)

    def test_main_passes_lap_episode_setting_into_training(self) -> None:
        train_module = importlib.import_module("src.layer3.train")
        with patch.object(train_module, "train", return_value="checkpoint.zip") as run:
            result = train_module.main(
                ["--n-envs", "1", "--laps-per-episode", "7", "--device", "cpu"]
            )
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.kwargs["laps_per_episode"], 7)

    def test_training_settings_have_no_lap_targets_or_evaluation_time_cap(self) -> None:
        settings = Settings()
        argv = settings.to_train_argv()
        args = build_arg_parser().parse_args(argv)
        self.assertEqual(settings.timesteps, 0)
        self.assertEqual(settings.frontier_stagnation_seconds, 10.0)
        self.assertEqual(args.frontier_stagnation_seconds, 10.0)
        self.assertEqual(settings.time_penalty_per_second, 5.0)
        self.assertEqual(settings.frontier_pace_target_mps, 6.0)
        self.assertEqual(args.evaluation_every_timesteps, 50_000)
        self.assertEqual(settings.evaluation_runs_per_snapshot, 3)
        self.assertEqual(args.evaluation_runs_per_snapshot, 3)
        self.assertEqual(settings.ppo_learning_rate, 3e-4)
        self.assertEqual(settings.ppo_n_epochs, 8)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.n_epochs, 8)
        self.assertEqual(settings.evaluation_metric, "total_reward")
        self.assertEqual(args.evaluation_metric, "total_reward")
        self.assertEqual(settings.laps_per_episode, 10)
        self.assertEqual(args.laps_per_episode, 10)
        self.assertFalse(hasattr(args, "evaluation_duration_seconds"))
        self.assertEqual(args.collision_penalty_magnitude, 100.0)
        self.assertEqual(args.collision_reward_percent, 100.0)
        self.assertEqual(args.episode_failure_reward_percent, 100.0)
        self.assertEqual(args.episode_failure_penalty_magnitude, 100.0)
        self.assertNotIn('--stop-after-laps', argv)
        self.assertNotIn('--lap-time-reward-scale', argv)
        self.assertIn('--laps-per-episode', argv)

    def test_ppo_optimizer_parameters_flow_from_settings_to_cli(self) -> None:
        settings = Settings(ppo_learning_rate=1e-4, ppo_n_epochs=4)
        args = build_arg_parser().parse_args(settings.to_train_argv())
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.n_epochs, 4)

    def test_settings_and_env_kwargs_are_backend_driven(self) -> None:
        configured = Settings(map_id='porto', n_envs=4,
                              frontier_pace_target_mps=4.5,
                              collision_penalty_magnitude=250,
                              collision_reward_percent=35,
                              episode_failure_reward_percent=40)
        args = build_arg_parser().parse_args(configured.to_train_argv())
        env_kwargs = env_kwargs_from_args(args)
        self.assertEqual(env_kwargs['map_id'], 'porto')
        self.assertEqual(env_kwargs['laps_per_episode'], 10)
        self.assertEqual(env_kwargs['time_penalty_per_second'], 5.0)
        self.assertEqual(env_kwargs['frontier_pace_target_mps'], 4.5)
        self.assertEqual(env_kwargs['collision_penalty_magnitude'], 250.0)
        self.assertEqual(env_kwargs['collision_reward_percent'], 35.0)
        self.assertEqual(env_kwargs['episode_failure_penalty_magnitude'], 100.0)
        self.assertEqual(env_kwargs['episode_failure_reward_percent'], 40.0)

    def test_wall_clock_stop_remains_optional(self) -> None:
        callback = RunStopCallback(max_duration_seconds=1)
        callback._started_at = time.monotonic() - 2
        callback.update_locals({})
        self.assertFalse(callback._on_step())
        self.assertEqual(callback.stop_reason, 'max_duration')

    def test_zero_wall_clock_limit_keeps_training_active(self) -> None:
        callback = RunStopCallback()
        callback.update_locals({})
        self.assertTrue(callback._on_step())
        self.assertIsNone(callback.stop_reason)


class SimulatorPauseCallbackTests(unittest.TestCase):
    def test_pause_control_event_is_dispatched_on_gevent_server_loop(self) -> None:
        class FakeLoop:
            def __init__(self):
                self.scheduled = 0

            def run_callback_threadsafe(self, callback):
                self.scheduled += 1
                callback()

        loop = FakeLoop()
        racer = Racer.__new__(Racer)
        racer.racer_id = 0
        racer.step_timeout = 0.5
        racer.sio = Mock()
        racer._wsgi_server = SimpleNamespace(loop=loop)

        racer._emit_socket_event("AICAR_SIMULATION_PAUSE", sid="unity-sid")

        self.assertEqual(loop.scheduled, 1)
        racer.sio.emit.assert_called_once_with("AICAR_SIMULATION_PAUSE", to="unity-sid")

    def test_socket_event_uses_direct_emit_without_gevent_loop(self) -> None:
        racer = Racer.__new__(Racer)
        racer.racer_id = 0
        racer.step_timeout = 0.5
        racer.sio = Mock()
        racer._wsgi_server = SimpleNamespace()

        racer._emit_socket_event("AICAR_SIMULATION_RESUME", sid="unity-sid")

        racer.sio.emit.assert_called_once_with("AICAR_SIMULATION_RESUME", to="unity-sid")

    def test_simulator_is_held_only_during_policy_updates(self) -> None:
        class FakeVecEnv:
            def __init__(self):
                self.pauses = []

            def env_method(self, method, *args):
                self.pauses.append((method, *args))

        callback = SimulatorPauseCallback()
        fake_env = FakeVecEnv()
        callback.model = SimpleNamespace(get_env=lambda: fake_env)

        callback._on_rollout_start()
        callback._on_rollout_end()
        callback._on_rollout_start()
        callback._on_training_end()

        self.assertEqual(
            fake_env.pauses,
            [
                ("set_simulation_paused", True),
                ("set_simulation_paused", False),
                ("resume_simulation",),
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
        env._positive_episode_return = 0.0
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
            progress.setdefault("advanced_m", 0.0)
            progress.setdefault("time_since_push_s", 0.0)
            progress.setdefault("current_progress_m", progress.get("progress_m", 0.0))
            progress.setdefault("current_line", progress.get("line"))
            progress.setdefault("current_projection_valid", True)
        env.frontier_stagnation_seconds = frontier_stagnation_seconds
        env.terminate_on_collision = terminate_on_collision
        env.lap_tracker = lap_tracker or SimpleNamespace(
            update=lambda *_: {"lap_count": 0},
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
        self.assertEqual(info["reward_components"]["collision"], -100.0)
        self.assertEqual(info["reward_components"]["episode_failure"], 0.0)

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

    def test_lap_crossing_never_ends_life_or_awards_terminal_bonus(self) -> None:
        class CompletingLapTracker:
            supported = True
            length_m = 120.0
            lap_count = 9
            def update(self, *_args):
                self.lap_count = 10
                return {"completed_attempt_average_frontier_speed_mps": 3.0}
            def sample(self, *_args):
                return {"lap_supported": True, "lap_count": self.lap_count}

        snap = SimpleNamespace(
            collision_count=0, timestamp=10.0, position=(1.0, 0.0, 1.0),
            v_long=2.0, true_speed=2.0, collision=False, heading_yaw=0.0,
            slip_angle=0.0, lidar=np.ones(1080),
        )
        progress = {
            "progress_m": 120.0, "advanced_m": 0.0,
            "time_since_push_s": 0.0, "line": [[0.0, 0.0], [1.0, 0.0]],
            "speed_mps": 0.0,
        }
        _, reward, terminated, truncated, info = self._step_with_collision(
            snap, False, progress=progress, lap_tracker=CompletingLapTracker(),
            laps_per_episode=0,
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["lap_count"], 10)
        self.assertFalse(info["episode_won"])
        self.assertNotIn("lap_reward", info)
        self.assertEqual(reward, -0.125)

    def test_ten_lap_target_ends_episode_as_success_without_failure_deduction(self) -> None:
        class CompletingLapTracker:
            supported = True
            length_m = 120.0
            lap_count = 9
            def update(self, *_args):
                self.lap_count = 10
                return {"lap_count": self.lap_count}
            def sample(self, *_args):
                return {"lap_supported": True, "lap_count": self.lap_count}

        snap = SimpleNamespace(
            collision_count=0, timestamp=10.0, position=(1.0, 0.0, 1.0),
            v_long=2.0, true_speed=2.0, collision=False, heading_yaw=0.0,
            slip_angle=0.0, lidar=np.ones(1080),
        )
        progress = {
            "progress_m": 120.0, "advanced_m": 0.0,
            "time_since_push_s": 0.0, "line": [[0.0, 0.0], [1.0, 0.0]],
            "speed_mps": 0.0,
        }
        _, reward, terminated, truncated, info = self._step_with_collision(
            snap, False, progress=progress, lap_tracker=CompletingLapTracker(),
            laps_per_episode=10,
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "lap_target")
        self.assertTrue(info["episode_won"])
        self.assertEqual(info["reward_components"]["episode_failure"], 0.0)
        self.assertEqual(reward, -0.125)

    def test_collision_remains_episode_end_even_on_lap_crossing(self) -> None:
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
            laps_per_episode=1,
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "collision")
        self.assertFalse(info["episode_won"])
        self.assertLess(reward, 0.0)

    def test_frontier_stall_resets_without_terminal_failure_cost(self) -> None:
        snap = SimpleNamespace(
            collision_count=1, timestamp=10.0, position=(1.0, 0.0, 1.0),
            v_long=1.0, true_speed=1.0, collision=False, heading_yaw=0.0,
            slip_angle=0.0, lidar=np.ones(1080),
        )
        progress = {
            "progress_m": 1.0,
            "advanced_m": 0.0,
            "time_since_push_s": 10.0,
            "line": [[0.0, 0.0], [1.0, 0.0]],
            "speed_mps": 0.0,
        }
        _, reward, terminated, truncated, info = self._step_with_collision(
            snap, True, progress=progress, frontier_stagnation_seconds=10.0,
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "frontier_stagnation")
        self.assertEqual(info["reward_components"]["episode_failure"], 0.0)
        self.assertEqual(reward, info["reward_components"]["time_cost"])

    def test_env_pays_frontier_high_water_push_not_current_route_motion(self) -> None:
        snap = SimpleNamespace(
            collision_count=1, timestamp=10.0, position=(1.0, 0.0, 1.0),
            v_long=1.0, true_speed=1.0, collision=False, heading_yaw=0.0,
            slip_angle=0.0, lidar=np.ones(1080),
        )
        progress = {
            "progress_m": 1.0,
            "advanced_m": 0.0,
            "current_delta_m": 0.5,
            "time_since_push_s": 0.0,
            "line": [[0.0, 0.0], [1.0, 0.0]],
            "speed_mps": 0.0,
        }
        _, reward, _, _, info = self._step_with_collision(
            snap, True, progress=progress,
        )
        self.assertEqual(info["reward_components"]["route_progress"], 0.0)
        self.assertEqual(info["reward_components"]["time_cost"], -0.125)
        self.assertAlmostEqual(reward, -0.125)

        progress["advanced_m"] = 0.02
        progress["current_delta_m"] = 0.0
        _, reward, _, _, info = self._step_with_collision(
            snap, True, progress=progress,
        )
        expected_progress_reward = 10.0 * (1.0 + (0.8 / 6.0) ** 2) * 0.02
        self.assertAlmostEqual(
            info["reward_components"]["route_progress"], expected_progress_reward
        )
        self.assertAlmostEqual(reward, expected_progress_reward - 0.125)


if __name__ == "__main__":
    unittest.main()
