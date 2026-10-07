"""Pure tests for deterministic model-evaluation summaries."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.layer3.evaluate import evaluate_until_episode_end
from src.layer3.train import EvaluationCallback


class EvaluationSummaryTests(unittest.TestCase):
    def test_evaluation_records_per_lap_speed_throttle_and_steering(self) -> None:
        class FakeModel:
            def predict(self, _obs, *, deterministic):
                self.assert_deterministic = deterministic
                return [0.0], None

        class FakeEnv:
            def reset(self):
                self.steps = 0
                return [[0.0]]

            def step(self, _action):
                self.steps += 1
                info = {
                    "frontier_advanced_m": 0.5,
                    "collision_event": False,
                    "lap_times_s": [10.0] if self.steps == 2 else [],
                    "step_duration_s": 1.0,
                    "true_speed": 4.0 if self.steps == 1 else 6.0,
                    "v_long": 3.0 if self.steps == 1 else 5.0,
                    "current_route_speed_mps": 4.0 if self.steps == 1 else 6.0,
                    "throttle_command": 0.5 if self.steps == 1 else 1.0,
                    "steering_command": -0.4 if self.steps == 1 else 0.2,
                }
                return [[0.0]], [1.0], [False], [info]

        observed = []
        result = evaluate_until_episode_end(
            FakeModel(), FakeEnv(), frame_skip=1, lap_target=1,
            on_step=observed.append,
        )

        diagnostic = result["lap_diagnostics"][0]
        self.assertEqual(diagnostic["lap"], 1)
        self.assertEqual(diagnostic["lap_time_s"], 10.0)
        self.assertAlmostEqual(diagnostic["mean_speed_mps"], 5.0)
        self.assertEqual(diagnostic["max_speed_mps"], 6.0)
        self.assertAlmostEqual(diagnostic["mean_forward_speed_mps"], 4.0)
        self.assertAlmostEqual(diagnostic["mean_route_speed_mps"], 5.0)
        self.assertAlmostEqual(diagnostic["mean_throttle"], 0.75)
        self.assertAlmostEqual(diagnostic["full_throttle_fraction"], 0.5)
        self.assertAlmostEqual(diagnostic["mean_steering"], -0.1)
        self.assertAlmostEqual(diagnostic["max_abs_steering"], 0.4)
        self.assertEqual(observed[-1]["_evaluation_metrics"]["lap_diagnostics"], [diagnostic])

    def test_live_evaluation_metrics_publish_without_pose_data(self) -> None:
        callback = EvaluationCallback.__new__(EvaluationCallback)
        callback._live_result = None
        callback._live_pose = None
        callback._active_snapshot_steps = 50_000
        callback._last_live_status_write = 0.0
        callback.runs_per_snapshot = 3
        callback._write_status = Mock()

        callback._on_evaluation_step({
            "_evaluation_metrics": {
                "frontier_distance_m": 12.5,
                "frontier_speed_mps": 2.5,
                "simulated_seconds": 5.0,
                "laps_observed": 1,
                "lap_times_s": [10.0],
                "best_10_lap_time_s": None,
                "reward_per_simulated_second": 3.0,
                "collisions": 0,
            },
        }, attempt_index=1)

        self.assertEqual(callback._live_result["frontier_distance_m"], 12.5)
        callback._write_status.assert_called_once_with(
            "running", snapshot_timesteps=50_000,
        )

    def test_evaluation_stops_on_failure_and_uses_actual_elapsed_time(self) -> None:
        class FakeModel:
            deterministic_flags = []
            def predict(self, _obs, *, deterministic):
                self.deterministic_flags.append(deterministic)
                return [0.0], None

        class FakeEnv:
            def __init__(self):
                self.steps = 0
            def reset(self):
                self.steps = 0
                return [[0.0]]
            def step(self, _action):
                self.steps += 1
                done = self.steps == 2
                return [[0.0]], [2.0], [done], [{
                    "frontier_advanced_m": 0.5,
                    "collision_event": False,
                    "lap_times_s": [],
                    "termination_reason": "frontier_stagnation" if done else None,
                }]

        model = FakeModel()
        observed = []
        metrics = evaluate_until_episode_end(
            model, FakeEnv(), frame_skip=4, timesteps=123,
            on_step=observed.append,
        )
        self.assertEqual(model.deterministic_flags, [True, True])
        self.assertEqual(metrics["timesteps"], 123)
        self.assertEqual(metrics["frontier_distance_m"], 1.0)
        self.assertEqual(metrics["frontier_speed_mps"], 5.0)
        self.assertEqual(metrics["reward_per_simulated_second"], 20.0)
        self.assertEqual(metrics["total_reward"], 4.0)
        self.assertEqual(metrics["failed_episodes"], 1)
        self.assertEqual(metrics["stop_reason"], "frontier_stagnation")
        self.assertAlmostEqual(metrics["simulated_seconds"], 0.2)
        self.assertIsNone(metrics["best_10_lap_time_s"])
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0]["_evaluation_metrics"]["frontier_distance_m"], 0.5)
        self.assertEqual(observed[0]["_evaluation_metrics"]["frontier_speed_mps"], 5.0)
        self.assertEqual(observed[1]["_evaluation_metrics"]["frontier_distance_m"], 1.0)
        self.assertEqual(observed[1]["_evaluation_metrics"]["laps_observed"], 0)
        self.assertEqual(observed[1]["_evaluation_metrics"]["lap_times_s"], [])

    def test_evaluation_stops_after_ten_laps(self) -> None:
        class FakeModel:
            def predict(self, _obs, *, deterministic):
                return [0.0], None

        class FakeEnv:
            def reset(self):
                self.steps = 0
                self.laps = []
                return [[0.0]]

            def step(self, _action):
                self.steps += 1
                self.laps.append(12.0)
                return [[0.0]], [1.0], [False], [{
                    "frontier_advanced_m": 0.25,
                    "collision_event": False,
                    "lap_times_s": list(self.laps),
                }]

        env = FakeEnv()
        observed = []
        metrics = evaluate_until_episode_end(
            FakeModel(), env, frame_skip=4, on_step=observed.append,
        )
        self.assertEqual(env.steps, 10)
        self.assertEqual(metrics["laps_observed"], 10)
        self.assertEqual(metrics["stop_reason"], "lap_target")
        self.assertEqual(metrics["failed_episodes"], 0)
        self.assertEqual(metrics["total_reward"], 10.0)
        self.assertEqual(metrics["best_10_lap_time_s"], 120.0)
        self.assertEqual(len(metrics["lap_times_s"]), 10)
        self.assertEqual(observed[-1]["_evaluation_metrics"]["best_10_lap_time_s"], 120.0)
        self.assertEqual(len(observed[-1]["_evaluation_metrics"]["lap_times_s"]), 10)



if __name__ == "__main__":
    unittest.main()
