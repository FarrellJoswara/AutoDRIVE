"""Tests for the bounded deterministic-evaluation action trace."""

from __future__ import annotations

import unittest

import numpy as np

from src.layer3.evaluate import evaluate_until_episode_end


class EvaluationActionTraceTests(unittest.TestCase):
    def test_failed_attempt_retains_policy_action_and_input_sensor_summary(self) -> None:
        class FakeModel:
            def predict(self, _obs, *, deterministic):
                assert deterministic is True
                return np.asarray([[0.2, -0.3]], dtype=np.float32), None

        class FakeEnv:
            def reset(self):
                self.steps = 0
                lidar = np.ones((1, 1081), dtype=np.float32)
                lidar[0, 500] = 0.25
                state = np.zeros((1, 9), dtype=np.float32)
                state[0, 0] = 0.4
                return {"lidar": lidar, "state": state}

            def step(self, _action):
                self.steps += 1
                next_obs = {
                    "lidar": np.zeros((1, 1081), dtype=np.float32),
                    "state": np.zeros((1, 9), dtype=np.float32),
                }
                return next_obs, np.asarray([-100.0]), np.asarray([True]), [{
                    "frontier_advanced_m": 0.1,
                    "collision_event": True,
                    "lap_times_s": [],
                    "step_duration_s": 0.025,
                    "termination_reason": "collision",
                    "throttle_command": 0.6,
                    "steering_command": -0.3,
                    "v_long": 2.0,
                    "true_speed": 2.1,
                    "yaw": 0.5,
                    "position": (1.0, 0.0, 2.0),
                }]

        result = evaluate_until_episode_end(
            FakeModel(), FakeEnv(), frame_skip=1,
        )

        self.assertEqual(result["stop_reason"], "collision")
        self.assertEqual(len(result["initial_action_trace"]), 1)
        sample = result["initial_action_trace"][0]
        self.assertAlmostEqual(sample["policy_action"][0], 0.2)
        self.assertAlmostEqual(sample["policy_action"][1], -0.3)
        self.assertEqual(sample["throttle_command"], 0.6)
        self.assertEqual(sample["steering_command"], -0.3)
        self.assertAlmostEqual(sample["state"][0], 0.4)
        self.assertAlmostEqual(sample["lidar_min_front"], 0.25)
        self.assertEqual(sample["position"], [1.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
