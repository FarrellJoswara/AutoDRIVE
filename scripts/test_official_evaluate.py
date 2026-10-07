"""Focused tests for IROS official race score validation and early exit."""

from __future__ import annotations

import unittest

import numpy as np

from src.layer3.official_evaluate import evaluate_attempt, summarize_attempt


class _FixedPolicy:
    def predict(self, observation, deterministic=True):
        del observation, deterministic
        return np.asarray([0.5, 0.0], dtype=np.float32), None


class _CollisionSequenceEnv:
    def __init__(self, collisions: int):
        self.target_collisions = collisions
        self.steps = 0

    def reset(self):
        return {"lidar": np.ones(1081, dtype=np.float32), "state": np.zeros(9, dtype=np.float32)}, {
            "race_position": [0.0, 0.0, 0.0],
        }

    def step(self, action):
        del action
        self.steps += 1
        info = {
            "race_position": [float(self.steps), 0.0, 0.0],
            "race_lap_times_s": [],
            "race_collisions": min(self.steps, self.target_collisions),
            "raw_collision_count": min(self.steps, self.target_collisions),
            "race_collision_baseline": 0,
            "warmup_lap_times_s": [],
            "warmup_collisions": 0,
            "laps_since_start": 0,
            "control_interval_s": 0.05,
            "lidar_scan_rate_hz": 40.0,
            "throttle_command": 0.5,
            "steering_command": 0.0,
        }
        return {"lidar": np.ones(1081, dtype=np.float32), "state": np.zeros(9, dtype=np.float32)}, 0.0, False, False, info


class OfficialEvaluationTests(unittest.TestCase):
    def test_more_than_ten_collisions_is_disqualified(self):
        result = summarize_attempt(lap_times_s=[80.0] * 10, race_collisions=11)

        self.assertTrue(result["complete"])
        self.assertTrue(result["disqualified"])
        self.assertIsNone(result["adjusted_race_time_s"])

    def test_attempt_stops_at_official_disqualification_threshold(self):
        env = _CollisionSequenceEnv(collisions=20)

        result = evaluate_attempt(
            _FixedPolicy(),
            env,
            wall_timeout_s=30.0,
            max_steps=100,
            attempt_index=1,
        )

        self.assertEqual(env.steps, 11)
        self.assertEqual(result["race_collisions"], 11)
        self.assertEqual(result["stop_reason"], "collision_disqualification_limit")
        self.assertEqual(result["score_status"], "disqualified")
        self.assertFalse(result["valid_for_comparison"])


if __name__ == "__main__":
    unittest.main()
