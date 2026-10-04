"""Smoke tests for initializing a sensor-only PPO actor from teacher labels."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer3.behavior_cloning import behavior_clone_actor
from src.layer3.train import make_model
from src.layer2.spaces import LIDAR_BEAMS, STATE_DIM


class _SensorEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(0.0, 1.0, shape=(LIDAR_BEAMS,), dtype=np.float32),
            "state": gym.spaces.Box(-5.0, 5.0, shape=(STATE_DIM,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"lidar": np.ones(LIDAR_BEAMS, np.float32), "state": np.zeros(STATE_DIM, np.float32)}, {}

    def step(self, action):
        obs, info = self.reset()
        return obs, 0.0, False, False, info


class BehaviorCloningTests(unittest.TestCase):
    def test_cloning_fits_sensor_observation_to_continuous_action_labels(self) -> None:
        env = DummyVecEnv([_SensorEnv])
        try:
            model = make_model(env, device="cpu", seed=12, tensorboard_log=None)
            rng = np.random.default_rng(12)
            count = 256
            states = rng.normal(size=(count, STATE_DIM)).astype(np.float32)
            lidar = rng.uniform(0.2, 1.0, size=(count, LIDAR_BEAMS)).astype(np.float32)
            actions = np.column_stack((
                np.tanh(states[:, 0] * 0.4),
                np.tanh(states[:, 1] * 0.4),
            )).astype(np.float32)
            summary = behavior_clone_actor(
                model,
                {"lidar": lidar, "state": states, "actions": actions},
                epochs=8,
                batch_size=64,
                seed=12,
            )
            self.assertEqual(summary["transitions"], count)
            self.assertLess(summary["final_validation_loss"], summary["initial_validation_loss"])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
