"""Smoke tests for initializing a sensor-only PPO actor from teacher labels."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer3.behavior_cloning import (
    behavior_clone_actor,
    collect_policy_demonstrations,
)
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
            model = make_model(
                env, device="cpu", seed=12, tensorboard_log=None,
                policy_architecture="lidar_cnn",
            )
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

    def test_camera_enabled_policy_accepts_uint8_demonstrations_during_cloning(self) -> None:
        class CameraEnv(_SensorEnv):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Dict({
                    "lidar": gym.spaces.Box(0.0, 1.0, shape=(LIDAR_BEAMS,), dtype=np.float32),
                    "state": gym.spaces.Box(-5.0, 5.0, shape=(STATE_DIM,), dtype=np.float32),
                    "camera": gym.spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
                })

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return {
                    "lidar": np.ones(LIDAR_BEAMS, np.float32),
                    "state": np.zeros(STATE_DIM, np.float32),
                    "camera": np.full((90, 160, 3), 128, np.uint8),
                }, {}

        env = DummyVecEnv([CameraEnv])
        try:
            model = make_model(
                env, device="cpu", seed=13, tensorboard_log=None,
                policy_architecture="lidar_camera_cnn",
            )
            rng = np.random.default_rng(13)
            count = 64
            state = rng.normal(size=(count, STATE_DIM)).astype(np.float32)
            summary = behavior_clone_actor(
                model,
                {
                    "lidar": rng.uniform(0.2, 1.0, size=(count, LIDAR_BEAMS)).astype(np.float32),
                    "state": state,
                    "camera": rng.integers(0, 256, size=(count, 90, 160, 3), dtype=np.uint8),
                    "actions": np.tanh(state[:, :2]).astype(np.float32),
                },
                epochs=1, batch_size=16, seed=13,
            )
            self.assertEqual(summary["transitions"], count)
            self.assertTrue(np.isfinite(summary["final_validation_loss"]))
            self.assertFalse(model.policy.normalize_images)
        finally:
            env.close()

    def test_policy_collection_persists_only_student_inputs_and_teacher_actions(self) -> None:
        class FakeVecEnv:
            num_envs = 2

            def reset(self):
                return {
                    "lidar": np.full((2, LIDAR_BEAMS), 0.25, np.float32),
                    "state": np.full((2, STATE_DIM), 0.5, np.float32),
                }

            def env_method(self, method):
                self.method = method
                return [
                    {"lidar": np.full(LIDAR_BEAMS, 0.9, np.float32),
                     "state": np.full(STATE_DIM, 0.9, np.float32)}
                    for _ in range(2)
                ]

            def step(self, actions):
                self.actions = actions
                return self.reset(), np.zeros(2), np.zeros(2, dtype=bool), [{}, {}]

        class FakeTeacher:
            def predict(self, observations, deterministic):
                assert deterministic
                assert np.all(observations["state"] == 0.9)
                return np.tile(np.asarray([0.1, -0.2], np.float32), (2, 1)), None

        with TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            dataset, summary = collect_policy_demonstrations(
                FakeVecEnv(), FakeTeacher(), steps=3, output_path=path
            )
            self.assertEqual(summary, {"transitions": 6, "environments": 2})
            self.assertEqual(set(dataset), {"lidar", "state", "actions"})
            self.assertTrue(np.all(dataset["lidar"] == 0.25))
            self.assertTrue(np.all(dataset["state"] == 0.5))
            self.assertTrue(np.allclose(dataset["actions"], [0.1, -0.2]))
            with np.load(path) as saved:
                self.assertEqual(set(saved.files), {"lidar", "state", "actions"})

    def test_dagger_collection_executes_student_but_stores_teacher_labels(self) -> None:
        class FakeVecEnv:
            num_envs = 1

            def reset(self):
                return {
                    "lidar": np.full((1, LIDAR_BEAMS), 0.25, np.float32),
                    "state": np.full((1, STATE_DIM), 0.5, np.float32),
                }

            def env_method(self, method):
                return [{
                    "lidar": np.full(LIDAR_BEAMS, 0.9, np.float32),
                    "state": np.full(STATE_DIM, 0.9, np.float32),
                }]

            def step(self, actions):
                self.executed_actions = actions.copy()
                return self.reset(), np.zeros(1), np.zeros(1, dtype=bool), [{}]

        class FakePolicy:
            def __init__(self, action):
                self.action = np.asarray([action], dtype=np.float32)

            def predict(self, observations, deterministic):
                self.deterministic = deterministic
                return self.action.copy(), None

        env = FakeVecEnv()
        with TemporaryDirectory() as directory:
            dataset, _ = collect_policy_demonstrations(
                env,
                FakePolicy([0.1, -0.2]),
                steps=1,
                output_path=Path(directory) / "dagger.npz",
                behavior_policy=FakePolicy([0.7, 0.6]),
            )
        self.assertTrue(np.allclose(env.executed_actions[0], [0.7, 0.6]))
        self.assertTrue(np.allclose(dataset["actions"][0], [0.1, -0.2]))

    def test_official_route_teacher_collection_keeps_camera_and_excludes_privileged_state(self) -> None:
        from src.layer3.behavior_cloning import collect_centerline_demonstrations

        class FakeVecEnv:
            num_envs = 1

            def reset(self):
                return {
                    "lidar": np.ones((1, LIDAR_BEAMS), np.float32),
                    "state": np.zeros((1, STATE_DIM), np.float32),
                    "camera": np.full((1, 90, 160, 3), 128, np.uint8),
                }

            def env_method(self, method):
                assert method == "get_current_info"
                return [{"position": [1.0, 2.0], "yaw": 0.25, "current_progress_m": 3.0}]

            def step(self, actions):
                assert actions.shape == (1, 2)
                return self.reset(), np.zeros(1), np.zeros(1, dtype=bool), [{}]

        class Expert:
            def action_batch(self, infos):
                assert infos[0]["position"] == [1.0, 2.0]
                return np.asarray([[0.2, -0.1]], dtype=np.float32)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "official-demo.npz"
            dataset, summary = collect_centerline_demonstrations(
                FakeVecEnv(), Expert(), steps=2, output_path=path
            )
            self.assertEqual(summary["transitions"], 2)
            self.assertEqual(set(dataset), {"lidar", "state", "camera", "actions"})
            with np.load(path) as saved:
                self.assertEqual(set(saved.files), {"lidar", "state", "camera", "actions"})
                self.assertEqual(saved["camera"].dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
