"""Focused tests for authoritative official PPO checkpoint resume."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer2.spaces import make_action_space, make_observation_space
from src.layer3.official_policy import load_ppo_checkpoint, resume_ppo_checkpoint
from src.layer3.official_train import _verify_official_ppo_config
from src.layer3.train import make_model


class _ObservationEnv(gym.Env):
    observation_space = make_observation_space()
    action_space = make_action_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {
            "lidar": np.ones(1081, dtype=np.float32),
            "state": np.zeros(9, dtype=np.float32),
        }, {}

    def step(self, action):
        del action
        return self.reset()[0], 0.0, False, False, {}


class OfficialPPOResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DummyVecEnv([_ObservationEnv])
        self.requested = {
            "n_steps": 64,
            "learning_rate": 5e-5,
            "n_epochs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        }

    def tearDown(self) -> None:
        self.env.close()

    def _source_model(self):
        model = make_model(
            self.env,
            device="cpu",
            seed=10,
            tensorboard_log=None,
            learning_rate=1e-5,
            n_steps=64,
            n_epochs=1,
            gamma=0.9995,
            gae_lambda=0.9,
            policy_architecture="lidar_cnn",
        )
        model.num_timesteps = 1234
        return model

    def test_resume_transfers_only_policy_weights_and_timestep(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source = self._source_model()
            source.learn(total_timesteps=64)
            self.assertTrue(source.policy.optimizer.state)
            source.num_timesteps = 1234
            source.save(str(source_path))

            resumed = resume_ppo_checkpoint(
                Path(f"{source_path}.zip"),
                env=self.env,
                observation_profile="official_sensors",
                seed=22,
                tensorboard_log=None,
                **self.requested,
            )

            self.assertEqual(resumed.num_timesteps, 1234)
            self.assertEqual(resumed.n_steps, 64)
            self.assertEqual(resumed.rollout_buffer.buffer_size, 64)
            self.assertEqual(resumed.n_epochs, 2)
            self.assertEqual(resumed.gamma, 0.99)
            self.assertEqual(resumed.gae_lambda, 0.95)
            self.assertEqual(resumed.batch_size, 64)
            self.assertEqual(resumed.ent_coef, 0.01)
            self.assertEqual(resumed.vf_coef, 0.5)
            self.assertEqual(resumed.max_grad_norm, 0.5)
            self.assertEqual(float(resumed.lr_schedule(1.0)), 5e-5)
            self.assertEqual(float(resumed.lr_schedule(0.0)), 5e-5)
            self.assertTrue(
                all(group["lr"] == 5e-5 for group in resumed.policy.optimizer.param_groups)
            )
            self.assertEqual(resumed.policy.optimizer.state, {})
            self.assertIsNot(resumed.policy.optimizer, source.policy.optimizer)
            for key, value in source.policy.state_dict().items():
                self.assertTrue(
                    torch.equal(value, resumed.policy.state_dict()[key]),
                    msg=f"policy parameter differs after resume: {key}",
                )

            effective = _verify_official_ppo_config(
                resumed,
                self.requested,
                observation_profile="official_sensors",
            )
            self.assertEqual(effective["rollout_buffer_size"], 64)
            self.assertEqual(effective["policy_architecture"], "lidar_cnn")
            self.assertEqual(effective["policy_extractor"], "LidarStateExtractor")

            resumed_path = Path(temp_dir) / "resumed"
            resumed.save(str(resumed_path))
            reloaded = load_ppo_checkpoint(Path(f"{resumed_path}.zip"), device="cpu")
            self.assertEqual(reloaded.num_timesteps, 1234)
            for key, value in resumed.policy.state_dict().items():
                self.assertTrue(torch.equal(value, reloaded.policy.state_dict()[key]))

    def test_verification_fails_before_learning_when_optimizer_differs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source"
            self._source_model().save(str(path))
            model = resume_ppo_checkpoint(
                Path(f"{path}.zip"),
                env=self.env,
                observation_profile="official_sensors",
                seed=22,
                tensorboard_log=None,
                **self.requested,
            )
            model.policy.optimizer.param_groups[0]["lr"] = 1e-5
            with self.assertRaisesRegex(RuntimeError, "optimizer_learning_rates"):
                _verify_official_ppo_config(
                    model,
                    self.requested,
                    observation_profile="official_sensors",
                )


if __name__ == "__main__":
    unittest.main()
