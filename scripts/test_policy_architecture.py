"""Tests for checkpoint-compatible Layer 3 LiDAR policy architectures."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer3.extractors import (
    LidarStateExtractor,
    PooledLidarStateExtractor,
    policy_kwargs_for_architecture,
)
from src.layer3.train import build_arg_parser
from src.layer4.settings import Settings
from src.layer2.spaces import make_action_space, make_observation_space
from src.layer3.official_policy import load_policy


class PolicyArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(0.0, 1.0, shape=(1081,), dtype=np.float32),
            "state": gym.spaces.Box(-5.0, 5.0, shape=(9,), dtype=np.float32),
        })

    def test_pooled_extractor_reduces_projection_and_preserves_output_shape(self) -> None:
        legacy = LidarStateExtractor(self.observation_space, features_dim=64)
        pooled = PooledLidarStateExtractor(self.observation_space, features_dim=64)
        sample = {
            "lidar": torch.rand(2, 1081),
            "state": torch.rand(2, 9),
        }

        self.assertEqual(tuple(pooled(sample).shape), (2, 64))
        self.assertLess(pooled.merge[0].in_features, legacy.merge[0].in_features)
        pooled(sample).sum().backward()
        self.assertIsNotNone(pooled.lidar_net[0].weight.grad)

    def test_architecture_names_keep_old_checkpoint_class_and_expose_new_one(self) -> None:
        self.assertIs(
            policy_kwargs_for_architecture("lidar_cnn")["features_extractor_class"],
            LidarStateExtractor,
        )
        self.assertIs(
            policy_kwargs_for_architecture("lidar_cnn_pooled")["features_extractor_class"],
            PooledLidarStateExtractor,
        )
        with self.assertRaisesRegex(ValueError, "Unknown policy architecture"):
            policy_kwargs_for_architecture("unknown")

    def test_architecture_is_configurable_from_cli_and_layer4_settings(self) -> None:
        args = build_arg_parser().parse_args([
            "--policy-architecture", "lidar_cnn_pooled",
        ])
        self.assertEqual(args.policy_architecture, "lidar_cnn_pooled")

        settings = Settings(policy_architecture="lidar_cnn_pooled")
        argv = settings.to_train_argv()
        index = argv.index("--policy-architecture")
        self.assertEqual(argv[index + 1], "lidar_cnn_pooled")

    def test_official_policy_loader_can_restore_pooled_checkpoint(self) -> None:
        class ObservationEnv(gym.Env):
            observation_space = make_observation_space()
            action_space = make_action_space()

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return {
                    "lidar": np.ones(1081, dtype=np.float32),
                    "state": np.zeros(9, dtype=np.float32),
                }, {}

            def step(self, action):
                return self.reset()[0], 0.0, False, False, {}

        model = PPO(
            "MultiInputPolicy",
            DummyVecEnv([ObservationEnv]),
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            policy_kwargs=policy_kwargs_for_architecture("lidar_cnn_pooled"),
            device="cpu",
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pooled-policy"
            model.save(str(checkpoint))
            restored = load_policy(Path(f"{checkpoint}.zip"), device="cpu")
            self.assertIsInstance(restored.policy.features_extractor, PooledLidarStateExtractor)


if __name__ == "__main__":
    unittest.main()
