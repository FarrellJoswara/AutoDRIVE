"""Regression checks for official sensor-only scan history observations."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.spaces import make_observation_space, snapshot_to_obs
from src.layer2.autodrive_env import AutoDriveEnv
from src.layer3.extractors import TemporalLidarStateExtractor, policy_kwargs_for_architecture
from src.layer3.official_policy import load_policy


class TemporalSensorObservationTests(unittest.TestCase):
    def snapshot(self, distance: float) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            lidar_ranges=np.full(1081, distance, dtype=np.float32),
            lidar_valid=True,
            lidar_range_min=0.06,
            lidar_range_max=10.0,
            v_long=99.0,
            v_lat=50.0,
        )

    def test_history_is_ordered_and_current_scan_is_last(self) -> None:
        previous = [np.full(1081, value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
        obs = snapshot_to_obs(
            self.snapshot(4.036), 0.0, 0.0,
            forward_speed_mps=2.0,
            lateral_speed_mps=0.0,
            lidar_history=previous,
        )
        self.assertEqual(obs["lidar"].shape, (4, 1081))
        self.assertTrue(np.allclose(obs["lidar"][:3], np.stack(previous)))
        self.assertTrue(np.allclose(obs["lidar"][3], 0.4, atol=1e-6))
        self.assertAlmostEqual(float(obs["state"][0]), 2.0 / 22.88)
        self.assertEqual(float(obs["state"][1]), 0.0)

    def test_single_scan_observation_contract_is_unchanged(self) -> None:
        obs = snapshot_to_obs(self.snapshot(5.03), 0.0, 0.0)
        self.assertEqual(obs["lidar"].shape, (1081,))
        self.assertEqual(make_observation_space().spaces["lidar"].shape, (1081,))
        self.assertEqual(
            make_observation_space(lidar_history_frames=4).spaces["lidar"].shape,
            (4, 1081),
        )

    def test_layer2_history_advances_and_keeps_privileged_state_out(self) -> None:
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        env.observation_profile = "official_sensors_history"
        env.lidar_beams = 1081
        env._lidar_history = [np.full(1081, value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
        env._observation_encoder_positions = (0.0, 0.0)
        snap = self.snapshot(5.03)
        snap.encoder_left = 2.0
        snap.encoder_right = 2.0
        obs = env._policy_observation(
            snap, 0.1, -0.1, lidar_beams=1081, elapsed_s=0.1
        )
        self.assertTrue(np.allclose(obs["lidar"][:3], np.stack([
            np.full(1081, value, dtype=np.float32) for value in (0.1, 0.2, 0.3)
        ])))
        self.assertTrue(np.allclose(
            env._lidar_history[0], np.full(1081, 0.2, dtype=np.float32)
        ))
        self.assertEqual(float(obs["state"][1]), 0.0)
        # The RL observation intentionally uses LiDAR ego-motion, never wheel
        # encoder speed (which can report spin while the car is stopped).
        self.assertEqual(float(obs["state"][0]), 0.0)

    def test_temporal_extractor_consumes_history_and_produces_features(self) -> None:
        obs_space = make_observation_space(lidar_history_frames=4)
        extractor = TemporalLidarStateExtractor(obs_space, features_dim=64)
        sample = {
            "lidar": torch.rand(2, 4, 1081),
            "state": torch.rand(2, 9),
        }
        self.assertEqual(tuple(extractor(sample).shape), (2, 64))
        self.assertIs(
            policy_kwargs_for_architecture("temporal_lidar_cnn")["features_extractor_class"],
            TemporalLidarStateExtractor,
        )

    def test_temporal_checkpoint_round_trips_through_competition_loader(self) -> None:
        class TemporalObservationEnv(gym.Env):
            observation_space = make_observation_space(lidar_history_frames=4)
            action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return {"lidar": np.ones((4, 1081), dtype=np.float32), "state": np.zeros(9, dtype=np.float32)}, {}

            def step(self, action):
                observation, info = self.reset()
                return observation, 0.0, False, False, info

        model = PPO(
            "MultiInputPolicy",
            DummyVecEnv([TemporalObservationEnv]),
            n_steps=64,
            batch_size=64,
            n_epochs=1,
            policy_kwargs=policy_kwargs_for_architecture("temporal_lidar_cnn"),
            device="cpu",
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "temporal-policy"
            model.save(str(checkpoint))
            restored = load_policy(
                Path(f"{checkpoint}.zip"),
                device="cpu",
                observation_profile="official_sensors_history",
            )
            sample = {"lidar": np.ones((4, 1081), dtype=np.float32), "state": np.zeros(9, dtype=np.float32)}
            action, _ = restored.predict(sample, deterministic=True)
            self.assertEqual(np.asarray(action).shape, (2,))


if __name__ == "__main__":
    unittest.main()
