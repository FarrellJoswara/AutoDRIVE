"""Unit test LidarStateExtractor shapes (no Unity)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import numpy as np
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layer2.spaces import LIDAR_BEAMS, STATE_DIM, make_observation_space
from src.layer3.extractors import LidarStateExtractor, LidarCameraStateExtractor, policy_kwargs_for_architecture


def test_lidar_state_extractor_forward_shape():
    space = make_observation_space()
    extractor = LidarStateExtractor(space, features_dim=256)
    batch = 4
    obs = {
        "lidar": torch.rand(batch, LIDAR_BEAMS),
        "state": torch.randn(batch, STATE_DIM),
    }
    out = extractor(obs)
    assert out.shape == (batch, 256)
    assert torch.isfinite(out).all()


def test_lidar_state_extractor_features_dim_matches():
    space = make_observation_space()
    extractor = LidarStateExtractor(space, features_dim=128)
    out = extractor({"lidar": torch.zeros(1, LIDAR_BEAMS), "state": torch.zeros(1, STATE_DIM)})
    assert out.shape == (1, 128)
    assert extractor.features_dim == 128


def test_camera_policy_scales_uint8_frames_once_inside_vision_branch():
    space = make_observation_space(include_camera=True)
    extractor = LidarCameraStateExtractor(space, features_dim=32)
    observed = []
    handle = extractor.camera_net[0].register_forward_pre_hook(
        lambda _module, inputs: observed.append(float(inputs[0].max()))
    )
    try:
        output = extractor({
            "lidar": torch.ones(1, LIDAR_BEAMS),
            "state": torch.zeros(1, STATE_DIM),
            "camera": torch.full((1, 90, 160, 3), 255, dtype=torch.uint8),
        })
    finally:
        handle.remove()
    assert output.shape == (1, 32)
    assert observed == [1.0]
    assert policy_kwargs_for_architecture("lidar_camera_cnn")["normalize_images"] is False


def test_loader_repairs_legacy_checkpoint_camera_scaling():
    import tempfile
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.layer3.official_policy import load_ppo_checkpoint

    class CameraEnv(gym.Env):
        def __init__(self):
            self.observation_space = make_observation_space(include_camera=True)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {
                "lidar": np.ones(LIDAR_BEAMS, np.float32),
                "state": np.zeros(STATE_DIM, np.float32),
                "camera": np.zeros((90, 160, 3), dtype=np.uint8),
            }, {}

        def step(self, action):
            observation, _ = self.reset()
            return observation, 0.0, False, False, {}

    env = DummyVecEnv([CameraEnv])
    try:
        policy_kwargs = policy_kwargs_for_architecture("lidar_camera_cnn")
        policy_kwargs["normalize_images"] = True  # emulate pre-fix checkpoints
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, n_steps=64, batch_size=64)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-camera.zip"
            model.save(checkpoint)
            loaded = load_ppo_checkpoint(checkpoint, observation_profile="official_sensors_camera")
            assert loaded.policy.normalize_images is False
    finally:
        env.close()
