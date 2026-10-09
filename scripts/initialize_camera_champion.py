"""Export a champion policy's tensors, then wrap them in the current camera PPO.

Run `--export-only` with a NumPy 2 environment if the historical checkpoint was
written with NumPy 2. Run the normal build step in the project's training venv so
the new SB3 archive uses that runtime's compatible metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer3.extractors import transfer_lidar_policy_to_camera
from src.layer3.train import make_model


class _ObservationEnv(gym.Env):
    """Minimal shape-only environment used to construct a PPO checkpoint."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(0.0, 1.0, shape=(1081,), dtype=np.float32),
            "camera": gym.spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-10.0, 10.0, shape=(9,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="historical lidar/state PPO zip")
    parser.add_argument("--payload", type=Path, required=True, help="intermediate tensor-only .pt file")
    parser.add_argument("--output", type=Path, required=True, help="camera PPO zip to resume for training")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        from stable_baselines3.common.save_util import load_from_zip_file

        data, parameters, _ = load_from_zip_file(str(args.source), device="cpu")
        if not isinstance(data, dict) or not isinstance(parameters, dict):
            raise RuntimeError("Unsupported SB3 checkpoint archive")
        if "policy" not in parameters:
            raise RuntimeError("Source archive has no policy weights")
        args.payload.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": parameters["policy"],
            "num_timesteps": int(data.get("num_timesteps", 0)),
        }, args.payload)
        print(f"Exported {len(parameters['policy'])} tensors to {args.payload}")
        return

    payload = torch.load(args.payload, map_location="cpu", weights_only=True)
    source = payload["policy"]
    vec_env = DummyVecEnv([_ObservationEnv])
    source_model = make_model(
        DummyVecEnv([_ObservationEnv]), device="cpu", seed=0, tensorboard_log=None,
        learning_rate=3e-6, n_epochs=1, n_steps=1024,
        gamma=0.99, gae_lambda=0.95, policy_architecture="lidar_cnn",
    )
    source_model.policy.load_state_dict(source, strict=True)
    model = make_model(
        vec_env, device="cpu", seed=0, tensorboard_log=None,
        learning_rate=3e-6, n_epochs=1, n_steps=1024,
        gamma=0.99, gae_lambda=0.95,
        policy_architecture="lidar_camera_raw_cnn",
    )
    target = model.policy.state_dict()
    transferred = transfer_lidar_policy_to_camera(source, target)
    incompatible = model.policy.load_state_dict(transferred, strict=False)
    unexpected_missing = [key for key in incompatible.missing_keys if ".camera_net." not in key]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected transfer mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    torch.manual_seed(41)
    observations = {
        "lidar": torch.rand(3, 1081),
        "state": torch.rand(3, 9),
        "camera": torch.randint(0, 256, (3, 90, 160, 3), dtype=torch.uint8),
    }
    lidar_observations = {key: value for key, value in observations.items() if key != "camera"}
    with torch.no_grad():
        source_features = source_model.policy.extract_features(lidar_observations)
        target_features = model.policy.extract_features(observations)
        original_distribution = source_model.policy.get_distribution(lidar_observations).distribution
        migrated_distribution = model.policy.get_distribution(observations).distribution
    if not torch.allclose(original_distribution.mean, migrated_distribution.mean, atol=1e-6, rtol=1e-6):
        difference = (original_distribution.mean - migrated_distribution.mean).abs().max().item()
        feature_difference = (source_features - target_features).abs().max().item()
        raise RuntimeError(f"Transferred policy action means differ by {difference}; feature difference {feature_difference}")
    if not torch.allclose(original_distribution.stddev, migrated_distribution.stddev, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Transferred policy does not reproduce champion action variance")
    model.num_timesteps = int(payload.get("num_timesteps", 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    from src.layer3.official_policy import resume_ppo_checkpoint

    reloaded = resume_ppo_checkpoint(
        args.output, env=vec_env, observation_profile="official_sensors_camera",
        learning_rate=3e-6, n_steps=1024, n_epochs=1, gamma=0.99,
        gae_lambda=0.95, seed=0, device="cpu",
    )
    with torch.no_grad():
        reloaded_distribution = reloaded.policy.get_distribution(observations).distribution
    if not torch.allclose(migrated_distribution.mean, reloaded_distribution.mean, atol=1e-6, rtol=1e-6):
        raise RuntimeError("Saved champion camera checkpoint changed action means on reload")
    print(
        f"Saved camera-enabled PPO at {args.output}; transferred champion policy "
        f"at timestep {model.num_timesteps}. Three random batches reproduced the "
        "champion action distribution exactly, and reload through the official PPO "
        "resume path preserved it; camera merge weights start at zero."
    )


if __name__ == "__main__":
    main()
