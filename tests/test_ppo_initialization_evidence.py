"""Offline evidence for conservative camera-PPO initialization and sampling."""

import math

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.initialize_camera_champion import _ObservationEnv
from src.layer3.official_exploration import OfficialExplorationSchedule
from src.layer3.official_policy import resume_ppo_checkpoint
from src.layer3.official_train import _resolve_exploration_settings
from src.layer3.train import make_model


def test_smoke_resume_initializes_exact_ppo_config_and_stochastic_policy(tmp_path):
    env = DummyVecEnv([_ObservationEnv])
    try:
        # A locally-created camera PPO archive exercises the actual SB3 save,
        # load, and clean-optimizer resume path without touching the simulator.
        source = make_model(
            env,
            device="cpu",
            seed=0,
            tensorboard_log=None,
            learning_rate=1e-5,
            n_epochs=4,
            n_steps=2048,
            gamma=0.9995,
            gae_lambda=0.95,
            target_kl=0.01,
            policy_architecture="lidar_camera_cnn",
            batch_size=64,
            clip_range=0.1,
            ent_coef=0.0,
        )
        with torch.no_grad():
            source.policy.log_std.copy_(torch.log(torch.tensor([0.01, 0.05])))
        archive = tmp_path / "camera_parent.zip"
        source.save(str(archive.with_suffix("")))

        model = resume_ppo_checkpoint(
            archive,
            env=env,
            observation_profile="official_sensors_camera",
            learning_rate=1e-5,
            n_steps=2048,
            n_epochs=4,
            gamma=0.9995,
            gae_lambda=0.95,
            target_kl=0.01,
            seed=0,
            device="cpu",
            batch_size=64,
            clip_range=0.1,
            ent_coef=0.0,
        )

        assert model.n_steps == 2048
        assert model.n_epochs == 4
        assert math.isclose(model.gamma, 0.9995)
        assert math.isclose(model.gae_lambda, 0.95)
        assert math.isclose(model.target_kl, 0.01)
        assert model.batch_size == 64
        assert math.isclose(model.clip_range(1.0), 0.1)
        assert math.isclose(model.ent_coef, 0.0)
        assert math.isclose(model.lr_schedule(1.0), 1e-5)
        assert model.policy.normalize_images is False

        current_std = tuple(float(value) for value in model.policy.log_std.detach().exp().cpu())
        exploration = _resolve_exploration_settings(
            requested_profile="bc_smoke",
            overrides={},
            saved=None,
            verified_bc_resume=True,
            current_std=current_std,
        )
        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.tensor(exploration["initial_std"])))
        schedule = OfficialExplorationSchedule(
            model,
            tmp_path,
            std_min=exploration["std_min"],
            std_max=exploration["std_max"],
        )
        schedule.on_rollout_start()
        assert np.allclose(
            model.policy.log_std.detach().exp().cpu().numpy(), [0.01, 0.05],
            rtol=1e-6,
        )

        # A representative official camera observation checks the live policy
        # path. No optimizer update or simulator process is involved.
        observations = {
            "lidar": np.full((1, 1081), 0.65, dtype=np.float32),
            "camera": np.full((1, 90, 160, 3), 96, dtype=np.uint8),
            "state": np.zeros((1, 9), dtype=np.float32),
        }
        samples = np.asarray([
            model.predict(observations, deterministic=False)[0][0]
            for _ in range(32)
        ])
        assert samples.shape == (32, 2)
        assert np.isfinite(samples).all()
        assert np.all(np.abs(samples) <= 1.0)
        assert np.all(samples.std(axis=0) > 0.0)
    finally:
        env.close()
