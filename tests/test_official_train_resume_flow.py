"""Exercise the full non-BC official trainer path with a fake simulator/model."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from src.layer3 import official_train


class LidarCameraStateExtractor:
    lidar_pooling = True


class _ResumeModel:
    def __init__(self, env):
        self.env = env
        self.num_timesteps = 0
        self._n_updates = 0
        self.n_steps = 2048
        self.n_envs = 1
        self.n_epochs = 4
        self.gamma = 0.9995
        self.gae_lambda = 0.95
        self.batch_size = 64
        self.learning_rate = 1e-5
        self.target_kl = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.ent_coef = 0.0
        self.clip_range = lambda _: 0.1
        self.lr_schedule = lambda _: 1e-5
        self.rollout_buffer = SimpleNamespace(buffer_size=2048, n_envs=1)
        self.observation_space = SimpleNamespace(spaces={
            "camera": None, "lidar": None, "state": None,
        })
        self.policy = SimpleNamespace(
            log_std=torch.log(torch.tensor([0.01, 0.05])),
            features_extractor=LidarCameraStateExtractor(),
            normalize_images=False,
            optimizer=SimpleNamespace(param_groups=[{"lr": 1e-5}]),
        )
        self.learn_call = None

    def get_env(self):
        return self.env

    def set_logger(self, logger):
        self.logger = logger

    def learn(self, *, total_timesteps, callback, progress_bar, reset_num_timesteps):
        self.learn_call = {
            "total_timesteps": total_timesteps,
            "callback": callback,
            "progress_bar": progress_bar,
            "reset_num_timesteps": reset_num_timesteps,
        }

    def save(self, path):
        Path(f"{path}.zip").write_bytes(b"fake trained PPO checkpoint")


class _FakeEnv:
    num_envs = 1


@pytest.mark.parametrize(
    "profile,expected_preset,expected_learning_rate",
    [
        ("bc_smoke", "bc_smoke_locked", 1e-5),
        ("bc_campaign", "bc_campaign_locked", 3e-5),
    ],
)
def test_authenticated_profile_resume_writes_effective_config_and_reaches_learn(
    tmp_path, monkeypatch, profile, expected_preset, expected_learning_rate,
):
    resume = tmp_path / "parent" / "ppo_warmstart.zip"
    resume.parent.mkdir()
    resume.write_bytes(b"authenticated warmstart fixture")
    out_dir = tmp_path / "official_run_smoke"
    route = tmp_path / "official_centerline.csv"
    route.write_text("x,y\n0,0\n1,0\n", encoding="utf-8")
    route.with_name("official_frontier.json").write_text(
        json.dumps({"route_length_m": 1.0}), encoding="utf-8"
    )
    env = _FakeEnv()
    model = _ResumeModel(env)
    model.learning_rate = expected_learning_rate
    model.lr_schedule = lambda _: expected_learning_rate
    model.policy.optimizer.param_groups[0]["lr"] = expected_learning_rate
    sidecars = []

    monkeypatch.setenv("AICAR_SIMULATOR_BACKEND", "docker")
    monkeypatch.delenv("HUB_URL", raising=False)
    monkeypatch.delenv("AICAR_RUN_ID", raising=False)
    monkeypatch.setattr(official_train, "_verified_ready_bc_resume_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(official_train, "_training_resume_lineage", lambda _: {
        "parent_checkpoint": {"path": {"host": str(resume)}, "sha256": "c" * 64},
        "bc_ancestor_checkpoint": {
            "path": {"host": str(tmp_path / "ancestor" / "behavior_cloned_policy.zip")},
            "sha256": "b" * 64,
        },
        "verification": "ppo_checkpoint_and_authenticated_bc_ancestry",
    })
    monkeypatch.setattr(
        "src.layer3.official_tracks.official_frontier_path", lambda: route
    )
    monkeypatch.setattr(
        "src.layer3.official_envs.make_official_vec_env", lambda *args, **kwargs: env
    )
    monkeypatch.setattr("src.layer3.official_envs.close_official_vec_env", lambda _: None)
    monkeypatch.setattr(
        "src.layer3.official_policy.resume_ppo_checkpoint",
        lambda path, **kwargs: model,
    )
    monkeypatch.setattr("src.layer3.hub_callback.maybe_hub_callback", lambda *args, **kwargs: None)
    monkeypatch.setattr("stable_baselines3.common.logger.configure", lambda **kwargs: object())

    def record_sidecar(path, **kwargs):
        sidecar = Path(f"{path}.lineage.json")
        sidecar.write_text(json.dumps({"verification": "test fixture"}), encoding="utf-8")
        sidecars.append((Path(path), kwargs))
        return sidecar

    monkeypatch.setattr(official_train, "write_ppo_lineage_sidecar", record_sidecar)

    result = official_train.train_official(
        out_dir=out_dir,
        total_timesteps=8192,
        resume=resume,
        device="cpu",
        automatic_evaluation_enabled=False,
        behavior_cloning_enabled=False,
        exploration_profile=profile,
        learning_rate=expected_learning_rate,
        target_kl=0.01,
        checkpoint_every=2048,
    )

    saved_config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert model.learn_call is not None
    assert model.learn_call["total_timesteps"] == 8192
    assert model.learn_call["reset_num_timesteps"] is False
    assert saved_config["behavior_cloning"]["enabled"] is False
    assert saved_config["behavior_cloning"]["teacher_controller"] is None
    assert saved_config["ppo"]["preset"] == expected_preset
    assert saved_config["ppo"]["effective"]["target_kl"] == 0.01
    assert saved_config["ppo"]["effective"]["clip_range"] == 0.1
    assert saved_config["ppo"]["effective"]["ent_coef"] == 0.0
    assert saved_config["ppo"]["effective"]["learning_rate"] == expected_learning_rate
    assert saved_config["ppo_exploration"]["profile"] == profile
    assert saved_config["ppo_exploration"]["warm_start_source"] == "verified_ppo_descendant"
    assert saved_config["ppo_exploration"]["resume_checkpoint_sha256"] == "c" * 64
    assert saved_config["ppo_exploration"]["bc_ancestor_checkpoint_sha256"] == "b" * 64
    assert saved_config["exploration_schedule"]["profile"] == profile
    assert saved_config["exploration_schedule"]["minimum_action_std_by_action"] == (
        [0.01, 0.05] if profile == "bc_smoke" else [0.005, 0.02]
    )
    assert saved_config["exploration_schedule"]["maximum_action_std_by_action"] == [0.01, 0.05]
    assert saved_config["automatic_evaluation"]["enabled"] is False
    assert result == out_dir / "official_policy_latest.zip"
    assert result.is_file()
    assert sidecars and sidecars[0][0] == result
