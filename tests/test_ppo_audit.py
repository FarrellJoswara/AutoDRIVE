import json
import math
from threading import Event

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.initialize_camera_champion import _ObservationEnv
from src.layer3.official_exploration import OfficialExplorationSchedule
from src.layer3.official_policy import resume_ppo_checkpoint
from src.layer3.ppo_audit import OperatorStopCallback, PostOptimizerAuditCallback
from src.layer3.train import make_model
from src.layer3.official_train import (
    AutomaticEvaluationSchedule,
    _BC_SMOKE_FIXED_PPO,
    _BC_SMOKE_PPO,
    _resolve_exploration_settings,
    _step_cadence_origin,
)


def test_exploration_profile_presets_overrides_and_resume_inheritance():
    smoke = _resolve_exploration_settings(
        requested_profile="bc_smoke",
        overrides={},
        saved=None,
        verified_bc_resume=True,
        current_std=(0.2, 0.2),
    )
    assert smoke["initial_std"] == (0.01, 0.05)
    assert smoke["std_min"] == smoke["std_max"] == (0.01, 0.05)
    try:
        _resolve_exploration_settings(
            requested_profile="bc_smoke",
            overrides={},
            saved=None,
            verified_bc_resume=False,
            current_std=(0.2, 0.2),
        )
    except ValueError as exc:
        assert "authenticated BC-derived resume" in str(exc)
    else:
        raise AssertionError("BC smoke profile accepted an unauthenticated parent")

    campaign = _resolve_exploration_settings(
        requested_profile="bc_campaign",
        overrides={},
        saved=None,
        verified_bc_resume=True,
        current_std=(0.2, 0.2),
    )
    assert campaign["initial_std"] == (0.01, 0.05)
    assert campaign["std_min"] == (0.005, 0.02)
    assert campaign["std_max"] == (0.01, 0.05)

    inherited = _resolve_exploration_settings(
        requested_profile=None,
        overrides={},
        saved={
            "profile": "bc_campaign",
            "minimum_action_std_by_action": [0.005, 0.02],
            "maximum_action_std_by_action": [0.01, 0.05],
        },
        verified_bc_resume=True,
        current_std=(0.008, 0.03),
    )
    assert inherited["profile"] == "bc_campaign"
    assert inherited["initial_std"] is None
    assert inherited["std_min"] == campaign["std_min"]
    assert inherited["std_max"] == campaign["std_max"]
    assert inherited["inherited_from_resume"] is True

    overridden = _resolve_exploration_settings(
        requested_profile="bc_campaign",
        overrides={"initial_throttle": 0.009},
        saved=None,
        verified_bc_resume=True,
        current_std=(0.2, 0.2),
    )
    assert overridden["initial_std"] == (0.009, 0.05)
    try:
        _resolve_exploration_settings(
            requested_profile="bc_smoke",
            overrides={"max_steering": 0.04},
            saved=None,
            verified_bc_resume=True,
            current_std=(0.2, 0.2),
        )
    except ValueError as exc:
        assert "minimum must not exceed maximum" in str(exc)
    else:
        raise AssertionError("inconsistent initial exploration settings were accepted")

    try:
        _resolve_exploration_settings(
            requested_profile="bc_campaign",
            overrides={"initial_throttle": 0.011},
            saved=None,
            verified_bc_resume=True,
            current_std=(0.2, 0.2),
        )
    except ValueError as exc:
        assert "within their bounds" in str(exc)
    else:
        raise AssertionError("initial exploration standard deviation outside bounds was accepted")


def test_exploration_resume_config_requires_two_action_values():
    try:
        _resolve_exploration_settings(
            requested_profile=None,
            overrides={},
            saved={
                "profile": "bc_campaign",
                "minimum_action_std_by_action": [0.005],
                "maximum_action_std_by_action": [0.01, 0.05],
            },
            verified_bc_resume=True,
            current_std=(0.01, 0.05),
        )
    except ValueError as exc:
        assert "exactly throttle and steering" in str(exc)
    else:
        raise AssertionError("malformed saved exploration bounds were accepted")


def test_smoke_ppo_profile_is_bounded_and_ordinary_defaults_stay_unchanged():
    assert _BC_SMOKE_PPO == {
        "n_envs": 1,
        "n_steps": 2048,
        "learning_rate": 1e-5,
        "n_epochs": 4,
        "gamma": 0.9995,
        "gae_lambda": 0.95,
        "target_kl": 0.01,
        "batch_size": 64,
        "clip_range": 0.1,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "max_transitions": 8192,
        "max_wall_seconds": 600,
        "save_every_transitions": 2048,
    }
    assert _BC_SMOKE_FIXED_PPO["clip_range"] == 0.1
    assert _BC_SMOKE_FIXED_PPO["ent_coef"] == 0.0
    env = DummyVecEnv([_ObservationEnv])
    try:
        model = make_model(
            env, device="cpu", seed=0, tensorboard_log=None,
            learning_rate=1e-5, n_epochs=4, n_steps=2048,
            gamma=0.9995, gae_lambda=0.95, target_kl=0.01,
            policy_architecture="lidar_camera_cnn", **_BC_SMOKE_FIXED_PPO,
        )
        assert model.batch_size == 64
        assert model.clip_range(1.0) == 0.1
        assert model.ent_coef == 0.0
        assert model.vf_coef == 0.5
        assert model.max_grad_norm == 0.5

        ordinary = make_model(
            env, device="cpu", seed=0, tensorboard_log=None,
            n_steps=2048, policy_architecture="lidar_camera_cnn",
        )
        assert ordinary.clip_range(1.0) == 0.2
        assert ordinary.ent_coef == 0.01
    finally:
        env.close()


def test_operator_stop_is_immediate_and_preserves_latest_post_update_checkpoint(tmp_path):
    env = DummyVecEnv([_ObservationEnv])
    try:
        model = make_model(
            env, device="cpu", seed=0, tensorboard_log=None, n_steps=2048,
            policy_architecture="lidar_camera_cnn",
        )
        model.num_timesteps = 2048
        model._n_updates = 1
        model.set_logger(configure(str(tmp_path / "sb3-stop-logs"), ["stdout"]))
        audit = PostOptimizerAuditCallback(tmp_path, save_every_transitions=2048)
        audit.init_callback(model)
        audit.on_training_start({}, {})
        audit.last_update = 0
        audit.last_saved_steps = 0
        model.logger.name_to_value.update({"train/loss": 0.5})
        audit.on_rollout_start()
        saved = next((tmp_path / "checkpoints").glob("post_optimizer_update_*.zip"))
        saved_hash = __import__("hashlib").sha256(saved.read_bytes()).hexdigest()

        request = Event()
        stop = OperatorStopCallback(request)
        stop.init_callback(model)
        stop.on_training_start({}, {})
        updates_at_request = int(model._n_updates)
        request.set()
        assert stop.on_step() is False
        assert stop.stop_reason == "operator_stop"
        assert int(model._n_updates) == updates_at_request
        assert __import__("hashlib").sha256(saved.read_bytes()).hexdigest() == saved_hash
    finally:
        env.close()


def test_campaign_checkpoint_and_probe_cadence_use_global_step_buckets(tmp_path):
    assert _step_cadence_origin("bc_campaign", "steps", 6144) == 0
    assert _step_cadence_origin("standard", "steps", 6144) == 6144
    assert _step_cadence_origin("bc_campaign", "episodes", 6144) == 6144

    schedule = AutomaticEvaluationSchedule("steps", 50_000, starting_steps=0)
    assert schedule.observe(49_152, []) == (False, 49_152, 0)
    assert schedule.observe(50_176, []) == (True, 50_176, 0)
    assert schedule.observe(99_328, []) == (False, 99_328, 0)
    assert schedule.observe(100_352, []) == (True, 100_352, 0)

    env = DummyVecEnv([_ObservationEnv])
    try:
        model = make_model(
            env, device="cpu", seed=0, tensorboard_log=None, n_steps=2048,
            policy_architecture="lidar_camera_cnn",
        )
        model.num_timesteps = 6144
        model._n_updates = 4
        model.set_logger(configure(str(tmp_path / "global-cadence-logs"), ["stdout"]))
        callback = PostOptimizerAuditCallback(
            tmp_path, save_every_transitions=10_000, global_step_cadence=True
        )
        callback.init_callback(model)
        callback.on_training_start({}, {})
        for step, epochs in ((8192, 5), (10240, 6), (18432, 7), (20480, 8)):
            model.num_timesteps = step
            model._n_updates = epochs
            model.logger.name_to_value.update({
                "train/loss": 0.1,
                "train/policy_gradient_loss": 0.01,
                "train/value_loss": 0.1,
                "train/entropy_loss": -1.0,
                "train/approx_kl": 0.001,
                "train/clip_fraction": 0.0,
            })
            callback.on_rollout_start()
            saved_steps = sorted(
                int(path.stem.split("_steps_")[1])
                for path in (tmp_path / "checkpoints").glob("post_optimizer_update_*.zip")
            ) if (tmp_path / "checkpoints").exists() else []
            expected_steps = (
                [] if step < 10240 else
                [10240] if step < 20480 else
                [10240, 20480]
            )
            assert saved_steps == expected_steps
    finally:
        env.close()


def test_camera_resume_exploration_audit_and_reload(tmp_path):
    env = DummyVecEnv([_ObservationEnv])
    try:
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
        )
        with torch.no_grad():
            source.policy.log_std.copy_(torch.log(torch.tensor([0.01, 0.05])))
        source_path = tmp_path / "bc_parent.zip"
        source.save(str(source_path.with_suffix("")))

        model = resume_ppo_checkpoint(
            source_path,
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
        )
        assert model.n_steps == 2048
        assert model.batch_size == 64
        assert model.n_epochs == 4
        assert model.target_kl == 0.01
        assert model.policy.normalize_images is False

        schedule = OfficialExplorationSchedule(
            model,
            tmp_path,
            std_min=(0.01, 0.05),
            std_max=(0.01, 0.05),
        )
        schedule.on_rollout_start()
        assert np.allclose(model.policy.log_std.detach().exp().cpu().numpy(), [0.01, 0.05])
        (tmp_path / "evaluation_status.json").write_text(json.dumps({
            "state": "incomplete",
            "evaluation_id": "nonranking-probe",
            "snapshot_timesteps": 2048,
            "validation_mode": "training_probe",
            "latest_result": {"valid": False},
        }), encoding="utf-8")
        assert schedule.observe(2048, force=True) is None
        assert schedule.stale_evaluations == 0

        rng = np.random.default_rng(4)
        observations = {
            "lidar": rng.uniform(0, 1, (1, 1081)).astype(np.float32),
            "camera": rng.integers(0, 256, (1, 90, 160, 3), dtype=np.uint8),
            "state": rng.uniform(-1, 1, (1, 9)).astype(np.float32),
        }
        deterministic_before = model.predict(observations, deterministic=True)[0]
        stochastic_a = model.predict(observations, deterministic=False)[0]
        stochastic_b = model.predict(observations, deterministic=False)[0]
        assert np.isfinite(deterministic_before).all()
        assert not np.allclose(stochastic_a, stochastic_b)

        # Exercise callback lifecycle against a constructed camera PPO model;
        # manually mark one completed update so the test does not train PPO.
        model.num_timesteps = 0
        model.set_logger(configure(str(tmp_path / "sb3-logs"), ["stdout"]))
        audit = PostOptimizerAuditCallback(tmp_path, save_every_transitions=2048)
        audit.init_callback(model)
        audit.on_training_start({}, {})
        model.num_timesteps = 2048
        model._n_updates = 1
        parameter = next(model.policy.parameters())
        parameter.grad = torch.zeros_like(parameter)
        model.logger.name_to_value.update({
            "train/loss": 1.0,
            "train/policy_gradient_loss": 0.1,
            "train/value_loss": 0.2,
            "train/entropy_loss": -0.3,
            "train/approx_kl": 0.001,
            "train/clip_fraction": 0.0,
        })
        audit.on_rollout_start()
        audit_record = json.loads(audit.audit_path.read_text(encoding="utf-8").splitlines()[-1])
        assert audit_record["optimizer_updates"] == 1
        assert audit_record["metrics_finite"] is True
        assert audit_record["gradients_finite"] is True
        assert audit_record["parameters_finite"] is True
        checkpoints = list((tmp_path / "checkpoints").glob("post_optimizer_update_*.zip"))
        assert len(checkpoints) == 1

        reloaded = PPO.load(str(checkpoints[0]), env=env, device="cpu")
        deterministic_after = reloaded.predict(observations, deterministic=True)[0]
        assert np.allclose(deterministic_after, deterministic_before, atol=1e-7, rtol=0)
        assert math.isclose(float(reloaded.target_kl), 0.01)
    finally:
        env.close()
