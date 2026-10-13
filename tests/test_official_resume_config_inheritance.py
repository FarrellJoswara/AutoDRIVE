import json

from src.layer4.hub import official_run_manager as manager
from src.layer4.official_settings import OfficialTrainSettings


def _parent(tmp_path, monkeypatch):
    root = tmp_path
    logs = root / "logs"
    logs.mkdir()
    checkpoint = logs / "parent.zip"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_name("parent.lineage.json").write_text("{}", encoding="utf-8")
    effective = logs / "config.json"
    effective.write_text(json.dumps({
        "simulator_backend": "windows_external",
        "device": "cuda",
        "observation_profile": "official_sensors_camera",
        "actions": {
            "throttle_mode": "bidirectional", "negative_throttle_mode": "allow",
            "steering_mode": "normal", "steering_action_scale": 1.0,
            "straight_throttle_gain": 1.0, "straight_throttle_steering_threshold": 0.15,
        },
        "episode": {"training_watchdog_s": 600.0, "frontier_stagnation_s": 5.0},
        "ppo_exploration": {
            "profile": "bc_smoke", "starting_action_std": [0.01, 0.05],
            "minimum_action_std_by_action": [0.01, 0.05],
            "maximum_action_std_by_action": [0.01, 0.05],
        },
        "ppo": {
            "n_envs": 1, "n_steps": 2048, "learning_rate": 1e-5,
            "n_epochs": 4, "gamma": 0.9995, "gae_lambda": 0.95,
            "target_kl": 0.01, "post_optimizer_save_every_transitions": 2048,
            "stop_on_plateau": False, "max_duration_hours": 1 / 6,
        },
        "automatic_evaluation": {"enabled": False, "trigger": "steps", "interval": 2048, "mode": "competitive"},
        "behavior_cloning": {"enabled": False},
    }), encoding="utf-8")
    monkeypatch.setattr(manager, "ROOT", root)
    monkeypatch.setattr(
        manager, "verify_ppo_lineage_checkpoint",
        lambda path: {"effective_config": str(effective)},
    )
    return checkpoint


def test_resume_inherits_authenticated_effective_settings_when_omitted(tmp_path, monkeypatch):
    checkpoint = _parent(tmp_path, monkeypatch)
    config = manager.normalize_official_config({
        "resume": str(checkpoint), "total_timesteps": 8192,
    }, explicit_fields={"resume", "total_timesteps"})

    assert config["learning_rate"] == 1e-5
    assert config["target_kl"] == 0.01
    assert config["exploration_profile"] == "bc_smoke"
    assert config["exploration_initial_std_throttle"] == 0.01
    assert config["exploration_max_std_steering"] == 0.05
    assert config["automatic_evaluation_enabled"] is False
    assert config["device"] == "cuda"
    assert config["resume"] == "/runs/parent.zip"


def test_resume_explicit_overrides_win_and_new_run_default_stays_unchanged(tmp_path, monkeypatch):
    checkpoint = _parent(tmp_path, monkeypatch)
    config = manager.normalize_official_config({
        "resume": str(checkpoint), "learning_rate": 2e-5,
        "target_kl": None, "total_timesteps": 4096,
    }, explicit_fields={"resume", "learning_rate", "target_kl", "total_timesteps"})
    assert config["learning_rate"] == 2e-5
    assert config["target_kl"] is None

    ordinary = OfficialTrainSettings.model_validate({"behavior_cloning_enabled": False})
    assert ordinary.learning_rate == 1e-4


def test_null_exploration_fields_mean_inherit_saved_profile(tmp_path, monkeypatch):
    checkpoint = _parent(tmp_path, monkeypatch)
    config = manager.normalize_official_config({
        "resume": str(checkpoint), "exploration_profile": None,
        "exploration_initial_std_throttle": None,
    }, explicit_fields={"resume", "exploration_profile", "exploration_initial_std_throttle"})
    assert config["exploration_profile"] == "bc_smoke"
    assert config["exploration_initial_std_throttle"] == 0.01


def test_campaign_experiment_metadata_survives_settings_normalization():
    config = manager.normalize_official_config({
        "campaign_experiment_id": "time_cost_3",
        "campaign_experiment_phase": "pace_experiments",
        "campaign_experiment_config_sha256": "a" * 64,
        "time_cost_per_simulated_second": 3.0,
    })

    assert config["campaign_experiment_id"] == "time_cost_3"
    assert config["campaign_experiment_phase"] == "pace_experiments"
    assert config["campaign_experiment_config_sha256"] == "a" * 64
    assert OfficialTrainSettings.model_validate(config).model_dump(mode="json")[
        "campaign_experiment_config_sha256"
    ] == "a" * 64


def test_windows_startup_timeout_is_normalized_before_campaign_hash():
    windows = manager.normalize_official_config({
        "simulator_backend": "windows_external", "timeout_s": 30.0,
    })
    docker = manager.normalize_official_config({
        "simulator_backend": "docker", "timeout_s": 30.0,
    })

    assert windows["timeout_s"] == 180.0
    assert docker["timeout_s"] == 30.0
