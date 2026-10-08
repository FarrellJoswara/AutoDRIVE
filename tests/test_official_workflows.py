"""Focused checks for official orchestration; FakeDocker is not simulator proof."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.test_official_run_manager import FakeDocker
from src.layer4.hub.official_run_manager import OfficialRunManager
from src.layer4.hub.official_models import checkpoint_settings, list_official_models
from src.layer4.hub.official_replay_job import OfficialReplayJob
from src.layer4.official_settings import OfficialTrainSettings


@pytest.fixture
def manager(tmp_path):
    backend = FakeDocker(tmp_path)
    result = OfficialRunManager(backend=backend, registry_path=tmp_path / "registry.json",
                               runs_root=tmp_path / "logs" / "rl", start_monitor=False)
    yield result
    result.close()


def test_parallel_pairs_have_separate_ros_domains_and_one_trainer(manager):
    run = manager.create_run({"n_envs": 3})
    containers = list(manager.backend.containers.values())
    assert len(containers) == 6
    apis = {item["Labels"]["aicar.role"]: dict(value.split("=", 1) for value in item["kwargs"].get("env", []))
            for item in containers if item["Labels"]["aicar.role"].startswith("api")}
    assert apis["api"]["AICAR_TRAIN_N_ENVS"] == "3"
    assert apis["api"]["AICAR_MODE"] == "train"
    assert apis["api-1"]["ROS_DOMAIN_ID"] == "1"
    assert apis["api-2"]["ROS_DOMAIN_ID"] == "2"
    assert all(apis[role]["AICAR_MODE"] == "bridge" for role in ("api-1", "api-2"))
    manager.stop_run(run["run_id"])
    assert not manager.backend.containers
    assert not manager.backend.networks


def test_cuda_run_uses_gpu_learner_image_and_keeps_workers_cpu(manager):
    with patch("src.layer4.hub.official_run_manager.GPU_API_IMAGE", "aicar-test:cuda"):
        run = manager.create_run({"n_envs": 2, "device": "cuda"})
    primary = next(item for item in manager.backend.containers.values()
                   if item["Labels"]["aicar.role"] == "api")
    bridge = next(item for item in manager.backend.containers.values()
                  if item["Labels"]["aicar.role"] == "api-1")
    assert primary["kwargs"]["image"] == "aicar-test:cuda"
    assert primary["kwargs"]["gpu"] is True
    assert "AICAR_PPO_DEVICE=cuda" in primary["kwargs"]["env"]
    assert bridge["kwargs"].get("gpu", False) is False
    manager.stop_run(run["run_id"])


def test_worker_failure_cleans_entire_run_but_leaves_other_run(manager):
    first = manager.create_run({"n_envs": 2})
    second = manager.create_run({})
    for item in manager.backend.containers.values():
        if item["Labels"]["aicar.run_id"] == first["run_id"] and item["Labels"]["aicar.role"] == "api-1":
            item["State"] = "exited"
    manager.reconcile(first["run_id"])
    assert manager.get_run(first["run_id"])["state"] == "failed"
    assert manager.get_run(second["run_id"])["state"] == "running"
    assert all(item["Labels"]["aicar.run_id"] == second["run_id"] for item in manager.backend.containers.values())


def test_transient_network_inventory_failure_does_not_destroy_run(manager):
    run = manager.create_run({"n_envs": 2})
    with patch.object(manager.backend, "list_official_run_networks", side_effect=RuntimeError("timeout")):
        manager.reconcile()
    assert manager.get_run(run["run_id"])["state"] == "running"
    assert len(manager.backend.containers) == 4


def test_parallel_run_recovered_from_labels_without_registry(manager, tmp_path):
    run = manager.create_run({"n_envs": 2})
    recovered = OfficialRunManager(backend=manager.backend, registry_path=tmp_path / "new.json",
                                  runs_root=manager.runs_root, start_monitor=False)
    try:
        assert recovered.get_run(run["run_id"])["config"]["n_envs"] == 2
        recovered.stop_run(run["run_id"])
        assert not manager.backend.containers
    finally:
        recovered.close()


def official_checkpoint(root, *, nested=True):
    folder = root / "logs" / "rl" / "official-example"
    folder.mkdir(parents=True)
    settings = OfficialTrainSettings()
    config = {"runtime": "official IROS 2026 API and simulator images", "observation_profile": "official_sensors_history",
              "actions": {key: getattr(settings, key) for key in ("throttle_mode", "negative_throttle_mode", "steering_mode", "steering_action_scale", "straight_throttle_gain", "straight_throttle_steering_threshold")}}
    config["actions"]["throttle_mode"] = "forward_only"
    (folder / "config.json").write_text(json.dumps(config))
    target = folder / "checkpoints" if nested else folder
    target.mkdir(exist_ok=True)
    checkpoint = target / "official_ppo_128_steps.zip"
    checkpoint.write_bytes(b"fixture: never unpickled")
    return checkpoint


def test_catalog_finds_official_checkpoint_subdirectory_and_filters_custom(tmp_path):
    path = official_checkpoint(tmp_path)
    custom = tmp_path / "logs/rl/custom"
    custom.mkdir(); (custom / "policy.zip").touch()
    models = list_official_models(tmp_path / "logs/rl")
    assert [item["id"] for item in models] == ["official-example/checkpoints/official_ppo_128_steps.zip"]
    assert checkpoint_settings(path)["throttle_mode"] == "forward_only"


def test_replay_uses_saved_profile_and_controls_in_fresh_official_pair(manager, tmp_path):
    path = official_checkpoint(tmp_path)
    replay = OfficialReplayJob(manager)
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        status = replay.start(model_path=path, trace_steps=500)
    run = manager.get_run(status["run_id"])
    assert run["kind"] == "evaluation"
    api = next(item for item in manager.backend.containers.values() if item["Labels"]["aicar.role"] == "api")
    env = dict(value.split("=", 1) for value in api["kwargs"]["env"])
    assert env["AICAR_MODE"] == "evaluate"
    assert env["AICAR_OBSERVATION_PROFILE"] == "official_sensors_history"
    assert env["AICAR_THROTTLE_MODE"] == "forward_only"
    assert env["AICAR_EVALUATION_TRACE_STEPS"] == "500"
    Path(run["output_dir"], "evaluation.json").write_text(json.dumps({"completed_attempts": 0, "runs": []}))
    manager.reconcile()
    assert replay.status()["evaluation"]["completed_attempts"] == 0
    assert replay.status()["state"] == "exited"
    assert not manager.backend.containers
    recovered = OfficialReplayJob(manager)
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        restarted = recovered.reset()
    assert restarted["run_id"] != status["run_id"]
    recovered.stop()


def test_replay_trace_steps_are_bounded(manager, tmp_path):
    path = official_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="between 0 and 500"):
        OfficialReplayJob(manager).start(model_path=path, trace_steps=501)
    assert not manager.backend.containers


def test_replay_rejects_checkpoint_without_provenance(manager, tmp_path):
    path = tmp_path / "unknown.zip"; path.touch()
    with pytest.raises(ValueError, match="config.json"):
        OfficialReplayJob(manager).start(model_path=path)
    assert not manager.backend.containers


def test_official_settings_bound_worker_count():
    assert OfficialTrainSettings(n_envs=2).to_container_env(hub_url="http://hub", run_id="r")["AICAR_TRAIN_N_ENVS"] == "2"
    with pytest.raises(ValueError):
        OfficialTrainSettings(n_envs=9)
    assert OfficialTrainSettings(device="cuda").to_container_env(hub_url="http://hub", run_id="r")["AICAR_PPO_DEVICE"] == "cuda"


def test_completed_progress_uses_saved_count_not_sampled_telemetry(manager):
    run = manager.create_run({"total_timesteps": 256})
    manager._runs[run["run_id"]]["latest_metrics"] = {"step": 200}
    Path(run["output_dir"], "stop_reason.json").write_text(json.dumps({"reason": "timestep_limit", "num_timesteps": 256}))
    manager.reconcile()
    result = manager.get_run(run["run_id"])
    assert result["state"] == "completed"
    assert result["latest_metrics"]["step"] == 256
    assert result["map_id"] == "none"
