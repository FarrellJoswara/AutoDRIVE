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
    assert run["simulator_image"].endswith(":2026-iros-practice")
    containers = list(manager.backend.containers.values())
    assert len(containers) == 6
    apis = {item["Labels"]["aicar.role"]: dict(value.split("=", 1) for value in item["kwargs"].get("env", []))
            for item in containers if item["Labels"]["aicar.role"].startswith("api")}
    assert apis["api"]["AICAR_TRAIN_N_ENVS"] == "3"
    assert apis["api"]["AICAR_MODE"] == "train"
    assert apis["api"]["AICAR_OFFICIAL_SIMULATOR_IMAGE"].endswith(":2026-iros-practice")
    assert apis["api-1"]["ROS_DOMAIN_ID"] == "1"
    assert apis["api-2"]["ROS_DOMAIN_ID"] == "2"
    assert all(apis[role]["AICAR_MODE"] == "bridge" for role in ("api-1", "api-2"))
    simulator_images = {
        item["kwargs"]["image"] for item in containers
        if item["Labels"]["aicar.role"].startswith("sim")
    }
    assert simulator_images == {"autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice"}
    simulators = [item for item in containers if item["Labels"]["aicar.role"].startswith("sim")]
    assert all(item["kwargs"]["gpu"] is True for item in simulators)
    assert all("xvfb-run" in item["kwargs"]["command"][0] for item in simulators)
    assert all("-nographics" not in item["kwargs"]["command"][0] for item in simulators)
    assert {item["kwargs"]["command"][0].split("-ip ", 1)[1].split()[0]
            for item in simulators} == {"api", "api-1", "api-2"}
    manager.stop_run(run["run_id"])
    assert not manager.backend.containers
    assert not manager.backend.networks


def test_windows_graphics_backend_publishes_one_local_bridge_per_env(manager):
    run = manager.create_run({"simulator_backend": "windows_external", "n_envs": 2})
    containers = list(manager.backend.containers.values())
    assert {item["Labels"]["aicar.role"] for item in containers} == {"api", "api-1"}
    assert run["simulator_backend"] == "windows_external"
    assert run["simulator_image"] == "official AutoDRIVE IROS 2026 Windows simulator"
    assert len(run["simulator_connections"]) == 2
    ports = [item["port"] for item in run["simulator_connections"]]
    assert ports[1] == ports[0] + 1
    apis = {item["Labels"]["aicar.role"]: item for item in containers}
    assert apis["api"]["kwargs"]["published_ports"] == {"4568": ports[0]}
    assert apis["api-1"]["kwargs"]["published_ports"] == {"4568": ports[1]}
    assert "AICAR_MODE=train" in apis["api"]["kwargs"]["env"]
    assert "AICAR_MODE=bridge" in apis["api-1"]["kwargs"]["env"]
    manager.stop_run(run["run_id"])
    assert not manager.backend.containers
    assert not manager.backend.networks


def test_native_backend_starts_windows_fleet_only_after_all_bridge_pairs_exist(manager):
    calls = []
    manager.backend.simulator_host = "172.22.16.1"
    manager.backend.start_simulators = lambda **kwargs: calls.append(kwargs)
    run = manager.create_run({"simulator_backend": "windows_external", "n_envs": 4})

    assert [item["host"] for item in run["simulator_connections"]] == ["172.22.16.1"] * 4
    assert len(calls) == 1
    assert len(calls[0]["connections"]) == 4
    assert calls[0]["run_id"] == run["run_id"]
    assert len(manager.backend.containers) == 4
    manager.stop_run(run["run_id"])


def test_native_backend_maps_container_log_paths_to_workspace_logs(tmp_path):
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    value = NativeOfficialBackend._local_logs_path("/runs/rl/official_run_abcd/checkpoint.zip")
    assert value.replace("\\", "/").endswith("logs/rl/official_run_abcd/checkpoint.zip")


def test_native_backend_translates_run_settings_to_native_trainer_arguments():
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    command = NativeOfficialBackend._training_command({
        "AICAR_TRAIN_OUT": "/tmp/run-output",
        "AICAR_TRAIN_N_ENVS": "4",
        "AICAR_TRAIN_TIMESTEPS": "256",
        "AICAR_PPO_DEVICE": "cuda",
        "AICAR_PPO_N_STEPS": "64",
        "AICAR_PPO_N_EPOCHS": "1",
        "AICAR_OBSERVATION_PROFILE": "official_sensors_camera",
        "AICAR_TRAIN_STOP_ON_PLATEAU": "0",
    })
    args = dict(zip(command[3::2], command[4::2]))
    assert args["--out"] == "/tmp/run-output"
    assert args["--n-envs"] == "4"
    assert args["--total-timesteps"] == "256"
    assert args["--device"] == "cuda"
    assert args["--n-steps"] == "64"
    assert args["--n-epochs"] == "1"
    assert args["--observation-profile"] == "official_sensors_camera"
    assert args["--stop-on-plateau"] == "0"


def test_official_relay_accepts_bridge_event_before_socketio_namespace_connect():
    import asyncio
    import engineio
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from src.layer1.official_bridge_relay import OfficialBridgeRelay

    relay = OfficialBridgeRelay()

    async def exercise_unity_event_order():
        app = web.Application()
        relay.sio.attach(app, socketio_path="socket.io")
        server = TestServer(app)
        await server.start_server()
        client = engineio.AsyncClient()
        try:
            await client.connect(
                str(server.make_url("/")).rstrip("/"),
                engineio_path="socket.io",
                transports=["websocket"],
            )
            # AutoDRIVE Unity sends an Engine.IO MESSAGE with a Socket.IO
            # Bridge event before the formal Socket.IO namespace connect.
            await client.send('2["Bridge",{}]')
            deadline = asyncio.get_running_loop().time() + 2.0
            while relay._incomplete_frames == 0 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert relay._incomplete_frames == 1
            assert len(relay._connected_simulators) == 1
        finally:
            if getattr(client, "state", None) == "connected":
                await client.disconnect()
            await server.close()

    asyncio.run(exercise_unity_event_order())


def test_legacy_docker_registry_paths_resolve_to_persistent_workspace_logs(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"runs": [{
        "run_id": "ab12cd34ef56", "state": "completed",
        "output_dir": "/app/logs/rl/official_run_ab12cd34ef56",
        "log_path": "/app/logs/rl/official_run_ab12cd34ef56/hub_train.log",
    }]}), encoding="utf-8")
    manager = OfficialRunManager(
        backend=FakeDocker(tmp_path), registry_path=registry,
        runs_root=tmp_path / "logs" / "rl", start_monitor=False,
    )
    try:
        run = manager.get_run("ab12cd34ef56")
        assert run["output_dir"] == str(tmp_path / "logs" / "rl" / "official_run_ab12cd34ef56")
        assert run["log_path"] == str(tmp_path / "logs" / "rl" / "official_run_ab12cd34ef56" / "hub_train.log")
    finally:
        manager.close()


def test_external_simulator_port_blocks_are_valid_and_unique(manager):
    first = manager.create_run({"simulator_backend": "windows_external", "n_envs": 8})
    second = manager.create_run({"simulator_backend": "windows_external", "n_envs": 8})
    first_ports = {item["port"] for item in first["simulator_connections"]}
    second_ports = {item["port"] for item in second["simulator_connections"]}
    assert len(first_ports) == 8
    assert len(second_ports) == 8
    assert all(49152 <= port <= 65535 for port in first_ports | second_ports)
    assert first_ports.isdisjoint(second_ports)
    manager.stop_run(first["run_id"])
    manager.stop_run(second["run_id"])


def test_external_run_creation_error_is_reported_and_cleaned_up(manager, monkeypatch):
    def fail_create(**kwargs):
        raise RuntimeError("simulated Docker port allocation failure")

    monkeypatch.setattr(manager.backend, "create_official_run_container", fail_create)
    with pytest.raises(RuntimeError, match="simulated Docker port allocation failure"):
        manager.create_run({"simulator_backend": "windows_external", "n_envs": 1})
    failed = manager.list_runs()["runs"][0]
    assert failed["state"] == "failed"
    assert failed["error"] == "simulated Docker port allocation failure"
    assert failed["cleanup_error"] is None
    assert not manager.backend.networks


def test_docker_api_binds_external_bridge_ports_to_loopback_only(monkeypatch):
    from src.layer4.hub import docker_control

    calls = []
    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"Id": "container-id"} if method == "POST" and "/containers/create" in path else None

    monkeypatch.setattr(docker_control, "_request", request)
    docker_control.create_official_run_container(
        run_id="0123456789ab", role="api", image="aicar-iros2026",
        network_name="aicar-run-0123456789ab", published_ports={"4568": 48123},
    )
    body = calls[0][2]["body"]
    assert body["ExposedPorts"] == {"4568/tcp": {}}
    assert body["HostConfig"]["PortBindings"] == {
        "4568/tcp": [{"HostIp": "127.0.0.1", "HostPort": "48123"}]
    }


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
    assert run["simulator_image"].endswith(":2026-iros-compete")
    api = next(item for item in manager.backend.containers.values() if item["Labels"]["aicar.role"] == "api")
    env = dict(value.split("=", 1) for value in api["kwargs"]["env"])
    assert env["AICAR_MODE"] == "evaluate"
    assert env["AICAR_OBSERVATION_PROFILE"] == "official_sensors_history"
    assert env["AICAR_THROTTLE_MODE"] == "forward_only"
    assert env["AICAR_EVALUATION_TRACE_STEPS"] == "500"
    sim = next(item for item in manager.backend.containers.values()
               if item["Labels"]["aicar.role"] == "sim")
    assert sim["kwargs"]["image"] == "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete"
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
    settings = OfficialTrainSettings(n_envs=2)
    environment = settings.to_container_env(hub_url="http://hub", run_id="r")
    assert environment["AICAR_TRAIN_N_ENVS"] == "2"
    assert settings.collision_reward_percent == 50.0
    assert environment["AICAR_TRAIN_COLLISION_REWARD_PERCENT"] == "50.0"
    assert settings.backward_speed_penalty_scale == 10.0
    assert environment["AICAR_TRAIN_BACKWARD_SPEED_PENALTY_SCALE"] == "10.0"
    assert environment["AICAR_TRAIN_FAILURE_REWARD_PERCENT"] == "50.0"
    with pytest.raises(ValueError):
        OfficialTrainSettings(n_envs=9)
    assert OfficialTrainSettings(device="cuda").to_container_env(hub_url="http://hub", run_id="r")["AICAR_PPO_DEVICE"] == "cuda"


def test_official_checkpoint_records_external_simulator_provenance(monkeypatch):
    from src.layer3.official_train import _official_simulator_image

    monkeypatch.setenv(
        "AICAR_OFFICIAL_SIMULATOR_IMAGE",
        "AutoDRIVE IROS 2026 compete Windows binary",
    )
    assert _official_simulator_image() == "AutoDRIVE IROS 2026 compete Windows binary"
    monkeypatch.delenv("AICAR_OFFICIAL_SIMULATOR_IMAGE")
    assert _official_simulator_image().endswith(":2026-iros-practice")


def test_camera_endpoint_only_serves_camera_runs(tmp_path, monkeypatch):
    from src.layer4.hub import app as hub_app

    output = tmp_path / "logs" / "rl" / "official_run_cam"
    camera_dir = output / "camera"
    camera_dir.mkdir(parents=True)
    image_path = camera_dir / "env-0.jpg"
    image_path.write_bytes(b"jpeg fixture")

    class Runs:
        profile = "official_sensors_camera"

        def get_run(self, run_id):
            return {"run_id": run_id, "config": {"observation_profile": self.profile}, "output_dir": str(output)}

    monkeypatch.setattr(hub_app, "official_runs", Runs())
    monkeypatch.setattr(hub_app, "ROOT", tmp_path)
    response = __import__("asyncio").run(hub_app.get_official_run_camera("cam", 0))
    assert Path(response.path) == image_path
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"] == "no-store, max-age=0"

    hub_app.official_runs.profile = "official_sensors"
    with pytest.raises(hub_app.HTTPException) as error:
        __import__("asyncio").run(hub_app.get_official_run_camera("cam", 1))
    assert error.value.status_code == 404


def test_completed_progress_uses_saved_count_not_sampled_telemetry(manager):
    run = manager.create_run({"total_timesteps": 256})
    manager._runs[run["run_id"]]["latest_metrics"] = {"step": 200}
    Path(run["output_dir"], "stop_reason.json").write_text(json.dumps({"reason": "timestep_limit", "num_timesteps": 256}))
    manager.reconcile()
    result = manager.get_run(run["run_id"])
    assert result["state"] == "completed"
    assert result["latest_metrics"]["step"] == 256
    assert result["map_id"] == "none"
