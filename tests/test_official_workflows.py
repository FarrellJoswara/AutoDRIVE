"""Focused checks for official orchestration; FakeDocker is not simulator proof."""
import json
import time
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.test_official_run_manager import FakeDocker
from src.layer4.hub.official_run_manager import OfficialRunManager
from src.layer4.hub.official_models import checkpoint_settings, list_official_models
from src.layer4.hub.official_replay_job import OfficialReplayJob
from src.layer4.hub.official_evaluation_service import OfficialEvaluationService, _attempt_collision_count
from src.layer4.hub.native_official_backend import _windows_simulator_mode_args
from src.layer4.official_settings import OfficialTrainSettings, load_official_train_settings
from src.layer3.official_train import train_official


@pytest.fixture
def manager(tmp_path):
    backend = FakeDocker(tmp_path)
    result = OfficialRunManager(backend=backend, registry_path=tmp_path / "registry.json",
                               runs_root=tmp_path / "logs" / "rl", start_monitor=False)
    yield result
    result.close()


def test_official_training_requires_camera_fused_observations(tmp_path):
    with pytest.raises(ValueError, match="requires camera, LiDAR, and state"):
        train_official(
            out_dir=tmp_path / "run", total_timesteps=1,
            observation_profile="official_sensors",
        )


def test_bc_only_trainer_rejects_disabled_cloning_and_resume(tmp_path):
    with pytest.raises(ValueError, match="requires a new run with behavior cloning enabled"):
        train_official(
            out_dir=tmp_path / "disabled", total_timesteps=0,
            behavior_cloning_only=True, behavior_cloning_enabled=False,
        )
    with pytest.raises(ValueError, match="requires a new run with behavior cloning enabled"):
        train_official(
            out_dir=tmp_path / "resume", total_timesteps=0,
            behavior_cloning_only=True, behavior_cloning_enabled=True,
            resume=tmp_path / "checkpoint.zip",
        )


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


def test_windows_simulator_offscreen_flag_keeps_default_gui_mode_unchanged():
    assert _windows_simulator_mode_args({"simulator_display_mode": "offscreen"}) == ["-BatchMode"]
    assert _windows_simulator_mode_args({"simulator_display_mode": "gui"}) == []
    assert _windows_simulator_mode_args({}) == []


def test_native_backend_starts_windows_fleet_only_after_all_bridge_pairs_exist(manager):
    calls = []
    manager.backend.simulator_host = "172.22.16.1"
    manager.backend.start_simulators = lambda **kwargs: calls.append(kwargs)
    run = manager.create_run({
        "simulator_backend": "windows_external", "simulator_display_mode": "offscreen", "n_envs": 4,
    })

    assert [item["host"] for item in run["simulator_connections"]] == ["172.22.16.1"] * 4
    assert len(calls) == 1
    assert len(calls[0]["connections"]) == 4
    assert calls[0]["run_id"] == run["run_id"]
    assert calls[0]["config"]["simulator_display_mode"] == "offscreen"
    assert len(manager.backend.containers) == 4
    manager.stop_run(run["run_id"])


def test_automatic_evaluation_quiesces_and_restores_active_windows_training(manager):
    run = manager.create_run({"simulator_backend": "windows_external", "n_envs": 2})
    calls = []
    manager.backend.pause_training_for_evaluation = lambda **kwargs: calls.append(("pause", kwargs))
    manager.backend.resume_training_after_evaluation = lambda **kwargs: calls.append(("resume", kwargs))

    paused = manager.pause_active_training_for_evaluation()
    assert paused == [run["run_id"]]
    assert calls[0][0] == "pause"
    assert len(calls[0][1]["connections"]) == 2

    manager.resume_training_after_evaluation(paused)
    assert calls[1][0] == "resume"
    assert calls[1][1]["run_id"] == run["run_id"]
    assert run["run_id"] not in manager._evaluation_paused
    manager.stop_run(run["run_id"])


def test_evaluation_worker_brackets_manual_evaluation_with_training_pause(manager, tmp_path):
    checkpoint = official_checkpoint(tmp_path)
    run = manager.create_run({"simulator_backend": "windows_external", "n_envs": 2})
    calls = []
    manager.backend.pause_training_for_evaluation = lambda **kwargs: calls.append("pause")
    manager.backend.resume_training_after_evaluation = lambda **kwargs: calls.append("resume")
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "official-evaluations.json",
        champion_dir=tmp_path / "best", model_root=tmp_path / "logs" / "rl",
    )
    service._run_batch = lambda job: job.update(state="incomplete")
    try:
        service.submit(checkpoint, source="manual")
        deadline = time.monotonic() + 3
        while service.status()["state"] not in {"incomplete", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.status()["state"] == "incomplete"
        assert calls == ["pause", "resume"]
        assert run["run_id"] not in manager._evaluation_paused
    finally:
        service.close()
        manager.stop_run(run["run_id"])


def test_native_training_pause_stops_sims_before_eval_and_resumes_after_fresh_camera_frames(tmp_path):
    import signal
    import threading
    from types import SimpleNamespace
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    backend = object.__new__(NativeOfficialBackend)
    backend._lock = threading.RLock()
    run_id = "a1b2c3d4e5f6"
    run_dir = tmp_path / f"official_run_{run_id}"
    run_dir.mkdir()
    (run_dir / "windows_simulator_pids.txt").write_text("", encoding="ascii")
    process = SimpleNamespace(pid=101, poll=lambda: None)
    group = {
        "Labels": {"aicar.run_id": run_id, "aicar.role": "api"},
        "records": [{"role": "trainer", "process": process}],
    }
    backend._groups = {"native-group": group}
    connections = [{"environment": 0, "port": 53001}]
    backend._run_dir_for_id = lambda _run_id: run_dir
    backend._read_simulator_health = lambda _port: {
        "complete_frames": 10, "camera_frames": 9,
    }
    wait_args = []
    backend._wait_for_simulator_streams = lambda _connections, _run_dir, **kwargs: wait_args.append(kwargs)

    with patch("src.layer4.hub.native_official_backend.os.killpg", create=True) as killpg:
        backend.pause_training_for_evaluation(run_id=run_id, connections=connections, hub_url="http://hub")
        assert (run_dir / "evaluation_pause.flag").is_file()
        assert killpg.call_args_list[-1].args == (101, getattr(signal, "SIGSTOP", 19))

        backend.resume_training_after_evaluation(
            run_id=run_id, connections=connections, hub_url="http://hub",
        )
        assert not (run_dir / "evaluation_pause.flag").exists()
        assert wait_args == [{"minimum_frames": {0: (11, 10)}}]
        assert killpg.call_args_list[-1].args == (101, getattr(signal, "SIGCONT", 18))
        assert "evaluation_frame_baselines" not in group


def test_native_backend_maps_container_log_paths_to_workspace_logs(tmp_path):
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    value = NativeOfficialBackend._local_logs_path("/runs/rl/official_run_abcd/checkpoint.zip")
    assert value.replace("\\", "/").endswith("logs/rl/official_run_abcd/checkpoint.zip")
    env = {
        "AICAR_MODEL_PATH": "/runs/rl/official_run_abcd/checkpoints/automatic_evaluation_1.zip",
        "AICAR_EVALUATION_OUTPUT": "/runs/rl/official_run_eval/evaluation.json",
    }
    NativeOfficialBackend._localize_runtime_paths(env)
    assert env["AICAR_MODEL_PATH"].replace("\\", "/").endswith(
        "logs/rl/official_run_abcd/checkpoints/automatic_evaluation_1.zip"
    )
    assert env["AICAR_EVALUATION_OUTPUT"].replace("\\", "/").endswith(
        "logs/rl/official_run_eval/evaluation.json"
    )


def test_native_backend_starts_policy_process_in_its_allocated_ros_domain(tmp_path):
    import shlex
    import threading
    from types import SimpleNamespace
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    backend = object.__new__(NativeOfficialBackend)
    backend._startup_lock = threading.RLock()
    backend._lock = threading.RLock()
    backend._groups = {}
    backend.ros_workspace = tmp_path / "ros_ws"
    backend._reserved_ros_domains = set()
    backend._reserved_devkit_ports = set()
    backend._allocate_ros_domain = lambda: 7
    backend._allocate_devkit_port = lambda: 4577
    bridge_domains = []
    backend._start_bridge_pair = lambda _records, _run_id, _index, env, _run_dir, **kwargs: (
        bridge_domains.append((env.get("ROS_DOMAIN_ID"), kwargs["ros_domain"]))
    )
    backend._evaluation_command = lambda _env: ["evaluate"]
    backend._persist_group = lambda _group: None
    spawned_env = {}
    spawned_command = []

    def fake_popen(command, *, env, **_kwargs):
        spawned_command.extend(command)
        spawned_env.update(env)
        return SimpleNamespace(pid=1234, poll=lambda: None)

    with patch("src.layer4.hub.native_official_backend.subprocess.Popen", side_effect=fake_popen):
        backend.create_official_run_container(
            run_id="a1b2c3d4e5f6", role="api", image="unused", network_name="unused",
            env=["AICAR_MODE=evaluate"], metadata_labels={
                "aicar.output_dir": str(tmp_path / "logs" / "rl" / "official_run_a1b2c3d4e5f6"),
                "aicar.run_id": "a1b2c3d4e5f6", "aicar.role": "api",
            },
        )

    assert spawned_env["ROS_DOMAIN_ID"] == "7"
    assert bridge_domains == [("7", 7)]
    assert spawned_command[:2] == ["bash", "-lc"]
    assert "source /opt/ros/jazzy/setup.bash" in spawned_command[2]
    assert shlex.quote(str(backend.ros_workspace / "install" / "setup.bash")) in spawned_command[2]
    assert spawned_command[3:] == ["aicar-native-policy", "evaluate"]


def test_native_bridge_resources_are_unique_and_skip_occupied_ports():
    import socket
    import threading
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    backend = object.__new__(NativeOfficialBackend)
    backend._lock = threading.RLock()
    backend._groups = {}
    backend._reserved_ros_domains = set()
    backend._reserved_devkit_ports = set()
    first_port = None
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for candidate in range(40000, 59000):
        try:
            probe.bind(("0.0.0.0", candidate))
        except OSError:
            continue
        first_port = candidate
        break
    assert first_port is not None
    probe.close()
    backend._devkit_port_start = first_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", first_port))
        assert backend._allocate_ros_domain() == 0
        assert backend._allocate_ros_domain() == 1
        assert backend._allocate_devkit_port() == first_port + 1
        assert backend._allocate_devkit_port() == first_port + 2

    backend._reserved_ros_domains.clear()
    backend._reserved_devkit_ports.clear()
    backend._groups["active-training"] = {
        "ros_domains": [0, 1, 2, 3],
        "devkit_ports": [first_port, first_port + 1, first_port + 2, first_port + 3],
    }
    assert backend._allocate_ros_domain() == 4
    assert backend._allocate_devkit_port() == first_port + 4


def test_reconcile_does_not_mark_a_stopping_run_orphaned(manager):
    run = manager.create_run({"simulator_backend": "windows_external", "n_envs": 1})
    run_id = run["run_id"]
    manager.backend.containers.clear()
    manager._stopping.add(run_id)
    try:
        manager.reconcile(run_id)
        assert manager.get_run(run_id)["state"] == "running"
        assert manager.get_run(run_id)["error"] is None
        assert manager.backend.networks
    finally:
        manager._stopping.discard(run_id)
        manager.backend.networks.clear()
        manager._runs[run_id]["state"] = "stopped"
        manager._runs[run_id]["network_id"] = None


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
        "AICAR_BEHAVIOR_CLONING_ONLY": "1",
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
    assert args["--behavior-cloning-only"] == "1"


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


def test_catalog_exposes_bc_only_checkpoint_for_official_replay(tmp_path):
    path = official_checkpoint(tmp_path, nested=False)
    bc_checkpoint = path.parent / "behavior_cloned_policy.zip"
    path.rename(bc_checkpoint)
    models = list_official_models(tmp_path / "logs" / "rl")
    assert any(item["id"].endswith("/behavior_cloned_policy.zip") for item in models)
    assert checkpoint_settings(bc_checkpoint)["observation_profile"] == "official_sensors_history"


def test_evaluation_batch_counts_warmup_collision_when_race_counter_is_zero():
    # An immediate first-impact stop may occur before the evaluator establishes
    # a scored-race collision baseline. Preserve it in Mission Control totals.
    attempt = {
        "race_collisions": 0,
        "race_collision_baseline": None,
        "raw_collision_count_at_finish": 1,
        "warmup_collisions": 0,
        "stop_reason": "collision_detected",
    }
    assert _attempt_collision_count(attempt) == 1
    assert _attempt_collision_count({"race_collisions": 2, "raw_collision_count_at_finish": 1}) == 2


def test_replay_uses_saved_profile_and_controls_in_fresh_official_pair(manager, tmp_path):
    path = official_checkpoint(tmp_path)
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "official-evaluations.json",
        champion_dir=tmp_path / "best_official_model",
        model_root=tmp_path / "logs" / "rl",
    )
    replay = OfficialReplayJob(manager, evaluation_service=service)
    original_create = manager.create_run
    attempt_envs = []

    def complete_evaluation(*args, **kwargs):
        row = original_create(*args, **kwargs)
        api = next(item for item in manager.backend.containers.values()
                   if item["Labels"]["aicar.run_id"] == row["run_id"] and item["Labels"]["aicar.role"] == "api")
        attempt_envs.append(dict(value.split("=", 1) for value in api["kwargs"]["env"]))
        attempt_number = len([run for run in manager.list_runs()["runs"] if run.get("kind") == "evaluation"])
        result = {
            "score_status": "valid", "valid_for_comparison": True,
            "adjusted_race_time_s": 101.0 + attempt_number, "race_time_s": 100.0 + attempt_number,
            "race_lap_times_s": [10.0] * 10, "race_collisions": 0,
            "stop_reason": "race_complete",
        }
        Path(row["output_dir"], "evaluation.json").write_text(json.dumps({"runs": [result]}))
        return row

    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        manager.create_run = complete_evaluation
        status = replay.start(model_path=path, trace_steps=500)
        assert status["state"] == "queued"
        deadline = time.monotonic() + 5
        while replay.status()["state"] not in {"complete", "incomplete", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
    completed = replay.status()
    assert completed["state"] == "complete"
    assert completed["evaluation"]["completed_attempts"] == 3
    assert completed["evaluation"]["median_adjusted_race_time_s"] == 103.0
    assert service.champion()["model_path"] == "best_official_model/model.zip"
    recovered_service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "official-evaluations.json",
        champion_dir=tmp_path / "best_official_model",
        model_root=tmp_path / "logs" / "rl",
    )
    assert recovered_service.champion()["model_sha256"] == service.champion()["model_sha256"]
    assert recovered_service.history()[0]["source"] == "manual"
    recovered_service.close()
    evaluated = [run for run in manager.list_runs()["runs"] if run.get("kind") == "evaluation"]
    assert len(evaluated) == 3
    assert all(run["simulator_image"].endswith(":2026-iros-compete") for run in evaluated)
    assert all(run["config"]["warmup_laps"] == 1 and run["config"]["race_laps"] == 10 for run in evaluated)
    for env in attempt_envs:
        assert env["AICAR_MODE"] == "evaluate"
        assert env["AICAR_OBSERVATION_PROFILE"] == "official_sensors_history"
        assert env["AICAR_THROTTLE_MODE"] == "forward_only"
        assert env["AICAR_EVALUATION_TRACE_STEPS"] == "500"
    assert not manager.backend.containers
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        restarted = replay.reset()
    assert restarted["state"] == "queued"
    replay.stop()
    service.close()


def test_evaluation_registry_discards_temporary_checkpoints(manager, tmp_path):
    outside = tmp_path / "pytest-temp" / "model.zip"
    outside.parent.mkdir()
    outside.touch()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"history": [{
        "evaluation_id": "stale", "model_path": str(outside), "state": "failed",
    }]}))
    service = OfficialEvaluationService(
        manager, registry_path=registry, champion_dir=tmp_path / "best",
        model_root=tmp_path / "logs" / "rl",
    )
    assert service.history() == []
    assert json.loads(registry.read_text())["history"] == []
    service.close()


def test_manual_evaluation_queues_behind_running_automatic_evaluation(manager, tmp_path):
    import threading

    path = official_checkpoint(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "registry.json",
        champion_dir=tmp_path / "best", model_root=tmp_path / "logs" / "rl",
    )

    def hold_current_job(job):
        entered.set()
        release.wait(timeout=2)
        job["state"] = "stopped"

    service._run_batch = hold_current_job
    try:
        service.submit(
            path, source="automatic", parent_run_id="training-run",
            trigger="steps", trigger_step=50_000,
        )
        assert entered.wait(timeout=2)
        queued = service.submit(path, source="manual")
        assert queued["state"] == "queued"
        assert service.status()["source"] == "automatic"
    finally:
        release.set()
        deadline = time.monotonic() + 2
        while service.status().get("source") == "automatic" and time.monotonic() < deadline:
            time.sleep(0.01)
        service.stop()
        service.close()


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
    assert settings.frontier_stagnation_s == 5.0
    assert environment["AICAR_TRAIN_FRONTIER_STAGNATION_S"] == "5.0"
    assert settings.time_cost_per_simulated_second == 0.0
    assert environment["AICAR_TRAIN_TIME_COST"] == "0.0"
    assert settings.lap_completion_reward == 0.0
    assert environment["AICAR_TRAIN_LAP_COMPLETION_REWARD"] == "0.0"
    assert settings.collision_penalty_magnitude == 200.0
    assert settings.collision_reward_percent == 0.0
    assert environment["AICAR_TRAIN_COLLISION_PENALTY_MAGNITUDE"] == "200.0"
    assert environment["AICAR_TRAIN_COLLISION_REWARD_PERCENT"] == "0.0"
    assert settings.backward_speed_penalty_scale == 10.0
    assert environment["AICAR_TRAIN_BACKWARD_SPEED_PENALTY_SCALE"] == "10.0"
    assert settings.failed_episode_penalty == 200.0
    assert environment["AICAR_TRAIN_FAILURE_REWARD_PERCENT"] == "0.0"
    assert environment["AICAR_TRAIN_FAILURE_PENALTY"] == "200.0"
    with pytest.raises(ValueError):
        OfficialTrainSettings(n_envs=9)
    assert OfficialTrainSettings(device="cuda").to_container_env(hub_url="http://hub", run_id="r")["AICAR_PPO_DEVICE"] == "cuda"
    assert environment["AICAR_AUTO_EVALUATION_ENABLED"] == "1"
    assert environment["AICAR_AUTO_EVALUATION_TRIGGER"] == "steps"
    assert environment["AICAR_AUTO_EVALUATION_INTERVAL"] == "50000"
    with pytest.raises(ValueError):
        OfficialTrainSettings(collision_penalty_magnitude=100.0)
    with pytest.raises(ValueError):
        OfficialTrainSettings(collision_penalty_magnitude=120.0)
    with pytest.raises(ValueError):
        OfficialTrainSettings(failed_episode_penalty=120.0)
    with pytest.raises(ValueError):
        OfficialTrainSettings(lap_completion_reward=1.0)


def test_old_reward_settings_migrate_to_safe_lap_learning(tmp_path):
    path = tmp_path / "official-settings.json"
    path.write_text(json.dumps({
        "time_cost_per_simulated_second": 5.0,
        "collision_penalty_magnitude": 100.0,
        "collision_reward_percent": 100.0,
        "lap_completion_reward": 100.0,
        "failed_episode_penalty": 100.0,
        "episode_failure_reward_percent": 100.0,
    }))

    settings = load_official_train_settings(path)

    assert settings.time_cost_per_simulated_second == 0.0
    assert settings.collision_penalty_magnitude == 200.0
    assert settings.collision_reward_percent == 0.0
    assert settings.lap_completion_reward == 0.0
    assert settings.failed_episode_penalty == 200.0
    assert settings.episode_failure_reward_percent == 0.0


def test_automatic_evaluation_schedule_counts_run_relative_steps_and_vector_episodes():
    from src.layer3.official_train import AutomaticEvaluationSchedule

    steps = AutomaticEvaluationSchedule("steps", 100, starting_steps=128)
    assert steps.observe(163, [{}, {}]) == (False, 163, 0)
    assert steps.observe(231, [{"episode": {"r": 1}}, {"episode": {"r": 2}}]) == (True, 231, 2)
    assert steps.observe(232, []) == (False, 232, 2)

    episodes = AutomaticEvaluationSchedule("episodes", 3, starting_steps=1000)
    assert episodes.observe(1008, [{"episode": {}}, {}, {"episode": {}}]) == (False, 1008, 2)
    assert episodes.observe(1012, [{"episode": {}}]) == (True, 1012, 3)


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
