from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.layer4.hub.linux_official_simulators import validate_linux_simulator_paths
from src.layer4.hub.native_linux_official_backend import NativeLinuxOfficialBackend
from src.layer4.hub.official_run_manager import normalize_official_config, _training_simulator_image
from src.layer4.official_settings import OfficialTrainSettings


def _runtime(tmp_path: Path, monkeypatch):
    runtime = tmp_path / "runtime"
    rootfs = runtime / "rootfs"
    executable = rootfs / "home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"practice player")
    executable.chmod(0o755)
    wrapper = tmp_path / "native-linux-simulator.sh"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    identity = {
        "repository": "autodriveecosystem/autodrive_roboracer_sim",
        "tag": "2026-iros-practice",
        "digest": "sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76",
        "architecture": "linux/amd64", "manifest_verified": True,
        "rootfs_tree_digest": "sha256:" + "a" * 64,
        "simulator_path": executable.relative_to(rootfs).as_posix(),
        "simulator_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    (rootfs / ".OCI-COMPLETE.json").write_text(json.dumps({"tag": identity["tag"], "digest": identity["digest"], "verified_layers": 30}))
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "IMAGE-IDENTITY.json").write_text(json.dumps(identity), encoding="utf-8")
    monkeypatch.setenv("AICAR_NATIVE_RUNTIME", str(runtime))
    monkeypatch.setenv("AICAR_OFFICIAL_SIMULATOR_WRAPPER", str(wrapper))
    monkeypatch.setenv("AICAR_OFFICIAL_SIMULATOR_EXE", str(executable))
    return runtime, executable, wrapper


def test_pinned_practice_image_and_executable_are_verified(tmp_path, monkeypatch):
    runtime, executable, wrapper = _runtime(tmp_path, monkeypatch)
    assert validate_linux_simulator_paths() == (runtime.resolve(), executable.resolve(), wrapper.resolve())
    executable.write_bytes(b"changed player")
    with pytest.raises(RuntimeError, match="executable hash"):
        validate_linux_simulator_paths()


def test_wrong_image_attestation_is_rejected(tmp_path, monkeypatch):
    runtime, _executable, _wrapper = _runtime(tmp_path, monkeypatch)
    marker = runtime / "IMAGE-IDENTITY.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["digest"] = "sha256:wrong"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned official"):
        validate_linux_simulator_paths()


def test_linux_settings_choose_practice_and_no_competitive_autoeval():
    settings = OfficialTrainSettings(simulator_backend="linux_external")
    assert settings.automatic_evaluation_enabled is False
    config = normalize_official_config({"simulator_backend": "linux_external", "automatic_evaluation_enabled": False})
    assert _training_simulator_image(config).startswith("official AutoDRIVE 2026 IROS Linux practice")


def test_linux_training_can_queue_automatic_evaluation_with_competition_probe_path(tmp_path, monkeypatch):
    from scripts.test_official_run_manager import FakeDocker
    from src.layer4.hub.official_run_manager import OfficialRunManager
    from src.layer4.settings import ROOT

    class FakeNativeLinux(FakeDocker):
        supports_linux_external = True
        required_simulator_backend = "linux_external"
        simulator_host = "127.0.0.1"

        def validate_linux_runtime(self, config):
            self.validated_profile = config["track_profile"]

        def start_simulators(self, **kwargs):
            self.started = kwargs

    backend = FakeNativeLinux(tmp_path)
    manager = OfficialRunManager(backend=backend, registry_path=tmp_path / "registry.json",
                                 runs_root=tmp_path / "runs", start_monitor=False)
    try:
        train = manager.create_run({
            "simulator_backend": "linux_external", "n_envs": 1,
            "automatic_evaluation_enabled": True,
        })
        assert train["kind"] == "training"
        assert backend.validated_profile == "practice"
        assert backend.started["config"]["automatic_evaluation_enabled"] is True

        checkpoint = ROOT / "logs" / "rl" / f"linux-probe-{tmp_path.name}.zip"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        probe = manager.create_run({
            "simulator_backend": "linux_external", "n_envs": 1, "resume": str(checkpoint),
            "automatic_evaluation_enabled": False,
        }, kind="evaluation", evaluation_options={
            "validation_mode": "training_probe", "lap_progress_timeout_s": 240,
            "lap_progress_wall_timeout_s": 240, "frontier_stall_after_s": 5,
            "frontier_path": str(ROOT / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"),
        })
        assert backend.validated_profile == "competition"
        assert backend.started["config"]["track_profile"] == "competition"
        assert probe["simulator_image"].endswith("Linux competition release (pinned extracted runtime)")
        manager.stop_run(probe["run_id"])
        manager.stop_run(train["run_id"])
        checkpoint.unlink(missing_ok=True)
    finally:
        manager.close()


def test_linux_external_manager_allocates_worker_domains_from_zero_without_docker(tmp_path, monkeypatch):
    from scripts.test_official_run_manager import FakeDocker
    from src.layer4.hub import docker_control
    from src.layer4.hub.official_run_manager import OfficialRunManager

    class FakeNativeLinux(FakeDocker):
        supports_linux_external = True
        required_simulator_backend = "linux_external"
        simulator_host = "127.0.0.1"

        def validate_linux_runtime(self, _config):
            return None

        def start_simulators(self, **kwargs):
            self.started = kwargs

    backend = FakeNativeLinux(tmp_path)
    manager = OfficialRunManager(
        backend=backend, registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "logs" / "rl", start_monitor=False,
    )
    monkeypatch.delenv("AICAR_ROS_DOMAIN_ID_BASE", raising=False)
    monkeypatch.setattr(docker_control, "_request", lambda *_a, **_kw: pytest.fail("Linux native path called Docker API"))
    try:
        run = manager.create_run({
            "simulator_backend": "linux_external", "n_envs": 3,
            "automatic_evaluation_enabled": False,
        })
        api_envs = {
            item["Labels"]["aicar.role"]: dict(value.split("=", 1) for value in item["kwargs"]["env"])
            for item in backend.containers.values()
            if item["Labels"]["aicar.role"].startswith("api")
        }
        assert api_envs["api-1"]["ROS_DOMAIN_ID"] == "1"
        assert api_envs["api-2"]["ROS_DOMAIN_ID"] == "2"
        assert manager._runs[run["run_id"]]["ros_domain_base"] is None
        assert len(backend.started["connections"]) == 3
    finally:
        manager.close()


def test_linux_evaluation_requires_porto_practice_diagnostic(tmp_path, monkeypatch):
    from scripts.test_official_run_manager import FakeDocker
    from src.layer4.hub.official_run_manager import OfficialRunManager
    from src.layer4.settings import ROOT

    class FakeNativeLinux(FakeDocker):
        supports_linux_external = True
        required_simulator_backend = "linux_external"
        simulator_host = "127.0.0.1"

        def validate_linux_runtime(self, _config):
            self.validated = True

        def start_simulators(self, **kwargs):
            self.started = kwargs

    checkpoint = ROOT / "logs" / "rl" / f"linux-eval-gate-{tmp_path.name}.zip"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    backend = FakeNativeLinux(tmp_path)
    manager = OfficialRunManager(
        backend=backend, registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "logs" / "rl", start_monitor=False,
    )
    config = {
        "simulator_backend": "linux_external", "automatic_evaluation_enabled": False,
        "resume": str(checkpoint),
    }
    try:
        for options in (
            {},
            {"validation_mode": "bc_diagnostic", "lap_progress_timeout_s": 240,
             "lap_progress_wall_timeout_s": 240, "frontier_stall_after_s": 5,
             "frontier_path": str(tmp_path / "frontier.csv")},
        ):
                with pytest.raises(ValueError, match="native Linux evaluation requires"):
                    manager.create_run(config, kind="evaluation", evaluation_options=options)

        assert not backend.containers
        assert not getattr(backend, "validated", False)
        competition = manager.create_run(config, kind="evaluation", evaluation_options={
            "route_monitor_only": True,
            "frontier_path": str(ROOT / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"),
        })
        assert backend.validated
        assert backend.started["config"]["track_profile"] == "competition"
        assert backend.started["config"]["track_profile"] == "competition"
        assert competition["simulator_image"].endswith("Linux competition release (pinned extracted runtime)")
        manager.stop_run(competition["run_id"])
    finally:
        manager.close()
        checkpoint.unlink(missing_ok=True)


def test_configured_ros_and_python_paths_are_used_for_preflight(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    ros = tmp_path / "ros" / "setup.bash"
    overlay = tmp_path / "ws" / "install" / "setup.bash"
    ros.parent.mkdir(parents=True); ros.write_text("# ROS")
    overlay.parent.mkdir(parents=True); overlay.write_text("# workspace")
    backend = object.__new__(NativeLinuxOfficialBackend)
    backend.native_ros_setup = ros
    backend.native_ros_workspace = overlay.parents[1]
    backend.native_python = Path(os.sys.executable)
    invoked = []
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.subprocess.run", lambda *a, **k: invoked.append((a, k)) or SimpleNamespace(returncode=0, stdout=""))
    backend.validate_linux_runtime()
    command = invoked[0][0][0][-1]
    assert str(ros) in command and str(overlay) in command
    assert str(Path(os.sys.executable).resolve()) in command


def test_missing_ros_prerequisites_fail_honestly(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch)
    backend = object.__new__(NativeLinuxOfficialBackend)
    backend.native_ros_setup = tmp_path / "missing-setup.bash"
    backend.native_ros_workspace = tmp_path / "missing-workspace"
    backend.native_python = tmp_path / "missing-python"
    with pytest.raises(RuntimeError, match="AICAR_NATIVE_ROS_SETUP"):
        backend.validate_linux_runtime()


def test_inherited_native_lifecycle_uses_local_relay_and_keeps_linux_backend(tmp_path, monkeypatch):
    backend = object.__new__(NativeLinuxOfficialBackend)
    backend._lock = threading.RLock(); backend._startup_lock = threading.Lock(); backend._groups = {}
    backend._reserved_ros_domains = set(); backend._reserved_devkit_ports = set(); backend._devkit_port_start = 4567
    backend.backend_instance_id = "test"; backend.hub_pid = os.getpid()
    backend.startup_timeout_s = 1.0
    backend.ros_workspace = tmp_path / "ws"; backend.ros_setup_script = tmp_path / "ros-setup.bash"
    backend.relay_bind_host = "127.0.0.1"
    calls = []
    class Proc:
        returncode = None
        def __init__(self, args): self.args, self.pid = args, 99000 + len(calls)
        def poll(self): return self.returncode
        def terminate(self): self.returncode = 0
    monkeypatch.setattr("src.layer4.hub.native_official_backend.subprocess.Popen", lambda args, **kw: calls.append((args, kw)) or Proc(args))
    monkeypatch.setattr("src.layer4.hub.native_official_backend.socket.create_connection", lambda *_a, **_k: MagicMock(__enter__=lambda _s: None, __exit__=lambda *_a: None))
    monkeypatch.setattr(backend, "_read_process_identity", lambda _pid: None)
    monkeypatch.setattr(backend, "_persist_group", lambda _group: None)
    identity = backend.create_official_run_container(
        run_id="linux-local", role="api", image="unused", network_name="local",
        env=["AICAR_SIMULATOR_BACKEND=linux_external", "AICAR_MODE=train"],
        metadata_labels={"aicar.output_dir": str(tmp_path / "run")}, published_ports={"4568": 49192},
    )
    assert identity.startswith("native-linux-local-api-")
    assert len(calls) == 3
    assert calls[1][0][0:5] == [os.sys.executable, "-m", "src.layer1.official_bridge_relay", "--host", "127.0.0.1"]
    assert {row["role"] for row in backend._groups[identity]["records"]} == {"devkit-0", "relay-0", "trainer"}
    assert all("docker" not in " ".join(map(str, args)) for args, _kw in calls)


def test_linux_backend_starts_selected_official_profile(monkeypatch):
    backend = object.__new__(NativeLinuxOfficialBackend)
    started = []
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.start_linux_simulators", lambda **kw: started.append(kw))
    monkeypatch.setattr(backend, "_wait_for_simulator_streams", lambda *_a, **_kw: None)
    backend.start_simulators(run_id="practice", connections=[{"environment": 0, "port": 49192}], hub_url="", config={"track_profile": "practice"})
    assert started[0]["run_id"] == "practice"
    backend.start_simulators(run_id="competition", connections=[{"environment": 0, "port": 49193}], hub_url="", config={"track_profile": "competition"})
    assert started[1]["config"]["track_profile"] == "competition"


def test_competition_profile_selects_and_requires_exact_competition_identity(tmp_path, monkeypatch):
    runtime, executable, wrapper = _runtime(tmp_path, monkeypatch)
    (runtime / "rootfs" / ".OCI-COMPLETE.json").write_text(json.dumps({
        "tag": "2026-iros-compete",
        "digest": "sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d",
        "verified_layers": 30,
    }))
    marker = runtime / "IMAGE-IDENTITY.json"
    identity = json.loads(marker.read_text())
    identity.update(tag="2026-iros-compete", digest="sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d")
    marker.write_text(json.dumps(identity))
    assert validate_linux_simulator_paths({"track_profile": "competition", "native_runtime": str(runtime)}) == (runtime.resolve(), executable.resolve(), wrapper.resolve())
    identity.update(tag="2026-iros-practice", digest="sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76")
    marker.write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="pinned official linux/amd64 competition"):
        validate_linux_simulator_paths({"track_profile": "competition", "native_runtime": str(runtime)})


def test_exporter_pins_both_release_tag_digest_pairs():
    import runpy
    releases = runpy.run_path("scripts/native-linux-export.py")["RELEASES"]
    assert releases["practice"] == ("2026-iros-practice", "sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76")
    assert releases["competition"] == ("2026-iros-compete", "sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d")


def test_competition_route_accepts_only_pinned_bytes_or_lf_checkout_form():
    from src.layer3.official_evaluate import _verified_competition_route_bytes
    route = Path("competition/iros2026/tracks/competition/official_centerline.csv").read_bytes()
    lf = route.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(crlf).hexdigest() == "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"
    lf_identity = _verified_competition_route_bytes(lf)
    crlf_identity = _verified_competition_route_bytes(crlf)
    assert lf_identity["route_sha256"] == crlf_identity["route_sha256"] == "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"
    assert lf_identity["route_file_sha256"] == hashlib.sha256(lf).hexdigest()
    assert crlf_identity["route_file_sha256"] == hashlib.sha256(crlf).hexdigest()
    assert lf_identity["route_line_ending_normalized"] is True
    assert crlf_identity["route_line_ending_normalized"] is False
    assert lf_identity["route_hash_reference_archive_sha256"] == "bc350d1f3211083c251d292de20fef6f9024ca5b6fe4a02a4985a754df727954"
    altered = lf.replace(b"0.000000", b"0.000001", 1)
    if altered == lf:
        altered = lf[:-1] + (b"x" if lf[-1:] != b"x" else b"y")
    with pytest.raises(ValueError, match="bytes do not match"):
        _verified_competition_route_bytes(altered)


def test_linux_competition_route_monitor_requires_competition_process_identity(tmp_path, monkeypatch):
    from src.layer3.official_evaluate import _route_monitor_identity
    runtime, executable, _wrapper = _runtime(tmp_path, monkeypatch)
    marker_path = runtime / "IMAGE-IDENTITY.json"
    marker = json.loads(marker_path.read_text())
    marker.update(tag="2026-iros-compete", digest="sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d")
    marker_path.write_text(json.dumps(marker))
    (runtime / "rootfs" / ".OCI-COMPLETE.json").write_text(json.dumps({"tag": marker["tag"], "digest": marker["digest"]}))
    run_id = "linux-competition-provenance"
    output = tmp_path / "run" / "evaluation.json"
    output.parent.mkdir()
    record = {
        "pid": 31, "simulator_pid": 32, "process_group_id": 31,
        "process_starttime_ticks": 101, "simulator_starttime_ticks": 202,
        "executable": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "simulator_image": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete",
        "simulator_image_digest": marker["digest"], "port": 47931,
    }
    manifest_path = output.parent / "linux_simulator_processes.json"
    manifest_path.write_text(json.dumps({"run_id": run_id, "processes": [record]}))
    monkeypatch.setenv("AICAR_SIMULATOR_BACKEND", "linux_external")
    monkeypatch.setattr("src.layer4.hub.linux_official_simulators._proc_start_ticks", lambda pid: {31: 101, 32: 202}.get(pid))
    monkeypatch.setattr("src.layer4.hub.linux_official_simulators._proc_group_id", lambda _pid: 31)
    route = Path("competition/iros2026/tracks/competition/official_centerline.csv").resolve()
    identity = _route_monitor_identity(output_path=output, frontier_path=route, expected_run_id=run_id)
    assert identity["simulator_image_digest"] == marker["digest"]
    assert identity["track_profile"] == "competition"
    assert identity["route_sha256"] == "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"
    assert identity["route_file_sha256"] == hashlib.sha256(route.read_bytes()).hexdigest()
    assert identity["route_line_ending_normalized"] is True
    record["simulator_image_digest"] = "sha256:practice"
    manifest_path.write_text(json.dumps({"run_id": run_id, "processes": [record]}))
    with pytest.raises(ValueError, match="pinned competition runtime"):
        _route_monitor_identity(output_path=output, frontier_path=route, expected_run_id=run_id)


def test_linux_practice_evaluator_requires_run_local_process_identity(tmp_path, monkeypatch):
    from src.layer3.official_evaluate import _verified_porto_practice_runtime
    monkeypatch.setenv("AICAR_SIMULATOR_BACKEND", "linux_external")
    monkeypatch.setenv("AICAR_OFFICIAL_TRACK_PROFILE", "practice")
    monkeypatch.delenv("AICAR_EVALUATION_OUTPUT", raising=False)
    monkeypatch.delenv("AICAR_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="run-local output path and run identity"):
        _verified_porto_practice_runtime({"route_path": str(tmp_path / "centerline.csv")})
