import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.layer4.hub.native_linux_official_backend import NativeLinuxOfficialBackend


def _backend(tmp_path, monkeypatch):
    run_id = "a1b2c3d4e5f6"
    run_dir = tmp_path / f"official_run_{run_id}"
    run_dir.mkdir()
    checkpoint = run_dir / "pause-checkpoint.zip"
    checkpoint.write_bytes(b"durable post-update checkpoint")
    process = SimpleNamespace(poll=lambda: None)
    group = {
        "Labels": {"aicar.run_id": run_id, "aicar.role": "api"},
        "records": [{"role": "trainer", "process": process}],
    }
    backend = object.__new__(NativeLinuxOfficialBackend)
    backend._lock = threading.RLock()
    backend._groups = {"api": group}
    backend._linux_launch_states = {}
    backend.startup_timeout_s = 0.1
    backend._run_dir_for_id = lambda _run_id: run_dir
    connections = [{"environment": 0, "host": "127.0.0.1", "port": 47931}]
    config = {"track_profile": "practice", "native_runtime": str(tmp_path / "runtime"),
              "simulator_executable": str(tmp_path / "runtime/player")}
    state = {"config": config, "connections": connections, "hub_url": "http://hub"}
    backend._linux_launch_states[run_id] = state
    (run_dir / "linux_simulator_lifecycle.json").write_text(json.dumps(state))
    monkeypatch.setattr(backend, "_read_simulator_health", lambda _port: {
        "simulator_clients": 1, "complete_frames": 8, "camera_frames": 8,
        "simulator_sessions": [{"sid": "old-session", "complete_frames": 8,
                                "camera_frames": 8, "devkit_commands": 8}],
    })
    backend._verify_pause_checkpoint = lambda _directory, ack: {
        **ack["checkpoint"], "path": str(checkpoint), "sha256": "a" * 64,
    }
    return backend, run_id, run_dir, connections, state, group, checkpoint


def test_linux_fresh_stream_gate_requires_readiness_on_new_sid(monkeypatch, tmp_path):
    backend, run_id, _run_dir, connections, _state, _group, _checkpoint = _backend(tmp_path, monkeypatch)
    backend._read_simulator_health = lambda _port: {
        "simulator_clients": 2,
        "simulator_sessions": [
            {"sid": "old-session", "complete_frames": 8, "camera_frames": 8,
             "devkit_commands": 8},
            {"sid": "new-session", "complete_frames": 0, "camera_frames": 0,
             "devkit_commands": 0},
        ],
    }
    clock_values = iter((0.0, 46.0))
    monkeypatch.setattr(
        "src.layer4.hub.native_linux_official_backend.time",
        SimpleNamespace(monotonic=lambda: next(clock_values), sleep=lambda _seconds: None),
    )

    with pytest.raises(TimeoutError, match="new relay session ids"):
        backend._wait_for_linux_fresh_streams(
            run_id=run_id, connections=connections, old_sids={"0": ["old-session"]},
        )


def test_linux_pause_retire_relaunch_fresh_session_then_resume(monkeypatch, tmp_path):
    backend, run_id, run_dir, connections, state, group, _checkpoint = _backend(tmp_path, monkeypatch)
    events = []
    ack = {"status": "paused", "checkpoint": {"path": "ignored", "sha256": "a" * 64,
            "post_optimizer_update": True}, "optimizer_updates": 3, "num_timesteps": 64}

    def handshake(**kwargs):
        events.append(kwargs["phase"])
        return ack if kwargs["phase"] == "pause" else {"status": "resumed", "reset_completed": True}

    backend._wait_for_training_handshake = handshake
    backend._wait_for_simulator_sessions_disconnected = lambda _connections: events.append("disconnected")
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.stop_linux_simulators",
                        lambda _run_id: events.append("retired"))
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.start_linux_simulators",
                        lambda **kwargs: events.append(("relaunch", kwargs["config"]["track_profile"])))
    backend._wait_for_linux_fresh_streams = lambda **kwargs: (
        events.append(("fresh_streams", kwargs["old_sids"]["0"])) or {0: {"simulator_clients": 1}}
    )
    backend._watch_step = lambda _directory: 100
    backend._wait_for_training_observation_after_resume = lambda **_kwargs: 101

    backend.pause_training_for_evaluation(run_id=run_id, connections=connections, hub_url="http://hub")
    assert events[:3] == ["pause", "retired", "disconnected"]
    assert state["paused"] is True
    assert group["evaluation_pause_ack"]["checkpoint"]["path"].endswith("pause-checkpoint.zip")

    backend.resume_training_after_evaluation(run_id=run_id, connections=connections, hub_url="http://hub")
    assert events[3:] == ["retired", "disconnected", ("relaunch", "practice"),
                          ("fresh_streams", ["old-session"]), "resume"]
    assert state["paused"] is False
    event = json.loads((run_dir / "evaluation_resume_events.jsonl").read_text().splitlines()[0])
    assert event["resumed_watch_step"] == 101
    assert event["fresh_simulator_session_required"] is True
    assert not (run_dir / "evaluation_resume.request.json").exists()


def test_linux_pause_timeout_leaves_matching_resume_release_for_callback(monkeypatch, tmp_path):
    backend, run_id, run_dir, connections, state, _group, checkpoint = _backend(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "_wait_for_training_handshake",
                        lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("pause timed out")))
    stopped = []
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.stop_linux_simulators",
                        lambda _run_id: stopped.append(_run_id))

    with pytest.raises(TimeoutError, match="pause timed out"):
        backend.pause_training_for_evaluation(run_id=run_id, connections=connections, hub_url="http://hub")

    request = json.loads((run_dir / "evaluation_resume.request.json").read_text())
    assert request["request_id"]
    assert not (run_dir / "evaluation_pause.request.json").exists()
    assert stopped == []
    assert checkpoint.is_file()
    assert state["paused"] is False


def test_linux_partial_pause_failure_relaunches_and_releases_waiting_trainer(monkeypatch, tmp_path):
    backend, run_id, run_dir, connections, state, _group, checkpoint = _backend(tmp_path, monkeypatch)
    events = []
    ack = {"status": "paused", "checkpoint": {"path": "ignored", "sha256": "a" * 64,
            "post_optimizer_update": True}, "optimizer_updates": 3, "num_timesteps": 64}

    def handshake(**kwargs):
        events.append(kwargs["phase"])
        return ack if kwargs["phase"] == "pause" else {"status": "resumed"}

    backend._wait_for_training_handshake = handshake
    backend._wait_for_training_observation_after_resume = lambda **_kwargs: 65
    disconnect_calls = []

    def disconnected(_connections):
        disconnect_calls.append(True)
        if len(disconnect_calls) == 1:
            raise TimeoutError("relay session did not retire")

    backend._wait_for_simulator_sessions_disconnected = disconnected
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.stop_linux_simulators",
                        lambda _run_id: events.append("stop"))
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.start_linux_simulators",
                        lambda **kwargs: events.append(("restart", kwargs["config"]["track_profile"])))
    backend._wait_for_linux_fresh_streams = lambda **kwargs: events.append("fresh-streams")

    with pytest.raises(TimeoutError, match="relay session did not retire"):
        backend.pause_training_for_evaluation(run_id=run_id, connections=connections, hub_url="http://hub")

    assert events == ["pause", "stop", "stop", ("restart", "practice"), "fresh-streams", "resume"]
    assert checkpoint.is_file()
    assert not (run_dir / "evaluation_pause.flag").exists()
    assert not (run_dir / "evaluation_resume.request.json").exists()
    assert state["pause_recovery_error"] is None


def test_linux_abort_pause_clears_gate_without_relaunching_simulator(tmp_path, monkeypatch):
    backend, run_id, run_dir, _connections, _state, _group, _checkpoint = _backend(tmp_path, monkeypatch)
    (run_dir / "evaluation_pause.flag").write_text("paused")
    backend.abort_evaluation_pause(run_id)
    assert not (run_dir / "evaluation_pause.flag").exists()


def test_linux_resume_stream_timeout_keeps_recovery_request_and_checkpoint(tmp_path, monkeypatch):
    backend, run_id, run_dir, connections, state, group, checkpoint = _backend(tmp_path, monkeypatch)
    state.update({"paused": True, "request_id": "resume-timeout", "old_sids": {"0": ["old-session"]},
                  "pause_ack": {"checkpoint": {"path": str(checkpoint)}}})
    group["evaluation_pause_request_id"] = "resume-timeout"
    (run_dir / "evaluation_pause.flag").write_text("paused")
    (run_dir / "evaluation_pause.request.json").write_text(json.dumps({"request_id": "resume-timeout"}))
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.stop_linux_simulators", lambda _run_id: None)
    monkeypatch.setattr("src.layer4.hub.native_linux_official_backend.start_linux_simulators", lambda **_kwargs: None)
    backend._wait_for_simulator_sessions_disconnected = lambda _connections: None
    backend._wait_for_linux_fresh_streams = lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("fresh session timeout"))

    with pytest.raises(RuntimeError, match="fresh session timeout"):
        backend.resume_training_after_evaluation(run_id=run_id, connections=connections, hub_url="http://hub")

    request = json.loads((run_dir / "evaluation_resume.request.json").read_text())
    assert request["request_id"] == "resume-timeout"
    assert json.loads((run_dir / "evaluation_pause.flag").read_text())["recovery_required"] is True
    assert checkpoint.is_file()
