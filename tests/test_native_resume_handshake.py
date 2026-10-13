import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from src.layer4.hub.native_official_backend import (
    NativeOfficialBackend,
    simulator_session_ready,
)
from src.layer1.official_bridge_relay import DEVKIT_REQUIRED_FIELDS, NEUTRAL_COMMAND, OfficialBridgeRelay


def test_cumulative_pre_pause_frames_do_not_make_a_new_session_ready():
    health = {
        "simulator_clients": 1,
        "complete_frames": 7463,
        "camera_frames": 7463,
        "simulator_sessions": [{
            "sid": "new-sid", "complete_frames": 0, "camera_frames": 0,
            "devkit_commands": 0,
        }],
    }
    assert not simulator_session_ready(health)


def test_current_session_must_have_camera_frame_and_devkit_command():
    health = {
        "simulator_clients": 1,
        "complete_frames": 7463,
        "camera_frames": 7463,
        "simulator_sessions": [{
            "sid": "new-sid", "complete_frames": 1, "camera_frames": 1,
            "devkit_commands": 1,
        }],
    }
    assert simulator_session_ready(health)
    health["simulator_sessions"][0]["devkit_commands"] = 0
    assert not simulator_session_ready(health)


def test_relay_session_health_counters_reset_and_track_devkit_response(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    relay = OfficialBridgeRelay()
    payload = {key: "value" for key in DEVKIT_REQUIRED_FIELDS}
    payload["V1 Front Camera Image"] = b"image"

    async def fake_upstream_emit(_event, _payload):
        relay.pending["fresh-sid"].set_result(dict(NEUTRAL_COMMAND))

    monkeypatch.setattr(relay.upstream, "emit", fake_upstream_emit)
    monkeypatch.setattr(relay.sio, "emit", AsyncMock())
    handlers = relay.sio.handlers["/"]

    async def exercise():
        await handlers["connect"]("fresh-sid", {})
        assert relay._simulator_sessions["fresh-sid"] == {
            "complete_frames": 0, "camera_frames": 0, "devkit_commands": 0,
        }
        await handlers["Bridge"]("fresh-sid", payload)

    asyncio.run(exercise())
    assert relay._simulator_sessions["fresh-sid"] == {
        "complete_frames": 1, "camera_frames": 1, "devkit_commands": 1,
    }
    relay.sio.emit.assert_awaited_once_with("Bridge", data=NEUTRAL_COMMAND, to="fresh-sid")


def test_resume_ack_requires_watch_step_to_advance(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = run_dir / "watch_latest.json"
    snapshot.write_text(json.dumps({"step": 70}), encoding="utf-8")
    process = SimpleNamespace(poll=lambda: None)
    backend = object.__new__(NativeOfficialBackend)
    backend.startup_timeout_s = 0.5

    def publish_resumed_step():
        time.sleep(0.05)
        snapshot.write_text(json.dumps({"step": 71}), encoding="utf-8")

    writer = threading.Thread(target=publish_resumed_step)
    writer.start()
    try:
        assert backend._wait_for_training_observation_after_resume(
            run_id="run-id", run_dir=run_dir, process=process, previous_step=70,
        ) == 71
    finally:
        writer.join()


def test_training_handshake_rejects_cancelled_pause_ack(tmp_path):
    path = tmp_path / "evaluation_pause.ack.json"
    path.write_text(json.dumps({
        "run_id": "run-id", "request_id": "request-id",
        "status": "cancelled_by_stop",
    }), encoding="utf-8")
    backend = object.__new__(NativeOfficialBackend)
    backend.startup_timeout_s = 0.5
    process = SimpleNamespace(poll=lambda: None)
    import pytest
    with pytest.raises(RuntimeError, match="cancelled_by_stop"):
        backend._wait_for_training_handshake(
            run_id="run-id", process=process, path=path,
            request_id="request-id", phase="pause", expected_status="paused",
        )


def test_training_handshake_fails_immediately_when_trainer_exits(tmp_path):
    path = tmp_path / "evaluation_pause.ack.json"
    backend = object.__new__(NativeOfficialBackend)
    backend.startup_timeout_s = 0.5
    process = SimpleNamespace(poll=lambda: 1, returncode=1)
    import pytest
    with pytest.raises(RuntimeError, match="exited before the pause handshake"):
        backend._wait_for_training_handshake(
            run_id="run-id", process=process, path=path,
            request_id="request-id", phase="pause", expected_status="paused",
        )


def test_abort_pause_removes_only_windows_retirement_flag(tmp_path):
    run_dir = tmp_path / "official_run_run-id"
    run_dir.mkdir()
    (run_dir / "evaluation_pause.flag").write_text("pause", encoding="utf-8")
    request = run_dir / "evaluation_pause.request.json"
    request.write_text('{"request_id":"request-id"}', encoding="utf-8")
    backend = object.__new__(NativeOfficialBackend)
    backend._run_dir_for_id = lambda _run_id: run_dir

    backend.abort_evaluation_pause("run-id")

    assert not (run_dir / "evaluation_pause.flag").exists()
    assert request.is_file()


def test_pause_wait_requires_old_sid_disconnect_before_return(tmp_path):
    backend = object.__new__(NativeOfficialBackend)
    backend.startup_timeout_s = 0.5
    observations = iter([
        {"simulator_clients": 1, "simulator_sessions": [{"sid": "old"}]},
        {"simulator_clients": 0, "simulator_sessions": []},
    ])
    backend._read_simulator_health = lambda _port: next(observations)
    backend._wait_for_simulator_sessions_disconnected(
        [{"environment": 0, "port": 4568}], timeout_s=1,
    )
