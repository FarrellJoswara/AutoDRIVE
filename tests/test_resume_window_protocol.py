"""Bounded evaluation handoff timing and stop responsiveness regressions."""

import json
import threading
import time as real_time
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.layer3.ppo_audit as ppo_audit
from src.layer3.ppo_audit import CooperativePauseCallback, CooperativePauseInterrupted
from src.layer4.hub.native_official_backend import NativeOfficialBackend


RUN_ID = "a1b2c3d4e5f6"
REQUEST_ID = "evaluation-resume-late"


def _transport(*, fresh: bool):
    ages = {name: 0.02 for name in (
        "scan", "imu", "throttle", "steering", "left_encoder", "right_encoder", "camera",
    )}
    if not fresh:
        ages["scan"] = 10.0
    return {
        "spin_state": "running", "topic_fresh": fresh,
        "required_topics": list(ages), "topic_age_s": ages,
    }


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep(self.now)


def _pause_callback(tmp_path: Path, stop: threading.Event, clock: _Clock):
    callback = CooperativePauseCallback(
        tmp_path, RUN_ID, stop, SimpleNamespace(),
        resume_request_wait_timeout_s=9000.0,
        resume_freshness_timeout_s=120.0,
        poll_s=0.2,
    )
    callback.model = SimpleNamespace(num_timesteps=8192, _n_updates=4)
    ppo_audit_clock = SimpleNamespace(
        monotonic=clock.monotonic, sleep=clock.sleep, time=real_time.time,
    )
    return callback, ppo_audit_clock


def test_late_resume_request_gets_full_freshness_window(tmp_path, monkeypatch):
    stop = threading.Event()
    clock = _Clock()
    callback, fake_time = _pause_callback(tmp_path, stop, clock)
    request_seen_at = []

    def read_request(_path, run_id):
        if clock.now < 8999.6:
            return None
        if not request_seen_at:
            request_seen_at.append(clock.now)
        return {"run_id": run_id, "request_id": REQUEST_ID}

    callback._read_request = read_request
    callback._env_method = lambda _method: _transport(
        fresh=bool(request_seen_at and clock.now - request_seen_at[0] >= 119.8),
    )
    monkeypatch.setattr(ppo_audit, "time", fake_time)

    result = callback._wait_for_resume(REQUEST_ID)

    assert request_seen_at and request_seen_at[0] >= 8999.6
    assert clock.now > 9000.0
    assert clock.now < request_seen_at[0] + 120.0
    assert callback._transport_is_fresh(result)


def test_operator_stop_interrupts_freshness_wait_promptly(tmp_path, monkeypatch):
    stop = threading.Event()
    clock = _Clock()
    callback, fake_time = _pause_callback(tmp_path, stop, clock)
    callback._read_request = lambda _path, run_id: {"run_id": run_id, "request_id": REQUEST_ID}
    callback._env_method = lambda _method: _transport(fresh=False)
    clock.on_sleep = lambda now: stop.set() if now >= 0.4 else None
    monkeypatch.setattr(ppo_audit, "time", fake_time)

    with pytest.raises(CooperativePauseInterrupted, match="operator stop"):
        callback._wait_for_resume(REQUEST_ID)

    acknowledgement = json.loads(callback.resume_ack_path.read_text(encoding="utf-8"))
    assert acknowledgement["status"] == "cancelled_by_stop"
    assert clock.now <= 0.4


def test_backend_signals_resume_immediately_after_fresh_simulator_streams(tmp_path):
    run_id = "b2c3d4e5f6a1"
    request_id = "completed-evaluation"
    run_dir = tmp_path / f"official_run_{run_id}"
    run_dir.mkdir()
    (run_dir / "evaluation_pause.flag").write_text("pause", encoding="utf-8")
    (run_dir / "evaluation_pause.request.json").write_text(json.dumps({
        "run_id": run_id, "request_id": request_id,
    }), encoding="utf-8")
    (run_dir / "evaluation_pause.ack.json").write_text(json.dumps({
        "run_id": run_id, "request_id": request_id, "status": "paused",
    }), encoding="utf-8")
    (run_dir / "watch_latest.json").write_text(json.dumps({"step": 100}), encoding="utf-8")

    process = SimpleNamespace(poll=lambda: None)
    group = {
        "Labels": {"aicar.run_id": run_id, "aicar.role": "api"},
        "records": [{"role": "trainer", "process": process}],
        "evaluation_pause_request_id": request_id,
        "evaluation_pause_ack": {"status": "paused"},
    }
    backend = object.__new__(NativeOfficialBackend)
    backend._lock = threading.RLock()
    backend._groups = {"test": group}
    backend.startup_timeout_s = 0.5
    backend._run_dir_for_id = lambda _run_id: run_dir
    events = []

    def wait_for_streams(_connections, _run_directory, *, minimum_frames):
        assert minimum_frames == {0: (1, 1)}
        events.append("fresh_simulator_streams_ready")
        return {0: {"simulator_clients": 1}}

    backend._wait_for_simulator_streams = wait_for_streams
    backend._write_json_atomic = lambda path, payload: (
        events.append("resume_request_written") if path.name == "evaluation_resume.request.json" else None,
        path.write_text(json.dumps(payload), encoding="utf-8"),
    )

    def wait_for_ack(**kwargs):
        assert kwargs["phase"] == "resume"
        assert kwargs["timeout_s"] == 120.0
        assert events == ["fresh_simulator_streams_ready", "resume_request_written"]
        request = json.loads(
            (run_dir / "evaluation_resume.request.json").read_text(encoding="utf-8")
        )
        assert request["request_id"] == request_id
        events.append("resume_acknowledged")
        return {"status": "resumed", "reset_completed": True}

    backend._wait_for_training_handshake = wait_for_ack
    backend._wait_for_training_observation_after_resume = lambda **_kwargs: 101

    backend.resume_training_after_evaluation(
        run_id=run_id,
        connections=[{"environment": 0, "port": 4568}],
        hub_url="http://127.0.0.1:8090",
    )

    assert events == [
        "fresh_simulator_streams_ready", "resume_request_written", "resume_acknowledged",
    ]
    event = json.loads((run_dir / "evaluation_resume_events.jsonl").read_text(encoding="utf-8"))
    assert event["previous_watch_step"] == 100
    assert event["resumed_watch_step"] == 101
    assert event["resume_ack"]["reset_completed"] is True
