"""Safe-boundary PPO pause handshake used around official evaluation."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.layer3.ppo_audit as ppo_audit
from src.layer3.ppo_audit import CooperativePauseCallback, CooperativePauseInterrupted


RUN_ID = "a1b2c3d4e5f6"
REQUEST_ID = "pause-20261012-001"


def _fresh_transport():
    return {
        "spin_state": "running",
        "step_counter": 42,
        "topic_age_s": {
            "scan": 0.03,
            "imu": 0.02,
            "throttle": 0.02,
            "steering": 0.02,
            "left_encoder": 0.03,
            "right_encoder": 0.03,
            "camera": 0.1,
        },
        "topic_fresh": True,
        "required_topics": [
            "scan", "imu", "throttle", "steering", "left_encoder", "right_encoder", "camera",
        ],
    }


class _Env:
    num_envs = 1

    def __init__(self):
        self.reset_count = 0
        self.pause_cancel_event = None

    def env_method(self, method, *args):
        if method == "set_pause_cancel_event":
            assert len(args) == 1
            self.pause_cancel_event = args[0]
            return [None]
        if method == "request_pause_neutral":
            return [{
                "published": True,
                "command": {"throttle": 0.0, "steering": 0.0},
                "transport_acknowledged": False,
                **_fresh_transport(),
            }]
        if method == "get_transport_status":
            return [_fresh_transport()]
        raise AssertionError(method)

    def reset(self):
        self.reset_count += 1
        return {"state": np.array([[1.0, 2.0]], dtype=np.float32)}


class _Audit:
    def __init__(self, directory):
        self.path = Path(directory) / "checkpoints" / "post_optimizer_update_00000004_steps_000000008192.zip"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"post update checkpoint")

    def ensure_post_optimizer_checkpoint(self):
        return self.path


def _callback(
    tmp_path, stop_event, *, request_wait_timeout_s=9000.0,
    freshness_timeout_s=120.0, poll_s=0.01,
):
    env = _Env()
    model = SimpleNamespace(num_timesteps=8192, _n_updates=4, _last_obs={"old": np.array([[0]])})
    model.get_env = lambda: env
    callback = CooperativePauseCallback(
        tmp_path, RUN_ID, stop_event, _Audit(tmp_path),
        resume_request_wait_timeout_s=request_wait_timeout_s,
        resume_freshness_timeout_s=freshness_timeout_s, poll_s=poll_s,
    )
    callback.model = model
    return callback, model, env


def _write_request(path, *, run_id=RUN_ID, request_id=REQUEST_ID):
    path.write_text(json.dumps({
        "run_id": run_id,
        "request_id": request_id,
        "requested_at_unix_s": time.time(),
    }), encoding="utf-8")


def test_pause_ack_follows_neutral_and_checkpoint_then_resume_resets_observation(tmp_path):
    stop = threading.Event()
    callback, model, env = _callback(tmp_path, stop)
    _write_request(callback.pause_request_path)
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(callback._on_rollout_start()))
    worker.start()
    deadline = time.monotonic() + 2.0
    while not callback.pause_ack_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    pause_ack = json.loads(callback.pause_ack_path.read_text(encoding="utf-8"))
    assert pause_ack["status"] == "paused"
    assert pause_ack["checkpoint"]["post_optimizer_update"] is True
    assert pause_ack["checkpoint"]["sha256"]
    assert pause_ack["neutral_command"]["command"] == {"throttle": 0.0, "steering": 0.0}

    _write_request(callback.resume_request_path)
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert outcome == [None]
    resume_ack = json.loads(callback.resume_ack_path.read_text(encoding="utf-8"))
    assert resume_ack["status"] == "resumed"
    assert resume_ack["reset_completed"] is True
    assert resume_ack["episode_boundary"]["policy_failure"] is False
    assert env.reset_count == 1
    assert env.pause_cancel_event is stop
    assert model._last_obs["state"].tolist() == [[1.0, 2.0]]
    assert model._last_episode_starts.tolist() == [True]


def test_operator_stop_interrupts_pause_wait_without_resuming_policy_action(tmp_path):
    stop = threading.Event()
    callback, _model, _env = _callback(tmp_path, stop)
    _write_request(callback.pause_request_path)
    errors = []

    def run_pause():
        try:
            callback._on_rollout_start()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_pause)
    worker.start()
    deadline = time.monotonic() + 2.0
    while not callback.pause_ack_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], CooperativePauseInterrupted)
    pause_ack = json.loads(callback.pause_ack_path.read_text(encoding="utf-8"))
    assert pause_ack["status"] == "paused"
    resume_ack = json.loads(callback.resume_ack_path.read_text(encoding="utf-8"))
    assert resume_ack["status"] == "cancelled_by_stop"


class _FakeClock:
    def __init__(self, now=0.0, on_sleep=None):
        self.now = now
        self.on_sleep = on_sleep

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration
        if self.on_sleep is not None:
            self.on_sleep(self.now)


def _install_fake_clock(monkeypatch, clock):
    monkeypatch.setattr(
        ppo_audit, "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep, time=time.time),
    )


def test_resume_request_wait_can_outlast_old_360_second_bound(tmp_path, monkeypatch):
    stop = threading.Event()
    callback, _model, _env = _callback(
        tmp_path, stop, request_wait_timeout_s=9000.0,
        freshness_timeout_s=120.0, poll_s=100.0,
    )
    clock = _FakeClock()

    def create_delayed_request(now):
        if now >= 400.0 and not callback.resume_request_path.exists():
            _write_request(callback.resume_request_path)

    clock.on_sleep = create_delayed_request
    _install_fake_clock(monkeypatch, clock)
    fresh = callback._wait_for_resume(REQUEST_ID)
    assert clock.now >= 400.0
    assert callback._transport_is_fresh(fresh)
    assert not callback.resume_ack_path.exists()


def test_resume_freshness_timeout_starts_after_matching_request(tmp_path, monkeypatch):
    stop = threading.Event()
    callback, _model, env = _callback(
        tmp_path, stop, request_wait_timeout_s=9000.0,
        freshness_timeout_s=1.0, poll_s=0.5,
    )
    _write_request(callback.resume_request_path)
    stale = dict(_fresh_transport(), topic_fresh=False)
    env.env_method = lambda method, *_args: [stale] if method == "get_transport_status" else [None]
    clock = _FakeClock(now=400.0)
    _install_fake_clock(monkeypatch, clock)

    with pytest.raises(TimeoutError, match="telemetry did not become fresh"):
        callback._wait_for_resume(REQUEST_ID)
    ack = json.loads(callback.resume_ack_path.read_text(encoding="utf-8"))
    assert ack["status"] == "timeout"
    assert ack["timeout_phase"] == "waiting_for_fresh_transport"
    assert clock.now >= 401.0


def test_operator_stop_remains_prompt_during_long_resume_request_wait(tmp_path):
    stop = threading.Event()
    callback, _model, _env = _callback(
        tmp_path, stop, request_wait_timeout_s=9000.0,
        freshness_timeout_s=120.0, poll_s=0.01,
    )
    _write_request(callback.pause_request_path)
    errors = []

    def run_pause():
        try:
            callback._on_rollout_start()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_pause)
    worker.start()
    deadline = time.monotonic() + 2.0
    while not callback.pause_ack_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert json.loads(callback.pause_ack_path.read_text(encoding="utf-8"))["status"] == "paused"
    stop.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], CooperativePauseInterrupted)
    resume_ack = json.loads(callback.resume_ack_path.read_text(encoding="utf-8"))
    assert resume_ack["status"] == "cancelled_by_stop"


def test_pause_rejects_invalid_run_id_and_stale_ack_payload(tmp_path):
    stop = threading.Event()
    callback, _model, _env = _callback(tmp_path, stop)
    _write_request(callback.pause_request_path, run_id="different-run")
    callback._on_rollout_start()
    assert not callback.pause_ack_path.exists()


def test_pause_boundary_requires_single_environment(tmp_path):
    stop = threading.Event()
    callback, _model, env = _callback(tmp_path, stop)
    _write_request(callback.pause_request_path)
    env.num_envs = 2
    with pytest.raises(RuntimeError, match="single environment"):
        callback._on_rollout_start()
