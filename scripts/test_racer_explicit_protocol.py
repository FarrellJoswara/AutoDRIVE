"""Unit tests for Racer's opt-in action/response Bridge protocol."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from src.layer1.racer import Racer
from src.layer1.telemetry import TelemetrySnapshot


class FakeSocketServer:
    def __init__(self) -> None:
        self.emitted: list[dict[str, object]] = []

    def emit(self, event: str, *, data=None, to=None) -> None:
        self.emitted.append({"event": event, "data": data, "to": to})


def _bare_racer(*, explicit: bool = True, step_timeout: float = 0.5) -> Racer:
    racer = Racer.__new__(Racer)
    racer.racer_id = 0
    racer.port = 4567
    racer.step_timeout = step_timeout
    racer.action_interval_s = 0.086 if explicit else None
    racer._explicit_mode = explicit
    racer._explicit_protocol_verified = False
    racer._explicit_protocol_error = None
    racer._explicit_timeout_error = False
    racer._explicit_pending_step_id = None
    racer._explicit_next_step_id = 1
    racer._explicit_sim_time = None
    racer.physics_ticks = None
    racer._explicit_expected_physics_ticks = None
    racer._explicit_last_response_id = None
    racer._explicit_last_response_data = None
    racer.latest_bridge_keys = frozenset()
    racer._bridge_condition = threading.Condition()
    racer.telemetry = TelemetrySnapshot()
    racer.logger = SimpleNamespace(log=lambda snap: None)
    racer.enable_logging = False
    racer._cmd_throttle = 0.0
    racer._cmd_steering = 0.0
    racer._reset_requested = False
    racer._reset_command_step = None
    racer.step_counter = 0
    racer.step_latency_ms = 0.0
    racer.steps_per_second = 0.0
    racer.real_time_factor = 0.0
    racer._last_step_time = time.time()
    racer._step_event = threading.Event()
    racer._reset_sent_event = threading.Event()
    racer._reset_event = threading.Event()
    racer._connected = False
    racer._is_alive = True
    racer.client_sid = "sim-sid"
    racer.sio = FakeSocketServer()
    return racer


def _packet(*, step_id: int, sim_time: float, physics_ticks: int) -> dict[str, str]:
    return {
        "AICAR Step Protocol": "1",
        "AICAR Action Interval": "0.086",
        "AICAR Sim Time": f"{sim_time:.17g}",
        "AICAR Step ID": str(step_id),
        "AICAR Physics Ticks": str(physics_ticks),
        "V1 Position": "0 0 0",
        "V1 Orientation Quaternion": "0 0 0 1",
        "V1 Linear Velocity": "1 0 0",
        "V1 Angular Velocity": "0 0 0",
        "V1 Linear Acceleration": "0 0 0",
        "V1 Throttle": "0",
        "V1 Steering": "0",
        "V1 Collisions": "0",
        # The telemetry parser tolerates absent range data; protocol tests are
        # focused on request/response sequencing rather than observation validity.
    }


def _handshake(racer: Racer) -> None:
    racer._on_bridge("sim-sid", _packet(step_id=0, sim_time=0.0, physics_ticks=0))


def test_explicit_handshake_does_not_automatically_reply() -> None:
    racer = _bare_racer()

    _handshake(racer)

    assert racer._explicit_protocol_verified
    assert racer.telemetry.step_id == 0
    assert racer.simulation_time() == 0.0
    assert racer.sio.emitted == []
    assert "AICAR Step Protocol" in racer.latest_bridge_keys


def test_explicit_step_sends_one_command_and_waits_for_exact_response() -> None:
    racer = _bare_racer()
    _handshake(racer)
    result: list[TelemetrySnapshot] = []
    errors: list[BaseException] = []

    def do_step() -> None:
        try:
            result.append(racer.step(0.4, -0.2))
        except BaseException as exc:  # surfaced in the test thread
            errors.append(exc)

    worker = threading.Thread(target=do_step)
    worker.start()
    deadline = time.monotonic() + 1.0
    while not racer.sio.emitted and time.monotonic() < deadline:
        time.sleep(0.001)
    assert racer.sio.emitted == [
        {
            "event": "Bridge",
            "data": {
                "V1 Throttle": "0.4",
                "V1 Steering": "-0.2",
                "V1 Reset": "False",
                "AICAR Step ID": "1",
            },
            "to": "sim-sid",
        }
    ]
    assert worker.is_alive(), "step() must wait for the matching simulator response"

    racer._on_bridge("sim-sid", _packet(step_id=1, sim_time=0.086, physics_ticks=86))
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert errors == []
    assert len(result) == 1
    assert result[0].step_id == 1
    assert racer.simulation_time() == pytest.approx(0.086)
    assert racer.physics_ticks == 86
    assert len(racer.sio.emitted) == 1, "responses must not trigger an automatic command"


def test_explicit_command_is_dispatched_through_server_loop() -> None:
    racer = _bare_racer()
    _handshake(racer)
    scheduled: list[bool] = []

    class FakeLoop:
        def run_callback_threadsafe(self, callback) -> None:
            scheduled.append(True)
            callback()

    racer._wsgi_server = SimpleNamespace(loop=FakeLoop())
    result: list[TelemetrySnapshot] = []
    worker = threading.Thread(target=lambda: result.append(racer.step(0.2, 0.1)))
    worker.start()
    deadline = time.monotonic() + 1.0
    while not racer.sio.emitted and time.monotonic() < deadline:
        time.sleep(0.001)

    assert scheduled == [True]
    assert racer.sio.emitted[0]["data"]["AICAR Step ID"] == "1"
    racer._on_bridge("sim-sid", _packet(step_id=1, sim_time=0.086, physics_ticks=86))
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert result[0].step_id == 1


def test_explicit_reset_is_a_matched_zero_control_request() -> None:
    racer = _bare_racer()
    _handshake(racer)
    _handshake_response = _packet(step_id=1, sim_time=0.086, physics_ticks=86)
    racer._explicit_next_step_id = 2
    racer._explicit_sim_time = 0.086
    racer._explicit_expected_physics_ticks = 86
    racer.telemetry = TelemetrySnapshot.from_raw_dict(_handshake_response, step_id=1)

    result: list[TelemetrySnapshot] = []
    worker = threading.Thread(target=lambda: result.append(racer.reset()))
    worker.start()
    deadline = time.monotonic() + 1.0
    while not racer.sio.emitted and time.monotonic() < deadline:
        time.sleep(0.001)
    assert racer.sio.emitted[0]["data"] == {
        "V1 Throttle": "0.0",
        "V1 Steering": "0.0",
        "V1 Reset": "True",
        "AICAR Step ID": "2",
    }
    racer._on_bridge("sim-sid", _packet(step_id=2, sim_time=0.172, physics_ticks=86))
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert result[0].step_id == 2
    assert racer.simulation_time() == pytest.approx(0.172)


def test_explicit_step_fails_closed_on_mismatched_id() -> None:
    racer = _bare_racer()
    _handshake(racer)
    result: list[BaseException] = []

    def do_step() -> None:
        try:
            racer.step(0.1, 0.2)
        except BaseException as exc:
            result.append(exc)

    worker = threading.Thread(target=do_step)
    worker.start()
    deadline = time.monotonic() + 1.0
    while not racer.sio.emitted and time.monotonic() < deadline:
        time.sleep(0.001)
    racer._on_bridge("sim-sid", _packet(step_id=2, sim_time=0.086, physics_ticks=86))
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], RuntimeError)
    assert "step ID mismatch" in str(result[0])


def test_duplicate_latest_explicit_response_is_ignored_but_conflict_fails() -> None:
    racer = _bare_racer()
    initial = _packet(step_id=0, sim_time=0.0, physics_ticks=0)
    racer._on_bridge("sim-sid", initial)
    racer._on_bridge("sim-sid", dict(initial))
    assert racer.step_counter == 1
    assert racer.telemetry.step_id == 0
    assert racer._explicit_protocol_error is None

    response = _packet(step_id=1, sim_time=0.086, physics_ticks=86)
    racer._explicit_pending_step_id = 1
    racer._on_bridge("sim-sid", response)
    assert racer.step_counter == 2
    racer._on_bridge("sim-sid", dict(response))
    assert racer.step_counter == 2
    assert racer.simulation_time() == pytest.approx(0.086)
    assert racer._explicit_protocol_error is None

    conflicting = dict(response)
    conflicting["V1 Linear Velocity"] = "9 0 0"
    racer._on_bridge("sim-sid", conflicting)
    assert racer._explicit_protocol_error is not None
    assert "unsolicited" in racer._explicit_protocol_error


def test_explicit_step_times_out_if_matching_response_never_arrives() -> None:
    racer = _bare_racer(step_timeout=0.02)
    _handshake(racer)

    with pytest.raises(TimeoutError, match="timed out waiting"):
        racer.step(0.0, 0.0)


def test_legacy_mode_still_replies_to_each_bridge_packet() -> None:
    racer = _bare_racer(explicit=False)
    racer._cmd_throttle = 0.3
    racer._cmd_steering = -0.1

    racer._on_bridge("sim-sid", {"V1 Linear Velocity": "1 0 0"})

    assert racer.step_counter == 1
    assert racer._step_event.is_set()
    assert racer.sio.emitted == [
        {
            "event": "Bridge",
            "data": {
                "V1 Throttle": "0.3",
                "V1 Steering": "-0.1",
                "V1 Reset": "False",
            },
            "to": "sim-sid",
        }
    ]


def test_legacy_client_rejects_explicit_unity_handshake_without_reply() -> None:
    racer = _bare_racer(explicit=False)

    racer._on_bridge("sim-sid", _packet(step_id=0, sim_time=0.0, physics_ticks=0))

    assert racer._connected is False
    assert racer.step_counter == 0
    assert racer.sio.emitted == []
    with pytest.raises(RuntimeError, match="action_interval_s"):
        _ = racer.is_connected
    with pytest.raises(RuntimeError, match="action_interval_s"):
        racer.step(0.2, 0.0)
    with pytest.raises(RuntimeError, match="action_interval_s"):
        racer.reset()


def test_legacy_reset_surfaces_protocol_mismatch_arriving_during_wait() -> None:
    racer = _bare_racer(explicit=False)

    class HandshakeDuringWait:
        def clear(self) -> None:
            pass

        def wait(self, timeout: float) -> bool:
            racer._on_bridge(
                "sim-sid", _packet(step_id=0, sim_time=0.0, physics_ticks=0)
            )
            return False

    racer._reset_sent_event = HandshakeDuringWait()

    with pytest.raises(RuntimeError, match="action_interval_s"):
        racer.reset()

    assert racer.sio.emitted == []


def test_explicit_handshake_rejects_missing_or_mismatched_metadata() -> None:
    missing = _bare_racer()
    missing._on_bridge("sim-sid", {"V1 Position": "0 0 0"})
    with pytest.raises(RuntimeError, match="protocol metadata is missing"):
        missing._wait_for_explicit_handshake()

    wrong_interval = _bare_racer()
    packet = _packet(step_id=0, sim_time=0.0, physics_ticks=0)
    packet["AICAR Action Interval"] = "0.1"
    wrong_interval._on_bridge("sim-sid", packet)
    with pytest.raises(RuntimeError, match="does not match requested"):
        wrong_interval._wait_for_explicit_handshake()
