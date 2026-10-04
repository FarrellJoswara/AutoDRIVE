"""Real Socket.IO transport regression for explicit request/response steps."""

from __future__ import annotations

import socket
import threading
import time

import socketio

from src.layer1.racer import Racer


def _packet(step_id: int, sim_time: float, ticks: int) -> dict[str, str]:
    return {
        "AICAR Step Protocol": "1",
        "AICAR Action Interval": "0.086",
        "AICAR Sim Time": f"{sim_time:.17g}",
        "AICAR Step ID": str(step_id),
        "AICAR Physics Ticks": str(ticks),
        "V1 Position": "0 0 0",
        "V1 Orientation Quaternion": "0 0 0 1",
        "V1 Linear Velocity": "1 0 0",
        "V1 Angular Velocity": "0 0 0",
        "V1 Linear Acceleration": "0 0 0",
        "V1 Throttle": "0",
        "V1 Steering": "0",
        "V1 Collisions": "0",
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_real_socketio_client_receives_reset_and_next_step_commands() -> None:
    racer = Racer(
        racer_id=991,
        port=_free_port(),
        auto_launch=False,
        enable_logging=False,
        action_interval_s=0.086,
        step_timeout=3.0,
    )
    client = socketio.Client(reconnection=False, logger=False, engineio_logger=False)
    client_errors: list[BaseException] = []
    client_commands: list[dict[str, str]] = []
    client_lock = threading.Lock()

    @client.on("Bridge")
    def on_bridge(data: dict[str, str]) -> None:
        try:
            command_id = int(data["AICAR Step ID"])
            with client_lock:
                client_commands.append(dict(data))
            client.emit("Bridge", _packet(command_id, command_id * 0.086, 86))
        except BaseException as exc:
            client_errors.append(exc)

    try:
        client.connect(
            f"http://127.0.0.1:{racer.port}",
            transports=["websocket"],
            wait_timeout=3.0,
        )
        client.emit("Bridge", _packet(0, 0.0, 0))
        deadline = time.monotonic() + 3.0
        while not racer._explicit_protocol_verified and time.monotonic() < deadline:
            time.sleep(0.01)
        assert racer._explicit_protocol_verified

        assert racer.reset().step_id == 1
        assert racer.simulation_time() == 0.086
        for index in range(10):
            # Leave the WebSocket receiver idle before each OS-thread emit so
            # this covers waking a waiting gevent outbound queue repeatedly.
            time.sleep(0.1)
            response = racer.step(0.25, -0.1)
            expected_id = index + 2
            assert response.step_id == expected_id
            assert racer.simulation_time() == expected_id * 0.086
        assert [command["AICAR Step ID"] for command in client_commands] == [
            str(step_id) for step_id in range(1, 12)
        ]
        assert client_errors == []
    finally:
        if client.connected:
            client.disconnect()
        racer.kill()
