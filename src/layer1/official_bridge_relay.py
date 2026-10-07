"""Layer 1 Socket.IO relay for AutoDRIVE's first partial startup frame.

The official Devkit bridge expects every incoming Bridge packet to contain all
sensor fields. The simulator's first packet omits LiDAR ranges, so the stock
callback raises before replying and the simulator never advances to its next
complete frame. This relay answers incomplete frames with neutral controls and
forwards complete packets unchanged to the unmodified official Devkit bridge.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional

from gevent import sleep as gevent_sleep


DEVKIT_REQUIRED_FIELDS = frozenset({
    "V1 Throttle",
    "V1 Steering",
    "V1 Encoder Angles",
    "V1 Position",
    "V1 Orientation Quaternion",
    "V1 Angular Velocity",
    "V1 Linear Acceleration",
    "V1 Linear Velocity",
    "V1 LIDAR Scan Rate",
    "V1 LIDAR Range Array",
    "V1 Front Camera Image",
    "V1 Lap Count",
    "V1 Lap Time",
    "V1 Last Lap Time",
    "V1 Best Lap Time",
    "V1 Collisions",
})

NEUTRAL_COMMAND = {
    "V1 Throttle": "0.0000",
    "V1 Steering": "0.0000",
    "V1 Reset": "False",
}


def missing_devkit_fields(payload: Any) -> tuple[str, ...]:
    """Return the official bridge fields absent from a simulator packet."""
    if not isinstance(payload, dict):
        return tuple(sorted(DEVKIT_REQUIRED_FIELDS))
    return tuple(sorted(DEVKIT_REQUIRED_FIELDS.difference(payload)))


class OfficialBridgeRelay:
    """Relay complete simulator events through the official Devkit bridge."""

    def __init__(
        self,
        *,
        devkit_url: str = "http://127.0.0.1:4567",
        response_timeout_s: float = 2.0,
        connect_timeout_s: float = 30.0,
    ) -> None:
        import socketio

        self._socketio = socketio
        self.response_timeout_s = float(response_timeout_s)
        self.sio = socketio.Server(async_mode="gevent", cors_allowed_origins="*")
        self.upstream = socketio.Client(reconnection=False)
        self.pending: Dict[str, Dict[str, Any]] = {}
        self._warned_missing: set[tuple[str, ...]] = set()
        self._forwarded_frames = 0
        self._commands_received = 0
        self._response_timeouts = 0

        @self.upstream.on("connect")
        def on_devkit_connect() -> None:
            print(json.dumps({"event": "devkit_socket_connected"}), flush=True)

        @self.upstream.on("disconnect")
        def on_devkit_disconnect() -> None:
            print(json.dumps({"event": "devkit_socket_disconnected"}), flush=True)

        @self.upstream.on("Bridge")
        def on_devkit_command(command: Any) -> None:
            # The official bridge broadcasts its actuator response to connected
            # Socket.IO clients. Only forward the command to the pending sim.
            self._commands_received += 1
            for response in tuple(self.pending.values()):
                response["command"] = command
                if self._commands_received == 1:
                    print(json.dumps({
                        "event": "first_devkit_command_received",
                        "keys": sorted(command) if isinstance(command, dict) else [],
                    }), flush=True)
                break
            else:
                if self._commands_received == 1:
                    print(json.dumps({
                        "event": "unsolicited_devkit_command",
                        "keys": sorted(command) if isinstance(command, dict) else [],
                    }), flush=True)

        @self.sio.on("connect")
        def on_simulator_connect(sid: str, environ: Dict[str, Any]) -> None:
            print(json.dumps({"event": "simulator_connected", "sid": sid}), flush=True)

        @self.sio.on("disconnect")
        def on_simulator_disconnect(sid: str) -> None:
            self.pending.pop(sid, None)
            print(json.dumps({"event": "simulator_disconnected", "sid": sid}), flush=True)

        @self.sio.on("Bridge")
        def on_simulator_frame(sid: str, payload: Any) -> None:
            missing = missing_devkit_fields(payload)
            if missing:
                if missing not in self._warned_missing:
                    self._warned_missing.add(missing)
                    print(json.dumps({
                        "event": "incomplete_startup_frame",
                        "sid": sid,
                        "missing_fields": missing,
                        "action": "neutral_controls",
                    }), flush=True)
                self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid)
                return

            response: Dict[str, Any] = {}
            self.pending[sid] = response
            try:
                # Do not decode, rewrite, or inspect values. The official Devkit
                # remains responsible for ROS publishing and policy-facing data.
                self._forwarded_frames += 1
                if self._forwarded_frames == 1:
                    print(json.dumps({
                        "event": "first_complete_frame_forwarded",
                        "keys": sorted(payload),
                    }), flush=True)
                self.upstream.emit("Bridge", payload)
                if self._forwarded_frames == 1:
                    print(json.dumps({
                        "event": "upstream_emit_returned",
                    }), flush=True)
                deadline = time.monotonic() + self.response_timeout_s
                while "command" not in response and time.monotonic() < deadline:
                    gevent_sleep(0.001)
                command = response.get("command")
                if isinstance(command, dict):
                    self.sio.emit("Bridge", data=command, to=sid)
                    return
                self._response_timeouts += 1
                if self._response_timeouts == 1 or self._response_timeouts % 100 == 0:
                    print(json.dumps({
                        "event": "devkit_response_timeout",
                        "count": self._response_timeouts,
                        "sid": sid,
                        "timeout_s": self.response_timeout_s,
                    }), flush=True)
                self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid)
            except Exception as exc:
                print(json.dumps({
                    "event": "relay_error",
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid)
            finally:
                self.pending.pop(sid, None)

        deadline = time.monotonic() + float(connect_timeout_s)
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                self.upstream.connect(devkit_url)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)
        else:
            raise TimeoutError(
                f"Could not connect relay to official Devkit at {devkit_url}: {last_error}"
            ) from last_error

        print(json.dumps({
            "event": "devkit_connected",
            "devkit_url": devkit_url,
        }), flush=True)

    def serve(self, host: str, port: int) -> None:
        from gevent import pywsgi
        from geventwebsocket.handler import WebSocketHandler

        app = self._socketio.WSGIApp(self.sio)
        server = pywsgi.WSGIServer((host, int(port)), app, handler_class=WebSocketHandler)
        print(json.dumps({"event": "relay_listening", "host": host, "port": int(port)}), flush=True)
        try:
            server.serve_forever()
        finally:
            self.upstream.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4568)
    parser.add_argument("--devkit-url", default="http://127.0.0.1:4567")
    parser.add_argument("--response-timeout-s", type=float, default=2.0)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.response_timeout_s <= 0 or args.connect_timeout_s <= 0:
        parser.error("timeouts must be positive")

    relay = OfficialBridgeRelay(
        devkit_url=args.devkit_url,
        response_timeout_s=args.response_timeout_s,
        connect_timeout_s=args.connect_timeout_s,
    )
    relay.serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
