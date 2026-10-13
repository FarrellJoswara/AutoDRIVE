"""Async Socket.IO relay for official AutoDRIVE simulator connections.

The official Devkit bridge expects every incoming Bridge packet to contain all
sensor fields. The simulator's first packet omits LiDAR ranges, so the stock
callback raises before replying and the simulator never advances to its next
complete frame. This relay answers incomplete frames with neutral controls and
forwards complete packets unchanged to the unmodified official Devkit bridge.
Its aiohttp server and Engine.IO compatibility hook match the official
AutoDRIVE Race Control Tower's handling of the Unity client.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import signal
from typing import Any, Dict, Optional


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
        from socketio import packet as socketio_packet

        self._socketio = socketio
        self._socketio_packet = socketio_packet
        self.response_timeout_s = float(response_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.sio = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins="*",
            max_http_buffer_size=128 * 1024 * 1024,
        )
        self.upstream = socketio.AsyncClient(reconnection=False)
        self.pending: Dict[str, asyncio.Future[Any]] = {}
        self._warned_missing: set[tuple[str, ...]] = set()
        self._forwarded_frames = 0
        self._commands_received = 0
        self._response_timeouts = 0
        self._connected_simulators: set[str] = set()
        # Counters are scoped to each simulator Socket.IO SID. Cumulative
        # counters alone can make a restarted simulator appear ready based on
        # frames emitted by the retired process.
        self._simulator_sessions: Dict[str, Dict[str, int]] = {}
        self._camera_frames = 0
        self._incomplete_frames = 0
        # Bound shutdown latency while releasing all connected simulators.
        # Socket.IO emit completion confirms transport enqueue/send only; it
        # is not an application-level actuator acknowledgement.
        self.shutdown_neutral_timeout_s = 0.5

        @self.upstream.on("connect")
        async def on_devkit_connect() -> None:
            print(json.dumps({"event": "devkit_socket_connected"}), flush=True)

        @self.upstream.on("disconnect")
        async def on_devkit_disconnect() -> None:
            print(json.dumps({"event": "devkit_socket_disconnected"}), flush=True)

        @self.upstream.on("Bridge")
        async def on_devkit_command(command: Any) -> None:
            # The official bridge broadcasts its actuator response to connected
            # Socket.IO clients. Only forward the command to the pending sim.
            self._commands_received += 1
            for response in tuple(self.pending.values()):
                if not response.done():
                    response.set_result(command)
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
        async def on_simulator_connect(sid: str, environ: Dict[str, Any], auth: Any = None) -> bool:
            self._connected_simulators.add(sid)
            self._simulator_sessions[sid] = {
                "complete_frames": 0,
                "camera_frames": 0,
                "devkit_commands": 0,
            }
            print(json.dumps({"event": "simulator_connected", "sid": sid}), flush=True)
            return True

        @self.sio.on("disconnect")
        async def on_simulator_disconnect(sid: str, reason: Any = None) -> None:
            self._connected_simulators.discard(sid)
            self._simulator_sessions.pop(sid, None)
            self.pending.pop(sid, None)
            print(json.dumps({"event": "simulator_disconnected", "sid": sid}), flush=True)

        @self.sio.on("Bridge")
        async def on_simulator_frame(sid: str, payload: Any) -> None:
            missing = missing_devkit_fields(payload)
            if missing:
                self._incomplete_frames += 1
                if missing not in self._warned_missing:
                    self._warned_missing.add(missing)
                    print(json.dumps({
                        "event": "incomplete_startup_frame",
                        "sid": sid,
                        "missing_fields": missing,
                        "action": "neutral_controls",
                    }), flush=True)
                await self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid)
                return

            response = asyncio.get_running_loop().create_future()
            self.pending[sid] = response
            try:
                # Do not decode, rewrite, or inspect values. The official Devkit
                # remains responsible for ROS publishing and policy-facing data.
                self._forwarded_frames += 1
                if isinstance(payload, dict) and payload.get("V1 Front Camera Image") is not None:
                    self._camera_frames += 1
                    session = self._simulator_sessions.get(sid)
                    if session is not None:
                        session["camera_frames"] += 1
                session = self._simulator_sessions.get(sid)
                if session is not None:
                    session["complete_frames"] += 1
                if self._forwarded_frames == 1:
                    print(json.dumps({
                        "event": "first_complete_frame_forwarded",
                        "keys": sorted(payload),
                    }), flush=True)
                await self.upstream.emit("Bridge", payload)
                if self._forwarded_frames == 1:
                    print(json.dumps({
                        "event": "upstream_emit_returned",
                    }), flush=True)
                command = await asyncio.wait_for(response, timeout=self.response_timeout_s)
                if isinstance(command, dict):
                    await self.sio.emit("Bridge", data=command, to=sid)
                    session = self._simulator_sessions.get(sid)
                    if session is not None:
                        session["devkit_commands"] += 1
                    return
                raise TypeError(f"official Devkit returned non-dictionary command: {type(command).__name__}")
            except asyncio.TimeoutError:
                self._response_timeouts += 1
                if self._response_timeouts == 1 or self._response_timeouts % 100 == 0:
                    print(json.dumps({
                        "event": "devkit_response_timeout",
                        "count": self._response_timeouts,
                        "sid": sid,
                        "timeout_s": self.response_timeout_s,
                    }), flush=True)
                await self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid)
            except Exception as exc:
                print(json.dumps({
                    "event": "relay_error",
                    "sid": sid,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                await self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid)
            finally:
                if self.pending.get(sid) is response:
                    self.pending.pop(sid, None)

        # AutoDRIVE's Unity client can emit its first Bridge event before it
        # sends the Socket.IO namespace-connect packet. AutoDRIVE's own Race
        # Control Tower handles this Engine.IO/Socket.IO ordering explicitly;
        # without it a TCP/WebSocket session looks connected while no Bridge
        # callback can run.
        original_message_handler = self.sio.eio.handlers["message"]

        async def compat_message(eio_sid: str, data: Any) -> Any:
            await self._ensure_socketio_namespace_for_event(eio_sid, data)
            result = original_message_handler(eio_sid, data)
            if inspect.isawaitable(result):
                return await result
            return result

        self.sio.eio.handlers["message"] = compat_message

    async def _ensure_socketio_namespace_for_event(self, eio_sid: str, data: Any) -> None:
        try:
            packet_class = getattr(self.sio, "packet_class", self._socketio_packet.Packet)
            packet = packet_class(encoded_packet=data)
        except (TypeError, ValueError):
            return
        if packet.packet_type != self._socketio_packet.EVENT:
            return
        namespace = packet.namespace or "/"
        sid_from_eio_sid = getattr(self.sio.manager, "sid_from_eio_sid", None)
        sid = sid_from_eio_sid(eio_sid, namespace) if callable(sid_from_eio_sid) else eio_sid
        if sid is not None and self.sio.manager.is_connected(sid, namespace):
            return
        event = packet.data[0] if isinstance(packet.data, list) and packet.data else "unknown"
        print(json.dumps({
            "event": "implicit_socketio_connect",
            "engineio_sid": eio_sid,
            "namespace": namespace,
            "first_event": event,
        }), flush=True)
        handle_connect = self.sio._handle_connect
        parameters = inspect.signature(handle_connect).parameters
        result = handle_connect(eio_sid, namespace, None) if len(parameters) >= 3 else handle_connect(eio_sid, namespace)
        if inspect.isawaitable(result):
            await result

    async def _release_simulators_on_shutdown(self) -> None:
        """Send neutral controls to connected simulators before relay teardown."""
        simulator_sids = tuple(self._connected_simulators)

        async def send_neutral(sid: str) -> bool:
            try:
                await asyncio.wait_for(
                    self.sio.emit("Bridge", data=NEUTRAL_COMMAND, to=sid),
                    timeout=self.shutdown_neutral_timeout_s,
                )
            except Exception as exc:
                print(json.dumps({
                    "event": "shutdown_neutral_command_failed",
                    "sid": sid,
                    "timeout_s": self.shutdown_neutral_timeout_s,
                    "error": f"{type(exc).__name__}: {exc}",
                }), flush=True)
                return False
            print(json.dumps({
                "event": "shutdown_neutral_command_sent",
                "sid": sid,
                "command": NEUTRAL_COMMAND,
            }), flush=True)
            return True

        # Emit concurrently so shutdown stays bounded by one timeout, even if
        # multiple simulator sockets are connected.
        results = await asyncio.gather(*(send_neutral(sid) for sid in simulator_sids))
        sent = sum(results)
        print(json.dumps({
            "event": "shutdown_neutral_release_complete",
            "attempted": len(simulator_sids),
            "sent": sent,
            "failed": len(simulator_sids) - sent,
            "transport_acknowledged": False,
        }), flush=True)

    async def serve(self, host: str, port: int, devkit_url: str) -> None:
        from aiohttp import web

        last_error: Optional[Exception] = None
        deadline = asyncio.get_running_loop().time() + self.connect_timeout_s
        while asyncio.get_running_loop().time() < deadline:
            try:
                await self.upstream.connect(devkit_url, wait_timeout=min(5.0, self.connect_timeout_s))
                break
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.25)
        else:
            raise TimeoutError(
                f"Could not connect relay to official Devkit at {devkit_url}: {last_error}"
            ) from last_error
        print(json.dumps({"event": "devkit_connected", "devkit_url": devkit_url}), flush=True)

        request_count = 0
        @web.middleware
        async def socketio_request_log(request: Any, handler: Any) -> Any:
            nonlocal request_count
            if request.path.rstrip("/") == "/socket.io" and request_count < 12:
                request_count += 1
                print(json.dumps({
                    "event": "socketio_handshake_request",
                    "method": request.method,
                    "query": request.query_string,
                    "upgrade": request.headers.get("Upgrade", ""),
                    "user_agent": request.headers.get("User-Agent", ""),
                }), flush=True)
            return await handler(request)

        async def health(_request: Any) -> Any:
            return web.json_response({
                "simulator_clients": len(self._connected_simulators),
                "complete_frames": self._forwarded_frames,
                "camera_frames": self._camera_frames,
                "incomplete_frames": self._incomplete_frames,
                "response_timeouts": self._response_timeouts,
                "simulator_sessions": [
                    {"sid": sid, **counters}
                    for sid, counters in sorted(self._simulator_sessions.items())
                ],
            }, headers={"Cache-Control": "no-store"})

        app = web.Application(
            client_max_size=128 * 1024 * 1024,
            middlewares=[socketio_request_log],
        )
        self.sio.attach(app, socketio_path="socket.io")
        app.router.add_get("/healthz", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, int(port))
        await site.start()
        print(json.dumps({"event": "relay_listening", "host": host, "port": int(port)}), flush=True)
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(signum, stopped.set)
            except NotImplementedError:
                pass
        try:
            await stopped.wait()
        finally:
            try:
                await self._release_simulators_on_shutdown()
            finally:
                await self.upstream.disconnect()
                await runner.cleanup()


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
        response_timeout_s=args.response_timeout_s,
        connect_timeout_s=args.connect_timeout_s,
    )
    asyncio.run(relay.serve(args.host, args.port, args.devkit_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
