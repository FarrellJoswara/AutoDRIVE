"""Inspect AutoDRIVE Bridge event shapes without modifying the official Devkit.

Run this in place of the official Devkit on a disposable Docker network while
the official simulator starts. It reports keys and value types only, then
replies with neutral actuator commands so the simulator can continue sending
frames. It is diagnostic instrumentation, not a race policy or score runner.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import socketio
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler


def packet_summary(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"payload_type": type(data).__name__}
    return {
        "keys": sorted(str(key) for key in data),
        "value_types": {str(key): type(value).__name__ for key, value in data.items()},
        "field_count": len(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--max-packets", type=int, default=12)
    args = parser.parse_args()
    if args.max_packets < 1:
        parser.error("--max-packets must be positive")

    sio = socketio.Server(async_mode="gevent", cors_allowed_origins="*")
    state = {"packets": 0}

    @sio.on("connect")
    def on_connect(sid: str, environ: dict[str, Any]) -> None:
        print(json.dumps({"event": "connected", "sid": sid}), flush=True)

    @sio.on("Bridge")
    def on_bridge(sid: str, data: Any) -> None:
        state["packets"] += 1
        print(json.dumps({
            "event": "Bridge",
            "index": state["packets"],
            **packet_summary(data),
        }), flush=True)
        sio.emit("Bridge", data={
            "V1 Throttle": "0.0000",
            "V1 Steering": "0.0000",
            "V1 Reset": "False",
        }, to=sid)
        if state["packets"] >= args.max_packets:
            server.stop(timeout=1)

    app = socketio.WSGIApp(sio)
    server = pywsgi.WSGIServer((args.host, args.port), app, handler_class=WebSocketHandler)
    print(json.dumps({"event": "listening", "host": args.host, "port": args.port}), flush=True)
    server.serve_forever()
    print(json.dumps({"event": "complete", "packets": state["packets"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
