"""Phase 5: AutoDRIVE Bridge + FTG (offline-safe; live when sim on :4567)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing sibling autodrive.py
_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from .contracts import decode_action
from .ftg import FollowTheGap
from .observation import downsample_lidar


def ensure_reset_api() -> bool:
    """Offline pass: Bridge vehicle API exposes reset_command."""
    import autodrive

    car = autodrive.F1TENTH()
    car.id = "V1"
    car.throttle_command = 0.0
    car.steering_command = 0.0
    car.reset_command = 1
    msg = car.generate_commands(verbose=False)
    return any("Reset" in k for k in msg.keys()) and hasattr(car, "reset_command")


def run_bridge(port: int = 4567, verbose: bool = False) -> None:
    import eventlet
    import socketio
    from flask import Flask
    import autodrive

    f1tenth_1 = autodrive.F1TENTH()
    f1tenth_1.id = "V1"
    f1tenth_1.reset_command = 0
    policy = FollowTheGap()
    sio = socketio.Server()
    app = Flask(__name__)

    @sio.on("connect")
    def connect(sid, environ):
        print("Connected to AutoDRIVE Simulator")
        f1tenth_1.reset_command = 1

    @sio.on("Bridge")
    def bridge(sid, data):
        if not data:
            return
        f1tenth_1.parse_data(data, verbose=verbose)
        # After first frame, clear reset
        f1tenth_1.reset_command = 0
        ranges = f1tenth_1.lidar_range_array
        if ranges is None:
            action = np.array([0, 5], dtype=np.int64)
        else:
            action = policy.act(np.asarray(ranges, dtype=np.float32))
        throttle, steering = decode_action(action)
        f1tenth_1.throttle_command = float(throttle)
        f1tenth_1.steering_command = float(steering)
        if verbose:
            print(f"FTG cmd throttle={throttle:.2f} steering={steering:.2f}")
        try:
            sio.emit("Bridge", data=f1tenth_1.generate_commands(verbose=False))
        except Exception as exc:
            print(exc)

    app = socketio.Middleware(sio, app)
    print(f"AutoDRIVE Bridge FTG listening on 0.0.0.0:{port}")
    eventlet.wsgi.server(eventlet.listen(("", port)), app)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5: AutoDRIVE Bridge FTG")
    parser.add_argument("--check", action="store_true", help="Offline API check only")
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    ok = ensure_reset_api()
    print(f"Reset API present: {ok}")
    if not ok:
        return 1
    if args.check:
        print("OK: offline Bridge + Reset API")
        return 0
    run_bridge(port=args.port, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
