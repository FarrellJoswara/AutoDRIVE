"""Phase 5: AutoDRIVE Bridge + FTG (offline-safe; live when sim on :4567).

Observation layout matches contracts v2 (`build_observation_from_f1tenth`) so
Bridge→PPO can reuse the same input width; FTG itself still uses raw LiDAR only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing sibling autodrive.py
_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from .contracts import N_LIDAR_DEFAULT, decode_action, obs_dim
from .ftg import FollowTheGap
from .observation import build_observation_from_f1tenth


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
    prev_action = (0.0, 0.0)
    prev_enc = None
    last_t = None

    @sio.on("connect")
    def connect(sid, environ):
        nonlocal prev_action, prev_enc, last_t
        print("Connected to AutoDRIVE Simulator")
        f1tenth_1.reset_command = 1
        prev_action = (0.0, 0.0)
        prev_enc = None
        last_t = None

    @sio.on("Bridge")
    def bridge(sid, data):
        nonlocal prev_action, prev_enc, last_t
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

        # Build v2 obs from Bridge sensors (lidar + encoders + IMU) for later PPO / logging
        import time as _time

        now = _time.time()
        dt = 0.05 if last_t is None else max(1e-3, now - last_t)
        last_t = now
        obs, speed_mps, prev_enc = build_observation_from_f1tenth(
            f1tenth_1,
            prev_action[0],
            prev_action[1],
            n_lidar=N_LIDAR_DEFAULT,
            prev_encoder_angles=prev_enc,
            dt=dt,
        )
        assert obs.shape[-1] == obs_dim(N_LIDAR_DEFAULT)

        f1tenth_1.throttle_command = float(throttle)
        f1tenth_1.steering_command = float(steering)
        prev_action = (float(throttle), float(steering))
        if verbose:
            print(
                f"FTG cmd throttle={throttle:.2f} steering={steering:.2f} "
                f"obs_dim={obs.shape[-1]} speed~={speed_mps:.2f}"
            )
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
        # Offline: verify v2 obs builder accepts a stub F1TENTH-like object
        class _Stub:
            lidar_range_array = np.full(180, 5.0, dtype=np.float32)
            encoder_angles = np.array([0.0, 0.0], dtype=np.float64)
            angular_velocity = np.array([0.0, 0.0, 0.1], dtype=np.float64)
            linear_acceleration = np.array([0.2, 0.0, 9.8], dtype=np.float64)

        obs, _, _ = build_observation_from_f1tenth(_Stub(), 0.0, 0.0)
        assert obs.shape[-1] == obs_dim(), f"obs dim {obs.shape[-1]} != {obs_dim()}"
        print(f"OK: offline Bridge + Reset API + obs_dim={obs.shape[-1]}")
        return 0
    run_bridge(port=args.port, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
