"""Compare encoder-derived and simulator-reported forward speed on live frames.

Run inside the brain container while one simulator is connected to ``--port``.
This is a local sensor-alignment diagnostic; ground-truth fields are printed
for comparison only and are never passed into the policy observation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np

from src.layer2 import AutoDriveEnv


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument(
        "--action-interval-s",
        type=float,
        default=None,
        help="Use a fixed-step simulator only when it advertises the step protocol; default uses the installed legacy build.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    env = AutoDriveEnv(
        port=args.port,
        map_id=args.map_id,
        headless=True,
        auto_launch=False,
        connect_timeout=args.timeout,
        frame_skip=1,
        action_interval_s=args.action_interval_s,
        observation_profile="official_sensors",
        laps_per_episode=0,
        max_episode_steps=0,
        terminate_on_collision=False,
    )
    rows: list[dict[str, float | int | None]] = []
    try:
        env.reset()
        for index in range(args.steps):
            observation, _, terminated, truncated, info = env.step(
                np.asarray([args.throttle, args.steering], dtype=np.float32)
            )
            snap = env._last_snap
            encoder_speed = float(observation["state"][0]) * 22.88
            row = {
                "step": index + 1,
                "simulated_v_long_mps": _finite(snap.v_long),
                "encoder_v_long_mps": encoder_speed,
                "encoder_left_rad": _finite(snap.encoder_left),
                "encoder_right_rad": _finite(snap.encoder_right),
                "step_duration_s": _finite(info.get("step_duration_s")),
                "throttle_command": _finite(info.get("throttle_command")),
                "pose_x": _finite(snap.position[0]),
                "pose_z": _finite(snap.position[2]),
            }
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)
            if terminated or truncated:
                break
    finally:
        env.close()

    simulator_speeds = [row["simulated_v_long_mps"] for row in rows if row["simulated_v_long_mps"] is not None]
    encoder_speeds = [row["encoder_v_long_mps"] for row in rows if row["encoder_v_long_mps"] is not None]
    result = {
        "samples": len(rows),
        "simulator_forward_speed_median_mps": statistics.median(simulator_speeds) if simulator_speeds else None,
        "encoder_forward_speed_median_mps": statistics.median(encoder_speeds) if encoder_speeds else None,
        "duration_seconds": sum(float(row["step_duration_s"] or 0.0) for row in rows),
        "created_wall_time": time.time(),
    }
    print(json.dumps({"summary": result}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
