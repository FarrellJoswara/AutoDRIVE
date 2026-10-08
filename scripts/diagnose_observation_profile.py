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
from pathlib import Path
from typing import Any

import numpy as np

from src.layer2 import AutoDriveEnv
from src.layer2.lidar_odometry import ScanMotion, estimate_scan_motion


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
    parser.add_argument("--out", type=Path, default=None)
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
    previous_yaw: float | None = None
    previous_position: tuple[float, float] | None = None
    previous_scan: np.ndarray | None = None
    previous_frame_scan: np.ndarray | None = None
    window_elapsed_s = 0.0
    window_measured_elapsed_s = 0.0
    window_imu_yaw_rad = 0.0
    last_imu_yaw_delta = 0.0
    window_position: tuple[float, float] | None = None
    window_start_yaw: float | None = None
    last_lidar_displacement = ScanMotion()
    last_lidar_unprior_displacement = ScanMotion()
    last_lidar_nominal_dt = 0.0
    last_lidar_measured_dt = 0.0
    last_pose_window_speed: float | None = None
    imu_integrated_forward_speed_mps = 0.0
    total_elapsed_s = 0.0
    try:
        env.reset()
        previous_scan = np.asarray(env._last_snap.lidar_ranges, dtype=np.float32).copy()
        previous_frame_scan = previous_scan.copy()
        previous_position = (float(env._last_snap.position[0]), float(env._last_snap.position[2]))
        window_position = previous_position
        previous_yaw = float(env._last_snap.heading_yaw)
        window_start_yaw = previous_yaw
        for index in range(args.steps):
            clock_before = float(env.racer.simulation_time())
            observation, _, terminated, truncated, info = env.step(
                np.asarray([args.throttle, args.steering], dtype=np.float32)
            )
            clock_after = float(env.racer.simulation_time())
            snap = env._last_snap
            encoder_speed = float(observation["state"][0]) * 22.88
            elapsed_s = float(info.get("step_duration_s", 0.0))
            frame_lidar_only = ScanMotion()
            frame_lidar_imu = ScanMotion()
            if previous_frame_scan is not None and elapsed_s > 0.0:
                frame_lidar_only = estimate_scan_motion(
                    previous_frame_scan,
                    snap.lidar_ranges,
                )
                frame_lidar_imu = estimate_scan_motion(
                    previous_frame_scan,
                    snap.lidar_ranges,
                    yaw_delta_rad=-float(snap.angular_velocity[2]) * elapsed_s,
                )
            previous_frame_scan = np.asarray(snap.lidar_ranges, dtype=np.float32).copy()
            total_elapsed_s += elapsed_s
            imu_integrated_forward_speed_mps += float(snap.linear_acceleration[0]) * elapsed_s
            measured_elapsed_s = max(0.0, clock_after - clock_before)
            window_elapsed_s += elapsed_s
            window_measured_elapsed_s += measured_elapsed_s
            position = (float(snap.position[0]), float(snap.position[2]))
            pose_ground_speed = None
            if previous_position is not None and measured_elapsed_s > 0.0:
                pose_ground_speed = float(
                    np.linalg.norm(np.asarray(position) - np.asarray(previous_position))
                    / measured_elapsed_s
                )
            previous_position = position
            if elapsed_s > 0.0:
                window_imu_yaw_rad += -float(snap.angular_velocity[2]) * elapsed_s
            pose_window_yaw_delta = 0.0
            if (index + 1) % 4 == 0 and previous_scan is not None:
                last_lidar_displacement = estimate_scan_motion(
                    previous_scan,
                    snap.lidar_ranges,
                    yaw_delta_rad=window_imu_yaw_rad,
                )
                # Compare the production estimator (LiDAR only) with an
                # IMU-yaw-constrained fit in the same sensor window. The IMU
                # signal is permitted, but keeping these diagnostics separate
                # shows whether it materially improves scan registration.
                last_lidar_unprior_displacement = estimate_scan_motion(
                    previous_scan,
                    snap.lidar_ranges,
                )
                last_imu_yaw_delta = window_imu_yaw_rad
                last_lidar_nominal_dt = window_elapsed_s
                last_lidar_measured_dt = window_measured_elapsed_s
                if window_position is not None and window_measured_elapsed_s > 0.0:
                    last_pose_window_speed = float(
                        np.linalg.norm(np.asarray(position) - np.asarray(window_position))
                        / window_measured_elapsed_s
                    )
                if window_start_yaw is not None:
                    pose_window_yaw_delta = float(
                        np.arctan2(
                            np.sin(snap.heading_yaw - window_start_yaw),
                            np.cos(snap.heading_yaw - window_start_yaw),
                        )
                    )
                previous_scan = np.asarray(snap.lidar_ranges, dtype=np.float32).copy()
                window_position = position
                window_start_yaw = float(snap.heading_yaw)
                window_elapsed_s = 0.0
                window_measured_elapsed_s = 0.0
                window_imu_yaw_rad = 0.0
            lidar_motion_valid = bool(last_lidar_displacement.valid)
            lidar_v_forward_nominal = (
                last_lidar_displacement.forward_m / last_lidar_nominal_dt
                if lidar_motion_valid and last_lidar_nominal_dt > 0.0 else 0.0
            )
            lidar_v_forward_measured = (
                last_lidar_displacement.forward_m / last_lidar_measured_dt
                if lidar_motion_valid and last_lidar_measured_dt > 0.0 else 0.0
            )
            lidar_yaw_rate_nominal = (
                last_lidar_displacement.yaw_rad / last_lidar_nominal_dt
                if lidar_motion_valid and last_lidar_nominal_dt > 0.0 else 0.0
            )
            lidar_unprior_valid = bool(last_lidar_unprior_displacement.valid)
            yaw_delta = 0.0
            if previous_yaw is not None and elapsed_s > 0.0:
                yaw_delta = float(
                    np.arctan2(
                        np.sin(snap.heading_yaw - previous_yaw),
                        np.cos(snap.heading_yaw - previous_yaw),
                    )
                ) / elapsed_s
            previous_yaw = float(snap.heading_yaw)
            row = {
                "step": index + 1,
                "elapsed_simulated_seconds": total_elapsed_s,
                "simulated_v_long_mps": _finite(snap.v_long),
                "encoder_v_long_mps": encoder_speed,
                "imu_forward_acceleration_mps2": _finite(snap.linear_acceleration[0]),
                "imu_integrated_v_long_mps": _finite(imu_integrated_forward_speed_mps),
                "lidar_v_long_nominal_mps": _finite(lidar_v_forward_nominal),
                "lidar_v_long_measured_mps": _finite(lidar_v_forward_measured),
                "lidar_only_valid": int(lidar_unprior_valid),
                "lidar_only_v_long_mps": _finite(
                    last_lidar_unprior_displacement.forward_m / last_lidar_nominal_dt
                    if lidar_unprior_valid and last_lidar_nominal_dt > 0.0 else 0.0
                ),
                "lidar_only_rmse_m": _finite(last_lidar_unprior_displacement.rmse_m),
                "lidar_only_inlier_fraction": _finite(last_lidar_unprior_displacement.inlier_fraction),
                "lidar_frame_valid": int(frame_lidar_only.valid),
                "lidar_frame_forward_speed_mps": _finite(
                    frame_lidar_only.forward_m / elapsed_s
                    if frame_lidar_only.valid and elapsed_s > 0.0 else 0.0
                ),
                "lidar_frame_imu_valid": int(frame_lidar_imu.valid),
                "lidar_frame_imu_forward_speed_mps": _finite(
                    frame_lidar_imu.forward_m / elapsed_s
                    if frame_lidar_imu.valid and elapsed_s > 0.0 else 0.0
                ),
                "lidar_v_lat_nominal_mps": _finite(
                    last_lidar_displacement.lateral_m / last_lidar_nominal_dt
                    if lidar_motion_valid and last_lidar_nominal_dt > 0.0 else 0.0
                ),
                "lidar_yaw_rate_nominal_rad_s": _finite(lidar_yaw_rate_nominal),
                "pose_window_ground_speed_mps": _finite(last_pose_window_speed),
                "imu_yaw_window_rad": _finite(last_imu_yaw_delta),
                "pose_yaw_window_rad": _finite(pose_window_yaw_delta),
                "pose_yaw_rate_rad_s": _finite(yaw_delta),
                "pose_ground_speed_mps": _finite(pose_ground_speed),
                "lidar_odometry_valid": int(lidar_motion_valid),
                "lidar_odometry_rmse_m": _finite(last_lidar_displacement.rmse_m),
                "lidar_odometry_inlier_fraction": _finite(last_lidar_displacement.inlier_fraction),
                "encoder_left_rad": _finite(snap.encoder_left),
                "encoder_right_rad": _finite(snap.encoder_right),
                "step_duration_s": _finite(elapsed_s),
                "measured_elapsed_s": _finite(measured_elapsed_s),
                "racer_step_latency_ms": _finite(getattr(env.racer, "step_latency_ms", None)),
                "lidar_scan_rate_hz": _finite(snap.lidar_scan_rate),
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
    imu_speed_errors = [
        abs(float(row["imu_integrated_v_long_mps"]) - float(row["simulated_v_long_mps"]))
        for row in rows
        if row["imu_integrated_v_long_mps"] is not None and row["simulated_v_long_mps"] is not None
    ]
    encoder_speed_errors = [
        abs(float(row["encoder_v_long_mps"]) - float(row["simulated_v_long_mps"]))
        for row in rows
        if row["encoder_v_long_mps"] is not None and row["simulated_v_long_mps"] is not None
    ]
    error_bins: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        second = int(float(row["elapsed_simulated_seconds"] or 0.0))
        bucket = error_bins.setdefault(second, {"encoder": [], "imu_integrated": []})
        if row["simulated_v_long_mps"] is not None:
            if row["encoder_v_long_mps"] is not None:
                bucket["encoder"].append(
                    abs(float(row["encoder_v_long_mps"]) - float(row["simulated_v_long_mps"]))
                )
            if row["imu_integrated_v_long_mps"] is not None:
                bucket["imu_integrated"].append(
                    abs(float(row["imu_integrated_v_long_mps"]) - float(row["simulated_v_long_mps"]))
                )
    result = {
        "samples": len(rows),
        "simulator_forward_speed_median_mps": statistics.median(simulator_speeds) if simulator_speeds else None,
        "encoder_forward_speed_median_mps": statistics.median(encoder_speeds) if encoder_speeds else None,
        "encoder_speed_median_absolute_error_mps": statistics.median(encoder_speed_errors) if encoder_speed_errors else None,
        "imu_integrated_speed_median_absolute_error_mps": statistics.median(imu_speed_errors) if imu_speed_errors else None,
        "speed_error_by_elapsed_second": {
            str(second): {
                "encoder_median_absolute_error_mps": statistics.median(values["encoder"]) if values["encoder"] else None,
                "imu_integrated_median_absolute_error_mps": statistics.median(values["imu_integrated"]) if values["imu_integrated"] else None,
                "samples": len(values["encoder"]),
            }
            for second, values in sorted(error_bins.items())
        },
        "lidar_odometry_valid_fraction": sum(int(row["lidar_odometry_valid"]) for row in rows) / len(rows) if rows else 0.0,
        "lidar_only_valid_fraction": sum(int(row["lidar_only_valid"]) for row in rows) / len(rows) if rows else 0.0,
        "lidar_frame_valid_fraction": sum(int(row["lidar_frame_valid"]) for row in rows) / len(rows) if rows else 0.0,
        "lidar_frame_imu_valid_fraction": sum(int(row["lidar_frame_imu_valid"]) for row in rows) / len(rows) if rows else 0.0,
        "duration_seconds": sum(float(row["step_duration_s"] or 0.0) for row in rows),
        "created_wall_time": time.time(),
    }
    report = {
        "map_id": args.map_id,
        "action_interval_s": args.action_interval_s,
        "frame_skip": 1,
        "policy_observation_profile": "official_sensors",
        "simulator_pose_and_velocity_used_for_diagnostics_only": True,
        "summary": result,
        "samples": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved diagnostic report: {args.out}", flush=True)
    print(json.dumps({"summary": result}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
