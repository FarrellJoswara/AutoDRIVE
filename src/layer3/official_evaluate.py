"""Evaluate a policy in the official IROS 2026 AutoDRIVE ROS 2 race flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import numpy as np
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


OFFICIAL_COMPETITION_ROUTE_SHA256 = "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"


def collision_penalty_seconds(collisions: int, seconds_per_collision: float = 10.0) -> float:
    """Official increasing penalty: 10 s, then 20 s, then 30 s, and so on."""
    count = max(0, int(collisions))
    return float(seconds_per_collision) * count * (count + 1) / 2.0


def summarize_attempt(
    *,
    lap_times_s: List[float],
    race_collisions: int,
    race_laps_required: int = 10,
    disqualification_limit: int = 10,
) -> Dict[str, Any]:
    valid_times = [float(value) for value in lap_times_s if math.isfinite(float(value)) and float(value) > 0]
    race_time = (
        sum(valid_times)
        if valid_times and len(valid_times) == len(lap_times_s)
        else None
    )
    disqualified = int(race_collisions) > int(disqualification_limit)
    penalty = collision_penalty_seconds(race_collisions)
    required_laps = max(1, int(race_laps_required))
    complete = len(valid_times) == required_laps and len(valid_times) == len(lap_times_s)
    return {
        "race_laps_required": required_laps,
        "race_laps_completed": len(lap_times_s),
        "race_lap_times_s": list(lap_times_s),
        "race_time_s": race_time,
        "race_collisions": int(race_collisions),
        "collision_penalty_s": penalty,
        "adjusted_race_time_s": (
            None if disqualified or not complete or race_time is None
            else race_time + penalty
        ),
        "disqualified": disqualified,
        "complete": complete,
        "best_lap_s": min(valid_times) if valid_times else None,
        "mean_lap_s": statistics.fmean(valid_times) if valid_times else None,
    }


def verify_vehicle_motion(
    positions: List[List[float]],
    elapsed_s: float,
    *,
    check_after_s: float = 3.0,
    minimum_displacement_m: float = 0.25,
) -> Dict[str, Any]:
    """Use evaluator-only IPS to reject stationary simulator attempts.

    IPS is restricted and deliberately used only as a test-validity diagnostic;
    it is never included in policy observations or actions.
    """
    common = {
        "source": "restricted_ips_diagnostic_only",
        "available": bool(positions),
        "elapsed_s": float(elapsed_s),
        "check_after_s": float(check_after_s),
        "minimum_displacement_m": float(minimum_displacement_m),
    }
    if not positions:
        return {**common, "verified": None, "max_displacement_m": None}
    origin = positions[0]
    max_displacement = max(
        math.sqrt(sum((float(point[i]) - float(origin[i])) ** 2 for i in range(3)))
        for point in positions
    )
    moved = max_displacement >= minimum_displacement_m
    verified = True if moved else (False if elapsed_s >= check_after_s else None)
    return {
        **common,
        "verified": verified,
        "max_displacement_m": max_displacement,
    }


def summarize_evaluator_positions(
    positions: List[List[float]], *, max_points: int = 600
) -> Dict[str, Any]:
    """Summarize and bound an evaluator-only official ROS IPS route trace.

    Position is restricted diagnostic data. This trace is written only to the
    evaluation report; it is never passed to the policy or included in its
    observation/action diagnostics. Official ROS IPS uses the AutoDRIVE frame
    (x/y horizontal, z vertical); it is not the Unity-swizzled TelemetrySnapshot
    position.
    """
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    valid = [
        [float(value) for value in point]
        for point in positions
        if len(point) >= 2 and all(math.isfinite(float(value)) for value in point)
    ]
    if not valid:
        return {
            "source": "restricted_ips_evaluator_only",
            "positions_collected": 0,
            "path_length_m": None,
            "net_displacement_m": None,
            "path_efficiency": None,
            "max_distance_from_start_m": None,
            "max_3d_displacement_from_start_m": None,
            "vertical_min_m": None,
            "vertical_max_m": None,
            "vertical_span_m": None,
            "sampled_xy_m": [],
            "sampled_xyz_m": [],
        }

    horizontal = [
        (point[0], point[1])
        for point in valid
    ]
    path_length = sum(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(horizontal, horizontal[1:])
    )
    origin = horizontal[0]
    net_displacement = math.hypot(
        horizontal[-1][0] - origin[0], horizontal[-1][1] - origin[1]
    )
    max_distance = max(
        math.hypot(point[0] - origin[0], point[1] - origin[1])
        for point in horizontal
    )
    max_3d_displacement = max(
        math.sqrt(sum((current[axis] - valid[0][axis]) ** 2 for axis in range(3)))
        for current in valid
    ) if all(len(point) >= 3 for point in valid) else None
    vertical_values = [point[2] for point in valid] if all(
        len(point) >= 3 for point in valid
    ) else []
    if len(horizontal) <= max_points:
        sample_indices = list(range(len(horizontal)))
    else:
        sample_indices = sorted({
            round(index * (len(horizontal) - 1) / (max_points - 1))
            for index in range(max_points)
        })
    return {
        "source": "restricted_ips_evaluator_only",
        "positions_collected": len(horizontal),
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "path_efficiency": net_displacement / path_length if path_length > 0 else None,
        "max_distance_from_start_m": max_distance,
        "max_3d_displacement_from_start_m": max_3d_displacement,
        "vertical_min_m": min(vertical_values) if vertical_values else None,
        "vertical_max_m": max(vertical_values) if vertical_values else None,
        "vertical_span_m": (
            max(vertical_values) - min(vertical_values) if vertical_values else None
        ),
        "sampled_xy_m": [list(horizontal[index]) for index in sample_indices],
        "sampled_xyz_m": [list(valid[index][:3]) for index in sample_indices]
        if all(len(point) >= 3 for point in valid) else [],
    }


def _distribution(values: List[float]) -> Dict[str, Optional[float]]:
    """Summarize a bounded stream of policy or observation values."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    array = [float(value) for value in values]
    return {
        "count": len(array),
        "min": min(array),
        "max": max(array),
        "mean": statistics.fmean(array),
        "std": statistics.pstdev(array),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_monitor_identity(
    *, output_path: Optional[Path], frontier_path: Path, expected_run_id: str,
) -> Dict[str, Any]:
    """Bind monitor-only route telemetry to the exact official native scene/run.

    The route tracker is diagnostic only. This identity record lets downstream
    validators distinguish an observed projection from a report that merely
    names a route; it does not participate in scoring or stop behavior.
    """
    if output_path is None:
        raise ValueError("route monitor requires a run-local evaluation output path")
    route = Path(frontier_path).resolve()
    expected_route = (
        Path(__file__).resolve().parents[2] / "competition" / "iros2026"
        / "tracks" / "competition" / "official_centerline.csv"
    ).resolve()
    if route != expected_route or not route.is_file():
        raise ValueError("route monitor accepts only the fixed official competition centerline")
    route_sha256 = _sha256(route)
    if route_sha256 != OFFICIAL_COMPETITION_ROUTE_SHA256:
        raise ValueError("official competition centerline bytes do not match the verified route")
    manifest_path = Path(output_path).resolve().parent / "windows_simulator_processes.json"
    if not manifest_path.is_file():
        raise ValueError(f"route monitor requires the run-local simulator identity manifest: {manifest_path}")
    # Windows PowerShell writes run-local process manifests with a UTF-8 BOM.
    # Accept that encoding while still hashing the original bytes below.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != expected_run_id:
        raise ValueError("simulator identity manifest does not match the evaluation run")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError("route monitor requires exactly one run-local simulator process identity")
    record = records[0]
    pid = record.get("pid")
    port = record.get("port")
    executable = record.get("executable_path")
    created = record.get("process_creation_utc")
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            or isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535)
            or not isinstance(executable, str) or not executable.strip()
            or not isinstance(created, str) or not created.strip()
            or record.get("batch_mode") is not False):
        raise ValueError("simulator process identity manifest is incomplete or invalid")
    from src.layer3.official_tracks import verify_windows_scene
    scene = verify_windows_scene(route, executable=executable)
    return {
        "official_scene_sha256": scene["scene_sha256"],
        "official_scene_path": scene["scene_path"],
        "route_sha256": route_sha256,
        "route_path": str(route),
        "track_profile": scene["track_profile"],
        "route_monitor_only": True,
        "simulator_identity": {
            "pid": pid,
            "process_creation_utc": created,
            "executable_path": executable,
            "port": port,
            "manifest_sha256": _sha256(manifest_path),
        },
    }


def _camera_topic_age_s(env: Any) -> Optional[float]:
    """Return camera receive age from the official transport when available."""
    racer = getattr(env, "racer", None)
    status_getter = getattr(racer, "transport_status", None)
    if not callable(status_getter):
        return None
    try:
        status = status_getter()
        age = (status.get("topic_age_s") or {}).get("camera")
        age = float(age)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return age if math.isfinite(age) and age >= 0.0 else None


def _valid_camera_frame(observation: Dict[str, Any]) -> Optional[np.ndarray]:
    frame = observation.get("camera")
    if frame is None:
        return None
    try:
        array = np.asarray(frame)
    except (TypeError, ValueError):
        return None
    if array.ndim != 3:
        return None
    if array.shape[-1] == 3 or array.shape[0] == 3:
        return array
    return None


def _frontier_fraction(
    progress_m: Optional[float], route_length_m: Optional[float], *, target_laps: int = 10
) -> Optional[float]:
    if progress_m is None or route_length_m is None or route_length_m <= 0.0:
        return None
    return min(1.0, max(0.0, float(progress_m) / (float(route_length_m) * max(1, int(target_laps)))))


def _policy_observation_trace(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact diagnostics from only the observation given to the policy."""
    lidar = observation["lidar"]
    if getattr(lidar, "ndim", 1) > 1:
        lidar = lidar[-1]
    lidar_values = [float(value) for value in lidar]
    state_values = [float(value) for value in observation["state"]]
    left_end = max(1, len(lidar_values) // 3)
    right_start = min(len(lidar_values) - 1, (2 * len(lidar_values)) // 3)
    return {
        "observation_state": state_values,
        "lidar_min_left": min(lidar_values[:left_end]),
        "lidar_min_front": min(lidar_values[left_end:right_start]),
        "lidar_min_right": min(lidar_values[right_start:]),
    }


def _simulator_alignment_trace(env: Any, info: Dict[str, Any]) -> Dict[str, Any]:
    """Return evaluator-only IPS/LiDAR pairing for map registration diagnostics.

    These restricted simulator fields are deliberately kept outside the policy
    observation and are emitted only when the caller requested a trace.
    """
    result: Dict[str, Any] = {}
    position = info.get("race_position")
    if position is not None:
        result["race_position"] = [float(value) for value in position]
    snap = getattr(env, "_last_snap", None)
    if snap is not None:
        ranges = getattr(snap, "lidar_ranges", None)
        if ranges is not None:
            result["simulator_lidar_ranges"] = [float(value) for value in ranges]
        heading = getattr(snap, "heading_yaw", None)
        if heading is not None:
            result["simulator_heading_yaw_rad"] = float(heading)
    return result


def _official_xy_position(position: Any) -> Optional[List[float]]:
    """Extract official IPS x/y, ignoring simulator height in 3D poses."""
    if position is None or len(position) < 2:
        return None
    return [float(position[0]), float(position[1])]


def action_observation_diagnostics(
    *,
    policy_throttle: List[float],
    policy_steering: List[float],
    applied_throttle: List[float],
    applied_steering: List[float],
    observation_states: List[List[float]],
    normalized_lidar_minima: List[float],
) -> Dict[str, Any]:
    """Report allowed policy I/O only; excludes pose, laps, and collisions."""
    state_names = (
        "forward_speed", "sideways_speed", "yaw_rate", "forward_acceleration",
        "sideways_acceleration", "throttle_feedback", "steering_feedback",
        "previous_throttle", "previous_steering",
    )
    if observation_states:
        states = list(zip(*observation_states))
        state_summary = {
            name: _distribution(list(channel))
            for name, channel in zip(state_names, states)
        }
    else:
        state_summary = {}
    return {
        "policy_throttle": _distribution(policy_throttle),
        "policy_steering": _distribution(policy_steering),
        "applied_throttle": _distribution(applied_throttle),
        "applied_steering": _distribution(applied_steering),
        "policy_reverse_fraction": (
            sum(value < 0.0 for value in policy_throttle) / len(policy_throttle)
            if policy_throttle else None
        ),
        "observation_state": state_summary,
        "normalized_lidar_minimum": _distribution(normalized_lidar_minima),
    }


def evaluate_attempt(
    model: Any,
    env: Any,
    *,
    wall_timeout_s: float,
    max_steps: int,
    attempt_index: int,
    race_laps_required: int = 10,
    frontier_progress_target_laps: Optional[int] = None,
    noncompetitive: bool = False,
    trace_steps: int = 0,
    motion_check_after_s: float = 3.0,
    motion_minimum_displacement_m: float = 0.25,
    motion_stall_after_s: float = 30.0,
    lap_progress_timeout_s: float = 90.0,
    lap_progress_wall_timeout_s: float = 90.0,
    frontier_tracker: Any = None,
    frontier_stall_after_s: float = 0.0,
    route_monitor_only: bool = False,
    camera_preview_writer: Any = None,
    camera_freshness_limit_s: float = 0.5,
    camera_preview_interval_s: float = 1.0,
    disqualification_limit: int = 10,
    on_step: Any = None,
) -> Dict[str, Any]:
    if frontier_progress_target_laps is None:
        # Normal competitive evaluations retain their historical scored-lap
        # denominator. Diagnostic evaluations may include a warmup traversal
        # in the route tracker's accumulated distance.
        frontier_progress_target_laps = race_laps_required
    frontier_progress_target_laps = max(1, int(frontier_progress_target_laps))
    obs, info = env.reset()
    started = time.monotonic()
    laps_reported = 0
    lap_progress_count = len(info.get("warmup_lap_times_s", [])) + len(info.get("race_lap_times_s", []))
    last_lap_progress_elapsed_s = 0.0
    last_lap_progress_wall_s = started
    collisions_reported = 0
    steps = 0
    control_intervals_s: List[float] = []
    scan_rates_hz: List[float] = []
    policy_throttle: List[float] = []
    policy_steering: List[float] = []
    applied_throttle: List[float] = []
    applied_steering: List[float] = []
    policy_inference_latencies_s: List[float] = []
    observation_states: List[List[float]] = []
    normalized_lidar_minima: List[float] = []
    camera_ages_s: List[float] = []
    camera_sample_count = 0
    camera_fresh_sample_count = 0
    camera_stale_sample_count = 0
    camera_missing_sample_count = 0
    camera_freshness_unknown_count = 0
    last_camera_preview_wall_s = 0.0
    action_trace: List[Dict[str, Any]] = []
    initial_position = info.get("race_position")
    position_trace: List[List[float]] = (
        [[float(value) for value in initial_position]]
        if initial_position is not None else []
    )
    control_elapsed_s = 0.0
    last_motion_position = initial_position
    last_motion_elapsed_s = 0.0
    last_frontier_elapsed_s = 0.0
    frontier_progress_m = 0.0
    frontier_projection_valid: Optional[bool] = None
    frontier_route_length_m = (
        float(getattr(frontier_tracker, "length_m"))
        if frontier_tracker is not None and getattr(frontier_tracker, "length_m", None) is not None
        else None
    )
    frontier_lateral_offset_m: Optional[float] = None
    frontier_state: Dict[str, Any] = {}
    diagnostic_frontier_required = frontier_stall_after_s > 0
    frontier_telemetry_invalid = False
    route_projection_sample_count = 0
    route_projection_valid_sample_count = 0
    route_projection_invalid_sample_count = 0
    if diagnostic_frontier_required and (frontier_tracker is None or initial_position is None):
        frontier_telemetry_invalid = True
    if route_monitor_only and (frontier_tracker is None or initial_position is None):
        route_projection_sample_count += 1
        route_projection_invalid_sample_count += 1
    if frontier_tracker is not None and initial_position is not None:
        try:
            initial_xy = _official_xy_position(initial_position)
            if initial_xy is None:
                raise ValueError("official IPS pose is missing x/y coordinates")
            frontier_state = frontier_tracker.reset(
                initial_xy[0], initial_xy[1], now=0.0
            )
            frontier_progress_m = float(frontier_state["progress_m"])
            frontier_projection_valid = bool(frontier_state.get("current_projection_valid", True))
            if route_monitor_only:
                route_projection_sample_count += 1
                route_projection_valid_sample_count += int(frontier_projection_valid)
                route_projection_invalid_sample_count += int(not frontier_projection_valid)
            frontier_lateral_offset_m = frontier_tracker.lateral_distance(
                initial_xy[0], initial_xy[1]
            )
        except (ValueError, TypeError, IndexError):
            if diagnostic_frontier_required:
                frontier_telemetry_invalid = True
            else:
                if route_monitor_only:
                    route_projection_sample_count += 1
                    route_projection_invalid_sample_count += 1
                    frontier_projection_valid = False
                frontier_tracker = None
    motion_ever_verified = False
    motion_status = verify_vehicle_motion(
        position_trace,
        control_elapsed_s,
        check_after_s=motion_check_after_s,
        minimum_displacement_m=motion_minimum_displacement_m,
    )
    while not bool(info.get("race_complete", False)):
        if time.monotonic() - started >= wall_timeout_s:
            stop_reason = "wall_timeout"
            break
        if steps >= max_steps:
            stop_reason = "step_guard"
            break
        observation_states.append([float(v) for v in obs["state"]])
        normalized_lidar_minima.append(float(np.min(obs["lidar"])))
        trace_observation = (
            _policy_observation_trace(obs)
            if len(action_trace) < max(0, int(trace_steps))
            else {}
        )
        camera_frame = _valid_camera_frame(obs)
        camera_age_s = _camera_topic_age_s(env) if "camera" in obs else None
        camera_fresh: Optional[bool] = None
        if "camera" in obs:
            if camera_frame is None:
                camera_missing_sample_count += 1
            else:
                camera_sample_count += 1
                if camera_age_s is None:
                    camera_freshness_unknown_count += 1
                else:
                    camera_ages_s.append(camera_age_s)
                    camera_fresh = camera_age_s <= float(camera_freshness_limit_s)
                    if camera_fresh:
                        camera_fresh_sample_count += 1
                    else:
                        camera_stale_sample_count += 1
                now_wall = time.monotonic()
                if (
                    camera_preview_writer is not None
                    and now_wall - last_camera_preview_wall_s >= camera_preview_interval_s
                ):
                    # PreviewWriter copies the frame and writes it on one
                    # bounded background thread, keeping image I/O off-policy.
                    if camera_preview_writer.submit([camera_frame]):
                        last_camera_preview_wall_s = now_wall
        inference_started = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        inference_latency_s = max(0.0, time.perf_counter() - inference_started)
        policy_inference_latencies_s.append(inference_latency_s)
        command = [float(v) for v in action.reshape(-1)]
        debug = getattr(model, "last_debug", {})
        policy_throttle.append(command[0])
        policy_steering.append(command[1])
        obs, _, terminated, _, info = env.step(action)
        if len(action_trace) < max(0, int(trace_steps)):
            next_observation_trace = _policy_observation_trace(obs)
            action_trace.append({
                "step": steps,
                **trace_observation,
                **_simulator_alignment_trace(env, info),
                "policy_throttle": command[0],
                "policy_steering": command[1],
                "applied_throttle": float(info.get("throttle_command", 0.0)),
                "applied_steering": float(info.get("steering_command", 0.0)),
                "policy_inference_latency_s": inference_latency_s,
                "camera_age_s": camera_age_s,
                "camera_fresh": camera_fresh,
                "next_observation_state": next_observation_trace["observation_state"],
                "next_lidar_min_left": next_observation_trace["lidar_min_left"],
                "next_lidar_min_front": next_observation_trace["lidar_min_front"],
                "next_lidar_min_right": next_observation_trace["lidar_min_right"],
                "evaluation_collision_event": bool(info.get("collision_event", False)),
                "evaluation_race_collisions": int(info.get("race_collisions", 0)),
                **{key: float(value) for key, value in debug.items()},
            })
        applied_throttle.append(float(info.get("throttle_command", 0.0)))
        applied_steering.append(float(info.get("steering_command", 0.0)))
        steps += 1
        interval = float(info.get("control_interval_s", 0.0))
        scan_rate = float(info.get("lidar_scan_rate_hz", 0.0))
        if math.isfinite(interval) and interval > 0:
            control_intervals_s.append(interval)
            control_elapsed_s += interval
        if math.isfinite(scan_rate) and scan_rate > 0:
            scan_rates_hz.append(scan_rate)
        current_position = info.get("race_position")
        if current_position is not None:
            current_position = [float(value) for value in current_position]
            position_trace.append(current_position)
            if last_motion_position is None:
                last_motion_position = current_position
            displacement = math.sqrt(sum(
                (current_position[i] - float(last_motion_position[i])) ** 2
                for i in range(min(3, len(current_position), len(last_motion_position)))
            ))
            if displacement >= motion_minimum_displacement_m:
                last_motion_position = current_position
                last_motion_elapsed_s = control_elapsed_s
            if frontier_tracker is not None:
                try:
                    current_xy = _official_xy_position(current_position)
                    if current_xy is None:
                        raise ValueError("official IPS pose is missing x/y coordinates")
                    progress = frontier_tracker.update(
                        current_xy[0], current_xy[1], now=control_elapsed_s
                    )
                    frontier_state = progress
                    frontier_progress_m = float(progress.get("progress_m") or frontier_progress_m)
                    frontier_projection_valid = bool(progress.get("current_projection_valid", False))
                    if route_monitor_only:
                        route_projection_sample_count += 1
                        route_projection_valid_sample_count += int(frontier_projection_valid)
                        route_projection_invalid_sample_count += int(not frontier_projection_valid)
                    frontier_lateral_offset_m = frontier_tracker.lateral_distance(
                        current_xy[0], current_xy[1]
                    )
                    if float(progress.get("advanced_m") or 0.0) >= 0.04:
                        last_frontier_elapsed_s = control_elapsed_s
                    if diagnostic_frontier_required and not progress.get("current_projection_valid", False):
                        frontier_telemetry_invalid = True
                except (ValueError, TypeError, IndexError):
                    if diagnostic_frontier_required:
                        frontier_telemetry_invalid = True
                        frontier_projection_valid = False
                    else:
                        if route_monitor_only:
                            route_projection_sample_count += 1
                            route_projection_invalid_sample_count += 1
                            frontier_projection_valid = False
                        frontier_tracker = None
        elif diagnostic_frontier_required:
            frontier_telemetry_invalid = True
            frontier_projection_valid = False
        elif route_monitor_only:
            route_projection_sample_count += 1
            route_projection_invalid_sample_count += 1
            frontier_projection_valid = False
        if len(action_trace) > 0 and action_trace[-1].get("step") == steps - 1:
            action_trace[-1].update({
                "frontier_progress_m": frontier_progress_m if frontier_tracker is not None else None,
                "frontier_progress_fraction": _frontier_fraction(
                    frontier_progress_m if frontier_tracker is not None else None,
                    frontier_route_length_m,
                    target_laps=frontier_progress_target_laps,
                ),
                "frontier_projection_valid": frontier_projection_valid,
                "route_projection_valid": frontier_projection_valid,
                "frontier_lateral_offset_m": frontier_lateral_offset_m,
            })
        if on_step is not None:
            callback_info = dict(info)
            callback_info.update({
                "frontier_line": frontier_state.get("line"),
                "frontier_progress_m": frontier_progress_m if frontier_tracker is not None else None,
                "frontier_progress_fraction": _frontier_fraction(
                    frontier_progress_m if frontier_tracker is not None else None,
                    frontier_route_length_m,
                    target_laps=frontier_progress_target_laps,
                ),
                "frontier_projection_valid": frontier_projection_valid,
                "route_projection_valid": frontier_projection_valid,
                "frontier_lateral_offset_m": frontier_lateral_offset_m,
                "camera_age_s": camera_age_s,
                "camera_fresh": camera_fresh,
            })
            on_step(steps, obs, callback_info)
        motion_status = verify_vehicle_motion(
            position_trace,
            time.monotonic() - started,
            check_after_s=motion_check_after_s,
            minimum_displacement_m=motion_minimum_displacement_m,
        )
        motion_ever_verified = motion_ever_verified or motion_status["verified"] is True

        lap_times = list(info.get("race_lap_times_s", []))
        while laps_reported < len(lap_times):
            laps_reported += 1
            print(
                f"attempt {attempt_index}: race lap {laps_reported}/{race_laps_required} "
                f"{float(lap_times[laps_reported - 1]):.3f}s",
                flush=True,
            )
        current_lap_progress_count = len(info.get("warmup_lap_times_s", [])) + len(lap_times)
        if current_lap_progress_count > lap_progress_count:
            lap_progress_count = current_lap_progress_count
            last_lap_progress_elapsed_s = control_elapsed_s
            last_lap_progress_wall_s = time.monotonic()
        collisions = int(info.get("race_collisions", 0))
        while collisions_reported < collisions:
            collisions_reported += 1
            print(
                f"attempt {attempt_index}: race collision {collisions_reported}; "
                f"penalty now {collision_penalty_seconds(collisions_reported):.0f}s",
                flush=True,
            )
        if bool(info.get("collision_event")):
            stop_reason = "collision_detected"
            break
        if frontier_telemetry_invalid:
            stop_reason = "frontier_telemetry_invalid"
            motion_status = {
                **motion_status,
                "frontier_telemetry_invalid": True,
                "frontier_progress_m": frontier_progress_m,
            }
            break
        if collisions > int(disqualification_limit):
            stop_reason = "collision_disqualification_limit"
            break
        if terminated:
            stop_reason = (
                "race_complete"
                if bool(info.get("race_complete", False))
                else "terminated"
            )
            break
        if motion_status["verified"] is False:
            stop_reason = "vehicle_motion_not_verified"
            break
        seconds_since_lap_progress_wall = time.monotonic() - last_lap_progress_wall_s
        if (
            control_elapsed_s - last_lap_progress_elapsed_s >= lap_progress_timeout_s
            or seconds_since_lap_progress_wall >= lap_progress_wall_timeout_s
        ):
            stop_reason = "lap_progress_stalled"
            motion_status = {
                **motion_status,
                "lap_progress_stalled": True,
                "lap_progress_timeout_s": float(lap_progress_timeout_s),
                "seconds_since_lap_progress": control_elapsed_s - last_lap_progress_elapsed_s,
                "seconds_since_lap_progress_wall": seconds_since_lap_progress_wall,
                "completed_laps_since_start": lap_progress_count,
            }
            break
        if (
            not diagnostic_frontier_required
            and motion_ever_verified
            and control_elapsed_s - last_motion_elapsed_s >= motion_stall_after_s
        ):
            stop_reason = "vehicle_motion_stalled"
            motion_status = {
                **motion_status,
                "stalled": True,
                "stall_after_s": float(motion_stall_after_s),
                "seconds_since_motion": control_elapsed_s - last_motion_elapsed_s,
            }
            break
        if (
            frontier_tracker is not None
            and frontier_stall_after_s > 0
            and control_elapsed_s - last_frontier_elapsed_s >= frontier_stall_after_s
        ):
            stop_reason = "frontier_progress_stalled"
            motion_status = {
                **motion_status,
                "frontier_progress_stalled": True,
                "frontier_stall_after_s": float(frontier_stall_after_s),
                "seconds_since_frontier_progress": control_elapsed_s - last_frontier_elapsed_s,
                "frontier_progress_m": frontier_progress_m,
            }
            break
    else:
        stop_reason = "race_complete"

    result = summarize_attempt(
        lap_times_s=list(info.get("race_lap_times_s", [])),
        race_collisions=int(info.get("race_collisions", 0)),
        race_laps_required=race_laps_required,
    )
    if stop_reason == "collision_detected":
        score_status = "collision_detected"
    elif motion_status["verified"] is not True:
        score_status = "vehicle_motion_not_verified"
    elif result["disqualified"]:
        score_status = "disqualified"
    elif not result["complete"]:
        score_status = "incomplete"
    else:
        score_status = "valid"
    if score_status != "valid":
        result["adjusted_race_time_s"] = None
    result.update({
        "attempt": attempt_index,
        "steps": steps,
        "warmup_lap_times_s": list(info.get("warmup_lap_times_s", [])),
        "warmup_collisions": int(info.get("warmup_collisions", 0)),
        "raw_collision_count_at_finish": int(info.get("raw_collision_count", 0)),
        "race_collision_baseline": info.get("race_collision_baseline"),
        "total_laps_since_start": int(info.get("laps_since_start", 0)),
        "stop_reason": stop_reason,
        "wall_duration_s": time.monotonic() - started,
        "mean_control_rate_hz": (
            1.0 / statistics.fmean(control_intervals_s)
            if control_intervals_s else None
        ),
        "median_control_interval_s": (
            statistics.median(control_intervals_s)
            if control_intervals_s else None
        ),
        "median_reported_lidar_scan_rate_hz": (
            statistics.median(scan_rates_hz) if scan_rates_hz else None
        ),
        "policy_inference_latency_s": _distribution(policy_inference_latencies_s),
        "camera_sample_count": camera_sample_count,
        "camera_fresh_sample_count": camera_fresh_sample_count,
        "camera_stale_sample_count": camera_stale_sample_count,
        "camera_missing_sample_count": camera_missing_sample_count,
        "camera_freshness_unknown_count": camera_freshness_unknown_count,
        "camera_age_s": _distribution(camera_ages_s),
        "camera_freshness_limit_s": float(camera_freshness_limit_s),
        "frontier_progress_m": frontier_progress_m if frontier_tracker is not None else None,
        "frontier_progress_fraction": _frontier_fraction(
            frontier_progress_m if frontier_tracker is not None else None,
            frontier_route_length_m,
            target_laps=frontier_progress_target_laps,
        ),
        "frontier_progress_target_laps": frontier_progress_target_laps,
        "frontier_progress_target_m": (
            frontier_route_length_m * frontier_progress_target_laps
            if frontier_route_length_m is not None else None
        ),
        "frontier_projection_valid": frontier_projection_valid,
        "route_projection_valid": frontier_projection_valid,
        "route_projection_sample_count": route_projection_sample_count,
        "route_projection_valid_sample_count": route_projection_valid_sample_count,
        "route_projection_invalid_sample_count": route_projection_invalid_sample_count,
        "frontier_lateral_offset_m": frontier_lateral_offset_m,
        "frontier_line": frontier_state.get("line"),
        "action_observation_diagnostics": action_observation_diagnostics(
            policy_throttle=policy_throttle,
            policy_steering=policy_steering,
            applied_throttle=applied_throttle,
            applied_steering=applied_steering,
            observation_states=observation_states,
            normalized_lidar_minima=normalized_lidar_minima,
        ),
        "action_trace": action_trace,
        "motion_verification": motion_status,
        "position_diagnostics": summarize_evaluator_positions(position_trace),
        "score_status": score_status,
        "valid_for_comparison": score_status == "valid" and not noncompetitive,
        "noncompetitive_diagnostic": bool(noncompetitive),
    })
    if noncompetitive:
        result["adjusted_race_time_s"] = None
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(
    model_path: Path,
    *,
    attempts: int = 1,
    device: str = "cpu",
    wall_timeout_s: float = 1800.0,
    max_steps: int = 150_000,
    timeout_s: float = 180.0,
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    negative_throttle_mode: str = "allow",
    steering_mode: str = "normal",
    throttle_mode: str = "bidirectional",
    output_path: Optional[Path] = None,
    controller: str = "ppo",
    trace_steps: int = 0,
    observation_profile: str = "official_sensors_camera",
    lap_progress_timeout_s: float = 90.0,
    lap_progress_wall_timeout_s: float = 90.0,
    frontier_path: Optional[Path] = None,
    frontier_stall_after_s: float = 0.0,
    route_monitor_only: bool = False,
    validation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    from src.layer2.official_race_env import OfficialRaceEnv
    from src.layer3.official_policy import load_policy

    checkpoint = Path(model_path).resolve()
    if controller == "ppo" and not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    model_sha256 = _sha256(checkpoint) if checkpoint.is_file() else None
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if trace_steps < 0 or trace_steps > 500:
        raise ValueError("trace_steps must be between 0 and 500")
    if validation_mode not in {None, "bc_diagnostic", "training_probe"}:
        raise ValueError("unsupported validation_mode")
    if route_monitor_only and (validation_mode is not None or frontier_path is None):
        raise ValueError("route_monitor_only requires a competitive evaluation with a verified route path")
    if validation_mode in {"bc_diagnostic", "training_probe"}:
        if frontier_path is None:
            raise ValueError(f"{validation_mode} requires the verified competition frontier path")
        if lap_progress_timeout_s < 240 or lap_progress_wall_timeout_s < 240:
            raise ValueError(f"{validation_mode} requires at least 240-second lap-progress deadlines")
        if frontier_stall_after_s != 5.0:
            raise ValueError(f"{validation_mode} requires the five-second frontier-stall check")
        if attempts != 1:
            raise ValueError(f"{validation_mode} requires exactly one fresh attempt")
    if validation_mode == "training_probe" and controller != "ppo":
        raise ValueError("training_probe requires the authenticated PPO controller")
    if attempts > 1:
        raise ValueError(
            "each official attempt requires a fresh simulator process; the "
            "restricted reset command is not used by the scored path"
        )

    # These are needed by route-monitor identity validation before the race
    # starts, as well as by the optional telemetry publisher below.
    hub_url = os.environ.get("HUB_URL", "")
    run_id = os.environ.get("AICAR_RUN_ID", "")

    model = load_policy(
        checkpoint, device=device, controller=controller,
        observation_profile=observation_profile,
    )
    route_identity_before = (
        _route_monitor_identity(
            output_path=output_path,
            frontier_path=Path(frontier_path),
            expected_run_id=run_id,
        )
        if route_monitor_only else None
    )
    frontier_tracker = None
    if frontier_path is not None:
        from src.layer2.route_progress import RouteProgressTracker
        frontier_tracker = RouteProgressTracker.from_csv(Path(frontier_path), min_push_m=0.04)
    race_laps_required = 3 if validation_mode == "training_probe" else 10
    frontier_progress_target_laps = (
        race_laps_required + 1 if validation_mode is not None else race_laps_required
    )
    env = OfficialRaceEnv(
        timeout_s=timeout_s,
        warmup_laps=1,
        race_laps=race_laps_required,
        steering_action_scale=steering_action_scale,
        straight_throttle_gain=straight_throttle_gain,
        straight_throttle_steering_threshold=straight_throttle_steering_threshold,
        throttle_mode=throttle_mode,
        negative_throttle_mode=negative_throttle_mode,
        steering_mode=steering_mode,
        observation_profile=observation_profile,
        stop_on_collision=True,
    )
    runs: List[Dict[str, Any]] = []
    publisher = None
    camera_preview_writer = None
    last_frame_at = 0.0
    if hub_url and run_id:
        from src.layer3.hub_callback import _Publisher
        publisher = _Publisher(hub_url, snapshot_path=(output_path.parent / "watch_latest.json") if output_path else None)
        if observation_profile == "official_sensors_camera":
            from src.layer3.preview_writer import PreviewWriter
            preview_root = output_path.parent if output_path is not None else checkpoint.parent
            camera_preview_writer = PreviewWriter(preview_root / "camera")
            camera_preview_writer.__enter__()

    def publish_frame(step, observation, info):
        nonlocal last_frame_at
        if publisher is None or time.monotonic() - last_frame_at < 0.1:
            return
        from src.layer3.hub_callback import _lap_fields, _min_pool_lidar
        last_frame_at = time.monotonic()
        position = info.get("race_position")
        route_position = _official_xy_position(position)
        scan = observation["lidar"]
        if getattr(scan, "ndim", 1) > 1:
            scan = scan[-1]
        publisher.enqueue({
            "kind": "fleet", "runtime": "official", "channel": "replay",
            "run_id": run_id, "step": step, "episode": 1, "map_id": "none",
            "ts": datetime.now(timezone.utc).isoformat(),
            "lidar_range_min": 0.06, "lidar_range_max": 10.0,
            "cars": [{"env_id": 0, "pose": route_position,
                      "yaw": info.get("yaw"), "speed": info.get("watch_speed_mps"),
                      "speed_source": "lidar_estimate", "episode_return": None,
                      "collision": bool(info.get("collision_event")),
                      "collision_count": info.get("race_collisions", 0),
                      "lidar": _min_pool_lidar(scan, 120),
                      "frontier_line": info.get("frontier_line"),
                      "frontier_progress_m": info.get("frontier_progress_m"),
                      "frontier_progress_fraction": info.get("frontier_progress_fraction"),
                      "frontier_projection_valid": info.get("frontier_projection_valid"),
                      "route_projection_valid": info.get("route_projection_valid"),
                      "frontier_lateral_offset_m": info.get("frontier_lateral_offset_m"),
                      "camera_age_s": info.get("camera_age_s"),
                      "camera_fresh": info.get("camera_fresh"),
                      **_lap_fields(info)}],
        })
    try:
        for attempt in range(1, attempts + 1):
            runs.append(evaluate_attempt(
                model,
                env,
                wall_timeout_s=wall_timeout_s,
                max_steps=max_steps,
                attempt_index=attempt,
                race_laps_required=race_laps_required,
                frontier_progress_target_laps=frontier_progress_target_laps,
                noncompetitive=validation_mode is not None,
                trace_steps=trace_steps,
                lap_progress_timeout_s=lap_progress_timeout_s,
                lap_progress_wall_timeout_s=lap_progress_wall_timeout_s,
                frontier_tracker=frontier_tracker,
                frontier_stall_after_s=frontier_stall_after_s,
                route_monitor_only=route_monitor_only,
                camera_preview_writer=camera_preview_writer,
                disqualification_limit=10,
                on_step=publish_frame,
            ))
        completed = [
            run for run in runs
            if run.get("score_status") == "valid"
            and run.get("complete") is True
        ]
        competitive = validation_mode is None
        adjusted = [
            float(run["adjusted_race_time_s"])
            for run in completed
            if competitive and run.get("adjusted_race_time_s") is not None
        ]
        result: Dict[str, Any] = {
            "model": str(checkpoint),
            "model_sha256": model_sha256,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "competition": "RoboRacer Sim Racing League @ IROS 2026",
            "official_base_image": "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete",
            "simulator_image": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete",
            "deterministic": True,
            "controller": controller,
            "observation_profile": observation_profile,
            "throttle_mode": throttle_mode,
            "steering_mode": steering_mode,
            "warmup_laps_ignored": 1,
            "race_laps": race_laps_required,
            "validation_mode": validation_mode,
            "competitive_score_valid": competitive,
            "noncompetitive_diagnostic": not competitive,
            "protocol": (
                "official_training_probe_1_plus_3" if validation_mode == "training_probe"
                else "official_bc_diagnostic_1_plus_10" if validation_mode == "bc_diagnostic"
                else "official_competitive_1_plus_10"
            ),
            "runs": runs,
            "completed_attempts": len(completed),
            "median_adjusted_race_time_s": statistics.median(adjusted) if competitive and adjusted else None,
            "best_adjusted_race_time_s": min(adjusted) if competitive and adjusted else None,
        }
        if route_monitor_only:
            route_identity = _route_monitor_identity(
                output_path=output_path,
                frontier_path=Path(frontier_path),
                expected_run_id=run_id,
            )
            if route_identity != route_identity_before:
                raise RuntimeError("official scene, route, or simulator identity changed during route monitoring")
            result.update(route_identity)
            for run in runs:
                run.update(route_identity)
                run["route_monitor"] = {
                    "projection_valid": (
                        run.get("route_projection_invalid_sample_count", 0) == 0
                        and run.get("route_projection_valid_sample_count", 0) > 0
                    ),
                    "projection_sample_count": run.get("route_projection_sample_count", 0),
                    "projection_valid_sample_count": run.get("route_projection_valid_sample_count", 0),
                    "projection_invalid_sample_count": run.get("route_projection_invalid_sample_count", 0),
                    "scoring_or_stop_effect": False,
                }
        target = output_path or checkpoint.parent / f"{checkpoint.stem}_iros2026_evaluation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved official evaluation: {target}", flush=True)
        print(json.dumps({
            "completed_attempts": result["completed_attempts"],
            "median_adjusted_race_time_s": result["median_adjusted_race_time_s"],
            "best_adjusted_race_time_s": result["best_adjusted_race_time_s"],
        }, indent=2), flush=True)
        return result
    finally:
        try:
            if camera_preview_writer is not None:
                camera_preview_writer.__exit__(None, None, None)
        finally:
            try:
                env.close()
            finally:
                if publisher is not None:
                    publisher.stop()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller", choices=("ppo", "lidar_gap", "straight"), default="ppo")
    parser.add_argument(
        "--observation-profile",
        choices=("official_sensors", "official_sensors_history", "official_sensors_camera"),
        default="official_sensors_camera",
    )
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--wall-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-steps", type=int, default=150_000, help="Failsafe guard per race, not a scoring cutoff")
    parser.add_argument("--timeout-s", type=float, default=180.0, help="Wait for initial official sensor topics")
    parser.add_argument("--steering-action-scale", type=float, default=1.0)
    parser.add_argument("--straight-throttle-gain", type=float, default=1.0)
    parser.add_argument("--straight-throttle-steering-threshold", type=float, default=0.15)
    parser.add_argument(
        "--negative-throttle-mode",
        choices=("allow", "zero", "positive_magnitude"),
        default="allow",
        help="How to map negative policy throttle actions to the actuator",
    )
    parser.add_argument(
        "--steering-mode",
        choices=("normal", "invert"),
        default="normal",
        help="Map policy steering into the official actuator direction",
    )
    parser.add_argument(
        "--throttle-mode",
        choices=("bidirectional", "forward_only"),
        default="bidirectional",
        help="Map normalized policy throttle into signed or forward-only actuator range",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trace-steps", type=int, default=0, help="Include a bounded action/sensor trace in the report")
    parser.add_argument("--lap-progress-timeout-s", type=float, default=90.0)
    parser.add_argument("--lap-progress-wall-timeout-s", type=float, default=90.0)
    parser.add_argument("--frontier-path", type=Path, default=None)
    parser.add_argument("--frontier-stall-after-s", type=float, default=0.0)
    parser.add_argument("--route-monitor-only", action="store_true")
    parser.add_argument("--validation-mode", choices=("bc_diagnostic", "training_probe"), default=None)
    args = parser.parse_args(argv)
    evaluate(
        args.model,
        attempts=args.attempts,
        device=args.device,
        wall_timeout_s=args.wall_timeout_s,
        max_steps=args.max_steps,
        timeout_s=args.timeout_s,
        steering_action_scale=args.steering_action_scale,
        straight_throttle_gain=args.straight_throttle_gain,
        straight_throttle_steering_threshold=args.straight_throttle_steering_threshold,
        negative_throttle_mode=args.negative_throttle_mode,
        steering_mode=args.steering_mode,
        throttle_mode=args.throttle_mode,
        output_path=args.out,
        controller=args.controller,
        trace_steps=args.trace_steps,
        observation_profile=args.observation_profile,
        lap_progress_timeout_s=args.lap_progress_timeout_s,
        lap_progress_wall_timeout_s=args.lap_progress_wall_timeout_s,
        frontier_path=args.frontier_path,
        frontier_stall_after_s=args.frontier_stall_after_s,
        route_monitor_only=args.route_monitor_only,
        validation_mode=args.validation_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
