"""Drive the official practice route with the pure-pursuit teacher only.

This entry point creates the official Gym environment but never constructs or
fits a PPO/BC policy. It sends ordinary step-based actions to the simulator;
it does not require simulator changes or lockstep physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.layer2.route_progress import RouteProgressTracker
from src.layer3.expert import CenterlineExpert
from src.layer3.official_tracks import WINDOWS_SCENES, official_frontier_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    """Replace non-finite telemetry diagnostics with null before JSON output."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _verified_lap_goal_reached(
    *,
    race_laps_completed: int,
    race_lap_times_s: list[float],
    current_route_frontier_m: float | None,
    initial_route_frontier_m: float | None,
    route_length_m: float,
    lap_goal: int,
    minimum_route_coverage: float = 0.95,
) -> bool:
    """Require scored official laps with valid times and route distance."""
    valid_times = [
        _finite_float(value) for value in race_lap_times_s
    ]
    if race_laps_completed < lap_goal or len(valid_times) < lap_goal:
        return False
    if any(value is None or value <= 0.0 for value in valid_times[:lap_goal]):
        return False
    if current_route_frontier_m is None or initial_route_frontier_m is None:
        return False
    route_delta = current_route_frontier_m - initial_route_frontier_m
    return route_delta >= route_length_m * lap_goal * minimum_route_coverage


def _official_simulator_provenance(route_path: Path) -> dict[str, Any]:
    """Record the matching native Windows runtime's preflighted hashes."""
    evidence_value = os.environ.get("AICAR_RUNTIME_PREFLIGHT", "").strip()
    if not evidence_value:
        raise RuntimeError("AICAR_RUNTIME_PREFLIGHT must point to the native runtime preflight JSON")
    evidence_path = Path(evidence_value).resolve()
    if not evidence_path.is_file():
        raise FileNotFoundError(f"native runtime preflight not found: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not str(evidence.get("runtime_kind", "")).startswith("native_windows_official_simulator"):
        raise RuntimeError(f"runtime preflight is not for the native official Windows simulator: {evidence_path}")
    if evidence.get("docker_used_for_runtime") is not False:
        raise RuntimeError(f"runtime preflight does not verify a Docker-free runtime: {evidence_path}")
    if evidence.get("profile") != "practice":
        raise RuntimeError(f"runtime preflight is not for the official practice profile: {evidence_path}")
    if evidence.get("scene_sha256") != WINDOWS_SCENES["practice"]:
        raise RuntimeError(f"runtime preflight scene hash does not match official practice: {evidence_path}")
    if evidence.get("route_sha256") != _sha256(route_path):
        raise RuntimeError(f"runtime preflight route hash does not match the teacher route: {evidence_path}")

    launch_config_path = evidence_path.with_name("launch_config.json")
    launch_config = json.loads(launch_config_path.read_text(encoding="utf-8"))
    if launch_config.get("track_profile") != "practice":
        raise RuntimeError(f"runtime launch config is not for the practice profile: {launch_config_path}")
    return {
        "backend": "native_windows_official_simulator",
        "executable_path": launch_config.get("simulator_executable"),
        "executable_sha256": evidence.get("executable_sha256"),
        "scene_sha256": evidence.get("scene_sha256"),
        "scene_verified_for_profile": "practice",
        "runtime_run_id": evidence.get("preflight_run_id", evidence_path.parent.name),
        "simulator_process_id": evidence.get("sim_pid"),
        "release_evidence_path": evidence.get("official_release_evidence"),
        "runtime_preflight_path": str(evidence_path),
        "runtime_preflight_sha256": _sha256(evidence_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-profile", choices=("practice", "competition"), default="practice")
    parser.add_argument("--laps", type=int, default=3, help="clean official lap goal for this episode")
    parser.add_argument("--max-steps", type=int, default=30000, help="hard control-step cap")
    parser.add_argument("--max-wall-seconds", type=float, default=900.0, help="hard wall-clock cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed-mps", type=float, default=0.3)
    parser.add_argument("--lookahead-m", type=float, default=0.8)
    parser.add_argument("--observation-profile", choices=("official_sensors", "official_sensors_history", "official_sensors_camera"), default="official_sensors")
    parser.add_argument("--pose-age-limit-s", type=float, default=1.0)
    parser.add_argument("--control-gap-limit-s", type=float, default=2.0)
    parser.add_argument("--stall-limit-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.laps < 1 or args.max_steps < 1 or args.max_wall_seconds <= 0:
        raise ValueError("laps, max-steps, and max-wall-seconds must be positive")
    if args.speed_mps < 0.2 or args.lookahead_m < 0.5:
        raise ValueError("speed must be at least 0.2 m/s and lookahead at least 0.5 m")
    if args.track_profile != "practice":
        raise ValueError("official_teacher is for the Porto practice route; select --track-profile practice")

    route_path = official_frontier_path(profile="practice").resolve()
    route_metadata_path = route_path.with_name("official_frontier.json")
    json.loads(route_metadata_path.read_text(encoding="utf-8"))
    simulator_provenance = _official_simulator_provenance(route_path)
    route = RouteProgressTracker.from_csv(route_path)
    expert = CenterlineExpert(
        route,
        max_speed_mps=args.speed_mps,
        lookahead_base_m=args.lookahead_m,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "teacher_trace.jsonl"
    summary_path = output_dir / "teacher_summary.json"

    # This mirrors the official training environment's route, actuator mapping,
    # and sensors while avoiding every PPO construction/training code path.
    from src.layer3.official_envs import close_official_vec_env, make_official_vec_env

    vec_env = make_official_vec_env(
        1,
        seed=args.seed,
        timeout_s=args.timeout_s,
        warmup_laps=0,
        race_laps=args.laps,
        steering_action_scale=1.0,
        straight_throttle_gain=1.0,
        straight_throttle_steering_threshold=0.15,
        throttle_mode="bidirectional",
        negative_throttle_mode="allow",
        steering_mode="normal",
        observation_profile=args.observation_profile,
        training_mode=True,
        training_timeout_s=args.max_wall_seconds,
        training_frontier_stagnation_s=args.stall_limit_s,
        stop_on_collision=True,
        frontier_path=route_path,
    )

    start = time.monotonic()
    previous_control = start
    initial_collision_count: int | None = None
    initial_expiry_count: int | None = None
    initial_route_progress_m: float | None = None
    initial_official_lap_count: int | None = None
    previous_lap_count: int | None = None
    termination_reason = "step_guard"
    completed_cleanly = False
    trace_count = 0
    step_durations: list[float] = []
    control_gaps: list[float] = []
    pose_ages: list[float] = []
    max_lateral_error = 0.0
    max_lateral_ratio = 0.0
    last_info: dict[str, Any] = {}
    official_lap_events: list[dict[str, Any]] = []
    run_error: str | None = None
    stop_requested = False
    old_sigint = signal.getsignal(signal.SIGINT)

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        observations = vec_env.reset()
        reset_info = vec_env.env_method("get_current_info")[0]
        initial_route_progress_m = _finite_float(reset_info.get("current_progress_m"))
        previous_lap_count = int(reset_info.get("lap_count", 0) or 0)
        initial_official_lap_count = previous_lap_count
        with trace_path.open("w", encoding="utf-8", buffering=1) as trace:
            for step_index in range(args.max_steps):
                now = time.monotonic()
                elapsed = now - start
                if stop_requested:
                    termination_reason = "operator_stop"
                    break
                if elapsed >= args.max_wall_seconds:
                    termination_reason = "wall_time_guard"
                    break

                info = vec_env.env_method("get_current_info")[0]
                last_info = dict(info)
                collision_count = int(info.get("collision_count", 0) or 0)
                expiry_count = int(info.get("command_expiry_count", 0) or 0)
                if initial_collision_count is None:
                    initial_collision_count = collision_count
                if initial_expiry_count is None:
                    initial_expiry_count = expiry_count
                if collision_count > initial_collision_count:
                    termination_reason = "collision_counter_changed"
                    break
                if expiry_count > initial_expiry_count:
                    termination_reason = "command_expiry"
                    break

                pose_age = _finite_float(info.get("position_age_s"))
                if pose_age is None or pose_age > args.pose_age_limit_s:
                    termination_reason = "stale_or_missing_pose"
                    break
                if _finite_float(info.get("yaw")) is None:
                    termination_reason = "invalid_pose_yaw"
                    break
                control_gap = now - previous_control
                if step_index and control_gap > args.control_gap_limit_s:
                    termination_reason = "control_gap_guard"
                    break
                previous_control = now

                route_diag = expert.route_diagnostics(info)
                route_s = route_diag["route_s_m"]
                lateral = route_diag["lateral_error_m"]
                lateral_bound = route_diag["lateral_bound_m"]
                if route_s is None or lateral is None or lateral_bound is None:
                    termination_reason = "route_projection_invalid"
                    break
                max_lateral_error = max(max_lateral_error, float(lateral))
                lateral_ratio = float(lateral) / max(1e-9, float(lateral_bound))
                max_lateral_ratio = max(max_lateral_ratio, lateral_ratio)
                if float(lateral) > float(lateral_bound):
                    termination_reason = "route_corridor_clearance_exceeded"
                    break

                action = expert.action(info, observations["lidar"][0], observations["state"][0])
                control_time = time.monotonic()
                # The check above used the previous action boundary. Capture
                # the actual interval between controller decisions as well.
                control_gaps.append(control_time - (start if step_index == 0 else last_control_time))
                last_control_time = control_time
                observations, _, dones, infos = vec_env.step(np.asarray([action], dtype=np.float32))
                post = dict(infos[0])
                current_lap_count = int(post.get("lap_count", 0) or 0)
                lap_delta = max(0, current_lap_count - int(previous_lap_count or 0))
                if lap_delta:
                    official_lap_events.append({
                        "step": step_index + 1,
                        "elapsed_wall_s": round(time.monotonic() - start, 6),
                        "official_lap_count": current_lap_count,
                        "lap_event_count": lap_delta,
                        "official_last_lap_time_s": _finite_float(post.get("last_lap_time_s")),
                        "route_frontier_m": _finite_float(post.get("frontier_progress_m")),
                    })
                duration = _finite_float(post.get("step_duration_s"))
                if duration is not None:
                    step_durations.append(duration)
                pose_age_post = _finite_float(post.get("position_age_s"))
                if pose_age_post is not None:
                    pose_ages.append(pose_age_post)
                watch_speed = _finite_float(info.get("watch_speed_mps"))
                speed_value = watch_speed if watch_speed is not None else _finite_float(info.get("true_speed"))
                row = {
                    "step": step_index + 1,
                    "elapsed_wall_s": round(time.monotonic() - start, 6),
                    "position_xy_m": info.get("position"),
                    "yaw_rad": info.get("yaw"),
                    "route_projection_m": route_s,
                    "lateral_error_m": lateral,
                    "lateral_bound_m": lateral_bound,
                    "corridor_left_half_width_m": route_diag.get("corridor_left_half_width_m"),
                    "corridor_right_half_width_m": route_diag.get("corridor_right_half_width_m"),
                    "action_throttle": float(action[0]),
                    "action_steering": float(action[1]),
                    "official_lap_count": post.get("official_lap_count"),
                    "lap_event_count": lap_delta,
                    "official_lap_time_s": post.get("lap_time_s"),
                    "official_last_lap_time_s": post.get("last_lap_time_s"),
                    "route_frontier_m": post.get("frontier_progress_m"),
                    "collision_count": post.get("collision_count"),
                    "collision_event": post.get("collision_event"),
                    "control_gap_s": control_gaps[-1],
                    "step_duration_s": duration,
                    "position_age_s": pose_age_post,
                    "command_expiry_count": post.get("command_expiry_count"),
                    "command_expiry_diagnostic": post.get("command_expiry_diagnostic"),
                    "neutral_fallback_count": post.get("neutral_fallback_count"),
                    "neutral_fallback_diagnostic": post.get("neutral_fallback_diagnostic"),
                    "control_interval_source": post.get("control_interval_source"),
                    "speed_mps": speed_value,
                    "speed_source": "watch_speed_mps" if watch_speed is not None else "true_speed",
                }
                trace.write(json.dumps(_json_safe(row), separators=(",", ":"), allow_nan=False) + "\n")
                trace_count += 1
                last_info = post
                previous_lap_count = current_lap_count

                if bool(post.get("collision_event")):
                    termination_reason = "collision"
                    break
                if int(post.get("command_expiry_count", 0) or 0) > int(initial_expiry_count or 0):
                    termination_reason = "command_expiry"
                    break
                if (post.get("position_age_s") is None or float(post["position_age_s"]) > args.pose_age_limit_s):
                    termination_reason = "stale_or_missing_pose"
                    break
                if bool(post.get("training_frontier_stagnated")):
                    termination_reason = "sustained_stall"
                    break
                if _verified_lap_goal_reached(
                    race_laps_completed=int(post.get("race_laps_completed", 0) or 0),
                    race_lap_times_s=list(post.get("race_lap_times_s", []) or []),
                    current_route_frontier_m=_finite_float(post.get("frontier_progress_m")),
                    initial_route_frontier_m=initial_route_progress_m,
                    route_length_m=route.length_m,
                    lap_goal=args.laps,
                ):
                    termination_reason = "clean_lap_goal"
                    completed_cleanly = True
                    break
                if dones[0]:
                    if bool(post.get("episode_won")) and bool(post.get("race_complete")):
                        termination_reason = "clean_lap_goal"
                        completed_cleanly = True
                    else:
                        termination_reason = str(post.get("termination_reason") or "episode_ended_unsuccessfully")
                    break
            else:
                termination_reason = "step_guard"
    except Exception as exc:
        termination_reason = f"exception:{type(exc).__name__}:{exc}"
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            vec_env.env_method("request_pause_neutral")
        except Exception:
            pass
        close_official_vec_env(vec_env)
        signal.signal(signal.SIGINT, old_sigint)

    duration_wall = time.monotonic() - start
    final_official_lap_count = int(last_info.get("official_lap_count", 0) or 0)
    laps_completed = (
        max(0, final_official_lap_count - initial_official_lap_count)
        if initial_official_lap_count is not None else 0
    )
    route_frontier_absolute_m = _finite_float(last_info.get("frontier_progress_m"))
    route_frontier_m = (
        max(0.0, route_frontier_absolute_m - initial_route_progress_m)
        if route_frontier_absolute_m is not None and initial_route_progress_m is not None
        else 0.0
    )
    route_target_m = route.length_m * args.laps
    route_target_met = route_frontier_m >= route_target_m * 0.95
    collisions = max(
        0,
        int(last_info.get("collision_count", 0) or 0) - int(initial_collision_count or 0),
    )
    summary = {
        "status": "accepted" if (
            completed_cleanly
            and collisions == 0
            and laps_completed == args.laps
            and route_target_met
            and termination_reason == "clean_lap_goal"
        ) else "failed",
        "termination_reason": termination_reason,
        "error": run_error,
        "clean_lap_goal": args.laps,
        "laps_completed": laps_completed,
        "official_lap_count": final_official_lap_count,
        "race_laps_completed_by_environment": int(last_info.get("race_laps_completed", 0) or 0),
        "official_lap_events": official_lap_events,
        "race_lap_times_s": last_info.get("race_lap_times_s", []),
        "route_length_m": route.length_m,
        "route_frontier_m": route_frontier_m,
        "route_frontier_start_m": initial_route_progress_m,
        "route_frontier_end_m": route_frontier_absolute_m,
        "route_target_m": route_target_m,
        "route_target_met_95_percent": route_target_met,
        "initial_official_lap_count": initial_official_lap_count,
        "final_official_lap_count": final_official_lap_count,
        "official_lap_counter_delta": laps_completed,
        "race_lap_times_s": last_info.get("race_lap_times_s", []),
        "neutral_fallback_count": int(last_info.get("neutral_fallback_count", 0) or 0),
        "verified_lap_goal_rule": (
            "environment-scored lap count and finite positive lap times meet goal, with route frontier delta >= 95% of route length times lap goal"
        ),
        "collision_count": collisions,
        "trace_rows": trace_count,
        "wall_time_s": round(duration_wall, 4),
        "max_steps": args.max_steps,
        "max_wall_seconds": args.max_wall_seconds,
        "timing": {
            "step_duration_count": len(step_durations),
            "step_duration_mean_s": statistics.fmean(step_durations) if step_durations else None,
            "step_duration_max_s": max(step_durations) if step_durations else None,
            "control_gap_mean_s": statistics.fmean(control_gaps) if control_gaps else None,
            "control_gap_max_s": max(control_gaps) if control_gaps else None,
            "pose_age_mean_s": statistics.fmean(pose_ages) if pose_ages else None,
            "pose_age_max_s": max(pose_ages) if pose_ages else None,
            "command_expiry_count": max(
                0,
                int(last_info.get("command_expiry_count", 0) or 0) - int(initial_expiry_count or 0),
            ),
            "command_expiry_counter_final": last_info.get("command_expiry_count"),
            "neutral_fallback_count": int(last_info.get("neutral_fallback_count", 0) or 0),
            "control_interval_source": last_info.get("control_interval_source"),
        },
        "corridor": {
            "vehicle_half_width_m": expert.vehicle_half_width_m,
            "safety_margin_m": expert.corridor_margin_m,
            "bound_rule": "min(interpolated route left/right half-width) - vehicle_half_width - safety_margin",
            "max_lateral_error_m": max_lateral_error,
            "max_lateral_error_to_bound_ratio": max_lateral_ratio,
        },
        "teacher": {
            "controller": type(expert).__name__,
            "max_speed_target_mps": args.speed_mps,
            "lookahead_base_m": args.lookahead_m,
            "route_coordinate_frame": "official practice route and IPS position in ROS XY meters",
            "yaw_conversion": "Unity Y-up yaw + pi/2 converted to route-bearing convention",
        },
        "runtime_contract": "Official step-based observation/action environment; no simulator modification or lockstep assumption.",
        "track": {
            "profile": "practice",
            "route_path": str(route_path),
            "route_sha256": _sha256(route_path),
            "route_source": "official IROS 2026 practice Porto mesh, rigidly aligned to the repository Porto centerline; repository map coordinates are not selected directly",
            "metadata_path": str(route_metadata_path),
            "metadata_sha256": _sha256(route_metadata_path),
            "simulator": simulator_provenance,
        },
        "configuration": {
            "seed": args.seed,
            "laps": args.laps,
            "observation_profile": args.observation_profile,
            "action_mapping": "bidirectional throttle, negative throttle allow, normal steering, scale 1, throttle gain 1",
            "pose_age_limit_s": args.pose_age_limit_s,
            "control_gap_limit_s": args.control_gap_limit_s,
            "stall_limit_s": args.stall_limit_s,
            "timeout_s": args.timeout_s,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        print(f"teacher run failed: {exc}")
        return 2
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
