"""Evaluate a policy in the official IROS 2026 AutoDRIVE ROS 2 race flow."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def collision_penalty_seconds(collisions: int, seconds_per_collision: float = 10.0) -> float:
    """Official increasing penalty: 10 s, then 20 s, then 30 s, and so on."""
    count = max(0, int(collisions))
    return float(seconds_per_collision) * count * (count + 1) / 2.0


def summarize_attempt(
    *,
    lap_times_s: List[float],
    race_collisions: int,
    disqualification_limit: int = 10,
) -> Dict[str, Any]:
    valid_times = [float(value) for value in lap_times_s if math.isfinite(float(value)) and float(value) > 0]
    race_time = sum(valid_times) if len(valid_times) == len(lap_times_s) else None
    disqualified = int(race_collisions) > int(disqualification_limit)
    penalty = collision_penalty_seconds(race_collisions)
    return {
        "race_laps_completed": len(lap_times_s),
        "race_lap_times_s": list(lap_times_s),
        "race_time_s": race_time,
        "race_collisions": int(race_collisions),
        "collision_penalty_s": penalty,
        "adjusted_race_time_s": None if disqualified or race_time is None else race_time + penalty,
        "disqualified": disqualified,
        "best_lap_s": min(valid_times) if valid_times else None,
        "mean_lap_s": statistics.fmean(valid_times) if valid_times else None,
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
) -> Dict[str, Any]:
    obs, info = env.reset()
    started = time.monotonic()
    laps_reported = 0
    collisions_reported = 0
    steps = 0
    control_intervals_s: List[float] = []
    scan_rates_hz: List[float] = []
    policy_throttle: List[float] = []
    policy_steering: List[float] = []
    applied_throttle: List[float] = []
    applied_steering: List[float] = []
    observation_states: List[List[float]] = []
    normalized_lidar_minima: List[float] = []
    while not bool(info.get("race_complete", False)):
        if time.monotonic() - started >= wall_timeout_s:
            stop_reason = "wall_timeout"
            break
        if steps >= max_steps:
            stop_reason = "step_guard"
            break
        observation_states.append([float(v) for v in obs["state"]])
        normalized_lidar_minima.append(float(min(obs["lidar"])))
        action, _ = model.predict(obs, deterministic=True)
        command = [float(v) for v in action.reshape(-1)]
        policy_throttle.append(command[0])
        policy_steering.append(command[1])
        obs, _, terminated, _, info = env.step(action)
        applied_throttle.append(float(info.get("throttle_command", 0.0)))
        applied_steering.append(float(info.get("steering_command", 0.0)))
        steps += 1
        interval = float(info.get("control_interval_s", 0.0))
        scan_rate = float(info.get("lidar_scan_rate_hz", 0.0))
        if math.isfinite(interval) and interval > 0:
            control_intervals_s.append(interval)
        if math.isfinite(scan_rate) and scan_rate > 0:
            scan_rates_hz.append(scan_rate)

        lap_times = list(info.get("race_lap_times_s", []))
        while laps_reported < len(lap_times):
            laps_reported += 1
            print(
                f"attempt {attempt_index}: race lap {laps_reported}/10 "
                f"{float(lap_times[laps_reported - 1]):.3f}s",
                flush=True,
            )
        collisions = int(info.get("race_collisions", 0))
        while collisions_reported < collisions:
            collisions_reported += 1
            print(
                f"attempt {attempt_index}: race collision {collisions_reported}; "
                f"penalty now {collision_penalty_seconds(collisions_reported):.0f}s",
                flush=True,
            )
        if terminated:
            stop_reason = "race_complete"
            break
    else:
        stop_reason = "race_complete"

    result = summarize_attempt(
        lap_times_s=list(info.get("race_lap_times_s", [])),
        race_collisions=int(info.get("race_collisions", 0)),
    )
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
        "action_observation_diagnostics": action_observation_diagnostics(
            policy_throttle=policy_throttle,
            policy_steering=policy_steering,
            applied_throttle=applied_throttle,
            applied_steering=applied_steering,
            observation_states=observation_states,
            normalized_lidar_minima=normalized_lidar_minima,
        ),
    })
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate(
    model_path: Path,
    *,
    attempts: int = 1,
    device: str = "cpu",
    wall_timeout_s: float = 300.0,
    max_steps: int = 150_000,
    timeout_s: float = 180.0,
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    negative_throttle_mode: str = "allow",
    steering_mode: str = "normal",
    throttle_mode: str = "bidirectional",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from src.layer2.official_race_env import OfficialRaceEnv
    from src.layer3.official_policy import load_policy

    checkpoint = Path(model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if attempts > 1:
        raise ValueError(
            "each official attempt requires a fresh simulator process; the "
            "restricted reset command is not used by the scored path"
        )

    model = load_policy(checkpoint, device=device)
    env = OfficialRaceEnv(
        timeout_s=timeout_s,
        warmup_laps=1,
        race_laps=10,
        steering_action_scale=steering_action_scale,
        straight_throttle_gain=straight_throttle_gain,
        straight_throttle_steering_threshold=straight_throttle_steering_threshold,
        throttle_mode=throttle_mode,
        negative_throttle_mode=negative_throttle_mode,
        steering_mode=steering_mode,
    )
    runs: List[Dict[str, Any]] = []
    try:
        for attempt in range(1, attempts + 1):
            runs.append(evaluate_attempt(
                model,
                env,
                wall_timeout_s=wall_timeout_s,
                max_steps=max_steps,
                attempt_index=attempt,
            ))
        completed = [run for run in runs if run["race_laps_completed"] == 10 and run["adjusted_race_time_s"] is not None]
        adjusted = [float(run["adjusted_race_time_s"]) for run in completed]
        result: Dict[str, Any] = {
            "model": str(checkpoint),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "competition": "RoboRacer Sim Racing League @ IROS 2026",
            "official_base_image": "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete",
            "simulator_image": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete",
            "deterministic": True,
            "throttle_mode": throttle_mode,
            "steering_mode": steering_mode,
            "warmup_laps_ignored": 1,
            "race_laps": 10,
            "runs": runs,
            "completed_attempts": len(completed),
            "median_adjusted_race_time_s": statistics.median(adjusted) if adjusted else None,
            "best_adjusted_race_time_s": min(adjusted) if adjusted else None,
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
        env.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--wall-timeout-s", type=float, default=300.0)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
