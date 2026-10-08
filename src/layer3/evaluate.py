"""Deterministic, repeatable Layer 3 policy evaluation on a selected map."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def evaluate_until_episode_end(
    model: Any,
    env: Any,
    *,
    frame_skip: int,
    timesteps: int = 0,
    lap_target: int = 10,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Evaluate deterministically until failure or the configured lap target."""
    from src.layer2.autodrive_env import SIMULATION_HZ

    if frame_skip < 1:
        raise ValueError("frame_skip must be at least one")
    if lap_target < 1:
        raise ValueError("lap_target must be at least one")
    obs = env.reset()
    step_seconds = frame_skip / SIMULATION_HZ
    total_reward = frontier_distance = simulated_seconds = 0.0
    collisions = failed_episodes = completed_laps = 0
    lap_times: List[float] = []
    lap_diagnostics: List[Dict[str, Any]] = []
    lap_samples: List[Dict[str, float]] = []
    initial_action_trace: List[Dict[str, Any]] = []
    current_life_laps: List[float] = []
    seen_lap_times = 0
    best_10_lap_time_s: Optional[float] = None
    termination_reasons: Dict[str, int] = {}

    stop_reason: Optional[str] = None
    while stop_reason is None:
        policy_obs = obs
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += float(rewards[0])
        info = infos[0] if infos else {}
        duration = info.get("step_duration_s", step_seconds) if isinstance(info, dict) else step_seconds
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Evaluation step_duration_s must be finite and positive")
        simulated_seconds += duration
        if not isinstance(info, dict):
            continue
        # A short, bounded trace makes deterministic failures diagnosable: it
        # shows the exact policy command alongside the sensor state used to
        # choose it, without changing the policy observation or action path.
        if len(initial_action_trace) < 64:
            try:
                import numpy as np

                action_values = np.asarray(action, dtype=float).reshape(-1, 2)[0]
                lidar_values = np.asarray(policy_obs["lidar"], dtype=float).reshape(-1, 1081)[0]
                state_values = np.asarray(policy_obs["state"], dtype=float).reshape(-1, 9)[0]
                initial_action_trace.append({
                    "step": len(initial_action_trace) + 1,
                    "simulated_seconds": simulated_seconds,
                    "policy_action": [float(action_values[0]), float(action_values[1])],
                    "throttle_command": info.get("throttle_command"),
                    "steering_command": info.get("steering_command"),
                    "forward_speed_mps": info.get("v_long"),
                    "speed_mps": info.get("true_speed"),
                    "yaw_rad": info.get("yaw"),
                    "position": list(info["position"]) if "position" in info else None,
                    "state": [float(value) for value in state_values],
                    "lidar_min_left": float(np.min(lidar_values[:360])),
                    "lidar_min_front": float(np.min(lidar_values[360:721])),
                    "lidar_min_right": float(np.min(lidar_values[721:])),
                })
            except (KeyError, IndexError, TypeError, ValueError):
                # Test doubles and legacy observation wrappers may not expose
                # the canonical dict observation; omit trace data in that case.
                pass
        frontier_distance += max(0.0, float(info.get("frontier_advanced_m", 0.0) or 0.0))
        collisions += int(bool(info.get("collision_event", False)))
        if bool(info.get("collision_event", False)):
            stop_reason = "collision"
        sample: Dict[str, float] = {}
        for output_key, input_key in (
            ("speed_mps", "true_speed"),
            ("forward_speed_mps", "v_long"),
            ("route_speed_mps", "current_route_speed_mps"),
            ("throttle", "throttle_command"),
            ("steering", "steering_command"),
        ):
            try:
                value = float(info[input_key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                sample[output_key] = value
        if sample:
            sample["step_duration_s"] = duration
            lap_samples.append(sample)
        times = info.get("lap_times_s")
        if isinstance(times, (list, tuple)):
            if len(times) < seen_lap_times:
                seen_lap_times = 0
            new_lap_times = times[seen_lap_times:]
            for raw in new_lap_times:
                try:
                    lap_time = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(lap_time) and lap_time > 0:
                    lap_times.append(lap_time)
                    current_life_laps.append(lap_time)
                    completed_laps += 1
                    diagnostic: Dict[str, Any] = {
                        "lap": completed_laps,
                        "lap_time_s": lap_time,
                        "samples": len(lap_samples),
                    }
                    metric_inputs = {
                        "speed_mps": "speed_mps",
                        "forward_speed_mps": "forward_speed_mps",
                        "route_speed_mps": "route_speed_mps",
                        "throttle": "throttle",
                        "steering": "steering",
                    }
                    for metric, sample_key in metric_inputs.items():
                        values = [
                            (item[sample_key], item["step_duration_s"])
                            for item in lap_samples
                            if sample_key in item
                        ]
                        weight = sum(item[1] for item in values)
                        diagnostic[f"mean_{metric}"] = (
                            sum(value * seconds for value, seconds in values) / weight
                            if weight > 0 else None
                        )
                    speeds = [item["speed_mps"] for item in lap_samples if "speed_mps" in item]
                    steerings = [item["steering"] for item in lap_samples if "steering" in item]
                    throttle_samples = [item for item in lap_samples if "throttle" in item]
                    throttle_duration = sum(item["step_duration_s"] for item in throttle_samples)
                    diagnostic["max_speed_mps"] = max(speeds) if speeds else None
                    diagnostic["max_abs_steering"] = max(map(abs, steerings)) if steerings else None
                    diagnostic["full_throttle_fraction"] = (
                        sum(
                            item["step_duration_s"]
                            for item in throttle_samples
                            if item["throttle"] >= 0.99
                        ) / throttle_duration
                        if throttle_duration > 0 else None
                    )
                    lap_diagnostics.append(diagnostic)
                    lap_samples = []
                    if len(current_life_laps) >= 10:
                        total_10 = sum(current_life_laps[-10:])
                        best_10_lap_time_s = (
                            total_10 if best_10_lap_time_s is None
                            else min(best_10_lap_time_s, total_10)
                        )
            seen_lap_times = len(times)
        if on_step is not None:
            live_info = dict(info)
            live_info["_evaluation_metrics"] = {
                "frontier_distance_m": frontier_distance,
                "frontier_speed_mps": frontier_distance / simulated_seconds,
                "simulated_seconds": simulated_seconds,
                "laps_observed": completed_laps,
                "lap_times_s": list(lap_times),
                "lap_diagnostics": list(lap_diagnostics),
                "best_10_lap_time_s": best_10_lap_time_s,
                "reward_per_simulated_second": total_reward / simulated_seconds,
                "total_reward": total_reward,
                "collisions": collisions,
                "initial_action_trace": list(initial_action_trace),
            }
            on_step(live_info)
        if completed_laps >= lap_target:
            stop_reason = "lap_target"
        if bool(dones[0]):
            reason = str(info.get("termination_reason") or info.get("truncate_reason") or "other")
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
            failed_episodes += int(reason not in {"collision", "lap_target"})
            seen_lap_times = 0
            current_life_laps.clear()
            stop_reason = stop_reason or reason

    return {
        "timesteps": int(timesteps),
        "simulated_seconds": simulated_seconds,
        "frontier_distance_m": frontier_distance,
        "frontier_speed_mps": frontier_distance / simulated_seconds,
        "reward_per_simulated_second": total_reward / simulated_seconds,
        "total_reward": total_reward,
        "collisions": collisions,
        "failed_episodes": failed_episodes,
        "laps_observed": completed_laps,
        "lap_times_s": lap_times,
        "lap_diagnostics": lap_diagnostics,
        "initial_action_trace": initial_action_trace,
        "best_10_lap_time_s": best_10_lap_time_s,
        "termination_reasons": termination_reasons,
        "stop_reason": stop_reason,
    }


def evaluate_policy(
    model_path: Path,
    *,
    map_id: str,
    frame_skip: int = 4,
    port_start: int = 4567,
    device: str = "auto",
    headless: bool = True,
    auto_launch: bool = False,
    connect_timeout: float = 90.0,
    output_path: Optional[Path] = None,
    action_interval_s: Optional[float] = None,
    steering_action_scale: float = 1.0,
    throttle_mode: str = "bidirectional",
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    observation_profile: str = "simulator_camera",
) -> Dict[str, Any]:
    """Run a deterministic policy until failure or ten laps, then save metrics."""
    from stable_baselines3 import PPO

    from src.layer3.envs import make_vec_env

    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO model does not exist: {model_path}")
    env = make_vec_env(
        1,
        seed=0,
        port_start=port_start,
        map_id=map_id,
        laps_per_episode=0,
        frame_skip=frame_skip,
        action_interval_s=action_interval_s,
        steering_action_scale=steering_action_scale,
        throttle_mode=throttle_mode,
        straight_throttle_gain=straight_throttle_gain,
        straight_throttle_steering_threshold=straight_throttle_steering_threshold,
        observation_profile=observation_profile,
        max_episode_steps=0,
        frontier_stagnation_seconds=10.0,
        terminate_on_collision=True,
        headless=headless,
        auto_launch=auto_launch,
        connect_timeout=connect_timeout,
    )
    try:
        # A previous trainer process may have exited while Unity was paused;
        # resume the player itself rather than trusting this new Racer's local
        # pause flag, then reset from a live physics frame.
        env.env_method("resume_simulation")
        model = PPO.load(str(model_path), env=env, device=device)
        summary = evaluate_until_episode_end(model, env, frame_skip=frame_skip)
        result = {
            "model": str(model_path),
            "map_id": map_id,
            "lap_target": 10,
            "frame_skip": frame_skip,
            "action_interval_s": action_interval_s,
            "steering_action_scale": steering_action_scale,
            "throttle_mode": throttle_mode,
            "straight_throttle_gain": straight_throttle_gain,
            "straight_throttle_steering_threshold": straight_throttle_steering_threshold,
            "observation_profile": observation_profile,
            "deterministic": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        }
        target = output_path or model_path.parent / f"{model_path.stem}_evaluation.json"
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print(f"saved evaluation: {target}")
        return result
    finally:
        env.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic Layer 3 policy evaluation")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--action-interval-s", type=float, default=None)
    parser.add_argument("--steering-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--throttle-mode",
        choices=("bidirectional", "forward_only"),
        default="bidirectional",
        help="Use the actuator mapping the policy was trained with",
    )
    parser.add_argument("--straight-throttle-gain", type=float, default=1.0)
    parser.add_argument("--straight-throttle-steering-threshold", type=float, default=0.15)
    parser.add_argument(
        "--observation-profile",
        choices=("simulator", "simulator_camera", "official_sensors", "official_sensors_history", "official_sensors_camera"),
        default="simulator_camera",
    )
    parser.add_argument("--port-start", type=int, default=4567)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--connect-timeout", type=float, default=90.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    evaluate_policy(
        args.model,
        map_id=args.map_id,
        frame_skip=args.frame_skip,
        action_interval_s=args.action_interval_s,
        steering_action_scale=args.steering_action_scale,
        throttle_mode=args.throttle_mode,
        straight_throttle_gain=args.straight_throttle_gain,
        straight_throttle_steering_threshold=args.straight_throttle_steering_threshold,
        observation_profile=args.observation_profile,
        port_start=args.port_start,
        device=args.device,
        headless=args.headless,
        auto_launch=args.auto_launch,
        connect_timeout=args.connect_timeout,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
