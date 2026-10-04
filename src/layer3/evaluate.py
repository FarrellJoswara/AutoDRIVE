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
    current_life_laps: List[float] = []
    seen_lap_times = 0
    best_10_lap_time_s: Optional[float] = None
    termination_reasons: Dict[str, int] = {}

    stop_reason: Optional[str] = None
    while stop_reason is None:
        action, _ = model.predict(obs, deterministic=True)
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
        frontier_distance += max(0.0, float(info.get("frontier_advanced_m", 0.0) or 0.0))
        collisions += int(bool(info.get("collision_event", False)))
        if bool(info.get("collision_event", False)):
            stop_reason = "collision"
        times = info.get("lap_times_s")
        if isinstance(times, (list, tuple)):
            if len(times) < seen_lap_times:
                seen_lap_times = 0
            for raw in times[seen_lap_times:]:
                try:
                    lap_time = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(lap_time) and lap_time > 0:
                    lap_times.append(lap_time)
                    current_life_laps.append(lap_time)
                    completed_laps += 1
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
                "best_10_lap_time_s": best_10_lap_time_s,
                "reward_per_simulated_second": total_reward / simulated_seconds,
                "total_reward": total_reward,
                "collisions": collisions,
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
        max_episode_steps=0,
        frontier_stagnation_seconds=5.0,
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
