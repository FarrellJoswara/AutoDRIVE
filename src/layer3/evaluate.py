"""Deterministic, repeatable Layer 3 policy evaluation on a selected map."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def summarize_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comparable lap success and pace metrics from terminal infos."""
    clean = [row for row in episodes if row.get("episode_won")]
    lap_times = [
        float(row["best_lap_time_s"])
        for row in clean
        if row.get("best_lap_time_s") is not None
        and math.isfinite(float(row["best_lap_time_s"]))
        and float(row["best_lap_time_s"]) > 0.0
    ]
    return {
        "episodes": len(episodes),
        "clean_episode_wins": len(clean),
        "success_rate": len(clean) / len(episodes) if episodes else 0.0,
        "completed_laps": sum(max(0, int(row.get("lap_count", 0) or 0)) for row in episodes),
        "best_lap_time_s": min(lap_times) if lap_times else None,
        "mean_clean_lap_time_s": sum(lap_times) / len(lap_times) if lap_times else None,
        "termination_reasons": {
            reason: sum(1 for row in episodes if row.get("termination_reason") == reason)
            for reason in sorted({
                str(row.get("termination_reason"))
                for row in episodes
                if row.get("termination_reason")
            })
        },
    }


def evaluate_policy(
    model_path: Path,
    *,
    map_id: str,
    episodes: int = 5,
    laps_per_episode: int = 1,
    frame_skip: int = 4,
    max_episode_steps: int = 5_000,
    device: str = "auto",
    headless: bool = True,
    auto_launch: bool = False,
    connect_timeout: float = 90.0,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a deterministic policy over independent resets and persist results."""
    from stable_baselines3 import PPO

    from src.layer3.envs import make_vec_env

    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"PPO model does not exist: {model_path}")
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    if laps_per_episode < 1:
        raise ValueError("evaluation requires a positive lap target")

    env = make_vec_env(
        1,
        seed=0,
        map_id=map_id,
        laps_per_episode=laps_per_episode,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
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
        obs = env.reset()
        rows: List[Dict[str, Any]] = []
        while len(rows) < episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            if bool(dones[0]):
                info = infos[0]
                rows.append({
                    "episode": len(rows) + 1,
                    "episode_won": bool(info.get("episode_won", False)),
                    "lap_count": int(info.get("lap_count", 0) or 0),
                    "best_lap_time_s": info.get("best_lap_time_s"),
                    "completed_attempt_average_frontier_speed_mps": info.get(
                        "completed_attempt_average_frontier_speed_mps"
                    ),
                    "termination_reason": info.get("termination_reason")
                    or info.get("truncate_reason")
                    or "unknown",
                    "position": list(info.get("position") or ()),
                    "yaw": info.get("yaw"),
                    "speed_mps": info.get("true_speed"),
                    "current_progress_m": info.get("current_progress_m"),
                    "frontier_progress_m": info.get("frontier_progress_m"),
                    "terminal_reward": float(rewards[0]),
                })
                print(
                    f"evaluation episode {len(rows)}/{episodes}: "
                    f"win={rows[-1]['episode_won']} laps={rows[-1]['lap_count']} "
                    f"reason={rows[-1]['termination_reason']} "
                    f"best_lap={rows[-1]['best_lap_time_s']}",
                    flush=True,
                )
        result = {
            "model": str(model_path),
            "map_id": map_id,
            "laps_per_episode": laps_per_episode,
            "frame_skip": frame_skip,
            "deterministic": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summarize_episodes(rows),
            "episodes": rows,
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--laps-per-episode", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=5_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--connect-timeout", type=float, default=90.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    evaluate_policy(
        args.model,
        map_id=args.map_id,
        episodes=args.episodes,
        laps_per_episode=args.laps_per_episode,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        headless=args.headless,
        auto_launch=args.auto_launch,
        connect_timeout=args.connect_timeout,
        output_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
