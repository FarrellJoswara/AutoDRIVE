"""Phase 3: run Follow-the-Gap in the gym backend and print competition-shaped metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import CONTRACTS_VERSION, N_LIDAR_DEFAULT, TIMEOUT_S, obs_dim
from .ftg import FollowTheGap
from .metrics_io import make_metrics, write_run_artifacts
from .racing_env import RacingEnv, assert_resolved_map_id, resolve_map_yaml


def run_episode(env: RacingEnv, policy: FollowTheGap, max_steps: int | None = None):
    obs, info = env.reset()
    done = False
    steps = 0
    total_return = 0.0
    collisions = 0
    lap_time = None
    limit = max_steps or env.max_steps
    while not done and steps < limit:
        # Prefer raw scan from env for FTG (meters)
        raw = env._raw_scan
        action = policy.act(raw)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        steps += 1
        if info.get("collision"):
            collisions += 1
        if info.get("lap_time") is not None:
            lap_time = info["lap_time"]
        done = terminated or truncated
    return {
        "return": total_return,
        "collisions": collisions,
        "lap_time": lap_time,
        "steps": steps,
        "timeout": bool(info.get("timeout")),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: Gym FTG baseline")
    parser.add_argument("--map", type=str, default="map0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_lidar", type=int, default=N_LIDAR_DEFAULT)
    parser.add_argument("--save", action="store_true", help="Write models/<run_id> artifacts")
    args = parser.parse_args(argv)

    maps_root = Path(__file__).resolve().parent / "maps"
    try:
        map_yaml = resolve_map_yaml(args.map, maps_root, strict=True)
        assert_resolved_map_id(args.map, map_yaml)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    env = RacingEnv(map_yaml=map_yaml, n_lidar=args.n_lidar, seed=args.seed)
    policy = FollowTheGap(n_lidar=args.n_lidar)

    lap_times = []
    total_collisions = 0
    returns = []
    for ep in range(args.episodes):
        result = run_episode(env, policy)
        returns.append(result["return"])
        total_collisions += result["collisions"]
        if result["lap_time"] is not None:
            lap_times.append(result["lap_time"])
        print(
            f"episode={ep} steps={result['steps']} return={result['return']:.3f} "
            f"collisions={result['collisions']} lap_time={result['lap_time']} timeout={result['timeout']}"
        )

    mean_lap = float(sum(lap_times) / len(lap_times)) if lap_times else None
    metrics = make_metrics(
        run_id=f"ftg_gym_{Path(map_yaml).stem}",
        policy="ftg",
        backend="gym",
        mean_lap_time=mean_lap,
        total_collisions=total_collisions,
        n_episodes=args.episodes,
        tracks_eval=[str(map_yaml.stem)],
        mean_return=float(sum(returns) / max(len(returns), 1)),
        n_lidar=args.n_lidar,
        timeout_s=TIMEOUT_S,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))

    if args.save:
        models_root = Path(__file__).resolve().parent / "models"
        config = {
            "contracts_version": CONTRACTS_VERSION,
            "run_id": metrics["run_id"],
            "policy": "ftg",
            "backend": "gym",
            "n_lidar": args.n_lidar,
            "action_space": "MultiDiscrete([4, 11])",
            "throttle_bins": [0.0, 0.33, 0.66, 1.0],
            "steering_bins": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "obs_include_speed": True,
            "obs_include_imu": True,
            "imu_dim": 3,
            "obs_dim": obs_dim(args.n_lidar),
            "timeout_s": TIMEOUT_S,
        }
        write_run_artifacts(models_root, metrics["run_id"], config, metrics)
        print(f"saved {models_root / metrics['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
