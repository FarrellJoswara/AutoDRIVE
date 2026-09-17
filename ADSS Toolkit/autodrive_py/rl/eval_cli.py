"""Phase 6: cross-backend eval CLI (gym / autodrive; ftg / ppo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import N_LIDAR_DEFAULT
from .ftg import FollowTheGap
from .metrics_io import make_metrics
from .racing_env import RacingEnv, resolve_map_yaml
from .run_ftg import run_episode


def eval_gym_ftg(map_yaml: Path, episodes: int, n_lidar: int, seed: int):
    env = RacingEnv(map_yaml=map_yaml, n_lidar=n_lidar, seed=seed)
    policy = FollowTheGap(n_lidar=n_lidar)
    laps, cols, rets = [], 0, []
    for _ in range(episodes):
        r = run_episode(env, policy)
        cols += r["collisions"]
        rets.append(r["return"])
        if r["lap_time"] is not None:
            laps.append(r["lap_time"])
    mean_lap = float(sum(laps) / len(laps)) if laps else None
    return make_metrics(
        run_id="eval_ftg_gym",
        policy="ftg",
        backend="gym",
        mean_lap_time=mean_lap,
        total_collisions=cols,
        n_episodes=episodes,
        tracks_eval=[map_yaml.stem],
        mean_return=float(np.mean(rets)),
        n_lidar=n_lidar,
        seed=seed,
    )


def eval_gym_ppo(map_yaml: Path, model_path: Path, episodes: int, n_lidar: int, seed: int):
    from stable_baselines3 import PPO

    env = RacingEnv(map_yaml=map_yaml, n_lidar=n_lidar, seed=seed)
    model = PPO.load(str(model_path))
    laps, cols, rets = [], 0, []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            if info.get("collision"):
                cols += 1
            if info.get("lap_time") is not None:
                laps.append(float(info["lap_time"]))
            done = terminated or truncated
        rets.append(ep_ret)
    mean_lap = float(sum(laps) / len(laps)) if laps else None
    return make_metrics(
        run_id="eval_ppo_gym",
        policy="ppo",
        backend="gym",
        mean_lap_time=mean_lap,
        total_collisions=cols,
        n_episodes=episodes,
        tracks_eval=[map_yaml.stem],
        mean_return=float(np.mean(rets)),
        n_lidar=n_lidar,
        seed=seed,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6: cross-eval CLI")
    parser.add_argument("--backend", choices=["gym", "autodrive"], default="gym")
    parser.add_argument("--policy", choices=["ftg", "ppo"], default="ftg")
    parser.add_argument("--map", type=str, default="map0")
    parser.add_argument("--model", type=str, default="", help="Path to SB3 zip for ppo")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--n_lidar", type=int, default=N_LIDAR_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.backend == "autodrive":
        print(
            "ERROR: AutoDRIVE live eval requires the Unity simulator connected to "
            "rl.bridge_ftg (port 4567). Start the sim, run bridge_ftg, then re-run eval."
        )
        return 2

    maps_root = Path(__file__).resolve().parent / "maps"
    try:
        map_yaml = resolve_map_yaml(args.map, maps_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.policy == "ftg":
        metrics = eval_gym_ftg(map_yaml, args.episodes, args.n_lidar, args.seed)
    else:
        if not args.model:
            # try latest ppo under models/
            models = sorted((Path(__file__).resolve().parent / "models").glob("*/best_model.zip"))
            if not models:
                print("ERROR: --policy ppo requires --model path (no best_model.zip found).")
                return 2
            model_path = models[-1]
        else:
            model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: model not found: {model_path}")
            return 2
        metrics = eval_gym_ppo(map_yaml, model_path, args.episodes, args.n_lidar, args.seed)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
