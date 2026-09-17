"""Phase 6: cross-backend eval CLI (gym / autodrive; ftg / ppo) + official protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compat import ContractsMismatch, load_ppo_refusing_mismatch
from .contracts import N_LIDAR_DEFAULT
from .eval_protocol import eval_ftg_protocol, eval_ppo_protocol, load_protocol
from .ftg import FollowTheGap
from .map_pack import SealBroken, assert_seals_intact
from .metrics_io import append_leaderboard, make_metrics
from .racing_env import RacingEnv, resolve_map_yaml
from .run_ftg import run_episode


def eval_gym_ftg(map_yaml: Path, episodes: int, n_lidar: int, seed: int, *, kind: str = "smoke"):
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
        kind=kind,
        mean_return=float(np.mean(rets)),
        n_lidar=n_lidar,
        seed=seed,
    )


def eval_gym_ppo(
    map_yaml: Path,
    model_path: Path,
    episodes: int,
    n_lidar: int,
    seed: int,
    *,
    kind: str = "smoke",
):
    env = RacingEnv(map_yaml=map_yaml, n_lidar=n_lidar, seed=seed)
    model = load_ppo_refusing_mismatch(model_path, n_lidar=n_lidar)
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
        kind=kind,
        mean_return=float(np.mean(rets)),
        n_lidar=n_lidar,
        seed=seed,
        model_path=str(model_path),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cross-eval CLI (+ official protocol)")
    parser.add_argument("--backend", choices=["gym", "autodrive"], default="gym")
    parser.add_argument("--policy", choices=["ftg", "ppo"], default="ftg")
    parser.add_argument("--map", type=str, default="map0")
    parser.add_argument("--model", type=str, default="", help="Path to SB3 zip for ppo")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Episodes per map. 0 = use the protocol's episodes_per_map "
        "(official runs must not quietly shrink their seed count)",
    )
    parser.add_argument("--n_lidar", type=int, default=N_LIDAR_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--official",
        action="store_true",
        help="Run frozen eval_protocol.yaml (map0 + holdout) and append kind=official",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Append metrics to models/leaderboard.csv (official always appends)",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="",
        help="Optional path to protocol YAML (default: rl/eval_protocol.yaml)",
    )
    args = parser.parse_args(argv)

    if args.backend == "autodrive":
        print(
            "ERROR: AutoDRIVE live eval requires the Unity simulator connected to "
            "rl.bridge_ftg (port 4567). Start the sim, run bridge_ftg, then re-run eval."
        )
        return 2

    models_root = Path(__file__).resolve().parent / "models"
    maps_root = Path(__file__).resolve().parent / "maps"

    try:
        if args.official:
            proto = load_protocol(Path(args.protocol) if args.protocol else None)
            if args.policy == "ftg":
                metrics = eval_ftg_protocol(
                    protocol=proto,
                    maps_root=maps_root,
                    n_lidar=args.n_lidar,
                    episodes_override=args.episodes if args.episodes > 0 else None,
                )
            else:
                if not args.model:
                    models = sorted(models_root.glob("*/best_model.zip"))
                    if not models:
                        print("ERROR: --policy ppo --official requires --model")
                        return 2
                    model_path = models[-1]
                else:
                    model_path = Path(args.model)
                if not model_path.exists():
                    print(f"ERROR: model not found: {model_path}")
                    return 2
                metrics = eval_ppo_protocol(
                    model_path,
                    protocol=proto,
                    maps_root=maps_root,
                    n_lidar=args.n_lidar,
                    episodes_override=args.episodes if args.episodes > 0 else None,
                )
            # A regenerated or edited holdout would silently invalidate the claim.
            try:
                assert_seals_intact(maps_root)
            except SealBroken as exc:
                print(f"ERROR: refusing to append official row - {exc}")
                return 2
            append_leaderboard(models_root / "leaderboard.csv", metrics)
            print(json.dumps(metrics, indent=2))
            print(f"appended official row -> {models_root / 'leaderboard.csv'}")
            return 0

        episodes = max(1, int(args.episodes))
        map_yaml = resolve_map_yaml(args.map, maps_root)
        if args.policy == "ftg":
            metrics = eval_gym_ftg(map_yaml, episodes, args.n_lidar, args.seed)
        else:
            if not args.model:
                models = sorted(models_root.glob("*/best_model.zip"))
                if not models:
                    print("ERROR: --policy ppo requires --model path (no best_model.zip found).")
                    return 2
                model_path = models[-1]
            else:
                model_path = Path(args.model)
            if not model_path.exists():
                print(f"ERROR: model not found: {model_path}")
                return 2
            metrics = eval_gym_ppo(map_yaml, model_path, episodes, args.n_lidar, args.seed)

        if args.save and metrics.get("adjusted_time") is not None:
            append_leaderboard(models_root / "leaderboard.csv", metrics)
            print(f"appended smoke row -> {models_root / 'leaderboard.csv'}")
        elif args.save:
            print("NOTE: refused to append null adjusted_time smoke row")

        print(json.dumps(metrics, indent=2))
        return 0
    except ContractsMismatch as exc:
        print(f"ERROR: contracts refuse - {exc}")
        return 2
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
