"""Phase 4: PPO training on RacingEnv + model registry artifacts."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .contracts import (
    CONTRACTS_VERSION,
    N_LIDAR_DEFAULT,
    STEERING_BINS,
    THROTTLE_BINS,
    TIMEOUT_S,
)
from .metrics_io import make_metrics, write_run_artifacts
from .racing_env import RacingEnv, resolve_map_yaml


def resolve_device(requested: str) -> str:
    import torch

    req = (requested or "auto").lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return req


def _eval_policy(env: RacingEnv, model, episodes: int = 3):
    lap_times = []
    collisions = 0
    returns = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            if info.get("collision"):
                collisions += 1
            if info.get("lap_time") is not None:
                lap_times.append(float(info["lap_time"]))
            done = terminated or truncated
        returns.append(ep_ret)
    mean_lap = float(sum(lap_times) / len(lap_times)) if lap_times else None
    return mean_lap, collisions, float(np.mean(returns))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4: Gym PPO smoke / train")
    parser.add_argument("--map", type=str, default="map0")
    parser.add_argument("--timesteps", type=int, default=2048, help="Use ~2048 for smoke; raise for real training")
    parser.add_argument("--n_lidar", type=int, default=N_LIDAR_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0 (auto prefers GPU)",
    )
    parser.add_argument(
        "--tb",
        action="store_true",
        default=True,
        help="Log to TensorBoard under rl/runs/ (default on)",
    )
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    args = parser.parse_args(argv)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    device = resolve_device(args.device)
    print(f"torch={torch.__version__} device={device}", end="")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
        # Larger rollout/batch amortizes CPU→GPU transfer for small MLP policies
        n_steps = min(4096, max(1024, args.timesteps // 4)) if args.timesteps >= 2048 else min(512, args.timesteps)
        batch_size = 256
    else:
        print()
        n_steps = min(2048, max(256, args.timesteps // 4)) if args.timesteps >= 2048 else min(512, args.timesteps)
        batch_size = 64

    maps_root = Path(__file__).resolve().parent / "maps"
    models_root = Path(__file__).resolve().parent / "models"
    runs_root = Path(__file__).resolve().parent / "runs"
    map_yaml = resolve_map_yaml(args.map, maps_root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{stamp}_ppo_gym_{map_yaml.stem}"

    env = Monitor(RacingEnv(map_yaml=map_yaml, n_lidar=args.n_lidar, seed=args.seed))
    tb_log = None if args.no_tb else str(runs_root / run_id)
    if tb_log:
        Path(tb_log).mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard: tensorboard --logdir \"{runs_root}\"")
        print("  then open http://localhost:6006  (metrics only, no sim cost)")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        device=device,
        tensorboard_log=tb_log,
    )

    t0 = time.time()
    model.learn(total_timesteps=int(args.timesteps), progress_bar=False, tb_log_name="ppo")
    train_s = time.time() - t0

    mean_lap, collisions, mean_ret = _eval_policy(env, model, episodes=args.eval_episodes)
    metrics = make_metrics(
        run_id=run_id,
        policy="ppo",
        backend="gym",
        mean_lap_time=mean_lap,
        total_collisions=collisions,
        n_episodes=args.eval_episodes,
        tracks_eval=[map_yaml.stem],
        mean_return=mean_ret,
        n_lidar=args.n_lidar,
        timeout_s=TIMEOUT_S,
        seed=args.seed,
        train_timesteps=args.timesteps,
        train_seconds=train_s,
        device=device,
    )
    config = {
        "contracts_version": CONTRACTS_VERSION,
        "run_id": run_id,
        "policy": "ppo",
        "backend": "gym",
        "n_lidar": args.n_lidar,
        "action_space": "MultiDiscrete([4, 11])",
        "throttle_bins": THROTTLE_BINS.tolist(),
        "steering_bins": STEERING_BINS.tolist(),
        "obs_include_speed": False,
        "timeout_s": TIMEOUT_S,
        "timesteps": args.timesteps,
        "map": str(map_yaml),
        "device": device,
    }
    run_dir = write_run_artifacts(models_root, run_id, config, metrics, train_tracks=[map_yaml.stem])
    model_path = run_dir / "best_model"
    model.save(str(model_path))
    print(f"saved {model_path}.zip")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
