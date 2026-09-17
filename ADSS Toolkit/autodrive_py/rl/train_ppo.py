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


def _parse_net_arch(s: str) -> list[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        return [256, 256]
    return [int(p) for p in parts]


def _make_env(map_yaml: Path, n_lidar: int, seed: int, rank: int):
    def _thunk():
        from stable_baselines3.common.monitor import Monitor

        env = RacingEnv(map_yaml=map_yaml, n_lidar=n_lidar, seed=seed + rank)
        # Expose collision into Monitor episode dict for live_status estimates
        return Monitor(env, info_keywords=("collision",))

    return _thunk


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
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2048,
        help="Use ~2048 for smoke; raise for real training (start_train.ps1 defaults higher)",
    )
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
        "--n-envs",
        type=int,
        default=8,
        help="Parallel envs (DummyVecEnv on Windows by default; more = higher throughput)",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=("dummy", "subproc"),
        default="dummy",
        help="dummy = DummyVecEnv (often better on Windows); subproc = SubprocVecEnv",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout length per env (total batch ≈ n_steps * n_envs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Minibatch size for PPO updates (larger = more GPU work)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per rollout (more = more GPU update time)",
    )
    parser.add_argument(
        "--net-arch",
        type=str,
        default="256,256",
        help="MLP hidden sizes, comma-separated (e.g. 512,512 for heavier GPU load)",
    )
    parser.add_argument(
        "--tb",
        action="store_true",
        default=True,
        help="Log to TensorBoard under rl/runs/ (default on)",
    )
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save SB3 checkpoint every N timesteps (0 disables). Survives interrupted runs.",
    )
    parser.add_argument(
        "--status-every-rollouts",
        type=int,
        default=1,
        help="Write rl/runs/<run_id>/live_status.json every N PPO rollouts (cheap; no render).",
    )
    parser.add_argument(
        "--no-live-status",
        action="store_true",
        help="Disable live_status.json trail (watch --follow needs it).",
    )
    parser.add_argument(
        "--save-latest-every-rollouts",
        type=int,
        default=1,
        help="Also save models/<run_id>/latest_model.zip every N rollouts (0 disables).",
    )
    args = parser.parse_args(argv)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .live_status import LiveStatusCallback

    device = resolve_device(args.device)
    n_envs = max(1, int(args.n_envs))
    net_arch = _parse_net_arch(args.net_arch)
    n_steps = max(64, int(args.n_steps))
    batch_size = max(32, int(args.batch_size))
    n_epochs = max(1, int(args.n_epochs))

    # Keep batch_size compatible with rollout buffer size
    buffer_size = n_steps * n_envs
    if batch_size > buffer_size:
        batch_size = buffer_size
        print(f"NOTE: batch_size clamped to buffer size {batch_size} (n_steps*n_envs)")

    print(f"torch={torch.__version__} device={device}", end="")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(
        f"vec={args.vec_env} n_envs={n_envs} n_steps={n_steps} "
        f"batch_size={batch_size} n_epochs={n_epochs} net_arch={net_arch}"
    )

    maps_root = Path(__file__).resolve().parent / "maps"
    models_root = Path(__file__).resolve().parent / "models"
    runs_root = Path(__file__).resolve().parent / "runs"
    map_yaml = resolve_map_yaml(args.map, maps_root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{stamp}_ppo_gym_{map_yaml.stem}"

    env_fns = [_make_env(map_yaml, args.n_lidar, args.seed, i) for i in range(n_envs)]
    if args.vec_env == "subproc" and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        # DummyVecEnv: sequential but no Windows spawn/pickle pain; still multiplies
        # rollout size and GPU update work via n_envs * n_steps.
        vec_env = DummyVecEnv(env_fns)

    tb_log = None if args.no_tb else str(runs_root / run_id)
    if tb_log:
        Path(tb_log).mkdir(parents=True, exist_ok=True)
        print(f"TensorBoard: tensorboard --logdir \"{runs_root}\"")
        print("  then open http://localhost:6006  (metrics only, no sim cost)")

    policy_kwargs = dict(net_arch=net_arch)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log,
    )

    run_dir_early = models_root / run_id
    run_dir_early.mkdir(parents=True, exist_ok=True)
    status_dir = runs_root / run_id
    status_dir.mkdir(parents=True, exist_ok=True)
    callbacks = []
    if args.checkpoint_every and args.checkpoint_every > 0:
        ckpt_dir = run_dir_early / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_every) // max(1, n_envs)),
                save_path=str(ckpt_dir),
                name_prefix="ppo",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
        print(f"Checkpoints every ~{args.checkpoint_every} steps -> {ckpt_dir}")

    if not args.no_live_status:
        latest_path = None
        save_latest_every = int(args.save_latest_every_rollouts)
        if save_latest_every > 0:
            latest_path = run_dir_early / "latest_model.zip"
        status_path = status_dir / "live_status.json"
        callbacks.append(
            LiveStatusCallback(
                status_path=status_path,
                run_id=run_id,
                every_rollouts=max(1, int(args.status_every_rollouts)),
                latest_model_path=latest_path,
                save_latest_every_rollouts=max(1, save_latest_every) if latest_path else 1,
            )
        )
        print(f"Live status -> {status_path}")
        if latest_path:
            print(f"Latest weights every {save_latest_every} rollout(s) -> {latest_path}")
        print("Follow (second terminal): python -m rl.watch --follow")

    t0 = time.time()
    model.learn(
        total_timesteps=int(args.timesteps),
        progress_bar=False,
        tb_log_name="ppo",
        callback=callbacks or None,
    )
    train_s = time.time() - t0
    vec_env.close()

    eval_env = RacingEnv(map_yaml=map_yaml, n_lidar=args.n_lidar, seed=args.seed + 10_000)
    mean_lap, collisions, mean_ret = _eval_policy(eval_env, model, episodes=args.eval_episodes)

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
        "obs_include_speed": True,
        "obs_include_imu": True,
        "imu_dim": 3,
        "obs_dim": int(env.observation_space.shape[-1]),
        "speed_max_mps": SPEED_MAX_MPS,
        "timeout_s": TIMEOUT_S,
        "timesteps": args.timesteps,
        "map": str(map_yaml),
        "device": device,
        "n_envs": n_envs,
        "vec_env": args.vec_env,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "net_arch": net_arch,
    }
    run_dir = write_run_artifacts(models_root, run_id, config, metrics, train_tracks=[map_yaml.stem])
    model_path = run_dir / "best_model"
    model.save(str(model_path))
    print(f"saved {model_path}.zip")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
