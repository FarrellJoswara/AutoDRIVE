"""Load a trained PPO checkpoint and drive (no learning)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def rollout(
    *,
    model_path: Path,
    port: int = 4567,
    steps: int = 2000,
    seed: int = 0,
    device: str = "auto",
    env_kwargs: Optional[dict] = None,
) -> int:
    import torch
    from stable_baselines3 import PPO

    from src.layer3.envs import build_env

    env_kwargs = dict(env_kwargs or {})
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"device: cuda ({torch.cuda.get_device_name(0)})")
        else:
            device = "cpu"
            print("WARN: CUDA unavailable — play on CPU (last resort)")
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback for --device cuda")

    env = build_env(port=port, seed=seed, **env_kwargs)
    try:
        model = PPO.load(str(model_path), env=env, device=device)
        obs, info = env.reset(seed=seed)
        print(f"play: model={model_path} port={port} steps={steps} device={device}")

        for i in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if i == 0 or (i + 1) % 50 == 0:
                print(
                    f"  step {i + 1}: v_long={info.get('v_long', float('nan')):.3f} "
                    f"reward={float(reward):.3f} trunc={truncated}"
                )
            if terminated or truncated:
                reason = info.get("truncate_reason")
                print(f"  episode end at step {i + 1} reason={reason}; reset")
                obs, info = env.reset()
        print("play: done")
        return 0
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 3 PPO play / eval")
    p.add_argument("--model", type=Path, required=True, help="Path to PPO .zip")
    p.add_argument("--port", type=int, default=4567)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-timeout", type=float, default=90.0)
    p.add_argument("--forward-scale", type=float, default=1.0)
    p.add_argument("--collision-penalty", type=float, default=0.0)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.model.exists():
        # SB3 save() may be given without .zip suffix
        alt = Path(str(args.model) + ".zip")
        if alt.exists():
            args.model = alt
        else:
            print(f"ERROR: model not found: {args.model}")
            return 1

    from src.layer3.envs import env_kwargs_from_args

    return rollout(
        model_path=args.model,
        port=args.port,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        env_kwargs=env_kwargs_from_args(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
