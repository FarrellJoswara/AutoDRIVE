"""PPO training entrypoint for Layer 3."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _device_arg(value: str) -> str:
    v = value.lower().strip()
    if v not in {"auto", "cpu", "cuda"}:
        raise argparse.ArgumentTypeError("device must be auto|cpu|cuda")
    return v


def _resolve_device(requested: str) -> str:
    """Prefer CUDA. Fall back to CPU only when CUDA truly unavailable."""
    import torch

    cuda_ok = torch.cuda.is_available()
    if requested == "cpu":
        print("WARN: --device cpu forced; GPU preferred for PPO updates")
        return "cpu"
    if requested in {"auto", "cuda"}:
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            print(f"device: cuda ({name})")
            return "cuda"
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested (--device cuda) but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch (see requirements / LAYER3.md). "
                "Refusing to silently train on CPU."
            )
        print(
            "WARN: CUDA unavailable — falling back to CPU (last resort). "
            "Install torch+cu12x for the RTX GPU."
        )
        return "cpu"
    return requested


def make_model(vec_env, *, device: str, seed: int, tensorboard_log: Optional[str]):
    from stable_baselines3 import PPO

    from src.layer3.extractors import LidarStateExtractor

    policy_kwargs = dict(
        features_extractor_class=LidarStateExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    return PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
    )


def train(
    *,
    n_envs: int = 1,
    base_port: int = 4567,
    timesteps: int = 10_000,
    out_dir: Path,
    seed: int = 0,
    device: str = "auto",
    resume: Optional[Path] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from src.layer3.envs import make_vec_env

    env_kwargs = dict(env_kwargs or {})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    tb_dir = out_dir / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    print(
        f"train: n_envs={n_envs} base_port={base_port} timesteps={timesteps} "
        f"device={resolved_device} out={out_dir}"
    )

    vec_env = make_vec_env(n_envs, base_port=base_port, seed=seed, **env_kwargs)
    try:
        if resume is not None:
            print(f"resuming from {resume}")
            model = PPO.load(str(resume), env=vec_env, device=resolved_device)
        else:
            model = make_model(
                vec_env,
                device=resolved_device,
                seed=seed,
                tensorboard_log=str(tb_dir),
            )

        config = {
            "n_envs": n_envs,
            "base_port": base_port,
            "ports": list(range(base_port, base_port + n_envs)),
            "timesteps": timesteps,
            "seed": seed,
            "device": resolved_device,
            "env_kwargs": env_kwargs,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resume": str(resume) if resume else None,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        # For short smokes, still checkpoint once near the end if long enough.
        save_freq = max(1, min(10_000, timesteps // max(n_envs, 1)))
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(ckpt_dir),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        model.learn(total_timesteps=int(timesteps), callback=checkpoint_cb, progress_bar=False)
        final_path = out_dir / "final_model"
        model.save(str(final_path))
        print(f"saved {final_path}.zip")
        return Path(str(final_path) + ".zip")
    finally:
        vec_env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 3 PPO train")
    p.add_argument("--n-envs", type=int, default=1, help="1=DummyVecEnv; >=2=SubprocVecEnv")
    p.add_argument("--base-port", type=int, default=4567)
    p.add_argument("--timesteps", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=None, help="Run directory under logs/rl/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=_device_arg, default="auto")
    p.add_argument("--resume", type=Path, default=None, help="Optional PPO .zip to continue")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-timeout", type=float, default=90.0)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=0)
    p.add_argument("--stagnation-speed-threshold", type=float, default=0.15)
    p.add_argument("--stagnation-steps", type=int, default=200)
    p.add_argument("--forward-scale", type=float, default=1.0)
    p.add_argument("--collision-penalty", type=float, default=0.0)
    p.add_argument("--slip-penalty", type=float, default=0.0)
    p.add_argument("--steer-jerk-penalty", type=float, default=0.0)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.n_envs < 1:
        print("ERROR: --n-envs must be >= 1")
        return 1

    from src.layer3.envs import env_kwargs_from_args

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or (ROOT / "logs" / "rl" / f"ppo_{stamp}")
    zip_path = train(
        n_envs=args.n_envs,
        base_port=args.base_port,
        timesteps=args.timesteps,
        out_dir=out,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        env_kwargs=env_kwargs_from_args(args),
    )
    print(f"done: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
