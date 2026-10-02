"""PPO training entrypoint for Layer 3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Practical first-policy PPO defaults for continuous lidar control.
# Short/medium runs should get frequent updates + some exploration.
_PPO_LR = 3e-4
_PPO_N_STEPS = 1024
_PPO_BATCH_SIZE = 64
_PPO_N_EPOCHS = 8
_PPO_GAMMA = 0.99
_PPO_GAE_LAMBDA = 0.95
_PPO_CLIP_RANGE = 0.2
_PPO_ENT_COEF = 0.01
_PPO_VF_COEF = 0.5
_PPO_MAX_GRAD_NORM = 0.5


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
        learning_rate=_PPO_LR,
        n_steps=_PPO_N_STEPS,
        batch_size=_PPO_BATCH_SIZE,
        n_epochs=_PPO_N_EPOCHS,
        gamma=_PPO_GAMMA,
        gae_lambda=_PPO_GAE_LAMBDA,
        clip_range=_PPO_CLIP_RANGE,
        ent_coef=_PPO_ENT_COEF,
        vf_coef=_PPO_VF_COEF,
        max_grad_norm=_PPO_MAX_GRAD_NORM,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
    )


class RunStopCallback(BaseCallback):
    """Stop Layer 3 training on a duration or any-car lap target."""

    def __init__(self, *, max_duration_seconds: float = 0.0, stop_after_laps: int = 0):
        super().__init__(verbose=0)
        self.max_duration_seconds = max(0.0, float(max_duration_seconds))
        self.stop_after_laps = max(0, int(stop_after_laps))
        self._started_at = 0.0
        self.stop_reason: Optional[str] = None
        self._lap_totals: list[int] = []
        self._episode_laps: list[int] = []

    def _on_training_start(self) -> None:
        self._started_at = time.monotonic()

    def _on_step(self) -> bool:
        if (
            self.max_duration_seconds > 0
            and time.monotonic() - self._started_at >= self.max_duration_seconds
        ):
            self.stop_reason = "max_duration"
            return False

        if self.stop_after_laps > 0:
            infos = self.locals.get("infos") or []
            dones = self.locals.get("dones")
            if dones is None:
                dones = []
            if len(self._lap_totals) != len(infos):
                self._lap_totals = [0] * len(infos)
                self._episode_laps = [0] * len(infos)
            for index, info in enumerate(infos):
                if isinstance(info, dict):
                    current = max(0, int(info.get("lap_count", 0) or 0))
                    previous = self._episode_laps[index]
                    if current >= previous:
                        self._lap_totals[index] += current - previous
                    self._episode_laps[index] = current
                if index < len(dones) and bool(dones[index]):
                    self._episode_laps[index] = 0
            if any(total >= self.stop_after_laps for total in self._lap_totals):
                self.stop_reason = "lap_target"
                return False
        return True


def train(
    *,
    n_envs: int = 1,
    timesteps: int = 50_000,
    out_dir: Path,
    seed: int = 0,
    device: str = "auto",
    resume: Optional[Path] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    max_duration_seconds: float = 0.0,
    stop_after_laps: int = 0,
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
        f"train: n_envs={n_envs} timesteps={timesteps} "
        f"device={resolved_device} out={out_dir}"
    )
    print(
        f"ppo: lr={_PPO_LR} n_steps={_PPO_N_STEPS} batch={_PPO_BATCH_SIZE} "
        f"n_epochs={_PPO_N_EPOCHS} gamma={_PPO_GAMMA} ent_coef={_PPO_ENT_COEF} "
        f"clip={_PPO_CLIP_RANGE}"
    )
    if env_kwargs:
        print(f"env: {env_kwargs}")

    vec_env = make_vec_env(n_envs, seed=seed, **env_kwargs)
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
            "bridge_ports": list(range(4567, 4567 + n_envs)),
            "timesteps": timesteps,
            "stopping": {
                "max_duration_seconds": max(0.0, float(max_duration_seconds)),
                "stop_after_laps": max(0, int(stop_after_laps)),
                "lap_target_semantics": "any_car_total_across_run",
            },
            "seed": seed,
            "device": resolved_device,
            "env_kwargs": env_kwargs,
            "ppo": {
                "learning_rate": _PPO_LR,
                "n_steps": _PPO_N_STEPS,
                "batch_size": _PPO_BATCH_SIZE,
                "n_epochs": _PPO_N_EPOCHS,
                "gamma": _PPO_GAMMA,
                "gae_lambda": _PPO_GAE_LAMBDA,
                "clip_range": _PPO_CLIP_RANGE,
                "ent_coef": _PPO_ENT_COEF,
                "vf_coef": _PPO_VF_COEF,
                "max_grad_norm": _PPO_MAX_GRAD_NORM,
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "features_dim": 256,
            },
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

        stop_cb = RunStopCallback(
            max_duration_seconds=max_duration_seconds,
            stop_after_laps=stop_after_laps,
        )
        callbacks = [checkpoint_cb, stop_cb]
        from src.layer3.hub_callback import maybe_hub_callback

        hub_cb = maybe_hub_callback(run_id=out_dir.name)
        if hub_cb is not None:
            callbacks.append(hub_cb)
            print(f"hub telemetry: HUB_URL set → publishing to hub (run_id={out_dir.name})")

        model.learn(
            total_timesteps=int(timesteps),
            callback=callbacks,
            progress_bar=False,
        )
        stop_record = {
            "reason": stop_cb.stop_reason or "timestep_limit",
            "num_timesteps": int(model.num_timesteps),
            "max_duration_seconds": max(0.0, float(max_duration_seconds)),
            "stop_after_laps": max(0, int(stop_after_laps)),
        }
        (out_dir / "stop_reason.json").write_text(
            json.dumps(stop_record, indent=2), encoding="utf-8"
        )
        print(f"training stopped: {stop_record['reason']} at {model.num_timesteps} timesteps")
        final_path = out_dir / "final_model"
        model.save(str(final_path))
        print(f"saved {final_path}.zip")
        return Path(str(final_path) + ".zip")
    finally:
        vec_env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 3 PPO train")
    p.add_argument("--n-envs", type=int, default=1, help="1=DummyVecEnv; >=2=SubprocVecEnv")
    p.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total env steps (50k ≈ usable first policy; smoke with >=5k)",
    )
    p.add_argument(
        "--max-duration-seconds", type=float, default=0.0,
        help="Optional wall-clock run limit; 0 disables it",
    )
    p.add_argument(
        "--stop-after-laps", type=int, default=0,
        help="Stop when any car completes this many laps; 0 disables it",
    )
    p.add_argument("--out", type=Path, default=None, help="Run directory under logs/rl/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=_device_arg, default="auto")
    p.add_argument("--resume", type=Path, default=None, help="Optional PPO .zip to continue")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-timeout", type=float, default=90.0)
    # ~10 Hz decisions at 40 Hz physics — easier credit assignment than 40 Hz.
    p.add_argument("--frame-skip", type=int, default=4)
    # Hard cap so stuck-but-wiggling episodes still reset (~100 s @ frame_skip=4).
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--stagnation-speed-threshold", type=float, default=0.15)
    # ~5 s idle @ frame_skip=4 before truncate.
    p.add_argument("--stagnation-steps", type=int, default=50)
    p.add_argument(
        "--terminate-on-collision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--map-id", type=str, default="none")
    p.add_argument("--frontier-stagnation-seconds", type=float, default=5.0)
    p.add_argument("--forward-scale", type=float, default=1.0)
    p.add_argument("--route-progress-scale", type=float, default=10.0)
    p.add_argument("--collision-penalty", type=float, default=-5.0)
    p.add_argument("--slip-penalty", type=float, default=0.2)
    p.add_argument("--steer-jerk-penalty", type=float, default=0.05)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.n_envs < 1:
        print("ERROR: --n-envs must be >= 1")
        return 1
    if args.n_envs > 16:
        print("ERROR: --n-envs must be <= 16 (the simulator bridge range)")
        return 1

    from src.layer3.envs import env_kwargs_from_args

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or (ROOT / "logs" / "rl" / f"ppo_{stamp}")
    zip_path = train(
        n_envs=args.n_envs,
        timesteps=args.timesteps,
        out_dir=out,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        env_kwargs=env_kwargs_from_args(args),
        max_duration_seconds=args.max_duration_seconds,
        stop_after_laps=args.stop_after_laps,
    )
    print(f"done: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
