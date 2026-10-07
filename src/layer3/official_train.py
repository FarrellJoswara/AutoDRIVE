"""Train PPO against the official IROS RoboRacer simulator and Devkit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def train_official(
    *,
    out_dir: Path,
    total_timesteps: int,
    seed: int = 0,
    resume: Optional[Path] = None,
    checkpoint_every: int = 10_000,
    n_steps: int = 2048,
    learning_rate: float = 1e-4,
    n_epochs: int = 4,
    gamma: float = 0.9995,
    gae_lambda: float = 0.95,
    timeout_s: float = 180.0,
    training_timeout_s: float = 600.0,
    observation_profile: str = "official_sensors",
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    throttle_mode: str = "bidirectional",
    negative_throttle_mode: str = "allow",
    steering_mode: str = "normal",
) -> Path:
    """Run single-simulator PPO; all vehicle IO uses the official ROS Devkit."""
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    if checkpoint_every < 1 or n_steps < 1 or n_epochs < 1:
        raise ValueError("checkpoint_every, n_steps, and n_epochs must be positive")

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.logger import configure

    from src.layer2.official_race_env import OfficialRaceEnv
    from src.layer3.train import make_model

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "tensorboard").mkdir(exist_ok=True)

    env = OfficialRaceEnv(
        timeout_s=timeout_s,
        warmup_laps=1,
        race_laps=10,
        steering_action_scale=steering_action_scale,
        straight_throttle_gain=straight_throttle_gain,
        straight_throttle_steering_threshold=straight_throttle_steering_threshold,
        throttle_mode=throttle_mode,
        negative_throttle_mode=negative_throttle_mode,
        steering_mode=steering_mode,
        observation_profile=observation_profile,
        training_mode=True,
        training_timeout_s=training_timeout_s,
    )
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    try:
        if resume is None:
            model = make_model(
                vec_env,
                device="cpu",
                seed=seed,
                tensorboard_log=str(out_dir / "tensorboard"),
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                policy_architecture=(
                    "temporal_lidar_cnn"
                    if observation_profile == "official_sensors_history"
                    else "lidar_cnn"
                ),
            )
        else:
            model = PPO.load(str(Path(resume).resolve()), env=vec_env, device="cpu")
            model.learning_rate = float(learning_rate)
            model.n_epochs = int(n_epochs)
            model.gamma = float(gamma)
            model.gae_lambda = float(gae_lambda)
            model.n_steps = int(n_steps)
            model.set_logger(
                configure(
                    folder=str(out_dir / "tensorboard"),
                    format_strings=["stdout", "csv", "tensorboard"],
                )
            )

        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "official IROS 2026 API and simulator images",
            "official_api_image": "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete",
            "official_simulator_image": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete",
            "episode": {
                "warmup_laps_ignored": 1,
                "race_laps": 10,
                "collision_behavior": "official checkpoint reset; terminate only after >10 scored collisions",
                "training_watchdog_s": training_timeout_s,
            },
            "observation_profile": observation_profile,
            "actions": {
                "throttle_mode": throttle_mode,
                "negative_throttle_mode": negative_throttle_mode,
                "steering_mode": steering_mode,
                "steering_action_scale": steering_action_scale,
                "straight_throttle_gain": straight_throttle_gain,
                "straight_throttle_steering_threshold": straight_throttle_steering_threshold,
            },
            "training_only_reward": {
                "time_cost_per_simulated_second": 1.0,
                "lap_completion_bonus": 100.0,
                "warmup_lap_completion_bonus": 100.0,
                "collision_penalty_seconds": "10 * collision_number",
                "disqualification_or_watchdog_penalty": 1000.0,
                "watchdog_scope": "full episode including warm-up",
                "restricted_topics_used_only_for_reward_and_episode_control": True,
                "restricted_topics_in_policy_observation": False,
            },
            "ppo": {
                "total_timesteps": total_timesteps,
                "n_steps": n_steps,
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "device": "cpu",
                "seed": seed,
            },
            "resume": str(Path(resume).resolve()) if resume else None,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, checkpoint_every),
            save_path=str(out_dir / "checkpoints"),
            name_prefix="official_ppo",
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
            reset_num_timesteps=resume is None,
        )
        target = out_dir / "official_policy_latest.zip"
        model.save(str(target.with_suffix("")))
        print(f"Saved official-simulator PPO checkpoint: {target}", flush=True)
        return target
    finally:
        vec_env.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/runs/official"))
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.9995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--training-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--observation-profile",
        choices=("official_sensors", "official_sensors_history"),
        default="official_sensors",
    )
    parser.add_argument("--steering-action-scale", type=float, default=1.0)
    parser.add_argument("--straight-throttle-gain", type=float, default=1.0)
    parser.add_argument("--straight-throttle-steering-threshold", type=float, default=0.15)
    parser.add_argument("--throttle-mode", choices=("bidirectional", "forward_only"), default="bidirectional")
    parser.add_argument("--negative-throttle-mode", choices=("allow", "zero", "positive_magnitude"), default="allow")
    parser.add_argument("--steering-mode", choices=("normal", "invert"), default="normal")
    args = parser.parse_args(argv)
    train_official(
        out_dir=args.out,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        timeout_s=args.timeout_s,
        training_timeout_s=args.training_timeout_s,
        observation_profile=args.observation_profile,
        steering_action_scale=args.steering_action_scale,
        straight_throttle_gain=args.straight_throttle_gain,
        straight_throttle_steering_threshold=args.straight_throttle_steering_threshold,
        throttle_mode=args.throttle_mode,
        negative_throttle_mode=args.negative_throttle_mode,
        steering_mode=args.steering_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
