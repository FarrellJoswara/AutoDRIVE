"""Train PPO against the official IROS RoboRacer simulator and Devkit."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_OFFICIAL_FIXED_PPO = {
    "batch_size": 64,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


class EpisodeReturnPlateau:
    """Stop only after repeated windows fail to improve mean episode return."""

    def __init__(
        self,
        *,
        window_episodes: int,
        patience: int,
        min_timesteps: int,
        min_improvement_pct: float,
    ) -> None:
        if window_episodes < 2 or patience < 1 or min_timesteps < 0 or min_improvement_pct < 0:
            raise ValueError("invalid episode-return plateau configuration")
        self.window_episodes = int(window_episodes)
        self.patience = int(patience)
        self.min_timesteps = int(min_timesteps)
        self.min_improvement_pct = float(min_improvement_pct)
        self.window_returns = deque()
        self.best_window_mean: Optional[float] = None
        self.stale_windows = 0

    def observe(self, episode_return: float, trained_steps: int) -> bool:
        self.window_returns.append(float(episode_return))
        if len(self.window_returns) < self.window_episodes:
            return False
        current_mean = sum(self.window_returns) / len(self.window_returns)
        self.window_returns.clear()
        if trained_steps < self.min_timesteps:
            return False
        if self.best_window_mean is None:
            self.best_window_mean = current_mean
            return False
        relative_gain_pct = (
            (current_mean - self.best_window_mean)
            / max(abs(self.best_window_mean), 1.0)
            * 100.0
        )
        if relative_gain_pct >= self.min_improvement_pct:
            self.best_window_mean = current_mean
            self.stale_windows = 0
        else:
            self.stale_windows += 1
        return self.stale_windows >= self.patience


def _official_ppo_config(model, *, observation_profile: str) -> dict:
    """Read effective PPO values from constructed model state, not CLI inputs."""
    architecture = {
        "official_sensors_history": "temporal_lidar_cnn",
        "official_sensors_camera": "lidar_camera_cnn",
    }.get(observation_profile, "lidar_cnn")
    optimizer = getattr(model.policy, "optimizer", None)
    if optimizer is None or not optimizer.param_groups:
        raise RuntimeError("PPO policy optimizer is missing or has no parameter groups")
    optimizer_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return {
        "learning_rate": float(model.learning_rate),
        "lr_schedule_at_1": float(model.lr_schedule(1.0)),
        "lr_schedule_at_0": float(model.lr_schedule(0.0)),
        "optimizer_learning_rates": optimizer_lrs,
        "n_steps": int(model.n_steps),
        "rollout_buffer_size": int(model.rollout_buffer.buffer_size),
        "n_envs": int(model.n_envs),
        "rollout_buffer_envs": int(model.rollout_buffer.n_envs),
        "rollout_transitions": int(model.rollout_buffer.buffer_size * model.rollout_buffer.n_envs),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "batch_size": int(model.batch_size),
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_at_1": float(model.clip_range(1.0)),
        "clip_range_at_0": float(model.clip_range(0.0)),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "policy_architecture": architecture,
        "policy_extractor": type(model.policy.features_extractor).__name__,
    }


def _verify_official_ppo_config(model, requested: dict, *, observation_profile: str) -> dict:
    """Fail before collection unless every effective PPO setting is exact."""
    effective = _official_ppo_config(model, observation_profile=observation_profile)
    expected = {
        **requested,
        **_OFFICIAL_FIXED_PPO,
        "rollout_buffer_size": int(requested["n_steps"]),
        "n_envs": int(model.get_env().num_envs),
        "rollout_buffer_envs": int(model.get_env().num_envs),
        "rollout_transitions": int(requested["n_steps"] * model.get_env().num_envs),
        "lr_schedule_at_1": float(requested["learning_rate"]),
        "lr_schedule_at_0": float(requested["learning_rate"]),
        "clip_range_at_1": 0.2,
        "clip_range_at_0": 0.2,
        "policy_architecture": {
            "official_sensors_history": "temporal_lidar_cnn",
            "official_sensors_camera": "lidar_camera_cnn",
        }.get(observation_profile, "lidar_cnn"),
    }
    errors = []
    for key, expected_value in expected.items():
        actual = effective.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-12
            ):
                errors.append(f"{key}: requested {expected_value!r}, effective {actual!r}")
        elif actual != expected_value:
            errors.append(f"{key}: requested {expected_value!r}, effective {actual!r}")
    for index, actual_lr in enumerate(effective["optimizer_learning_rates"]):
        if not math.isclose(
            actual_lr, float(requested["learning_rate"]), rel_tol=1e-9, abs_tol=1e-12
        ):
            errors.append(
                f"optimizer_learning_rates[{index}]: requested "
                f"{requested['learning_rate']!r}, effective {actual_lr!r}"
            )
    expected_extractor = {
        "official_sensors_history": "TemporalLidarStateExtractor",
        "official_sensors_camera": "LidarCameraStateExtractor",
    }.get(observation_profile, "LidarStateExtractor")
    if effective["policy_extractor"] != expected_extractor:
        errors.append(
            f"policy_extractor: expected {expected_extractor!r}, "
            f"effective {effective['policy_extractor']!r}"
        )
    if errors:
        raise RuntimeError("Official PPO configuration verification failed before learn(): " + "; ".join(errors))
    return effective


def train_official(
    *,
    out_dir: Path,
    total_timesteps: int,
    n_envs: int = 1,
    stop_on_plateau: bool = True,
    plateau_min_timesteps: int = 100_000,
    plateau_window_episodes: int = 5,
    plateau_patience: int = 5,
    plateau_min_improvement_pct: float = 1.0,
    max_duration_hours: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    resume: Optional[Path] = None,
    checkpoint_every: int = 10_000,
    n_steps: int = 2048,
    learning_rate: float = 1e-4,
    n_epochs: int = 4,
    gamma: float = 0.9995,
    gae_lambda: float = 0.95,
    timeout_s: float = 180.0,
    training_timeout_s: float = 600.0,
    race_laps: int = 10,
    warmup_laps: int = 0,
    time_cost_per_simulated_second: float = 5.0,
    collision_penalty_magnitude: float = 100.0,
    collision_reward_percent: float = 100.0,
    failed_episode_penalty: float = 100.0,
    frontier_stagnation_s: float = 10.0,
    observation_profile: str = "official_sensors",
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    throttle_mode: str = "bidirectional",
    negative_throttle_mode: str = "allow",
    steering_mode: str = "normal",
) -> Path:
    """Run one PPO learner over isolated official ROS simulator domains."""
    if not 1 <= n_envs <= 8:
        raise ValueError("n_envs must be between 1 and 8")
    if total_timesteps < 0:
        raise ValueError("total_timesteps must be zero (unlimited) or positive")
    if checkpoint_every < 1 or n_steps < 1 or n_epochs < 1:
        raise ValueError("checkpoint_every, n_steps, and n_epochs must be positive")
    if plateau_window_episodes < 2 or plateau_patience < 1:
        raise ValueError("plateau_window_episodes must be >= 2 and plateau_patience >= 1")
    if min(plateau_min_timesteps, max_duration_hours) < 0:
        raise ValueError("plateau_min_timesteps and max_duration_hours must be non-negative")

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback

    from src.layer3.official_envs import make_official_vec_env, close_official_vec_env
    from src.layer3.hub_callback import maybe_hub_callback
    from src.layer3.official_policy import resume_ppo_checkpoint
    from src.layer3.train import make_model

    stop_requested = threading.Event()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

    class _TrainingStopCallback(BaseCallback):
        """Apply the user's independent wall-clock, plateau, and operator stops."""

        def __init__(self, starting_steps: int, started_at: float) -> None:
            super().__init__(verbose=0)
            self.starting_steps = int(starting_steps)
            self.started_at = started_at
            self.plateau = EpisodeReturnPlateau(
                window_episodes=plateau_window_episodes,
                patience=plateau_patience,
                min_timesteps=plateau_min_timesteps,
                min_improvement_pct=plateau_min_improvement_pct,
            )
            self.stop_reason: Optional[str] = None

        def _on_step(self) -> bool:
            if stop_requested.is_set():
                self.stop_reason = "operator_stop"
                return False
            if max_duration_hours > 0 and time.monotonic() - self.started_at >= max_duration_hours * 3600:
                self.stop_reason = "max_duration"
                return False
            if not stop_on_plateau:
                return True
            infos = self.locals.get("infos") or []
            for info in infos:
                episode = info.get("episode") if isinstance(info, dict) else None
                if not isinstance(episode, dict) or "r" not in episode:
                    continue
                trained_steps = int(self.model.num_timesteps) - self.starting_steps
                if self.plateau.observe(float(episode["r"]), trained_steps):
                    self.stop_reason = "progress_plateau"
                    print(
                        "Training plateau reached: "
                        f"{self.plateau.stale_windows} episode windows without >= "
                        f"{plateau_min_improvement_pct:g}% mean-return improvement.",
                        flush=True,
                    )
                    return False
            return True

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "tensorboard").mkdir(exist_ok=True)

    vec_env = None
    model = None
    try:
        vec_env = make_official_vec_env(
            n_envs, seed=seed,
            timeout_s=timeout_s,
            warmup_laps=warmup_laps,
            race_laps=race_laps,
            steering_action_scale=steering_action_scale,
            straight_throttle_gain=straight_throttle_gain,
            straight_throttle_steering_threshold=straight_throttle_steering_threshold,
            throttle_mode=throttle_mode,
            negative_throttle_mode=negative_throttle_mode,
            steering_mode=steering_mode,
            observation_profile=observation_profile,
            training_mode=True,
            training_timeout_s=training_timeout_s,
            training_failure_penalty=failed_episode_penalty,
            training_time_cost_per_simulated_second=time_cost_per_simulated_second,
            training_collision_penalty_magnitude=collision_penalty_magnitude,
            training_collision_reward_percent=collision_reward_percent,
            training_frontier_stagnation_s=frontier_stagnation_s,
            frontier_path=Path(__file__).resolve().parents[2]
            / "competition" / "iros2026" / "official_centerline.csv",
        )
        requested_ppo = {
            "n_steps": int(n_steps),
            "learning_rate": float(learning_rate),
            "n_epochs": int(n_epochs),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
        }
        if resume is None:
            model = make_model(
                vec_env,
                device=device,
                seed=seed,
                tensorboard_log=str(out_dir / "tensorboard"),
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                policy_architecture={
                    "official_sensors_history": "temporal_lidar_cnn",
                    "official_sensors_camera": "lidar_camera_cnn",
                }.get(observation_profile, "lidar_cnn"),
            )
        else:
            model = resume_ppo_checkpoint(
                resume,
                env=vec_env,
                observation_profile=observation_profile,
                learning_rate=learning_rate,
                n_steps=n_steps,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                seed=seed,
                device=device,
                tensorboard_log=str(out_dir / "tensorboard"),
            )
            model.set_logger(
                configure(
                    folder=str(out_dir / "tensorboard"),
                    format_strings=["stdout", "csv", "tensorboard"],
                )
            )

        effective_ppo = _official_ppo_config(
            model, observation_profile=observation_profile
        )
        print(
            "Official PPO requested/effective configuration before verification: "
            + json.dumps({"requested": requested_ppo, "effective": effective_ppo}, sort_keys=True),
            flush=True,
        )
        _verify_official_ppo_config(
            model, requested_ppo, observation_profile=observation_profile
        )
        print("Official PPO configuration verified before rollout.", flush=True)

        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "official IROS 2026 API and simulator images",
            "official_api_image": "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete",
            "official_simulator_image": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice",
            "episode": {
                "frontier_source": "restricted official IPS position; Porto practice-track mesh-aligned centerline",
                "frontier_stagnation_s": frontier_stagnation_s,
                "collision_behavior": "end this car life on each new collision",
                "training_watchdog_s": training_timeout_s,
                "laps": "diagnostics only; do not terminate or reward lap crossings",
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
                "time_cost_per_simulated_second": time_cost_per_simulated_second,
                "elapsed_time_source": "ROS LaserScan header timestamps when advancing; monotonic receipt-clock fallback",
                "positive_reward": "new high-water route frontier distance only",
                "collision_cost": f"{collision_penalty_magnitude:g} + {collision_reward_percent:g}% of positive frontier return",
                "non_collision_failure_cost": failed_episode_penalty,
                "frontier_stall": "end and reset life; time cost only",
                "lap_completion_bonus": 0,
                "restricted_topics_used_only_for_reward_and_episode_control": True,
                "restricted_topics_in_policy_observation": False,
            },
            "ppo": {
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "rollout_transitions": n_steps * n_envs,
                "stop_on_plateau": stop_on_plateau,
                "plateau_min_timesteps": plateau_min_timesteps,
                "plateau_window_episodes": plateau_window_episodes,
                "plateau_patience": plateau_patience,
                "plateau_min_improvement_pct": plateau_min_improvement_pct,
                "max_duration_hours": max_duration_hours,
                # Retain the original flat fields for existing log readers.
                **requested_ppo,
                "requested": requested_ppo,
                "effective": effective_ppo,
                "device": device,
                "seed": seed,
            },
            "resume": str(Path(resume).resolve()) if resume else None,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, math.ceil(checkpoint_every / n_envs)),
            save_path=str(out_dir / "checkpoints"),
            name_prefix="official_ppo",
        )
        hub_callback = maybe_hub_callback(
            os.environ.get("AICAR_RUN_ID", out_dir.name),
            runtime="official",
            run_dir=out_dir,
        )
        callbacks = [checkpoint_callback]
        if hub_callback is not None:
            callbacks.append(hub_callback)
            print(
                f"Watch telemetry enabled: HUB_URL is set (run_id={out_dir.name})",
                flush=True,
            )
        starting_steps = int(model.num_timesteps)
        stop_callback = _TrainingStopCallback(starting_steps, time.monotonic())
        callbacks.append(stop_callback)
        model.learn(
            total_timesteps=total_timesteps if total_timesteps > 0 else sys.maxsize,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=resume is None,
        )
        target = out_dir / "official_policy_latest.zip"
        model.save(str(target.with_suffix("")))
        print(f"Saved official-simulator PPO checkpoint: {target}", flush=True)
        stop_reason = stop_callback.stop_reason or (
            "operator_stop" if stop_requested.is_set() else "timestep_limit"
        )
        (out_dir / "stop_reason.json").write_text(
            json.dumps({"reason": stop_reason, "num_timesteps": int(model.num_timesteps)}, indent=2),
            encoding="utf-8",
        )
        return target
    except BaseException:
        if model is not None:
            try:
                model.save(str(out_dir / "official_policy_recovery"))
                print("Saved recovery checkpoint after interrupted/failed training.", flush=True)
            except Exception as save_error:
                print(f"Recovery checkpoint failed: {save_error}", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            close_official_vec_env(vec_env)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/runs/official"))
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--stop-on-plateau", type=int, choices=(0, 1), default=1)
    parser.add_argument("--plateau-min-timesteps", type=int, default=100_000)
    parser.add_argument("--plateau-window-episodes", type=int, default=5)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-min-improvement-pct", type=float, default=1.0)
    parser.add_argument("--max-duration-hours", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.9995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--training-timeout-s", type=float, default=600.0)
    parser.add_argument("--race-laps", type=int, default=10)
    parser.add_argument("--warmup-laps", type=int, default=0)
    parser.add_argument("--time-cost-per-simulated-second", type=float, default=5.0)
    parser.add_argument("--collision-penalty-magnitude", type=float, default=100.0)
    parser.add_argument("--collision-reward-percent", type=float, default=20.0)
    parser.add_argument("--failed-episode-penalty", type=float, default=100.0)
    parser.add_argument("--frontier-stagnation-s", type=float, default=10.0)
    parser.add_argument(
        "--observation-profile",
        choices=("official_sensors", "official_sensors_history", "official_sensors_camera"),
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
        n_envs=args.n_envs,
        stop_on_plateau=bool(args.stop_on_plateau),
        plateau_min_timesteps=args.plateau_min_timesteps,
        plateau_window_episodes=args.plateau_window_episodes,
        plateau_patience=args.plateau_patience,
        plateau_min_improvement_pct=args.plateau_min_improvement_pct,
        max_duration_hours=args.max_duration_hours,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        timeout_s=args.timeout_s,
        training_timeout_s=args.training_timeout_s,
        race_laps=args.race_laps,
        warmup_laps=args.warmup_laps,
        time_cost_per_simulated_second=args.time_cost_per_simulated_second,
        collision_penalty_magnitude=args.collision_penalty_magnitude,
        collision_reward_percent=args.collision_reward_percent,
        failed_episode_penalty=args.failed_episode_penalty,
        frontier_stagnation_s=args.frontier_stagnation_s,
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
