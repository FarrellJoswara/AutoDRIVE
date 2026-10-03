"""PPO training entrypoint for Layer 3."""

from __future__ import annotations

import argparse
import json
import math
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


class LapCurriculumCallback(BaseCallback):
    """Start with one-lap episodes, then promote to the configured lap target."""

    def __init__(self, *, target_laps: int, successful_single_laps: int) -> None:
        super().__init__(verbose=0)
        self.target_laps = max(0, int(target_laps))
        self.successful_single_laps = max(0, int(successful_single_laps))
        self.single_lap_wins = 0
        self.promoted = self.target_laps <= 1 or self.successful_single_laps == 0
        self.promotion_step: Optional[int] = None

    def _on_step(self) -> bool:
        if self.promoted or self.target_laps <= 1:
            return True
        infos = self.locals.get("infos") or []
        self.single_lap_wins += sum(
            1 for info in infos
            if isinstance(info, dict) and bool(info.get("episode_won", False))
        )
        if self.single_lap_wins >= self.successful_single_laps:
            self.training_env.env_method("set_laps_per_episode", self.target_laps)
            self.promoted = True
            self.promotion_step = int(self.num_timesteps)
            if self.verbose:
                print(
                    f"lap curriculum: {self.single_lap_wins} clean one-lap episodes; "
                    f"promoted to {self.target_laps} laps/episode at step {self.num_timesteps}"
                )
        return True
class RunStopCallback(BaseCallback):
    """Stop on configured limits or when measured learning progress plateaus."""

    def __init__(
        self,
        *,
        max_duration_seconds: float = 0.0,
        stop_after_laps: int = 0,
        plateau_min_timesteps: int = 100_000,
        plateau_window_timesteps: int = 25_000,
        plateau_patience: int = 5,
        plateau_min_improvement_pct: float = 1.0,
        plateau_min_successful_laps: int = 10,
    ):
        super().__init__(verbose=0)
        self.max_duration_seconds = max(0.0, float(max_duration_seconds))
        self.stop_after_laps = max(0, int(stop_after_laps))
        self.plateau_min_timesteps = max(0, int(plateau_min_timesteps))
        self.plateau_window_timesteps = max(1, int(plateau_window_timesteps))
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_min_improvement_pct = max(0.0, float(plateau_min_improvement_pct))
        self.plateau_min_successful_laps = max(0, int(plateau_min_successful_laps))
        self._started_at = 0.0
        self.stop_reason: Optional[str] = None
        self._lap_totals: list[int] = []
        self._episode_laps: list[int] = []
        self._completed_laps = 0
        self._window_progress_m = 0.0
        self._window_reward = 0.0
        self._window_frontier_samples = 0
        self._window_samples = 0
        self._best_progress_rate: Optional[float] = None
        self._stale_windows = 0
        self._evaluated_windows = 0
        self.plateau_summary: Dict[str, Any] = {}

    def _on_training_start(self) -> None:
        self._started_at = time.monotonic()

    def _on_step(self) -> bool:
        if (
            self.max_duration_seconds > 0
            and time.monotonic() - self._started_at >= self.max_duration_seconds
        ):
            self.stop_reason = "max_duration"
            return False

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
                delta = max(0, current - previous)
                self._lap_totals[index] += delta
                self._completed_laps += delta
                self._episode_laps[index] = current
            if index < len(dones) and bool(dones[index]):
                self._episode_laps[index] = 0

        if self.stop_after_laps > 0:
            if any(total >= self.stop_after_laps for total in self._lap_totals):
                self.stop_reason = "lap_target"
                return False

        # Prefer actual route distance. Builtin maps fall back to mean reward.
        rewards = self.locals.get("rewards")
        for index, info in enumerate(infos):
            reward = float(rewards[index]) if rewards is not None and index < len(rewards) else 0.0
            if not math.isfinite(reward):
                reward = 0.0
            self._window_reward += reward
            self._window_samples += 1
            if isinstance(info, dict) and "frontier_advanced_m" in info:
                advanced_m = float(info.get("frontier_advanced_m", 0.0) or 0.0)
                if math.isfinite(advanced_m):
                    self._window_progress_m += max(0.0, advanced_m)
                    self._window_frontier_samples += 1

        if self._window_samples >= self.plateau_window_timesteps:
            if self._window_frontier_samples == self._window_samples:
                progress_rate = self._window_progress_m / self._window_samples
                metric = "frontier_m_per_env_step"
            else:
                progress_rate = self._window_reward / max(1, self._window_samples)
                metric = "reward_per_env_step"

            self._window_progress_m = 0.0
            self._window_reward = 0.0
            self._window_frontier_samples = 0
            self._window_samples = 0
            self._evaluated_windows += 1

            # Observe windows during warm-up to establish a baseline, but do
            # not count them toward patience until minimum training is reached.
            if self._best_progress_rate is None:
                self._best_progress_rate = progress_rate
            else:
                improvement_pct = (
                    (progress_rate - self._best_progress_rate)
                    / max(abs(self._best_progress_rate), 1e-9)
                    * 100.0
                )
                if progress_rate > self._best_progress_rate and improvement_pct >= self.plateau_min_improvement_pct:
                    self._best_progress_rate = progress_rate
                    self._stale_windows = 0
                elif self.num_timesteps >= self.plateau_min_timesteps:
                    self._stale_windows += 1

            self.plateau_summary = {
                "metric": metric,
                "latest_window_rate": progress_rate,
                "best_window_rate": self._best_progress_rate,
                "evaluated_windows": self._evaluated_windows,
                "stale_windows": self._stale_windows,
                "minimum_timesteps": self.plateau_min_timesteps,
                "window_timesteps": self.plateau_window_timesteps,
                "patience": self.plateau_patience,
                "minimum_improvement_pct": self.plateau_min_improvement_pct,
                "completed_laps": self._completed_laps,
                "minimum_successful_laps": self.plateau_min_successful_laps,
            }
            if (
                self.num_timesteps >= self.plateau_min_timesteps
                and self._stale_windows >= self.plateau_patience
                and self._completed_laps >= self.plateau_min_successful_laps
            ):
                self.stop_reason = "progress_plateau"
                return False
        return True


class SimulatorPauseCallback(BaseCallback):
    """Freeze Unity between PPO rollouts so cached actions cannot keep driving."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._paused = False

    def _on_training_start(self) -> None:
        # Fail before collecting a rollout if this player was not rebuilt with
        # the lockstep gate; otherwise stale controls could still drive physics.
        self.training_env.env_method("set_simulation_paused", True)
        self._paused = True
        self.training_env.env_method("set_simulation_paused", False)
        self._paused = False

    def _on_rollout_end(self) -> None:
        self.training_env.env_method("set_simulation_paused", True)
        self._paused = True

    def _on_rollout_start(self) -> None:
        if self._paused:
            self.training_env.env_method("set_simulation_paused", False)
            self._paused = False

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # SB3 finishes on a rollout boundary, where Unity is paused for PPO
        # optimization. Explicitly resume even when this callback's local flag
        # was lost or the final stop callback ended the rollout early.
        self.training_env.env_method("resume_simulation")
        self._paused = False


class CleanLapCheckpointCallback(BaseCallback):
    """Keep only PPO policies that repeatedly demonstrate clean laps."""

    MINIMUM_SUCCESSFUL_EPISODES = 3

    def __init__(self, path: Path) -> None:
        super().__init__(verbose=0)
        self.path = Path(path)
        self.successful_episodes = 0
        self.best_lap_time_s: Optional[float] = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        for index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            if dones is not None and index < len(dones) and not bool(dones[index]):
                continue
            if not bool(info.get("episode_won", False)):
                continue
            self.successful_episodes += 1
            lap_time = info.get("best_lap_time_s")
            try:
                lap_time = float(lap_time)
            except (TypeError, ValueError):
                lap_time = None
            if self.successful_episodes < self.MINIMUM_SUCCESSFUL_EPISODES:
                continue
            if lap_time is not None and math.isfinite(lap_time) and lap_time > 0:
                if self.best_lap_time_s is None or lap_time < self.best_lap_time_s:
                    self.best_lap_time_s = lap_time
                    self.model.save(str(self.path))
            elif self.best_lap_time_s is None and not self.path.with_suffix(".zip").exists():
                # Older simulator builds may report a clean win without lap
                # timing; retain the first successful policy as a fallback.
                self.model.save(str(self.path))
        return True


def train(
    *,
    n_envs: int = 1,
    timesteps: int = 0,
    out_dir: Path,
    seed: int = 0,
    device: str = "auto",
    resume: Optional[Path] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    max_duration_seconds: float = 0.0,
    stop_after_laps: int = 0,
    plateau_min_timesteps: int = 100_000,
    plateau_window_timesteps: int = 25_000,
    plateau_patience: int = 5,
    plateau_min_improvement_pct: float = 1.0,
    plateau_min_successful_laps: int = 10,
    expert_pretrain_steps: int = 0,
    curriculum_single_lap_successes: int = 10,
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
        f"train: n_envs={n_envs} timesteps={'unbounded' if timesteps == 0 else timesteps} "
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
        trusted_initial_path = out_dir / "trusted_initial_model"
        if resume is not None:
            print(f"resuming from {resume}")
            model = PPO.load(str(resume), env=vec_env, device=resolved_device)
            model.save(str(trusted_initial_path))
        else:
            model = make_model(
                vec_env,
                device=resolved_device,
                seed=seed,
                tensorboard_log=str(tb_dir),
            )

        expert_summary = None
        cloning_summary = None
        final_lap_target = max(0, int(env_kwargs.get("laps_per_episode", 10)))
        curriculum = LapCurriculumCallback(
            target_laps=final_lap_target,
            successful_single_laps=curriculum_single_lap_successes,
        )
        route_training_supported = False
        centerline = None
        route = None
        if resume is None and final_lap_target > 0:
            from src.layer2.route_progress import RouteProgressTracker, map_centerline_path
            centerline = map_centerline_path(str(env_kwargs.get("map_id", "none")))
            if centerline is not None:
                route = RouteProgressTracker.from_csv(centerline)
                route_training_supported = bool(route.closed)
            if route_training_supported:
                # Start with one circuit even when expert data collection is
                # disabled; the curriculum promotes only after clean success.
                initial_laps = (
                    1
                    if final_lap_target > 1 and curriculum_single_lap_successes > 0
                    else final_lap_target
                )
                vec_env.env_method("set_laps_per_episode", initial_laps)
                curriculum.promoted = initial_laps == final_lap_target
                if expert_pretrain_steps > 0:
                    from src.layer3.behavior_cloning import (
                        behavior_clone_actor,
                        collect_centerline_demonstrations,
                    )
                    from src.layer3.expert import CenterlineExpert

                    expert = CenterlineExpert(route)
                    # Privileged map geometry generates action labels only.
                    # The policy observation remains LiDAR + vehicle state.
                    dataset, expert_summary = collect_centerline_demonstrations(
                        vec_env,
                        expert,
                        steps=expert_pretrain_steps,
                        output_path=out_dir / "expert_demonstrations.npz",
                    )
                    print(
                        "expert collection summary: "
                        f"{expert_summary['completed_episodes']} episodes, "
                        f"{expert_summary['successful_episodes']} clean laps, "
                        f"endings={expert_summary['termination_counts']}, "
                        f"examples={expert_summary['termination_examples']}",
                        flush=True,
                    )
                    if expert_summary["successful_episodes"] < 1:
                        raise RuntimeError(
                            "centerline teacher produced no clean one-lap demonstrations; "
                            f"endings={expert_summary['termination_counts']}; "
                            "refusing to initialize the learned policy from unvalidated labels"
                        )
                    cloning_summary = behavior_clone_actor(model, dataset, seed=seed)
                    model.save(str(out_dir / "expert_initialized_model"))
                    print(
                        "expert warmup: "
                        f"{expert_summary['transitions']} transitions, "
                        f"{expert_summary['successful_episodes']} clean one-lap episodes, "
                        f"endings={expert_summary['termination_counts']}, "
                        f"validation loss {cloning_summary['initial_validation_loss']:.4f} -> "
                        f"{cloning_summary['final_validation_loss']:.4f}"
                    )
            else:
                curriculum.promoted = True
                print("map has no closed centerline; one-lap curriculum and expert warmup skipped")
        elif resume is not None:
            curriculum.promoted = True
            vec_env.env_method("set_laps_per_episode", final_lap_target)

        config = {
            "n_envs": n_envs,
            "bridge_ports": list(range(4567, 4567 + n_envs)),
            "timesteps": timesteps,
            "stopping": {
                "max_duration_seconds": max(0.0, float(max_duration_seconds)),
                "stop_after_laps": max(0, int(stop_after_laps)),
                "lap_target_semantics": "any_car_total_across_run",
                "plateau_min_timesteps": max(0, int(plateau_min_timesteps)),
                "plateau_window_timesteps": max(1, int(plateau_window_timesteps)),
                "plateau_patience": max(1, int(plateau_patience)),
                "plateau_min_improvement_pct": max(0.0, float(plateau_min_improvement_pct)),
                "plateau_min_successful_laps": (
                    max(0, int(plateau_min_successful_laps))
                    if route_training_supported else 0
                ),
                "plateau_metric": "frontier_m_per_env_step; reward_per_env_step_without_route",
            },
            "learning_strategy": {
                "expert_pretrain_steps": max(0, int(expert_pretrain_steps)),
                "expert_demonstrations": expert_summary,
                "behavior_cloning": cloning_summary,
                "curriculum_single_lap_successes": max(0, int(curriculum_single_lap_successes)),
                "curriculum_final_laps_per_episode": final_lap_target,
                "teacher_map_geometry_in_policy_observation": False,
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

        # Keep regular recovery points even when the maximum step count is off.
        save_freq = (
            max(1, min(10_000, timesteps // max(n_envs, 1)))
            if timesteps > 0 else max(1, 10_000 // max(n_envs, 1))
        )
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
            plateau_min_timesteps=plateau_min_timesteps,
            plateau_window_timesteps=plateau_window_timesteps,
            plateau_patience=plateau_patience,
            plateau_min_improvement_pct=plateau_min_improvement_pct,
            plateau_min_successful_laps=(
                plateau_min_successful_laps if route_training_supported else 0
            ),
        )
        simulator_pause_cb = SimulatorPauseCallback()
        clean_lap_cb = CleanLapCheckpointCallback(out_dir / "best_clean_lap_model")
        curriculum.verbose = 1
        callbacks = [simulator_pause_cb, checkpoint_cb, curriculum, clean_lap_cb, stop_cb]
        from src.layer3.hub_callback import maybe_hub_callback

        hub_cb = maybe_hub_callback(run_id=out_dir.name)
        if hub_cb is not None:
            callbacks.append(hub_cb)
            print(f"hub telemetry: HUB_URL set → publishing to hub (run_id={out_dir.name})")

        model.learn(
            # SB3 requires a finite target; plateau stopping ends unbounded runs.
            total_timesteps=int(timesteps) if timesteps > 0 else sys.maxsize,
            callback=callbacks,
            progress_bar=False,
        )
        ppo_timesteps = int(model.num_timesteps)
        last_model_path = out_dir / "ppo_last_model"
        model.save(str(last_model_path))
        final_policy_source = "ppo_last_model"
        trusted_model = out_dir / "expert_initialized_model.zip"
        if not trusted_model.is_file():
            trusted_model = trusted_initial_path.with_suffix(".zip")
        if clean_lap_cb.successful_episodes >= clean_lap_cb.MINIMUM_SUCCESSFUL_EPISODES:
            selected_path = out_dir / "best_clean_lap_model.zip"
            if selected_path.is_file():
                model = PPO.load(str(selected_path), env=vec_env, device=resolved_device)
                final_policy_source = "best_clean_lap_model"
        elif (
            clean_lap_cb.successful_episodes < clean_lap_cb.MINIMUM_SUCCESSFUL_EPISODES
            and trusted_model.is_file()
        ):
            # PPO can lose the demonstrator before it has learned to finish a
            # lap. Keep the initial sensor-only policy (or the user's resumed
            # checkpoint) as the usable model rather than publishing an
            # unvalidated, non-driving checkpoint.
            model = PPO.load(
                str(trusted_model),
                env=vec_env,
                device=resolved_device,
            )
            final_policy_source = (
                "expert_initialized_model_no_clean_ppo_lap"
                if trusted_model.name == "expert_initialized_model.zip"
                else "trusted_initial_model_no_clean_ppo_lap"
            )
        stop_record = {
            "reason": stop_cb.stop_reason or "timestep_limit",
            "num_timesteps": ppo_timesteps,
            "max_duration_seconds": max(0.0, float(max_duration_seconds)),
            "stop_after_laps": max(0, int(stop_after_laps)),
            "plateau": stop_cb.plateau_summary,
            "completed_laps": stop_cb._completed_laps,
            "successful_ppo_episodes": clean_lap_cb.successful_episodes,
            "minimum_successful_ppo_episodes_for_promotion": clean_lap_cb.MINIMUM_SUCCESSFUL_EPISODES,
            "best_ppo_lap_time_s": clean_lap_cb.best_lap_time_s,
            "final_policy_source": final_policy_source,
            "curriculum": {
                "single_lap_wins": curriculum.single_lap_wins,
                "promoted": curriculum.promoted,
                "promotion_step": curriculum.promotion_step,
                "final_laps_per_episode": curriculum.target_laps,
            },
        }
        (out_dir / "stop_reason.json").write_text(
            json.dumps(stop_record, indent=2), encoding="utf-8"
        )
        print(f"training stopped: {stop_record['reason']} at {ppo_timesteps} timesteps")
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
        default=0,
        help="Maximum total env steps; 0 disables the cap and stops on progress plateau",
    )
    p.add_argument(
        "--max-duration-seconds", type=float, default=0.0,
        help="Optional wall-clock run limit; 0 disables it",
    )
    p.add_argument(
        "--stop-after-laps", type=int, default=0,
        help="Stop when any car completes this many laps; 0 disables it",
    )
    p.add_argument("--plateau-min-timesteps", type=int, default=100_000)
    p.add_argument("--plateau-window-timesteps", type=int, default=25_000)
    p.add_argument("--plateau-patience", type=int, default=5)
    p.add_argument("--plateau-min-improvement-pct", type=float, default=1.0)
    p.add_argument("--plateau-min-successful-laps", type=int, default=10)
    p.add_argument("--expert-pretrain-steps", type=int, default=0)
    p.add_argument("--curriculum-single-lap-successes", type=int, default=10)
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
    p.add_argument("--max-episode-steps", type=int, default=0)
    p.add_argument("--stagnation-speed-threshold", type=float, default=0.15)
    # ~5 s idle @ frame_skip=4 before truncate.
    p.add_argument("--stagnation-steps", type=int, default=50)
    p.add_argument(
        "--terminate-on-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--map-id", type=str, default="none")
    p.add_argument("--laps-per-episode", type=int, default=10)
    p.add_argument("--frontier-stagnation-seconds", type=float, default=5.0)
    p.add_argument("--forward-scale", type=float, default=0.0)
    p.add_argument("--backward-speed-penalty-scale", type=float, default=1.0)
    p.add_argument("--route-progress-scale", type=float, default=10.0)
    p.add_argument("--time-penalty-per-second", type=float, default=1.0)
    p.add_argument("--collision-penalty", type=float, default=-100.0)
    p.add_argument("--episode-failure-penalty", type=float, default=-100.0)
    p.add_argument("--slip-penalty", type=float, default=0.2)
    p.add_argument("--steer-jerk-penalty", type=float, default=0.05)
    p.add_argument("--lap-time-reward-scale", type=float, default=1000.0)
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
        plateau_min_timesteps=args.plateau_min_timesteps,
        plateau_window_timesteps=args.plateau_window_timesteps,
        plateau_patience=args.plateau_patience,
        plateau_min_improvement_pct=args.plateau_min_improvement_pct,
        plateau_min_successful_laps=args.plateau_min_successful_laps,
        expert_pretrain_steps=args.expert_pretrain_steps,
        curriculum_single_lap_successes=args.curriculum_single_lap_successes,
    )
    print(f"done: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
