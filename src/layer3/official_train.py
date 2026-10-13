"""Train PPO against the official IROS RoboRacer simulator and Devkit."""

from __future__ import annotations

import argparse
import json
import importlib.metadata
import math
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.layer4.official_settings import validate_bc_teacher_action_mapping
from src.layer3.bc_artifact import (
    OFFICIAL_BC_SOURCE,
    checkpoint_path_aliases,
    sha256_file as _sha256_file,
    verify_bc_checkpoint,
    verify_ppo_lineage_checkpoint,
    verify_ppo_warmstart,
    write_ppo_lineage_sidecar,
)

_OFFICIAL_FIXED_PPO = {
    "batch_size": 64,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
_BC_SMOKE_PPO = {
    "n_envs": 1,
    "n_steps": 2048,
    "learning_rate": 1e-5,
    "n_epochs": 4,
    "gamma": 0.9995,
    "gae_lambda": 0.95,
    "target_kl": 0.01,
    "batch_size": 64,
    "clip_range": 0.1,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "max_transitions": 8192,
    "max_wall_seconds": 600,
    "save_every_transitions": 2048,
}
_BC_SMOKE_FIXED_PPO = {
    "batch_size": _BC_SMOKE_PPO["batch_size"],
    "clip_range": _BC_SMOKE_PPO["clip_range"],
    "ent_coef": _BC_SMOKE_PPO["ent_coef"],
    "vf_coef": _BC_SMOKE_PPO["vf_coef"],
    "max_grad_norm": _BC_SMOKE_PPO["max_grad_norm"],
}
_BC_SMOKE_REWARDS = {
    "time_cost_per_simulated_second": 0.0,
    "collision_penalty_magnitude": 200.0,
    "collision_reward_percent": 0.0,
    "lap_completion_reward": 0.0,
    "failed_episode_penalty": 200.0,
    "backward_speed_penalty_scale": 10.0,
    "episode_failure_reward_percent": 0.0,
    "frontier_stagnation_s": 5.0,
}

_DEFAULT_OFFICIAL_SIMULATOR_IMAGE = (
    "autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice"
)

_BC_RESUME_STD_MIN = (0.01, 0.05)
_BC_RESUME_STD_MAX = (0.8, 0.8)


def _verified_ready_bc_resume_sha256(model_path: Optional[Path]) -> Optional[str]:
    """Return a BC parent hash only after strict BC or derived-candidate proof."""
    if model_path is None:
        return None
    verified_bc = verify_bc_checkpoint(model_path)
    if verified_bc:
        return str(verified_bc["checkpoint_sha256"])
    verified_source_legacy = verify_bc_checkpoint(
        model_path,
        expected_sha256=OFFICIAL_BC_SOURCE["checkpoint_sha256"],
        allow_legacy=True,
    )
    if verified_source_legacy:
        return str(verified_source_legacy["checkpoint_sha256"])
    verified_candidate = verify_ppo_warmstart(model_path)
    if verified_candidate:
        return str(verified_candidate["parent_checkpoint_sha256"])
    verified_descendant = verify_ppo_lineage_checkpoint(model_path)
    if verified_descendant:
        return str(verified_descendant["bc_ancestor_checkpoint_sha256"])
    return None


def _training_resume_lineage(model_path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Describe the verified immediate PPO parent and its BC ancestor."""
    if model_path is None:
        return None
    path = Path(model_path).resolve()
    if not path.is_file():
        return None
    parent_hash = _sha256_file(path)
    rebuilt = verify_bc_checkpoint(path, expected_sha256=parent_hash)
    if rebuilt:
        return {
            "parent_checkpoint": {"path": checkpoint_path_aliases(path), "sha256": parent_hash},
            "bc_ancestor_checkpoint": {"path": checkpoint_path_aliases(path), "sha256": parent_hash},
            "verification": rebuilt["verification"],
        }
    legacy = verify_bc_checkpoint(
        path,
        expected_sha256=OFFICIAL_BC_SOURCE["checkpoint_sha256"],
        allow_legacy=True,
    )
    if legacy:
        return {
            "parent_checkpoint": {"path": checkpoint_path_aliases(path), "sha256": parent_hash},
            "bc_ancestor_checkpoint": {"path": checkpoint_path_aliases(path), "sha256": parent_hash},
            "verification": legacy["verification"],
        }
    warmstart = verify_ppo_warmstart(path)
    if warmstart:
        parent = Path(warmstart["parent_checkpoint"]).resolve()
        return {
            "parent_checkpoint": {"path": checkpoint_path_aliases(path), "sha256": parent_hash},
            "bc_ancestor_checkpoint": {
                "path": checkpoint_path_aliases(parent),
                "sha256": warmstart["parent_checkpoint_sha256"],
            },
            "verification": warmstart["verification"],
        }
    descendant = verify_ppo_lineage_checkpoint(path)
    if descendant:
        ancestor = Path(descendant["bc_ancestor_checkpoint"]).resolve()
        return {
            "parent_checkpoint": {"path": checkpoint_path_aliases(path), "sha256": parent_hash},
            "bc_ancestor_checkpoint": {
                "path": checkpoint_path_aliases(ancestor),
                "sha256": descendant["bc_ancestor_checkpoint_sha256"],
            },
            "verification": descendant["verification"],
        }
    return None


def _training_context_metadata() -> dict[str, Any]:
    package_names = ("stable-baselines3", "torch", "numpy", "gymnasium", "cloudpickle")
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("official_policy.py").resolve(),
        Path(__file__).with_name("train.py").resolve(),
        Path(__file__).with_name("ppo_audit.py").resolve(),
        Path(__file__).with_name("bc_artifact.py").resolve(),
    )
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": packages,
        "code_sha256": {path.name: _sha256_file(path) for path in source_files if path.is_file()},
    }


def _make_training_exploration_schedule(
    model: Any,
    run_dir: Path,
    *,
    initial_std: Optional[tuple[float, float]],
    std_min: tuple[float, float],
    std_max: tuple[float, float],
) -> Any:
    """Install the resolved per-action bounds before exploration adapts."""
    from src.layer3.official_exploration import OfficialExplorationSchedule

    if initial_std is not None:
        import torch

        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.as_tensor(
                initial_std,
                device=model.policy.log_std.device,
                dtype=model.policy.log_std.dtype,
            )))
    return OfficialExplorationSchedule(model, run_dir, std_min=std_min, std_max=std_max)


def _exploration_profile_values(
    profile: str, *, verified_bc_resume: bool
) -> tuple[Optional[tuple[float, float]], tuple[float, float], tuple[float, float]]:
    if profile == "bc_smoke":
        return (0.01, 0.05), (0.01, 0.05), (0.01, 0.05)
    if profile == "bc_campaign":
        return (0.01, 0.05), (0.005, 0.02), (0.01, 0.05)
    minimum = _BC_RESUME_STD_MIN if verified_bc_resume else (0.2, 0.2)
    maximum = _BC_RESUME_STD_MAX if verified_bc_resume else (0.8, 0.8)
    return None, minimum, maximum


def _read_resume_exploration_config(model_path: Optional[Path]) -> Optional[dict[str, Any]]:
    if model_path is None:
        return None
    path = Path(model_path).resolve()
    for directory in (path.parent, path.parent.parent):
        config_path = directory / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(config, dict) and isinstance(config.get("ppo_exploration"), dict):
            return config["ppo_exploration"]
    return None


def _resolve_exploration_settings(
    *,
    requested_profile: Optional[str],
    overrides: dict[str, Optional[float]],
    saved: Optional[dict[str, Any]],
    verified_bc_resume: bool,
    current_std: tuple[float, float],
) -> dict[str, Any]:
    def action_pair(name: str, value: Any) -> tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"saved {name} must contain exactly throttle and steering values")
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"saved {name} must contain numeric values") from exc

    profile = requested_profile or (str(saved.get("profile")) if saved else "standard")
    if profile not in {"standard", "bc_smoke", "bc_campaign"}:
        raise ValueError("saved exploration profile is invalid")
    if profile != "standard" and not verified_bc_resume:
        raise ValueError(f"{profile} requires an authenticated BC-derived resume checkpoint")
    profile_initial, profile_min, profile_max = _exploration_profile_values(
        profile, verified_bc_resume=verified_bc_resume
    )
    if requested_profile is None and saved is not None:
        profile_min = action_pair(
            "minimum_action_std_by_action",
            saved.get("minimum_action_std_by_action", profile_min),
        )
        profile_max = action_pair(
            "maximum_action_std_by_action",
            saved.get("maximum_action_std_by_action", profile_max),
        )
    initial = profile_initial if requested_profile is not None else None
    if requested_profile is None and saved is None:
        initial = None
    if requested_profile is None and saved is not None:
        # Keep the resumed policy's actual log_std. The saved values below
        # describe the schedule; the checkpoint itself owns the current noise.
        initial = None
    if any(value is not None for value in overrides.values()):
        initial_values = list(initial or current_std)
        minimum_values = list(profile_min)
        maximum_values = list(profile_max)
        for key, index in (
            ("initial_throttle", 0), ("initial_steering", 1),
            ("min_throttle", 0), ("min_steering", 1),
            ("max_throttle", 0), ("max_steering", 1),
        ):
            value = overrides.get(key)
            if value is None:
                continue
            if key.startswith("initial"):
                initial_values[index] = float(value)
            elif key.startswith("min"):
                minimum_values[index] = float(value)
            else:
                maximum_values[index] = float(value)
        initial = tuple(initial_values)
        profile_min = tuple(minimum_values)
        profile_max = tuple(maximum_values)
    values = [*current_std, *profile_min, *profile_max]
    if initial is not None:
        values.extend(initial)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("per-action exploration standard deviations must be finite and positive")
    if any(lower > upper for lower, upper in zip(profile_min, profile_max)):
        raise ValueError("per-action exploration minimum must not exceed maximum")
    if initial is not None and any(
        value < lower or value > upper
        for value, lower, upper in zip(initial, profile_min, profile_max)
    ):
        raise ValueError("initial per-action exploration standard deviations must fit within their bounds")
    return {
        "profile": profile,
        "initial_std": initial,
        "std_min": tuple(profile_min),
        "std_max": tuple(profile_max),
        "inherited_from_resume": requested_profile is None and saved is not None,
    }


def _official_simulator_image() -> str:
    """Return the simulator provenance recorded with an official checkpoint."""
    value = os.environ.get("AICAR_OFFICIAL_SIMULATOR_IMAGE", "").strip()
    return value or _DEFAULT_OFFICIAL_SIMULATOR_IMAGE


def _demonstration_progress_publisher(route: Any, publish_phase: Any) -> Any:
    """Adapt collector progress into the public BC phase telemetry payload."""
    def publish(
        completed_steps: int,
        target_steps: int,
        route_progress_m: list[Optional[float]],
        teacher_poses: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        publish_phase(
            "demonstration_collection",
            completed_steps_per_env=completed_steps,
            target_steps_per_env=target_steps,
            route_progress_m=route_progress_m,
            route_length_m=route.length_m,
            teacher_poses=teacher_poses or [],
        )

    return publish


class _AsyncProgress:
    """Keep hub I/O off the driving loop, with only one update in flight."""
    def __init__(self, callback: Any):
        self.callback = callback
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bc-monitor")
        self.pending = None

    def __enter__(self):
        return self

    def __call__(self, *args):
        if self.pending is None or self.pending.done():
            self.pending = self.pool.submit(self.callback, *args)

    def __exit__(self, *_):
        # Finish the last update before terminal phase publication, so an old
        # collection update cannot overwrite "completed" or "rejected".
        self.pool.shutdown(wait=True)


def _step_cadence_origin(exploration_profile: str, trigger: str, starting_steps: int) -> int:
    """Use global transition buckets for the campaign profile only."""
    if trigger == "steps" and exploration_profile == "bc_campaign":
        return 0
    return int(starting_steps)


class AutomaticEvaluationSchedule:
    """Step-triggered schedule from an explicit origin, or vector-episode based."""

    def __init__(self, trigger: str, interval: int, starting_steps: int) -> None:
        if trigger not in {"steps", "episodes"} or interval < 1:
            raise ValueError("automatic evaluation trigger/interval is invalid")
        self.trigger = trigger
        self.interval = int(interval)
        self.starting_steps = int(starting_steps)
        self.completed_episodes = 0
        self.next_trigger = self.interval

    def observe(self, num_timesteps: int, infos: list[dict]) -> tuple[bool, int, int]:
        self.completed_episodes += sum(
            1 for info in infos
            if isinstance(info, dict) and isinstance(info.get("episode"), dict)
        )
        step = int(num_timesteps)
        elapsed = step - self.starting_steps if self.trigger == "steps" else self.completed_episodes
        due = elapsed >= self.next_trigger
        if due:
            while self.next_trigger <= elapsed:
                self.next_trigger += self.interval
        return due, step, self.completed_episodes


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
    if (
        observation_profile == "official_sensors_camera"
        and not getattr(model.policy.features_extractor, "lidar_pooling", True)
    ):
        architecture = "lidar_camera_raw_cnn"
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
        "target_kl": model.target_kl,
        "policy_architecture": architecture,
        "policy_extractor": type(model.policy.features_extractor).__name__,
        "normalize_images": bool(model.policy.normalize_images),
    }


def _verify_official_ppo_config(model, requested: dict, *, observation_profile: str) -> dict:
    """Fail before collection unless every effective PPO setting is exact."""
    effective = _official_ppo_config(model, observation_profile=observation_profile)
    expected = {
        **requested,
        **_OFFICIAL_FIXED_PPO,
        **requested,
        "rollout_buffer_size": int(requested["n_steps"]),
        "n_envs": int(model.get_env().num_envs),
        "rollout_buffer_envs": int(model.get_env().num_envs),
        "rollout_transitions": int(requested["n_steps"] * model.get_env().num_envs),
        "lr_schedule_at_1": float(requested["learning_rate"]),
        "lr_schedule_at_0": float(requested["learning_rate"]),
        "clip_range_at_1": float(requested.get("clip_range", _OFFICIAL_FIXED_PPO["clip_range"])),
        "clip_range_at_0": float(requested.get("clip_range", _OFFICIAL_FIXED_PPO["clip_range"])),
        "policy_architecture": {
            "official_sensors_history": "temporal_lidar_cnn",
            "official_sensors_camera": "lidar_camera_cnn",
        }.get(observation_profile, "lidar_cnn"),
        "normalize_images": False,
    }
    if observation_profile == "official_sensors_camera" and not getattr(
        model.policy.features_extractor, "lidar_pooling", True
    ):
        expected["policy_architecture"] = "lidar_camera_raw_cnn"
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
    automatic_evaluation_enabled: bool = True,
    automatic_evaluation_trigger: str = "steps",
    automatic_evaluation_interval: int = 50_000,
    automatic_evaluation_mode: str = "competitive",
    exploration_profile: Optional[str] = None,
    exploration_initial_std_throttle: Optional[float] = None,
    exploration_initial_std_steering: Optional[float] = None,
    exploration_min_std_throttle: Optional[float] = None,
    exploration_min_std_steering: Optional[float] = None,
    exploration_max_std_throttle: Optional[float] = None,
    exploration_max_std_steering: Optional[float] = None,
    behavior_cloning_enabled: bool = True,
    behavior_cloning_only: bool = False,
    behavior_cloning_steps_per_env: int = 4000,
    behavior_cloning_epochs: int = 8,
    behavior_cloning_batch_size: int = 256,
    behavior_cloning_learning_rate: float = 1e-3,
    behavior_teacher_max_speed_mps: float = 0.3,
    n_steps: int = 2048,
    learning_rate: float = 1e-4,
    n_epochs: int = 4,
    gamma: float = 0.9995,
    gae_lambda: float = 0.95,
    target_kl: Optional[float] = None,
    timeout_s: float = 180.0,
    training_timeout_s: float = 600.0,
    race_laps: int = 10,
    warmup_laps: int = 0,
    time_cost_per_simulated_second: float = 0.0,
    collision_penalty_magnitude: float = 200.0,
    collision_reward_percent: float = 0.0,
    lap_completion_reward: float = 0.0,
    failed_episode_penalty: float = 200.0,
    backward_speed_penalty_scale: float = 10.0,
    episode_failure_reward_percent: float = 0.0,
    frontier_stagnation_s: float = 5.0,
    observation_profile: str = "official_sensors_camera",
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    throttle_mode: str = "bidirectional",
    negative_throttle_mode: str = "allow",
    steering_mode: str = "normal",
) -> Path:
    """Run one PPO learner over isolated official ROS simulator domains."""
    if exploration_profile is not None and exploration_profile not in {"standard", "bc_smoke", "bc_campaign"}:
        raise ValueError("exploration_profile must be standard, bc_smoke, or bc_campaign")
    saved_exploration_config = _read_resume_exploration_config(resume)
    effective_profile = exploration_profile or (
        str(saved_exploration_config.get("profile"))
        if saved_exploration_config else "standard"
    )
    # Campaign branches share a global transition clock. Ordinary training
    # keeps the historical run-relative cadence when resuming a checkpoint.
    campaign_global_step_cadence = effective_profile == "bc_campaign"
    bc_resume_sha256 = _verified_ready_bc_resume_sha256(resume)
    smoke_preset_applied = effective_profile == "bc_smoke"
    fixed_profile_applied = effective_profile in {"bc_smoke", "bc_campaign"}
    if smoke_preset_applied:
        if not bc_resume_sha256:
            raise ValueError("bc_smoke requires an authenticated BC-derived resume checkpoint")
        if int(total_timesteps) < 0:
            raise ValueError("total_timesteps must be non-negative")
        n_envs = _BC_SMOKE_PPO["n_envs"]
        n_steps = _BC_SMOKE_PPO["n_steps"]
        learning_rate = _BC_SMOKE_PPO["learning_rate"]
        n_epochs = _BC_SMOKE_PPO["n_epochs"]
        gamma = _BC_SMOKE_PPO["gamma"]
        gae_lambda = _BC_SMOKE_PPO["gae_lambda"]
        target_kl = _BC_SMOKE_PPO["target_kl"]
        requested_transitions = (
            int(total_timesteps) if total_timesteps > 0
            else _BC_SMOKE_PPO["max_transitions"]
        )
        transitions_rounded_to_rollout = int(
            math.ceil(requested_transitions / _BC_SMOKE_PPO["n_steps"])
            * _BC_SMOKE_PPO["n_steps"]
        )
        total_timesteps = min(
            _BC_SMOKE_PPO["max_transitions"], transitions_rounded_to_rollout
        )
        checkpoint_every = min(
            max(1, int(checkpoint_every)), _BC_SMOKE_PPO["save_every_transitions"]
        )
        smoke_time_cap_hours = _BC_SMOKE_PPO["max_wall_seconds"] / 3600.0
        max_duration_hours = min(
            max_duration_hours if max_duration_hours > 0 else smoke_time_cap_hours,
            smoke_time_cap_hours,
        )
        time_cost_per_simulated_second = _BC_SMOKE_REWARDS[
            "time_cost_per_simulated_second"
        ]
        collision_penalty_magnitude = _BC_SMOKE_REWARDS["collision_penalty_magnitude"]
        collision_reward_percent = _BC_SMOKE_REWARDS["collision_reward_percent"]
        lap_completion_reward = _BC_SMOKE_REWARDS["lap_completion_reward"]
        failed_episode_penalty = _BC_SMOKE_REWARDS["failed_episode_penalty"]
        backward_speed_penalty_scale = _BC_SMOKE_REWARDS["backward_speed_penalty_scale"]
        episode_failure_reward_percent = _BC_SMOKE_REWARDS[
            "episode_failure_reward_percent"
        ]
        frontier_stagnation_s = _BC_SMOKE_REWARDS["frontier_stagnation_s"]
    if not 1 <= n_envs <= 8:
        raise ValueError("n_envs must be between 1 and 8")
    if total_timesteps < 0:
        raise ValueError("total_timesteps must be zero (unlimited) or positive")
    if target_kl is not None and (not math.isfinite(float(target_kl)) or float(target_kl) <= 0):
        raise ValueError("target_kl must be a finite positive value or None")
    if observation_profile != "official_sensors_camera":
        raise ValueError("official PPO training requires camera, LiDAR, and state observations")
    if checkpoint_every < 1 or n_steps < 1 or n_epochs < 1 or automatic_evaluation_interval < 1:
        raise ValueError("checkpoint and automatic evaluation intervals, n_steps, and n_epochs must be positive")
    if automatic_evaluation_trigger not in {"steps", "episodes"}:
        raise ValueError("automatic_evaluation_trigger must be steps or episodes")
    if automatic_evaluation_mode not in {"competitive", "training_probe"}:
        raise ValueError("automatic_evaluation_mode must be competitive or training_probe")
    if automatic_evaluation_mode == "training_probe" and not automatic_evaluation_enabled:
        raise ValueError("training_probe mode requires automatic evaluation to be enabled")
    resume_lineage = _training_resume_lineage(resume)
    if automatic_evaluation_mode == "training_probe" and not resume_lineage:
        raise ValueError("training_probe evaluation requires a verified BC-derived PPO resume checkpoint")
    if behavior_cloning_enabled and behavior_cloning_steps_per_env < 2:
        raise ValueError("behavior_cloning_steps_per_env must be >= 2 when cloning is enabled")
    if behavior_cloning_only and (not behavior_cloning_enabled or resume is not None):
        raise ValueError("behavior_cloning_only requires a new run with behavior cloning enabled")
    if behavior_cloning_enabled and (
        behavior_cloning_epochs < 1 or behavior_cloning_batch_size < 1
        or behavior_cloning_learning_rate <= 0 or behavior_teacher_max_speed_mps < 0.2
    ):
        raise ValueError("behavior-cloning optimizer and teacher settings are invalid")
    validate_bc_teacher_action_mapping(
        behavior_cloning_enabled=behavior_cloning_enabled,
        resume=resume,
        throttle_mode=throttle_mode,
        negative_throttle_mode=negative_throttle_mode,
        steering_mode=steering_mode,
        steering_action_scale=steering_action_scale,
        straight_throttle_gain=straight_throttle_gain,
    )
    exploration_overrides = {
        "initial_throttle": exploration_initial_std_throttle,
        "initial_steering": exploration_initial_std_steering,
        "min_throttle": exploration_min_std_throttle,
        "min_steering": exploration_min_std_steering,
        "max_throttle": exploration_max_std_throttle,
        "max_steering": exploration_max_std_steering,
    }
    if effective_profile in {"bc_smoke", "bc_campaign"} and not bc_resume_sha256:
        raise ValueError(f"{effective_profile} requires an authenticated BC-derived resume checkpoint")
    if plateau_window_episodes < 2 or plateau_patience < 1:
        raise ValueError("plateau_window_episodes must be >= 2 and plateau_patience >= 1")
    if min(plateau_min_timesteps, max_duration_hours) < 0:
        raise ValueError("plateau_min_timesteps and max_duration_hours must be non-negative")
    if collision_penalty_magnitude <= 120 or failed_episode_penalty <= 120:
        raise ValueError("official terminal penalties must exceed the 120-point progress and centerline reward cap")
    if lap_completion_reward != 0:
        raise ValueError("official lap-learning reward uses verified lap completion as a terminal goal, not a lap bonus")

    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback

    from src.layer3.ppo_audit import (
        CooperativePauseCallback,
        CooperativePauseInterrupted,
        OperatorStopCallback,
        PostOptimizerAuditCallback,
    )
    from src.layer3.official_envs import make_official_vec_env, close_official_vec_env
    from src.layer3.hub_callback import maybe_hub_callback
    from src.layer3.official_policy import resume_ppo_checkpoint
    from src.layer3.train import make_model

    class _AutomaticEvaluationTrigger(BaseCallback):
        """Save immutable PPO snapshots and enqueue official race evaluations."""

        def __init__(self, starting_steps: int) -> None:
            super().__init__(verbose=0)
            self.schedule = AutomaticEvaluationSchedule(
                automatic_evaluation_trigger, automatic_evaluation_interval, starting_steps
            )
            self.index = 0
            self.exploration = None
            self.pending_snapshots: list[tuple[int, int, int]] = []

        def _on_training_start(self) -> None:
            self.exploration = _make_training_exploration_schedule(
                self.model,
                out_dir,
                initial_std=exploration_settings["initial_std"],
                std_min=exploration_settings["std_min"],
                std_max=exploration_settings["std_max"],
            )

        def _on_rollout_start(self) -> None:
            if self.exploration is not None:
                self.exploration.on_rollout_start()
            self._publish_ready_snapshots()

        def _freeze_snapshot(
            self, trigger_step: int, completed_episodes: int, trigger_update: int
        ) -> None:
            self.index += 1
            update = int(getattr(self.model, "_n_updates", 0))
            step = int(self.model.num_timesteps)
            snapshot = out_dir / "checkpoints" / f"automatic_evaluation_{self.index:06d}_steps_{trigger_step}.zip"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(snapshot.with_suffix("")))
            if resume_lineage:
                try:
                    write_ppo_lineage_sidecar(
                        snapshot,
                        parent_lineage=resume_lineage,
                        effective_config=out_dir / "config.json",
                        optimizer_updates=update,
                        num_timesteps=step,
                        training_context=training_context,
                        run_id=os.environ.get("AICAR_RUN_ID", out_dir.name),
                        kind=(
                            "official_training_probe"
                            if automatic_evaluation_mode == "training_probe"
                            else "official_ppo_training_checkpoint"
                        ),
                        extra={
                            "optimizer_updates_at_trigger": trigger_update,
                            "trigger_step": trigger_step,
                            "completed_episodes_at_trigger": completed_episodes,
                        },
                    )
                except (OSError, ValueError) as exc:
                    print(f"Automatic snapshot was saved but lineage was not authenticated: {exc}", flush=True)
                    if automatic_evaluation_mode == "training_probe":
                        return
            hub_url = os.environ.get("HUB_URL", "").rstrip("/")
            run_id = os.environ.get("AICAR_RUN_ID", "")
            payload = json.dumps({
                "model_path": str(snapshot), "trigger": automatic_evaluation_trigger,
                "step": trigger_step, "episode": completed_episodes,
                "mode": automatic_evaluation_mode,
            }).encode("utf-8")
            import urllib.request
            request = urllib.request.Request(
                f"{hub_url}/train/runs/{run_id}/evaluation", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10.0) as response:
                    queued = json.loads(response.read().decode("utf-8"))
                print(
                    "Automatic official evaluation queued: "
                    f"trigger={automatic_evaluation_trigger} step={trigger_step} "
                    f"episode={completed_episodes} id={queued.get('evaluation_id')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Could not queue official evaluation snapshot {snapshot}: {exc}", file=sys.stderr, flush=True)

        def _publish_ready_snapshots(self) -> None:
            if not self.pending_snapshots:
                return
            current_update = int(getattr(self.model, "_n_updates", 0))
            ready = [item for item in self.pending_snapshots if current_update > item[2]]
            self.pending_snapshots = [item for item in self.pending_snapshots if current_update <= item[2]]
            for trigger_step, completed_episodes, trigger_update in ready:
                self._freeze_snapshot(trigger_step, completed_episodes, trigger_update)

        def _on_training_end(self) -> None:
            # SB3 calls this after its final optimizer pass, so a final queued
            # evaluation still receives a post-update immutable checkpoint.
            if self.pending_snapshots:
                self._publish_ready_snapshots()

        def _on_step(self) -> bool:
            nonlocal out_dir
            if self.exploration is not None:
                self.exploration.observe(int(self.model.num_timesteps))
            infos = self.locals.get("infos") or []
            due, step, completed_episodes = self.schedule.observe(int(self.model.num_timesteps), infos)
            if not automatic_evaluation_enabled or not due:
                return True
            hub_url = os.environ.get("HUB_URL", "").rstrip("/")
            run_id = os.environ.get("AICAR_RUN_ID", "")
            if not hub_url or not run_id:
                print("Automatic official evaluation skipped: HUB_URL or AICAR_RUN_ID is unavailable.", flush=True)
                return True
            self.pending_snapshots.append((step, completed_episodes, int(getattr(self.model, "_n_updates", 0))))
            return True

    stop_requested = threading.Event()
    from src.layer3.official_tracks import official_frontier_path, verify_windows_scene
    selected_frontier_path = official_frontier_path()
    scene_verification = (
        verify_windows_scene(selected_frontier_path)
        if os.environ.get("AICAR_SIMULATOR_BACKEND") == "windows_external" else None
    )


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

    (out_dir / "track_profile.json").write_text(json.dumps({
        "centerline_path": str(selected_frontier_path),
        "scene_verification": scene_verification,
        "metadata": json.loads(selected_frontier_path.with_name("official_frontier.json").read_text()),
    }, indent=2), encoding="utf-8")

    vec_env = None
    model = None
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
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
            training_backward_speed_penalty_scale=backward_speed_penalty_scale,
            training_failure_reward_percent=episode_failure_reward_percent,
            training_time_cost_per_simulated_second=time_cost_per_simulated_second,
            training_collision_penalty_magnitude=collision_penalty_magnitude,
            training_collision_reward_percent=collision_reward_percent,
            training_lap_completion_reward=lap_completion_reward,
            training_frontier_stagnation_s=frontier_stagnation_s,
            frontier_path=selected_frontier_path,
        )
        # The BC smoke and campaign profiles both freeze these PPO values.
        # Campaign-specific rollout size, learning rate, and horizon still
        # come from its effective config; ordinary runs retain legacy defaults.
        fixed_ppo = _BC_SMOKE_FIXED_PPO if fixed_profile_applied else _OFFICIAL_FIXED_PPO
        requested_ppo = {
            "n_steps": int(n_steps),
            "learning_rate": float(learning_rate),
            "n_epochs": int(n_epochs),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "target_kl": None if target_kl is None else float(target_kl),
            **fixed_ppo,
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
                target_kl=target_kl,
                **fixed_ppo,
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
                target_kl=target_kl,
                **fixed_ppo,
                seed=seed,
                device=device,
                tensorboard_log=str(out_dir / "tensorboard"),
            )
        cloning_summary = None
        # The centerline teacher is only constructed for a fresh BC collection
        # run. PPO resumes (including authenticated BC-derived warmstarts) still
        # write the same effective configuration, so initialize this explicitly
        # and serialize a null controller when no teacher participated.
        expert = None
        if resume is None and behavior_cloning_enabled:
            from src.layer2.route_progress import RouteProgressTracker
            from src.layer3.behavior_cloning import (
                assess_teacher_collection, behavior_clone_actor,
                collect_centerline_demonstrations,
                initialize_behavior_cloned_action_std,
                save_behavior_cloned_checkpoint,
            )
            from src.layer3.expert import CenterlineExpert

            def publish_phase(phase: str, **details: object) -> None:
                payload = {
                    "kind": "phase", "phase": phase, "step": int(model.num_timesteps),
                    "run_id": os.environ.get("AICAR_RUN_ID", out_dir.name),
                    "runtime": "official", "rollout_size": int(n_steps * n_envs),
                    "ts": datetime.now(timezone.utc).isoformat(), **details,
                }
                (out_dir / "latest_training_phase.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                hub_url = os.environ.get("HUB_URL", "").rstrip("/")
                if hub_url:
                    try:
                        import urllib.request
                        request = urllib.request.Request(
                            hub_url + "/api/train/phase",
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"}, method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=2.0):
                            pass
                    except Exception as exc:
                        print(f"Could not publish {phase} phase to Mission Control: {exc}", file=sys.stderr, flush=True)

            route = RouteProgressTracker.from_csv(selected_frontier_path)
            expert = CenterlineExpert(
                route,
                max_speed_mps=behavior_teacher_max_speed_mps,
                lookahead_base_m=0.8,
            )
            publish_phase(
                "demonstration_collection",
                teacher_controller=type(expert).__name__,
                teacher_inputs=[
                    "official simulator pose and selected track centerline for teacher steering",
                    "measured vehicle speed and front LiDAR for teacher throttle control",
                    "camera, LiDAR, and state observations for the BC student",
                ],
            )

            with _AsyncProgress(_demonstration_progress_publisher(route, publish_phase)) as progress_writer:
                demonstrations, collection_summary = collect_centerline_demonstrations(
                    vec_env, expert,
                    steps=behavior_cloning_steps_per_env,
                    output_path=out_dir / "behavior_cloning_demonstrations.npz",
                    stagnation_s=frontier_stagnation_s,
                    should_stop=stop_requested.is_set,
                    progress_callback=progress_writer,
                )
            if stop_requested.is_set():
                target = out_dir / "official_policy_latest.zip"
                model.save(str(target.with_suffix("")))
                (out_dir / "stop_reason.json").write_text(
                    json.dumps({"reason": "operator_stop_during_behavior_cloning", "num_timesteps": 0}, indent=2),
                    encoding="utf-8",
                )
                publish_phase("stopped", stop_reason="operator_stop_during_behavior_cloning")
                print("Training stopped during teacher collection; saved the initialized policy.", flush=True)
                return target
            if collection_summary["transitions"] < 2:
                raise RuntimeError("official teacher did not produce enough demonstrations to initialize PPO")
            collection_gate = assess_teacher_collection(collection_summary)
            if not collection_gate["ready"]:
                failure = {
                    "status": "rejected",
                    "usable": False,
                    "reason": collection_gate["reason"],
                    "readiness": collection_gate,
                    "transitions": int(collection_summary["transitions"]),
                    "collection": collection_summary,
                    "checkpoint": None,
                }
                (out_dir / "behavior_cloning_summary.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                (out_dir / "stop_reason.json").write_text(
                    json.dumps({
                        "reason": f"behavior_cloning_rejected: {collection_gate['reason']}",
                        "failure_code": "behavior_cloning_rejected",
                        "detail": collection_gate["reason"],
                        "usable_checkpoint": False,
                        "num_timesteps": int(model.num_timesteps),
                    }, indent=2),
                    encoding="utf-8",
                )
                publish_phase(
                    "behavior_cloning_rejected",
                    reason=collection_gate["reason"],
                    readiness=collection_gate,
                    transitions=int(collection_summary["transitions"]),
                    teacher_poses=collection_summary.get("latest_teacher_poses", []),
                )
                print(
                    "Behavior cloning rejected; PPO will not start: "
                    f"{collection_gate['reason']}. Demonstrations and diagnostics were preserved.",
                    file=sys.stderr, flush=True,
                )
                raise RuntimeError(
                    "behavior cloning safety gate rejected teacher collection: "
                    f"{collection_gate['reason']}"
                )
            publish_phase(
                "behavior_cloning",
                transitions=collection_summary["transitions"],
                teacher_poses=collection_summary.get("latest_teacher_poses", []),
                route_progress_m=[
                    item.get("end_progress_m")
                    for item in collection_summary.get("route_direction_diagnostics", [])
                ],
                route_length_m=route.length_m,
            )
            cloning_summary = behavior_clone_actor(
                model, demonstrations, epochs=behavior_cloning_epochs,
                batch_size=behavior_cloning_batch_size,
                learning_rate=behavior_cloning_learning_rate, seed=seed,
                should_stop=stop_requested.is_set,
            )
            if not stop_requested.is_set():
                cloning_summary["initial_action_std"] = (
                    initialize_behavior_cloned_action_std(model)
                )
            cloning_summary["collection"] = collection_summary
            cloning_summary["status"] = "ready"
            cloning_summary["usable"] = True
            cloning_summary["readiness"] = collection_gate
            cloned_checkpoint = save_behavior_cloned_checkpoint(model, out_dir)
            cloning_summary["checkpoint"] = str(cloned_checkpoint)
            cloning_summary["checkpoint_sha256"] = _sha256_file(cloned_checkpoint)
            (out_dir / "behavior_cloning_summary.json").write_text(
                json.dumps(cloning_summary, indent=2), encoding="utf-8"
            )
            print(
                "Official behavior cloning completed: "
                f"{collection_summary['transitions']} sensor/action pairs; "
                f"validation loss {cloning_summary['initial_validation_loss']:.5f} -> "
                f"{cloning_summary['final_validation_loss']:.5f}", flush=True,
            )
            if behavior_cloning_only and not stop_requested.is_set():
                bc_only_config = {
                    "runtime": "official IROS 2026 API and simulator images",
                    "simulator_backend": os.environ.get("AICAR_SIMULATOR_BACKEND", "docker"),
                    "observation_profile": observation_profile,
                    "policy_image_normalization": "custom extractor divides uint8 RGB by 255 exactly once; SB3 image normalization disabled",
                    "actions": {
                        "throttle_mode": throttle_mode,
                        "negative_throttle_mode": negative_throttle_mode,
                        "steering_mode": steering_mode,
                        "steering_action_scale": steering_action_scale,
                        "straight_throttle_gain": straight_throttle_gain,
                        "straight_throttle_steering_threshold": straight_throttle_steering_threshold,
                    },
                    "behavior_cloning": {
                        "enabled": True,
                        "only": True,
                        "teacher": "pure pursuit on the selected official centerline",
                        "steps_per_environment": behavior_cloning_steps_per_env,
                        "checkpoint": cloned_checkpoint.name,
                        "status": "ready",
                        "usable": True,
                        "readiness": collection_gate,
                        "summary": cloning_summary,
                        "policy_inputs": list(model.observation_space.spaces.keys()),
                        "privileged_teacher_state_saved": False,
                    },
                }
                (out_dir / "config.json").write_text(
                    json.dumps(bc_only_config, indent=2), encoding="utf-8"
                )
                (out_dir / "stop_reason.json").write_text(
                    json.dumps({
                        "reason": "behavior_cloning_only_complete",
                        "num_timesteps": 0,
                        "checkpoint": cloned_checkpoint.name,
                    }, indent=2),
                    encoding="utf-8",
                )
                publish_phase(
                    "behavior_cloning_ready",
                    checkpoint=cloned_checkpoint.name,
                    transitions=collection_summary["transitions"],
                    readiness=collection_gate,
                    teacher_poses=collection_summary.get("latest_teacher_poses", []),
                )
                print(
                    f"BC-only run complete; saved policy for official evaluation: {cloned_checkpoint}",
                    flush=True,
                )
                return cloned_checkpoint
            if stop_requested.is_set():
                target = out_dir / "official_policy_latest.zip"
                model.save(str(target.with_suffix("")))
                (out_dir / "stop_reason.json").write_text(
                    json.dumps({"reason": "operator_stop_during_behavior_cloning", "num_timesteps": 0}, indent=2),
                    encoding="utf-8",
                )
                publish_phase("stopped", stop_reason="operator_stop_during_behavior_cloning")
                print("Training stopped during behavior cloning; saved the current policy.", flush=True)
                return target

        model.set_logger(
            configure(
                folder=str(out_dir / "tensorboard"),
                format_strings=["stdout", "csv", "tensorboard"],
            )
        )

        current_std = tuple(
            float(value) for value in model.policy.log_std.detach().exp().cpu().tolist()
        )
        exploration_settings = _resolve_exploration_settings(
            requested_profile=exploration_profile,
            overrides=exploration_overrides,
            saved=saved_exploration_config,
            verified_bc_resume=bool(bc_resume_sha256),
            current_std=current_std,
        )
        if exploration_settings["initial_std"] is not None:
            import torch

            with torch.no_grad():
                model.policy.log_std.copy_(torch.log(torch.as_tensor(
                    exploration_settings["initial_std"],
                    device=model.policy.log_std.device,
                    dtype=model.policy.log_std.dtype,
                )))

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

        current_std = [
            float(value)
            for value in model.policy.log_std.detach().exp().cpu().tolist()
        ]
        resume_parent_record = (
            resume_lineage.get("parent_checkpoint") if resume_lineage else None
        )
        resume_ancestor_record = (
            resume_lineage.get("bc_ancestor_checkpoint") if resume_lineage else None
        )
        resume_verification = str(resume_lineage.get("verification", "")) if resume_lineage else ""
        if resume is None:
            warm_start_source = "new_policy"
        elif "ppo_checkpoint" in resume_verification:
            warm_start_source = "verified_ppo_descendant"
        elif "derived_candidate" in resume_verification:
            warm_start_source = "verified_ppo_warmstart"
        elif "legacy" in resume_verification:
            warm_start_source = "verified_legacy_bc_checkpoint"
        else:
            warm_start_source = "verified_behavior_cloned_checkpoint"
        effective_start_std = [
            min(upper, max(lower, value))
            for value, lower, upper in zip(
                current_std, exploration_settings["std_min"], exploration_settings["std_max"]
            )
        ]
        exploration_config = {
            "profile": exploration_settings["profile"],
            "inherited_from_parent": exploration_settings["inherited_from_resume"],
            "configured_initial_action_std_by_action": (
                None if exploration_settings["initial_std"] is None
                else list(exploration_settings["initial_std"])
            ),
            "warm_start_source": warm_start_source,
            "resume_checkpoint_sha256": (
                resume_parent_record.get("sha256")
                if isinstance(resume_parent_record, dict) else None
            ),
            "bc_ancestor_checkpoint_sha256": (
                resume_ancestor_record.get("sha256")
                if isinstance(resume_ancestor_record, dict) else None
            ),
            "starting_action_std": effective_start_std,
            "minimum_action_std_by_action": list(exploration_settings["std_min"]),
            "maximum_action_std_by_action": list(exploration_settings["std_max"]),
        }

        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": "official IROS 2026 API and simulator images",
            "official_api_image": "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete",
            "official_simulator_image": _official_simulator_image(),
            "simulator_backend": os.environ.get("AICAR_SIMULATOR_BACKEND", "docker"),
            "episode": {
                "frontier_source": f"restricted official IPS position; selected route {selected_frontier_path}",
                "frontier_stagnation_s": frontier_stagnation_s,
                "collision_behavior": "apply a fixed penalty; terminate the episode and reset through the official training API to the configured spawn",
                "training_watchdog_s": training_timeout_s,
                "laps": "end successfully after the configured official lap count when route-frontier distance also confirms at least 95% of the target distance",
            },
            "observation_profile": observation_profile,
            "policy_image_normalization": "custom extractor divides uint8 RGB by 255 exactly once; SB3 image normalization disabled",
            "actions": {
                "throttle_mode": throttle_mode,
                "negative_throttle_mode": negative_throttle_mode,
                "steering_mode": steering_mode,
                "steering_action_scale": steering_action_scale,
                "straight_throttle_gain": straight_throttle_gain,
                "straight_throttle_steering_threshold": straight_throttle_steering_threshold,
            },
            "ppo_exploration": exploration_config,
            "training_only_reward": {
                "time_cost_per_simulated_second": time_cost_per_simulated_second,
                "elapsed_time_source": "ROS LaserScan header timestamps when advancing; monotonic receipt-clock fallback",
                "positive_reward": "new high-water route frontier distance; normalized to a 100-point cap at the configured lap target; no pace multiplier or lap bonus",
                "frontier_progress_per_m": "100 / (frontier_route_length_m * race_laps)",
                "frontier_reward_cap": 100.0,
                "centerline_potential_cap": 20.0,
                "maximum_shaped_progress_return": 120.0,
                "pace_target_mps": 6.0,
                "pace_bonus_strength": 0.0,
                "pace_source": "disabled; each new frontier metre has fixed reward",
                "backward_speed_penalty_per_m": backward_speed_penalty_scale,
                "backward_speed_deadband_mps": 0.1,
                "collision_cost": f"{collision_penalty_magnitude:g} fixed; no frontier clawback; maximum shaped progress return is 120",
                "non_collision_failure_cost": f"{failed_episode_penalty:g} fixed; no frontier clawback; maximum shaped progress return is 120",
                "frontier_stall": f"end and reset life after {frontier_stagnation_s:g} seconds without frontier progress; apply the configured fixed non-collision failure penalty",
                "slip_penalty": 0.0,
                "steering_change_penalty": 0.0,
                "lap_completion_bonus": lap_completion_reward,
                "restricted_topics_used_only_for_reward_and_episode_control": True,
                "restricted_topics_in_policy_observation": False,
            },
            "ppo": {
                "preset": (
                    "bc_smoke_locked" if smoke_preset_applied else
                    "bc_campaign_locked" if effective_profile == "bc_campaign" else
                    "ordinary"
                ),
                "total_timesteps": total_timesteps,
                "maximum_transitions": (
                    _BC_SMOKE_PPO["max_transitions"] if smoke_preset_applied else None
                ),
                "maximum_wall_seconds": (
                    _BC_SMOKE_PPO["max_wall_seconds"] if smoke_preset_applied else None
                ),
                "post_optimizer_save_every_transitions": checkpoint_every,
                "checkpoint_interval_scope": (
                    "global_step_buckets" if campaign_global_step_cadence else "run_relative"
                ),
                "rewards_locked_to_official_defaults": fixed_profile_applied,
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
            "automatic_evaluation": {
                "enabled": automatic_evaluation_enabled,
                "trigger": automatic_evaluation_trigger,
                "interval": automatic_evaluation_interval,
                "mode": automatic_evaluation_mode,
                "interval_scope": (
                    "global_step_buckets"
                    if campaign_global_step_cadence and automatic_evaluation_trigger == "steps"
                    else "run_relative" if automatic_evaluation_trigger == "steps"
                    else "completed_episodes"
                ),
                "attempts_per_snapshot": 1 if automatic_evaluation_mode == "training_probe" else 3,
                "protocol": (
                    "one warmup lap followed by three nonranking probe laps"
                    if automatic_evaluation_mode == "training_probe"
                    else "one warmup lap followed by ten scored race laps"
                ),
            },
            "automatic_evaluation_mode": automatic_evaluation_mode,
            "training_probe": {
                "enabled": automatic_evaluation_mode == "training_probe",
                "ranking": False,
                "simulation_timeout_s": 240.0,
                "wall_timeout_s": 240.0,
                "frontier_stagnation_s": 5.0,
                "lap_counts_by_environment": [1, 3],
                "stop_on_first_collision_including_warmup": True,
            },
            "optimizer_audit": {
                "enabled": True,
                "save_after_optimizer_update": True,
                "save_every_transitions": int(checkpoint_every),
                "record": "ppo_optimizer_audit.jsonl",
            },
            "cooperative_pause": {
                "enabled": True,
                "request_file": "evaluation_pause.request.json",
                "ack_file": "evaluation_pause.ack.json",
                "resume_request_file": "evaluation_resume.request.json",
                "resume_ack_file": "evaluation_resume.ack.json",
                "safe_boundary": "after_optimizer_update_before_next_rollout_action",
                "pause_boundary_timeout_seconds": 240,
                "resume_request_wait_timeout_seconds": 9000,
                "resume_request_wait_timeout_basis": (
                    "3 competitive attempts x (11 laps x 240s wall-progress limit + 180s startup) "
                    "+ 540s orchestration margin"
                ),
                "resume_request_wait_timeout_starts_after": "paused_ack_written",
                "resume_freshness_timeout_seconds": 120,
                "resume_freshness_timeout_starts_after": "matching_resume_request_received",
                "reset_episode_after_simulator_restart": True,
                "interruption_is_policy_failure": False,
            },
            "behavior_cloning": {
                "enabled": bool(behavior_cloning_enabled and resume is None),
                "teacher": "pure pursuit on the selected official centerline",
                "teacher_max_speed_mps": behavior_teacher_max_speed_mps,
                "teacher_controller": type(expert).__name__ if expert is not None else None,
                "steps_per_environment": behavior_cloning_steps_per_env,
                "epochs": behavior_cloning_epochs,
                "batch_size": behavior_cloning_batch_size,
                "learning_rate": behavior_cloning_learning_rate,
                "summary": cloning_summary,
                "policy_inputs": list(model.observation_space.spaces.keys()),
                "privileged_teacher_state_saved": False,
            },
            "exploration_schedule": {
                "profile": exploration_settings["profile"],
                "minimum_action_std_by_action": list(exploration_settings["std_min"]),
                "maximum_action_std_by_action": list(exploration_settings["std_max"]),
                "after_better_valid_ten_lap_score": 0.9,
                "after_three_stale_official_evaluations": 1.1,
                "evaluation_metric": "official_valid_ten_lap_time",
                "nonranking_modes_ignored": ["bc_diagnostic", "training_probe"],
            },
            "resume": str(Path(resume).resolve()) if resume else None,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        training_context = _training_context_metadata()
        checkpoint_callback = PostOptimizerAuditCallback(
            out_dir,
            save_every_transitions=checkpoint_every,
            global_step_cadence=campaign_global_step_cadence,
            lineage={
                "parent_lineage": resume_lineage,
                "effective_config": out_dir / "config.json",
                "training_context": training_context,
                "run_id": os.environ.get("AICAR_RUN_ID", out_dir.name),
            } if resume_lineage else None,
        )
        hub_callback = maybe_hub_callback(
            os.environ.get("AICAR_RUN_ID", out_dir.name),
            runtime="official",
            run_dir=out_dir,
        )
        operator_stop_callback = OperatorStopCallback(stop_requested)
        callbacks = [checkpoint_callback, operator_stop_callback]
        starting_steps = int(model.num_timesteps)
        if automatic_evaluation_enabled or exploration_settings["profile"] != "standard":
            schedule_origin = _step_cadence_origin(
                effective_profile, automatic_evaluation_trigger, starting_steps
            )
            callbacks.append(_AutomaticEvaluationTrigger(schedule_origin))
        if hub_callback is not None:
            callbacks.append(hub_callback)
            print(
                f"Watch telemetry enabled: HUB_URL is set (run_id={out_dir.name})",
                flush=True,
            )
        stop_callback = _TrainingStopCallback(starting_steps, time.monotonic())
        callbacks.append(stop_callback)
        cooperative_pause_callback = CooperativePauseCallback(
            out_dir,
            os.environ.get("AICAR_RUN_ID", out_dir.name),
            stop_requested,
            checkpoint_callback,
            resume_request_wait_timeout_s=9000.0,
            resume_freshness_timeout_s=120.0,
        )
        callbacks.append(cooperative_pause_callback)
        try:
            model.learn(
                total_timesteps=total_timesteps if total_timesteps > 0 else sys.maxsize,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=resume is None,
            )
        except CooperativePauseInterrupted:
            # The callback runs before the next rollout starts, after the last
            # optimizer update. Stop here without collecting another action;
            # the prior update checkpoint remains complete and authenticated.
            checkpoint_callback.on_training_end()
            print("Training stopped at a cooperative pause boundary.", flush=True)
        if os.environ.get("HUB_URL") and behavior_cloning_enabled and resume is None:
            payload = {
                "kind": "phase", "phase": "rollout", "step": int(model.num_timesteps),
                "run_id": os.environ.get("AICAR_RUN_ID", out_dir.name),
                "runtime": "official", "rollout_size": int(n_steps * n_envs),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            try:
                import urllib.request
                request = urllib.request.Request(
                    os.environ["HUB_URL"].rstrip("/") + "/api/train/phase",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(request, timeout=2.0):
                    pass
            except Exception:
                pass
        target = out_dir / "official_policy_latest.zip"
        model.save(str(target.with_suffix("")))
        if resume_lineage:
            try:
                write_ppo_lineage_sidecar(
                    target,
                    parent_lineage=resume_lineage,
                    effective_config=out_dir / "config.json",
                    optimizer_updates=int(getattr(model, "_n_updates", 0)),
                    num_timesteps=int(model.num_timesteps),
                    training_context=training_context,
                    run_id=os.environ.get("AICAR_RUN_ID", out_dir.name),
                )
            except (OSError, ValueError) as exc:
                print(f"Final checkpoint saved without authenticated PPO lineage: {exc}", flush=True)
        print(f"Saved official-simulator PPO checkpoint: {target}", flush=True)
        stop_reason = operator_stop_callback.stop_reason or stop_callback.stop_reason or (
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
    parser.add_argument("--automatic-evaluation-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--automatic-evaluation-trigger", choices=("steps", "episodes"), default="steps")
    parser.add_argument("--automatic-evaluation-interval", type=int, default=50_000)
    parser.add_argument("--automatic-evaluation-mode", choices=("competitive", "training_probe"), default="competitive")
    parser.add_argument("--exploration-profile", choices=("standard", "bc_smoke", "bc_campaign"))
    parser.add_argument("--exploration-initial-std-throttle", type=float)
    parser.add_argument("--exploration-initial-std-steering", type=float)
    parser.add_argument("--exploration-min-std-throttle", type=float)
    parser.add_argument("--exploration-min-std-steering", type=float)
    parser.add_argument("--exploration-max-std-throttle", type=float)
    parser.add_argument("--exploration-max-std-steering", type=float)
    parser.add_argument("--behavior-cloning-enabled", type=int, choices=(0, 1), default=1)
    parser.add_argument("--behavior-cloning-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--behavior-cloning-steps-per-env", type=int, default=4000)
    parser.add_argument("--behavior-cloning-epochs", type=int, default=8)
    parser.add_argument("--behavior-cloning-batch-size", type=int, default=256)
    parser.add_argument("--behavior-cloning-learning-rate", type=float, default=1e-3)
    parser.add_argument("--behavior-teacher-max-speed-mps", type=float, default=1.2)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.9995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--training-timeout-s", type=float, default=600.0)
    parser.add_argument("--race-laps", type=int, default=10)
    parser.add_argument("--warmup-laps", type=int, default=0)
    parser.add_argument("--time-cost-per-simulated-second", type=float, default=0.0)
    parser.add_argument("--collision-penalty-magnitude", type=float, default=200.0)
    parser.add_argument("--collision-reward-percent", type=float, default=0.0)
    parser.add_argument("--lap-completion-reward", type=float, default=0.0)
    parser.add_argument("--failed-episode-penalty", type=float, default=200.0)
    parser.add_argument("--backward-speed-penalty-scale", type=float, default=10.0)
    parser.add_argument("--episode-failure-reward-percent", type=float, default=0.0)
    parser.add_argument("--frontier-stagnation-s", type=float, default=5.0)
    parser.add_argument(
        "--observation-profile",
        choices=("official_sensors", "official_sensors_history", "official_sensors_camera"),
        default="official_sensors_camera",
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
        automatic_evaluation_enabled=bool(args.automatic_evaluation_enabled),
        automatic_evaluation_trigger=args.automatic_evaluation_trigger,
        automatic_evaluation_interval=args.automatic_evaluation_interval,
        automatic_evaluation_mode=args.automatic_evaluation_mode,
        exploration_profile=args.exploration_profile,
        exploration_initial_std_throttle=args.exploration_initial_std_throttle,
        exploration_initial_std_steering=args.exploration_initial_std_steering,
        exploration_min_std_throttle=args.exploration_min_std_throttle,
        exploration_min_std_steering=args.exploration_min_std_steering,
        exploration_max_std_throttle=args.exploration_max_std_throttle,
        exploration_max_std_steering=args.exploration_max_std_steering,
        behavior_cloning_enabled=bool(args.behavior_cloning_enabled),
        behavior_cloning_only=bool(args.behavior_cloning_only),
        behavior_cloning_steps_per_env=args.behavior_cloning_steps_per_env,
        behavior_cloning_epochs=args.behavior_cloning_epochs,
        behavior_cloning_batch_size=args.behavior_cloning_batch_size,
        behavior_cloning_learning_rate=args.behavior_cloning_learning_rate,
        behavior_teacher_max_speed_mps=args.behavior_teacher_max_speed_mps,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        target_kl=args.target_kl,
        timeout_s=args.timeout_s,
        training_timeout_s=args.training_timeout_s,
        race_laps=args.race_laps,
        warmup_laps=args.warmup_laps,
        time_cost_per_simulated_second=args.time_cost_per_simulated_second,
        collision_penalty_magnitude=args.collision_penalty_magnitude,
        collision_reward_percent=args.collision_reward_percent,
        lap_completion_reward=args.lap_completion_reward,
        failed_episode_penalty=args.failed_episode_penalty,
        backward_speed_penalty_scale=args.backward_speed_penalty_scale,
        episode_failure_reward_percent=args.episode_failure_reward_percent,
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
