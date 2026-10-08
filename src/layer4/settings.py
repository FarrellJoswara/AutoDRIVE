"""Shared Mission Control Settings — maps 1:1 to train.py CLI flags + hub-only fields."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = ROOT / "logs" / "layer4" / "settings.json"


class Settings(BaseModel):
    """Train CLI flags + hub-only telemetry knobs."""

    # Train / job — defaults aligned with src/layer3/train.py
    n_envs: int = Field(default=1, ge=1, le=16)
    # 0 means no total-step cap; the plateau callback decides when to stop.
    timesteps: int = Field(default=0, ge=0)
    max_duration_seconds: float = Field(default=0.0, ge=0)
    plateau_min_timesteps: int = Field(default=100_000, ge=0)
    plateau_patience: int = Field(default=5, ge=1)
    plateau_min_improvement_pct: float = Field(default=1.0, ge=0)
    evaluation_every_timesteps: int = Field(default=50_000, ge=1)
    evaluation_runs_per_snapshot: int = Field(default=3, ge=1, le=10)
    evaluation_metric: Literal[
        "frontier_speed", "reward_per_simulated_second", "total_reward", "ten_lap_time"
    ] = "total_reward"
    ppo_learning_rate: float = Field(default=3e-4, gt=0, le=0.01)
    ppo_n_steps: int = Field(default=1024, ge=64, le=8192, multiple_of=64)
    ppo_n_epochs: int = Field(default=8, ge=1, le=20)
    ppo_gamma: float = Field(default=0.99, gt=0, le=1)
    ppo_gae_lambda: float = Field(default=0.95, ge=0, le=1)
    exploration_std_min: float = Field(default=0.2, gt=0)
    exploration_std_max: float = Field(default=0.8, gt=0)
    exploration_improvement_scale: float = Field(default=0.9, gt=0, le=1)
    exploration_plateau_scale: float = Field(default=1.1, ge=1)
    out: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    resume: Optional[str] = None

    # Env kwargs — match train.py CLI defaults for a usable first policy
    headless: bool = True
    auto_launch: bool = True
    simulator_mode: Literal["legacy", "fixed_camera_on", "fixed_camera_off"] = "legacy"
    action_interval_s: Optional[float] = Field(default=None, gt=0)
    observation_profile: Literal["simulator", "simulator_camera", "official_sensors", "official_sensors_history", "official_sensors_camera"] = "simulator_camera"
    throttle_mode: Literal["bidirectional", "forward_only"] = "bidirectional"
    policy_architecture: Literal["lidar_cnn", "lidar_cnn_pooled", "temporal_lidar_cnn", "lidar_camera_cnn"] = "lidar_camera_cnn"
    steering_action_scale: float = Field(default=1.0, ge=0, le=1)
    straight_throttle_gain: float = Field(default=1.0, ge=1.0, le=2.0)
    straight_throttle_steering_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    connect_timeout: float = Field(default=90.0, gt=0)
    frame_skip: int = Field(default=4, ge=1)
    max_episode_steps: int = Field(default=0, ge=0)
    laps_per_episode: int = Field(default=10, ge=0)
    stagnation_speed_threshold: float = 0.15
    stagnation_steps: int = Field(default=50, ge=0)
    frontier_stagnation_seconds: float = Field(default=10.0, ge=0)
    terminate_on_collision: bool = True
    forward_scale: float = 0.0
    backward_speed_penalty_scale: float = Field(default=1.0, ge=0)
    route_progress_scale: float = Field(default=10.0, ge=0)
    frontier_pace_target_mps: float = Field(default=6.0, gt=0)
    frontier_pace_bonus_strength: float = Field(default=1.0, ge=0)
    frontier_pace_source: Literal["episode_average", "current_push"] = "episode_average"
    time_penalty_per_second: float = Field(default=5.0, ge=0)
    collision_penalty_magnitude: float = Field(default=100.0, ge=0)
    collision_reward_percent: float = Field(default=100.0, ge=0, le=100)
    episode_failure_penalty_magnitude: float = Field(default=100.0, ge=0)
    episode_failure_reward_percent: float = Field(default=100.0, ge=0, le=100)
    slip_penalty: float = 0.2
    steer_jerk_penalty: float = 0.05

    # Hub-only (not train argv)
    telemetry_every_n: int = Field(default=200, ge=1)
    fleet_hz: float = Field(default=15.0, gt=0)
    lidar_display_beams: int = Field(default=120, ge=1)
    telemetry_lidar_max_envs: int = Field(default=4, ge=0)
    docker_mode: bool = False
    stop_sims_on_train_exit: bool = True
    stop_stack_on_train_exit: bool = False
    # Mission Control map selection — locked in on Train Start (Watch underlay).
    # "none" = builtin Unity track (grid underlay only).
    map_id: str = "none"

    @field_validator("exploration_std_max")
    @classmethod
    def _exploration_max_gte_min(cls, value: float, info: Any) -> float:
        minimum = info.data.get("exploration_std_min")
        if minimum is not None and value < minimum:
            raise ValueError("exploration_std_max must be >= exploration_std_min")
        return value

    @field_validator("device", mode="before")
    @classmethod
    def _norm_device(cls, v: Any) -> str:
        s = str(v).lower().strip()
        if s not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto|cpu|cuda")
        return s

    def resolve_out(self) -> Path:
        """Resolve run output directory under logs/rl/."""
        if self.out:
            p = Path(self.out)
            return p if p.is_absolute() else ROOT / p
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (self.run_name or "ppo").strip() or "ppo"
        # sanitize simple path segment
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return ROOT / "logs" / "rl" / f"{safe}_{stamp}"

    def effective_auto_launch(self) -> bool:
        if self.docker_mode:
            return False
        return self.auto_launch

    def effective_action_interval(self) -> Optional[float]:
        if self.simulator_mode == "legacy":
            return None
        return self.action_interval_s or 0.086

    def simulator_env(self, *, containerized: Optional[bool] = None) -> Dict[str, str]:
        """Environment for the selected simulator mode.

        Docker replicas need the volume-mounted path, while a host-launched
        simulator needs the repository path. Callers may override detection
        when the hub is managing containers from outside Docker.
        """
        if containerized is None:
            containerized = os.environ.get("AICAR_IN_DOCKER", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
        legacy_path = (
            "/app/simulator/AutoDRIVE Simulator.x86_64"
            if containerized
            else str(ROOT / "simulator" / "AutoDRIVE Simulator.x86_64")
        )
        fixed_path = (
            "/app/simulator/_build/linux-fixed-step-experiment/AutoDRIVE Simulator.x86_64"
            if containerized
            else str(ROOT / "simulator" / "_build" / "linux-fixed-step-experiment" / "AutoDRIVE Simulator.x86_64")
        )
        if self.simulator_mode == "legacy":
            return {
                "AICAR_ACTION_INTERVAL_SECONDS": "",
                "AICAR_ACTION_IDLE_TARGET_FPS": "",
                "AICAR_DISABLE_CAMERA_STREAM": "",
                "AICAR_SIMULATOR_PATH": legacy_path,
            }
        return {
            "AICAR_ACTION_INTERVAL_SECONDS": f"{self.effective_action_interval():.17g}",
            # Lower idle polling only when the camera-free mode is scaling to a
            # larger pool. The cap applies while physics is paused between
            # actions; the fixed-tick action batch remains uncapped.
            "AICAR_ACTION_IDLE_TARGET_FPS": (
                "10" if self.simulator_mode == "fixed_camera_off" and self.n_envs >= 8
                else "30"
            ),
            "AICAR_DISABLE_CAMERA_STREAM": "1" if self.simulator_mode == "fixed_camera_off" else "0",
            "AICAR_SIMULATOR_PATH": fixed_path,
        }

    def to_env_kwargs(
        self, *, headless: bool = True, auto_launch: bool = False,
        map_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the Layer 2 environment settings used by replay/playback."""
        return {
            "headless": headless,
            "auto_launch": auto_launch,
            "connect_timeout": self.connect_timeout,
            "frame_skip": self.frame_skip,
            "action_interval_s": self.effective_action_interval(),
            "steering_action_scale": self.steering_action_scale,
            "throttle_mode": self.throttle_mode,
            "straight_throttle_gain": self.straight_throttle_gain,
            "straight_throttle_steering_threshold": self.straight_throttle_steering_threshold,
            "observation_profile": self.observation_profile,
            "max_episode_steps": self.max_episode_steps,
            "stagnation_speed_threshold": self.stagnation_speed_threshold,
            "stagnation_steps": self.stagnation_steps,
            "map_id": self.map_id if map_id is None else map_id,
            "laps_per_episode": self.laps_per_episode,
            "frontier_stagnation_seconds": self.frontier_stagnation_seconds,
            "terminate_on_collision": self.terminate_on_collision,
            "route_progress_scale": self.route_progress_scale,
            "frontier_pace_target_mps": self.frontier_pace_target_mps,
            "frontier_pace_bonus_strength": self.frontier_pace_bonus_strength,
            "frontier_pace_source": self.frontier_pace_source,
            "time_penalty_per_second": self.time_penalty_per_second,
            "forward_scale": self.forward_scale,
            "backward_speed_penalty_scale": self.backward_speed_penalty_scale,
            "collision_penalty_magnitude": self.collision_penalty_magnitude,
            "collision_reward_percent": self.collision_reward_percent,
            "episode_failure_penalty_magnitude": self.episode_failure_penalty_magnitude,
            "episode_failure_reward_percent": self.episode_failure_reward_percent,
            "slip_penalty": self.slip_penalty,
            "steer_jerk_penalty": self.steer_jerk_penalty,
        }

    def to_train_argv(self) -> List[str]:
        """Build argv list for `python -m src.layer3.train` (flags only, no module)."""
        out = self.resolve_out()
        auto_launch = self.effective_auto_launch()
        argv: List[str] = [
            "--n-envs",
            str(self.n_envs),
            "--timesteps",
            str(self.timesteps),
            "--max-duration-seconds",
            str(self.max_duration_seconds),
            "--plateau-min-timesteps",
            str(self.plateau_min_timesteps),
            "--plateau-patience",
            str(self.plateau_patience),
            "--plateau-min-improvement-pct",
            str(self.plateau_min_improvement_pct),
            "--evaluation-every-timesteps",
            str(self.evaluation_every_timesteps),
            "--evaluation-runs-per-snapshot",
            str(self.evaluation_runs_per_snapshot),
            "--learning-rate",
            str(self.ppo_learning_rate),
            "--n-steps",
            str(self.ppo_n_steps),
            "--n-epochs",
            str(self.ppo_n_epochs),
            "--policy-architecture",
            self.policy_architecture,
            "--gamma",
            str(self.ppo_gamma),
            "--gae-lambda",
            str(self.ppo_gae_lambda),
            "--evaluation-metric",
            self.evaluation_metric,
            "--exploration-std-min",
            str(self.exploration_std_min),
            "--exploration-std-max",
            str(self.exploration_std_max),
            "--exploration-improvement-scale",
            str(self.exploration_improvement_scale),
            "--exploration-plateau-scale",
            str(self.exploration_plateau_scale),
            "--out",
            str(out),
            "--seed",
            str(self.seed),
            "--device",
            self.device,
            "--connect-timeout",
            str(self.connect_timeout),
            "--frame-skip",
            str(self.frame_skip),
            "--observation-profile",
            self.observation_profile,
            "--throttle-mode",
            self.throttle_mode,
            "--steering-action-scale",
            str(self.steering_action_scale),
            "--straight-throttle-gain",
            str(self.straight_throttle_gain),
            "--straight-throttle-steering-threshold",
            str(self.straight_throttle_steering_threshold),
            "--max-episode-steps",
            str(self.max_episode_steps),
            "--laps-per-episode",
            str(self.laps_per_episode),
            "--stagnation-speed-threshold",
            str(self.stagnation_speed_threshold),
            "--stagnation-steps",
            str(self.stagnation_steps),
            "--map-id",
            self.map_id,
            "--frontier-stagnation-seconds",
            str(self.frontier_stagnation_seconds),
            "--terminate-on-collision" if self.terminate_on_collision else "--no-terminate-on-collision",
            "--forward-scale",
            str(self.forward_scale),
            "--backward-speed-penalty-scale",
            str(self.backward_speed_penalty_scale),
            "--route-progress-scale",
            str(self.route_progress_scale),
            "--frontier-pace-target-mps",
            str(self.frontier_pace_target_mps),
            "--frontier-pace-bonus-strength",
            str(self.frontier_pace_bonus_strength),
            "--frontier-pace-source",
            self.frontier_pace_source,
            "--time-penalty-per-second",
            str(self.time_penalty_per_second),
            "--collision-penalty-magnitude",
            str(self.collision_penalty_magnitude),
            "--collision-reward-percent",
            str(self.collision_reward_percent),
            "--episode-failure-penalty-magnitude",
            str(self.episode_failure_penalty_magnitude),
            "--episode-failure-reward-percent",
            str(self.episode_failure_reward_percent),
            "--slip-penalty",
            str(self.slip_penalty),
            "--steer-jerk-penalty",
            str(self.steer_jerk_penalty),
        ]
        action_interval = self.effective_action_interval()
        if action_interval is not None:
            argv.extend(["--action-interval-s", str(action_interval)])
        if self.resume:
            argv.extend(["--resume", str(self.resume)])
        argv.append("--headless" if self.headless else "--no-headless")
        argv.append("--auto-launch" if auto_launch else "--no-auto-launch")
        return argv

    def hub_env(self) -> Dict[str, str]:
        """Extra env vars for the train child (telemetry knobs)."""
        return {
            "AICAR_TELEMETRY_EVERY_N": str(self.telemetry_every_n),
            "AICAR_FLEET_HZ": str(self.fleet_hz),
            "AICAR_LIDAR_DISPLAY_BEAMS": str(self.lidar_display_beams),
            "AICAR_TELEMETRY_LIDAR_MAX_ENVS": str(self.telemetry_lidar_max_envs),
            **self.simulator_env(),
        }


def load_settings(path: Optional[Path] = None) -> Settings:
    path = path or DEFAULT_SETTINGS_PATH
    if not path.is_file():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Settings.model_validate(data)
    except Exception:
        return Settings()


def save_settings(settings: Settings, path: Optional[Path] = None) -> Path:
    path = path or DEFAULT_SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path
