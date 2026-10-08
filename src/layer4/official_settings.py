"""Authoritative Layer 4 settings for official-simulator PPO runs."""

from __future__ import annotations

from typing import Literal, Optional
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.layer4.settings import ROOT


OFFICIAL_SETTINGS_PATH = ROOT / "logs" / "layer4" / "official_train_settings.json"


class OfficialTrainSettings(BaseModel):
    """Competition settings accepted by the official-only run manager.

    Run identity, output directory, simulator ports, and container names are
    assigned by Layer 4 and intentionally cannot be supplied here.
    """

    # Zero means no timestep ceiling; plateau and optional wall-clock stop rules still apply.
    n_envs: int = Field(default=1, ge=1, le=8)
    total_timesteps: int = Field(default=1_000_000, ge=0)
    stop_on_plateau: bool = True
    plateau_min_timesteps: int = Field(default=100_000, ge=0)
    plateau_window_episodes: int = Field(default=5, ge=2, le=100)
    plateau_patience: int = Field(default=5, ge=1, le=100)
    plateau_min_improvement_pct: float = Field(default=1.0, ge=0, le=100)
    max_duration_hours: float = Field(default=0.0, ge=0)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    device: Literal["cpu", "cuda"] = "cpu"
    resume: Optional[str] = None
    checkpoint_every: int = Field(default=10_000, ge=1)
    n_steps: int = Field(default=2048, ge=64, multiple_of=64)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.01)
    n_epochs: int = Field(default=4, ge=1, le=20)
    gamma: float = Field(default=0.9995, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    timeout_s: float = Field(default=30.0, gt=0)
    training_timeout_s: float = Field(default=600.0, gt=0)
    frontier_stagnation_s: float = Field(default=10.0, ge=0)
    race_laps: int = Field(default=10, ge=1, le=100)
    warmup_laps: int = Field(default=0, ge=0, le=20)
    time_cost_per_simulated_second: float = Field(default=5.0, ge=0)
    collision_penalty_magnitude: float = Field(default=100.0, ge=0)
    collision_reward_percent: float = Field(default=20.0, ge=0, le=1000)
    failed_episode_penalty: float = Field(default=100.0, ge=0)
    observation_profile: Literal[
        "official_sensors", "official_sensors_history", "official_sensors_camera"
    ] = "official_sensors"
    steering_action_scale: float = Field(default=1.0, ge=0, le=1)
    straight_throttle_gain: float = Field(default=1.0, ge=1, le=2)
    straight_throttle_steering_threshold: float = Field(default=0.15, ge=0, le=1)
    throttle_mode: Literal["bidirectional", "forward_only"] = "bidirectional"
    negative_throttle_mode: Literal["allow", "zero", "positive_magnitude"] = "allow"
    steering_mode: Literal["normal", "invert"] = "normal"

    def to_container_env(self, *, hub_url: str, run_id: str) -> dict[str, str]:
        """Build the exact trainer environment for one manager-assigned run."""
        values = {
            "AICAR_MODE": "train",
            "AICAR_TRAIN_N_ENVS": str(self.n_envs),
            "HUB_URL": hub_url,
            "AICAR_RUN_ID": run_id,
            "AICAR_TRAIN_OUT": f"/runs/rl/official_run_{run_id}",
            "AICAR_TRAIN_TIMESTEPS": str(self.total_timesteps),
            "AICAR_TRAIN_STOP_ON_PLATEAU": "1" if self.stop_on_plateau else "0",
            "AICAR_TRAIN_PLATEAU_MIN_TIMESTEPS": str(self.plateau_min_timesteps),
            "AICAR_TRAIN_PLATEAU_WINDOW_EPISODES": str(self.plateau_window_episodes),
            "AICAR_TRAIN_PLATEAU_PATIENCE": str(self.plateau_patience),
            "AICAR_TRAIN_PLATEAU_MIN_IMPROVEMENT_PCT": str(self.plateau_min_improvement_pct),
            "AICAR_TRAIN_MAX_DURATION_HOURS": str(self.max_duration_hours),
            "AICAR_TRAIN_SEED": str(self.seed),
            "AICAR_PPO_DEVICE": self.device,
            "AICAR_TRAIN_CHECKPOINT_EVERY": str(self.checkpoint_every),
            "AICAR_PPO_N_STEPS": str(self.n_steps),
            "AICAR_PPO_LEARNING_RATE": str(self.learning_rate),
            "AICAR_PPO_N_EPOCHS": str(self.n_epochs),
            "AICAR_PPO_GAMMA": str(self.gamma),
            "AICAR_PPO_GAE_LAMBDA": str(self.gae_lambda),
            "AICAR_SENSOR_TIMEOUT_S": str(self.timeout_s),
            "AICAR_TRAIN_WATCHDOG_S": str(self.training_timeout_s),
            "AICAR_TRAIN_FRONTIER_STAGNATION_S": str(self.frontier_stagnation_s),
            "AICAR_TRAIN_RACE_LAPS": str(self.race_laps),
            "AICAR_TRAIN_WARMUP_LAPS": str(self.warmup_laps),
            "AICAR_TRAIN_TIME_COST": str(self.time_cost_per_simulated_second),
            "AICAR_TRAIN_COLLISION_PENALTY_MAGNITUDE": str(self.collision_penalty_magnitude),
            "AICAR_TRAIN_COLLISION_REWARD_PERCENT": str(self.collision_reward_percent),
            "AICAR_TRAIN_FAILURE_PENALTY": str(self.failed_episode_penalty),
            "AICAR_OBSERVATION_PROFILE": self.observation_profile,
            "AICAR_STEERING_ACTION_SCALE": str(self.steering_action_scale),
            "AICAR_STRAIGHT_THROTTLE_GAIN": str(self.straight_throttle_gain),
            "AICAR_STRAIGHT_THROTTLE_STEERING_THRESHOLD": str(self.straight_throttle_steering_threshold),
            "AICAR_THROTTLE_MODE": self.throttle_mode,
            "AICAR_NEGATIVE_THROTTLE_MODE": self.negative_throttle_mode,
            "AICAR_STEERING_MODE": self.steering_mode,
        }
        if self.resume:
            values["AICAR_TRAIN_RESUME"] = self.resume
        return values
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def load_official_train_settings(path: Path = OFFICIAL_SETTINGS_PATH) -> OfficialTrainSettings:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return OfficialTrainSettings()
    if isinstance(raw, dict):
        # Older UI settings used lap bonuses and escalating collision penalties.
        # Drop those obsolete controls while retaining the user's unrelated PPO/run settings.
        legacy_reward_schema = "lap_completion_reward" in raw or "collision_penalty_base" in raw
        for legacy in ("lap_completion_reward", "collision_penalty_base", "lidar_brake_distance_m"):
            raw.pop(legacy, None)
        if legacy_reward_schema:
            raw.update({
                "warmup_laps": 0,
                "time_cost_per_simulated_second": 5.0,
                "collision_penalty_magnitude": 100.0,
                "collision_reward_percent": 20.0,
                "failed_episode_penalty": 100.0,
                "frontier_stagnation_s": 10.0,
            })
    return OfficialTrainSettings.model_validate(raw)


def save_official_train_settings(
    settings: OfficialTrainSettings, path: Path = OFFICIAL_SETTINGS_PATH
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    temp.replace(path)
    return path
