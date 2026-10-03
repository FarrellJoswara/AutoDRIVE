"""Shared Mission Control Settings — maps 1:1 to train.py CLI flags + hub-only fields."""

from __future__ import annotations

import json
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
    stop_after_laps: int = Field(default=0, ge=0)
    plateau_min_timesteps: int = Field(default=100_000, ge=0)
    plateau_window_timesteps: int = Field(default=25_000, ge=1)
    plateau_patience: int = Field(default=5, ge=1)
    plateau_min_improvement_pct: float = Field(default=1.0, ge=0)
    out: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    resume: Optional[str] = None

    # Env kwargs — match train.py CLI defaults for a usable first policy
    headless: bool = True
    auto_launch: bool = True
    connect_timeout: float = Field(default=90.0, gt=0)
    frame_skip: int = Field(default=4, ge=1)
    max_episode_steps: int = Field(default=0, ge=0)
    stagnation_speed_threshold: float = 0.15
    stagnation_steps: int = Field(default=50, ge=0)
    frontier_stagnation_seconds: float = Field(default=5.0, ge=0)
    terminate_on_collision: bool = True
    forward_scale: float = 0.0
    backward_speed_penalty_scale: float = Field(default=1.0, ge=0)
    route_progress_scale: float = Field(default=10.0, ge=0)
    collision_penalty: float = -100.0
    slip_penalty: float = 0.2
    steer_jerk_penalty: float = 0.05
    lap_time_reward_scale: float = Field(default=1000.0, ge=0)

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
    laps_per_episode: int = Field(default=10, ge=0)

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
            "--stop-after-laps",
            str(self.stop_after_laps),
            "--plateau-min-timesteps",
            str(self.plateau_min_timesteps),
            "--plateau-window-timesteps",
            str(self.plateau_window_timesteps),
            "--plateau-patience",
            str(self.plateau_patience),
            "--plateau-min-improvement-pct",
            str(self.plateau_min_improvement_pct),
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
            "--max-episode-steps",
            str(self.max_episode_steps),
            "--stagnation-speed-threshold",
            str(self.stagnation_speed_threshold),
            "--stagnation-steps",
            str(self.stagnation_steps),
            "--map-id",
            self.map_id,
            "--laps-per-episode",
            str(self.laps_per_episode),
            "--frontier-stagnation-seconds",
            str(self.frontier_stagnation_seconds),
            "--terminate-on-collision" if self.terminate_on_collision else "--no-terminate-on-collision",
            "--forward-scale",
            str(self.forward_scale),
            "--backward-speed-penalty-scale",
            str(self.backward_speed_penalty_scale),
            "--route-progress-scale",
            str(self.route_progress_scale),
            "--collision-penalty",
            str(self.collision_penalty),
            "--slip-penalty",
            str(self.slip_penalty),
            "--steer-jerk-penalty",
            str(self.steer_jerk_penalty),
            "--lap-time-reward-scale",
            str(self.lap_time_reward_scale),
        ]
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
