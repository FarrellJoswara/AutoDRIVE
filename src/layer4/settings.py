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

    # Train / job
    n_envs: int = Field(default=1, ge=1)
    base_port: int = Field(default=4567, ge=1)
    timesteps: int = Field(default=10_000, ge=1)
    out: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    resume: Optional[str] = None

    # Env kwargs
    headless: bool = True
    auto_launch: bool = True
    connect_timeout: float = Field(default=90.0, gt=0)
    frame_skip: int = Field(default=1, ge=1)
    max_episode_steps: int = Field(default=0, ge=0)
    stagnation_speed_threshold: float = 0.15
    stagnation_steps: int = Field(default=200, ge=0)
    forward_scale: float = 1.0
    collision_penalty: float = 0.0
    slip_penalty: float = 0.0
    steer_jerk_penalty: float = 0.0

    # Hub-only (not train argv)
    telemetry_every_n: int = Field(default=200, ge=1)
    fleet_hz: float = Field(default=15.0, gt=0)
    lidar_display_beams: int = Field(default=120, ge=1)
    telemetry_lidar_max_envs: int = Field(default=4, ge=0)
    docker_mode: bool = False
    stop_sims_on_train_exit: bool = True
    stop_stack_on_train_exit: bool = False

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
            "--base-port",
            str(self.base_port),
            "--timesteps",
            str(self.timesteps),
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
            "--forward-scale",
            str(self.forward_scale),
            "--collision-penalty",
            str(self.collision_penalty),
            "--slip-penalty",
            str(self.slip_penalty),
            "--steer-jerk-penalty",
            str(self.steer_jerk_penalty),
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
