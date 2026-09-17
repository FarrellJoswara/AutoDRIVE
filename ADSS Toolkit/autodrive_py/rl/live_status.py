"""Cheap train-side live status trail (JSON only — no OpenCV / rendering)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_live_status(path: Path, payload: dict[str, Any]) -> None:
    """Atomic-ish JSON write so ``watch --follow`` never reads a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_live_status(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_latest_status(runs_root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Pick the newest ``live_status.json`` under ``rl/runs/``."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return None
    best: tuple[float, Path, dict[str, Any]] | None = None
    for status_path in runs_root.glob("*/live_status.json"):
        data = read_live_status(status_path)
        if not data:
            continue
        mtime = status_path.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, status_path, data)
    if best is None:
        return None
    return best[1], best[2]


def resolve_latest_weights(run_models_dir: Path, status: dict[str, Any] | None = None) -> Path | None:
    """Prefer explicit path / latest_model.zip, else newest checkpoint, else best_model."""
    run_models_dir = Path(run_models_dir)
    if status:
        hint = status.get("latest_model") or status.get("checkpoint")
        if hint:
            p = Path(str(hint))
            if p.is_file():
                return p
            if p.with_suffix(".zip").is_file():
                return p.with_suffix(".zip")
    latest = run_models_dir / "latest_model.zip"
    if latest.is_file():
        return latest
    ckpt_dir = run_models_dir / "checkpoints"
    if ckpt_dir.is_dir():
        zips = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if zips:
            return zips[-1]
    best = run_models_dir / "best_model.zip"
    if best.is_file():
        return best
    return None


def LiveStatusCallback(*args, **kwargs):
    """Factory so helper imports do not require SB3 until train constructs the callback."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _LiveStatusCallback(BaseCallback):
        """
        Write a small status snapshot every N PPO rollouts.

        Intentionally cheap: JSON + optional ``model.save`` only. Never renders.
        """

        def __init__(
            self,
            status_path: Path,
            run_id: str,
            *,
            every_rollouts: int = 1,
            latest_model_path: Path | None = None,
            save_latest_every_rollouts: int = 1,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.status_path = Path(status_path)
            self.run_id = run_id
            self.every_rollouts = max(1, int(every_rollouts))
            self.latest_model_path = Path(latest_model_path) if latest_model_path else None
            self.save_latest_every_rollouts = max(1, int(save_latest_every_rollouts))
            self._rollouts = 0
            self._ep_count = 0
            self._collision_eps = 0
            self._last_latest_path: str | None = None

        def _on_training_start(self) -> None:
            self._write(force_save_latest=False)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []) or []:
                ep = info.get("episode")
                if ep is None:
                    continue
                self._ep_count += 1
                if ep.get("collision"):
                    self._collision_eps += 1
            return True

        def _on_rollout_end(self) -> None:
            self._rollouts += 1
            if self._rollouts % self.every_rollouts == 0:
                save_latest = (
                    self.latest_model_path is not None
                    and self._rollouts % self.save_latest_every_rollouts == 0
                )
                self._write(force_save_latest=save_latest)

        def _on_training_end(self) -> None:
            self._write(force_save_latest=self.latest_model_path is not None)

        def _ep_rew_mean(self) -> float | None:
            buf = getattr(self.model, "ep_info_buffer", None)
            if not buf:
                return None
            rewards = [float(x["r"]) for x in buf if "r" in x]
            if not rewards:
                return None
            return float(sum(rewards) / len(rewards))

        def _write(self, *, force_save_latest: bool) -> None:
            latest_str = self._last_latest_path
            if force_save_latest and self.latest_model_path is not None:
                self.latest_model_path.parent.mkdir(parents=True, exist_ok=True)
                dest = self.latest_model_path
                save_stem = dest.with_suffix("") if dest.suffix == ".zip" else dest
                self.model.save(str(save_stem))
                latest_str = str(save_stem.with_suffix(".zip"))
                self._last_latest_path = latest_str

            payload = {
                "run_id": self.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unix_time": time.time(),
                "timesteps": int(self.num_timesteps),
                "rollouts": int(self._rollouts),
                "ep_rew_mean": self._ep_rew_mean(),
                "episode_count_estimate": int(self._ep_count),
                "collisions_estimate": int(self._collision_eps),
                "latest_model": latest_str,
            }
            write_live_status(self.status_path, payload)

    return _LiveStatusCallback(*args, **kwargs)
