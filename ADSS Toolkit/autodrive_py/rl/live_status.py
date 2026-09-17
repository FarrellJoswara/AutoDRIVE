"""Cheap train-side live status trail (JSON only — no OpenCV / rendering)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Windows: control_ui / watch / AV can briefly lock live_status.json so
# os.replace(tmp, dest) raises WinError 5 (access denied) or 32 (sharing).
_WIN_LOCK_ERRORS = frozenset({5, 32})


def write_live_status(path: Path, payload: dict[str, Any]) -> None:
    """Atomic-ish JSON write so ``watch --follow`` never reads a half file.

    Retries replace on Windows lock races; falls back to in-place overwrite
    rather than raising (a status trail must never kill a PPO run).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp.write_text(text, encoding="utf-8")

    last_err: OSError | None = None
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except OSError as exc:
            last_err = exc
            winerr = getattr(exc, "winerror", None)
            transient = (
                winerr in _WIN_LOCK_ERRORS
                or isinstance(exc, PermissionError)
                or getattr(exc, "errno", None) in (11, 13, 16)  # EAGAIN / EACCES / EBUSY
            )
            if not transient:
                raise
            time.sleep(0.02 * (attempt + 1))

    # Last resort: non-atomic overwrite so train keeps going.
    try:
        path.write_text(text, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return
    except OSError as exc:
        last_err = last_err or exc

    # Still locked: drop the trail update rather than abort PPO.
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"[live_status] write skipped (non-fatal): {last_err}")


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
        Write a small status snapshot often enough for the control UI.

        - JSON every ``every_steps`` env timesteps (cheap; no model save)
        - JSON every ``every_rollouts`` PPO rollouts (plus optional latest.zip)

        Intentionally cheap: never renders.
        """

        def __init__(
            self,
            status_path: Path,
            run_id: str,
            *,
            every_rollouts: int = 1,
            every_steps: int = 512,
            latest_model_path: Path | None = None,
            save_latest_every_rollouts: int = 1,
            n_envs: int = 1,
            vec_env_active: str = "dummy",
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self.status_path = Path(status_path)
            self.run_id = run_id
            self.every_rollouts = max(1, int(every_rollouts))
            self.every_steps = max(0, int(every_steps))  # 0 = only on rollouts
            self.latest_model_path = Path(latest_model_path) if latest_model_path else None
            self.save_latest_every_rollouts = max(1, int(save_latest_every_rollouts))
            self.n_envs = max(1, int(n_envs))
            self.vec_env_active = str(vec_env_active or "dummy")
            self._rollouts = 0
            self._ep_count = 0
            self._collision_eps = 0
            self._last_latest_path: str | None = None
            self._last_status_ts: int = -10**9
            self._t0 = time.time()
            self._ts0 = 0

        def _on_training_start(self) -> None:
            self._t0 = time.time()
            self._ts0 = int(self.num_timesteps)
            self._write(force_save_latest=False)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []) or []:
                ep = info.get("episode")
                if ep is None:
                    continue
                self._ep_count += 1
                if ep.get("collision"):
                    self._collision_eps += 1
            # Mid-rollout trail so the UI timesteps tick without waiting for a full PPO buffer.
            if self.every_steps > 0:
                ts = int(self.num_timesteps)
                if ts - self._last_status_ts >= self.every_steps:
                    # One early weight dump so watch --follow need not wait for rollout 1
                    # (with n_envs=14 that can be ~28k steps of empty cars).
                    early = (
                        self.latest_model_path is not None
                        and self._last_latest_path is None
                        and ts >= max(2048, self.every_steps)
                    )
                    self._write(force_save_latest=early)
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
            # Do not clobber RaceBestModelCallback's early_stopped / validating phase.
            self._write(
                force_save_latest=self.latest_model_path is not None,
                preserve_terminal_phase=True,
            )

        def _ep_rew_mean(self) -> float | None:
            buf = getattr(self.model, "ep_info_buffer", None)
            if not buf:
                return None
            rewards = [float(x["r"]) for x in buf if "r" in x]
            if not rewards:
                return None
            return float(sum(rewards) / len(rewards))

        def _write(
            self,
            *,
            force_save_latest: bool,
            preserve_terminal_phase: bool = False,
        ) -> None:
            latest_str = self._last_latest_path
            if force_save_latest and self.latest_model_path is not None:
                from .metrics_io import atomic_save_sb3

                self.latest_model_path.parent.mkdir(parents=True, exist_ok=True)
                dest = self.latest_model_path
                final = atomic_save_sb3(self.model, dest)
                latest_str = str(final)
                self._last_latest_path = latest_str

            n_updates = getattr(self.model, "_n_updates", None)
            try:
                n_updates = int(n_updates) if n_updates is not None else None
            except (TypeError, ValueError):
                n_updates = None

            elapsed = max(time.time() - self._t0, 1e-6)
            ts = int(self.num_timesteps)
            steps_per_sec = float(ts - self._ts0) / elapsed
            crash_rate = (
                float(self._collision_eps) / float(self._ep_count) if self._ep_count > 0 else None
            )

            phase = "learning"
            msg = "learning"
            if preserve_terminal_phase:
                prev = read_live_status(self.status_path)
                if prev and prev.get("phase") in (
                    "early_stopped",
                    "early-stop",
                    "validating",
                ):
                    phase = str(prev["phase"])
                    if prev.get("msg"):
                        msg = str(prev["msg"])

            payload = {
                "run_id": self.run_id,
                "phase": phase,
                "msg": msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "unix_time": time.time(),
                "timesteps": ts,
                "rollouts": int(self._rollouts),
                # PPO update count when available (else watch falls back to rollouts).
                "n_updates": n_updates,
                "ep_rew_mean": self._ep_rew_mean(),
                "episode_count_estimate": int(self._ep_count),
                "collisions_estimate": int(self._collision_eps),
                "crash_rate_estimate": crash_rate,
                "steps_per_sec": steps_per_sec,
                "latest_model": latest_str,
                "n_envs": int(self.n_envs),
                "vec_env_active": self.vec_env_active,
                # Reminder: reward/env edits need Stop+Start train to reload worker code.
                "reward_shaping": "forward_s_only+stall_timeout; restart train after env changes",
            }
            try:
                write_live_status(self.status_path, payload)
            except OSError as exc:
                # Status I/O must never abort PPO (Windows file locks from UI).
                if self.verbose:
                    print(f"[live_status] write skipped (non-fatal): {exc}")
                return
            self._last_status_ts = ts

    return _LiveStatusCallback(*args, **kwargs)
