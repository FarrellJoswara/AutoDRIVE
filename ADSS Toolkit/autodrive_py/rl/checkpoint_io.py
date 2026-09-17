"""Shared checkpoint / artifact IO for train + operator siblings.

Public API
----------
atomic_save_sb3(model, dest: Path) -> Path
    Save SB3 zip via temp stem then rename (incomplete never looks like best).

find_last_complete_checkpoint(run_dir: Path) -> Path | None
    Prefer last complete checkpoints/*.zip, else latest_model, else best_model.

acquire_run_lock(run_dir: Path, pid: int | None = None) -> Path
release_run_lock(run_dir: Path) -> None
    Refuse dual writers on the same run_id (train.lock).

config_fingerprint(config: dict) -> str
git_sha_short(cwd: Path | None = None) -> str | None
"""

from __future__ import annotations

from .metrics_io import (
    acquire_run_lock,
    atomic_save_sb3,
    config_fingerprint,
    find_last_complete_checkpoint,
    git_sha_short,
    release_run_lock,
    run_lock_path,
)

__all__ = [
    "acquire_run_lock",
    "atomic_save_sb3",
    "config_fingerprint",
    "find_last_complete_checkpoint",
    "git_sha_short",
    "release_run_lock",
    "run_lock_path",
]
