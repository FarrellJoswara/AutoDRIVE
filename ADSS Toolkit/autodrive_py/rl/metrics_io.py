"""Metrics + leaderboard helpers (contracts.md §6–7)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import CONTRACTS_VERSION, SCORING_REV

LEADERBOARD_HEADER = [
    "run_id",
    "policy",
    "backend",
    "adjusted_time",
    "mean_lap_time",
    "total_collisions",
    "n_episodes",
    "timestamp",
    "contracts_version",
    "kind",
    "protocol_id",
    "dnf",
    "mean_progress_frac",
    "scoring_rev",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_metrics(
    *,
    run_id: str,
    policy: str,
    backend: str,
    mean_lap_time: float | None,
    total_collisions: int,
    n_episodes: int,
    tracks_eval: list[str],
    kind: str = "smoke",
    **extra: Any,
) -> dict:
    adjusted = None
    if mean_lap_time is not None:
        adjusted = float(mean_lap_time) + 10.0 * int(total_collisions)
    out = {
        "contracts_version": CONTRACTS_VERSION,
        "scoring_rev": SCORING_REV,
        "run_id": run_id,
        "policy": policy,
        "backend": backend,
        "mean_lap_time": mean_lap_time,
        "total_collisions": int(total_collisions),
        "adjusted_time": adjusted,
        "n_episodes": int(n_episodes),
        "tracks_eval": list(tracks_eval),
        "timestamp": utc_now(),
        "kind": str(kind or "smoke"),
    }
    out.update(extra)
    return out


def git_sha_short(cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def config_fingerprint(config: dict) -> str:
    """Stable short hash of the training fingerprint fields."""
    keys = (
        "contracts_version",
        "obs_dim",
        "n_lidar",
        "map",
        "maps",
        "map_hashes",
        "seed",
        "device",
        "n_envs",
        "vec_env",
        "n_steps",
        "batch_size",
        "n_epochs",
        "net_arch",
        "timesteps",
        "spawn_jitter",
        "collision_first",
        "speed_gate",
        "ttc_truncate",
        "lidar_dr",
        "git_sha",
        "map_pack",
        "validation_map",
    )
    payload = {k: config.get(k) for k in keys if k in config}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def atomic_write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_save_sb3(model, dest: Path) -> Path:
    """Save SB3 zip via temp stem then rename so incomplete never looks like best."""
    dest = Path(dest)
    if dest.suffix == ".zip":
        final = dest
    else:
        final = Path(str(dest) + ".zip")
    final.parent.mkdir(parents=True, exist_ok=True)
    # SB3 appends .zip only when the path has no suffix — avoid dots in temp stem.
    tmp_stem = final.parent / f"{final.stem}_tmpsave"
    tmp_zip = Path(str(tmp_stem) + ".zip")
    for p in (tmp_zip, tmp_stem):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    model.save(str(tmp_stem))
    produced = tmp_zip if tmp_zip.is_file() else (tmp_stem if tmp_stem.is_file() else None)
    if produced is None:
        raise OSError(f"SB3 save did not produce {tmp_zip}")
    bak = final.with_suffix(final.suffix + ".bak")
    if final.exists():
        try:
            if bak.exists():
                bak.unlink()
            final.replace(bak)
        except OSError:
            try:
                final.unlink()
            except OSError:
                pass
    produced.replace(final)
    if bak.exists():
        try:
            bak.unlink()
        except OSError:
            pass
    return final


def write_run_artifacts(
    models_root: Path,
    run_id: str,
    config: dict,
    metrics: dict,
    train_tracks: list[str] | None = None,
    *,
    append_board: bool = True,
) -> Path:
    run_dir = models_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    if "fingerprint" not in cfg:
        cfg["fingerprint"] = config_fingerprint(cfg)
    atomic_write_json(run_dir / "config.json", cfg)
    atomic_write_json(run_dir / "metrics.json", metrics)
    tracks = train_tracks or metrics.get("tracks_eval") or []
    (run_dir / "train_tracks.txt").write_text(
        "\n".join(str(t) for t in tracks) + ("\n" if tracks else ""),
        encoding="utf-8",
    )
    if append_board:
        append_leaderboard(models_root / "leaderboard.csv", metrics)
    return run_dir


def append_leaderboard(path: Path, metrics: dict) -> None:
    """Append a row. Prefer kind=official for race claims; smoke rows stay tagged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Migrate header if needed
    # Header migration must preserve every row verbatim, including legacy ones.
    existing = read_leaderboard(
        path,
        hide_null=False,
        official_only=False,
        contracts_only=False,
        current_scoring_only=False,
    )
    new_file = not path.exists()
    needs_rewrite = False
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header and any(col not in header for col in LEADERBOARD_HEADER):
            needs_rewrite = True

    row = {
        "run_id": metrics.get("run_id"),
        "policy": metrics.get("policy"),
        "backend": metrics.get("backend"),
        "adjusted_time": metrics.get("adjusted_time"),
        "mean_lap_time": metrics.get("mean_lap_time"),
        "total_collisions": metrics.get("total_collisions"),
        "n_episodes": metrics.get("n_episodes"),
        "timestamp": metrics.get("timestamp"),
        "contracts_version": metrics.get("contracts_version", CONTRACTS_VERSION),
        "kind": metrics.get("kind", "smoke"),
        "protocol_id": metrics.get("protocol_id", ""),
        "dnf": metrics.get("dnf", metrics.get("mean_lap_time") is None),
        "mean_progress_frac": metrics.get("mean_progress_frac"),
        "scoring_rev": metrics.get("scoring_rev", SCORING_REV),
    }

    if needs_rewrite:
        # Rewrite with extended header, preserving old rows (kind blank → smoke)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER, extrasaction="ignore")
            w.writeheader()
            for old in existing:
                old.setdefault("kind", "smoke")
                old.setdefault("protocol_id", "")
                # Pre-migration rows predate the lap-gate fix; stamp them rev 1.
                if not str(old.get("scoring_rev") or "").strip():
                    old["scoring_rev"] = 1
                w.writerow({k: old.get(k, "") for k in LEADERBOARD_HEADER})
            w.writerow(row)
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def read_leaderboard(
    path: Path,
    *,
    hide_null: bool = True,
    official_only: bool = False,
    contracts_only: bool = True,
    current_scoring_only: bool = True,
) -> list[dict]:
    """Read leaderboard rows, dropping anything not comparable by default.

    ``current_scoring_only`` hides rows written under an older ``scoring_rev``.
    Rev 1 rows can carry laps that were never driven, so they must never be
    ranked against rev 2 rows.
    """
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict] = []
    for r in rows:
        if contracts_only:
            ver = str(r.get("contracts_version") or "")
            if ver and ver != CONTRACTS_VERSION:
                continue
        if current_scoring_only:
            try:
                rev = int(str(r.get("scoring_rev") or "1").strip() or 1)
            except ValueError:
                rev = 1
            if rev != SCORING_REV:
                continue
        kind = str(r.get("kind") or "smoke").lower()
        if official_only and kind != "official":
            continue
        if hide_null:
            adj = r.get("adjusted_time")
            if adj is None or str(adj).strip() in ("", "None", "null"):
                continue
        out.append(r)
    return out


def find_last_complete_checkpoint(run_dir: Path) -> Path | None:
    """Prefer last complete checkpoint zip; else latest_model; else best_model."""
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        zips = sorted(
            [p for p in ckpt_dir.glob("*.zip") if p.stat().st_size > 1024],
            key=lambda p: p.stat().st_mtime,
        )
        # Skip obvious partials
        zips = [p for p in zips if not p.name.startswith(".")]
        if zips:
            return zips[-1]
    for name in ("latest_model.zip", "best_model.zip"):
        p = run_dir / name
        if p.is_file() and p.stat().st_size > 1024:
            return p
    return None


def run_lock_path(run_dir: Path) -> Path:
    return Path(run_dir) / "train.lock"


def acquire_run_lock(run_dir: Path, pid: int | None = None) -> Path:
    """Refuse dual writers: create exclusive train.lock with pid."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_lock_path(run_dir)
    pid = int(pid or os.getpid())
    if lock.exists():
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
            old_pid = int(data.get("pid", -1))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            old_pid = -1
        # Stale if process gone
        alive = False
        if old_pid > 0:
            try:
                if os.name == "nt":
                    import ctypes

                    k = ctypes.windll.kernel32  # type: ignore[attr-defined]
                    handle = k.OpenProcess(0x1000, False, old_pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                    if handle:
                        alive = True
                        k.CloseHandle(handle)
                else:
                    os.kill(old_pid, 0)
                    alive = True
            except (OSError, AttributeError):
                alive = False
        if alive:
            raise RuntimeError(
                f"Refuse dual-writer: {lock} held by pid={old_pid}. Stop that train first."
            )
    atomic_write_json(lock, {"pid": pid, "timestamp": utc_now()})
    return lock


def release_run_lock(run_dir: Path) -> None:
    lock = run_lock_path(run_dir)
    try:
        if lock.exists():
            lock.unlink()
    except OSError:
        pass
