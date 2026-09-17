"""Metrics + leaderboard helpers (contracts.md §6–7).

Train/eval write rows; ``compare_models`` ranks by ``adjusted_time``.
Also owns atomic SB3 zip save + ``train.lock`` acquire/release used by ``train_ppo``.
"""

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
    """Save SB3 zip via temp stem then rename so incomplete never looks like best.

    Never deletes the previous ``final`` unless a complete replacement is already
    on disk (or safely moved to ``.bak``). A failed Windows lock / replace must
    leave the prior zip intact — do not ``unlink(final)`` as a fallback.
    """
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
    if bak.exists():
        try:
            bak.unlink()
        except OSError:
            pass
    moved_aside = False
    if final.exists():
        try:
            final.replace(bak)
            moved_aside = True
        except OSError as move_exc:
            # Cannot move the live zip (Windows share lock). Try replace-in-place;
            # if that also fails, keep ``final`` and drop the temp — never unlink.
            try:
                produced.replace(final)
            except OSError as replace_exc:
                try:
                    produced.unlink(missing_ok=True)
                except OSError:
                    pass
                raise OSError(
                    f"atomic_save_sb3: could not replace locked {final}: {replace_exc}"
                ) from move_exc
            return final
    try:
        produced.replace(final)
    except OSError:
        # Restore prior weights if we had moved them aside.
        if moved_aside and bak.exists() and not final.exists():
            try:
                bak.replace(final)
            except OSError:
                pass
        try:
            if produced.exists():
                produced.unlink()
        except OSError:
            pass
        raise
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


def write_official_per_map_sidecar(
    metrics: dict,
    *,
    logs_root: Path | None = None,
) -> Path | None:
    """Persist per-map official breakdown beside the CSV row (FTGΔ localization).

    Leaderboard CSV stays aggregate; this sidecar carries ``per_map`` from
    ``eval_protocol`` so the next official re-eval can show where PPO loses
    seconds vs FTG without re-parsing logs. No-op for non-official / missing
    ``per_map`` (does not invent data for prior rows).
    """
    if str(metrics.get("kind") or "") != "official":
        return None
    per_map = metrics.get("per_map")
    if not isinstance(per_map, dict) or not per_map:
        return None
    root = logs_root or (Path(__file__).resolve().parent / "logs")
    root.mkdir(parents=True, exist_ok=True)
    rid = str(metrics.get("run_id") or "unknown")
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in rid)[:120]
    out = root / f"official_per_map_{safe}.json"
    payload = {
        "run_id": rid,
        "policy": metrics.get("policy"),
        "protocol_id": metrics.get("protocol_id"),
        "kind": "official",
        "adjusted_time": metrics.get("adjusted_time"),
        "mean_lap_time": metrics.get("mean_lap_time"),
        "total_collisions": metrics.get("total_collisions"),
        "n_episodes": metrics.get("n_episodes"),
        "mean_progress_frac": metrics.get("mean_progress_frac"),
        "dnf": metrics.get("dnf"),
        "tracks_eval": metrics.get("tracks_eval"),
        "timestamp": metrics.get("timestamp"),
        "per_map": per_map,
    }
    atomic_write_json(out, payload)
    return out


def append_leaderboard(path: Path, metrics: dict) -> None:
    """Append a row. Prefer kind=official for race claims; smoke rows stay tagged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Side-channel for FTGΔ: keep per-map times when protocol eval provides them.
    try:
        write_official_per_map_sidecar(metrics)
    except OSError:
        pass
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


def _zip_looks_complete(path: Path) -> bool:
    """True if ``path`` is a non-tiny zip that opens and passes ``testzip``."""
    import zipfile

    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size <= 1024:
            return False
    except OSError:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def find_last_complete_checkpoint(run_dir: Path) -> Path | None:
    """Prefer last complete checkpoint zip; else latest_model; else best_model.

    Also accepts ``*.zip.bak`` left by a failed ``atomic_save_sb3`` replace so
    Continue can recover instead of claiming no checkpoint. Skips zips that
    fail a zipfile open/test (partial SB3 writes).
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        zips = sorted(
            [
                p
                for p in ckpt_dir.glob("*.zip")
                if not p.name.startswith(".") and _zip_looks_complete(p)
            ],
            key=lambda p: p.stat().st_mtime,
        )
        if zips:
            return zips[-1]
    for name in (
        "latest_model.zip",
        "best_model.zip",
        "latest_model.zip.bak",
        "best_model.zip.bak",
    ):
        p = run_dir / name
        if _zip_looks_complete(p):
            return p
    return None


def run_lock_path(run_dir: Path) -> Path:
    return Path(run_dir) / "train.lock"


def _pid_alive(pid: int) -> bool:
    """Best-effort: True if ``pid`` appears to refer to a live process."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            k = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                k.CloseHandle(handle)
                return True
            return False
        os.kill(int(pid), 0)
        return True
    except (OSError, AttributeError):
        return False


def acquire_run_lock(run_dir: Path, pid: int | None = None) -> Path:
    """Refuse dual writers: exclusive ``train.lock`` create (O_EXCL) with pid.

    Stale locks (dead PID) are removed once, then create is retried. Two live
    trainers racing the same ``run_id`` cannot both win the exclusive create.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_lock_path(run_dir)
    pid = int(pid or os.getpid())
    payload = json.dumps({"pid": pid, "timestamp": utc_now()}, indent=2)

    def _try_excl_create() -> bool:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY  # type: ignore[attr-defined]
        try:
            fd = os.open(str(lock), flags, 0o644)
        except FileExistsError:
            return False
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        return True

    if _try_excl_create():
        return lock

    # Lock exists — refuse if holder looks alive; else clear stale once and retry.
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
        old_pid = int(data.get("pid", -1))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        old_pid = -1
    if _pid_alive(old_pid):
        raise RuntimeError(
            f"Refuse dual-writer: {lock} held by pid={old_pid}. Stop that train first."
        )
    try:
        lock.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Refuse dual-writer: {lock} exists but could not clear stale lock: {exc}"
        ) from exc
    if _try_excl_create():
        return lock
    # Lost the race to another trainer that created between unlink and create.
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
        winner = int(data.get("pid", -1))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        winner = -1
    raise RuntimeError(
        f"Refuse dual-writer: {lock} taken by pid={winner} during acquire. "
        "Stop that train first."
    )


def release_run_lock(run_dir: Path) -> None:
    lock = run_lock_path(run_dir)
    try:
        if lock.exists():
            lock.unlink()
    except OSError:
        pass
