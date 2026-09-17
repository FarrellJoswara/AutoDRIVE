"""Operator helpers for control_ui (testable without Gradio/browser).

Locks, presets, Continue argv, banner classification, map labels, coach hints.
Multi-run inventory: :func:`find_live_run_locks` + :func:`lock_owner`.
Status-file ranking across runs lives in ``live_status.pick_status_for_operator``
(Control UI still ranks live PIDs via its own pin / timesteps scorer).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .compat import ContractsMismatch, refuse_config_contracts
from .eval_protocol import is_holdout_map, load_protocol, row_race_score_key, sealed_map_ids
from .metrics_io import find_last_complete_checkpoint, read_leaderboard, run_lock_path


PRESETS = {
    "debug": {"timesteps": 4096, "n_envs": 1, "label": "Debug (1 env)"},
    "quick": {"timesteps": 50_000, "n_envs": 8, "label": "Quick (50k)"},
    # Overnight: budget OFF + early-stop with warmup so first validations cannot abort.
    "overnight": {
        "timesteps": 500_000,
        "n_envs": 8,
        "label": "Overnight (early-stop)",
        "stop_on_budget": False,
        "early_stop_patience": 5,
        "early_stop_min_improve": 0.5,
        "race_eval_every": 50_000,
        "early_stop_warmup_evals": 2,
        "early_stop_min_timesteps": 100_000,
        "select_timeout": 220,
    },
}

# Hard ceiling when "Stop on timesteps budget" is OFF (early-stop is the real exit).
UNLIMITED_TIMESTEPS_SAFETY = 50_000_000
# Nominal budget for auto --race-eval-every when timesteps are unlimited (~every 10%).
# 500k → auto every 50k (was 100k→10k / 250k→25k; overnight needs sparse evals).
UNLIMITED_EVAL_REFERENCE_TS = 500_000
# Floor for auto race-eval interval (timesteps). Higher = fewer mid-train pauses.
AUTO_RACE_EVAL_EVERY_FLOOR = 50_000
# Mid-train validation sim-time budget (s). Official protocol still uses 400s.
# >=180–220 so a real lap (~180–210s) can finish; 60s forced permanent DNF.
MID_TRAIN_SELECT_TIMEOUT_S = 220.0
# Overnight-safe early-stop defaults (unlimited + patience).
DEFAULT_EARLY_STOP_PATIENCE_UNLIMITED = 5
DEFAULT_EARLY_STOP_WARMUP_EVALS = 2
DEFAULT_EARLY_STOP_MIN_TIMESTEPS = 100_000

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RESERVED_MAP_PREFIXES = {"map"}  # protocol namespace: map0/map1/map2
MIN_COMPLETE_ZIP_BYTES = 1024


def read_operator_run_pin(logs_dir: Path) -> str | None:
    """Read ``logs/CURRENT_RUN.txt`` (one run_id line) or None if missing/blank.

    Control UI and Watch use this pin so a short A/B smoke does not eclipse
    overnight when several trains are live.
    """
    pin = Path(logs_dir) / "CURRENT_RUN.txt"
    try:
        text = pin.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def write_operator_run_pin(logs_dir: Path, run_id: str) -> Path:
    """Write operator pin; returns the pin path."""
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "CURRENT_RUN.txt"
    path.write_text(str(run_id).strip() + "\n", encoding="utf-8")
    return path


def clear_operator_run_pin(logs_dir: Path) -> bool:
    """Remove Focus pin (``CURRENT_RUN.txt``). Returns True if a pin was cleared."""
    pin = Path(logs_dir) / "CURRENT_RUN.txt"
    try:
        if not pin.is_file():
            return False
        pin.unlink()
        return True
    except OSError:
        try:
            pin.write_text("\n", encoding="utf-8")
            return True
        except OSError:
            return False


def resolve_stop_budget(
    *,
    stop_on_budget: bool = True,
    timesteps: int,
    early_stop_patience: int = 0,
) -> dict[str, Any]:
    """Resolve timesteps ceiling vs early-stop-only mode for Start/Continue/preview.

    When ``stop_on_budget`` is False, training uses ``UNLIMITED_TIMESTEPS_SAFETY``
    as a hard abort only; early-stop patience must be > 0 or Start is refused.
    """
    patience = max(0, int(early_stop_patience))
    ts = max(1, int(timesteps))
    if stop_on_budget:
        return {
            "ok": True,
            "stop_on_budget": True,
            "unlimited": False,
            "effective_timesteps": ts,
            "patience": patience,
            "budget_label": f"budget: {ts:,}",
            "error": None,
        }
    if patience <= 0:
        return {
            "ok": False,
            "stop_on_budget": False,
            "unlimited": True,
            "effective_timesteps": UNLIMITED_TIMESTEPS_SAFETY,
            "patience": patience,
            "budget_label": "budget: off (early-stop only)",
            "error": (
                "Refuse: Stop on timesteps budget is OFF but early-stop "
                "patience is 0 - training would never stop except manual Stop. "
                "Set patience to 3-5 (or >=1), or turn budget stop ON."
            ),
        }
    return {
        "ok": True,
        "stop_on_budget": False,
        "unlimited": True,
        "effective_timesteps": UNLIMITED_TIMESTEPS_SAFETY,
        "patience": patience,
        "budget_label": "budget: off (early-stop only)",
        "error": None,
    }


def _safe_id(value: str) -> bool:
    return bool(SAFE_ID.match(str(value or ""))) and ".." not in str(value)


def safe_run_id(value: str) -> bool:
    """Public alias: True when ``value`` is a safe models/<run_id> directory name."""
    return _safe_id(value)


STILL_ACTIVE = 259


def _pid_alive(pid: int) -> bool:
    """True only for a genuinely running process.

    On Windows a killed process stays openable while any handle survives, so
    ``OpenProcess`` alone reports zombies as alive — that false positive is what
    makes a freshly stopped run look permanently locked. Ask for the exit code.
    """
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            k = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = k.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                k.CloseHandle(handle)
        os.kill(int(pid), 0)
        return True
    except (OSError, AttributeError, ValueError):
        return False


def start_preview(
    map_id: str,
    timesteps: int,
    n_envs: int,
    run_id: str,
    models_dir: Path,
    maps_dir: Path | None = None,
    *,
    stop_on_budget: bool = True,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 0.5,
    race_eval_every: int = 0,
    collision_first: bool = False,
    speed_gate: bool = False,
) -> dict:
    """What Start will do — shown before/with Start."""
    vec = "subproc" if int(n_envs) > 1 else "dummy"
    info = map_role(map_id, maps_dir)
    blocked = not info["train_safe"]
    tag = " [HOLDOUT]" if info["sealed"] else (" [VALIDATION PIN]" if blocked else "")
    budget = resolve_stop_budget(
        stop_on_budget=stop_on_budget,
        timesteps=timesteps,
        early_stop_patience=early_stop_patience,
    )
    steps_note = (
        budget["budget_label"]
        if budget["unlimited"]
        else f"steps={budget['effective_timesteps']}"
    )
    min_imp = max(0.0, float(early_stop_min_improve))
    eval_every = max(0, int(race_eval_every))
    cf = bool(collision_first)
    sg = bool(speed_gate)
    return {
        "map": map_id,
        "map_role": info["role"],
        "map_label": info["label"],
        "timesteps": int(timesteps),
        "effective_timesteps": budget["effective_timesteps"],
        "stop_on_budget": budget["stop_on_budget"],
        "unlimited": budget["unlimited"],
        "budget_label": budget["budget_label"],
        "early_stop_patience": budget["patience"],
        "early_stop_min_improve": min_imp,
        "race_eval_every": eval_every,
        "budget_error": budget["error"],
        "n_envs": int(n_envs),
        "vec_env": vec,
        "collision_first": cf,
        "speed_gate": sg,
        "run_id": run_id,
        "run_path": str(Path(models_dir) / run_id),
        "holdout_warn": bool(blocked),
        "summary": (
            f"Start -> map={map_id} {steps_note} patience={budget['patience']} "
            f"min_improve={min_imp:g}s eval_every={eval_every or 'auto'} "
            f"n_envs={n_envs} vec={vec} collision_first={cf} speed_gate={sg} run={run_id}"
            + (f"{tag} - refuse unless allowed" if blocked else "")
        ),
    }


def run_curriculum_flags(models_dir: Path, run_id: str) -> dict[str, bool]:
    """Read collision_first / speed_gate from a prior run's config.json (Continue parity)."""
    defaults = {"collision_first": False, "speed_gate": False}
    if not _safe_id(run_id):
        return defaults
    cfg_path = Path(models_dir) / run_id / "config.json"
    if not cfg_path.is_file():
        return defaults
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    return {
        "collision_first": bool(cfg.get("collision_first", False)),
        "speed_gate": bool(cfg.get("speed_gate", False)),
    }


def seal_verify_summary(maps_dir: Path | None = None) -> dict[str, Any]:
    """Operator-facing pack seal check (MUST #5) — never mutates maps."""
    from . import map_pack

    report = map_pack.verify_pack(maps_dir)
    broken_seals = [
        r["id"] for r in report.get("maps", []) if r.get("sealed") and not r.get("ok")
    ]
    bad_maps = [r["id"] for r in report.get("maps", []) if not r.get("ok")]
    ok = bool(report.get("ok"))
    parts = [
        "SEALS OK" if ok and not broken_seals else ("SEAL BROKEN" if broken_seals else "PACK ISSUES"),
        f"maps_ok={sum(1 for r in report.get('maps', []) if r.get('ok'))}/{len(report.get('maps', []))}",
    ]
    if broken_seals:
        parts.append("broken=" + ",".join(broken_seals))
    elif bad_maps:
        parts.append("bad=" + ",".join(bad_maps[:6]))
    pack_issues = list(report.get("issues") or [])
    if pack_issues:
        parts.append("pack: " + "; ".join(pack_issues[:3]))
    return {
        "ok": ok,
        "msg": " | ".join(parts),
        "broken_seals": broken_seals,
        "bad_maps": bad_maps,
        "pack_issues": pack_issues,
        "maps": report.get("maps", []),
        "maps_root": report.get("maps_root"),
    }


def classify_banner(
    *,
    train_alive: bool,
    timesteps: int | None,
    prev_timesteps: int | None,
    status_age_s: float | None,
    saving: bool = False,
    stopping: bool = False,
    exit_code: int | None = None,
    phase: str | None = None,
    msg: str | None = None,
) -> dict[str, str]:
    """Idle / Learning / Validating / Saving / Stopping / EarlyStop / Crashed / Stale."""
    note = (str(msg).strip() if msg else "") or None
    if stopping:
        return {"state": "Stopping", "reason": "Stop requested - killing train tree"}
    if phase == "starting" and not train_alive and exit_code is None:
        return {"state": "Idle", "reason": "trainer spawning - prior live_status cleared"}
    # Early-stop is a clean exit 0 — surface the reason, never "Crashed" / mysterious Idle.
    if phase in ("early_stopped", "early-stop") and not train_alive:
        return {
            "state": "EarlyStop",
            "reason": note or "early-stop: no meaningful improvement",
        }
    if saving:
        return {"state": "Saving", "reason": "Checkpoint / latest_model write"}
    if train_alive and phase == "validating":
        return {
            "state": "Validating",
            "reason": note
            or "scheduled race eval — not hung (timesteps paused; mid-train ≠ official 400s×5)",
        }
    if train_alive:
        if status_age_s is not None and status_age_s > 90:
            return {
                "state": "Stale",
                "reason": f"live_status age {status_age_s:.0f}s while process alive",
            }
        if (
            prev_timesteps is not None
            and timesteps is not None
            and int(timesteps) == int(prev_timesteps)
            and status_age_s is not None
            and status_age_s > 45
        ):
            return {
                "state": "Stale",
                "reason": "timesteps not advancing",
            }
        return {
            "state": "Learning",
            "reason": note or "train_ppo alive",
        }
    if exit_code is not None and exit_code != 0:
        return {"state": "Crashed", "reason": f"exit code {exit_code}"}
    if note and ("early-stop" in note.lower() or "early stop" in note.lower()):
        return {"state": "EarlyStop", "reason": note}
    return {"state": "Idle", "reason": note or "no trainer"}


def map_role(map_id: str, maps_dir: Path | None = None) -> dict[str, Any]:
    """Role of a map for Start decisions: sealed holdout / validation pin / train_ok.

    Prefers the map pack manifest (knows every seal); falls back to the frozen
    eval protocol when the pack is unavailable.
    """
    mid = str(map_id)
    try:
        from . import map_pack

        rows = {r["id"]: r for r in map_pack.list_maps(maps_dir)}
        row = rows.get(mid)
        if row is not None:
            return {
                "id": mid,
                "role": row["role"],
                "sealed": bool(row["sealed"]),
                "train_safe": bool(row["train_safe"]),
                "label": row["label"],
                "source": "map_pack",
            }
    except Exception:
        pass
    try:
        sealed = is_holdout_map(mid)
    except Exception:
        sealed = False
    return {
        "id": mid,
        "role": "holdout" if sealed else "train_ok",
        "sealed": sealed,
        "train_safe": not sealed,
        "label": f"{mid} [HOLDOUT]" if sealed else mid,
        "source": "eval_protocol",
    }


def start_guard(map_id: str, *, allow_holdout: bool = False, maps_dir: Path | None = None) -> tuple[bool, str]:
    """Refuse Start on a sealed holdout / pinned validation map unless overridden."""
    mid = str(map_id or "").strip()
    if not mid:
        return False, (
            "Refuse Start: no track selected (empty map). "
            "Pick a train_ok map in the Track dropdown — blank used to silently become map0."
        )
    info = map_role(mid, maps_dir)
    if info["train_safe"]:
        return True, ""

    what = (
        "a sealed holdout map (official scoring only)"
        if info["sealed"]
        else "the pinned validation map (used to promote best_model - training on it poisons selection)"
    )
    if not allow_holdout:
        return False, (
            f"Refuse Start: '{mid}' is {what}. "
            "Tick 'allow sealed/pinned map' if you really mean to burn it."
        )
    return True, (
        f"WARNING: training on '{mid}' - {what}; official claims involving this map are void."
    )


def clear_stale_lock(models_dir: Path, run_id: str) -> str | None:
    """Drop models/<run_id>/train.lock when nobody holds it (post-kill cleanup).

    Returns a note when a lock was removed, else None. Never touches a live lock.
    """
    owner = lock_owner(models_dir, run_id)
    if owner is None or owner.get("alive"):
        return None
    lock = run_lock_path(Path(models_dir) / run_id)
    try:
        lock.unlink()
    except OSError:
        return None
    return f"cleared stale train.lock (pid={owner.get('pid')})"


def lock_owner(models_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Read models/<run_id>/train.lock — who claims this run_id right now."""
    if not _safe_id(run_id):
        return None
    lock = run_lock_path(Path(models_dir) / run_id)
    if not lock.is_file():
        return None
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pid": None, "alive": False, "stale": True, "timestamp": None}
    try:
        pid = int(data.get("pid", -1))
    except (TypeError, ValueError):
        pid = -1
    alive = _pid_alive(pid)
    return {
        "pid": pid if pid > 0 else None,
        "alive": alive,
        "stale": not alive,
        "timestamp": data.get("timestamp"),
    }


def find_live_run_locks(models_dir: Path) -> list[dict[str, Any]]:
    """All models/*/train.lock entries whose PID is still alive (Start/Continue refuse)."""
    root = Path(models_dir)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in root.iterdir():
        if not d.is_dir() or not _safe_id(d.name):
            continue
        owner = lock_owner(root, d.name)
        if owner and owner.get("alive"):
            out.append({"run_id": d.name, **owner})
    return out


def list_run_models(models_dir: Path) -> list[dict[str, Any]]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    out = []
    for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        best = d / "best_model.zip"
        latest = d / "latest_model.zip"
        ckpt = find_last_complete_checkpoint(d)
        cfg = {}
        cfg_path = d / "config.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cfg = {}
        out.append(
            {
                "run_id": d.name,
                "has_best": best.is_file(),
                "has_latest": latest.is_file(),
                "checkpoint": str(ckpt) if ckpt else None,
                "fingerprint": cfg.get("fingerprint"),
                "contracts_version": cfg.get("contracts_version"),
            }
        )
    return out


def delete_run_model(models_dir: Path, run_id: str, *, active_run_id: str | None = None) -> str:
    """Delete models/<run_id>/ tree. Refuses traversal, the live run, and locked runs."""
    rid = str(run_id).strip()
    if not rid or not _safe_id(rid):
        return f"Refuse delete: bad run_id {run_id!r}"
    if active_run_id and rid == str(active_run_id):
        return f"Refuse delete: {rid} is the run this UI is training. Stop first."
    owner = lock_owner(models_dir, rid)
    if owner and owner.get("alive"):
        return f"Refuse delete: {rid} is locked by live pid={owner.get('pid')}. Stop that train first."
    target = Path(models_dir) / rid
    if not target.is_dir():
        return f"No model dir: {rid}"
    shutil.rmtree(target)
    return f"Deleted models/{rid}"


def infer_map_ids_from_run_id(run_id: str) -> list[str]:
    """Recover ``gen231_0`` from ``…_ppo_gym_gen231_0_hard`` when config.json is missing."""
    rid = str(run_id or "")
    marker = "_ppo_gym_"
    if marker not in rid:
        return []
    rest = rid.split(marker, 1)[1]
    for suffix in ("_hard", "_easy", "_medium"):
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
            break
    return [rest] if rest and _safe_id(rest) else []


def run_map_ids(models_dir: Path, run_id: str) -> list[str]:
    """Map ids a run was trained on (from config.json) so Continue never silently swaps maps."""
    if not _safe_id(run_id):
        return []
    cfg_path = Path(models_dir) / run_id / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = None
        if cfg is not None:
            maps = cfg.get("maps") or ([cfg["map"]] if cfg.get("map") else [])
            out: list[str] = []
            for m in maps:
                stem = Path(str(m)).stem
                if stem and stem not in out:
                    out.append(stem)
            if out:
                return out
    return infer_map_ids_from_run_id(run_id)


def continue_train_argv(
    *,
    run_id: str,
    models_dir: Path,
    timesteps: int,
    n_envs: int,
    map_id: str | None = None,
    stop_on_budget: bool = True,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 0.5,
    race_eval_every: int = 0,
    early_stop_warmup_evals: int | None = None,
    early_stop_min_timesteps: int | None = None,
    select_timeout: float | None = None,
    collision_first: bool | None = None,
    speed_gate: bool | None = None,
) -> tuple[list[str], str]:
    """Build train_ppo argv for Continue (resume last complete ckpt).

    Curriculum flags default from the run's ``config.json`` so Continue does not
    silently drop ``--collision-first`` / ``--speed-gate`` (fingerprint parity).
    Pass explicit bools only to override.
    """
    run_dir = Path(models_dir) / run_id
    ckpt = find_last_complete_checkpoint(run_dir)
    if ckpt is None:
        return [], f"No complete checkpoint under {run_id}"
    budget = resolve_stop_budget(
        stop_on_budget=stop_on_budget,
        timesteps=timesteps,
        early_stop_patience=early_stop_patience,
    )
    if not budget["ok"]:
        return [], budget["error"] or "Invalid stop budget"
    prior = run_curriculum_flags(models_dir, run_id)
    cf = prior["collision_first"] if collision_first is None else bool(collision_first)
    sg = prior["speed_gate"] if speed_gate is None else bool(speed_gate)
    vec = "subproc" if int(n_envs) > 1 else "dummy"
    min_imp = max(0.0, float(early_stop_min_improve))
    eval_every = max(0, int(race_eval_every))
    argv = [
        "--resume",
        run_id,
        "--run_id",
        run_id,
        "--timesteps",
        str(int(budget["effective_timesteps"])),
        "--n-envs",
        str(int(n_envs)),
        "--vec-env",
        vec,
        "--checkpoint-every",
        "25000",
    ]
    if budget["unlimited"]:
        argv.append("--unlimited-timesteps")
    if budget["patience"] > 0:
        argv.extend(["--early-stop-patience", str(int(budget["patience"]))])
        argv.extend(["--early-stop-min-improve", f"{min_imp:g}"])
        if early_stop_warmup_evals is not None:
            argv.extend(["--early-stop-warmup-evals", str(int(early_stop_warmup_evals))])
        if early_stop_min_timesteps is not None:
            argv.extend(["--early-stop-min-timesteps", str(int(early_stop_min_timesteps))])
    if eval_every > 0:
        argv.extend(["--race-eval-every", str(eval_every)])
    if select_timeout is not None and float(select_timeout) > 0:
        argv.extend(["--select-timeout", f"{float(select_timeout):g}"])
    if map_id:
        argv.extend(["--map", str(map_id)])
    if cf:
        argv.append("--collision-first")
    if sg:
        argv.append("--speed-gate")
    steps_note = (
        "unlimited/early-stop"
        if budget["unlimited"]
        else f"+{budget['effective_timesteps']} steps"
    )
    return (
        argv,
        (
            f"Continue {run_id} from {ckpt.name} ({steps_note}, n_envs={n_envs}, "
            f"collision_first={cf}, speed_gate={sg})"
        ),
    )


def map_labels(maps_dir: Path) -> list[dict[str, Any]]:
    """Map picker entries tagged [HOLDOUT] / [VALIDATION] from the map pack."""
    try:
        from . import map_pack

        rows = map_pack.list_maps(maps_dir)
        if rows:
            return [
                {
                    "id": r["id"],
                    "sealed": bool(r["sealed"]),
                    "role": r["role"],
                    "train_safe": bool(r["train_safe"]),
                    "label": r["label"],
                }
                for r in rows
            ]
    except Exception:
        pass

    # Pack unavailable: fall back to protocol seals over whatever is on disk.
    sealed = set()
    try:
        sealed = set(sealed_map_ids(load_protocol()))
    except Exception:
        sealed = set()
    out = []
    if not Path(maps_dir).is_dir():
        return out
    for d in sorted(Path(maps_dir).iterdir()):
        if not d.is_dir():
            continue
        mid = d.name
        out.append(
            {
                "id": mid,
                "sealed": mid in sealed,
                "role": "holdout" if mid in sealed else "train_ok",
                "train_safe": mid not in sealed,
                "label": f"{mid} [HOLDOUT]" if mid in sealed else mid,
            }
        )
    return out


def coach_hints(
    live: dict | None,
    *,
    watch_opened: bool,
    n_envs: int,
    live_run_count: int = 0,
) -> list[str]:
    hints = []
    live = live or {}
    if int(live_run_count or 0) >= 2:
        hints.append(
            f"{int(live_run_count)} live train.lock(s) - use Live runs Focus to bind banner; "
            "Start from this panel stays refused (multi-train = separate ownership)."
        )
    crash = live.get("crash_rate_estimate")
    sps = live.get("steps_per_sec")
    if crash is not None:
        try:
            if float(crash) > 0.5:
                hints.append(
                    "Crash rate high - tick Collision-first on Start (or --collision-first); "
                    "do not morph a live overnight mid-run."
                )
        except (TypeError, ValueError):
            pass
    if sps is not None:
        try:
            sps_f = float(sps)
            if sps_f < 30 and n_envs > 8:
                hints.append(
                    "steps/sec low with many workers - try fewer n_envs or DummyVecEnv "
                    "(n_envs knee measure is deferred this wave)."
                )
            elif sps_f < 80 and n_envs >= 12:
                hints.append(
                    "steps/sec soft - n_envs may be past the knee; lower workers before overnight."
                )
        except (TypeError, ValueError):
            pass
    if not watch_opened:
        hints.append("Tip: Open Watch in a second window (lag-behind; does not slow train).")
    if n_envs > 16:
        hints.append(
            "Many workers - CPU-bound gym; idle GPU is normal. Measure n_envs knee before raising further."
        )
    if not hints:
        hints.append("Coach: looking healthy. Rank models by official adjusted_time, not ep_rew.")
    return hints


def _zip_entry(path: Path, kind: str) -> dict[str, Any]:
    try:
        stat = path.stat()
        size, mtime = stat.st_size, stat.st_mtime
    except OSError:
        size, mtime = 0, 0.0
    return {
        "kind": kind,
        "name": path.name,
        "path": str(path),
        "size_mb": round(size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(mtime, timezone.utc).isoformat(timespec="seconds")
        if mtime
        else None,
        "complete": size > MIN_COMPLETE_ZIP_BYTES,
    }


def model_timeline(models_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Best / Latest / Checkpoints for one run, newest first."""
    if not _safe_id(run_id):
        return []
    run_dir = Path(models_dir) / run_id
    if not run_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for name, kind in (("best_model.zip", "best"), ("latest_model.zip", "latest")):
        p = run_dir / name
        if p.is_file():
            entries.append(_zip_entry(p, kind))
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for p in ckpt_dir.glob("*.zip"):
            if not p.name.startswith("."):
                entries.append(_zip_entry(p, "checkpoint"))
    entries.sort(key=lambda e: e.get("modified") or "", reverse=True)
    return entries


def resolve_model_choice(models_dir: Path, run_id: str, which: str = "best") -> tuple[Path | None, str]:
    """Pick a zip from a run: best / latest / checkpoint (last complete) / auto."""
    if not _safe_id(run_id):
        return None, f"Bad run_id {run_id!r}"
    run_dir = Path(models_dir) / run_id
    if not run_dir.is_dir():
        return None, f"No model dir: {run_id}"
    which = (which or "best").lower()
    if which in ("best", "latest"):
        p = run_dir / f"{which}_model.zip"
        if p.is_file() and p.stat().st_size > MIN_COMPLETE_ZIP_BYTES:
            return p, f"{which}_model.zip"
        return None, f"{run_id}: no complete {which}_model.zip"
    ckpt = find_last_complete_checkpoint(run_dir)
    if ckpt is None:
        return None, f"{run_id}: no complete checkpoint"
    return ckpt, ckpt.name


def precheck_model(model_path: Path) -> tuple[bool, str]:
    """Cheap contracts gate before handing a zip to Watch / resume (no torch load)."""
    path = Path(model_path)
    if not path.is_file():
        return False, f"Missing model file: {path}"
    if path.stat().st_size <= MIN_COMPLETE_ZIP_BYTES:
        return False, f"Refuse load: {path.name} looks partial ({path.stat().st_size} bytes)"
    cfg_path = path.parent / "config.json"
    if not cfg_path.is_file() and path.parent.name == "checkpoints":
        cfg_path = path.parent.parent / "config.json"
    if not cfg_path.is_file():
        return True, f"{path.name}: no config.json - Watch will hard-check contracts on load"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return True, f"{path.name}: unreadable config.json ({exc})"
    try:
        refuse_config_contracts(cfg, label=str(cfg_path))
    except ContractsMismatch as exc:
        return False, f"Refuse load: {exc}"
    return True, f"{path.name}: contracts ok ({cfg.get('contracts_version')})"


def generate_maps_op(
    maps_dir: Path,
    *,
    count: int = 3,
    seed: int = 123,
    prefix: str | None = None,
    generator: Callable[..., list[Path]] | None = None,
) -> dict[str, Any]:
    """Generate N procedural maps without ever clobbering existing / sealed maps."""
    maps_dir = Path(maps_dir)
    try:
        count = int(count)
        seed = int(seed)
    except (TypeError, ValueError):
        return {"ok": False, "created": [], "msg": "Generate refused: count/seed must be integers"}
    if not 1 <= count <= 10:
        return {"ok": False, "created": [], "msg": "Generate refused: count must be 1..10"}

    prefix = (prefix or f"gen{seed}_").strip()
    if not _safe_id(prefix):
        return {"ok": False, "created": [], "msg": f"Generate refused: bad prefix {prefix!r}"}
    if prefix.lower() in RESERVED_MAP_PREFIXES:
        return {
            "ok": False,
            "created": [],
            "msg": f"Generate refused: prefix '{prefix}' is the protocol namespace (map0/map1/map2)",
        }

    if generator is None:
        # Map pack path: cached generation that registers roles/hashes and never
        # redraws a sealed map (skip_existing keeps prior tracks byte-identical).
        try:
            from . import map_pack

            result = map_pack.ensure_maps(
                count,
                seed=seed,
                prefix=prefix,
                maps_root=maps_dir,
                role="train_ok",
            )
        except Exception as exc:
            return {"ok": False, "created": [], "msg": f"Generate failed: {exc}"}

        created = list(result.get("created") or [])
        cached = list(result.get("cached") or [])
        if not created:
            return {
                "ok": False,
                "created": [],
                "cached": cached,
                "msg": (
                    f"Generate made nothing new: {', '.join(cached)} already exist for seed {seed}"
                    if cached
                    else f"Generate produced nothing for seed {seed} - try another seed"
                ),
            }
        note = f" ({len(cached)} already existed: {', '.join(cached)})" if cached else ""
        return {
            "ok": True,
            "created": created,
            "cached": cached,
            "msg": f"Generated {len(created)} map(s): {', '.join(created)}{note}",
        }

    # Injected generator (tests / no map pack): refuse to touch anything existing.
    try:
        sealed = set(sealed_map_ids(load_protocol()))
    except Exception:
        sealed = set()
    planned = [f"{prefix}{i}" for i in range(count)]
    clash = [n for n in planned if n in sealed or (maps_dir / n).is_dir()]
    if clash:
        return {
            "ok": False,
            "created": [],
            "msg": f"Generate refused: would overwrite {', '.join(clash)} - pick another seed/prefix",
        }

    try:
        paths = generator(maps_dir, num_maps=count, seed=seed, prefix=prefix)
    except Exception as exc:  # generator is procedural and can fail on bad seeds
        return {"ok": False, "created": [], "msg": f"Generate failed: {exc}"}

    created = [Path(p).parent.name for p in paths]
    if not created:
        return {"ok": False, "created": [], "msg": f"Generate produced nothing for seed {seed} - try another seed"}
    partial = " (partial)" if len(created) < count else ""
    return {
        "ok": True,
        "created": created,
        "msg": f"Generated {len(created)}/{count} maps{partial}: {', '.join(created)}",
    }


def race_candidate(models_dir: Path) -> dict | None:
    """Best official PPO under the current protocol — same rules as ``compare_models``.

    Filters to ``protocol_id`` from ``eval_protocol.yaml`` and ranks with
    ``row_race_score_key`` so a short DNF proxy cannot beat a finisher.
    """
    rows = read_leaderboard(Path(models_dir) / "leaderboard.csv", official_only=True, hide_null=True)
    try:
        proto_id = str(load_protocol().get("protocol_id") or "")
    except Exception:
        proto_id = ""
    if proto_id:
        rows = [r for r in rows if str(r.get("protocol_id") or "") in ("", proto_id)]
    ppo = [r for r in rows if str(r.get("policy", "")).lower() == "ppo"]
    if not ppo:
        return None

    best = sorted(ppo, key=row_race_score_key)[0]
    finished = best.get("mean_lap_time") not in (None, "", "None", "null")
    return {
        "run_id": best.get("run_id"),
        "adjusted_time": best.get("adjusted_time"),
        "total_collisions": best.get("total_collisions"),
        "protocol_id": best.get("protocol_id") or proto_id or None,
        "dnf": not finished,
        "note": "recommended by race_score_key on current official protocol (not ep_rew)",
    }
