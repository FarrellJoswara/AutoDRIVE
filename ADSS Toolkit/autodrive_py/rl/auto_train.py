"""W4 thin Auto-train scaffold — outer chain only; never mutates overnight.

Contract (CAMPAIGN_BOARD ``20260917-W4-auto-train-pipeline`` SHIP thin)::

    train_ok map → early-stop / short-budget train → official eval → next pack
    train_ok id / seed

Hard rules
----------
* **No** reward / curriculum mutation (no ``--collision-first``, ``--speed-gate``,
  Discord dials, or sacred early-stop floor edits in the frozen template).
* **No** holdout / validation training (``assert_train_safe``; refuse
  ``--allow-holdout``).
* **No** Start/Continue/kill/morph of a live overnight soak — refuse any live
  ``train.lock`` / overnight PID; never call ``_kill_train_tree``.
* Default **dry-run** (prints plan, spawns nothing). Pass ``--execute`` to run.
* ``--max-runs`` default **1**, hard-capped at **3**.

Usage (from ``ADSS Toolkit/autodrive_py``, with ``rl/.venv``)::

    python -m rl.auto_train --help
    python -m rl.auto_train --dry-run
    python -m rl.auto_train --map map2          # refuse (holdout)
    python -m rl.auto_train --execute --smoke   # Dummy short chain when idle
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RL_DIR = Path(__file__).resolve().parent
ROOT = RL_DIR.parent
MAPS_DIR = RL_DIR / "maps"
MODELS_DIR = RL_DIR / "models"
RUNS_DIR = RL_DIR / "runs"
LOGS_DIR = RL_DIR / "logs"
VENV_PY = RL_DIR / ".venv" / "Scripts" / "python.exe"

PROTECTED_OVERNIGHT_RUN = "overnight_soak_20260917_082739"
PROTECTED_PREFIX = "overnight_soak_"
MAX_RUNS_HARD_CAP = 3

# Frozen template knobs — sacred early-stop floors (do not lower for product).
FROZEN = {
    "n_envs": 8,
    "vec_env": "subproc",
    "device": "auto",
    "n_steps": 2048,
    "batch_size": 1024,
    "n_epochs": 10,
    "net_arch": "512,512",
    "checkpoint_every": 25000,
    "timesteps": 500_000,
    "unlimited": True,
    "early_stop_patience": 5,
    "early_stop_min_improve": 0.5,
    "early_stop_warmup_evals": 2,
    "early_stop_min_timesteps": 100_000,
    "race_eval_every": 50_000,
    "select_timeout": 220.0,
}

# Disposable Dummy smoke (acceptance #6 only) — short budget, not overnight math.
SMOKE = {
    "n_envs": 1,
    "vec_env": "dummy",
    "device": "cpu",
    "n_steps": 256,
    "batch_size": 128,
    "n_epochs": 2,
    "net_arch": "64,64",
    "checkpoint_every": 512,
    "timesteps": 2048,
    "unlimited": False,
    "early_stop_patience": 1,
    "early_stop_min_improve": 0.5,
    "early_stop_warmup_evals": 0,
    "early_stop_min_timesteps": 0,
    "race_eval_every": 1024,
    "select_timeout": 5.0,
}


def _py() -> str:
    return str(VENV_PY if VENV_PY.is_file() else sys.executable)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _is_protected_run_id(run_id: str) -> bool:
    rid = str(run_id or "")
    return (
        rid == PROTECTED_OVERNIGHT_RUN
        or rid.startswith(PROTECTED_PREFIX)
        or PROTECTED_OVERNIGHT_RUN in rid
    )


def refuse_live_train(*, force: bool = False) -> str | None:
    """Refuse when any live train.lock / overnight soak is active.

    Auto-train must never Start beside overnight and never kill anything.
    ``force`` is intentionally unsupported for product use (always refuse).
    """
    del force  # never override — overnight observe-only
    from .ui_ops import find_live_run_locks, lock_owner
    from .live_status import find_latest_status

    live_locks = find_live_run_locks(MODELS_DIR)
    if live_locks:
        bits = [
            f"{row.get('run_id')} pid={row.get('pid')}"
            for row in live_locks
        ]
        return (
            "Refuse: live train.lock present ("
            + "; ".join(bits)
            + "). Auto-train will not Start beside an active trainer "
            "(including overnight soak). Spawns nothing; kills nothing."
        )

    found = find_latest_status(RUNS_DIR)
    if found is not None:
        _path, live = found
        live = live or {}
        phase = str(live.get("phase") or "")
        rid = str(live.get("run_id") or "")
        if phase in ("learning", "validating", "saving", "starting") and rid:
            owner = lock_owner(MODELS_DIR, rid)
            if owner and owner.get("alive"):
                return (
                    f"Refuse: live train {rid} phase={phase} pid={owner.get('pid')}. "
                    "Auto-train will not compete with overnight / active soak."
                )
    return None


def resolve_queue(
    *,
    maps: list[str] | None,
    start_map: str | None,
    max_runs: int,
    maps_root: Path,
) -> list[dict[str, Any]]:
    """Build ordered train_ok jobs from the pack (map id + declared seed)."""
    from .map_pack import HoldoutViolation, assert_train_safe, load_pack, train_safe_maps

    pack = load_pack(maps_root)
    safe = train_safe_maps(maps_root, pack=pack)
    if not safe:
        raise RuntimeError("no train_ok maps in pack")

    if maps:
        wanted = [str(m) for m in maps]
        assert_train_safe(wanted, maps_root=maps_root, pack=pack)
        for mid in wanted:
            if mid not in safe:
                raise HoldoutViolation(
                    f"{mid} is not in train_safe_maps()={safe}"
                )
        order = wanted
    else:
        order = list(safe)
        if start_map:
            start_map = str(start_map)
            assert_train_safe([start_map], maps_root=maps_root, pack=pack)
            if start_map not in order:
                raise HoldoutViolation(
                    f"{start_map} is not in train_safe_maps()={safe}"
                )
            # Rotate so start_map is first, then continue through pack.
            idx = order.index(start_map)
            order = order[idx:] + order[:idx]

    jobs: list[dict[str, Any]] = []
    for i, mid in enumerate(order[: max(1, int(max_runs))]):
        entry = (pack.get("maps") or {}).get(mid) or {}
        gen = entry.get("gen") or {}
        seed = gen.get("seed")
        if seed is None:
            seed = i
        jobs.append(
            {
                "map": mid,
                "seed": int(seed),
                "pack_role": entry.get("role") or "train_ok",
                "content_hash": entry.get("content_hash"),
            }
        )
    return jobs


def build_train_argv(
    *,
    run_id: str,
    map_id: str,
    seed: int,
    knobs: dict[str, Any],
) -> list[str]:
    """Fingerprinted frozen train_ppo argv (no reward / holdout flags)."""
    argv = [
        "--map",
        str(map_id),
        "--timesteps",
        str(int(knobs["timesteps"])),
        "--device",
        str(knobs["device"]),
        "--run_id",
        run_id,
        "--n-envs",
        str(int(knobs["n_envs"])),
        "--vec-env",
        str(knobs["vec_env"]),
        "--n-steps",
        str(int(knobs["n_steps"])),
        "--batch-size",
        str(int(knobs["batch_size"])),
        "--n-epochs",
        str(int(knobs["n_epochs"])),
        "--net-arch",
        str(knobs["net_arch"]),
        "--checkpoint-every",
        str(int(knobs["checkpoint_every"])),
        "--seed",
        str(int(seed)),
    ]
    if knobs.get("unlimited"):
        argv.append("--unlimited-timesteps")
    patience = int(knobs.get("early_stop_patience") or 0)
    if patience > 0:
        argv.extend(["--early-stop-patience", str(patience)])
        argv.extend(
            ["--early-stop-min-improve", f"{float(knobs['early_stop_min_improve']):g}"]
        )
        warm = int(knobs.get("early_stop_warmup_evals") or 0)
        if warm > 0:
            argv.extend(["--early-stop-warmup-evals", str(warm)])
        min_ts = int(knobs.get("early_stop_min_timesteps") or 0)
        if min_ts > 0:
            argv.extend(["--early-stop-min-timesteps", str(min_ts)])
    eval_every = int(knobs.get("race_eval_every") or 0)
    if eval_every > 0:
        argv.extend(["--race-eval-every", str(eval_every)])
    sel = float(knobs.get("select_timeout") or 0)
    if sel > 0:
        argv.extend(["--select-timeout", f"{sel:g}"])
    return argv


def plan_run_id(map_id: str, *, smoke: bool, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    stamp = _utc_stamp()
    tag = "auto_smoke" if smoke else "auto_train"
    return f"{tag}_{stamp}_{map_id}"


def _wait_proc(proc: subprocess.Popen, *, label: str, timeout_s: float, log_path: Path) -> int:
    t0 = time.time()
    while proc.poll() is None:
        if time.time() - t0 > timeout_s:
            print(
                f"FAIL: {label} hung >{timeout_s:.0f}s — terminating OUR child only "
                f"(never overnight)",
                flush=True,
            )
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 124
        time.sleep(1.0)
    code = int(proc.returncode or 0)
    print(f"{label} exited code={code} log={log_path.name}", flush=True)
    return code


def _read_phase(run_id: str) -> str | None:
    path = RUNS_DIR / run_id / "live_status.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(data.get("phase") or "") or None


def _best_model(run_id: str) -> Path | None:
    run_dir = MODELS_DIR / run_id
    for name in ("best_model.zip", "latest_model.zip"):
        p = run_dir / name
        if p.is_file() and p.stat().st_size > 1024:
            return p
    from .metrics_io import find_last_complete_checkpoint

    return find_last_complete_checkpoint(run_dir)


def run_official_eval(
    *,
    model_path: Path,
    run_id: str,
    episodes: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Official protocol path only (seals asserted inside eval_cli)."""
    argv = [
        _py(),
        "-m",
        "rl.eval_cli",
        "--official",
        "--policy",
        "ppo",
        "--model",
        str(model_path),
    ]
    if episodes is not None and int(episodes) > 0:
        argv.extend(["--episodes", str(int(episodes))])
    plan = {
        "cmd": argv,
        "model": str(model_path),
        "run_id": run_id,
        "kind": "official",
        "via": "eval_cli --official (+ seal assert)",
    }
    if dry_run:
        print("DRY-RUN official eval:", " ".join(argv), flush=True)
        return {"planned": plan, "executed": False}

    print("OFFICIAL EVAL", " ".join(argv), flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}_auto_eval.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        # Official can be long (400s × maps × episodes); smoke uses episodes=1.
        code = _wait_proc(
            proc,
            label="official-eval",
            timeout_s=7200.0,
            log_path=log_path,
        )
    out: dict[str, Any] = {"planned": plan, "executed": True, "exit_code": code, "log": str(log_path)}
    if code != 0:
        out["error"] = f"official eval exited {code}"
        return out

    # Archive a copy of the last official-ish metrics blob if present in log.
    archive = MODELS_DIR / run_id / "auto_train_official.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # Best-effort: last JSON object in log.
        start = text.rfind("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            metrics = json.loads(text[start : end + 1])
            metrics["_auto_train"] = {
                "run_id": run_id,
                "model": str(model_path),
                "archived_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            archive.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            out["archive"] = str(archive)
            # Also copy best_model sidecar stamp.
            side = MODELS_DIR / run_id / "auto_train_best_model.txt"
            side.write_text(str(model_path.resolve()) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        out["archive_warn"] = str(exc)
    return out


def execute_job(
    job: dict[str, Any],
    *,
    knobs: dict[str, Any],
    smoke: bool,
    dry_run: bool,
    run_id: str,
    eval_episodes: int | None,
    train_timeout_s: float,
) -> dict[str, Any]:
    from .map_pack import assert_train_safe, pack_fingerprint

    map_id = str(job["map"])
    seed = int(job["seed"])
    assert_train_safe([map_id], maps_root=MAPS_DIR)

    if _is_protected_run_id(run_id):
        return {
            "ok": False,
            "error": (
                f"Refuse: run_id '{run_id}' collides with protected overnight naming "
                f"({PROTECTED_OVERNIGHT_RUN}). Auto-train never targets soak."
            ),
        }

    argv = build_train_argv(run_id=run_id, map_id=map_id, seed=seed, knobs=knobs)
    cmd = [_py(), "-m", "rl.train_ppo", *argv]
    fp = pack_fingerprint(MAPS_DIR)
    result: dict[str, Any] = {
        "map": map_id,
        "seed": seed,
        "run_id": run_id,
        "smoke": smoke,
        "cmd": cmd,
        "knobs": {k: knobs[k] for k in knobs},
        "pack_fingerprint": fp,
        "reward_mutation": False,
        "allow_holdout": False,
    }
    print("PLAN train:", " ".join(cmd), flush=True)
    print(
        f"  map={map_id} seed={seed} run_id={run_id} smoke={smoke} "
        f"train_ok={fp.get('train_ok')}",
        flush=True,
    )

    if dry_run:
        result["dry_run"] = True
        result["eval"] = run_official_eval(
            model_path=MODELS_DIR / run_id / "best_model.zip",
            run_id=run_id,
            episodes=eval_episodes,
            dry_run=True,
        )
        result["ok"] = True
        result["next"] = "would advance to next train_ok after early_stopped/clean exit"
        return result

    block = refuse_live_train()
    if block:
        result["ok"] = False
        result["error"] = block
        return result

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        log_f.write(json.dumps({"auto_train": result}, indent=2) + "\n---\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        code = _wait_proc(
            proc,
            label=f"train[{run_id}]",
            timeout_s=train_timeout_s,
            log_path=log_path,
        )
    result["train_exit"] = code
    result["phase"] = _read_phase(run_id)
    if code not in (0,):
        result["ok"] = False
        result["error"] = f"train exited {code}"
        return result

    model = _best_model(run_id)
    if model is None:
        result["ok"] = False
        result["error"] = f"no best/complete model under models/{run_id}"
        return result

    # Archive best_model copy under run dir for the chain artifact trail.
    archive_zip = MODELS_DIR / run_id / "auto_train_archived_best.zip"
    try:
        shutil.copy2(model, archive_zip)
        result["archived_best"] = str(archive_zip)
    except OSError as exc:
        result["archive_warn"] = str(exc)

    result["eval"] = run_official_eval(
        model_path=model,
        run_id=run_id,
        episodes=eval_episodes,
        dry_run=False,
    )
    result["ok"] = result["eval"].get("exit_code", 1) == 0
    if not result["ok"]:
        result["error"] = result["eval"].get("error") or "official eval failed"
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rl.auto_train",
        description=(
            "W4 thin Auto-train: chain train_ok -> early-stop train -> official eval -> "
            "next pack map/seed. Refuses overnight soak, live train.lock, holdouts, "
            "and reward mutation. Default dry-run (spawns nothing)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Refuses (nonzero, no spawn, no kill):\n"
            f"  * live models/*/train.lock or overnight PID ({PROTECTED_OVERNIGHT_RUN})\n"
            "  * --run-id matching overnight_soak_* / protected soak name\n"
            "  * holdout/validation maps (map2/map3/map4) via assert_train_safe\n"
            "  * --allow-holdout / reward-mutation flags (unsupported)\n"
            "  * --max-runs > 3\n"
            "\n"
            "Does NOT: mutate rewards, morph overnight knobs, hot-cycle soak maps,\n"
            "or call _kill_train_tree.\n"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="print planned argv + map ids; spawn nothing (default)",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="actually spawn train_ppo + official eval (still refuses live overnight)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="DummyVecEnv short disposable chain (acceptance #6); still refuses live lock",
    )
    p.add_argument("--map", type=str, default="", help="start / single train_ok map id")
    p.add_argument(
        "--maps",
        type=str,
        default="",
        help="comma-separated train_ok ids (order preserved); default = pack train_safe",
    )
    p.add_argument("--run-id", type=str, default="", help="optional disposable run_id")
    p.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="how many pack maps to chain (default 1, hard cap 3)",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="override official episodes_per_map (0 = protocol default; smoke uses 1)",
    )
    p.add_argument(
        "--train-timeout",
        type=float,
        default=0.0,
        help="seconds to wait on train child (0 = smoke 600 / frozen 86400)",
    )
    # Explicit poison pills — present so --help documents the refuse, and
    # so accidental flags fail closed instead of being silently ignored.
    p.add_argument(
        "--allow-holdout",
        action="store_true",
        help="UNSUPPORTED: auto-train always refuses holdout/validation training",
    )
    p.add_argument(
        "--mutate-reward",
        action="store_true",
        help="UNSUPPORTED: reward mutation is SKIP forever this campaign",
    )
    p.add_argument(
        "--collision-first",
        action="store_true",
        help="UNSUPPORTED in auto-train (no curriculum Discord dial)",
    )
    p.add_argument(
        "--speed-gate",
        action="store_true",
        help="UNSUPPORTED in auto-train (no curriculum Discord dial)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.allow_holdout:
        print(
            "REFUSE: --allow-holdout is unsupported in auto_train "
            "(holdout/validation training voids official claims).",
            flush=True,
        )
        return 2
    if args.mutate_reward or args.collision_first or args.speed_gate:
        print(
            "REFUSE: reward / curriculum mutation flags are SKIP forever "
            "(no --mutate-reward / --collision-first / --speed-gate in auto_train).",
            flush=True,
        )
        return 2

    max_runs = int(args.max_runs)
    if max_runs < 1:
        print("REFUSE: --max-runs must be >= 1", flush=True)
        return 2
    if max_runs > MAX_RUNS_HARD_CAP:
        print(
            f"REFUSE: --max-runs={max_runs} exceeds hard cap {MAX_RUNS_HARD_CAP} "
            "(thin scaffold smoke only).",
            flush=True,
        )
        return 2

    dry_run = not bool(args.execute)
    smoke = bool(args.smoke)
    knobs = dict(SMOKE if smoke else FROZEN)

    if args.run_id and _is_protected_run_id(args.run_id):
        print(
            f"REFUSE: --run-id '{args.run_id}' is protected overnight naming. "
            "Auto-train never targets or morphs the soak.",
            flush=True,
        )
        return 2

    # Live-lock refuse even on dry-run for execute path; dry-run still prints plan
    # but warns loudly. Execute always hard-refuses.
    block = refuse_live_train()
    if block and not dry_run:
        print(block, flush=True)
        return 2
    if block and dry_run:
        print(f"NOTE (dry-run): {block}", flush=True)

    maps_arg = [m.strip() for m in str(args.maps).split(",") if m.strip()]
    start = str(args.map).strip() or None
    try:
        jobs = resolve_queue(
            maps=maps_arg or None,
            start_map=start if not maps_arg else None,
            max_runs=max_runs,
            maps_root=MAPS_DIR,
        )
    except Exception as exc:  # HoldoutViolation / RuntimeError
        print(f"REFUSE: {exc}", flush=True)
        return 2

    if start and maps_arg:
        # Single-map pin already in maps list; start ignored.
        pass
    elif start and not maps_arg and max_runs == 1:
        jobs = [j for j in jobs if j["map"] == start][:1] or jobs[:1]

    eval_eps = int(args.eval_episodes)
    if eval_eps <= 0:
        eval_eps = 1 if smoke else 0
    eval_episodes = eval_eps if eval_eps > 0 else None

    train_timeout = float(args.train_timeout)
    if train_timeout <= 0:
        train_timeout = 600.0 if smoke else 86400.0

    print(
        f"auto_train dry_run={dry_run} smoke={smoke} max_runs={max_runs} "
        f"jobs={[j['map'] for j in jobs]}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    for i, job in enumerate(jobs):
        rid = plan_run_id(
            job["map"],
            smoke=smoke,
            explicit=args.run_id if (args.run_id and max_runs == 1) else None,
        )
        if max_runs > 1 and args.run_id:
            rid = f"{args.run_id}_{job['map']}_{i}"
            if _is_protected_run_id(rid):
                print(f"REFUSE: derived run_id '{rid}' protected", flush=True)
                return 2
        print(f"\n=== job {i + 1}/{len(jobs)} map={job['map']} seed={job['seed']} ===", flush=True)
        # Re-check lock before each execute job (overnight may have started).
        if not dry_run:
            block = refuse_live_train()
            if block:
                print(block, flush=True)
                return 2
        res = execute_job(
            job,
            knobs=knobs,
            smoke=smoke,
            dry_run=dry_run,
            run_id=rid,
            eval_episodes=eval_episodes,
            train_timeout_s=train_timeout,
        )
        results.append(res)
        if not res.get("ok"):
            print(f"FAIL: {res.get('error')}", flush=True)
            return 2
        print(
            f"OK job map={job['map']} run_id={rid} "
            f"{'dry-run' if dry_run else 'executed'} -> next pack slot ready",
            flush=True,
        )

    summary = {
        "ok": True,
        "dry_run": dry_run,
        "smoke": smoke,
        "n_jobs": len(results),
        "maps": [r.get("map") for r in results],
        "run_ids": [r.get("run_id") for r in results],
    }
    print("\nSUMMARY", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
