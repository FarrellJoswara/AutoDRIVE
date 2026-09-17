"""Tick0 / W2 verify smokes — overnight observe-only (never touches protected soak).

Runs: overnight constant audit, holdout refuse, collision-first UI argv check,
progress_probe smoke, disposable Continue argv + optional short train proof.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from . import control_ui as cui
from . import ui_ops
from .map_pack import HoldoutViolation, assert_train_safe, verify_pack


RL = Path(__file__).resolve().parent
ROOT = RL.parent  # autodrive_py
PROTECTED = "overnight_soak_20260917_082739"


def _ok(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise SystemExit(1)


def audit_overnight_constants() -> None:
    print("=== overnight constant audit ===")
    _ok("MID_TRAIN_SELECT_TIMEOUT_S >= 220", ui_ops.MID_TRAIN_SELECT_TIMEOUT_S >= 220.0, str(ui_ops.MID_TRAIN_SELECT_TIMEOUT_S))
    _ok("AUTO_RACE_EVAL_EVERY_FLOOR >= 50000", ui_ops.AUTO_RACE_EVAL_EVERY_FLOOR >= 50_000, str(ui_ops.AUTO_RACE_EVAL_EVERY_FLOOR))
    _ok("DEFAULT_EARLY_STOP_WARMUP_EVALS >= 2", ui_ops.DEFAULT_EARLY_STOP_WARMUP_EVALS >= 2)
    _ok("DEFAULT_EARLY_STOP_MIN_TIMESTEPS >= 100000", ui_ops.DEFAULT_EARLY_STOP_MIN_TIMESTEPS >= 100_000)
    night = ui_ops.PRESETS["overnight"]
    _ok("overnight stop_on_budget False", night.get("stop_on_budget") is False)
    _ok("overnight patience >= 5", int(night.get("early_stop_patience", 0)) >= 5)
    _ok("overnight eval_every >= 50000", int(night.get("race_eval_every", 0)) >= 50_000)
    _ok("overnight warmup >= 2", int(night.get("early_stop_warmup_evals", 0)) >= 2)
    _ok("overnight min_ts >= 100000", int(night.get("early_stop_min_timesteps", 0)) >= 100_000)
    _ok("overnight select_timeout >= 220", float(night.get("select_timeout", 0)) >= 220.0)


def holdout_refuse() -> None:
    print("=== holdout / validation refuse ===")
    maps = RL / "maps"
    for mid in ("map2", "map3", "map4"):
        try:
            assert_train_safe([mid], maps_root=maps)
            _ok(f"assert_train_safe({mid}) raises", False, "did not raise")
        except HoldoutViolation as exc:
            _ok(f"assert_train_safe({mid}) raises", True, str(exc)[:80])
    # CLI exit codes (device cpu, tiny — should die before env build if guard works)
    py = sys.executable
    for mid in ("map2", "map3"):
        r = subprocess.run(
            [
                py,
                "-m",
                "rl.train_ppo",
                "--map",
                mid,
                "--timesteps",
                "64",
                "--n-envs",
                "1",
                "--vec-env",
                "dummy",
                "--device",
                "cpu",
                "--eval_episodes",
                "0",
                "--run_id",
                f"refuse_{mid}_smoke",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        _ok(f"train_ppo --map {mid} exit!=0", r.returncode != 0, f"code={r.returncode}")
    vr = verify_pack(maps)
    _ok("verify_pack ok", bool(vr.get("ok")), str(vr.get("issues", []))[:160])


def collision_first_ui_argv() -> None:
    print("=== collision-first UI wiring ===")
    prev = ui_ops.start_preview(
        "map0", 4096, 1, "preview_cf", RL / "models", RL / "maps", collision_first=True
    )
    _ok("preview collision_first True", prev.get("collision_first") is True, str(prev.get("summary")))
    # Build Start argv path without spawning: monkeypatch spawn
    spawned: list[list[str]] = []

    def _fake_spawn(argv, run_id, n_envs=1, action="Training started", log_mode="w"):
        spawned.append(list(argv))
        return cui._set_msg(f"FAKE {action}: {run_id}")

    old_busy, old_kill, old_spawn = cui._train_busy, cui._kill_train_tree, cui._spawn_train
    try:
        cui._train_busy = lambda force=True: None  # type: ignore[assignment]
        cui._kill_train_tree = lambda: 0  # type: ignore[assignment]
        cui._spawn_train = _fake_spawn  # type: ignore[assignment]
        msg = cui._start_train("map0", 2048, 1, collision_first=True)
        _ok("Start with collision_first did not refuse", "FAKE" in msg or "Training" in msg, msg)
        _ok("argv contains --collision-first", spawned and "--collision-first" in spawned[0], str(spawned))
        msg2 = cui._start_train("map0", 2048, 1, collision_first=False)
        _ok(
            "argv omits --collision-first when off",
            spawned and "--collision-first" not in spawned[-1],
            str(spawned[-1]),
        )
    finally:
        cui._train_busy = old_busy
        cui._kill_train_tree = old_kill
        cui._spawn_train = old_spawn


def progress_probe_smoke() -> None:
    print("=== progress_probe (FTG, short) ===")
    from .progress_probe import main as probe_main

    rc = probe_main(["--map", "map0", "--policy", "ftg", "--episodes", "1", "--timeout-s", "8"])
    _ok("progress_probe exit 0", rc == 0)


def continue_argv_smoke() -> None:
    print("=== Continue argv from disposable prior zip (if any) ===")
    models = RL / "models"
    # Prefer a tiny smoke run if present; else skip Continuity train, still test helper.
    candidates = sorted(models.glob("smoke_*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    rid = None
    for c in candidates:
        if ui_ops.find_last_complete_checkpoint and False:
            pass
        from .metrics_io import find_last_complete_checkpoint

        if find_last_complete_checkpoint(c) is not None:
            rid = c.name
            break
    if rid is None:
        print("[SKIP] no disposable smoke zip for Continue argv — constants/UI already covered")
        return
    argv, why = ui_ops.continue_train_argv(
        run_id=rid,
        models_dir=models,
        timesteps=2048,
        n_envs=1,
        stop_on_budget=True,
    )
    _ok("continue_train_argv non-empty", bool(argv), why)
    _ok("--resume in argv", "--resume" in argv, str(argv[:8]))
    _ok("does not target protected soak", rid != PROTECTED, rid)


def protected_untouched() -> None:
    print("=== protected overnight still present ===")
    ls = RL / "runs" / PROTECTED / "live_status.json"
    _ok("live_status exists", ls.is_file())
    data = json.loads(ls.read_text(encoding="utf-8"))
    _ok("run_id matches", data.get("run_id") == PROTECTED, str(data.get("run_id")))
    print(f"  timesteps={data.get('timesteps')} phase={data.get('phase')} msg={data.get('msg')}")


def main() -> int:
    audit_overnight_constants()
    holdout_refuse()
    collision_first_ui_argv()
    progress_probe_smoke()
    continue_argv_smoke()
    protected_untouched()
    print("ALL W2 / Tick0 verify smokes PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
