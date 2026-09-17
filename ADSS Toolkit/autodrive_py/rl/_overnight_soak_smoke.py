"""Soak: multi-eval overnight early-stop gate (post-fix code).

DummyVecEnv, patience=3, warmup_evals=2, race-eval-every=2048, short select-timeout.
Asserts validating->learning cycles and no early_stopped before warmup+patience.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RL = Path(__file__).resolve().parent
ROOT = RL.parent
VENV_PY = RL / ".venv" / "Scripts" / "python.exe"
WARMUP_EVALS = 2
PATIENCE = 3
# baseline uses is_baseline; warmup covers eval_count<=2; then patience misses.
# Earliest legal stop: eval_count >= warmup + patience (=5) when every post-warmup is a miss.
MIN_LEGAL_STOP_EVAL = WARMUP_EVALS + PATIENCE
TARGET_VALS = 4  # past first cycles; prove survival beyond warmup window


def main() -> int:
    run_id = f"overnight_soak_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    status_path = RL / "runs" / run_id / "live_status.json"
    log_path = RL / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        str(VENV_PY if VENV_PY.is_file() else sys.executable),
        "-m",
        "rl.train_ppo",
        "--map",
        "map0",
        "--n-envs",
        "2",
        "--vec-env",
        "dummy",
        "--timesteps",
        "25000",
        "--n-steps",
        "1024",
        "--batch-size",
        "256",
        "--n-epochs",
        "2",
        "--eval_episodes",
        "1",
        "--race-eval-every",
        "2048",
        "--early-stop-patience",
        str(PATIENCE),
        "--early-stop-warmup-evals",
        str(WARMUP_EVALS),
        "--early-stop-min-timesteps",
        "0",
        "--early-stop-min-improve",
        "50",
        "--select-timeout",
        "5",
        "--checkpoint-every",
        "999999",
        "--save-latest-every-rollouts",
        "0",
        "--run_id",
        run_id,
        "--device",
        "cpu",
        "--net-arch",
        "64,64",
    ]
    print("SOAK", run_id, flush=True)
    print("CMD", " ".join(argv), flush=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

        phases: list[str] = []
        last: str | None = None
        vals = 0
        v2l = 0
        saw_v = False
        early_at: int | None = None
        max_eval = 0
        illegal_early = False
        t0 = time.time()
        deadline = t0 + 600

        try:
            while time.time() < deadline:
                st = None
                if status_path.is_file():
                    try:
                        st = json.loads(status_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        st = None
                if st:
                    phase = str(st.get("phase") or "")
                    try:
                        ev = int(st.get("eval_count") or 0)
                    except (TypeError, ValueError):
                        ev = 0
                    max_eval = max(max_eval, ev)
                    if phase and phase != last:
                        print(
                            f"phase {last}->{phase} ts={st.get('timesteps')} "
                            f"eval={st.get('eval_count')} no_improve={st.get('no_improve')} "
                            f"msg={str(st.get('msg') or '')[:90]}",
                            flush=True,
                        )
                        phases.append(phase)
                        if phase == "validating":
                            vals += 1
                            saw_v = True
                        if phase == "learning" and saw_v:
                            v2l += 1
                            saw_v = False
                        if phase == "early_stopped":
                            early_at = ev
                            if ev < MIN_LEGAL_STOP_EVAL:
                                illegal_early = True
                            break
                        last = phase

                    if v2l >= TARGET_VALS and proc.poll() is None and not illegal_early:
                        print(f"PASS gate: {v2l} validating->learning, still alive", flush=True)
                        break

                if proc.poll() is not None:
                    break
                time.sleep(0.35)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

        final = None
        if status_path.is_file():
            try:
                final = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                final = None
        if final and final.get("phase") == "early_stopped":
            try:
                early_at = int(final.get("eval_count") or early_at or 0)
            except (TypeError, ValueError):
                pass
            if early_at is not None and early_at < MIN_LEGAL_STOP_EVAL:
                illegal_early = True

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        val_lines = log_text.count("Validation race-eval @")
        warmup_cfg = f"warmup_evals={WARMUP_EVALS}" in log_text

        asserts = {
            "warmup_configured_in_log": warmup_cfg,
            "survived_multi_validations": v2l >= TARGET_VALS or vals >= TARGET_VALS or val_lines >= TARGET_VALS,
            "validating_to_learning": any(
                phases[i - 1] == "validating" and phases[i] == "learning"
                for i in range(1, len(phases))
            ),
            "no_early_stop_on_first_cycles": not illegal_early
            and not (early_at is not None and early_at <= WARMUP_EVALS),
            "early_stop_obeys_warmup_patience": (
                early_at is None or early_at >= MIN_LEGAL_STOP_EVAL
            ),
        }
        ok = all(asserts.values())
        result = {
            "run_id": run_id,
            "ok": ok,
            "assertions": asserts,
            "validating_cycles": vals,
            "learning_after_validating": v2l,
            "validation_log_lines": val_lines,
            "max_eval_count": max_eval,
            "early_stopped_at_eval": early_at,
            "min_legal_stop_eval": MIN_LEGAL_STOP_EVAL,
            "phases": phases,
            "proc_exit": proc.returncode,
            "wall_s": round(time.time() - t0, 1),
            "final_phase": (final or {}).get("phase"),
            "final_msg": (final or {}).get("msg"),
        }
        out = RL / "logs" / f"{run_id}_soak_result.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"wrote {out}", flush=True)
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
