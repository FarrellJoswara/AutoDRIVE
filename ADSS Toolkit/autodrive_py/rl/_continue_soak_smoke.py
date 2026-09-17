"""Disposable Continue soak proof (MUST #2) — never touches overnight soak.

Short train → wait for complete checkpoint → resume same run_id → assert
timesteps increase. Uses CLI only (no control_ui Start/Continue → no kill-tree).

Protected overnight run_id substring is refused hard.
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
PROTECTED = "overnight_soak_20260917_082739"


def _py() -> str:
    return str(VENV_PY if VENV_PY.is_file() else sys.executable)


def _read_ts(status_path: Path) -> int | None:
    if not status_path.is_file():
        return None
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return int(data.get("timesteps") or 0) or None
    except (TypeError, ValueError):
        return None


def _wait_proc(proc: subprocess.Popen, log_path: Path, *, label: str, timeout_s: float) -> int:
    t0 = time.time()
    while proc.poll() is None:
        if time.time() - t0 > timeout_s:
            print(f"FAIL: {label} hung >{timeout_s:.0f}s — killing OUR pid only", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 124
        time.sleep(1.0)
    code = int(proc.returncode or 0)
    print(f"{label} exited code={code} log={log_path.name}", flush=True)
    return code


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"continue_soak_{stamp}"
    if PROTECTED in run_id or run_id.startswith("overnight_soak_"):
        print("REFUSE: run_id collides with protected overnight naming")
        return 2

    models = RL / "models" / run_id
    status_path = RL / "runs" / run_id / "live_status.json"
    log_path = RL / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    base = [
        _py(),
        "-m",
        "rl.train_ppo",
        "--map",
        "map0",
        "--n-envs",
        "1",
        "--vec-env",
        "dummy",
        "--device",
        "cpu",
        "--n-steps",
        "256",
        "--batch-size",
        "128",
        "--n-epochs",
        "2",
        "--net-arch",
        "64,64",
        "--eval_episodes",
        "1",
        "--select-timeout",
        "5",
        "--save-latest-every-rollouts",
        "0",
        "--run_id",
        run_id,
    ]

    phase1 = base + [
        "--timesteps",
        "1024",
        "--checkpoint-every",
        "512",
        "--race-eval-every",
        "0",
        "--early-stop-patience",
        "0",
    ]
    print("CONTINUE-SOAK phase1", run_id, flush=True)
    print("CMD", " ".join(phase1), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            phase1,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        code = _wait_proc(proc, log_path, label="phase1", timeout_s=300)
    if code != 0:
        print(f"FAIL: phase1 exit {code}")
        return code

    from rl.metrics_io import find_last_complete_checkpoint

    ckpt = find_last_complete_checkpoint(models)
    if ckpt is None:
        # Fallback: latest/best if checkpoint cadence missed (tiny budget).
        for name in ("latest_model.zip", "best_model.zip"):
            p = models / name
            if p.is_file() and p.stat().st_size > 1024:
                ckpt = p
                break
    if ckpt is None:
        print(f"FAIL: no complete checkpoint under {models}")
        return 1
    ts_after_1 = _read_ts(status_path)
    print(f"phase1 ckpt={ckpt.name} ts={ts_after_1}", flush=True)

    # Clear stale lock if any (train_ppo should have released; don't touch other runs).
    lock = models / "train.lock"
    if lock.is_file():
        try:
            lock.unlink()
            print("cleared leftover train.lock", flush=True)
        except OSError as exc:
            print(f"WARN: could not clear lock: {exc}", flush=True)

    phase2 = base + [
        "--resume",
        run_id,
        "--timesteps",
        "1024",
        "--checkpoint-every",
        "512",
        "--race-eval-every",
        "0",
        "--early-stop-patience",
        "0",
    ]
    print("CONTINUE-SOAK phase2 resume", run_id, flush=True)
    print("CMD", " ".join(phase2), flush=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_f:
        log_f.write("\n--- CONTINUE phase2 ---\n")
        proc = subprocess.Popen(
            phase2,
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        code = _wait_proc(proc, log_path, label="phase2", timeout_s=300)
    if code != 0:
        print(f"FAIL: phase2 exit {code}")
        return code

    ts_after_2 = _read_ts(status_path)
    print(f"phase2 ts={ts_after_2} (phase1 ts={ts_after_1})", flush=True)
    if ts_after_2 is None:
        print("FAIL: no timesteps in live_status after Continue")
        return 1
    if ts_after_1 is not None and ts_after_2 <= ts_after_1:
        print(f"FAIL: timesteps did not increase ({ts_after_1} -> {ts_after_2})")
        return 1
    if ts_after_1 is None and ts_after_2 < 512:
        print(f"FAIL: continue timesteps too low ({ts_after_2})")
        return 1

    # Sanity: never touched protected overnight lock.
    overnight_lock = RL / "models" / PROTECTED / "train.lock"
    if overnight_lock.is_file():
        try:
            data = json.loads(overnight_lock.read_text(encoding="utf-8"))
            print(f"overnight lock intact pid={data.get('pid')} (observe-only)", flush=True)
        except (OSError, json.JSONDecodeError):
            print("overnight lock present (unreadable) — left alone", flush=True)

    print(f"PASS continue soak: {run_id} timesteps {ts_after_1} -> {ts_after_2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
