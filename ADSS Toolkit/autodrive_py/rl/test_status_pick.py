"""Smoke: multi-run status pick prefers CURRENT_RUN / highest timesteps.

    python -m rl.test_status_pick
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .live_status import find_latest_status, pick_status_for_operator, write_live_status
from .ui_ops import read_operator_run_pin, write_operator_run_pin


def _write(runs: Path, run_id: str, timesteps: int, *, sleep_s: float = 0.0) -> Path:
    d = runs / run_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "live_status.json"
    write_live_status(
        path,
        {
            "run_id": run_id,
            "timesteps": timesteps,
            "phase": "learning",
            "unix_time": time.time(),
        },
    )
    if sleep_s:
        time.sleep(sleep_s)
    return path


def main() -> int:
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        if ok:
            print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
        else:
            fails += 1
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

    print("status pick (multi-run)")
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        # Newer mtime but tiny timesteps (smoke) vs older overnight-like trail.
        _write(runs, "overnight_soak_demo", 400_000, sleep_s=0.02)
        _write(runs, "smoke_ab_6k", 6_000)

        latest = find_latest_status(runs)
        check(
            "find_latest_status is mtime-based (smoke can win)",
            latest is not None and latest[1].get("run_id") == "smoke_ab_6k",
            f"got={None if latest is None else latest[1].get('run_id')}",
        )

        picked = pick_status_for_operator(runs)
        check(
            "pick_status_for_operator prefers highest timesteps",
            picked is not None and picked[1].get("run_id") == "overnight_soak_demo",
            f"got={None if picked is None else picked[1].get('run_id')}",
        )

        pinned = pick_status_for_operator(runs, preferred_run_id="smoke_ab_6k")
        check(
            "preferred_run_id wins even if lower timesteps",
            pinned is not None and pinned[1].get("run_id") == "smoke_ab_6k",
        )

        among = pick_status_for_operator(
            runs,
            candidate_run_ids=["smoke_ab_6k", "overnight_soak_demo"],
        )
        check(
            "candidate_run_ids picks highest among locks",
            among is not None and among[1].get("run_id") == "overnight_soak_demo",
        )

        missing = pick_status_for_operator(runs, preferred_run_id="no_such_run")
        check(
            "missing pin falls back to highest timesteps",
            missing is not None and missing[1].get("run_id") == "overnight_soak_demo",
        )

    print("operator CURRENT_RUN pin")
    with tempfile.TemporaryDirectory() as tmp:
        logs = Path(tmp) / "logs"
        check("missing pin returns None", read_operator_run_pin(logs) is None)
        write_operator_run_pin(logs, "overnight_soak_demo")
        check(
            "write/read roundtrip",
            read_operator_run_pin(logs) == "overnight_soak_demo",
        )

    print("all checks passed" if fails == 0 else f"{fails} check(s) failed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
