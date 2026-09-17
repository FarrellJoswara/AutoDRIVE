"""CURRENT_RUN pin hygiene selftest (multi-run Focus / Watch bind).

Does not rewrite the live operator pin unless --write-live is passed
(default: temp-dir only). Observe overnight — never kill trains.

Usage (from ADSS Toolkit/autodrive_py):
  python -m rl._pin_selftest
  python -m rl._pin_selftest --observe-live
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .ui_ops import read_operator_run_pin, write_operator_run_pin

RL_ROOT = Path(__file__).resolve().parent
LIVE_LOGS = RL_ROOT / "logs"
PROTECTED = "overnight_soak_20260917_082739"


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"[PASS] {msg}")


def test_temp_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix="pin_selftest_") as td:
        logs = Path(td)
        _check(read_operator_run_pin(logs) is None, "missing pin -> None")
        write_operator_run_pin(logs, "  smoke_ab_demo  ")
        _check(
            read_operator_run_pin(logs) == "smoke_ab_demo",
            "write strips whitespace + roundtrip",
        )
        write_operator_run_pin(logs, PROTECTED)
        _check(
            read_operator_run_pin(logs) == PROTECTED,
            "overnight-shaped id roundtrips",
        )
        # blank write clears to empty file -> None
        (logs / "CURRENT_RUN.txt").write_text("\n", encoding="utf-8")
        _check(read_operator_run_pin(logs) is None, "blank pin file -> None")


def observe_live() -> None:
    pin = read_operator_run_pin(LIVE_LOGS)
    print(f"[OBS] live logs/CURRENT_RUN.txt -> {pin!r}")
    if pin == PROTECTED:
        print("[OBS] pin matches protected overnight (good for multi-run Focus).")
    elif pin:
        print("[OBS] pin is non-overnight - Focus may prefer a disposable run.")
    else:
        print("[OBS] pin missing/blank - status ranking falls back to max timesteps.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--observe-live",
        action="store_true",
        help="print live pin without modifying it",
    )
    args = ap.parse_args()
    try:
        test_temp_roundtrip()
    except AssertionError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if args.observe_live:
        observe_live()
    print("pin selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
