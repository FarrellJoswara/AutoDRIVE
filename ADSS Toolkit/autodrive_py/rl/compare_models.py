"""Phase 4 helper: print model leaderboard."""

from __future__ import annotations

import argparse
from pathlib import Path

from .metrics_io import read_leaderboard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare registered RL runs")
    parser.add_argument(
        "--leaderboard",
        type=str,
        default=str(Path(__file__).resolve().parent / "models" / "leaderboard.csv"),
    )
    args = parser.parse_args(argv)
    rows = read_leaderboard(Path(args.leaderboard))
    if not rows:
        print("No rows in leaderboard (train or run_ftg --save first).")
        return 1
    # Sort by adjusted_time ascending when present
    def key(r):
        v = r.get("adjusted_time")
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, 1e18)

    rows = sorted(rows, key=key)
    print(
        f"{'run_id':40} {'policy':6} {'backend':9} {'adj_t':>8} {'lap':>8} {'col':>4} {'n':>3}"
    )
    for r in rows:
        print(
            f"{r.get('run_id',''):40} {r.get('policy',''):6} {r.get('backend',''):9} "
            f"{str(r.get('adjusted_time')):>8} {str(r.get('mean_lap_time')):>8} "
            f"{str(r.get('total_collisions')):>4} {str(r.get('n_episodes')):>3}"
        )
    print(f"OK: {len(rows)} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
