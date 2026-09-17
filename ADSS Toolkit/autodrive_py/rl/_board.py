"""Scratch: print official leaderboard rows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.metrics_io import read_leaderboard  # noqa: E402

rows = read_leaderboard(Path(__file__).parent / "models" / "leaderboard.csv",
                        hide_null=True, official_only=True)
print(f"{'run_id':30s} {'pol':4s} {'adj':>8s} {'dnf':>6s} {'prog':>8s} {'cols':>5s}")
for r in rows:
    prog = (r.get("mean_progress_frac") or "")[:6]
    print(f"{r['run_id']:30s} {r['policy']:4s} {r['adjusted_time']:>8s} "
          f"{str(r.get('dnf')):>6s} {prog:>8s} {r['total_collisions']:>5s}")
