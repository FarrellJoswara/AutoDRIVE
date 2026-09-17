"""Compare registered RL runs; prefer official adjusted_time; beat-FTG delta."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import CONTRACTS_VERSION
from .eval_protocol import load_protocol, row_race_score_key
from .metrics_io import read_leaderboard


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare registered RL runs")
    parser.add_argument(
        "--leaderboard",
        type=str,
        default=str(Path(__file__).resolve().parent / "models" / "leaderboard.csv"),
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Show only kind=official rows",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include null adjusted_time / all kinds (debug)",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Print recommended race candidate (best official PPO adjusted_time)",
    )
    parser.add_argument(
        "--protocol-id",
        type=str,
        default="",
        help="Filter to this protocol_id (default: current eval_protocol.yaml when --official)",
    )
    args = parser.parse_args(argv)
    path = Path(args.leaderboard)
    rows = read_leaderboard(
        path,
        hide_null=not args.all,
        official_only=bool(args.official),
        contracts_only=not args.all,
    )
    proto_id = (args.protocol_id or "").strip()
    if not proto_id and args.official and not args.all:
        try:
            proto_id = str(load_protocol().get("protocol_id") or "")
        except Exception:
            proto_id = ""
    if proto_id:
        rows = [r for r in rows if str(r.get("protocol_id") or "") in ("", proto_id)]
    if not rows:
        print(
            "No usable rows (need non-null adjusted_time under contracts "
            f"{CONTRACTS_VERSION}). Train, or: python -m rl.eval_cli --official --policy ftg"
        )
        return 1

    rows = sorted(rows, key=row_race_score_key)
    ftg = [r for r in rows if str(r.get("policy", "")).lower() == "ftg"]
    # Prefer a finished FTG pin when present (DNF proxy is not a beatable baseline).
    ftg_finishers = [r for r in ftg if r.get("mean_lap_time") not in (None, "", "None", "null")]
    ftg_best = (ftg_finishers or ftg)[0] if ftg else None

    print(
        f"{'run_id':40} {'pol':5} {'kind':8} {'adj_t':>8} {'lap':>8} {'col':>4} {'n':>3} {'dFTG':>8}"
    )
    for r in rows:
        adj = r.get("adjusted_time")
        d_ftg = ""
        if ftg_best is not None and adj not in (None, "", "None"):
            try:
                d_ftg = f"{float(adj) - float(ftg_best['adjusted_time']):+.2f}"
            except (TypeError, ValueError):
                d_ftg = ""
        print(
            f"{r.get('run_id', ''):40} {r.get('policy', ''):5} {str(r.get('kind') or ''):8} "
            f"{str(adj):>8} {str(r.get('mean_lap_time')):>8} "
            f"{str(r.get('total_collisions')):>4} {str(r.get('n_episodes')):>3} {d_ftg:>8}"
        )

    if args.recommend:
        ppo = [r for r in rows if str(r.get("policy", "")).lower() == "ppo"]
        if not ppo:
            print("recommended: (none - no PPO rows with adjusted_time)")
        else:
            best = ppo[0]
            print(
                f"recommended_race_candidate: {best.get('run_id')} "
                f"adjusted_time={best.get('adjusted_time')} "
                f"(selected by race_score_key / adjusted_time, not ep_rew)"
            )
            if ftg_best is not None:
                try:
                    beat = row_race_score_key(best) < row_race_score_key(ftg_best)
                    print(
                        f"beats_pinned_ftg: {beat} "
                        f"(ftg={ftg_best.get('adjusted_time')} run={ftg_best.get('run_id')})"
                    )
                except (TypeError, ValueError):
                    pass

    suffix = f" protocol={proto_id}" if proto_id else ""
    print(f"OK: {len(rows)} row(s) contracts={CONTRACTS_VERSION}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
