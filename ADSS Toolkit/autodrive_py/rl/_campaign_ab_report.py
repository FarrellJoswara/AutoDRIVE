"""Collision-first A/B honesty report (MUST #3). Observe-only — no train kill/start.

Usage (from autodrive_py):
  python -m rl._campaign_ab_report
  python -m rl._campaign_ab_report --pair cf_ab50k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent
RUNS = RL_ROOT / "runs"
MODELS = RL_ROOT / "models"

# Known campaign pairs (base, cf). Newest / longest first in default order.
KNOWN_PAIRS: dict[str, tuple[str, str]] = {
    "cf_ab50k": (
        "cf_ab50k_base_20260917_041400",
        "cf_ab50k_cf_20260917_041400",
    ),
    "tick0_ab2": (
        "tick0_ab2_base_20260917_040725",
        "tick0_ab2_cf_20260917_040725",
    ),
    "tick0_ab": (
        "tick0_ab_base_20260917_040352",
        "tick0_ab_cf_20260917_040352",
    ),
}


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _arm(run_id: str) -> dict:
    cfg = _load_json(MODELS / run_id / "config.json") or {}
    live = _load_json(RUNS / run_id / "live_status.json") or {}
    meta = _load_json(MODELS / run_id / "best_model_meta.json") or {}
    crash = live.get("crash_rate_estimate")
    # Final RaceBest overwrite often drops crash_rate — fall back to estimates.
    if crash is None and live.get("episode_count_estimate"):
        eps = float(live["episode_count_estimate"])
        cols = float(live.get("collisions_estimate") or 0)
        crash = (cols / eps) if eps > 0 else None
    return {
        "run_id": run_id,
        "exists": (MODELS / run_id).is_dir() or (RUNS / run_id).is_dir(),
        "collision_first": cfg.get("collision_first"),
        "config_timesteps": cfg.get("timesteps"),
        "n_envs": cfg.get("n_envs"),
        "vec_env": cfg.get("vec_env") or cfg.get("vec_env_type"),
        "live_timesteps": live.get("timesteps"),
        "phase": live.get("phase"),
        "crash_rate_estimate": crash,
        "collisions_estimate": live.get("collisions_estimate"),
        "episode_count_estimate": live.get("episode_count_estimate"),
        "steps_per_sec": live.get("steps_per_sec"),
        "msg": live.get("msg"),
        "best_adj": (meta.get("metrics") or {}).get("adjusted_time"),
        "best_dnf": meta.get("dnf"),
        "best_ts": meta.get("timesteps"),
    }


def report_pair(name: str, base_id: str, cf_id: str) -> dict:
    base, cf = _arm(base_id), _arm(cf_id)
    crash_b, crash_c = base["crash_rate_estimate"], cf["crash_rate_estimate"]
    delta = None
    if crash_b is not None and crash_c is not None:
        delta = float(crash_c) - float(crash_b)
    wiring_ok = base.get("collision_first") is False and cf.get("collision_first") is True
    conclusive = (
        wiring_ok
        and crash_b is not None
        and crash_c is not None
        and not (float(crash_b) == 0.0 and float(crash_c) == 0.0)
    )
    if not wiring_ok:
        verdict = "BAD_WIRING"
    elif crash_b is None or crash_c is None:
        verdict = "INCONCLUSIVE_MISSING_CRASH"
    elif float(crash_b) == 0.0 and float(crash_c) == 0.0:
        verdict = "INCONCLUSIVE_ZERO_CRASH"
    elif delta is not None and delta < 0:
        verdict = "CF_HELPS"  # lower crash on collision_first arm
    elif delta is not None and delta > 0:
        verdict = "CF_HURTS"
    else:
        verdict = "NO_DELTA"
    return {
        "pair": name,
        "wiring_ok": wiring_ok,
        "crash_delta_cf_minus_base": delta,
        "conclusive": conclusive,
        "verdict": verdict,
        "base": base,
        "cf": cf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", choices=list(KNOWN_PAIRS) + ["all"], default="all")
    ap.add_argument("--json", action="store_true", help="machine-readable dump")
    args = ap.parse_args()
    names = list(KNOWN_PAIRS) if args.pair == "all" else [args.pair]
    rows = [report_pair(n, *KNOWN_PAIRS[n]) for n in names]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print("=== Collision-first A/B honesty report (MUST #3) ===")
    print("Observe-only. Mid-train crash_rate is NOT official race score.\n")
    for r in rows:
        # ASCII-only for Windows cp1252 consoles (same class of bug as auto_train --help).
        print(f"## {r['pair']} -> {r['verdict']} (conclusive={r['conclusive']})")
        print(f"  wiring collision_first base/cf: {r['base']['collision_first']}/{r['cf']['collision_first']}")
        print(
            f"  timesteps live base/cf: {r['base']['live_timesteps']}/{r['cf']['live_timesteps']}"
        )
        print(
            f"  crash_rate base/cf/dlt: {r['base']['crash_rate_estimate']}/"
            f"{r['cf']['crash_rate_estimate']}/{r['crash_delta_cf_minus_base']}"
        )
        print(
            f"  best_adj (train_eval) base/cf: {r['base']['best_adj']}/{r['cf']['best_adj']}"
        )
        if r["verdict"] == "INCONCLUSIVE_ZERO_CRASH":
            print("  note: both arms crash~0 -- stall-heavy early policy; need longer budget or harder map.")
        if r["verdict"] == "INCONCLUSIVE_MISSING_CRASH":
            print("  note: crash_rate missing (final val trail may have overwritten learning fields).")
        print()
    any_help = any(r["verdict"] == "CF_HELPS" for r in rows)
    print("SUMMARY:", "collision_first crash-down seen" if any_help else "no conclusive crash-down yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
