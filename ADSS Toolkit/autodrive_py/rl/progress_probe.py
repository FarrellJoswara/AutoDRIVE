"""Honest mid-train / operator progress probe — NEVER official, NEVER promote.

Labeled ``kind=smoke`` / status-only. Reports mean progress fraction, crash tags,
and (if any) finished adjusted_time. Does not write leaderboard official rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compat import load_ppo_refusing_mismatch
from .contracts import N_LIDAR_DEFAULT
from .ftg import FollowTheGap
from .racing_env import RacingEnv, assert_resolved_map_id, resolve_map_yaml


def _progress_frac(info: dict) -> float:
    for key in ("progress_frac", "s_frac", "lap_progress"):
        if key in info and info[key] is not None:
            try:
                return float(np.clip(float(info[key]), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
    s = info.get("s")
    length = info.get("track_length") or info.get("centerline_length")
    if s is not None and length:
        try:
            return float(np.clip(float(s) / float(length), 0.0, 1.0))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return 0.0


def probe_ftg(map_yaml: Path, *, episodes: int, n_lidar: int, seed: int, timeout_s: float) -> dict:
    env = RacingEnv(map_yaml=map_yaml, n_lidar=n_lidar, seed=seed, timeout_s=timeout_s)
    policy = FollowTheGap(n_lidar=n_lidar)
    progresses, crashes, adj = [], 0, []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        cols = 0
        best_p = 0.0
        lap_t = None
        while not done:
            raw = env._raw_scan
            action = policy.act(raw)
            obs, _reward, terminated, truncated, info = env.step(action)
            best_p = max(best_p, _progress_frac(info))
            if info.get("collision"):
                cols += 1
            if info.get("lap_time") is not None:
                lap_t = float(info["lap_time"])
            done = terminated or truncated
        progresses.append(best_p)
        crashes += cols
        if lap_t is not None:
            adj.append(lap_t + 10.0 * cols)
    return _pack("ftg", map_yaml.stem, progresses, crashes, adj, episodes, seed, timeout_s)


def probe_ppo(
    map_yaml: Path,
    model_path: Path,
    *,
    episodes: int,
    n_lidar: int,
    seed: int,
    timeout_s: float,
) -> dict:
    env = RacingEnv(map_yaml=map_yaml, n_lidar=n_lidar, seed=seed, timeout_s=timeout_s)
    model = load_ppo_refusing_mismatch(model_path, n_lidar=n_lidar)
    progresses, crashes, adj = [], 0, []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        cols = 0
        best_p = 0.0
        lap_t = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            best_p = max(best_p, _progress_frac(info))
            if info.get("collision"):
                cols += 1
            if info.get("lap_time") is not None:
                lap_t = float(info["lap_time"])
            done = terminated or truncated
        progresses.append(best_p)
        crashes += cols
        if lap_t is not None:
            adj.append(lap_t + 10.0 * cols)
    return _pack("ppo", map_yaml.stem, progresses, crashes, adj, episodes, seed, timeout_s)


def _pack(
    policy: str,
    map_id: str,
    progresses: list[float],
    crashes: int,
    adj: list[float],
    episodes: int,
    seed: int,
    timeout_s: float,
) -> dict:
    return {
        "kind": "smoke",
        "label": "progress_probe",
        "official": False,
        "promotes_best_model": False,
        "policy": policy,
        "map": map_id,
        "episodes": episodes,
        "seed": seed,
        "timeout_s": float(timeout_s),
        "mean_progress_frac": float(np.mean(progresses)) if progresses else 0.0,
        "crash_events": int(crashes),
        "finishers": len(adj),
        "mean_adjusted_time_finishers_only": float(np.mean(adj)) if adj else None,
        "note": (
            "NOT official / NOT beat-FTG. Finishers use adjusted_time among "
            "completed laps only; DNFs reported via progress + crash_events."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Honest progress probe (smoke only — never official)"
    )
    p.add_argument("--map", default="map0")
    p.add_argument("--policy", choices=("ftg", "ppo"), default="ftg")
    p.add_argument("--model", type=str, default="", help="PPO zip (required for --policy ppo)")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=30.0, help="Short probe budget (not official 400)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-lidar", type=int, default=N_LIDAR_DEFAULT)
    p.add_argument("--maps-root", type=str, default="")
    args = p.parse_args(argv)

    root = Path(args.maps_root) if args.maps_root else Path(__file__).resolve().parent / "maps"
    try:
        map_yaml = resolve_map_yaml(args.map, root, strict=True)
        assert_resolved_map_id(args.map, map_yaml)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2
    if args.policy == "ftg":
        out = probe_ftg(
            map_yaml,
            episodes=max(1, args.episodes),
            n_lidar=args.n_lidar,
            seed=args.seed,
            timeout_s=max(1.0, float(args.timeout_s)),
        )
    else:
        if not args.model:
            print("ERROR: --model required for ppo probe")
            return 2
        out = probe_ppo(
            map_yaml,
            Path(args.model),
            episodes=max(1, args.episodes),
            n_lidar=args.n_lidar,
            seed=args.seed,
            timeout_s=max(1.0, float(args.timeout_s)),
        )
    print(json.dumps(out, indent=2))
    print(
        f"[progress_probe] kind={out['kind']} mean_progress={out['mean_progress_frac']:.3f} "
        f"crashes={out['crash_events']} finishers={out['finishers']} "
        f"(NOT official)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
