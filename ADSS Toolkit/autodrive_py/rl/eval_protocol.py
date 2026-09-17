"""Frozen held-out eval protocol loader + runners (official race scores)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .compat import ContractsMismatch, load_ppo_refusing_mismatch
from .contracts import CONTRACTS_VERSION, N_LIDAR_DEFAULT, TIMEOUT_S
from .ftg import FollowTheGap
from .metrics_io import make_metrics
from .racing_env import RacingEnv, assert_resolved_map_id, resolve_map_yaml
from .run_ftg import run_episode

PROTOCOL_PATH = Path(__file__).resolve().parent / "eval_protocol.yaml"


def _adjusted_or_dq(mean_lap: float | None, total_cols: int, timeout_s: float = TIMEOUT_S) -> float:
    """Race score: lap+10·cols, or timeout DQ proxy when no scored lap.

    The DNF proxy is only a reported number — it is NOT comparable to a finisher's
    score (a slow but completed lap exceeds the timeout by construction). Ranking
    must go through ``race_score_key``, which puts every finisher ahead of every DNF.
    """
    if mean_lap is not None:
        return float(mean_lap) + 10.0 * int(total_cols)
    return float(timeout_s) + 10.0 * int(total_cols)


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else PROTOCOL_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid protocol file: {p}")
    if str(data.get("contracts_version")) != CONTRACTS_VERSION:
        raise ContractsMismatch(
            f"protocol contracts_version={data.get('contracts_version')!r} "
            f"!= frozen {CONTRACTS_VERSION}"
        )
    return data


def map_file_hash(map_yaml: Path) -> str:
    """Short content hash for fingerprint / seal checks."""
    h = hashlib.sha256()
    h.update(Path(map_yaml).read_bytes())
    img_name = None
    try:
        meta = yaml.safe_load(Path(map_yaml).read_text(encoding="utf-8"))
        img_name = meta.get("image") if isinstance(meta, dict) else None
    except Exception:
        img_name = None
    if img_name:
        img = Path(map_yaml).parent / str(img_name)
        if img.is_file():
            h.update(img.read_bytes())
    return h.hexdigest()[:16]


def sealed_map_ids(protocol: dict | None = None) -> list[str]:
    proto = protocol or load_protocol()
    out = []
    for m in proto.get("maps") or []:
        if m.get("sealed"):
            out.append(str(m["id"]))
    return out


def protocol_map_ids(protocol: dict | None = None) -> list[str]:
    proto = protocol or load_protocol()
    return [str(m["id"]) for m in (proto.get("maps") or [])]


def is_holdout_map(map_id: str, protocol: dict | None = None) -> bool:
    return str(map_id) in sealed_map_ids(protocol)


def _mean_or_none(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def eval_ftg_protocol(
    *,
    protocol: dict | None = None,
    maps_root: Path | None = None,
    n_lidar: int = N_LIDAR_DEFAULT,
    episodes_override: int | None = None,
) -> dict:
    """Run FTG on every protocol map × seed; return official metrics."""
    proto = protocol or load_protocol()
    maps_root = maps_root or (Path(__file__).resolve().parent / "maps")
    episodes = int(episodes_override or proto.get("episodes_per_map", 5))
    seeds = [int(s) for s in (proto.get("seeds") or [0])]
    # Use one episode per listed seed (seeds list length == episodes when matched).
    if len(seeds) >= episodes:
        seed_list = seeds[:episodes]
    else:
        seed_list = (seeds * episodes)[:episodes]

    timeout_s = float(proto.get("timeout_s") or TIMEOUT_S)
    eval_jitter = bool(proto.get("eval_spawn_jitter", False))

    all_laps: list[float] = []
    all_progress: list[float] = []
    total_cols = 0
    per_map: dict[str, Any] = {}
    track_ids: list[str] = []

    for m in proto.get("maps") or []:
        mid = str(m["id"])
        track_ids.append(mid)
        map_yaml = resolve_map_yaml(mid, maps_root, strict=True)
        assert_resolved_map_id(mid, map_yaml)
        laps, cols, rets, progress = [], 0, [], []
        for seed in seed_list:
            env = RacingEnv(
                map_yaml=map_yaml,
                n_lidar=n_lidar,
                seed=seed,
                timeout_s=timeout_s,
                spawn_jitter=eval_jitter,
            )
            policy = FollowTheGap(n_lidar=n_lidar)
            r = run_episode(env, policy)
            cols += int(r["collisions"])
            rets.append(float(r["return"]))
            progress.append(env.progress_frac)
            if r["lap_time"] is not None:
                laps.append(float(r["lap_time"]))
        total_cols += cols
        all_laps.extend(laps)
        all_progress.extend(progress)
        per_map[mid] = {
            "mean_lap_time": _mean_or_none(laps),
            "total_collisions": cols,
            "n_episodes": len(seed_list),
            "mean_return": float(np.mean(rets)) if rets else None,
            "mean_progress_frac": _mean_or_none(progress),
            "map_hash": map_file_hash(map_yaml),
            "sealed": bool(m.get("sealed")),
        }

    n_eps = len(seed_list) * max(1, len(track_ids))
    dq = int(proto.get("collision_dq", 10))
    mean_lap = _mean_or_none(all_laps)
    adj = _adjusted_or_dq(mean_lap, total_cols, timeout_s)
    metrics = make_metrics(
        run_id=str(proto.get("ftg_run_id") or "ftg_official_v1"),
        policy="ftg",
        backend=str(proto.get("backend") or "gym"),
        mean_lap_time=mean_lap,
        total_collisions=total_cols,
        n_episodes=n_eps,
        tracks_eval=track_ids,
        kind="official",
        protocol_id=str(proto.get("protocol_id")),
        per_map=per_map,
        collision_dq=dq,
        dq=bool(total_cols > dq) or mean_lap is None,
        incomplete_laps=mean_lap is None,
        mean_return=float(np.mean([v["mean_return"] for v in per_map.values() if v.get("mean_return") is not None]))
        if any(v.get("mean_return") is not None for v in per_map.values())
        else None,
        n_lidar=n_lidar,
        seeds=seed_list,
        timeout_s=timeout_s,
        eval_spawn_jitter=eval_jitter,
        lap_times=all_laps,
        mean_progress_frac=_mean_or_none(all_progress),
        dnf=mean_lap is None,
    )
    metrics["adjusted_time"] = adj
    return metrics


def eval_ppo_protocol(
    model_path: Path,
    *,
    protocol: dict | None = None,
    maps_root: Path | None = None,
    n_lidar: int = N_LIDAR_DEFAULT,
    run_id: str | None = None,
    episodes_override: int | None = None,
) -> dict:
    """Run PPO on every protocol map × seed under the same protocol as FTG."""
    proto = protocol or load_protocol()
    maps_root = maps_root or (Path(__file__).resolve().parent / "maps")
    episodes = int(episodes_override or proto.get("episodes_per_map", 5))
    seeds = [int(s) for s in (proto.get("seeds") or [0])]
    if len(seeds) >= episodes:
        seed_list = seeds[:episodes]
    else:
        seed_list = (seeds * episodes)[:episodes]

    model = load_ppo_refusing_mismatch(model_path, n_lidar=n_lidar, device="auto")
    timeout_s = float(proto.get("timeout_s") or TIMEOUT_S)
    eval_jitter = bool(proto.get("eval_spawn_jitter", False))

    all_laps: list[float] = []
    all_progress: list[float] = []
    total_cols = 0
    per_map: dict[str, Any] = {}
    track_ids: list[str] = []

    for m in proto.get("maps") or []:
        mid = str(m["id"])
        track_ids.append(mid)
        map_yaml = resolve_map_yaml(mid, maps_root, strict=True)
        assert_resolved_map_id(mid, map_yaml)
        laps, cols, rets, progress = [], 0, [], []
        for seed in seed_list:
            env = RacingEnv(
                map_yaml=map_yaml,
                n_lidar=n_lidar,
                seed=seed,
                timeout_s=timeout_s,
                spawn_jitter=eval_jitter,
            )
            obs, info = env.reset(seed=seed)
            done = False
            ep_ret = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                if info.get("collision"):
                    cols += 1
                if info.get("lap_time") is not None:
                    laps.append(float(info["lap_time"]))
                done = terminated or truncated
            rets.append(ep_ret)
            progress.append(env.progress_frac)
        total_cols += cols
        all_laps.extend(laps)
        all_progress.extend(progress)
        per_map[mid] = {
            "mean_lap_time": _mean_or_none(laps),
            "total_collisions": cols,
            "n_episodes": len(seed_list),
            "mean_return": float(np.mean(rets)) if rets else None,
            "mean_progress_frac": _mean_or_none(progress),
            "map_hash": map_file_hash(map_yaml),
            "sealed": bool(m.get("sealed")),
        }

    n_eps = len(seed_list) * max(1, len(track_ids))
    dq = int(proto.get("collision_dq", 10))
    mean_lap = _mean_or_none(all_laps)
    rid = run_id or f"eval_ppo_{Path(model_path).parent.name}"
    adj = _adjusted_or_dq(mean_lap, total_cols, timeout_s)
    metrics = make_metrics(
        run_id=rid,
        policy="ppo",
        backend=str(proto.get("backend") or "gym"),
        mean_lap_time=mean_lap,
        total_collisions=total_cols,
        n_episodes=n_eps,
        tracks_eval=track_ids,
        kind="official",
        protocol_id=str(proto.get("protocol_id")),
        per_map=per_map,
        collision_dq=dq,
        dq=bool(total_cols > dq) or mean_lap is None,
        incomplete_laps=mean_lap is None,
        model_path=str(model_path),
        mean_return=float(np.mean([v["mean_return"] for v in per_map.values() if v.get("mean_return") is not None]))
        if any(v.get("mean_return") is not None for v in per_map.values())
        else None,
        n_lidar=n_lidar,
        seeds=seed_list,
        timeout_s=timeout_s,
        eval_spawn_jitter=eval_jitter,
        lap_times=all_laps,
        mean_progress_frac=_mean_or_none(all_progress),
        dnf=mean_lap is None,
    )
    metrics["adjusted_time"] = adj
    return metrics


def protocol_timeout_s(path: Path | None = None) -> float:
    """Official sim-time budget from ``eval_protocol.yaml`` (laps need ~180–210 s).

    Train episodes may still use ``contracts.TIMEOUT_S`` (60 s) for shaping.
    Viewer / ghost / official eval should call this so FTG is not a false DNF.
    """
    try:
        return float(load_protocol(path).get("timeout_s") or TIMEOUT_S)
    except Exception:
        return float(TIMEOUT_S)


def race_score_key(metrics: dict) -> tuple:
    """Sort key for race ranking — lower is better.

    Ordering, in priority order:
      1. finishers (a scored lap) ahead of every DNF, so a slow completed lap can
         never lose to a car that never finished;
      2. among finishers: adjusted_time = lap + 10·collisions (then progress, cols);
      3. among DNFs: **progress first**, then fewer collisions, then adj proxy —
         so far-then-crash beats idle-timeout-at-2% (adj-before-progress was wrong
         when mid-train timeout makes adj ≈ timeout + 10·cols).
    """
    finished = metrics.get("mean_lap_time") is not None
    adj = metrics.get("adjusted_time")
    try:
        adj_f = float(adj)
    except (TypeError, ValueError):
        adj_f = 1e18
    try:
        progress = float(metrics.get("mean_progress_frac") or 0.0)
    except (TypeError, ValueError):
        progress = 0.0
    try:
        cols_i = int(metrics.get("total_collisions"))
    except (TypeError, ValueError):
        cols_i = 10**9
    if finished:
        return (0, adj_f, -progress, cols_i)
    # DNF: progress ≻ collisions ≻ timeout/collision adj proxy.
    return (1, -progress, cols_i, adj_f)


def row_race_score_key(row: dict) -> tuple:
    """Leaderboard-row adapter for ``race_score_key`` (CSV nulls / dnf flags)."""
    mean_lap = row.get("mean_lap_time")
    finished = mean_lap not in (None, "", "None", "null")
    dnf_flag = str(row.get("dnf", "")).strip().lower() in ("1", "true", "yes")
    metrics = {
        "mean_lap_time": float(mean_lap) if finished else None,
        "adjusted_time": row.get("adjusted_time"),
        "mean_progress_frac": row.get("mean_progress_frac"),
        "total_collisions": row.get("total_collisions"),
        "dnf": (not finished) or dnf_flag,
    }
    return race_score_key(metrics)
