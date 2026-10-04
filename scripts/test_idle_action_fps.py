"""Measure the fixed-step player's idle-only Unity frame cap.

Runs one staged camera-off player at a time at two idle target FPS values.
Physics remains on the existing uncapped fixed-tick action batches. The script
compares exact per-action telemetry and records Unity CPU and wall throughput.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from test_camera_fixed_interval import (
    COMPARISON_FIELDS,
    _critical_artifact_hashes,
    _field_diff,
    _run_one,
)


def run_at_fps(*, fps: int, repeat: int, simulator_path: Path, port: int,
               interval: float, seconds: float, map_id: str, seed: int,
               worker: int = 0) -> dict[str, Any]:
    key = "AICAR_ACTION_IDLE_TARGET_FPS"
    previous = os.environ.get(key)
    os.environ[key] = str(fps)
    try:
        return _run_one(
            label=f"idle-{fps}-r{repeat}", camera_enabled=False,
            simulator_path=simulator_path, port=port, interval_s=interval,
            seconds=seconds, map_id=map_id, step_timeout=15.0,
            policy_path=None, seed=seed, repeat_index=repeat, worker_index=worker,
        )
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _run_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_at_fps(**kwargs)
    except BaseException as exc:
        import traceback
        return {"_error": f"{type(exc).__name__}: {exc}", "_traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-path", type=Path, required=True)
    parser.add_argument("--reference-fps", type=int, default=30)
    parser.add_argument("--candidate-fps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel-envs", type=int, default=1)
    parser.add_argument("--reference-envs", type=int)
    parser.add_argument("--candidate-envs", type=int)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.086)
    parser.add_argument("--port-base", type=int, default=4730)
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_envs is None:
        args.reference_envs = args.parallel_envs
    if args.candidate_envs is None:
        args.candidate_envs = args.parallel_envs
    if min(args.reference_fps, args.candidate_fps, args.repeats,
           args.parallel_envs, args.reference_envs, args.candidate_envs) < 1 \
            or args.seconds <= 0 or args.interval <= 0:
        parser.error("FPS, repeats, pool size, duration, and interval must be positive")
    sim = args.simulator_path.resolve()
    hashes_start = _critical_artifact_hashes(sim)
    runs = []
    batches = []
    for repeat in range(args.repeats):
        order = [args.reference_fps, args.candidate_fps]
        if repeat % 2:
            order.reverse()
        for fps in order:
            env_count = args.reference_envs if fps == args.reference_fps else args.candidate_envs
            jobs = [dict(
                fps=fps, repeat=repeat, simulator_path=sim,
                port=args.port_base + len(runs) + worker, interval=args.interval,
                seconds=args.seconds, map_id=args.map_id, seed=args.seed + worker,
                worker=worker,
            ) for worker in range(env_count)]
            print(f"[idle-fps] start fps={fps} repeat={repeat + 1} pool={env_count}", flush=True)
            batch_start = time.perf_counter()
            if env_count == 1:
                batch = [run_at_fps(**jobs[0])]
            else:
                with ProcessPoolExecutor(max_workers=env_count) as pool:
                    batch = list(pool.map(_run_worker, jobs))
            batch_wall = time.perf_counter() - batch_start
            if any("_error" in run for run in batch):
                raise RuntimeError(json.dumps(batch, indent=2, default=str))
            runs.extend(batch)
            sim_s = sum(float(run["simulated_seconds"]) for run in batch)
            cpu_s = sum(float(run["unity_cpu_seconds"]) for run in batch)
            batches.append({
                "fps": fps,
                "repeat": repeat,
                "environment_count": env_count,
                "wall_seconds": batch_wall,
                "aggregate_simulated_seconds": sim_s,
                "aggregate_unity_cpu_seconds": cpu_s,
                "aggregate_simulated_seconds_per_wall_second": sim_s / batch_wall,
                "unity_cpu_seconds_per_wall_second": cpu_s / batch_wall,
                "unity_cpu_seconds_per_simulated_second": cpu_s / sim_s,
            })
            print(
                f"[idle-fps] done fps={fps}: pool sim-s/wall-s={sim_s / batch_wall:.4f}, "
                f"cpu/sim-s={cpu_s / sim_s:.4f}", flush=True,
            )
    comparisons = []
    pool_comparisons = []
    for repeat in range(args.repeats):
        ref_batch = next(b for b in batches if b["repeat"] == repeat and b["fps"] == args.reference_fps)
        cand_batch = next(b for b in batches if b["repeat"] == repeat and b["fps"] == args.candidate_fps)
        for worker in range(args.candidate_envs):
            reference = next(r for r in runs if r["repeat_index"] == repeat
                             and r["worker_index"] == min(worker, args.reference_envs - 1)
                             and r["label"].split("-")[1] == str(args.reference_fps))
            candidate = next(r for r in runs if r["repeat_index"] == repeat
                             and r["worker_index"] == worker
                             and r["label"].split("-")[1] == str(args.candidate_fps))
            comparison = {
                "repeat": repeat + 1,
                "worker": worker,
                "exact_trace_diff": _field_diff(reference["trace"], candidate["trace"]),
                "actions_equal": np.array_equal(reference["actions"], candidate["actions"]),
                "physics_ticks_equal": np.array_equal(
                    [x["physics_ticks"] for x in reference["trace"]],
                    [x["physics_ticks"] for x in candidate["trace"]],
                ),
                "step_ids_equal": np.array_equal(
                    [x["protocol_step_id"] for x in reference["trace"]],
                    [x["protocol_step_id"] for x in candidate["trace"]],
                ),
                "simulated_seconds_delta": abs(reference["simulated_seconds"] - candidate["simulated_seconds"]),
            }
            comparisons.append(comparison)
        pool_comparisons.append({
            "repeat": repeat + 1,
            "cpu_saving_fraction": 1.0 - cand_batch["unity_cpu_seconds_per_simulated_second"] / ref_batch["unity_cpu_seconds_per_simulated_second"],
            "throughput_ratio": cand_batch["aggregate_simulated_seconds_per_wall_second"] / ref_batch["aggregate_simulated_seconds_per_wall_second"],
            "cpu_demand_per_wall_second_ratio": cand_batch["unity_cpu_seconds_per_wall_second"] / ref_batch["unity_cpu_seconds_per_wall_second"],
            "reference_pool": ref_batch,
            "candidate_pool": cand_batch,
        })
    hashes_end = _critical_artifact_hashes(sim)
    report = {
        "benchmark": "fixed_step_idle_target_fps_ab",
        "created_unix": time.time(),
        "simulator_path": str(sim),
        "simulator_sha256": hashes_start.get("launcher"),
        "critical_artifact_hashes_start": hashes_start,
        "critical_artifact_hashes_end": hashes_end,
        "configuration": {
            "reference_fps": args.reference_fps,
            "candidate_fps": args.candidate_fps,
            "repeats": args.repeats,
            "parallel_envs": args.parallel_envs,
            "reference_envs": args.reference_envs,
            "candidate_envs": args.candidate_envs,
            "interval_s": args.interval,
            "requested_simulated_seconds": args.seconds,
            "camera_enabled": False,
            "seed": args.seed,
            "map_id": args.map_id,
        },
        "runs": [{k: v for k, v in r.items() if k not in ("trace", "actions")} for r in runs],
        "batches": batches,
        "comparisons": comparisons,
        "pool_comparisons": pool_comparisons,
        "behavior_exact": all(
            c["actions_equal"] and c["physics_ticks_equal"] and c["step_ids_equal"]
            and c["simulated_seconds_delta"] == 0
            and all(v["exact_value_mismatches"] == 0
                    for v in c["exact_trace_diff"]["fields"].values())
            for c in comparisons
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    arrays = {}
    for run in runs:
        prefix = run["label"].replace("-", "_")
        arrays[f"{prefix}_sim_time"] = np.asarray([x["sim_time_s"] for x in run["trace"]])
        arrays[f"{prefix}_physics_ticks"] = np.asarray([x["physics_ticks"] for x in run["trace"]])
        for field in COMPARISON_FIELDS:
            try:
                arrays[f"{prefix}_{field}"] = np.asarray([x[field] for x in run["trace"]])
            except (TypeError, ValueError):
                pass
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    if hashes_start != hashes_end:
        raise RuntimeError("staged simulator changed during the benchmark")
    print(f"[idle-fps] behavior_exact={report['behavior_exact']}; report={args.output}", flush=True)
    return 0 if report["behavior_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
