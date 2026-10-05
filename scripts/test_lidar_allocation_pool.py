"""Compare LiDAR temporary-array/intensity reuse in fixed-step Unity pools."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from test_camera_fixed_interval import _critical_artifact_hashes
from test_simulator_build_equivalence import compare, run_one


def _run(job: dict[str, Any]) -> dict[str, Any]:
    return run_one(
        label=job["label"], simulator=Path(job["simulator_path"]),
        port=job["port"], repeat=job["repeat"], seconds=job["seconds"],
        interval=job["interval"], map_id=job["map_id"], seed=job["seed"],
        worker_index=job["worker_index"],
    )


def pool_run(*, label: str, simulator: Path, envs: int, repeat: int,
             seconds: float, interval: float, port_base: int,
             seed: int, map_id: str) -> dict[str, Any]:
    jobs = [
        {
            "label": f"{label}-env{index}",
            "simulator_path": str(simulator),
            "port": port_base + index,
            "repeat": repeat,
            "seconds": seconds,
            "interval": interval,
            "map_id": map_id,
            "seed": seed + index * 1009,
            "worker_index": index,
        }
        for index in range(envs)
    ]
    print(f"[lidar-pool] start {label} repeat={repeat + 1} envs={envs}", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=envs) as pool:
        runs = list(pool.map(_run, jobs))
    wall = time.perf_counter() - started
    sim_seconds = sum(run["simulated_seconds"] for run in runs)
    cpu_seconds = sum(run["unity_cpu_seconds"] or 0.0 for run in runs)
    return {
        "label": label,
        "repeat": repeat + 1,
        "envs": envs,
        "wall_seconds": wall,
        "simulated_env_seconds": sim_seconds,
        "unity_cpu_seconds": cpu_seconds,
        "pool_simulated_env_seconds_per_wall_second": sim_seconds / wall,
        "unity_cpu_seconds_per_simulated_second": cpu_seconds / sim_seconds,
        "runs": runs,
    }


def compare_pool(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    per_env = []
    for ref, cand in zip(reference["runs"], candidate["runs"]):
        result = compare(ref, cand)
        fields = result["trace_diff"]["fields"]
        per_env.append({
            "env": ref["worker_index"],
            "actions_equal": result["actions_equal"],
            "physics_ticks_equal": result["physics_ticks_equal"],
            "protocol_step_ids_equal": result["protocol_step_ids_equal"],
            "simulated_seconds_delta": result["simulated_seconds_delta"],
            "exact_value_mismatches": {
                name: field["exact_value_mismatches"] for name, field in fields.items()
            },
        })
    exact = all(
        row["actions_equal"] and row["physics_ticks_equal"]
        and row["protocol_step_ids_equal"] and row["simulated_seconds_delta"] == 0
        and all(count == 0 for count in row["exact_value_mismatches"].values())
        for row in per_env
    )
    return {
        "behavior_exact": exact,
        "environments": per_env,
        "pool_throughput_ratio": (
            candidate["pool_simulated_env_seconds_per_wall_second"]
            / reference["pool_simulated_env_seconds_per_wall_second"]
        ),
        "cpu_per_simulated_second_ratio": (
            candidate["unity_cpu_seconds_per_simulated_second"]
            / reference["unity_cpu_seconds_per_simulated_second"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.086)
    parser.add_argument("--port-base", type=int, default=5160)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.envs, args.repeats) < 1 or min(args.seconds, args.interval) <= 0:
        parser.error("envs, repeats, seconds, and interval must be positive")
    paths = {"reference": args.reference.resolve(), "candidate": args.candidate.resolve()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_start = {key: _critical_artifact_hashes(path) for key, path in paths.items()}
    runs = []
    comparisons = []
    for repeat in range(args.repeats):
        order = ["reference", "candidate"] if repeat % 2 == 0 else ["candidate", "reference"]
        pair = {}
        for label in order:
            run = pool_run(
                label=label, simulator=paths[label], envs=args.envs, repeat=repeat,
                seconds=args.seconds, interval=args.interval,
                port_base=args.port_base + len(runs) * args.envs,
                seed=args.seed, map_id=args.map_id,
            )
            runs.append(run)
            pair[label] = run
            print(
                f"[lidar-pool] done {label}: "
                f"pool-sim-s/wall-s={run['pool_simulated_env_seconds_per_wall_second']:.4f}, "
                f"CPU/sim-s={run['unity_cpu_seconds_per_simulated_second']:.4f}",
                flush=True,
            )
        comparisons.append(compare_pool(pair["reference"], pair["candidate"]))
    hashes_end = {key: _critical_artifact_hashes(path) for key, path in paths.items()}
    report = {
        "benchmark": "fixed_step_lidar_allocation_pool_ab",
        "created_unix": time.time(),
        "simulator_paths": {key: str(path) for key, path in paths.items()},
        "critical_artifact_hashes_unchanged": hashes_start == hashes_end,
        "configuration": {
            "envs": args.envs, "repeats": args.repeats,
            "seconds_per_env": args.seconds, "interval_s": args.interval,
            "map_id": args.map_id, "seed": args.seed,
        },
        "runs": [{key: value for key, value in run.items() if key != "runs"} for run in runs],
        "comparisons": comparisons,
        "behavior_exact": all(row["behavior_exact"] for row in comparisons),
        "median_pool_throughput_ratio": float(np.median([
            row["pool_throughput_ratio"] for row in comparisons
        ])),
        "median_cpu_per_simulated_second_ratio": float(np.median([
            row["cpu_per_simulated_second_ratio"] for row in comparisons
        ])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    trace_fields = (
        "position", "orientation_quat", "linear_velocity", "angular_velocity",
        "linear_acceleration", "encoder_left", "encoder_right", "lidar_ranges",
        "observation_lidar", "observation_state", "reward", "physics_ticks",
        "protocol_step_id", "sim_time_s", "action",
    )
    arrays = {}
    for run in runs:
        for env in run["runs"]:
            prefix = f"{run['label']}_r{run['repeat']}_env{env['worker_index']}"
            for field in trace_fields:
                try:
                    arrays[f"{prefix}_{field}"] = np.asarray([row[field] for row in env["trace"]])
                except (KeyError, ValueError, TypeError):
                    pass
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    if hashes_start != hashes_end:
        raise RuntimeError("a player payload changed during the benchmark")
    print(f"[lidar-pool] behavior_exact={report['behavior_exact']}; report={args.output}", flush=True)
    return 0 if report["behavior_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
