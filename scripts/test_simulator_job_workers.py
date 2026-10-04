"""Compare Unity job-worker limits with identical fixed-step pool traces."""

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


def set_job_worker_count(boot_config: Path, count: int) -> None:
    lines = boot_config.read_text(encoding="utf-8").splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith("job-worker-count=")]
    if len(matches) > 1:
        raise RuntimeError(f"multiple job-worker-count settings in {boot_config}")
    setting = f"job-worker-count={count}"
    if matches:
        lines[matches[0]] = setting
    else:
        lines.append(setting)
    boot_config.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_pool_worker(job: dict[str, Any]) -> dict[str, Any]:
    return run_one(
        label=job["label"], simulator=Path(job["simulator_path"]),
        port=job["port"], repeat=job["repeat"], seconds=job["seconds"],
        interval=job["interval"], map_id=job["map_id"], seed=job["seed"],
        worker_index=job["worker_index"],
    )


def run_pool(*, label: str, simulator: Path, workers: int, envs: int,
             repeat: int, seconds: float, interval: float,
             port_base: int, seed: int, map_id: str) -> dict[str, Any]:
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
    print(f"[job-workers] start {label} repeat={repeat + 1} envs={envs}", flush=True)
    wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(_run_pool_worker, jobs))
    wall_seconds = time.perf_counter() - wall_start
    simulated_seconds = sum(run["simulated_seconds"] for run in runs)
    cpu_seconds = sum(run["unity_cpu_seconds"] or 0.0 for run in runs)
    summary = {
        "label": label,
        "repeat": repeat + 1,
        "job_worker_count": workers,
        "envs": envs,
        "actions_per_env": [run["action_count"] for run in runs],
        "wall_seconds": wall_seconds,
        "simulated_env_seconds": simulated_seconds,
        "unity_cpu_seconds": cpu_seconds,
        "pool_simulated_env_seconds_per_wall_second": (
            simulated_seconds / wall_seconds if wall_seconds else None
        ),
        "unity_cpu_cores": cpu_seconds / wall_seconds if wall_seconds else None,
        "unity_cpu_seconds_per_simulated_second": (
            cpu_seconds / simulated_seconds if simulated_seconds else None
        ),
        "runs": runs,
    }
    print(
        f"[job-workers] done {label}: "
        f"pool sim-s/wall-s={summary['pool_simulated_env_seconds_per_wall_second']:.4f}, "
        f"Unity CPU cores={summary['unity_cpu_cores']:.3f}", flush=True,
    )
    return summary


def exact_pair(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if len(reference["runs"]) != len(candidate["runs"]):
        raise RuntimeError("paired pool sizes differ")
    per_env = []
    for ref, cand in zip(reference["runs"], candidate["runs"]):
        check = compare(ref, cand)
        per_env.append({
            "env": ref["worker_index"],
            "actions_equal": check["actions_equal"],
            "physics_ticks_equal": check["physics_ticks_equal"],
            "protocol_step_ids_equal": check["protocol_step_ids_equal"],
            "simulated_seconds_delta": check["simulated_seconds_delta"],
            "fields": check["trace_diff"]["fields"],
        })
    exact = all(
        row["actions_equal"] and row["physics_ticks_equal"]
        and row["protocol_step_ids_equal"] and row["simulated_seconds_delta"] == 0
        and all(field["exact_value_mismatches"] == 0 for field in row["fields"].values())
        for row in per_env
    )
    return {"behavior_exact": exact, "environments": per_env}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-path", type=Path, required=True)
    parser.add_argument("--reference-workers", type=int, default=1)
    parser.add_argument("--candidate-workers", type=int, default=2)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.086)
    parser.add_argument("--port-base", type=int, default=4980)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.reference_workers, args.candidate_workers, args.envs, args.repeats) < 1:
        parser.error("worker counts, env count, and repeats must be positive")
    if args.reference_workers == args.candidate_workers:
        parser.error("reference and candidate worker counts must differ")
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("seconds and interval must be positive")

    executable = args.simulator_path.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    boot_config = executable.parent / "AutoDRIVE Simulator_Data" / "boot.config"
    original_config = boot_config.read_text(encoding="utf-8")
    payload_hashes_start = _critical_artifact_hashes(executable)
    original_workers = [
        int(line.partition("=")[2]) for line in original_config.splitlines()
        if line.startswith("job-worker-count=")
    ]
    if len(original_workers) != 1:
        raise RuntimeError(f"expected one job-worker-count in {boot_config}")
    if original_workers[0] != args.reference_workers:
        raise RuntimeError(
            f"staged player is configured for {original_workers[0]} workers, "
            f"not the expected reference value {args.reference_workers}"
        )

    runs = []
    comparisons = []
    try:
        for repeat in range(args.repeats):
            order = ["reference", "candidate"] if repeat % 2 == 0 else ["candidate", "reference"]
            pair = {}
            for label in order:
                worker_count = (
                    args.reference_workers if label == "reference" else args.candidate_workers
                )
                set_job_worker_count(boot_config, worker_count)
                pool_run = run_pool(
                    label=label, simulator=executable, workers=args.envs,
                    envs=args.envs, repeat=repeat, seconds=args.seconds,
                    interval=args.interval, port_base=args.port_base + len(runs) * args.envs,
                    seed=args.seed, map_id=args.map_id,
                )
                runs.append(pool_run)
                pair[label] = pool_run
            comparisons.append({
                "repeat": repeat + 1,
                **exact_pair(pair["reference"], pair["candidate"]),
                "pool_throughput_ratio": (
                    pair["candidate"]["pool_simulated_env_seconds_per_wall_second"]
                    / pair["reference"]["pool_simulated_env_seconds_per_wall_second"]
                ),
                "cpu_core_ratio": (
                    pair["candidate"]["unity_cpu_cores"] / pair["reference"]["unity_cpu_cores"]
                ),
                "cpu_per_simulated_second_ratio": (
                    pair["candidate"]["unity_cpu_seconds_per_simulated_second"]
                    / pair["reference"]["unity_cpu_seconds_per_simulated_second"]
                ),
            })
    finally:
        boot_config.write_text(original_config, encoding="utf-8")

    all_exact = all(pair["behavior_exact"] for pair in comparisons)
    payload_hashes_end = _critical_artifact_hashes(executable)
    if payload_hashes_start != payload_hashes_end:
        raise RuntimeError("player payload changed during the worker-count experiment")
    report = {
        "benchmark": "fixed_step_unity_job_worker_count_pool_ab",
        "created_unix": time.time(),
        "simulator_path": str(executable),
        "boot_config_restored": boot_config.read_text(encoding="utf-8") == original_config,
        "critical_payload_hashes_unchanged": payload_hashes_start == payload_hashes_end,
        "configuration": {
            "reference_workers": args.reference_workers,
            "candidate_workers": args.candidate_workers,
            "envs": args.envs,
            "repeats": args.repeats,
            "seconds_per_env": args.seconds,
            "interval_s": args.interval,
            "map_id": args.map_id,
            "seed": args.seed,
        },
        "runs": [{k: v for k, v in run.items() if k != "runs"} for run in runs],
        "comparisons": comparisons,
        "behavior_exact": all_exact,
        "median_pool_throughput_ratio": float(np.median([
            row["pool_throughput_ratio"] for row in comparisons
        ])),
        "median_cpu_core_ratio": float(np.median([
            row["cpu_core_ratio"] for row in comparisons
        ])),
        "median_cpu_per_simulated_second_ratio": float(np.median([
            row["cpu_per_simulated_second_ratio"] for row in comparisons
        ])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    trace_arrays = {}
    for run in runs:
        for env in run["runs"]:
            prefix = f"{run['label']}_r{run['repeat']}_env{env['worker_index']}"
            for field in ("position", "orientation_quat", "linear_velocity", "angular_velocity",
                          "linear_acceleration", "encoder_left", "encoder_right", "lidar_ranges",
                          "observation_lidar", "observation_state", "reward", "physics_ticks",
                          "protocol_step_id", "sim_time_s", "action"):
                try:
                    trace_arrays[f"{prefix}_{field}"] = np.asarray([row[field] for row in env["trace"]])
                except (KeyError, ValueError, TypeError):
                    pass
    np.savez_compressed(args.output.with_suffix(".npz"), **trace_arrays)
    print(f"[job-workers] behavior_exact={all_exact}; report={args.output}", flush=True)
    return 0 if all_exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
