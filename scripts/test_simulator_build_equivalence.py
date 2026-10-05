"""Compare two staged fixed-step Linux players under identical controls."""

from __future__ import annotations

import argparse
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


def run_one(*, label: str, simulator: Path, port: int, repeat: int,
            seconds: float, interval: float, map_id: str, seed: int,
            worker_index: int = 0) -> dict[str, Any]:
    prior_camera = os.environ.get("AICAR_DISABLE_CAMERA_STREAM")
    prior_fps = os.environ.get("AICAR_ACTION_IDLE_TARGET_FPS")
    os.environ["AICAR_DISABLE_CAMERA_STREAM"] = "1"
    os.environ["AICAR_ACTION_IDLE_TARGET_FPS"] = "30"
    try:
        return _run_one(
            label=label, camera_enabled=False, simulator_path=simulator,
            port=port, interval_s=interval, seconds=seconds, map_id=map_id,
            step_timeout=15.0, policy_path=None, seed=seed,
            repeat_index=repeat, worker_index=worker_index,
        )
    finally:
        for key, value in (("AICAR_DISABLE_CAMERA_STREAM", prior_camera),
                           ("AICAR_ACTION_IDLE_TARGET_FPS", prior_fps)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_diff": _field_diff(a["trace"], b["trace"]),
        "actions_equal": np.array_equal(a["actions"], b["actions"]),
        "physics_ticks_equal": np.array_equal(
            [r["physics_ticks"] for r in a["trace"]],
            [r["physics_ticks"] for r in b["trace"]],
        ),
        "protocol_step_ids_equal": np.array_equal(
            [r["protocol_step_id"] for r in a["trace"]],
            [r["protocol_step_id"] for r in b["trace"]],
        ),
        "simulated_seconds_delta": abs(a["simulated_seconds"] - b["simulated_seconds"]),
        "cpu_saving_fraction": 1.0 - b["unity_cpu_seconds_per_simulated_second"]
        / a["unity_cpu_seconds_per_simulated_second"],
        "throughput_ratio": b["simulated_seconds_per_wall_second"]
        / a["simulated_seconds_per_wall_second"],
    }


def json_safe(value: Any) -> Any:
    """Represent non-finite diff metrics as strings in strict JSON reports."""
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-simulator-path", type=Path, required=True)
    parser.add_argument("--candidate-simulator-path", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.086)
    parser.add_argument("--port-base", type=int, default=4940)
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.seconds <= 0 or args.interval <= 0:
        parser.error("repeats must be >=2 and duration/interval must be positive")
    paths = {"reference": args.reference_simulator_path.resolve(),
             "candidate": args.candidate_simulator_path.resolve()}
    hashes_start = {key: _critical_artifact_hashes(path) for key, path in paths.items()}
    runs = []
    comparisons = []
    for repeat in range(args.repeats):
        order = ["reference", "candidate"] if repeat % 2 == 0 else ["candidate", "reference"]
        pair = {}
        for label in order:
            print(f"[build-ab] start {label} repeat={repeat + 1}", flush=True)
            run = run_one(
                label=f"{label}-r{repeat + 1}", simulator=paths[label],
                port=args.port_base + len(runs), repeat=repeat,
                seconds=args.seconds, interval=args.interval,
                map_id=args.map_id, seed=args.seed,
            )
            runs.append(run)
            pair[label] = run
            print(
                f"[build-ab] done {label}: CPU/sim-s="
                f"{run['unity_cpu_seconds_per_simulated_second']:.4f}, "
                f"sim-s/wall-s={run['simulated_seconds_per_wall_second']:.4f}",
                flush=True,
            )
        comparisons.append({"repeat": repeat + 1,
                            **compare(pair["reference"], pair["candidate"])})
    hashes_end = {key: _critical_artifact_hashes(path) for key, path in paths.items()}
    exact = all(
        c["actions_equal"] and c["physics_ticks_equal"] and c["protocol_step_ids_equal"]
        and c["simulated_seconds_delta"] == 0
        and all(v["exact_value_mismatches"] == 0 for v in c["trace_diff"]["fields"].values())
        for c in comparisons
    )
    report = {
        "benchmark": "fixed_step_staged_player_ab",
        "created_unix": time.time(),
        "simulator_paths": {k: str(v) for k, v in paths.items()},
        "critical_artifact_hashes_start": hashes_start,
        "critical_artifact_hashes_end": hashes_end,
        "configuration": {
            "repeats": args.repeats, "seconds": args.seconds,
            "interval_s": args.interval, "map_id": args.map_id,
            "seed": args.seed, "camera_enabled": False,
            "idle_target_fps": 30,
        },
        "runs": [{k: v for k, v in r.items() if k not in ("trace", "actions")} for r in runs],
        "comparisons": comparisons,
        "behavior_exact": exact,
        "mean_reference_cpu_seconds_per_simulated_second": float(np.mean([
            r["unity_cpu_seconds_per_simulated_second"] for r in runs if r["label"].startswith("reference")
        ])),
        "mean_candidate_cpu_seconds_per_simulated_second": float(np.mean([
            r["unity_cpu_seconds_per_simulated_second"] for r in runs if r["label"].startswith("candidate")
        ])),
        "median_cpu_saving_fraction": float(np.median([c["cpu_saving_fraction"] for c in comparisons])),
        "median_throughput_ratio": float(np.median([c["throughput_ratio"] for c in comparisons])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    arrays = {}
    for run in runs:
        prefix = run["label"].replace("-", "_")
        arrays[f"{prefix}_sim_time"] = np.asarray([r["sim_time_s"] for r in run["trace"]])
        arrays[f"{prefix}_physics_ticks"] = np.asarray([r["physics_ticks"] for r in run["trace"]])
        for field in COMPARISON_FIELDS:
            try:
                arrays[f"{prefix}_{field}"] = np.asarray([r[field] for r in run["trace"]])
            except (TypeError, ValueError):
                pass
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    if hashes_start != hashes_end:
        raise RuntimeError("a staged player changed while the benchmark was running")
    print(f"[build-ab] behavior_exact={exact}; report={args.output}", flush=True)
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
