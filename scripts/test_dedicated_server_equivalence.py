"""Compare a normal fixed-step Linux player with its Dedicated Server build.

Run inside ``aicar_brain`` after staging both IL2CPP players. This test launches
one simulator at a time, with camera streaming disabled and an identical
scripted action trace, then compares bridge telemetry, full LiDAR, observations,
rewards, action timing, and physics tick counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from test_camera_fixed_interval import (
    COMPARISON_FIELDS,
    _field_diff,
    _run_one,
    _saved_reward_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _size_bytes(executable: Path) -> int:
    return sum(path.stat().st_size for path in executable.parent.rglob("*") if path.is_file())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k not in {"trace", "actions"}}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _run(
    *, label: str, simulator_path: Path, port: int, interval: float, seconds: float,
    map_id: str, seed: int, step_timeout: float, policy: Optional[Path],
    repeat_index: int, reward_config: Dict[str, Any], output: Path,
) -> Dict[str, Any]:
    return _run_one(
        label=label,
        camera_enabled=False,
        simulator_path=simulator_path,
        port=port,
        interval_s=interval,
        seconds=seconds,
        map_id=map_id,
        step_timeout=step_timeout,
        policy_path=policy,
        seed=seed,
        repeat_index=repeat_index,
        worker_index=0,
        reward_config=reward_config,
        simulator_log_path=output.with_name(f"{output.stem}_{label}.unity.log"),
    )


def _compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    trace_diff = _field_diff(reference["trace"], candidate["trace"])
    action_sequence_exact = bool(np.array_equal(reference["actions"], candidate["actions"]))
    ticks_per_action_equal = np.array_equal(
        [row["physics_ticks"] for row in reference["trace"]],
        [row["physics_ticks"] for row in candidate["trace"]],
    )
    protocol_ids_equal = np.array_equal(
        [row["protocol_step_id"] for row in reference["trace"]],
        [row["protocol_step_id"] for row in candidate["trace"]],
    )
    ref_times = np.asarray(
        [reference["simulated_time_first"]] + [row["sim_time_s"] for row in reference["trace"]],
        dtype=np.float64,
    )
    candidate_times = np.asarray(
        [candidate["simulated_time_first"]] + [row["sim_time_s"] for row in candidate["trace"]],
        dtype=np.float64,
    )
    ref_deltas = np.diff(ref_times)
    candidate_deltas = np.diff(candidate_times)
    sim_dt_max_abs_delta = (
        float(np.max(np.abs(ref_deltas - candidate_deltas)))
        if ref_deltas.size == candidate_deltas.size and ref_deltas.size
        else math.inf
    )
    exact_field_match = all(
        metrics["exact_value_mismatches"] == 0
        for metrics in trace_diff["fields"].values()
    )
    behavior_equivalent = bool(
        action_sequence_exact
        and len(reference["trace"]) == len(candidate["trace"])
        and ticks_per_action_equal
        and protocol_ids_equal
        and sim_dt_max_abs_delta == 0.0
        and exact_field_match
    )
    baseline_cpu = reference["unity_cpu_seconds_per_simulated_second"]
    candidate_cpu = candidate["unity_cpu_seconds_per_simulated_second"]
    cpu_saving_fraction = (
        (baseline_cpu - candidate_cpu) / baseline_cpu
        if baseline_cpu is not None and candidate_cpu is not None and baseline_cpu > 0
        else None
    )
    return {
        "behavior_equivalent_exact": behavior_equivalent,
        "action_sequence_exact": action_sequence_exact,
        "physics_ticks_per_action_equal": bool(ticks_per_action_equal),
        "protocol_step_ids_equal": bool(protocol_ids_equal),
        "per_action_sim_dt_max_abs_delta_s": sim_dt_max_abs_delta,
        "telemetry_observations_and_rewards": trace_diff,
        "baseline_cpu_seconds_per_simulated_second": baseline_cpu,
        "dedicated_server_cpu_seconds_per_simulated_second": candidate_cpu,
        "cpu_saving_fraction": cpu_saving_fraction,
        "aggregate_simulated_seconds_per_wall_second": {
            "normal_player": reference["simulated_seconds_per_wall_second"],
            "dedicated_server": candidate["simulated_seconds_per_wall_second"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-player", type=Path, required=True)
    parser.add_argument("--server-player", type=Path, required=True)
    parser.add_argument("--port-base", type=int, default=4680)
    parser.add_argument("--interval", type=float, default=0.086)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--step-timeout", type=float, default=15.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, default=Path("logs/diagnostics/dedicated_server_equivalence.json"))
    args = parser.parse_args()

    if args.interval <= 0 or args.seconds <= 0:
        parser.error("--interval and --seconds must be positive")
    for path in (args.baseline_player, args.server_player):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.policy is not None and not args.policy.is_file():
        raise FileNotFoundError(args.policy)

    reward_config = _saved_reward_config(args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 for a useful paired performance estimate")

    runs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        pair: Dict[str, Dict[str, Any]] = {}
        order = (
            [("normal_player", args.baseline_player), ("dedicated_server", args.server_player)]
            if repeat % 2 == 0
            else [("dedicated_server", args.server_player), ("normal_player", args.baseline_player)]
        )
        for order_index, (variant, simulator_path) in enumerate(order):
            label = f"{variant}_r{repeat + 1}"
            print(f"[server-ab] start {label} at {simulator_path}", flush=True)
            run = _run(
                label=label,
                simulator_path=simulator_path,
                port=args.port_base + repeat * 2 + order_index,
                interval=args.interval,
                seconds=args.seconds,
                map_id=args.map_id,
                seed=args.seed,
                step_timeout=args.step_timeout,
                policy=args.policy,
                repeat_index=repeat,
                reward_config=reward_config,
                output=args.output,
            )
            pair[variant] = run
            runs.append(run)
            print(
                f"[server-ab] done {label}: sim={run['simulated_seconds']:.3f}s "
                f"wall={run['wall_seconds']:.3f}s UnityCPU={run['unity_cpu_seconds']}s "
                f"CPU/sim-s={run['unity_cpu_seconds_per_simulated_second']}",
                flush=True,
            )
        comparison = _compare_pair(pair["normal_player"], pair["dedicated_server"])
        comparison["repeat_index"] = repeat + 1
        comparisons.append(comparison)
        print(
            f"[server-ab] repeat {repeat + 1}: equivalent={comparison['behavior_equivalent_exact']} "
            f"CPU savings={comparison['cpu_saving_fraction']}",
            flush=True,
        )

    behavior_equivalent = all(item["behavior_equivalent_exact"] for item in comparisons)
    cpu_savings = [item["cpu_saving_fraction"] for item in comparisons if item["cpu_saving_fraction"] is not None]
    baseline_cpu_values = [item["baseline_cpu_seconds_per_simulated_second"] for item in comparisons]
    candidate_cpu_values = [item["dedicated_server_cpu_seconds_per_simulated_second"] for item in comparisons]
    baseline_throughput = [item["aggregate_simulated_seconds_per_wall_second"]["normal_player"] for item in comparisons]
    candidate_throughput = [item["aggregate_simulated_seconds_per_wall_second"]["dedicated_server"] for item in comparisons]
    report = {
        "benchmark": "dedicated_server_fixed_interval_ab",
        "created_unix": time.time(),
        "configuration": {
            "baseline_player": str(args.baseline_player),
            "dedicated_server_player": str(args.server_player),
            "baseline_sha256": _sha256(args.baseline_player),
            "dedicated_server_sha256": _sha256(args.server_player),
            "baseline_build_bytes": _size_bytes(args.baseline_player),
            "dedicated_server_build_bytes": _size_bytes(args.server_player),
            "camera_enabled": False,
            "map_id": args.map_id,
            "interval_s": args.interval,
            "requested_simulated_seconds": args.seconds,
            "seed": args.seed,
            "policy": str(args.policy) if args.policy else None,
            "reward_config": reward_config,
        },
        "runs": [_json_safe(run) for run in runs],
        "comparison": {
            "behavior_equivalent_exact": behavior_equivalent,
            "paired_repeats": comparisons,
            "median_baseline_cpu_seconds_per_simulated_second": float(np.median(baseline_cpu_values)),
            "median_dedicated_server_cpu_seconds_per_simulated_second": float(np.median(candidate_cpu_values)),
            "median_cpu_saving_fraction": float(np.median(cpu_savings)),
            "median_normal_simulated_seconds_per_wall_second": float(np.median(baseline_throughput)),
            "median_server_simulated_seconds_per_wall_second": float(np.median(candidate_throughput)),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    arrays: Dict[str, np.ndarray] = {}
    for run in runs:
        prefix = run["label"]
        arrays[f"{prefix}_actions"] = np.asarray(run["actions"], dtype=np.float32)
        arrays[f"{prefix}_physics_ticks"] = np.asarray(
            [row["physics_ticks"] for row in run["trace"]], dtype=np.int64
        )
        for field in COMPARISON_FIELDS:
            arrays[f"{prefix}_{field}"] = np.asarray([row[field] for row in run["trace"]])
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(f"[server-ab] report: {args.output}", flush=True)
    print(
        f"[server-ab] behavior equivalent: {behavior_equivalent}; "
        f"median CPU savings={report['comparison']['median_cpu_saving_fraction']:.3%}",
        flush=True,
    )
    return 0 if behavior_equivalent else 2


if __name__ == "__main__":
    raise SystemExit(main())
