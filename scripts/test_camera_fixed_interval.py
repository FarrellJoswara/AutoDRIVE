"""Benchmark camera readback suppression with fixed simulated action intervals.

This is an opt-in integration benchmark, not a training entry point. It launches
one Unity player at a time, first using an identical scripted action trace and
optionally using the saved newmodel3 policy. Run it inside ``aicar_brain`` once
the simulator and Racer fixed-interval protocol are ready. It never launches a
simulator merely by being imported or by asking for ``--help``.

Example (inside aicar_brain; confirm the staged path/port with the operator):
    python scripts/test_camera_fixed_interval.py \
      --simulator-path '/app/simulator/_build/linux-cpu-experiment/AutoDRIVE Simulator.x86_64' \
      --port-base 4584 --interval 0.086 --seconds 20 --repeats 3 \
      --model logs/rl/newmodel3/best_evaluated_model.zip

Camera capture is enabled unless ``AICAR_DISABLE_CAMERA_STREAM=1``. The
benchmark flips that variable only for the launched simulator process and
restores the caller's environment after each run. Simulations are sequential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # Keep --help usable from a host without the training stack.
    np = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TELEMETRY_FIELDS: Tuple[str, ...] = (
    "position",
    "orientation_quat",
    "linear_velocity",
    "angular_velocity",
    "linear_acceleration",
    "encoder_left",
    "encoder_right",
    "encoder_ticks_left",
    "encoder_ticks_right",
    "lidar_ranges",
    "lidar_scan_rate",
    "lidar_range_min",
    "lidar_range_max",
    "throttle",
    "steering",
    "lap_count",
    "lap_time",
    "last_lap_time",
    "best_lap_time",
    "collision",
    "collision_count",
    "true_speed",
    "heading_yaw",
    "v_long",
    "v_lat",
    "slip_angle",
    "lateral_g",
)
COMPARISON_FIELDS: Tuple[str, ...] = TELEMETRY_FIELDS + (
    "observation_lidar",
    "observation_state",
    "reward",
)


def fixed_actions(count: int) -> np.ndarray:
    """Repeatable steering/throttle excitation shared by every open-loop run."""
    # Long enough straight to acquire speed, then matched left/right arcs and a
    # coast phase. Keep controls identical by index; never adapt to observations.
    cycle = np.asarray(
        [
            [0.28, 0.00],
            [0.36, 0.10],
            [0.36, -0.10],
            [0.18, 0.00],
        ],
        dtype=np.float32,
    )
    block = 30
    return np.stack([cycle[(i // block) % len(cycle)] for i in range(count)])


def _proc_cpu_seconds(pid: Optional[int]) -> Optional[float]:
    """Read one Unity process's user+system CPU time from Linux /proc."""
    if pid is None or not Path(f"/proc/{pid}/stat").is_file():
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        # comm may contain spaces/parentheses; split only after its final ')'.
        fields = raw[raw.rfind(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])  # stat fields 14 (utime), 15 (stime)
        return ticks / float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _critical_artifact_hashes(simulator_path: Path) -> Dict[str, str]:
    """Hash launcher and Unity payload files that must stay fixed during A/B."""
    data_dir = simulator_path.parent / "AutoDRIVE Simulator_Data"
    critical = {
        "launcher": simulator_path,
        "GameAssembly": simulator_path.parent / "GameAssembly.so",
        "globalgamemanagers": data_dir / "globalgamemanagers",
    }
    missing = [str(path) for path in critical.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing staged simulator payload(s): " + ", ".join(missing))
    return {name: _sha256(path) for name, path in critical.items()}


def _snapshot_row(snapshot: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for name in TELEMETRY_FIELDS:
        value = getattr(snapshot, name)
        if name == "lidar_ranges":
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if not bool(getattr(snapshot, "lidar_valid", False)):
                raise RuntimeError("Bridge returned a missing/invalid LiDAR scan")
            if arr.size != 1081:
                raise RuntimeError(f"Expected full 1081-beam LiDAR; received {arr.size}")
            row[name] = arr.copy()
        elif isinstance(value, (tuple, list, np.ndarray)):
            row[name] = np.asarray(value).copy()
        elif isinstance(value, (bool, int, float, np.generic)):
            row[name] = value.item() if isinstance(value, np.generic) else value
        else:
            row[name] = value
    return row


def _metadata(racer: Any) -> Dict[str, Any]:
    """Collect integration metadata exposed by the fixed-interval Racer API."""
    interval = getattr(racer, "action_interval_s", None)
    sim_time_fn = getattr(racer, "simulation_time", None)
    ticks = getattr(racer, "physics_ticks", None)
    if interval is None or not callable(sim_time_fn) or ticks is None:
        raise RuntimeError(
            "Racer lacks fixed-interval metadata: expected action_interval_s, "
            "simulation_time(), and physics_ticks after each response"
        )
    return {
        "step_counter": int(getattr(racer, "step_counter", 0)),
        "sim_time_s": float(sim_time_fn()),
        "physics_ticks": int(ticks),
    }


def _set_camera_env(enabled: bool) -> Optional[str]:
    """Set inherited child env for one launched process; return prior value."""
    key = "AICAR_DISABLE_CAMERA_STREAM"
    previous = os.environ.get(key)
    if enabled:
        os.environ.pop(key, None)
    else:
        os.environ[key] = "1"
    return previous


def _restore_camera_env(previous: Optional[str]) -> None:
    key = "AICAR_DISABLE_CAMERA_STREAM"
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def _create_env(
    *, simulator_path: Path, port: int, interval_s: float, camera_enabled: bool,
    map_id: str, step_timeout: float, reward_config: Any = None,
    simulator_log_path: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """Launch one simulator and wrap its Racer in the normal Layer 2 env."""
    from src.layer1.racer import Racer
    from src.layer2.autodrive_env import AutoDriveEnv

    old_camera_env = _set_camera_env(camera_enabled)
    old_map_env = os.environ.get("AICAR_MAP_ID")
    os.environ["AICAR_MAP_ID"] = map_id
    try:
        racer = Racer(
            racer_id=port,
            port=port,
            simulator_path=simulator_path,
            auto_launch=True,
            headless=True,
            step_timeout=step_timeout,
            enable_logging=False,
            action_interval_s=interval_s,
            simulator_log_path=simulator_log_path,
        )
    except BaseException:
        _restore_camera_env(old_camera_env)
        raise
    finally:
        # The child has inherited its environment. Restore immediately so one
        # mode cannot leak into the next run or into the caller's shell.
        _restore_camera_env(old_camera_env)
        if old_map_env is None:
            os.environ.pop("AICAR_MAP_ID", None)
        else:
            os.environ["AICAR_MAP_ID"] = old_map_env

    env = AutoDriveEnv(
        racer=racer,
        auto_launch=False,
        headless=True,
        frame_skip=1,
        map_id=map_id,
        laps_per_episode=0,
        terminate_on_collision=False,
        frontier_stagnation_seconds=0.0,
        reward_config=reward_config,
    )
    return racer, env


def _run_one(
    *, label: str, camera_enabled: bool, simulator_path: Path, port: int,
    interval_s: float, seconds: float, map_id: str, step_timeout: float,
    policy_path: Optional[Path], seed: int, repeat_index: int = 0, worker_index: int = 0,
    reward_config: Optional[Dict[str, Any]] = None,
    simulator_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    # Number of actions is common for every paired run; the fixed interval and
    # returned Unity clock/tick metadata establish whether horizons matched.
    action_count = max(1, int(round(seconds / interval_s)))
    scripted = fixed_actions(action_count)
    racer = env = None
    trace: List[Dict[str, Any]] = []
    action_trace: List[np.ndarray] = []
    try:
        from src.layer2.rewards import RewardConfig
        env_reward = RewardConfig(**(reward_config or {}))
        racer, env = _create_env(
            simulator_path=simulator_path,
            port=port,
            interval_s=interval_s,
            camera_enabled=camera_enabled,
            map_id=map_id,
            step_timeout=step_timeout,
            reward_config=env_reward,
            simulator_log_path=simulator_log_path,
        )
        observation, reset_info = env.reset(seed=seed)
        model = None
        if policy_path is not None:
            from stable_baselines3 import PPO
            model = PPO.load(str(policy_path), device="cpu")
        sim_start = float(racer.simulation_time())
        ticks_start = int(racer.physics_ticks)
        cpu_start = _proc_cpu_seconds(getattr(racer._sim_process, "pid", None))
        wall_start = time.perf_counter()

        for index in range(action_count):
            if model is None:
                action = scripted[index].copy()
            else:
                action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(2)
            observation, reward, terminated, truncated, info = env.step(action)
            snap = racer.telemetry
            meta = _metadata(racer)
            row = _snapshot_row(snap)
            row.update(meta)
            commanded_action = np.asarray(
                [info["throttle_command"], info["steering_command"]], dtype=np.float32
            )
            row["action"] = commanded_action.copy()
            row["observation_lidar"] = np.asarray(observation["lidar"], dtype=np.float32).copy()
            row["observation_state"] = np.asarray(observation["state"], dtype=np.float32).copy()
            row["reward"] = float(reward)
            row["terminated"] = bool(terminated)
            row["truncated"] = bool(truncated)
            row["frontier_progress_m"] = info.get("frontier_progress_m")
            row["frontier_advanced_m"] = info.get("frontier_advanced_m")
            row["episode_won"] = bool(info.get("episode_won", False))
            row["protocol_step_id"] = int(snap.step_id)
            bridge_keys = set(getattr(racer, "latest_bridge_keys", ()))
            camera_keys = sorted(key for key in bridge_keys if "Camera Image" in key)
            row["camera_payload_keys"] = camera_keys
            if bool(camera_enabled) != bool(camera_keys):
                raise RuntimeError(
                    f"camera mode {label} did not match Bridge camera payload keys "
                    f"at action {index}: {camera_keys}"
                )
            if trace:
                expected_step_id = int(trace[-1]["protocol_step_id"]) + 1
                if int(snap.step_id) != expected_step_id:
                    raise RuntimeError(
                        f"protocol step IDs skipped/repeated: expected {expected_step_id}, got {snap.step_id}"
                    )
            trace.append(row)
            action_trace.append(commanded_action.copy())

        wall_elapsed = time.perf_counter() - wall_start
        cpu_end = _proc_cpu_seconds(getattr(racer._sim_process, "pid", None))
        sim_end = float(racer.simulation_time())
        ticks_end = int(racer.physics_ticks)
        cpu_seconds = (
            None if cpu_start is None or cpu_end is None else max(0.0, cpu_end - cpu_start)
        )
        sim_elapsed = max(0.0, sim_end - sim_start)
        expected_delta = interval_s
        per_step_deltas = np.diff(
            np.asarray([sim_start] + [float(row["sim_time_s"]) for row in trace], dtype=np.float64)
        )
        if per_step_deltas.size and np.any(per_step_deltas <= 0.0):
            raise RuntimeError("Unity simulated clock did not advance monotonically per action")
        return {
            "label": label,
            "repeat_index": int(repeat_index),
            "worker_index": int(worker_index),
            "camera_enabled": bool(camera_enabled),
            "policy": str(policy_path) if policy_path else None,
            "simulator_log_path": str(simulator_log_path) if simulator_log_path else None,
            "interval_requested_s": float(interval_s),
            "action_count": action_count,
            "simulated_seconds": sim_elapsed,
            "simulated_time_first": sim_start,
            "simulated_time_last": sim_end,
            "physics_ticks_delta": int(sum(int(row["physics_ticks"]) for row in trace)),
            "physics_ticks_first": ticks_start,
            "physics_ticks_per_response": sorted({int(row["physics_ticks"]) for row in trace}),
            "physics_ticks_last": ticks_end,
            "protocol_step_id_first": int(trace[0]["protocol_step_id"]) if trace else None,
            "protocol_step_id_last": int(trace[-1]["protocol_step_id"]) if trace else None,
            "camera_payload_response_count": sum(bool(row["camera_payload_keys"]) for row in trace),
            "camera_payload_key_examples": sorted({key for row in trace for key in row["camera_payload_keys"]}),
            "per_action_sim_dt_mean_s": float(np.mean(per_step_deltas)) if per_step_deltas.size else None,
            "per_action_sim_dt_max_abs_error_s": (
                float(np.max(np.abs(per_step_deltas - expected_delta))) if per_step_deltas.size else None
            ),
            "wall_seconds": wall_elapsed,
            "parallel_batch_wall_seconds": None,
            "simulated_seconds_per_wall_second": sim_elapsed / wall_elapsed if wall_elapsed else None,
            "unity_cpu_seconds": cpu_seconds,
            "unity_cpu_seconds_per_simulated_second": (
                cpu_seconds / sim_elapsed if cpu_seconds is not None and sim_elapsed > 0 else None
            ),
            "unity_average_cpu_percent_one_core": (
                100.0 * cpu_seconds / wall_elapsed if cpu_seconds is not None and wall_elapsed else None
            ),
            "final_position": [float(x) for x in racer.telemetry.position],
            "final_speed_mps": float(racer.telemetry.true_speed),
            "collision_count_final": int(racer.telemetry.collision_count),
            "collision_flag_final": bool(racer.telemetry.collision),
            "frontier_progress_final_m": trace[-1].get("frontier_progress_m") if trace else None,
            "frontier_advanced_total_m": float(sum(float(row.get("frontier_advanced_m") or 0.0) for row in trace)),
            "trace": trace,
            "actions": action_trace,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if racer is not None:
            try:
                racer.kill()
            except Exception:
                pass


def _field_diff(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    count = min(len(a), len(b))
    result: Dict[str, Any] = {"steps_compared": count, "length_a": len(a), "length_b": len(b)}
    if len(a) != len(b):
        result["length_mismatch"] = True
    fields: Dict[str, Any] = {}
    for name in COMPARISON_FIELDS:
        exact_mismatches = 0
        total_values = 0
        max_abs = 0.0
        finite_pairs = 0
        for i in range(count):
            av = np.asarray(a[i][name])
            bv = np.asarray(b[i][name])
            if av.shape != bv.shape:
                exact_mismatches += max(av.size, bv.size, 1)
                total_values += max(av.size, bv.size, 1)
                max_abs = math.inf
                continue
            if np.issubdtype(av.dtype, np.bool_) or np.issubdtype(av.dtype, np.integer):
                equal = av == bv
                exact_mismatches += int(np.count_nonzero(~equal))
                total_values += int(av.size)
                if av.size:
                    max_abs = max(max_abs, float(np.max(np.not_equal(av, bv))))
                continue
            avf = av.astype(np.float64, copy=False)
            bvf = bv.astype(np.float64, copy=False)
            equal = (avf == bvf) | (np.isnan(avf) & np.isnan(bvf))
            exact_mismatches += int(np.count_nonzero(~equal))
            total_values += int(avf.size)
            finite = np.isfinite(avf) & np.isfinite(bvf)
            finite_pairs += int(np.count_nonzero(finite))
            if np.any(finite):
                max_abs = max(max_abs, float(np.max(np.abs(avf[finite] - bvf[finite]))))
            # Different infinity/NaN patterns count as unequal and infinite error.
            if np.any(~finite & ~equal):
                max_abs = math.inf
        fields[name] = {
            "exact_value_mismatches": exact_mismatches,
            "value_count": total_values,
            "exact_match_fraction": (1.0 - exact_mismatches / total_values) if total_values else 1.0,
            "max_abs_error": max_abs,
            "finite_pairs": finite_pairs,
        }
    result["fields"] = fields
    return result


def _worker_entry(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """ProcessPool-safe wrapper that transports failures back to the parent."""
    try:
        return _run_one(**kwargs)
    except BaseException as exc:
        import traceback
        return {
            "_error": f"{type(exc).__name__}: {exc}",
            "_traceback": traceback.format_exc(),
            "label": kwargs.get("label"),
            "worker_index": kwargs.get("worker_index"),
        }


def _saved_reward_config(model_path: Optional[Path]) -> Dict[str, Any]:
    """Reuse the policy's saved reward weights when its training config exists."""
    if model_path is None:
        return {}
    config_path = model_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Policy run config is required for a comparable evaluation: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env_kwargs = config.get("env_kwargs", {})
    from src.layer2.rewards import RewardConfig
    fields = set(RewardConfig.__dataclass_fields__)
    return {key: value for key, value in env_kwargs.items() if key in fields}


def _compare_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[bool, bool, int], List[Dict[str, Any]]] = {}
    for run in runs:
        key = (bool(run["camera_enabled"]), run["policy"] is not None,
               int(run.get("worker_index", 0)))
        grouped.setdefault(key, []).append(run)

    def dt(run: Dict[str, Any]) -> np.ndarray:
        times = [run["simulated_time_first"]] + [float(row["sim_time_s"]) for row in run["trace"]]
        return np.diff(np.asarray(times, dtype=np.float64))

    def compare(kind: str, a: Dict[str, Any], b: Dict[str, Any], repeat: int) -> Dict[str, Any]:
        a_dt, b_dt = dt(a), dt(b)
        shared = min(len(a_dt), len(b_dt))
        return {
            "kind": kind,
            "policy_evaluation": a["policy"] is not None,
            "worker_index": int(a.get("worker_index", 0)),
            "repeat_index": repeat,
            "reference": a["label"],
            "other": b["label"],
            "telemetry_and_observations": _field_diff(a["trace"], b["trace"]),
            "action_sequence_exact": bool(np.array_equal(a["actions"], b["actions"])),
            "simulated_time_abs_delta_s": abs(a["simulated_seconds"] - b["simulated_seconds"]),
            "physics_tick_abs_delta": abs(a["physics_ticks_delta"] - b["physics_ticks_delta"]),
            "physics_ticks_per_action_equal": bool(np.array_equal(
                [row["physics_ticks"] for row in a["trace"]],
                [row["physics_ticks"] for row in b["trace"]],
            )),
            "per_action_sim_dt_max_abs_delta_s": float(np.max(np.abs(a_dt[:shared] - b_dt[:shared]))) if shared else None,
            "protocol_step_ids_equal": bool(np.array_equal(
                [row["protocol_step_id"] for row in a["trace"]],
                [row["protocol_step_id"] for row in b["trace"]],
            )),
            "camera_payload_keys_equal": (
                len(a["trace"]) == len(b["trace"])
                and all(x["camera_payload_keys"] == y["camera_payload_keys"]
                        for x, y in zip(a["trace"], b["trace"]))
            ),
            "unity_cpu_seconds_per_simulated_second_delta": (
                None if a["unity_cpu_seconds_per_simulated_second"] is None
                or b["unity_cpu_seconds_per_simulated_second"] is None
                else a["unity_cpu_seconds_per_simulated_second"] - b["unity_cpu_seconds_per_simulated_second"]
            ),
        }

    comparisons: List[Dict[str, Any]] = []
    for (camera_enabled, _, _), group in grouped.items():
        group = sorted(group, key=lambda run: int(run.get("repeat_index", 0)))
        for a, b in zip(group, group[1:]):
            item = compare("same_mode_repeat", a, b, int(b.get("repeat_index", 0)))
            item["camera_enabled"] = camera_enabled
            comparisons.append(item)

    policies = sorted({is_policy for _, is_policy, _ in grouped})
    workers = sorted({worker for _, _, worker in grouped})
    for is_policy in policies:
        for worker in workers:
            off = {int(r.get("repeat_index", 0)): r for r in grouped.get((False, is_policy, worker), [])}
            on = {int(r.get("repeat_index", 0)): r for r in grouped.get((True, is_policy, worker), [])}
            for repeat in sorted(set(off) & set(on)):
                comparisons.append(compare("camera_off_vs_on", off[repeat], on[repeat], repeat))

    batches: Dict[Tuple[bool, bool, int], List[Dict[str, Any]]] = {}
    for run in runs:
        key = (bool(run["camera_enabled"]), run["policy"] is not None,
               int(run.get("repeat_index", 0)))
        batches.setdefault(key, []).append(run)
    pool_batches = []
    for (camera_enabled, is_policy, repeat), batch in sorted(batches.items()):
        sim_s = sum(float(run["simulated_seconds"]) for run in batch)
        cpu_values = [run["unity_cpu_seconds"] for run in batch]
        cpu_s = None if any(v is None for v in cpu_values) else sum(float(v) for v in cpu_values)
        wall_s = max([float(r["parallel_batch_wall_seconds"]) for r in batch
                      if r.get("parallel_batch_wall_seconds") is not None]
                     or [float(r["wall_seconds"]) for r in batch])
        pool_batches.append({
            "camera_enabled": camera_enabled,
            "policy_evaluation": is_policy,
            "repeat_index": repeat,
            "parallel_envs": len(batch),
            "aggregated_simulated_seconds": sim_s,
            "aggregated_unity_cpu_seconds": cpu_s,
            "aggregated_cpu_seconds_per_simulated_second": cpu_s / sim_s if cpu_s is not None and sim_s else None,
            "pool_simulated_seconds_per_wall_second": sim_s / wall_s if wall_s else None,
            "pool_wall_seconds": wall_s,
        })
    return {"comparisons": comparisons, "parallel_pool_batches": pool_batches}


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


def _save_report(path: Path, runs: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "camera_fixed_interval",
        "created_unix": time.time(),
        "configuration": {
            "simulator_path": str(args.simulator_path),
            "simulator_sha256": args.simulator_sha256,
            "critical_artifact_hashes_start": args.critical_artifact_hashes_start,
            "critical_artifact_hashes_end": getattr(args, "critical_artifact_hashes_end", None),
            "map_id": args.map_id,
            "interval_s": args.interval,
            "requested_simulated_seconds": args.seconds,
            "repeats": args.repeats,
            "parallel_envs": args.parallel_envs,
            "seed": args.seed,
            "model": str(args.model) if args.model else None,
            "reward_config": _saved_reward_config(args.model),
        },
        "runs": [_json_safe(run) for run in runs],
        **_compare_runs(runs),
    }
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    # Retain exact per-step vectors, especially full 1081-ray LiDAR, separately.
    arrays: Dict[str, np.ndarray] = {}
    for run in runs:
        prefix = run["label"].replace("-", "_")
        arrays[f"{prefix}_actions"] = np.asarray(run["actions"], dtype=np.float32)
        arrays[f"{prefix}_sim_time"] = np.asarray([r["sim_time_s"] for r in run["trace"]], dtype=np.float64)
        arrays[f"{prefix}_physics_ticks"] = np.asarray([r["physics_ticks"] for r in run["trace"]], dtype=np.int64)
        for field in COMPARISON_FIELDS:
            dtype = np.float32 if field == "lidar_ranges" else None
            try:
                arrays[f"{prefix}_{field}"] = np.asarray([r[field] for r in run["trace"]], dtype=dtype)
            except (TypeError, ValueError):
                pass
    np.savez_compressed(path.with_suffix(".npz"), **arrays)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-path", type=Path, required=True, help="Staged Unity player executable")
    parser.add_argument("--port-base", type=int, default=4584, help="First local bridge port; runs increment it")
    parser.add_argument("--interval", type=float, default=0.086, help="Requested simulated action duration in seconds")
    parser.add_argument("--seconds", type=float, default=20.0, help="Same action-count/simulated-time horizon per run")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per camera mode for baseline variability")
    parser.add_argument("--parallel-envs", type=int, default=1, help="Independent Unity workers to run concurrently")
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--step-timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--model", type=Path, help="Optional deterministic PPO policy for closed-loop A/B")
    parser.add_argument("--output", type=Path, default=Path("logs/diagnostics/camera_fixed_interval.json"))
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.seconds <= 0:
        parser.error("--interval and --seconds must be positive")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2 to estimate same-mode repeat variation")
    if args.parallel_envs < 1:
        parser.error("--parallel-envs must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if np is None:
        raise RuntimeError("Benchmark execution requires NumPy; run inside the aicar_brain container")
    if not args.simulator_path.is_file():
        raise FileNotFoundError(f"Staged simulator binary not found: {args.simulator_path}")
    args.simulator_sha256 = _sha256(args.simulator_path)
    args.critical_artifact_hashes_start = _critical_artifact_hashes(args.simulator_path)
    if args.model is not None and not args.model.is_file():
        raise FileNotFoundError(f"Policy model not found: {args.model}")
    reward_config = _saved_reward_config(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    run_index = 0
    # Alternating order reduces systematic bias from heat/clock/load drift.
    for repeat in range(args.repeats):
        order = [False, True] if repeat % 2 == 0 else [True, False]
        for camera_enabled in order:
            for policy_path in ([None, args.model] if args.model else [None]):
                policy_label = "policy" if policy_path else "open"
                mode_label = "camera_on" if camera_enabled else "camera_off"
                jobs = []
                for worker_index in range(args.parallel_envs):
                    run_index += 1
                    label = f"{policy_label}-{mode_label}-r{repeat + 1}-w{worker_index}"
                    job = {
                        "label": label,
                        "camera_enabled": camera_enabled,
                        "simulator_path": args.simulator_path,
                        "port": args.port_base + run_index - 1,
                        "interval_s": args.interval,
                        "seconds": args.seconds,
                        "map_id": args.map_id,
                        "step_timeout": args.step_timeout,
                        "policy_path": policy_path,
                        "seed": args.seed,
                        "repeat_index": repeat,
                        "worker_index": worker_index,
                        "reward_config": reward_config,
                        "simulator_log_path": (
                            args.output.parent.resolve()
                            / f"{args.output.stem}_{label}.unity.log"
                        ),
                    }
                    jobs.append(job)
                    print(
                        f"[benchmark] start {label} port={job['port']} "
                        f"interval={args.interval:.6f}s target={args.seconds:.3f} sim-s",
                        flush=True,
                    )

                batch_start = time.perf_counter()
                if args.parallel_envs == 1:
                    batch_results = [_worker_entry(jobs[0])]
                else:
                    from concurrent.futures import ProcessPoolExecutor
                    with ProcessPoolExecutor(max_workers=args.parallel_envs) as pool:
                        batch_results = list(pool.map(_worker_entry, jobs))
                batch_wall = time.perf_counter() - batch_start
                failures = [result for result in batch_results if "_error" in result]
                if failures:
                    raise RuntimeError("benchmark worker failed: " + json.dumps(failures, indent=2))
                for run in batch_results:
                    run["parallel_batch_wall_seconds"] = batch_wall
                    runs.append(run)
                    print(
                        f"[benchmark] done {run['label']}: sim={run['simulated_seconds']:.3f}s "
                        f"wall={run['wall_seconds']:.3f}s UnityCPU={run['unity_cpu_seconds']}s "
                        f"CPU/sim-s={run['unity_cpu_seconds_per_simulated_second']}",
                        flush=True,
                    )
                # Keep partial evidence if a later mode/repeat fails or is interrupted.
                _save_report(args.output, runs, args)
    args.critical_artifact_hashes_end = _critical_artifact_hashes(args.simulator_path)
    _save_report(args.output, runs, args)
    if args.critical_artifact_hashes_end != args.critical_artifact_hashes_start:
        raise RuntimeError(
            "Staged Unity payload changed during benchmark; see recorded start/end hashes"
        )
    print(f"[benchmark] report: {args.output.resolve()}", flush=True)
    print(f"[benchmark] raw traces: {args.output.with_suffix('.npz').resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
