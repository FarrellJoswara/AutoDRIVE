#!/usr/bin/env python3
"""Summarize existing official reward experiment run directories.

This is an offline, read-only report generator. It does not launch or stop runs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                    else:
                        invalid += 1
                except (json.JSONDecodeError, UnicodeError):
                    invalid += 1
    except OSError:
        pass
    return rows, invalid


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None,
                "max": None, "values": []}
    return {"count": len(values), "min": min(values),
            "mean": statistics.mean(values), "median": statistics.median(values),
            "max": max(values), "values": values}


def reward_metrics_summary(path: Path) -> tuple[dict[str, Any], int]:
    rows, malformed = read_jsonl(path)
    fields = {
        "speed": "speed", "v_long": "v_long", "route_speed": "current_route_speed_mps",
        "throttle": "throttle_command", "steering": "steering_command",
    }
    per_car: dict[str, dict[str, list[float]]] = {}
    rewards: dict[str, list[float]] = {}
    resets: dict[str, dict[str, Any]] = {}
    counters: dict[str, dict[str, dict[str, float]]] = {}
    sample_count = 0
    car_sample_count = 0
    for row in rows:
        cars = row.get("cars")
        if not isinstance(cars, list):
            continue
        sample_count += 1
        for car in cars:
            if not isinstance(car, dict):
                continue
            car_sample_count += 1
            car_id = str(car.get("env_id", "unknown"))
            target = per_car.setdefault(car_id, {name: [] for name in fields})
            for name, source in fields.items():
                value = finite_number(car.get(source))
                if value is not None:
                    target[name].append(value)
            components = car.get("reward_components")
            if isinstance(components, dict):
                for name, value in components.items():
                    number = finite_number(value)
                    if number is not None:
                        rewards.setdefault(str(name), []).append(number)
            reset = resets.setdefault(car_id, {"samples": 0, "reset_true_samples": 0, "reset_observed": False,
                                               "fault_fields_available_samples": 0,
                                               "neutral_fallback_diagnostic_observed": False,
                                               "command_expiry_diagnostic_observed": False})
            reset["samples"] += 1
            if car.get("reset") is True:
                reset["reset_true_samples"] += 1
                reset["reset_observed"] = True
            if car.get("fault_fields_available") is True:
                reset["fault_fields_available_samples"] += 1
            for diagnostic in ("neutral_fallback_diagnostic", "command_expiry_diagnostic"):
                if isinstance(car.get(diagnostic), dict):
                    reset[f"{diagnostic}_observed"] = True
            for name in ("neutral_fallback_count",):
                number = finite_number(car.get(name))
                if number is not None:
                    item = counters.setdefault(car_id, {}).setdefault(name, {"first": number, "last": number, "max": number})
                    item["last"] = number
                    item["max"] = max(item["max"], number)
    return ({
        "available": path.is_file(),
        "sample_weighting": "Each recorded car snapshot contributes one observation; pooled means are sample weighted, not simulation-time weighted.",
        "fleet_samples": sample_count, "car_samples": car_sample_count,
        "per_car": {car: {field: distribution(values) for field, values in metrics.items()}
                    for car, metrics in sorted(per_car.items())},
        "reward_components": {name: distribution(values) for name, values in sorted(rewards.items())},
        "reset_observations_by_car": resets if resets else None,
        "cumulative_counters_by_car": counters if counters else None,
        "counter_handling": "Cumulative counters are reported per car as first/last/max; they are never summed across samples.",
        "missing_fields": (["car metric values (no valid car samples)"] if not car_sample_count else []) +
                          (["reward components"] if not rewards else []) +
                          (["reset indicator"] if not any(x["reset_observed"] for x in resets.values()) else []),
        "malformed_lines": malformed,
    }, malformed)


def evaluation_summary(path: Path) -> dict[str, Any]:
    report = read_json(path)
    if not isinstance(report, dict):
        return {"available": False, "missing_fields": ["evaluation.json"]}
    runs = report.get("runs")
    if not isinstance(runs, list):
        return {"available": True, "run_records": None, "missing_fields": ["runs array"]}
    lap_values: list[float] = []
    first_laps: list[float] = []
    later_laps: list[float] = []
    complete_count = collision_free = scored_valid = 0
    safety_fields = ("race_collisions", "warmup_collisions", "raw_collision_count_at_finish",
                     "control_gap_fault_count", "ros_sensor_stamp_gap_count",
                     "route_projection_invalid_sample_count", "camera_missing_sample_count",
                     "camera_stale_sample_count", "camera_freshness_unknown_count")
    safety: dict[str, dict[str, float]] = {key: {"observed_runs": 0, "runs_nonzero": 0, "max": 0}
                                          for key in safety_fields}
    for run in runs:
        if not isinstance(run, dict):
            continue
        times = run.get("race_lap_times_s")
        if isinstance(times, list):
            valid_times = [n for value in times if (n := finite_number(value)) is not None and n > 0]
            lap_values.extend(valid_times)
            if valid_times:
                first_laps.append(valid_times[0])
                later_laps.extend(valid_times[1:])
        complete_count += run.get("complete") is True
        scored_valid += run.get("score_status") == "valid"
        collision_free += run.get("score_status") == "valid" and run.get("raw_collision_count_at_finish") == 0
        for key in safety_fields:
            value = finite_number(run.get(key))
            if value is not None:
                item = safety[key]
                item["observed_runs"] += 1
                item["runs_nonzero"] += value > 0
                item["max"] = max(item["max"], value)
    missing = []
    if not runs:
        missing.append("evaluation run records")
    for field, stats in safety.items():
        if stats["observed_runs"] == 0:
            missing.append(field)
    validation_mode = report.get("validation_mode")
    practice_passed = report.get("practice_diagnostic_passed")
    score_valid = report.get("competitive_score_valid")
    return {"available": True, "mode": "evaluation",
            "validation_mode": validation_mode,
            "noncompetitive_diagnostic": report.get("noncompetitive_diagnostic"),
            "competitive_score_valid": score_valid,
            "valid_for_comparison": score_valid,
            "practice_diagnostic_passed": practice_passed,
            "measurement_status": (
                "practice_diagnostic_passed" if practice_passed is True else
                "practice_diagnostic_failed" if practice_passed is False else
                "scored_valid" if scored_valid else "unclassified"
            ),
            "run_records": len(runs),
            "complete_runs": complete_count, "score_valid_runs": scored_valid,
            "valid_collision_free_runs": collision_free,
            "heldout_lap_times_s": distribution(lap_values),
            "first_lap_times_s": distribution(first_laps),
            "later_lap_times_s": distribution(later_laps),
            "safety_faults": safety, "missing_fields": missing}


def detect_mode(root: Path, rows: list[dict[str, Any]], telemetry: Any) -> tuple[str, str]:
    metadata: list[Any] = [telemetry]
    for name in ("run_manifest.json", "config.json", "run_config.json", "metadata.json",
                 "latest_training_phase.json"):
        data = read_json(root / name)
        if data is not None:
            metadata.append(data)
    metadata.extend(rows)
    for item in metadata:
        if not isinstance(item, dict):
            continue
        for key in ("mode", "run_type", "kind", "phase"):
            value = item.get(key)
            if isinstance(value, str) and value.lower() in {"training", "evaluation", "eval"}:
                return ("evaluation" if value.lower() == "eval" else value.lower(), f"{key}")
            if key == "phase" and isinstance(value, str) and value.lower() in {
                "rollout", "training", "optimizing", "optimization"
            }:
                return "training", "latest_training_phase.json:phase"
        if item.get("training_mode") is True or item.get("is_training") is True:
            return "training", "training_mode/is_training"
        if item.get("training_mode") is False or item.get("is_training") is False:
            return "evaluation", "training_mode/is_training"
    # Journals generated by official PPO training have checkpoints and optimizer audits.
    return "training" if any(r.get("event_type") == "checkpoint" for r in rows) else "unknown", "journal evidence" if any(r.get("event_type") == "checkpoint" for r in rows) else "missing"


def summarize_run(run_dir: Path) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    race_path = root / "race_metrics.jsonl"
    rows, malformed = read_jsonl(race_path)
    telemetry = read_json(root / "latest_telemetry.json")
    audit, audit_malformed = read_jsonl(root / "ppo_optimizer_audit.jsonl")
    reward_metrics, _ = reward_metrics_summary(root / "reward_experiment_metrics.jsonl")
    evaluation = evaluation_summary(root / "evaluation.json")
    mode, mode_source = detect_mode(root, rows, telemetry)
    if mode == "unknown" and evaluation.get("available"):
        mode, mode_source = "evaluation", "evaluation.json"
    laps = [r for r in rows if r.get("event_type") == "lap" and r.get("phase", "race") == "race"]
    lap_times = [n for r in laps if r.get("timing_valid") is True
                 if (n := finite_number(r.get("lap_time_s"))) is not None and n > 0]
    episodes = [r for r in rows if r.get("event_type") == "episode"]
    clean = [r for r in episodes if r.get("clean_10_lap_episode") is True]
    collision_total = sum(max(0, int(finite_number(r.get("race_collisions")) or 0)) for r in episodes)
    collision_lap_events = sum(r.get("collision_event") is True for r in laps)
    failures: dict[str, int] = {}
    for row in episodes:
        reason = row.get("termination_reason")
        if reason and reason not in ("completed", "episode_won", "clean_10_lap_episode"):
            failures[str(reason)] = failures.get(str(reason), 0) + 1
    finite_flags = [audit_row.get(key) for audit_row in audit
                    for key in ("metrics_finite", "gradients_finite", "parameters_finite", "optimizer_state_finite")
                    if isinstance(audit_row.get(key), bool)]
    audit_steps = [n for row in audit if (n := finite_number(row.get("num_timesteps"))) is not None]
    throughput: dict[str, Any] = {"available": False, "source": None, "values": None}
    timing = read_json(root / "timing_diagnostics.json")
    if isinstance(timing, dict):
        ratios = timing.get("ros_to_wall_elapsed_ratio_by_env")
        if isinstance(ratios, dict) and ratios:
            vals = [n for x in ratios.values() if (n := finite_number(x)) is not None]
            if vals:
                throughput = {"available": True, "source": "timing_diagnostics.json ros_to_wall_elapsed_ratio_by_env",
                              "values": distribution(vals)}
        if not throughput["available"]:
            for key in ("simulated_seconds_per_wall_second", "aggregate_simulated_seconds_per_wall_second", "fps", "throughput_fps"):
                val = finite_number(timing.get(key))
                if val is not None:
                    throughput = {"available": True, "source": f"timing_diagnostics.json:{key}", "values": val}
                    break
    missing_fields = [name for name, present in (("race_metrics.jsonl", race_path.is_file()),
        ("latest_telemetry.json", isinstance(telemetry, dict)), ("ppo_optimizer_audit.jsonl", (root / "ppo_optimizer_audit.jsonl").is_file()),
        ("timing_diagnostics.json", isinstance(timing, dict))) if not present]
    if not episodes:
        missing_fields.extend(("episode completion/collision/failure outcomes (no episode events)",))
    if not audit:
        missing_fields.append("PPO finite-state health (no audit rows)")
    if not throughput["available"]:
        missing_fields.append("throughput (no supported timing field)")
    if not reward_metrics["available"]:
        missing_fields.append("reward_experiment_metrics.jsonl")
    return {
        "run_dir": str(root), "run_id": (telemetry.get("run_id") if isinstance(telemetry, dict) else None) or (rows[0].get("run_id") if rows else root.name),
        "mode": mode, "mode_source": mode_source,
        "lap_times_s": distribution(lap_times),
        "lap_events": len(laps), "invalid_or_missing_lap_times": len(laps) - len(lap_times),
        "episodes_observed": len(episodes), "clean_completions": len(clean) if episodes else None,
        "collisions": {"episode_total": collision_total if episodes else None, "lap_events_with_collision": collision_lap_events,
                       "episodes_with_collision": sum((finite_number(r.get("race_collisions")) or 0) > 0 for r in episodes) if episodes else None},
        "failures_by_termination_reason": failures if episodes else None,
        "ppo_health": {"audit_rows": len(audit), "audit_malformed_lines": audit_malformed,
                       "optimizer_updates": audit[-1].get("optimizer_updates") if audit else None,
                       "last_num_timesteps": audit_steps[-1] if audit_steps else None,
                       "all_recorded_finite_checks_pass": all(finite_flags) if finite_flags else None,
                       "finite_check_count": len(finite_flags)},
        "throughput": throughput,
        "reward_experiment_metrics": reward_metrics,
        "heldout_evaluation": evaluation,
        "missing_fields": missing_fields,
        "malformed_race_metrics_lines": malformed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Existing experiment run directories")
    parser.add_argument("-o", "--output", type=Path, help="Write JSON here; defaults to stdout")
    args = parser.parse_args()
    report = {"schema_version": 1, "runs": [summarize_run(path) for path in args.run_dirs]}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
