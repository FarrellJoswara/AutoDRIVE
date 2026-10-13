"""Deterministic decision-plan builder for the bounded official PPO campaign.

This module does not start, stop, or evaluate runs. It turns the immutable
campaign clock plus collected evidence into a machine-readable plan for the
runtime owner to execute through the normal API. No per-step model/LLM loop is
involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.layer4.hub.campaign_deadline import (
    CAMPAIGN_PLAN,
    _lr_experiment_evidence_is_healthy,
    _positive_finite,
    _valid_sha256,
    BC_TIME_COST_CALIBRATION_LAP_S,
    TIME_COST_VARIANTS,
    CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE,
    campaign_effective_config_sha256,
    campaign_clock_status,
    default_state_path,
    load_campaign_state,
)


def _utc(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("decision-plan timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _regression_count(evidence: dict[str, Any]) -> int:
    value = evidence.get("consecutive_regressing_probes", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("consecutive_regressing_probes must be a non-negative integer")
    return value


def _plateau_count(evidence: dict[str, Any]) -> int:
    value = evidence.get("consecutive_plateau_reviews", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("consecutive_plateau_reviews must be a non-negative integer")
    return value


def _candidate(
    experiment_id: str,
    phase: str,
    base_request: Any,
    parent_sha256: Any,
    changes: dict[str, Any],
    evidence: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not isinstance(base_request, dict) or not _valid_sha256(parent_sha256):
        return None
    settings = dict(base_request)
    if not isinstance(settings.get("resume"), str) or not settings["resume"].strip():
        return None
    settings.update(changes)
    settings.update({
        "campaign_experiment_id": experiment_id,
        "campaign_experiment_phase": phase,
    })
    config_hash = campaign_effective_config_sha256(settings)
    settings["campaign_experiment_config_sha256"] = config_hash
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "parent_checkpoint_sha256": str(parent_sha256).lower(),
        "config_sha256": config_hash,
        "request_settings": settings,
        "declaration_evidence": evidence,
        "execution_status": "proposed_only; requires parent authentication and append-only declaration before API submission",
    }


def build_campaign_decision_plan(
    state: dict[str, Any],
    *,
    evidence: Optional[dict[str, Any]] = None,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build a reviewable next-action plan without mutating campaign state."""
    if state.get("status") != "active":
        raise ValueError("decision plan requires an active campaign")
    if state.get("campaign_plan") != CAMPAIGN_PLAN:
        raise ValueError("decision plan rejects a campaign with a modified fixed plan")
    observed = evidence or {}
    if not isinstance(observed, dict):
        raise ValueError("campaign decision evidence must be an object")
    clock = campaign_clock_status(state, utc_now=utc_now, monotonic_ns=monotonic_ns, boot_id=boot_id)
    elapsed_h = float(clock.get("elapsed_seconds", 0.0)) / 3600.0
    if clock.get("phase") == "expired":
        phase = "hard_deadline"
        actions = ["verify_all_run_and_simulator_processes_are_stopped", "finalize_ledger"]
    elif clock.get("phase") in {"final_validation", "final_validation_requested"}:
        phase = "final_validation"
        actions = [
            "freeze_training_and_select_at_most_two_finalists",
            "clean_reload_each_authenticated_checkpoint",
            "run_three_attempts_per_finalist_with_one_warmup_and_ten_scored_laps",
            "require_zero_collisions_and_preserve_failed_attempts",
        ]
    elif elapsed_h >= 5:
        phase = "best_validated_trend"
        regressions = _regression_count(observed)
        plateaus = _plateau_count(observed)
        boundary_recovery = observed.get("boundary_recovery_candidate")
        if (isinstance(boundary_recovery, dict)
                and boundary_recovery.get("boundary_recovery") == CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE):
            actions = ["resume_exact_204800_checkpoint_after_verified_runner_boundary_stop",
                       "preserve_selected_reward_noise_and_learning_rate"]
        elif regressions >= 2:
            actions = ["revert_to_most_recent_passing_authenticated_checkpoint", "set_lr_1e5_only", "preserve_regression_evidence"]
        elif plateaus >= 3:
            actions = ["review_plateau_parent_at_lr_3e5", "consider_lr_1e5_or_health_gated_lr_1e4_only", "preserve_reward_and_noise"]
        else:
            actions = ["continue_best_passing_authenticated_candidate_settings_exactly", "preserve_reward_and_noise"]
    elif elapsed_h >= 2:
        phase = "pace_experiments"
        same_parent_control_ready = (
            isinstance(observed.get("baseline_control_parent_sha256"), str)
            and _valid_sha256(observed.get("baseline_control_parent_sha256"))
            and observed.get("baseline_control_probe_passed") is True
            and _valid_sha256(observed.get("baseline_control_probe_report_sha256"))
            and _positive_finite(observed.get("baseline_control_median_scored_lap_s"))
        )
        if same_parent_control_ready:
            actions = [
                "compare_zero_cost_control_with_time_cost_3_and_time_cost_5",
                "start_both_reward_variants_from_the_same_authenticated_parent",
            ]
        else:
            actions = [
                "record_time_cost_3_and_time_cost_5_as_tbc",
                "wait_for_a_passing_zero_cost_probe_and_authenticated_parent_before_branching",
            ]
        lr_evidence = observed.get("lr_experiment_evidence")
        lr_eligible = (
            isinstance(lr_evidence, dict)
            and observed.get("best_reward_parent_sha256") == lr_evidence.get("best_reward_parent_sha256")
            and _lr_experiment_evidence_is_healthy(lr_evidence)
        )
        if lr_eligible:
            actions.append("consider_lr_1e4_only_from_best_reward_parent_after_health_gate")
        else:
            actions.append("skip_lr_1e4_until_probe_and_audit_health_gate_passes")
    else:
        phase = "baseline_observation"
        actions = [
            "keep_learning_rate_and_reward_unchanged",
            "record_wall_time_compute_cost_and_official_probe_trend",
            "keep_one_environment",
        ]

    faults = []
    if observed.get("nonfinite_update") is True:
        faults.append("nonfinite_update")
    if observed.get("actor_fault") is True:
        faults.append("actor_fault")
    if observed.get("sensor_fault") is True:
        faults.append("sensor_fault")
    if faults:
        actions.insert(0, "stop_current_branch_and_preserve_failed_artifacts")
        actions.insert(1, "do_not_launch_a_descendant_until_fault_review")

    candidates: list[dict[str, Any]] = []
    if phase == "pace_experiments":
        if same_parent_control_ready:
            base_request = observed.get("base_train_request")
            parent_hash = observed.get("baseline_control_parent_sha256")
            control_evidence = {
                "baseline_control_parent_sha256": parent_hash,
                "baseline_control_probe_passed": True,
                "baseline_control_probe_report_sha256": observed.get("baseline_control_probe_report_sha256"),
                "baseline_control_median_scored_lap_s": observed.get("baseline_control_median_scored_lap_s"),
            }
            for experiment_id, time_cost in TIME_COST_VARIANTS.items():
                candidate = _candidate(experiment_id, "pace_experiments", base_request, parent_hash,
                                       {"time_cost_per_simulated_second": time_cost}, control_evidence)
                if candidate is not None:
                    candidates.append(candidate)
        lr_evidence = observed.get("lr_experiment_evidence")
        best_request = observed.get("best_reward_parent_train_request")
        best_parent_hash = observed.get("best_reward_parent_sha256")
        if (isinstance(lr_evidence, dict) and best_parent_hash == lr_evidence.get("best_reward_parent_sha256")
                and _lr_experiment_evidence_is_healthy(lr_evidence)):
            candidate = _candidate("lr_1e4_best_stable", "pace_experiments", best_request, best_parent_hash,
                                   {"learning_rate": 1e-4}, lr_evidence)
            if candidate is not None:
                candidates.append(candidate)
    elif phase == "best_validated_trend":
        regressions = _regression_count(observed)
        plateaus = _plateau_count(observed)
        boundary_recovery = observed.get("boundary_recovery_candidate")
        if (isinstance(boundary_recovery, dict)
                and boundary_recovery.get("boundary_recovery") == CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE):
            candidate = _candidate(
                "continue_best_boundary_recovery", phase,
                boundary_recovery.get("train_request"),
                boundary_recovery.get("checkpoint_sha256"),
                {},
                {key: value for key, value in boundary_recovery.items()
                 if key not in {"train_request", "checkpoint_sha256"}},
            )
            if candidate is not None:
                candidates.append(candidate)
        elif regressions >= 2:
            branch = observed.get("regression_recovery_evidence")
            if isinstance(branch, dict):
                request = observed.get("most_recent_passing_train_request")
                parent_hash = observed.get("most_recent_passing_checkpoint_sha256")
                candidate = _candidate("regression_recovery", phase, request, parent_hash,
                                       {"learning_rate": 1e-5}, branch)
                if candidate is not None:
                    candidates.append(candidate)
        elif plateaus >= 3:
            branch = observed.get("plateau_experiment_evidence")
            request = observed.get("plateau_parent_train_request")
            parent_hash = observed.get("plateau_parent_checkpoint_sha256")
            if isinstance(branch, dict):
                conservative = _candidate("plateau_lr_conservative", phase, request, parent_hash,
                                          {"learning_rate": 1e-5}, branch)
                if conservative is not None:
                    candidates.append(conservative)
                if _lr_experiment_evidence_is_healthy(branch):
                    healthy = _candidate("plateau_lr_healthy", phase, request, parent_hash,
                                         {"learning_rate": 1e-4}, branch)
                    if healthy is not None:
                        candidates.append(healthy)
        else:
            branch = observed.get("selected_passing_candidate_evidence")
            request = observed.get("selected_passing_train_request")
            parent_hash = observed.get("selected_passing_checkpoint_sha256")
            if isinstance(branch, dict):
                candidate = _candidate("continue_best", phase, request, parent_hash, {}, branch)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            actions.append("record_tbc_until_required_parent_and_evidence_are_available")

    now = (utc_now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema_version": 1,
        "campaign_id": state.get("campaign_id"),
        "campaign_status": state.get("status"),
        "phase": phase,
        "observed_at_utc": now.isoformat(),
        "elapsed_hours": elapsed_h,
        "seconds_until_training_freeze": clock.get("seconds_until_training_freeze"),
        "seconds_until_hard_deadline": clock.get("remaining_seconds"),
        "clock_reason": clock.get("reason"),
        "actions": actions,
        "branch_candidates": candidates,
        "faults": faults,
        "evidence": observed,
        "execution": "plan_only; runtime owner executes approved actions through the normal API",
        "fixed_policy": {
            "training_freeze_hours": 9,
            "hard_deadline_hours": 12,
            "max_finalists": 2,
            "final_attempts_per_finalist": 3,
            "final_warmup_laps": 1,
            "final_scored_laps": 10,
            "final_zero_collision_required": True,
            "single_environment": True,
            "audit_counter_semantics": {
                "serialized_optimizer_updates_field": "SB3 model._n_updates; completed PPO training epochs, not optimizer.step calls or rollout cycles",
                "rollout_cycles": "derive from global timestep delta / effective n_steps / n_envs",
                "sgd_step_count": "report only when directly measured from optimizer state",
            },
            "campaign_plan_sha256": hashlib.sha256(
                json.dumps(CAMPAIGN_PLAN, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "comparison_rules": {
                "time_cost_variants_share_one_authenticated_campaign_checkpoint_parent": True,
                "bc_11_lap_median_calibration_reference_s": 110.255550384521,
                "time_cost_variant_semantics": "time_cost_3 and time_cost_5 are total penalty points per BC-calibration lap, converted to reward units per simulated second",
                "time_cost_variant_per_simulated_second": dict(TIME_COST_VARIANTS),
                "time_cost_variant_penalty_per_bc_calibration_lap": {
                    "time_cost_3": 3.0,
                    "time_cost_5": 5.0,
                },
                "bc_calibration_is_not_the_campaign_policy_control": True,
                "compare_official_probe_medians_and_collisions_not_raw_return": True,
                "regression_revert_after_consecutive_probes": 2,
                "one_group_adjustment_after_plateau_reviews": 3,
            },
            "phase_schedule": [
                {"hours": [0, 2], "phase": "baseline_observation", "learning_rate": 3e-5, "reward_time_cost": 0.0},
                {"hours": [2, 5], "phase": "pace_experiments",
                 "reward_time_cost_candidates": [0.0, *TIME_COST_VARIANTS.values()],
                 "reward_time_cost_candidate_ids": {key: value for key, value in TIME_COST_VARIANTS.items()},
                 "time_cost_calibration_lap_s": BC_TIME_COST_CALIBRATION_LAP_S,
                 "all_variants_share_parent": True},
                {"hours": [5, 9], "phase": "best_validated_trend", "revert_after_regressions": 2, "single_group_adjust_after_plateau_reviews": 3},
                {"hours": [9, 12], "phase": "final_validation", "training_frozen": True, "max_finalists": 2},
                {"after": "readiness_backed_campaign_validation_requested", "phase": "final_validation", "training_frozen": True, "max_finalists": 2,
                 "applies_when_deadline_policy": "unbounded_until_readiness"},
            ],
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=default_state_path())
    parser.add_argument("--evidence", type=Path, help="optional JSON snapshot of run/probe/audit evidence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--at-utc", help="test/replay timestamp; omitted uses current UTC")
    args = parser.parse_args(argv)
    state = load_campaign_state(args.state)
    if state is None:
        raise FileNotFoundError(f"campaign state not found: {args.state}")
    evidence = _read_json(args.evidence) if args.evidence else {}
    plan = build_campaign_decision_plan(state, evidence=evidence, utc_now=_utc(args.at_utc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"decision-plan artifact already exists: {args.output}")
    encoded = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"phase": plan["phase"], "output": str(args.output), "actions": plan["actions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
