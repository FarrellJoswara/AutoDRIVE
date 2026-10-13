"""Pure derivation of conservative 5–9h campaign trend evidence.

The caller supplies per-evaluation facts only after authenticating the loaded
checkpoint and validating the 1+3 report and sensor freshness. This module
orders those facts, ignores infrastructure/incomplete observations, and emits
the evidence keys consumed by the campaign decision planner.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from src.layer4.hub.campaign_deadline import TIME_COST_VARIANTS, campaign_effective_config_sha256


IMPROVEMENT_THRESHOLD = 0.005
REGRESSION_THRESHOLD = 0.005


def _timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (OverflowError, TypeError, ValueError):
        return False


def _reward_declaration_hash(request: dict[str, Any]) -> Optional[str]:
    cost = request.get("time_cost_per_simulated_second", 0.0)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    try:
        if not math.isfinite(float(cost)):
            return None
    except (OverflowError, TypeError, ValueError):
        return None
    if math.isclose(float(cost), 0.0, rel_tol=0.0, abs_tol=1e-15):
        return ""
    experiment_id = request.get("campaign_experiment_id")
    expected = TIME_COST_VARIANTS.get(experiment_id)
    config_hash = request.get("campaign_experiment_config_sha256")
    if (expected is None or not math.isclose(float(cost), expected, rel_tol=0.0, abs_tol=1e-15)
            or not _sha(config_hash)):
        return None
    return config_hash


def _record_reward_declaration_hash(record: dict[str, Any]) -> Optional[str]:
    request = record.get("train_request")
    if not isinstance(request, dict):
        return None
    direct = _reward_declaration_hash(request)
    if direct is not None:
        return direct
    cost = request.get("time_cost_per_simulated_second", 0.0)
    linked = record.get("reward_experiment_config_sha256")
    try:
        if (isinstance(cost, (int, float)) and not isinstance(cost, bool)
                and math.isfinite(float(cost)) and not math.isclose(float(cost), 0.0, abs_tol=1e-15)
                and _sha(linked)):
            return linked
    except (OverflowError, TypeError, ValueError):
        return None
    return None


def _eligible(record: Any) -> bool:
    """Accept only authenticated scheduled 1+3 observations with live sensors."""
    if not isinstance(record, dict):
        return False
    facts = record.get("trend_validation")
    if not isinstance(facts, dict):
        return False
    model_sha = record.get("model_sha256")
    request = record.get("train_request")
    validation_mode = record.get("validation_mode")
    report_validation_mode = record.get("report_validation_mode")
    if (validation_mode is not None and report_validation_mode is not None
            and validation_mode != report_validation_mode):
        return False
    if validation_mode is None:
        validation_mode = report_validation_mode
    if not isinstance(request, dict) or _record_reward_declaration_hash(record) is None:
        return False
    outcome = facts.get("outcome")
    collision_count = facts.get("collision_count")
    median = facts.get("median_scored_lap_s")
    outcome_shape_valid = (
        outcome == "pass"
        and isinstance(collision_count, int) and not isinstance(collision_count, bool)
        and collision_count == 0
        and _finite_positive(median)
    ) or (
        outcome == "collision"
        and isinstance(collision_count, int) and not isinstance(collision_count, bool)
        and collision_count > 0
    )
    return bool(
        record.get("state") == "complete"
        and validation_mode == "training_probe"
        and record.get("report_protocol") == "official_training_probe_1_plus_3"
        and record.get("model_lineage_verified") is True
        and _sha(model_sha)
        and facts.get("authenticated") is True
        and facts.get("scheduled_probe") is True
        and isinstance(facts.get("completed_rollouts_since_branch"), int)
        and not isinstance(facts.get("completed_rollouts_since_branch"), bool)
        and facts.get("completed_rollouts_since_branch", 0) >= 2
        and facts.get("protocol_valid") is True
        and facts.get("sensor_valid") is True
        and facts.get("loaded_model_sha256") == model_sha
        and outcome_shape_valid
        and isinstance(record.get("evaluation_id"), str)
        and bool(record["evaluation_id"])
        and isinstance(request.get("resume"), str)
        and bool(request["resume"].strip())
    )


def _branch_key(record: dict[str, Any]) -> str:
    request = dict(record["train_request"])
    request.pop("resume", None)
    return campaign_effective_config_sha256(request)


def derive_campaign_trend_evidence(
    probe_records: list[Any],
    *,
    declared_experiment_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Derive regression/plateau evidence from authenticated scheduled probes.

    Infra, missing, manual, malformed, unauthenticated, and sensor-invalid
    records are excluded entirely. A valid collision is a policy regression
    and a non-improvement. A valid completed probe is a regression only when
    its median is over 0.5% slower than the best passing median seen so far.
    Plateau counts advance for eligible probes without at least 0.5% new-best
    improvement. Only one conservative plateau LR adjustment may be proposed.
    """
    excluded = [
        str(row.get("evaluation_id", "unknown"))
        for row in probe_records
        if not _eligible(row)
    ]
    eligible = [row for row in probe_records if _eligible(row)]
    id_counts: dict[str, int] = {}
    for row in probe_records:
        if isinstance(row, dict) and isinstance(row.get("evaluation_id"), str) and row["evaluation_id"]:
            id_counts[row["evaluation_id"]] = id_counts.get(row["evaluation_id"], 0) + 1
    duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
    if duplicate_ids:
        excluded.extend(duplicate_ids)
        eligible = [row for row in eligible if row["evaluation_id"] not in set(duplicate_ids)]
    # Missing/unparseable times cannot define temporal order, so fail closed by
    # excluding those rows instead of falling back to a lexical ID ordering.
    ordered = [row for row in eligible if _timestamp(row["trend_validation"].get("timestamp_utc")) is not None]
    ordered.sort(key=lambda row: (
        _timestamp(row["trend_validation"].get("timestamp_utc")) or datetime.max.replace(tzinfo=timezone.utc),
        row["evaluation_id"],
    ))
    excluded.extend(row["evaluation_id"] for row in eligible
                    if _timestamp(row["trend_validation"].get("timestamp_utc")) is None)

    # Compare only one effective training branch at a time. A reward, LR, or
    # exploration change starts a new trend; selecting the branch is based on
    # the globally fastest authenticated passing checkpoint.
    passing_for_selection = [row for row in ordered if row["trend_validation"].get("outcome") == "pass"]
    selected_branch = None
    if passing_for_selection:
        selected_global = min(
            passing_for_selection,
            key=lambda row: (float(row["trend_validation"]["median_scored_lap_s"]), row["evaluation_id"]),
        )
        selected_branch = _branch_key(selected_global)
    elif ordered:
        selected_branch = _branch_key(ordered[-1])
    other_branch_count = sum(1 for row in ordered if _branch_key(row) != selected_branch)
    ordered = [row for row in ordered if selected_branch is not None and _branch_key(row) == selected_branch]

    best_median: Optional[float] = None
    best_record: Optional[dict[str, Any]] = None
    most_recent_passing: Optional[dict[str, Any]] = None
    regressions = 0
    plateau = 0
    useful_improvements = 0
    for record in ordered:
        facts = record["trend_validation"]
        outcome = facts["outcome"]
        request = record["train_request"]
        if outcome == "collision":
            regressions += 1
            plateau += 1
            continue

        median = facts.get("median_scored_lap_s")
        if not _finite_positive(median):
            excluded.append(record["evaluation_id"])
            continue
        median = float(median)
        if best_median is None:
            best_median, best_record = median, record
            most_recent_passing = record
            regressions = 0
            plateau = 0
            continue
        if median > best_median * (1.0 + REGRESSION_THRESHOLD):
            regressions += 1
            plateau += 1
            continue
        # A zero-collision probe can still represent a policy regression when
        # its median exceeds the accepted best by the 0.5% threshold. Preserve
        # the last nonregressing checkpoint as the recovery parent.
        most_recent_passing = record
        regressions = 0
        if median <= best_median * (1.0 - IMPROVEMENT_THRESHOLD):
            best_median, best_record = median, record
            plateau = 0
            useful_improvements += 1
        else:
            plateau += 1

    output: dict[str, Any] = {
        "trend_probe_count": len(ordered),
        "trend_excluded_evaluation_ids": excluded,
        "infrastructure_failures_excluded": True,
        "regression_threshold_fraction": REGRESSION_THRESHOLD,
        "improvement_threshold_fraction": IMPROVEMENT_THRESHOLD,
        "consecutive_regressing_probes": regressions,
        "consecutive_plateau_reviews": plateau,
        "scheduled_probe_count": len(ordered),
        "useful_improvement_count": useful_improvements,
        "trend_duplicate_evaluation_ids": duplicate_ids,
        "trend_selected_branch_config_sha256": selected_branch,
        "trend_other_branch_probe_count": other_branch_count,
    }
    if most_recent_passing is not None:
        output.update({
            "most_recent_passing_checkpoint_sha256": most_recent_passing["model_sha256"],
            "most_recent_passing_train_request": most_recent_passing["train_request"],
        })
    if best_record is not None:
        output.update({
            "best_trend_checkpoint_sha256": best_record["model_sha256"],
            "best_trend_train_request": best_record["train_request"],
            "best_trend_median_scored_lap_s": best_median,
        })
    recovery_ok = regressions >= 2 and most_recent_passing is not None
    if recovery_ok:
        request = dict(most_recent_passing["train_request"])
        output["regression_recovery_evidence"] = {
            "consecutive_regressing_probes": regressions,
            "infrastructure_failures_excluded": True,
            "most_recent_passing_checkpoint_sha256": most_recent_passing["model_sha256"],
            "parent_effective_config": {
                key: value for key, value in request.items()
                if key not in {"campaign_experiment_id", "campaign_experiment_phase",
                               "campaign_experiment_config_sha256"}
            },
            "parent_effective_config_sha256": campaign_effective_config_sha256(request),
            **({"reward_experiment_config_sha256": reward_hash} if (reward_hash := _record_reward_declaration_hash(most_recent_passing)) else {}),
        }
    plateau_available = "plateau_lr_conservative" not in declared_experiment_ids
    plateau_parent = best_record
    plateau_parent_request = plateau_parent.get("train_request") if plateau_parent else None
    plateau_parent_eligible = False
    if isinstance(plateau_parent_request, dict):
        try:
            plateau_parent_eligible = abs(float(plateau_parent_request.get("learning_rate", 0.0) or 0.0) - 3e-5) <= 1e-12
        except (OverflowError, TypeError, ValueError):
            plateau_parent_eligible = False
    if plateau_available and plateau >= 3 and plateau_parent_eligible and plateau_parent is not None:
        request = dict(plateau_parent_request)
        output.update({
            "plateau_parent_checkpoint_sha256": plateau_parent["model_sha256"],
            "plateau_parent_train_request": request,
            "plateau_experiment_evidence": {
                "consecutive_plateau_reviews": plateau,
                "scheduled_probe_count": len(ordered),
                # This is the current uninterrupted plateau window. Earlier
                # gains already reset `plateau` and do not disqualify it.
                "useful_improvement_count": 0,
                "parent_effective_config": {
                    key: value for key, value in request.items()
                    if key not in {"campaign_experiment_id", "campaign_experiment_phase",
                                   "campaign_experiment_config_sha256"}
                },
                "parent_effective_config_sha256": campaign_effective_config_sha256(request),
                **({"reward_experiment_config_sha256": reward_hash} if (reward_hash := _record_reward_declaration_hash(plateau_parent)) else {}),
            },
        })
    return output
