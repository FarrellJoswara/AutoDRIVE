from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.layer4.hub.campaign_deadline import TIME_COST_VARIANTS, campaign_effective_config_sha256
from src.layer4.hub.campaign_trend import derive_campaign_trend_evidence


T0 = datetime(2026, 10, 13, 4, 0, tzinfo=timezone.utc)


def _record(index: int, *, outcome: str = "pass", median: float | None = 100.0,
            sensor_valid: bool = True, authenticated: bool = True,
            scheduled: bool = True, collision_count: int | None = None,
            timestamp: datetime | None = None) -> dict:
    model_sha = f"{index + 1:064x}"
    request = {
        "resume": f"/runs/rl/official_run_{index + 1:012x}/checkpoint.zip",
        "learning_rate": 3e-5,
        "time_cost_per_simulated_second": 0.0,
        "campaign_experiment_id": "continue_best",
        "campaign_experiment_phase": "best_validated_trend",
        "campaign_experiment_config_sha256": "f" * 64,
    }
    return {
        "state": "complete",
        "validation_mode": "training_probe",
        "evaluation_id": f"eval-{index:03d}",
        "model_sha256": model_sha,
        "model_lineage_verified": True,
        "report_protocol": "official_training_probe_1_plus_3",
        "train_request": request,
        "trend_validation": {
            "authenticated": authenticated,
            "scheduled_probe": scheduled,
            "completed_rollouts_since_branch": 2,
            "protocol_valid": True,
            "sensor_valid": sensor_valid,
            "loaded_model_sha256": model_sha,
            "outcome": outcome,
            "collision_count": collision_count if collision_count is not None else (1 if outcome == "collision" else 0),
            "median_scored_lap_s": median,
            "timestamp_utc": (timestamp or (T0 + timedelta(minutes=index))).isoformat(),
        },
    }


def test_two_consecutive_policy_regressions_exclude_infra_and_revert_latest_passing():
    rows = [
        _record(0, median=100),
        {"evaluation_id": "infra", "failure_classification": "infrastructure_failure"},
        _record(1, outcome="collision", median=None),
        _record(2, median=100.6),
    ]
    evidence = derive_campaign_trend_evidence(rows)
    assert evidence["consecutive_regressing_probes"] == 2
    assert evidence["infrastructure_failures_excluded"] is True
    assert evidence["trend_probe_count"] == 3
    assert "infra" in evidence["trend_excluded_evaluation_ids"]
    assert evidence["most_recent_passing_checkpoint_sha256"] == rows[0]["model_sha256"]
    recovery = evidence["regression_recovery_evidence"]
    assert recovery["most_recent_passing_checkpoint_sha256"] == rows[0]["model_sha256"]
    assert recovery["infrastructure_failures_excluded"] is True
    assert recovery["parent_effective_config_sha256"] == campaign_effective_config_sha256(rows[0]["train_request"])


def test_two_clean_slow_laps_restore_pre_regression_checkpoint():
    rows = [
        _record(0, median=100),
        _record(1, median=101),
        _record(2, median=102),
    ]
    evidence = derive_campaign_trend_evidence(rows)
    assert evidence["consecutive_regressing_probes"] == 2
    assert evidence["most_recent_passing_checkpoint_sha256"] == rows[0]["model_sha256"]
    assert evidence["regression_recovery_evidence"]["most_recent_passing_checkpoint_sha256"] == rows[0]["model_sha256"]


def test_three_no_improvement_probes_offer_one_conservative_adjustment():
    rows = [
        _record(0, median=100),
        _record(1, median=99.8),
        _record(2, outcome="collision", median=None),
        _record(3, median=99.7),
    ]
    evidence = derive_campaign_trend_evidence(rows)
    assert evidence["consecutive_plateau_reviews"] == 3
    assert evidence["scheduled_probe_count"] == 4
    assert evidence["useful_improvement_count"] == 0
    assert evidence["plateau_parent_checkpoint_sha256"] == rows[0]["model_sha256"]
    assert evidence["plateau_experiment_evidence"]["parent_effective_config_sha256"] == campaign_effective_config_sha256(rows[0]["train_request"])

    already_used = derive_campaign_trend_evidence(
        rows, declared_experiment_ids={"plateau_lr_conservative"},
    )
    assert "plateau_experiment_evidence" not in already_used


def test_nonzero_reward_parent_carries_append_only_reward_declaration_hash():
    rows = [_record(0, median=100), _record(1, median=99.8),
            _record(2, median=99.7), _record(3, median=99.6)]
    for row in rows:
        request = row["train_request"]
        request["campaign_experiment_id"] = "time_cost_3_calibrated_retry"
        request["time_cost_per_simulated_second"] = TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
        request["campaign_experiment_config_sha256"] = campaign_effective_config_sha256(request)
    evidence = derive_campaign_trend_evidence(rows)
    assert evidence["consecutive_plateau_reviews"] == 3
    assert evidence["plateau_experiment_evidence"]["reward_experiment_config_sha256"] == rows[0]["train_request"]["campaign_experiment_config_sha256"]


def test_nonzero_reward_continue_best_descendant_uses_verified_reward_link():
    rows = [_record(0, median=100), _record(1, median=99.8),
            _record(2, median=99.7), _record(3, median=99.6)]
    reward_hash = "a" * 64
    for row in rows:
        request = row["train_request"]
        request["campaign_experiment_id"] = "continue_best"
        request["time_cost_per_simulated_second"] = TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
        request["campaign_experiment_config_sha256"] = "b" * 64
        row["reward_experiment_config_sha256"] = reward_hash
    evidence = derive_campaign_trend_evidence(rows)
    assert evidence["trend_probe_count"] == 4
    assert evidence["plateau_experiment_evidence"]["reward_experiment_config_sha256"] == reward_hash


def test_improvement_resets_regression_and_plateau_sequences():
    rows = [
        _record(0, median=100),
        _record(1, median=101),
        _record(2, median=99.4),
        _record(3, median=100),
    ]
    evidence = derive_campaign_trend_evidence(rows)
    assert evidence["consecutive_regressing_probes"] == 1
    assert evidence["consecutive_plateau_reviews"] == 1
    assert "regression_recovery_evidence" not in evidence
    assert "plateau_experiment_evidence" not in evidence


@pytest.mark.parametrize("changes", [
    {"sensor_valid": False},
    {"authenticated": False},
    {"scheduled": False},
])
def test_invalid_or_unscheduled_observations_do_not_advance_counters(changes):
    row = _record(0, **changes)
    evidence = derive_campaign_trend_evidence([row])
    assert evidence["trend_probe_count"] == 0
    assert evidence["consecutive_regressing_probes"] == 0
    assert evidence["consecutive_plateau_reviews"] == 0


def test_first_branch_probe_is_excluded_until_two_optimizer_rollouts():
    first_branch = _record(0, outcome="collision", median=None)
    first_branch["trend_validation"]["completed_rollouts_since_branch"] = 1
    evidence = derive_campaign_trend_evidence([first_branch])
    assert evidence["trend_probe_count"] == 0
    assert evidence["consecutive_regressing_probes"] == 0


def test_probe_order_uses_recorded_timestamp_and_missing_time_is_excluded():
    older = _record(0, median=100.2, timestamp=T0 + timedelta(minutes=2))
    newer = _record(1, median=101, timestamp=T0 + timedelta(minutes=3))
    earlier = _record(2, median=100, timestamp=T0 + timedelta(minutes=1))
    missing_time = _record(3, median=1)
    missing_time["trend_validation"].pop("timestamp_utc")
    evidence = derive_campaign_trend_evidence([older, newer, earlier, missing_time])
    assert evidence["trend_probe_count"] == 3
    assert evidence["consecutive_regressing_probes"] == 1
    assert evidence["most_recent_passing_checkpoint_sha256"] == older["model_sha256"]
    assert "eval-003" in evidence["trend_excluded_evaluation_ids"]


def test_probe_trends_do_not_mix_reward_branch_histories():
    zero_cost = [_record(0, median=100), _record(1, median=103), _record(2, median=101)]
    retry = [_record(3, median=90), _record(4, median=91)]
    for row in retry:
        request = row["train_request"]
        request["campaign_experiment_id"] = "time_cost_5_calibrated_retry"
        request["time_cost_per_simulated_second"] = TIME_COST_VARIANTS["time_cost_5_calibrated_retry"]
        request["campaign_experiment_config_sha256"] = campaign_effective_config_sha256(request)
    evidence = derive_campaign_trend_evidence(zero_cost + retry)
    assert evidence["trend_probe_count"] == 2
    assert evidence["trend_other_branch_probe_count"] == 3
    assert evidence["consecutive_regressing_probes"] == 1
    assert evidence["most_recent_passing_checkpoint_sha256"] == retry[0]["model_sha256"]


def test_duplicate_evaluation_ids_do_not_inflate_trend_counts():
    first = _record(0, outcome="collision", median=None)
    duplicate = _record(0, outcome="collision", median=None, timestamp=T0 + timedelta(minutes=1))
    duplicate["trend_validation"]["sensor_valid"] = False
    evidence = derive_campaign_trend_evidence([first, duplicate])
    assert evidence["trend_probe_count"] == 0
    assert evidence["consecutive_regressing_probes"] == 0
    assert evidence["trend_duplicate_evaluation_ids"] == ["eval-000"]


def test_malformed_overflow_numbers_fail_closed_without_raising():
    huge_median = _record(0, median=10**400)
    huge_cost = _record(1, median=90)
    huge_cost["train_request"]["time_cost_per_simulated_second"] = 10**400
    evidence = derive_campaign_trend_evidence([huge_median, huge_cost])
    assert evidence["trend_probe_count"] == 0
    assert evidence["consecutive_regressing_probes"] == 0


def test_report_validation_mode_alias_is_accepted_but_conflicts_fail_closed():
    report_schema = _record(0, median=100)
    report_schema.pop("validation_mode")
    report_schema["report_validation_mode"] = "training_probe"

    evidence = derive_campaign_trend_evidence([report_schema])
    assert evidence["trend_probe_count"] == 1
    assert evidence["best_trend_median_scored_lap_s"] == 100

    conflicting = dict(report_schema)
    conflicting["validation_mode"] = "competitive"
    rejected = derive_campaign_trend_evidence([conflicting])
    assert rejected["trend_probe_count"] == 0
    assert rejected["trend_excluded_evaluation_ids"] == ["eval-000"]
