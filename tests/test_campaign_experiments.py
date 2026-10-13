from datetime import datetime, timedelta, timezone
import json

from src.layer4.hub.campaign_deadline import (
    BC_TIME_COST_CALIBRATION_LAP_S,
    CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE,
    TIME_COST_VARIANTS,
    campaign_effective_config_sha256,
    initialize_campaign,
    amend_campaign_deadline_policy,
    request_campaign_final_validation,
)
from src.layer4.hub.campaign_experiments import build_campaign_decision_plan


START = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
BOOT = "linux-boot-test"


def _state(tmp_path):
    gates = tmp_path / "gates.json"
    checkpoint = "a" * 64
    gates.write_text(json.dumps({
        "updated_policy_probe": {"status": "PASS", "useful": True, "checkpoint_sha256": checkpoint},
        "single_environment_queue_interruption": {"status": "PASS", "passed": True},
    }), encoding="utf-8")
    return initialize_campaign(tmp_path / "active.json", gates, utc_now=START,
                               monotonic_ns=0, boot_id=BOOT)


def _plan(state, hours, **kwargs):
    return build_campaign_decision_plan(
        state, utc_now=START + timedelta(hours=hours),
        monotonic_ns=int(hours * 3600 * 1_000_000_000), boot_id=BOOT, **kwargs,
    )


def test_campaign_phase_planner_has_fixed_clock_boundaries_and_no_execution(tmp_path):
    state = _state(tmp_path)
    assert _plan(state, 1)["phase"] == "baseline_observation"
    pace = _plan(state, 2, evidence={"lr_experiment_evidence": {}})
    assert pace["phase"] == "pace_experiments"
    assert "record_time_cost_3_and_time_cost_5_as_tbc" in pace["actions"]
    parent = "b" * 64
    ready = _plan(state, 2, evidence={
        "baseline_control_parent_sha256": parent,
        "baseline_control_probe_passed": True,
        "baseline_control_probe_report_sha256": "c" * 64,
        "baseline_control_median_scored_lap_s": 105.5,
    })
    assert "start_both_reward_variants_from_the_same_authenticated_parent" in ready["actions"]
    assert pace["execution"].startswith("plan_only")
    best = _plan(state, 5, evidence={"consecutive_plateau_reviews": 3})
    assert best["phase"] == "best_validated_trend"
    assert "review_plateau_parent_at_lr_3e5" in best["actions"]
    final = _plan(state, 9)
    assert final["phase"] == "final_validation"
    assert final["fixed_policy"]["max_finalists"] == 2
    assert final["fixed_policy"]["campaign_plan_sha256"]
    expired = _plan(state, 12)
    assert expired["phase"] == "hard_deadline"


def test_unbounded_planner_continues_after_old_deadline_until_readiness_request(tmp_path):
    state_path = tmp_path / "active.json"
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({
        "updated_policy_probe": {"status": "PASS", "useful": True, "checkpoint_sha256": "a" * 64},
        "single_environment_queue_interruption": {"status": "PASS", "passed": True},
    }), encoding="utf-8")
    state = initialize_campaign(state_path, gates, utc_now=START, monotonic_ns=0, boot_id=BOOT)
    state = amend_campaign_deadline_policy(
        state_path, authorization="user_override", reason="explicit user direction",
        utc_now=START + timedelta(hours=13), monotonic_ns=13 * 3600 * 1_000_000_000,
    )
    after_old_deadline = _plan(state, 16, evidence={
        "selected_passing_checkpoint_sha256": "b" * 64,
        "selected_passing_train_request": {
            "resume": "logs/rl/parent/checkpoints/p.zip",
            "learning_rate": 3e-5,
            "time_cost_per_simulated_second": 0.0,
            "exploration_profile": "bc_campaign",
        },
        "selected_passing_candidate_evidence": {
            "selected_passing_checkpoint_sha256": "b" * 64,
        },
    })
    assert after_old_deadline["phase"] == "best_validated_trend"
    assert [row["experiment_id"] for row in after_old_deadline["branch_candidates"]] == ["continue_best"]
    assert after_old_deadline["seconds_until_hard_deadline"] is None

    request = request_campaign_final_validation(
        state_path,
        candidate_checkpoint_sha256="b" * 64,
        evaluation_id="stable-probe-1",
        probe_report_sha256="c" * 64,
        readiness_evidence={
            "stable_candidate_verified": True,
            "source_evidence_sha256": "d" * 64,
            "recent_audit_evidence_sha256": "e" * 64,
            "recent_audits_finite": True,
            "sensor_health_passed": True,
            "zero_collision_probes": True,
            "passing_nonregressing_probe_count": 2,
            "sol_readiness_recommendation": {
                "decision": "request_final_validation",
                "reviewer": "sol",
                "reason": "Useful learning has plateaued; this authenticated candidate is ready for competitive validation.",
                "candidate_checkpoint_sha256": "b" * 64,
                "recommendation_evidence_sha256": "f" * 64,
            },
        },
        utc_now=START + timedelta(hours=16), monotonic_ns=16 * 3600 * 1_000_000_000,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    final_plan = _plan(state, 16)
    assert request["event"] == "campaign_validation_requested"
    assert final_plan["phase"] == "final_validation"
    assert "freeze_training_and_select_at_most_two_finalists" in final_plan["actions"]
    assert final_plan["branch_candidates"] == []


def test_campaign_phase_planner_reverts_two_regressions_and_flags_faults(tmp_path):
    state = _state(tmp_path)
    plan = _plan(state, 6, evidence={
        "consecutive_regressing_probes": 2,
        "nonfinite_update": True,
        "actor_fault": False,
        "sensor_fault": False,
    })
    assert plan["phase"] == "best_validated_trend"
    assert "revert_to_most_recent_passing_authenticated_checkpoint" in plan["actions"]
    assert plan["faults"] == ["nonfinite_update"]
    assert plan["actions"][0] == "stop_current_branch_and_preserve_failed_artifacts"


def test_campaign_phase_planner_emits_hash_locked_same_parent_variants(tmp_path):
    state = _state(tmp_path)
    parent_hash = "f" * 64
    request = {"resume": "logs/rl/official_run_parent/checkpoints/parent.zip",
               "learning_rate": 3e-5, "time_cost_per_simulated_second": 0.0}
    plan = _plan(state, 3, evidence={
        "baseline_control_parent_sha256": parent_hash,
        "baseline_control_probe_passed": True,
        "baseline_control_probe_report_sha256": "e" * 64,
        "baseline_control_median_scored_lap_s": 105.5,
        "base_train_request": request,
    })
    assert [row["experiment_id"] for row in plan["branch_candidates"]] == [
        "time_cost_3_calibrated_retry", "time_cost_5_calibrated_retry"
    ]
    left, right = plan["branch_candidates"]
    assert left["parent_checkpoint_sha256"] == right["parent_checkpoint_sha256"] == parent_hash
    for row in (left, right):
        settings = row["request_settings"]
        assert settings["campaign_experiment_config_sha256"] == campaign_effective_config_sha256(settings)
        assert settings["resume"] == request["resume"]
    assert left["request_settings"]["time_cost_per_simulated_second"] == TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    assert right["request_settings"]["time_cost_per_simulated_second"] == TIME_COST_VARIANTS["time_cost_5_calibrated_retry"]
    assert left["request_settings"]["time_cost_per_simulated_second"] * BC_TIME_COST_CALIBRATION_LAP_S == 3.0
    assert right["request_settings"]["time_cost_per_simulated_second"] * BC_TIME_COST_CALIBRATION_LAP_S == 5.0
    assert plan["fixed_policy"]["comparison_rules"]["time_cost_variant_semantics"].startswith(
        "time_cost_3 and time_cost_5 are total penalty points per BC-calibration lap"
    )


def test_campaign_phase_planner_emits_only_authorized_trend_candidates(tmp_path):
    state = _state(tmp_path)
    parent_hash = "f" * 64
    parent_request = {"resume": "logs/rl/parent/checkpoints/p.zip", "learning_rate": 3e-5,
                      "time_cost_per_simulated_second": TIME_COST_VARIANTS["time_cost_3_calibrated_retry"],
                      "exploration_profile": "bc_campaign"}
    continue_plan = _plan(state, 6, evidence={
        "selected_passing_checkpoint_sha256": parent_hash,
        "selected_passing_train_request": parent_request,
        "selected_passing_candidate_evidence": {"selected_passing_checkpoint_sha256": parent_hash},
    })
    assert [row["experiment_id"] for row in continue_plan["branch_candidates"]] == ["continue_best"]
    assert continue_plan["branch_candidates"][0]["request_settings"]["time_cost_per_simulated_second"] == TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    assert continue_plan["branch_candidates"][0]["request_settings"]["learning_rate"] == 3e-5

    recovery = _plan(state, 6, evidence={
        "consecutive_regressing_probes": 2,
        "most_recent_passing_checkpoint_sha256": parent_hash,
        "most_recent_passing_train_request": parent_request,
        "regression_recovery_evidence": {"most_recent_passing_checkpoint_sha256": parent_hash},
    })
    assert [row["experiment_id"] for row in recovery["branch_candidates"]] == ["regression_recovery"]
    assert recovery["branch_candidates"][0]["request_settings"]["learning_rate"] == 1e-5
    assert recovery["branch_candidates"][0]["request_settings"]["time_cost_per_simulated_second"] == TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]

    plateau = _plan(state, 6, evidence={
        "consecutive_plateau_reviews": 3,
        "plateau_parent_checkpoint_sha256": parent_hash,
        "plateau_parent_train_request": parent_request,
        "plateau_experiment_evidence": {"consecutive_plateau_reviews": 3},
    })
    assert [row["experiment_id"] for row in plateau["branch_candidates"]] == ["plateau_lr_conservative"]
    assert plateau["branch_candidates"][0]["request_settings"]["learning_rate"] == 1e-5


def test_campaign_phase_planner_emits_only_exact_boundary_recovery(tmp_path):
    state = _state(tmp_path)
    parent_hash = "3a51ed5ffea719eb62240f720efcbfc78bcaae11f21e34271093cc3cbe7ebda7"
    request = {
        "resume": "logs/rl/official_run_381e8446cc36/checkpoints/"
                 "post_optimizer_update_00000002_steps_000000204800.zip",
        "learning_rate": 3e-5,
        "time_cost_per_simulated_second": TIME_COST_VARIANTS["time_cost_3_calibrated_retry"],
        "exploration_profile": "bc_campaign",
    }
    parent_request = dict(request, resume=(
        "logs/rl/official_run_a6272c234772/checkpoints/automatic_evaluation_000002_steps_200000.zip"
    ))
    recovery = {
        "boundary_recovery": dict(CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE),
        "train_request": request,
        "checkpoint_sha256": CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["checkpoint_sha256"],
        "parent_effective_config": parent_request,
        "parent_effective_config_sha256": campaign_effective_config_sha256(parent_request),
    }
    plan = _plan(state, 6, evidence={"boundary_recovery_candidate": recovery})
    candidates = plan["branch_candidates"]
    assert [row["experiment_id"] for row in candidates] == ["continue_best_boundary_recovery"]
    assert candidates[0]["request_settings"]["resume"] == request["resume"]
    assert candidates[0]["request_settings"]["learning_rate"] == 3e-5
    assert candidates[0]["parent_checkpoint_sha256"] == recovery["checkpoint_sha256"]


def test_campaign_phase_planner_rejects_modified_plan_and_invalid_counters(tmp_path):
    state = _state(tmp_path)
    altered = dict(state, campaign_plan={**state["campaign_plan"], "learning_rate": 1e-4})
    try:
        build_campaign_decision_plan(altered, utc_now=START, monotonic_ns=0, boot_id=BOOT)
    except ValueError as exc:
        assert "modified fixed plan" in str(exc)
    else:
        raise AssertionError("modified plan was accepted")
    try:
        _plan(state, 6, evidence={"consecutive_regressing_probes": -1})
    except ValueError as exc:
        assert "non-negative integer" in str(exc)
    else:
        raise AssertionError("negative decision counter was accepted")
