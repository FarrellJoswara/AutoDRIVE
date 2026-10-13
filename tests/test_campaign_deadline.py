from datetime import datetime, timedelta, timezone
import json
import hashlib
import os
import shutil
import signal
from types import SimpleNamespace

import pytest

from src.layer4.hub.campaign_deadline import (
    CAMPAIGN_PLAN,
    CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE,
    amend_campaign_deadline_policy,
    assert_campaign_allows,
    assert_campaign_training_config,
    TIME_COST_VARIANTS,
    campaign_effective_config_sha256,
    campaign_admission_decision,
    campaign_clock_status,
    campaign_deadline_status,
    effective_campaign_deadline_policy,
    declare_campaign_experiment,
    initialize_campaign,
    load_campaign_state,
    append_campaign_event,
    record_invalidated_time_cost_experiments,
)
from src.layer4.hub.campaign_supervisor import (
    _SIGCONT,
    _SIGKILL,
    force_stop_native_training_run,
    force_stop_windows_simulators,
    supervise_once,
)


START = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
BOOT = "linux-boot-test"
CHECKPOINT = "a" * 64


def _gate(path):
    evidence = {
        "updated_policy_probe": {
            "status": "PASS",
            "useful": True,
            "checkpoint_sha256": CHECKPOINT,
            "run_id": "probe-run",
        },
        "single_environment_queue_interruption": {
            "status": "PASS",
            "passed": True,
            "test_id": "queue-gate",
        },
    }
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_campaign_clock_requires_both_gates_and_cannot_be_restarted(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    gate_path = tmp_path / "gates.json"
    incomplete = {"updated_policy_probe": {"status": "PASS", "useful": True, "checkpoint_sha256": CHECKPOINT}}
    gate_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="single_environment_queue_interruption"):
        initialize_campaign(state_path, gate_path, utc_now=START, monotonic_ns=1000, boot_id=BOOT)
    assert not state_path.exists()

    _gate(gate_path)
    state = initialize_campaign(
        state_path, gate_path, campaign_id="test-campaign",
        utc_now=START, monotonic_ns=1000, boot_id=BOOT,
    )
    assert state["training_freeze_at_utc"] == (START + timedelta(hours=9)).isoformat()
    assert state["hard_deadline_at_utc"] == (START + timedelta(hours=12)).isoformat()
    with pytest.raises(FileExistsError, match="refusing to reset"):
        initialize_campaign(state_path, gate_path, utc_now=START, monotonic_ns=5000, boot_id=BOOT)


def _campaign_config():
    plan = CAMPAIGN_PLAN
    return {
        **{field: plan[field] for field in (
            "learning_rate", "total_timesteps", "stop_on_plateau", "max_duration_hours",
            "behavior_cloning_enabled", "behavior_cloning_only", "observation_profile",
            "simulator_backend", "throttle_mode", "negative_throttle_mode", "steering_mode",
        )},
        "resume": "logs/rl/official_run_89647bf65e50/checkpoints/step4096.zip",
        "target_kl": plan["target_kl"],
        "exploration_profile": plan["exploration_profile"],
        "n_envs": plan["ppo_settings"]["n_envs"],
        "n_steps": plan["ppo_settings"]["n_steps"],
        "batch_size": plan["ppo_settings"]["batch_size"],
        "n_epochs": plan["ppo_settings"]["n_epochs"],
        "gamma": plan["ppo_settings"]["gamma"],
        "gae_lambda": plan["ppo_settings"]["gae_lambda"],
        "clip_range": plan["ppo_settings"]["clip_range"],
        "ent_coef": plan["ppo_settings"]["ent_coef"],
        "vf_coef": plan["ppo_settings"]["vf_coef"],
        "max_grad_norm": plan["ppo_settings"]["max_grad_norm"],
        "device": plan["ppo_settings"]["device"],
        "checkpoint_every": plan["checkpoint_every_transitions"],
        "automatic_evaluation_enabled": plan["automatic_evaluation_enabled"],
        "automatic_evaluation_mode": plan["automatic_evaluation_mode"],
        "automatic_evaluation_interval": plan["automatic_evaluation_interval"],
        **plan["reward_settings"],
    }


def test_campaign_config_is_frozen_only_when_clock_is_active(tmp_path):
    config = _campaign_config()
    missing_campaign = tmp_path / "not_started.json"
    assert_campaign_training_config(config, state_path=missing_campaign)

    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    assert_campaign_training_config(config, state_path=state_path)
    config["learning_rate"] = 1e-5
    with pytest.raises(ValueError, match="learning_rate must be 3e-05"):
        assert_campaign_training_config(config, state_path=state_path)


def test_campaign_config_accepts_trainer_owned_fixed_values_when_omitted(tmp_path):
    config = _campaign_config()
    for field in ("batch_size", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"):
        config.pop(field)

    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    assert_campaign_training_config(config, state_path=state_path)

    # If a caller does expose these fields, they must agree with the values
    # selected by the authenticated bc_campaign preset.
    config["clip_range"] = 0.2
    with pytest.raises(ValueError, match="clip_range must be 0.1"):
        assert_campaign_training_config(config, state_path=state_path)


@pytest.mark.parametrize(("experiment_id", "time_cost"), list(TIME_COST_VARIANTS.items()))
def test_campaign_time_cost_variants_are_append_only_hashed_and_one_variable(tmp_path, experiment_id, time_cost):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=10, boot_id=BOOT)
    config = _campaign_config()
    config["time_cost_per_simulated_second"] = time_cost
    parent_hash = "b" * 64
    declaration = declare_campaign_experiment(
        state_path,
        experiment_id=experiment_id,
        phase="pace_experiments",
        config=config,
        parent_checkpoint_sha256=parent_hash,
        evidence={"baseline_control_parent_sha256": parent_hash,
                  "baseline_control_probe_passed": True,
                  "baseline_control_probe_report_sha256": "e" * 64,
                  "baseline_control_median_scored_lap_s": 105.5, "status": "TBC"},
        utc_now=START + timedelta(hours=3),
        monotonic_ns=3 * 3600 * 1_000_000_000,
    )
    submitted = declaration["config"]
    assert submitted["campaign_experiment_config_sha256"] == campaign_effective_config_sha256(submitted)
    assert_campaign_training_config(submitted, state_path=state_path)
    state = load_campaign_state(state_path)
    assert state["campaign_plan"] == CAMPAIGN_PLAN
    assert state["hard_deadline_at_utc"] == (START + timedelta(hours=12)).isoformat()
    assert [row["event"] for row in state["events"]] == ["campaign_experiment_declared"]

    tampered = dict(submitted)
    tampered["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="hash does not match"):
        assert_campaign_training_config(tampered, state_path=state_path)


def test_campaign_experiment_rejects_wrong_phase_or_unapproved_variable(tmp_path):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    config = _campaign_config()
    config["time_cost_per_simulated_second"] = TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    parent_hash = "c" * 64
    with pytest.raises(ValueError, match="outside its approved"):
        declare_campaign_experiment(
            state_path, experiment_id="time_cost_3_calibrated_retry", phase="pace_experiments", config=config,
            parent_checkpoint_sha256=parent_hash,
            evidence={"baseline_control_parent_sha256": parent_hash,
                      "baseline_control_probe_passed": True,
                      "baseline_control_probe_report_sha256": "e" * 64,
                      "baseline_control_median_scored_lap_s": 105.5},
            utc_now=START + timedelta(hours=6), monotonic_ns=6 * 3600 * 1_000_000_000,
        )
    config["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="unapproved variable"):
        declare_campaign_experiment(
            state_path, experiment_id="time_cost_3_calibrated_retry", phase="pace_experiments", config=config,
            parent_checkpoint_sha256=parent_hash,
            evidence={"baseline_control_parent_sha256": parent_hash,
                      "baseline_control_probe_passed": True,
                      "baseline_control_probe_report_sha256": "e" * 64,
                      "baseline_control_median_scored_lap_s": 105.5},
            utc_now=START + timedelta(hours=3), monotonic_ns=3 * 3600 * 1_000_000_000,
        )


@pytest.mark.parametrize("experiment_id", ["time_cost_3", "time_cost_5"])
def test_campaign_blocks_invalidated_literal_per_second_experiment_ids(tmp_path, experiment_id):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    config = _campaign_config()
    config["time_cost_per_simulated_second"] = 3.0 if experiment_id == "time_cost_3" else 5.0
    config["campaign_experiment_id"] = experiment_id
    config["campaign_experiment_phase"] = "pace_experiments"
    config["campaign_experiment_config_sha256"] = campaign_effective_config_sha256(config)
    with pytest.raises(ValueError, match="invalidated"):
        assert_campaign_training_config(config, state_path=state_path)


def test_legacy_invalid_time_costs_are_append_only_and_do_not_move_clock(tmp_path):
    state_path = tmp_path / "active.json"
    state = initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                                utc_now=START, monotonic_ns=0, boot_id=BOOT)
    declared = _campaign_config()
    declared["time_cost_per_simulated_second"] = 3.0
    append_campaign_event(state_path, {
        "event": "campaign_experiment_declared", "campaign_id": state["campaign_id"],
        "experiment_id": "time_cost_3", "phase": "pace_experiments",
        "config_sha256": "a" * 64, "declared_config": declared,
    }, utc_now=START + timedelta(hours=3), monotonic_ns=3 * 3600 * 1_000_000_000)
    before = load_campaign_state(state_path)
    invalidated = record_invalidated_time_cost_experiments(
        state_path, utc_now=START + timedelta(hours=3, minutes=1),
        monotonic_ns=(3 * 3600 + 60) * 1_000_000_000,
    )
    after = load_campaign_state(state_path)
    assert len(invalidated) == 1
    assert invalidated[0]["experiment_id"] == "time_cost_3"
    assert invalidated[0]["invalidated_config_sha256"] == "a" * 64
    assert invalidated[0]["calibrated_time_cost_per_simulated_second"] == TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    assert after["started_at_utc"] == before["started_at_utc"]
    assert after["training_freeze_at_utc"] == before["training_freeze_at_utc"]
    assert after["hard_deadline_at_utc"] == before["hard_deadline_at_utc"]
    assert record_invalidated_time_cost_experiments(
        state_path, utc_now=START + timedelta(hours=3, minutes=2),
        monotonic_ns=(3 * 3600 + 120) * 1_000_000_000,
    ) == []


def test_optional_lr_variant_requires_best_reward_parent_and_healthy_probe_audits(tmp_path):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    parent_hash = "d" * 64
    config = _campaign_config()
    config["learning_rate"] = 1e-4
    healthy = {
        "best_reward_parent_sha256": parent_hash,
        "best_reward_parent_probe_passed": True,
        "best_reward_parent_probe_report_sha256": "a" * 64,
        "all_finite": True,
        "actor_fault": False,
        "sensor_fault": False,
        "nonregressing_probe_medians": True,
        "passing_zero_collision_probes": 2,
        "passing_zero_collision_probe_medians_s": [105.5, 104.9],
        "passing_zero_collision_probe_report_sha256": ["b" * 64, "c" * 64],
        "last_three_optimizer_audit_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "last_three_approx_kl": [0.008, 0.009, 0.01],
        "last_three_clip_fraction": [0.1, 0.2, 0.3],
    }
    declaration = declare_campaign_experiment(
        state_path, experiment_id="lr_1e4_best_stable", phase="pace_experiments",
        config=config, parent_checkpoint_sha256=parent_hash, evidence=healthy,
        utc_now=START + timedelta(hours=3), monotonic_ns=3 * 3600 * 1_000_000_000,
    )
    assert_campaign_training_config(declaration["config"], state_path=state_path)

    weak_state = tmp_path / "weak.json"
    initialize_campaign(weak_state, _gate(tmp_path / "gates-weak.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    weak = dict(healthy, last_three_approx_kl=[0.01, 0.011, 0.012])
    with pytest.raises(ValueError, match="health evidence"):
        declare_campaign_experiment(
            weak_state, experiment_id="lr_1e4_best_stable", phase="pace_experiments",
            config=config, parent_checkpoint_sha256=parent_hash, evidence=weak,
            utc_now=START + timedelta(hours=3), monotonic_ns=3 * 3600 * 1_000_000_000,
        )


@pytest.mark.parametrize("experiment_id,learning_rate", [
    ("continue_best", 3e-5),
    ("regression_recovery", 1e-5),
])
def test_best_validated_trend_variants_are_parent_relative_and_hashed(tmp_path, experiment_id, learning_rate):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    parent_hash = "d" * 64
    parent_config = _campaign_config()
    parent_config["resume"] = "logs/rl/official_run_parent/checkpoints/parent.zip"
    child_config = dict(parent_config, resume="logs/rl/official_run_parent/checkpoints/next.zip")
    child_config["learning_rate"] = learning_rate
    evidence = {
        "parent_effective_config": parent_config,
        "parent_effective_config_sha256": campaign_effective_config_sha256(parent_config),
    }
    if experiment_id == "continue_best":
        evidence.update({"selected_passing_candidate": True,
                         "selected_passing_checkpoint_sha256": parent_hash})
    else:
        evidence.update({"consecutive_regressing_probes": 2,
                         "infrastructure_failures_excluded": True,
                         "most_recent_passing_checkpoint_sha256": parent_hash})
    declaration = declare_campaign_experiment(
        state_path, experiment_id=experiment_id, phase="best_validated_trend",
        config=child_config, parent_checkpoint_sha256=parent_hash, evidence=evidence,
        utc_now=START + timedelta(hours=6), monotonic_ns=6 * 3600 * 1_000_000_000,
    )
    assert_campaign_training_config(declaration["config"], state_path=state_path)

    tampered = dict(declaration["config"], exploration_max_std_steering=0.8)
    with pytest.raises(ValueError, match="effective-config hash does not match"):
        assert_campaign_training_config(tampered, state_path=state_path)


def test_boundary_recovery_requires_exact_runner_stop_and_checkpoint_evidence(tmp_path):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    parent_hash = CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["parent_probe_sha256"]
    control_config = _campaign_config()
    control_config["resume"] = "logs/rl/official_run_parent/checkpoints/control.zip"
    reward_config = dict(
        control_config,
        time_cost_per_simulated_second=TIME_COST_VARIANTS["time_cost_3_calibrated_retry"],
    )
    reward_declaration = declare_campaign_experiment(
        state_path, experiment_id="time_cost_3_calibrated_retry", phase="pace_experiments",
        config=reward_config, parent_checkpoint_sha256=parent_hash,
        evidence={
            "baseline_control_parent_sha256": parent_hash,
            "baseline_control_probe_passed": True,
            "baseline_control_probe_report_sha256": "f" * 64,
            "baseline_control_median_scored_lap_s": 35.6141,
        }, utc_now=START + timedelta(hours=3), monotonic_ns=3 * 3600 * 1_000_000_000,
    )
    parent_config = reward_declaration["config"]
    child_config = dict(parent_config, resume=(
        "logs/rl/official_run_381e8446cc36/checkpoints/"
        "post_optimizer_update_00000002_steps_000000204800.zip"
    ))
    evidence = {
        "parent_effective_config": parent_config,
        "parent_effective_config_sha256": campaign_effective_config_sha256(parent_config),
        "boundary_recovery": dict(CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE),
        "reward_experiment_config_sha256": reward_declaration["config_sha256"],
    }
    declaration = declare_campaign_experiment(
        state_path, experiment_id="continue_best_boundary_recovery",
        phase="best_validated_trend", config=child_config,
        parent_checkpoint_sha256=CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["checkpoint_sha256"],
        evidence=evidence, utc_now=START + timedelta(hours=6),
        monotonic_ns=6 * 3600 * 1_000_000_000,
    )
    assert declaration["config"]["learning_rate"] == parent_config["learning_rate"]
    assert declaration["config"]["time_cost_per_simulated_second"] == TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    assert_campaign_training_config(declaration["config"], state_path=state_path)

    invalid_state = tmp_path / "invalid.json"
    initialize_campaign(invalid_state, _gate(tmp_path / "gates-invalid.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    bad_evidence = dict(evidence, boundary_recovery={
        **CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE,
        "phase_runner_reason": "ordinary_stop",
    })
    with pytest.raises(ValueError, match="exact authenticated 5h-boundary recovery evidence"):
        declare_campaign_experiment(
            invalid_state, experiment_id="continue_best_boundary_recovery",
            phase="best_validated_trend", config=child_config,
            parent_checkpoint_sha256=CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["checkpoint_sha256"],
            evidence=bad_evidence, utc_now=START + timedelta(hours=6),
            monotonic_ns=6 * 3600 * 1_000_000_000,
        )
    wrong_path_config = dict(child_config, resume="logs/rl/another_run/checkpoints/latest.zip")
    with pytest.raises(ValueError, match="exact saved checkpoint path"):
        declare_campaign_experiment(
            invalid_state, experiment_id="continue_best_boundary_recovery",
            phase="best_validated_trend", config=wrong_path_config,
            parent_checkpoint_sha256=CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["checkpoint_sha256"],
            evidence=evidence, utc_now=START + timedelta(hours=6),
            monotonic_ns=6 * 3600 * 1_000_000_000,
        )


def test_plateau_variant_requires_three_reviews_and_preserves_selected_noise(tmp_path):
    state_path = tmp_path / "active.json"
    initialize_campaign(state_path, _gate(tmp_path / "gates.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    parent_hash = "e" * 64
    parent_config = _campaign_config()
    parent_config["resume"] = "logs/rl/official_run_parent/checkpoints/parent.zip"
    child_config = dict(parent_config, resume="logs/rl/official_run_parent/checkpoints/next.zip",
                        learning_rate=1e-5)
    evidence = {
        "parent_effective_config": parent_config,
        "parent_effective_config_sha256": campaign_effective_config_sha256(parent_config),
        "consecutive_plateau_reviews": 3,
        "scheduled_probe_count": 3,
        "useful_improvement_count": 0,
    }
    declaration = declare_campaign_experiment(
        state_path, experiment_id="plateau_lr_conservative", phase="best_validated_trend",
        config=child_config, parent_checkpoint_sha256=parent_hash, evidence=evidence,
        utc_now=START + timedelta(hours=6), monotonic_ns=6 * 3600 * 1_000_000_000,
    )
    assert_campaign_training_config(declaration["config"], state_path=state_path)

    weak_state = tmp_path / "weak.json"
    initialize_campaign(weak_state, _gate(tmp_path / "gates-weak.json"),
                        utc_now=START, monotonic_ns=0, boot_id=BOOT)
    with pytest.raises(ValueError, match="three scheduled plateau reviews"):
        weak = dict(evidence, consecutive_plateau_reviews=2)
        declare_campaign_experiment(
            weak_state, experiment_id="plateau_lr_conservative", phase="best_validated_trend",
            config=child_config, parent_checkpoint_sha256=parent_hash, evidence=weak,
            utc_now=START + timedelta(hours=6), monotonic_ns=6 * 3600 * 1_000_000_000,
        )
def test_windows_deadline_helper_is_run_scoped_and_requires_verified_json(tmp_path):
    (tmp_path / "scripts").mkdir()
    helper = tmp_path / "scripts" / "terminate_verified_official_simulators.ps1"
    helper.write_text("# test stub\n", encoding="utf-8")
    run_id = "a1b2c3d4e5f6"
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "run_id": run_id,
            "result": "success",
            "api_state": "hard_deadline_override",
            "targets": [],
            "terminated_pids": [4321],
            "already_exited_pids": [],
            "rejected_pids": [],
            "errors": [],
            "recorded_at_utc": "2026-10-12T23:00:00+00:00",
        }), stderr="")

    result = force_stop_windows_simulators(
        run_id, repository_root=tmp_path, runner=runner, hard_deadline_reached=True,
    )
    assert result["status"] == "verified"
    assert result["payload"]["terminated_pids"] == [4321]
    args = calls[0][0]
    assert args[args.index("-RunId") + 1] == run_id
    assert "-ApiStopAttempted" in args and "-GraceExpired" in args
    assert "-HardDeadlineReached" in args

    def mismatched_runner(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({"run_id": "f" * 12}), stderr="")

    mismatch = force_stop_windows_simulators(
        run_id, repository_root=tmp_path, runner=mismatched_runner,
    )
    assert mismatch["status"] == "refused"
    assert mismatch["reason"] == "helper_output_run_id_mismatch"

    malformed = force_stop_windows_simulators(
        run_id,
        repository_root=tmp_path,
        runner=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"run_id": run_id, "result": "terminated"}),
            stderr="",
        ),
    )
    assert malformed["status"] == "failed"
    assert malformed["reason"] == "helper_output_schema_invalid"


def test_deadline_uses_monotonic_and_utc_conservatively_across_monitor_restart(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    state = initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=1_000_000_000, boot_id=BOOT,
    )
    # The monitor restarted, but the operating-system boot identity and
    # monotonic clock are continuous. A backwards wall-clock adjustment cannot
    # buy more time than the persisted monotonic deadline.
    status = campaign_clock_status(
        state, utc_now=START - timedelta(hours=1),
        monotonic_ns=1_000_000_000 + 9 * 3600 * 1_000_000_000,
        boot_id=BOOT,
    )
    assert status["phase"] == "final_validation"
    assert status["remaining_seconds"] == pytest.approx(3 * 3600)

    expired = campaign_clock_status(
        state, utc_now=START + timedelta(hours=11),
        monotonic_ns=1_000_000_000 + 12 * 3600 * 1_000_000_000,
        boot_id=BOOT,
    )
    assert expired["phase"] == "expired"
    assert expired["remaining_seconds"] == 0

    rebooted = campaign_clock_status(
        state, utc_now=START + timedelta(minutes=1), monotonic_ns=500,
        boot_id="different-boot",
    )
    assert rebooted["phase"] == "expired"
    assert rebooted["reason"] == "boot_id_changed"


def test_admission_freezes_training_at_nine_hours_and_all_work_at_twelve(tmp_path):
    state = initialize_campaign(
        tmp_path / "active_campaign.json", _gate(tmp_path / "gates.json"),
        utc_now=START, monotonic_ns=0, boot_id=BOOT,
    )
    allowed, _ = campaign_admission_decision(
        state, "training", utc_now=START + timedelta(hours=8),
        monotonic_ns=8 * 3600 * 1_000_000_000, boot_id=BOOT,
    )
    assert allowed

    allowed, reason = campaign_admission_decision(
        state, "training", utc_now=START + timedelta(hours=9),
        monotonic_ns=9 * 3600 * 1_000_000_000, boot_id=BOOT,
    )
    assert not allowed and "training window ended" in reason
    allowed, _ = campaign_admission_decision(
        state, "evaluation", utc_now=START + timedelta(hours=11),
        monotonic_ns=11 * 3600 * 1_000_000_000, boot_id=BOOT,
    )
    assert allowed
    allowed, _ = campaign_admission_decision(
        state, "evaluation", utc_now=START + timedelta(hours=12),
        monotonic_ns=12 * 3600 * 1_000_000_000, boot_id=BOOT,
    )
    assert not allowed


def test_user_amendment_preserves_original_clock_and_allows_unbounded_work(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    original = initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=1000, boot_id=BOOT,
    )
    append_campaign_event(
        state_path, {"event": "pre_amendment_evidence", "value": "preserved"},
        utc_now=START + timedelta(hours=1), monotonic_ns=1000 + 3600 * 1_000_000_000,
    )
    before = load_campaign_state(state_path)
    before_bytes = state_path.read_bytes()
    original_events = list(before["events"])
    amended = amend_campaign_deadline_policy(
        state_path,
        authorization="user_override",
        reason="User explicitly directed training to continue until readiness without a fixed freeze or deadline.",
        utc_now=START + timedelta(hours=13),
        monotonic_ns=1000 + 13 * 3600 * 1_000_000_000,
    )

    assert amended["started_at_utc"] == original["started_at_utc"]
    assert amended["training_freeze_at_utc"] == original["training_freeze_at_utc"]
    assert amended["hard_deadline_at_utc"] == original["hard_deadline_at_utc"]
    assert amended["start_monotonic_ns"] == original["start_monotonic_ns"]
    assert amended["training_freeze_monotonic_ns"] == original["training_freeze_monotonic_ns"]
    assert amended["hard_deadline_monotonic_ns"] == original["hard_deadline_monotonic_ns"]
    assert amended["events"][:-1] == original_events
    amendment_event = amended["events"][-1]
    assert amendment_event["event"] == "campaign_deadline_policy_amended"
    assert amendment_event["prior_state_sha256"] == hashlib.sha256(before_bytes).hexdigest()
    assert effective_campaign_deadline_policy(amended)["mode"] == "unbounded_until_readiness"

    status = campaign_clock_status(
        amended,
        utc_now=START + timedelta(hours=15),
        monotonic_ns=1000 + 15 * 3600 * 1_000_000_000,
        boot_id=BOOT,
    )
    assert status["phase"] == "training_unbounded"
    assert status["seconds_until_training_freeze"] is None
    assert status["remaining_seconds"] is None
    for kind in ("training", "evaluation"):
        allowed, _ = campaign_admission_decision(
            amended, kind,
            utc_now=START + timedelta(hours=15),
            monotonic_ns=1000 + 15 * 3600 * 1_000_000_000,
            boot_id=BOOT,
        )
        assert allowed
    assert_campaign_allows("training", state_path=state_path)
    health = campaign_deadline_status(state_path=state_path)
    assert health["deadline_policy"]["mode"] == "unbounded_until_readiness"
    assert health["training_freeze_at_utc"] is None
    assert health["hard_deadline_at_utc"] is None

    with pytest.raises(ValueError, match="already been amended"):
        amend_campaign_deadline_policy(
            state_path, authorization="user_override", reason="duplicate", utc_now=START,
        )


def test_unbounded_amendment_requires_user_override_and_valid_audit_binding(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    with pytest.raises(ValueError, match="user_override"):
        amend_campaign_deadline_policy(state_path, authorization="inferred", reason="test")

    amended = amend_campaign_deadline_policy(
        state_path, authorization="user_override", reason="explicit user instruction",
        utc_now=START + timedelta(hours=10), monotonic_ns=10 * 3600 * 1_000_000_000,
    )
    amended["deadline_policy_amendment"]["original_hard_deadline_at_utc"] = "tampered"
    state_path.write_text(json.dumps(amended), encoding="utf-8")
    with pytest.raises(ValueError, match="preserve the original hard deadline"):
        load_campaign_state(state_path)


def test_assertion_and_health_payload_fail_closed_for_corrupt_state(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    state_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="campaign deadline state is invalid"):
        assert_campaign_allows("training", state_path=state_path)
    status = campaign_deadline_status(state_path=state_path)
    assert status["status"] == "invalid"
    assert status["phase"] == "fail_closed"
    assert status["remaining_seconds"] == 0

    assert load_campaign_state(tmp_path / "missing.json") is None

    valid_path = tmp_path / "valid.json"
    valid = initialize_campaign(valid_path, _gate(tmp_path / "valid-gates.json"),
                                utc_now=START, monotonic_ns=0, boot_id=BOOT)
    valid["hard_deadline_monotonic_ns"] += 3600 * 1_000_000_000
    valid_path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="monotonic hard deadline"):
        load_campaign_state(valid_path)


class _FakeSupervisorApi:
    def __init__(self, active):
        self.active = active
        self.calls = []

    def active_work(self, _hub_url):
        if getattr(self, "fail_status", False):
            raise TimeoutError("simulated hub outage")
        return self.active

    def stop_training(self, _hub_url, run_id):
        self.calls.append(("training", run_id))
        return {"state": "stopping"}

    def stop_evaluation(self, _hub_url):
        self.calls.append(("evaluation", None))
        return {"state": "stopping"}

    def escalate_training(self, _hub_url, run_id):
        self.calls.append(("escalate", run_id))
        return {"status": "force_stopped", "transport_acknowledged": False}


def test_supervisor_freezes_training_and_keeps_validation_evaluation_available(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    api = _FakeSupervisorApi({
        "training": [{"run_id": "training-1", "state": "running"}],
        "evaluation": {"run_id": "eval-1", "state": "running"},
    })
    result = supervise_once(
        state_path, utc_now=START + timedelta(hours=9),
        monotonic_ns=9 * 3600 * 1_000_000_000, boot_id=BOOT, api=api,
    )
    assert api.calls == [("training", "training-1")]
    assert result["actions"][0]["reason"] == "training_freeze"
    assert load_campaign_state(state_path)["status"] == "active"


def test_supervisor_requests_bounded_stops_before_hard_deadline_and_marks_expired(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    api = _FakeSupervisorApi({
        "training": [{"run_id": "training-2", "state": "running"}],
        "evaluation": {"run_id": "eval-2", "state": "running"},
    })
    result = supervise_once(
        state_path, utc_now=START + timedelta(hours=12) - timedelta(seconds=30),
        monotonic_ns=12 * 3600 * 1_000_000_000 - 30 * 1_000_000_000,
        boot_id=BOOT, api=api,
    )
    assert api.calls == [("training", "training-2"), ("evaluation", None)]
    assert len(result["actions"]) == 2

    api.active = {"training": [], "evaluation": None}
    expired = supervise_once(
        state_path, utc_now=START + timedelta(hours=12),
        monotonic_ns=12 * 3600 * 1_000_000_000, boot_id=BOOT, api=api,
    )
    assert expired["status"] == "expired"
    assert load_campaign_state(state_path)["status"] == "expired"


def test_supervisor_does_not_apply_old_deadlines_after_user_amendment(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    amend_campaign_deadline_policy(
        state_path,
        authorization="user_override",
        reason="User explicitly directed training until readiness without a fixed deadline.",
        utc_now=START + timedelta(hours=8),
        monotonic_ns=8 * 3600 * 1_000_000_000,
    )
    api = _FakeSupervisorApi({
        "training": [{"run_id": "training-unbounded", "state": "running"}],
        "evaluation": {"run_id": "eval-unbounded", "state": "running"},
    })

    result = supervise_once(
        state_path,
        utc_now=START + timedelta(hours=15),
        monotonic_ns=15 * 3600 * 1_000_000_000,
        boot_id=BOOT,
        api=api,
    )
    assert result["status"] == "active"
    assert result["clock"]["phase"] == "training_unbounded"
    assert result["actions"] == []
    assert api.calls == []
    assert load_campaign_state(state_path)["status"] == "active"


def test_supervisor_keeps_reconciling_active_runs_after_hard_deadline(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    api = _FakeSupervisorApi({
        "training": [{"run_id": "training-3", "state": "running"}],
        "evaluation": None,
    })
    expired_pass = supervise_once(
        state_path, utc_now=START + timedelta(hours=12),
        monotonic_ns=12 * 3600 * 1_000_000_000, boot_id=BOOT, api=api,
    )
    assert expired_pass["status"] == "active"
    assert any(action["kind"] == "training_escalation" for action in expired_pass["actions"])
    assert load_campaign_state(state_path)["status"] == "active"

    api.active = {"training": [], "evaluation": None}
    finalized = supervise_once(
        state_path, utc_now=START + timedelta(hours=12, seconds=1),
        monotonic_ns=12 * 3600 * 1_000_000_000 + 1_000_000_000,
        boot_id=BOOT, api=api,
    )
    assert finalized["status"] == "expired"


def test_supervisor_uses_last_durable_run_ids_when_hub_is_down_at_deadline(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    api = _FakeSupervisorApi({
        "training": [{"run_id": "training-4", "state": "running"}],
        "evaluation": None,
    })
    supervise_once(
        state_path, utc_now=START + timedelta(hours=11),
        monotonic_ns=11 * 3600 * 1_000_000_000, boot_id=BOOT, api=api,
    )
    api.fail_status = True
    result = supervise_once(
        state_path, utc_now=START + timedelta(hours=12),
        monotonic_ns=12 * 3600 * 1_000_000_000, boot_id=BOOT, api=api,
    )
    assert result["status"] == "active"
    assert any(call == ("escalate", "training-4") for call in api.calls)


def test_supervisor_waits_for_graceful_stop_before_escalating_and_rate_limits(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    api = _FakeSupervisorApi({
        "training": [{"run_id": "training-grace", "state": "running"}],
        "evaluation": None,
    })
    freeze_mono = 9 * 3600 * 1_000_000_000
    supervise_once(state_path, utc_now=START + timedelta(hours=9),
                   monotonic_ns=freeze_mono, boot_id=BOOT, api=api)
    assert api.calls == [("training", "training-grace")]

    before_grace = freeze_mono + 59 * 1_000_000_000
    result = supervise_once(state_path, utc_now=START + timedelta(hours=9, seconds=59),
                             monotonic_ns=before_grace, boot_id=BOOT, api=api)
    assert not any(action["kind"] == "training_escalation" for action in result["actions"])
    assert not any(call == ("escalate", "training-grace") for call in api.calls)

    after_grace = freeze_mono + 60 * 1_000_000_000
    result = supervise_once(state_path, utc_now=START + timedelta(hours=9, seconds=60),
                             monotonic_ns=after_grace, boot_id=BOOT, api=api)
    assert api.calls[-1] == ("escalate", "training-grace")
    assert result["actions"][0]["kind"] == "training_escalation"

    supervise_once(state_path, utc_now=START + timedelta(hours=9, seconds=61),
                   monotonic_ns=after_grace + 1_000_000_000, boot_id=BOOT, api=api)
    assert api.calls.count(("escalate", "training-grace")) == 1


def test_supervisor_skips_status_calls_inside_final_second(tmp_path):
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate(tmp_path / "gates.json"), utc_now=START,
        monotonic_ns=0, boot_id=BOOT,
    )
    api = _FakeSupervisorApi({"training": [], "evaluation": None})
    remaining = 0.5
    result = supervise_once(
        state_path, utc_now=START + timedelta(hours=12) - timedelta(seconds=remaining),
        monotonic_ns=12 * 3600 * 1_000_000_000 - int(remaining * 1_000_000_000),
        boot_id=BOOT, api=api,
    )
    assert result["reason"].startswith("hard deadline is within one second")
    assert api.calls == []


def _write_fake_linux_process(proc_root, pid, *, ppid, pgid, starttime, argv):
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    stat_fields = ["S"] + ["0"] * 19
    stat_fields[1] = str(ppid)
    stat_fields[2] = str(pgid)
    stat_fields[19] = str(starttime)
    (process_dir / "stat").write_text(f"{pid} (fake process) " + " ".join(stat_fields), encoding="ascii")
    (process_dir / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")


def _fake_training_manifest(root, proc_root, *, alter_cmdline_pid=None):
    run_id = "campaign-test"
    run_dir = root / "logs" / "rl" / f"official_run_{run_id}"
    run_dir.mkdir(parents=True)
    rows = []
    for role, pid, argv in (
        ("devkit-0", 203, ["ros2", "launch", "bringup"]),
        ("relay-0", 201, ["python", "-m", "src.layer1.official_bridge_relay"]),
        ("trainer", 202, ["python", "-m", "src.layer3.official_train"]),
    ):
        _write_fake_linux_process(proc_root, pid, ppid=99, pgid=pid, starttime=pid + 10,
                                  argv=(argv + ["tampered"] if pid == alter_cmdline_pid else argv))
        rows.append({"role": role, "pid": pid, "process_group_id": pid,
                     "process_starttime_ticks": pid + 10, "handle_origin": "created_by_popen",
                     "process_handle_class": "subprocess.Popen", "launch_parent_pid": 99,
                     "launch_command": argv,
                     "log": str(run_dir / ("relay_env0.log" if role.startswith("relay-") else f"{role}.log"))})
    manifest = {"hub_pid": 99, "backend_instance_id": "instance-test",
                "Labels": {"aicar.run_id": run_id, "aicar.kind": "training",
                           "aicar.output_dir": str(run_dir).replace("\\", "/")},
                "records": rows}
    (run_dir / "native_group_api.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_id


def test_emergency_cleanup_neutralizes_relay_then_kills_only_identity_matched_groups(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    run_id = _fake_training_manifest(tmp_path, proc_root)
    sent, waits = [], []

    def fake_killpg(pgid, sig):
        sent.append((pgid, sig))
        if pgid == 201 and sig == signal.SIGTERM:
            relay_log = tmp_path / "logs" / "rl" / f"official_run_{run_id}" / "relay_env0.log"
            relay_log.write_text(
                json.dumps({"event": "shutdown_neutral_command_sent", "sid": "test"}) + "\n"
                + json.dumps({"event": "shutdown_neutral_release_complete", "attempted": 1, "sent": 1,
                              "failed": 0, "transport_acknowledged": False}) + "\n",
                encoding="utf-8",
            )
        if sig == _SIGKILL:
            shutil.rmtree(proc_root / str(pgid), ignore_errors=True)

    result = force_stop_native_training_run(
        run_id, repository_root=tmp_path, proc_root=proc_root,
        signal_group=fake_killpg, sleep=waits.append,
    )
    assert result["status"] == "force_stopped"
    assert result["neutral_release_sent"] is True
    assert result["neutral_release_reports"][0]["sent"] == 1
    assert result["transport_acknowledged"] is False
    assert waits == [0.6, 0.8]
    assert sent[:2] == [(201, _SIGCONT), (201, signal.SIGTERM)]
    assert (202, _SIGKILL) in sent and (203, _SIGKILL) in sent


def test_emergency_cleanup_refuses_cmdline_mismatch_without_signaling(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    run_id = _fake_training_manifest(tmp_path, proc_root, alter_cmdline_pid=202)
    sent = []
    result = force_stop_native_training_run(
        run_id, repository_root=tmp_path, proc_root=proc_root,
        signal_group=lambda *args: sent.append(args), sleep=lambda _seconds: None,
    )
    assert result["status"] == "refused"
    assert "identity changed" in result["reason"]
    assert sent == []


def test_emergency_cleanup_does_not_wait_for_neutral_log_before_identity_checked_termination(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    run_id = _fake_training_manifest(tmp_path, proc_root)

    def fake_killpg(pgid, sig):
        if sig == _SIGKILL:
            shutil.rmtree(proc_root / str(pgid), ignore_errors=True)

    result = force_stop_native_training_run(
        run_id, repository_root=tmp_path, proc_root=proc_root,
        signal_group=fake_killpg, sleep=lambda _seconds: None,
    )
    assert result["status"] == "force_stopped"
    assert result["neutral_release_sent"] is False
    assert result["transport_acknowledged"] is False
    assert result["simulator_cleanup"].startswith("Windows simulator cleanup")
