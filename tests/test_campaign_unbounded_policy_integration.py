"""Regression coverage for the explicit no-fixed-cutoff campaign override."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.layer4.hub.campaign_deadline import (
    CAMPAIGN_ID,
    CAMPAIGN_PLAN,
    amend_campaign_deadline_policy,
    append_campaign_event,
    assert_campaign_allows,
    assert_campaign_training_config,
    campaign_admission_decision,
    campaign_clock_status,
    campaign_deadline_status,
    campaign_effective_config_sha256,
    initialize_campaign,
    load_campaign_state,
    request_campaign_final_validation,
)
from src.layer4.hub import campaign_phase_runner as runner_module
from src.layer4.hub.campaign_phase_runner import _sensor_fault_from_freshness
from src.layer4.hub.campaign_phase_runner import _valid_completed_rollout_steps
from src.layer4.hub.campaign_supervisor import force_stop_native_training_run, supervise_once
from src.layer4.hub.campaign_experiments import build_campaign_decision_plan
import src.layer4.hub.campaign_finalists as campaign_finalists


START = datetime(2026, 10, 12, 23, 26, 44, tzinfo=timezone.utc)
BOOT = "test-boot-id"
MONO_START = 17_000
CHECKPOINT_SHA = "a" * 64


def _gate_file(path: Path) -> Path:
    path.write_text(json.dumps({
        "updated_policy_probe": {
            "status": "PASS", "useful": True,
            "checkpoint_sha256": CHECKPOINT_SHA, "run_id": "accepted-probe",
        },
        "single_environment_queue_interruption": {
            "status": "PASS", "passed": True, "test_id": "queue-interruption",
        },
    }), encoding="utf-8")
    return path


def _campaign_config() -> dict:
    plan = CAMPAIGN_PLAN
    return {
        **{key: plan[key] for key in (
            "learning_rate", "total_timesteps", "stop_on_plateau", "max_duration_hours",
            "behavior_cloning_enabled", "behavior_cloning_only", "observation_profile",
            "simulator_backend", "throttle_mode", "negative_throttle_mode", "steering_mode",
        )},
        "resume": "logs/rl/official_run_parent/official_policy_latest.zip",
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


def _amended_state(tmp_path: Path) -> tuple[Path, dict]:
    state_path = tmp_path / "active_campaign.json"
    initialize_campaign(
        state_path, _gate_file(tmp_path / "gate.json"), campaign_id=CAMPAIGN_ID,
        utc_now=START, monotonic_ns=MONO_START, boot_id=BOOT,
    )
    append_campaign_event(
        state_path, {"event": "preserved_evidence", "evidence_id": "before-override"},
        utc_now=START + timedelta(hours=1), monotonic_ns=MONO_START + 3600_000_000_000,
    )
    pre_amend_bytes = state_path.read_bytes()
    pre_amend = load_campaign_state(state_path)
    amended = amend_campaign_deadline_policy(
        state_path, authorization="user_override",
        reason="User explicitly removed fixed freeze/deadline; continue until readiness.",
        utc_now=START + timedelta(hours=13),
        monotonic_ns=MONO_START + 13 * 3600_000_000_000,
    )
    assert amended["events"][:-1] == pre_amend["events"]
    event = amended["events"][-1]
    assert event["event"] == "campaign_deadline_policy_amended"
    assert event["prior_state_sha256"] == hashlib.sha256(pre_amend_bytes).hexdigest()
    for field in (
        "started_at_utc", "training_freeze_at_utc", "hard_deadline_at_utc",
        "start_monotonic_ns", "training_freeze_monotonic_ns", "hard_deadline_monotonic_ns",
        "campaign_plan",
    ):
        assert amended[field] == pre_amend[field]
    return state_path, amended


def test_unbounded_policy_preserves_clock_history_and_only_removes_time_cutoffs(tmp_path):
    state_path, state = _amended_state(tmp_path)
    now = START + timedelta(hours=16)
    mono = MONO_START + 16 * 3600_000_000_000

    clock = campaign_clock_status(state, utc_now=now, monotonic_ns=mono, boot_id=BOOT)
    assert clock["phase"] == "training_unbounded"
    assert clock["seconds_until_training_freeze"] is None
    assert clock["remaining_seconds"] is None
    assert_campaign_allows("training", state_path=state_path)
    assert_campaign_allows("evaluation", state_path=state_path)
    assert campaign_deadline_status(state_path=state_path)["deadline_policy"]["mode"] == "unbounded_until_readiness"
    for kind in ("training", "evaluation"):
        allowed, _ = campaign_admission_decision(
            state, kind, utc_now=now, monotonic_ns=mono, boot_id=BOOT,
        )
        assert allowed


def test_amended_policy_keeps_frozen_training_config_and_reward_contract(tmp_path):
    state_path, _ = _amended_state(tmp_path)
    config = _campaign_config()
    assert_campaign_training_config(config, state_path=state_path)
    assert campaign_effective_config_sha256(config)

    config["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="learning_rate"):
        assert_campaign_training_config(config, state_path=state_path)


class _Api:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []
        self.fail = False

    def active_work(self, _url):
        if self.fail:
            raise TimeoutError("read-only test outage")
        return {
            "training": [{"run_id": "healthy-training", "state": "running"}],
            "evaluation": {"run_id": "active-finalist", "state": "running"},
        }

    def stop_training(self, _url, run_id):
        self.calls.append(("stop_training", run_id))
        return {"state": "stopping"}

    def stop_evaluation(self, _url):
        self.calls.append(("stop_evaluation", None))
        return {"state": "stopping"}

    def escalate_training(self, _url, run_id):
        self.calls.append(("escalate_training", run_id))
        return {"status": "force_stopped"}


def test_supervisor_keeps_healthy_work_running_after_old_hard_deadline(tmp_path):
    state_path, _ = _amended_state(tmp_path)
    api = _Api()
    result = supervise_once(
        state_path, utc_now=START + timedelta(hours=20),
        monotonic_ns=MONO_START + 20 * 3600_000_000_000,
        boot_id=BOOT, api=api,
    )
    assert result["status"] == "active"
    assert result["actions"] == []
    assert api.calls == []
    assert load_campaign_state(state_path)["status"] == "active"


def test_unbounded_supervisor_api_outage_does_not_trigger_time_fallback_stop(tmp_path):
    state_path, _ = _amended_state(tmp_path)
    api = _Api()
    api.fail = True
    result = supervise_once(
        state_path, utc_now=START + timedelta(hours=20),
        monotonic_ns=MONO_START + 20 * 3600_000_000_000,
        boot_id=BOOT, api=api,
    )
    assert result["status"] == "active"
    assert result["actions"] == []
    assert "read-only test outage" in result["error"]
    assert api.calls == []
    assert load_campaign_state(state_path)["status"] == "active"


def test_unbounded_campaign_keeps_watch_sensor_fault_gate_and_process_identity_guard(tmp_path):
    state_path, _ = _amended_state(tmp_path)
    stale_watch = {"watch": {"age_s": 46.0, "valid_sensor_frame": False,
                              "reset_in_progress": False}, "sensor_fault": True}
    assert _sensor_fault_from_freshness(stale_watch, replay_active=False) is True

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    run_id = "bad-owner-run"
    run_dir = tmp_path / "logs" / "rl" / f"official_run_{run_id}"
    run_dir.mkdir(parents=True)
    (run_dir / "native_group_api.json").write_text(json.dumps({
        "hub_pid": 77,
        "backend_instance_id": "test-backend",
        "Labels": {
            "aicar.run_id": run_id,
            "aicar.output_dir": str(run_dir).replace("\\", "/"),
            "aicar.kind": "training",
        },
        "records": [{"role": "hub", "pid": 77, "process_group_id": 77,
                     "process_starttime_ticks": 10, "launch_parent_pid": 1,
                     "launch_command": ["python", "hub"],
                     "handle_origin": "created_by_popen",
                     "process_handle_class": "subprocess.Popen"}],
    }), encoding="utf-8")
    signals: list[tuple[int, int]] = []
    result = force_stop_native_training_run(
        run_id, repository_root=tmp_path,
        signal_group=lambda pgid, sig: signals.append((pgid, sig)),
        proc_root=proc_root,
    )
    assert result["status"] == "refused"
    assert "non-owned process role" in result["reason"]
    assert signals == []
    assert load_campaign_state(state_path)["status"] == "active"


def test_unbounded_policy_does_not_admit_nonfinite_optimizer_audits(tmp_path):
    state_path, _ = _amended_state(tmp_path)
    finite = {
        "num_timesteps": 10_240,
        "metrics_finite": True,
        "gradients_finite": True,
        "parameters_finite": True,
        "optimizer_state_finite": True,
    }
    nonfinite = {**finite, "num_timesteps": 12_288, "parameters_finite": False}
    assert _valid_completed_rollout_steps([finite, nonfinite], parent_step=8_192, trigger_step=16_384) == {10_240}
    assert load_campaign_state(state_path)["deadline_policy_amendment"]["policy"] == "unbounded_until_readiness"


def test_readiness_request_is_manual_transition_to_hash_bound_finalists(tmp_path, monkeypatch):
    state_path, state = _amended_state(tmp_path)
    now = START + timedelta(hours=16)
    mono = MONO_START + 16 * 3600_000_000_000
    plan = build_campaign_decision_plan(state, utc_now=now, monotonic_ns=mono, boot_id=BOOT)
    assert plan["phase"] == "best_validated_trend"

    candidate = {
        "checkpoint_sha256": "c" * 64,
        "evaluation_id": "readiness-evaluation-1",
        "probe_report_sha256": "d" * 64,
        "median_scored_lap_s": 25.0,
    }
    evidence = {"campaign_id": CAMPAIGN_ID, "candidates": [candidate]}
    with pytest.raises(ValueError, match="readiness-backed validation request"):
        campaign_finalists.build_finalist_manifest(state, evidence, repo_root=tmp_path,
                                                   utc_now=now, monotonic_ns=mono, boot_id=BOOT)

    request = request_campaign_final_validation(
        state_path,
        candidate_checkpoint_sha256=candidate["checkpoint_sha256"],
        evaluation_id=candidate["evaluation_id"],
        probe_report_sha256=candidate["probe_report_sha256"],
        readiness_evidence={
            "stable_candidate_verified": True,
            "source_evidence_sha256": "e" * 64,
            "recent_audit_evidence_sha256": "f" * 64,
            "recent_audits_finite": True,
            "sensor_health_passed": True,
            "zero_collision_probes": True,
            "passing_nonregressing_probe_count": 2,
            "sol_readiness_recommendation": {
                "decision": "request_final_validation",
                "reviewer": "sol",
                "reason": "The authenticated candidate has reached a useful stable trend and is ready for competitive validation.",
                "candidate_checkpoint_sha256": "c" * 64,
                "recommendation_evidence_sha256": "9" * 64,
            },
        },
        utc_now=now, monotonic_ns=mono,
    )
    amended = load_campaign_state(state_path)
    plan = build_campaign_decision_plan(amended, utc_now=now, monotonic_ns=mono, boot_id=BOOT)
    assert plan["phase"] == "final_validation"
    allowed, reason = campaign_admission_decision(
        amended, "training", utc_now=now, monotonic_ns=mono, boot_id=BOOT,
    )
    assert not allowed and "training is frozen" in reason
    assert campaign_admission_decision(
        amended, "evaluation", utc_now=now, monotonic_ns=mono, boot_id=BOOT,
    )[0]

    class _VerificationContext:
        def seal(self):
            return True

    monkeypatch.setattr(campaign_finalists, "PPOVerificationContext", _VerificationContext)
    monkeypatch.setattr(campaign_finalists, "_probe_candidate", lambda row, *_args: dict(row))
    manifest = campaign_finalists.build_finalist_manifest(
        amended, evidence, repo_root=tmp_path,
        utc_now=now, monotonic_ns=mono, boot_id=BOOT,
    )
    assert manifest["validation_request_id"] == request["request_id"]
    assert [row["checkpoint_sha256"] for row in manifest["finalists"]] == [candidate["checkpoint_sha256"]]
    assert manifest["final_validation"] == {
        "attempts_per_finalist": 3,
        "warmup_laps_per_attempt": 1,
        "scored_laps_per_attempt": 10,
        "zero_collisions_required": True,
        "clean_reload_required": True,
    }

    wrong_candidate = {**candidate, "probe_report_sha256": "9" * 64}
    with pytest.raises(ValueError, match="exact candidate/report"):
        campaign_finalists.build_finalist_manifest(
            amended, {"campaign_id": CAMPAIGN_ID, "candidates": [wrong_candidate]}, repo_root=tmp_path,
            utc_now=now, monotonic_ns=mono, boot_id=BOOT,
        )


@pytest.mark.parametrize("recommendation", [
    None,
    {
        "decision": "request_final_validation",
        "reviewer": "agent",
        "reason": "candidate is stable",
        "candidate_checkpoint_sha256": "c" * 64,
        "recommendation_evidence_sha256": "9" * 64,
    },
])
def test_validation_request_rejects_missing_or_non_sol_readiness_attestation(tmp_path, recommendation):
    from src.layer4.hub.campaign_deadline import campaign_validation_request

    state_path, _ = _amended_state(tmp_path)
    readiness = {
        "stable_candidate_verified": True,
        "source_evidence_sha256": "e" * 64,
        "recent_audit_evidence_sha256": "f" * 64,
        "recent_audits_finite": True,
        "sensor_health_passed": True,
        "zero_collision_probes": True,
        "passing_nonregressing_probe_count": 2,
    }
    if recommendation is not None:
        readiness["sol_readiness_recommendation"] = recommendation

    with pytest.raises(ValueError, match="stable candidate readiness evidence"):
        request_campaign_final_validation(
            state_path,
            candidate_checkpoint_sha256="c" * 64,
            evaluation_id="readiness-eval",
            probe_report_sha256="d" * 64,
            readiness_evidence=readiness,
            utc_now=START + timedelta(hours=16),
            monotonic_ns=MONO_START + 16 * 3600_000_000_000,
        )
    assert campaign_validation_request(load_campaign_state(state_path)) is None


class _RunnerApi:
    def __init__(self):
        self.posts = []

    def post(self, path, payload=None):
        self.posts.append((path, payload))
        return {"state": "stopping"}


def _campaign_runner(tmp_path, monkeypatch, state_path, snapshot):
    api = _RunnerApi()
    runner = runner_module.CampaignPhaseRunner(
        state_path=state_path, repo_root=tmp_path, api=api,
        finalists_manifest=tmp_path / "finalists.json",
    )
    monkeypatch.setattr(runner, "_api_snapshot", lambda: snapshot)
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: {})
    return runner, api


def _pace_training_row(state="running"):
    from src.layer4.hub.campaign_deadline import TIME_COST_VARIANTS

    return {
        "run_id": "campaign-branch",
        "kind": "training",
        "state": state,
        "display_name": "Campaign corrected retry",
        "config": {
            "campaign_experiment_id": "time_cost_3_calibrated_retry",
            "time_cost_per_simulated_second": TIME_COST_VARIANTS["time_cost_3_calibrated_retry"],
        },
    }


def test_unbounded_phase_runner_does_not_stop_pace_branch_or_create_finalists(tmp_path, monkeypatch):
    from src.layer4.hub import campaign_phase_runner as runner_module

    state_path, _ = _amended_state(tmp_path)
    snapshot = {
        "runs": [_pace_training_row()], "evaluation_history": [], "telemetry": {},
        "replay": {"state": "idle"}, "health": {"status": "ok"},
    }
    runner, api = _campaign_runner(tmp_path, monkeypatch, state_path, snapshot)
    result = runner.tick()

    assert result["phase"] == "best_validated_trend"
    assert result["action"] == "observe_only"
    assert api.posts == []
    assert not runner.finalists_manifest.exists()


def test_improving_probes_do_not_automatically_request_final_validation(tmp_path, monkeypatch):
    """Removing the clock cutoff does not implicitly authorize freezing training."""
    from src.layer4.hub.campaign_deadline import campaign_validation_request

    state_path, _ = _amended_state(tmp_path)
    request = {}
    branch_hash = campaign_effective_config_sha256(request)
    first_sha, second_sha = "1" * 64, "2" * 64
    now_text = (START + timedelta(hours=16)).isoformat()

    def probe_row(evaluation_id, model_sha, median, minute):
        return {
            "evaluation_id": evaluation_id,
            "parent_run_id": "campaign-branch",
            "state": "complete",
            "model_sha256": model_sha,
            "model_lineage_verified": True,
            "report_protocol": "official_training_probe_1_plus_3",
            "probe": {"readiness_passed": True},
            "train_request": request,
            "trend_validation": {
                "authenticated": True,
                "scheduled_probe": True,
                "protocol_valid": True,
                "sensor_valid": True,
                "loaded_model_sha256": model_sha,
                "outcome": "pass",
                "collision_count": 0,
                "completed_rollouts_since_branch": 2,
                "median_scored_lap_s": median,
                "timestamp_utc": (START + timedelta(hours=16, minutes=minute)).isoformat(),
            },
        }

    evidence = {
        "sensor_fault": False,
        "nonfinite_update": False,
        "actor_fault": False,
        "trend_selected_branch_config_sha256": branch_hash,
        "best_trend_checkpoint_sha256": second_sha,
        "probe_evidence": [
            probe_row("probe-1", first_sha, 40.0, 1),
            probe_row("probe-2", second_sha, 33.6, 2),
        ],
        "active_run_diagnostics": [{
            "run_id": "campaign-branch",
            "last_three_audits": [
                {"num_timesteps": 4096, "metrics_finite": True, "gradients_finite": True,
                 "parameters_finite": True, "optimizer_state_finite": True},
                {"num_timesteps": 6144, "metrics_finite": True, "gradients_finite": True,
                 "parameters_finite": True, "optimizer_state_finite": True},
            ],
            "audit_log_sha256": "3" * 64,
            "last_three_audit_record_sha256": "4" * 64,
            "telemetry": {"local_liveness": {"watch": {
                "valid_sensor_frame": True, "reset_in_progress": False, "age_s": 0.1,
                "timestamp_utc": now_text, "step": 6144,
            }}},
        }],
    }
    snapshot = {
        "runs": [_pace_training_row()], "evaluation_history": [], "telemetry": {},
        "replay": {"state": "idle"}, "health": {"status": "ok"},
    }
    runner, api = _campaign_runner(tmp_path, monkeypatch, state_path, snapshot)
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: evidence)

    result = runner.tick()

    assert result["phase"] == "best_validated_trend"
    assert api.posts == []
    assert campaign_validation_request(load_campaign_state(state_path)) is None
    assert not runner.finalists_manifest.exists()


def test_readiness_request_stops_only_via_api_and_waits_for_quiescence_and_manifest_gate(tmp_path, monkeypatch):
    from src.layer4.hub import campaign_phase_runner as runner_module

    state_path, _ = _amended_state(tmp_path)
    request_campaign_final_validation(
        state_path,
        candidate_checkpoint_sha256="c" * 64,
        evaluation_id="readiness-eval",
        probe_report_sha256="d" * 64,
        readiness_evidence={
            "stable_candidate_verified": True,
            "source_evidence_sha256": "e" * 64,
            "recent_audit_evidence_sha256": "f" * 64,
            "recent_audits_finite": True,
            "sensor_health_passed": True,
            "zero_collision_probes": True,
            "passing_nonregressing_probe_count": 2,
            "sol_readiness_recommendation": {
                "decision": "request_final_validation",
                "reviewer": "sol",
                "reason": "The authenticated candidate has reached a useful stable trend and is ready for competitive validation.",
                "candidate_checkpoint_sha256": "c" * 64,
                "recommendation_evidence_sha256": "9" * 64,
            },
        },
        utc_now=START + timedelta(hours=16),
        monotonic_ns=MONO_START + 16 * 3600_000_000_000,
    )
    running_snapshot = {
        "runs": [_pace_training_row()], "evaluation_history": [], "telemetry": {},
        "replay": {"state": "idle"}, "health": {"status": "ok"},
    }
    runner, api = _campaign_runner(tmp_path, monkeypatch, state_path, running_snapshot)
    stopped = runner.tick()

    assert stopped["phase"] == "final_validation"
    assert stopped["action"] == "stop_training_at_freeze"
    assert api.posts == [("/train/runs/campaign-branch/stop", None)]
    assert not runner.finalists_manifest.exists()

    # A live replay blocks finalist publication even after training is stopped.
    api.posts.clear()
    replay_snapshot = {
        **running_snapshot, "runs": [],
        "replay": {"state": "running", "evaluation_id": "existing-eval"},
    }
    monkeypatch.setattr(runner, "_api_snapshot", lambda: replay_snapshot)
    waiting = runner.tick()
    assert waiting["action"] == "wait_for_existing_evaluation"
    assert api.posts == []
    assert not runner.finalists_manifest.exists()

    # Once idle, the real create-only manifest gate runs; with no authenticated
    # candidate evidence it remains closed rather than publishing a shortlist.
    idle_snapshot = {**replay_snapshot, "replay": {"state": "idle"}}
    monkeypatch.setattr(runner, "_api_snapshot", lambda: idle_snapshot)
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: {"probe_evidence": []})
    idle_result = runner.tick()
    assert idle_result["action"] == "waiting_for_authenticated_passing_probe_candidate"
    assert not runner.finalists_manifest.exists()
    assert api.posts == []
