from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import src.layer4.hub.campaign_phase_runner as runner_module
from src.layer4.hub.campaign_phase_runner import (
    _local_run_liveness,
    _model_alias,
    _resolve_model_path,
    _campaign_run_reward_is_eligible,
    _pace_phase_boundary_stop_allowed,
    _campaign_evidence_state_signature,
    _pace_probe_is_screening_complete,
    _declared_reward_branch_hash,
    _is_calibrated_pace_branch,
    _sensor_fault_from_freshness,
    _valid_completed_rollout_steps,
    validate_final_evaluation,
    validate_training_probe,
)


SHA = "a" * 64


def test_trend_reward_declaration_is_hash_bound_to_tested_branch():
    from src.layer4.hub.campaign_deadline import TIME_COST_VARIANTS

    experiment_id = "time_cost_3_calibrated_retry"
    config_hash = "b" * 64
    request = {
        "time_cost_per_simulated_second": TIME_COST_VARIANTS[experiment_id],
        "campaign_experiment_id": experiment_id,
        "campaign_experiment_config_sha256": config_hash,
    }
    event = {
        "event": "campaign_experiment_declared", "experiment_id": experiment_id,
        "config_sha256": config_hash,
        "declared_config": {"time_cost_per_simulated_second": TIME_COST_VARIANTS[experiment_id]},
    }
    assert _declared_reward_branch_hash({"events": [event]}, request) == config_hash
    assert _declared_reward_branch_hash({"events": []}, request) is None
    assert _declared_reward_branch_hash({"events": [event]}, {**request, "time_cost_per_simulated_second": 3.0}) is None
    assert _declared_reward_branch_hash({"events": [event]}, {"time_cost_per_simulated_second": 0.0}) == ""


def test_inherited_reward_links_continue_best_to_original_calibrated_declaration():
    from src.layer4.hub.campaign_deadline import TIME_COST_VARIANTS

    declared_hash = "b" * 64
    coefficient = TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    state = {"events": [{
        "event": "campaign_experiment_declared",
        "experiment_id": "time_cost_3_calibrated_retry",
        "config_sha256": declared_hash,
        "declared_config": {"time_cost_per_simulated_second": coefficient},
    }]}
    child_request = {
        "campaign_experiment_id": "continue_best_boundary_recovery",
        "campaign_experiment_config_sha256": "c" * 64,
        "time_cost_per_simulated_second": coefficient,
    }

    assert _declared_reward_branch_hash(state, child_request) == declared_hash


def test_collect_evidence_rejects_dependency_mutated_with_mtime_restored(tmp_path: Path, monkeypatch):
    from src.layer4.hub.campaign_deadline import CAMPAIGN_ID

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=object(),
    )
    state = {"campaign_id": CAMPAIGN_ID, "status": "active", "events": []}
    runner.state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner_module, "load_campaign_state", lambda _path: state)
    monkeypatch.setattr(runner_module, "campaign_clock_status", lambda _state: {"phase": "training_window"})

    artifact = tmp_path / "candidate.zip"
    artifact.write_bytes(b"authenticated-bytes")

    def mutate_during_collection(_snapshot, _state, verification_context):
        verification_context.hash_file(artifact)
        original = artifact.stat()
        artifact.write_bytes(b"tampered-bytes!!")
        os.utime(artifact, ns=(original.st_atime_ns, original.st_mtime_ns))
        return []

    monkeypatch.setattr(runner, "_report_evidence", mutate_during_collection)
    with pytest.raises(RuntimeError, match="dependency changed during collection"):
        runner.collect_evidence({"runs": [], "evaluation_history": [], "telemetry": {}, "replay": {}})


def test_collect_evidence_does_not_select_a_clean_seed_probe_without_two_updates(tmp_path: Path, monkeypatch):
    from src.layer4.hub.campaign_deadline import CAMPAIGN_ID

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=object(),
    )
    state = {"campaign_id": CAMPAIGN_ID, "status": "active", "events": []}
    runner.state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner_module, "load_campaign_state", lambda _path: state)
    monkeypatch.setattr(runner_module, "campaign_clock_status", lambda _state: {"phase": "training_window"})
    model = tmp_path / "logs" / "rl" / "official_run_child" / "checkpoints" / "seed.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"seed")
    monkeypatch.setattr(runner, "_report_evidence", lambda _snapshot, _state, _context: [{
        "evaluation_id": "seed-eval", "parent_run_id": "child", "model_path": str(model),
        "model_sha256": "a" * 64, "report_sha256": "b" * 64,
        "probe": {"median_scored_lap_s": 34.0}, "train_request": {"time_cost_per_simulated_second": 0.0},
        "trend_validation": None,
    }])

    evidence = runner.collect_evidence({
        "runs": [{"run_id": "child", "config": {"time_cost_per_simulated_second": 0.0}}],
        "evaluation_history": [], "telemetry": {}, "replay": {},
    })
    assert evidence["passing_candidate_count"] == 0
    assert "selected_passing_checkpoint_sha256" not in evidence


def test_five_hour_boundary_only_stops_declared_pace_experiments():
    from src.layer4.hub.campaign_deadline import TIME_COST_VARIANTS

    coefficient = TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]
    assert _is_calibrated_pace_branch({
        "campaign_experiment_id": "time_cost_3_calibrated_retry",
        "time_cost_per_simulated_second": coefficient,
    })
    assert not _is_calibrated_pace_branch({
        "campaign_experiment_id": "continue_best",
        "time_cost_per_simulated_second": coefficient,
    })
    assert not _is_calibrated_pace_branch({
        "campaign_experiment_id": "time_cost_3_calibrated_retry",
        "time_cost_per_simulated_second": 3.0,
    })


def test_declared_but_unposted_candidate_intent_can_retry_after_validation_failure(tmp_path: Path):
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=object(),
    )
    intent = {
        "event": "candidate_launch_intent", "intent_id": "intent-1",
        "experiment_id": "continue_best", "phase": "best_validated_trend",
        "config_sha256": "b" * 64, "resume_path": "/runs/rl/parent.zip",
    }
    aborted = {
        "event": "candidate_launch_aborted", "intent_id": "intent-1",
        "experiment_id": "continue_best", "stage": "declaration",
    }
    runner.output_dir.mkdir(parents=True)
    runner.events_path.write_text("\n".join(json.dumps(row) for row in (intent, aborted)) + "\n", encoding="utf-8")
    assert runner._unresolved_candidate_intent([]) is None


def test_known_legacy_prepost_failure_can_be_resolved_by_timestamp(tmp_path: Path):
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=object(),
    )
    timestamp = "2026-10-13T04:35:18.414784+00:00"
    intent = {
        "event": "candidate_launch_intent", "experiment_id": "continue_best",
        "phase": "best_validated_trend", "config_sha256": "b" * 64,
        "resume_path": "/runs/rl/parent.zip", "timestamp_utc": timestamp,
    }
    aborted = {
        "event": "candidate_launch_aborted", "experiment_id": "continue_best",
        "config_sha256": "b" * 64, "stage": "declaration", "intent_timestamp_utc": timestamp,
    }
    runner.output_dir.mkdir(parents=True)
    runner.events_path.write_text("\n".join(json.dumps(row) for row in (intent, aborted)) + "\n", encoding="utf-8")
    assert runner._unresolved_candidate_intent([]) is None


def test_five_hour_boundary_waits_for_active_probe_but_allows_normal_stop_after():
    assert not _pace_phase_boundary_stop_allowed(5.0, replay_active=True)
    assert not _pace_phase_boundary_stop_allowed(4.99, replay_active=False)
    assert _pace_phase_boundary_stop_allowed(5.0, replay_active=False)


def test_trend_rollout_gate_counts_distinct_updates_only_through_snapshot():
    valid = {"metrics_finite": True, "gradients_finite": True,
             "parameters_finite": True, "optimizer_state_finite": True}
    rows = [
        {**valid, "num_timesteps": 152048},
        {**valid, "num_timesteps": 152048},  # duplicate audit line is not another rollout
        {**valid, "num_timesteps": 154096},
        {**valid, "num_timesteps": 156144},  # after the evaluated snapshot
        {**valid, "num_timesteps": 153072, "optimizer_state_finite": False},
    ]
    assert _valid_completed_rollout_steps(rows, parent_step=150000, trigger_step=154096) == {
        152048, 154096,
    }


def test_pace_branch_ignores_resume_seed_probe_until_two_new_updates(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "logs" / "rl" / "official_run_child"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    parent = tmp_path / "logs" / "rl" / "official_run_parent" / "checkpoints" / "parent.zip"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"parent")
    model = checkpoint_dir / "automatic_evaluation_000001_steps_151553.zip"
    model.write_bytes(b"child")
    audit = [
        {"num_timesteps": 152048, "metrics_finite": True, "gradients_finite": True,
         "parameters_finite": True, "optimizer_state_finite": True},
        {"num_timesteps": 154096, "metrics_finite": True, "gradients_finite": True,
         "parameters_finite": True, "optimizer_state_finite": True},
    ]
    (run_dir / "ppo_optimizer_audit.jsonl").write_text(
        "\n".join(json.dumps(row) for row in audit), encoding="utf-8",
    )
    parent_sha, child_sha = "b" * 64, "c" * 64
    def verify(path):
        if Path(path) == parent:
            return {"checkpoint_sha256": parent_sha, "num_timesteps": 150000}
        return {"checkpoint_sha256": child_sha, "parent_checkpoint_sha256": parent_sha,
                "lineage": {"trigger_step": 151553},
                "num_timesteps": 151553}
    monkeypatch.setattr(runner_module, "verify_ppo_lineage_checkpoint", verify)
    train = {"run_id": "child", "output_dir": str(run_dir), "config": {"resume": str(parent)}}
    record = {
        "state": "complete", "validation_mode": "training_probe", "model_path": str(model),
        "model_sha256": child_sha, "trigger_step": 151553,
        "attempts": [{"stop_reason": "race_complete", "frontier_projection_valid": True,
                      "camera_sample_count": 10, "camera_stale_sample_count": 0,
                      "camera_missing_sample_count": 0, "camera_freshness_unknown_count": 0}],
    }
    assert not _pace_probe_is_screening_complete(record, train, tmp_path)
    record["trigger_step"] = 154096
    record["model_path"] = str(checkpoint_dir / "automatic_evaluation_000002_steps_154096.zip")
    Path(record["model_path"]).write_bytes(b"trained child")
    def verify_updated(path):
        if Path(path) == parent:
            return {"checkpoint_sha256": parent_sha, "num_timesteps": 150000}
        return {"checkpoint_sha256": child_sha, "parent_checkpoint_sha256": parent_sha,
                "lineage": {"trigger_step": 154096},
                "num_timesteps": 154096}
    monkeypatch.setattr(runner_module, "verify_ppo_lineage_checkpoint", verify_updated)
    assert _pace_probe_is_screening_complete(record, train, tmp_path)


def _write_live_watch(run_dir: Path, *, ts: str, step: int = 10) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "watch_latest.json").write_text(json.dumps({
        "run_id": "run-live", "ts": ts, "step": step,
        "cars": [{"pose": [1.0, 2.0], "lidar": [3.0], "route_projection_valid": True}],
    }), encoding="utf-8")


def test_stale_api_publication_does_not_fault_when_local_watch_is_fresh(tmp_path: Path):
    now = datetime(2026, 10, 13, 1, 12, tzinfo=timezone.utc)
    _write_live_watch(tmp_path, ts=(now - timedelta(seconds=6)).isoformat(), step=156383)
    local = _local_run_liveness(tmp_path, "run-live", now=now)
    api_age_s = 46.0  # delayed /train telemetry publication, not ROS sensor age
    assert api_age_s > 45
    assert local["freshest"]["source"] == "watch_latest.json"
    assert local["freshest"]["step"] == 156383
    assert _sensor_fault_from_freshness(local, replay_active=False) is False


def test_stale_local_watch_remains_sensor_fault_without_evaluator(tmp_path: Path):
    now = datetime(2026, 10, 13, 1, 12, tzinfo=timezone.utc)
    _write_live_watch(tmp_path, ts=(now - timedelta(seconds=46)).isoformat())
    local = _local_run_liveness(tmp_path, "run-live", now=now)
    assert local["sensor_fault"] is True
    assert _sensor_fault_from_freshness(local, replay_active=False) is True


def test_fresh_training_metrics_do_not_hide_stale_official_watch(tmp_path: Path):
    now = datetime(2026, 10, 13, 1, 12, tzinfo=timezone.utc)
    _write_live_watch(tmp_path, ts=(now - timedelta(seconds=46)).isoformat())
    (tmp_path / "latest_telemetry.json").write_text(json.dumps({
        "run_id": "run-live", "ts": (now - timedelta(seconds=1)).isoformat(), "step": 99,
    }), encoding="utf-8")
    local = _local_run_liveness(tmp_path, "run-live", now=now)
    assert local["freshest"]["source"] == "latest_telemetry.json"
    assert local["sensor_fault"] is True
    assert _sensor_fault_from_freshness(local, replay_active=False) is True


def test_fresh_timestamp_with_explicit_stale_camera_is_sensor_fault(tmp_path: Path):
    now = datetime(2026, 10, 13, 1, 12, tzinfo=timezone.utc)
    _write_live_watch(tmp_path, ts=(now - timedelta(seconds=2)).isoformat())
    path = tmp_path / "watch_latest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cars"][0].update(camera_fresh=False, camera_age_s=1.2)
    path.write_text(json.dumps(value), encoding="utf-8")
    local = _local_run_liveness(tmp_path, "run-live", now=now)
    assert local["watch"]["valid_sensor_frame"] is False
    assert local["sensor_fault"] is True
    assert _sensor_fault_from_freshness(local, replay_active=False) is True


def test_fresh_watch_reset_row_is_not_misclassified_as_sensor_fault(tmp_path: Path):
    now = datetime(2026, 10, 13, 1, 12, tzinfo=timezone.utc)
    _write_live_watch(tmp_path, ts=(now - timedelta(seconds=1)).isoformat())
    path = tmp_path / "watch_latest.json"
    path.write_text(json.dumps({
        "run_id": "run-live", "ts": (now - timedelta(seconds=1)).isoformat(), "step": 12,
        "cars": [{"pose": None, "lidar": [], "route_projection_valid": False, "reset": True}],
    }), encoding="utf-8")
    local = _local_run_liveness(tmp_path, "run-live", now=now)
    assert local["watch"]["reset_in_progress"] is True
    assert local["sensor_fault"] is False
    assert _sensor_fault_from_freshness(local, replay_active=False, run_state="running") is False


def test_startup_without_first_watch_frame_gets_bounded_grace(tmp_path: Path):
    local = _local_run_liveness(tmp_path, "run-live")
    assert _sensor_fault_from_freshness(local, replay_active=False, run_state="starting") is False
    assert _sensor_fault_from_freshness(local, replay_active=False, run_state="running", run_age_s=30) is False
    assert _sensor_fault_from_freshness(local, replay_active=False, run_state="running", run_age_s=121) is True


def test_legacy_literal_reward_runs_and_all_descendants_are_quarantined():
    corrected = runner_module.TIME_COST_VARIANTS["time_cost_3_calibrated_retry"] if "time_cost_3_calibrated_retry" in runner_module.TIME_COST_VARIANTS else next(iter(runner_module.TIME_COST_VARIANTS.values()))
    rows = {
        "dcda7cefacef": {"run_id": "dcda7cefacef", "config": {"time_cost_per_simulated_second": 3.0}},
        "aabbccddeeff": {"run_id": "aabbccddeeff", "config": {
            "time_cost_per_simulated_second": 0.0,
            "resume": "/runs/rl/official_run_dcda7cefacef/checkpoints/post_optimizer_update_1.zip",
        }},
        "ccddeeff0011": {"run_id": "ccddeeff0011", "config": {
            "time_cost_per_simulated_second": corrected,
            "resume": "/runs/rl/official_run_aabbccddeeff/checkpoints/post_optimizer_update_1.zip",
        }},
    }
    assert not _campaign_run_reward_is_eligible(rows["dcda7cefacef"]["config"], run_id="dcda7cefacef", run_index=rows)
    assert not _campaign_run_reward_is_eligible(rows["aabbccddeeff"]["config"], run_id="aabbccddeeff", run_index=rows)
    assert not _campaign_run_reward_is_eligible(rows["ccddeeff0011"]["config"], run_id="ccddeeff0011", run_index=rows)


def test_stale_training_watch_is_expected_while_evaluator_owns_simulator(tmp_path: Path):
    local = {"sensor_fault": True, "freshest": None, "sources": []}
    assert _sensor_fault_from_freshness(local, replay_active=True) is False


def test_campaign_evidence_signature_ignores_only_supervisor_observation_updates():
    baseline = {
        "campaign_id": "campaign", "status": "active", "started_at_utc": "start",
        "training_freeze_at_utc": "freeze", "hard_deadline_at_utc": "deadline",
        "revision": 1, "last_observed_at_utc": "t1",
        "events": [{"event": "supervisor_active_work_observed", "active": {"training": ["run-a"]},
                    "timestamp_utc": "t1", "monotonic_ns": 1}],
    }
    heartbeat = {
        **baseline, "revision": 2, "last_observed_at_utc": "t2",
        "events": [{"event": "supervisor_active_work_observed", "active": {"training": ["run-a"]},
                    "timestamp_utc": "t2", "monotonic_ns": 2}],
    }
    assert _campaign_evidence_state_signature(baseline) == _campaign_evidence_state_signature(heartbeat)
    changed_active = {**heartbeat, "events": [{"event": "supervisor_active_work_observed",
                                               "active": {"training": ["run-b"]},
                                               "timestamp_utc": "t3", "monotonic_ns": 3}]}
    assert _campaign_evidence_state_signature(heartbeat) != _campaign_evidence_state_signature(changed_active)
    api_outage = {**heartbeat, "events": [{"event": "supervisor_api_unavailable", "error": "TimeoutError"}]}
    api_outage_changed = {**heartbeat, "events": [{"event": "supervisor_api_unavailable", "error": "ConnectionError"}]}
    assert _campaign_evidence_state_signature(api_outage) != _campaign_evidence_state_signature(api_outage_changed)
    declared = {**heartbeat, "events": [*heartbeat["events"], {
        "event": "campaign_experiment_declared", "experiment_id": "variant", "config_sha256": "a" * 64,
    }]}
    assert _campaign_evidence_state_signature(heartbeat) != _campaign_evidence_state_signature(declared)


def _attempt(*, laps: int, collisions: int = 0, run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "stop_reason": "race_complete",
        "frontier_projection_valid": True,
        "collision_count": collisions,
        "race_collisions": collisions,
        "warmup_collisions": 0,
        "raw_collision_count_at_finish": collisions,
        "camera_sample_count": 100,
        "camera_fresh_sample_count": 100,
        "camera_stale_sample_count": 0,
        "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
        "race_lap_times_s": [100.0 + index for index in range(laps)],
        "warmup_lap_times_s": [101.0],
    }


def _probe_row(report_path: Path, *, run_id: str = "run-1") -> dict:
    return {
        "state": "complete", "validation_mode": "training_probe", "model_sha256": SHA,
        "result": {
            "readiness_passed": True,
            "runs": [{"run_id": run_id, "evaluation_report": str(report_path),
                      "warmup_laps": 1, "race_laps_completed": 3,
                      "race_laps_required": 3, "collision_count": 0,
                      "race_collisions": 0, "score_status": "valid",
                      "readiness_passed": True, "valid_for_comparison": False}],
        },
    }


def test_training_probe_requires_authenticated_model_and_complete_zero_collision_1_plus_3(tmp_path: Path):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    row = {
        "state": "complete",
        "validation_mode": "training_probe",
        "model_sha256": SHA,
        "result": {"readiness_passed": True, "runs": [{
            "run_id": "run-1", "evaluation_report": str(report_path), "warmup_laps": 1,
            "race_laps_completed": 3, "race_laps_required": 3,
            "collision_count": 0, "race_collisions": 0, "score_status": "valid",
            "readiness_passed": True, "valid_for_comparison": False,
        }]},
    }
    report = {"model_sha256": SHA, "protocol": "official_training_probe_1_plus_3",
              "validation_mode": "training_probe", "runs": [{
                  **_attempt(laps=3), "race_collisions": 0,
                  "race_laps_completed": 3, "race_laps_required": 3,
                  "camera_sample_count": 10, "camera_fresh_sample_count": 10,
                  "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                  "camera_freshness_unknown_count": 0,
              }]}
    evidence = validate_training_probe(row, report, SHA, report_path=report_path, repo_root=tmp_path)
    assert evidence is not None
    assert evidence["median_scored_lap_s"] == 101.0
    assert evidence["collision_count"] == 0


def test_training_probe_rejects_incomplete_collision_or_wrong_loaded_hash(tmp_path: Path):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    row = _probe_row(report_path)
    kwargs = {"report_path": report_path, "repo_root": tmp_path}
    good = {"model_sha256": SHA, "protocol": "official_training_probe_1_plus_3",
            "validation_mode": "training_probe", "runs": [_attempt(laps=3)]}
    assert validate_training_probe(row, good, SHA, **kwargs) is None
    assert validate_training_probe(row, {**good, "runs": [_attempt(laps=2)]}, SHA, **kwargs) is None
    assert validate_training_probe(row, {**good, "runs": [_attempt(laps=3, collisions=1)]}, SHA, **kwargs) is None
    assert validate_training_probe(row, {**good, "model_sha256": "b" * 64}, SHA, **kwargs) is None


@pytest.mark.parametrize("field", ["collision_count", "race_collisions"])
def test_training_probe_rejects_boolean_service_zero_counters(tmp_path: Path, field: str):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    row = _probe_row(report_path)
    row["result"]["runs"][0][field] = False
    report = {"model_sha256": SHA, "protocol": "official_training_probe_1_plus_3",
              "validation_mode": "training_probe", "runs": [{
                  **_attempt(laps=3), "race_collisions": 0,
                  "race_laps_completed": 3, "race_laps_required": 3,
                  "camera_sample_count": 10, "camera_fresh_sample_count": 10,
                  "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                  "camera_freshness_unknown_count": 0,
              }]}
    assert validate_training_probe(row, report, SHA, report_path=report_path, repo_root=tmp_path) is None


@pytest.mark.parametrize("field", ["collision_count", "race_collisions"])
def test_training_probe_rejects_boolean_report_zero_counters(tmp_path: Path, field: str):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    attempt = {
        **_attempt(laps=3), "race_collisions": 0,
        "race_laps_completed": 3, "race_laps_required": 3,
        "camera_sample_count": 10, "camera_fresh_sample_count": 10,
        "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
    }
    attempt[field] = False
    report = {"model_sha256": SHA, "protocol": "official_training_probe_1_plus_3",
              "validation_mode": "training_probe", "runs": [attempt]}
    assert validate_training_probe(
        _probe_row(report_path), report, SHA, report_path=report_path, repo_root=tmp_path,
    ) is None


@pytest.mark.parametrize("field", [
    "camera_stale_sample_count", "camera_missing_sample_count", "camera_freshness_unknown_count",
])
def test_training_probe_rejects_boolean_camera_zero_counters(tmp_path: Path, field: str):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    report = {"model_sha256": SHA, "protocol": "official_training_probe_1_plus_3",
              "validation_mode": "training_probe", "runs": [{
                  **_attempt(laps=3), "race_collisions": 0,
                  "race_laps_completed": 3, "race_laps_required": 3,
                  "camera_sample_count": 10, "camera_fresh_sample_count": 10,
                  "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                  "camera_freshness_unknown_count": 0,
              }]}
    report["runs"][0][field] = False
    assert validate_training_probe(
        _probe_row(report_path), report, SHA, report_path=report_path, repo_root=tmp_path,
    ) is None


@pytest.mark.parametrize("lap_time", [float("nan"), float("inf"), 10 ** 400])
def test_training_probe_rejects_nonfinite_and_overflowing_lap_times(tmp_path: Path, lap_time):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    report = {"model_sha256": SHA, "protocol": "official_training_probe_1_plus_3",
              "validation_mode": "training_probe", "runs": [{
                  **_attempt(laps=3), "race_collisions": 0,
                  "race_laps_completed": 3, "race_laps_required": 3,
                  "camera_sample_count": 10, "camera_fresh_sample_count": 10,
                  "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                  "camera_freshness_unknown_count": 0,
              }]}
    report["runs"][0]["race_lap_times_s"][0] = lap_time
    assert validate_training_probe(
        _probe_row(report_path), report, SHA,
        report_path=report_path, repo_root=tmp_path,
    ) is None


@pytest.mark.parametrize("field,value", [
    ("run_id", "wrong-run"),
    ("evaluation_report", "/runs/rl/official_run_other/evaluation.json"),
])
def test_training_probe_binds_report_to_history_attempt(tmp_path: Path, field: str, value: str):
    report_path = tmp_path / "logs" / "rl" / "official_run_run-1" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    row = _probe_row(report_path)
    row["result"]["runs"][0][field] = value
    assert validate_training_probe(
        row, {"model_sha256": SHA, "runs": [_attempt(laps=3)]}, SHA,
        report_path=report_path, repo_root=tmp_path,
    ) is None


def test_wsl_report_path_resolves_only_within_local_rl_evidence_root(tmp_path: Path):
    repo_root = tmp_path / "AiCar"
    expected = (repo_root / "logs" / "rl" / "official_run_run-1" / "evaluation.json").resolve()
    assert _resolve_model_path(
        "/mnt/c/Users/farre/Documents/Github/AiCar/logs/rl/official_run_run-1/evaluation.json",
        repo_root,
    ).resolve() == expected
    assert _resolve_model_path(
        "/mnt/c/Users/farre/Documents/Github/Other/logs/rl/official_run_run-1/evaluation.json",
        repo_root,
    ) is None


def test_final_evaluation_requires_three_clean_attempts_with_loaded_hash(tmp_path: Path):
    from copy import deepcopy

    reports = []
    service_rows = []
    report_root = tmp_path / "logs" / "rl"
    report_root.mkdir(parents=True)
    for index in range(3):
        run_id = f"run-{index}"
        run_dir = report_root / f"official_run_{run_id}"
        run_dir.mkdir(parents=True)
        process_record = {
            "pid": 5100 + index,
            "port": 58000 + index,
            "executable_path": rf"C:\Users\farre\Temp\aicar-official-iros-camera\windows\autodrive_simulator\AutoDRIVE Simulator.exe",
            "process_creation_utc": f"2026-10-13T12:00:0{index}Z",
        }
        manifest_path = run_dir / "windows_simulator_processes.json"
        manifest_bytes = json.dumps({"run_id": run_id, "records": [process_record]}).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        simulator_identity = {
            "pid": 5100 + index,
            "process_creation_utc": f"2026-10-13T12:00:0{index}Z",
            "executable_path": rf"C:\Users\farre\Temp\aicar-official-iros-camera\windows\autodrive_simulator\AutoDRIVE Simulator.exe",
            "port": 58000 + index,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        monitor = {
            "projection_valid": True, "projection_sample_count": 100,
            "projection_valid_sample_count": 100,
            "projection_invalid_sample_count": 0, "scoring_or_stop_effect": False,
        }
        report_attempt = {
            **_attempt(laps=10, run_id=run_id),
            "frontier_projection_valid": True,
            "route_projection_valid": True,
            "route_monitor_only": True,
            "route_monitor": monitor,
            "official_scene_sha256": "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674",
            "route_sha256": "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb",
            "official_scene_path": r"C:\sim\Data\level0",
            "route_path": r"C:\repo\competition\iros2026\tracks\competition\official_centerline.csv",
            "track_profile": "competition",
            "simulator_identity": simulator_identity,
        }
        report = run_dir / "evaluation.json"
        report.write_text(
            json.dumps({
                "model_sha256": SHA, "validation_mode": None,
                "protocol": "official_competitive_1_plus_10", "competitive_score_valid": True,
                "route_monitor_only": True,
                "official_scene_sha256": report_attempt["official_scene_sha256"],
                "route_sha256": report_attempt["route_sha256"],
                "official_scene_path": report_attempt["official_scene_path"],
                "route_path": report_attempt["route_path"],
                "track_profile": "competition", "simulator_identity": simulator_identity,
                "runs": [report_attempt],
            }),
            encoding="utf-8",
        )
        reports.append(report)
    copied = reports
    for index, target in enumerate(copied):
        saved = json.loads(target.read_text(encoding="utf-8"))
        service_attempt = saved["runs"][0]
        service_rows.append({
            "evaluation_report": str(target), "run_id": service_attempt["run_id"], "warmup_laps": 1,
            "race_laps_completed": 10, "collision_count": 0, "race_collisions": 0,
            "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
            "camera_freshness_unknown_count": 0, "camera_sample_count": 100,
            "camera_fresh_sample_count": 100, "valid_for_comparison": True,
            "frontier_projection_valid": True, "route_projection_valid": True,
            "route_monitor_only": True, "route_monitor": deepcopy(service_attempt["route_monitor"]),
            "official_scene_sha256": service_attempt["official_scene_sha256"],
            "route_sha256": service_attempt["route_sha256"],
            "official_scene_path": service_attempt["official_scene_path"],
            "route_path": service_attempt["route_path"], "track_profile": "competition",
            "simulator_identity": deepcopy(service_attempt["simulator_identity"]),
        })
    record = {
        "state": "complete",
        "evaluation_id": "eval-1",
        "source": "manual",
        "model_sha256": SHA,
        "result": {
            "model_sha256": SHA,
            "protocol": "official_10_lap_race",
            "validation_mode": None,
            "attempts_required": 3,
            "competitive_score_valid": True,
                "runs": service_rows,
        },
    }
    result = validate_final_evaluation(record, copied, expected_sha256=SHA)
    assert result["passed"] is True
    assert len(result["attempts"]) == 3
    assert all(row["scored_laps"] == 10 and row["collision_count"] == 0 for row in result["attempts"])


def test_final_evaluation_fails_closed_on_collision_or_wrong_loaded_hash(tmp_path: Path):
    reports = []
    for index in range(3):
        report = tmp_path / f"attempt-{index}.json"
        report.write_text(
            json.dumps({"model_sha256": SHA, "runs": [_attempt(laps=10, collisions=1 if index == 1 else 0, run_id=f"run-{index}")]}),
            encoding="utf-8",
        )
        reports.append(report)
    report_root = tmp_path / "logs" / "rl"
    report_root.mkdir(parents=True)
    copied = []
    for index, report in enumerate(reports):
        target = report_root / f"official_run_run-{index}" / "evaluation.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(report.read_bytes())
        copied.append(target)
    record = {
        "state": "complete", "source": "manual", "model_sha256": "b" * 64,
        "result": {
            "model_sha256": "b" * 64, "protocol": "official_10_lap_race",
            "validation_mode": None, "attempts_required": 3,
            "competitive_score_valid": True,
            "runs": [
                {"evaluation_report": str(path), "run_id": f"run-{index}", "warmup_laps": 1,
                 "race_laps_completed": 10, "collision_count": 0, "race_collisions": 0,
                 "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                 "camera_freshness_unknown_count": 0, "valid_for_comparison": True}
                for index, path in enumerate(copied)
            ],
        },
    }
    result = validate_final_evaluation(record, copied, expected_sha256=SHA)
    assert result["passed"] is False
    assert any("loaded SHA" in error for error in result["errors"])
    assert any("zero-collision" in error for error in result["errors"])


@pytest.mark.parametrize("field", [
    "collision_count", "race_collisions", "camera_stale_sample_count",
    "camera_missing_sample_count", "camera_freshness_unknown_count",
])
def test_final_evaluation_rejects_boolean_service_zero_counters(tmp_path: Path, field: str):
    root = tmp_path / "logs" / "rl"
    paths = []
    service_runs = []
    for index in range(3):
        run_id = f"run-{index}"
        run_dir = root / f"official_run_{run_id}"
        run_dir.mkdir(parents=True)
        path = run_dir / "evaluation.json"
        path.write_text(json.dumps({"model_sha256": SHA, "runs": [_attempt(laps=10, run_id=run_id)]}), encoding="utf-8")
        paths.append(path)
        service_run = {"evaluation_report": str(path), "run_id": run_id, "warmup_laps": 1,
                       "race_laps_completed": 10, "collision_count": 0, "race_collisions": 0,
                       "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                       "camera_freshness_unknown_count": 0, "valid_for_comparison": True}
        if index == 0:
            service_run[field] = False
        service_runs.append(service_run)
    record = {"state": "complete", "source": "manual", "model_sha256": SHA,
              "result": {"model_sha256": SHA, "protocol": "official_10_lap_race",
                         "validation_mode": None, "attempts_required": 3,
                         "competitive_score_valid": True, "runs": service_runs}}
    result = validate_final_evaluation(record, paths, expected_sha256=SHA)
    assert result["passed"] is False
    assert any("zero-collision" in error for error in result["errors"])


@pytest.mark.parametrize("field", [
    "collision_count", "race_collisions", "camera_stale_sample_count", "camera_missing_sample_count",
    "camera_freshness_unknown_count",
])
def test_final_evaluation_rejects_boolean_report_zero_counters(tmp_path: Path, field: str):
    root = tmp_path / "logs" / "rl"
    paths = []
    service_runs = []
    for index in range(3):
        run_id = f"run-{index}"
        run_dir = root / f"official_run_{run_id}"
        run_dir.mkdir(parents=True)
        attempt = _attempt(laps=10, run_id=run_id)
        if index == 0:
            attempt[field] = False
        path = run_dir / "evaluation.json"
        path.write_text(json.dumps({"model_sha256": SHA, "runs": [attempt]}), encoding="utf-8")
        paths.append(path)
        service_runs.append({"evaluation_report": str(path), "run_id": run_id, "warmup_laps": 1,
                             "race_laps_completed": 10, "collision_count": 0, "race_collisions": 0,
                             "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                             "camera_freshness_unknown_count": 0, "valid_for_comparison": True})
    record = {"state": "complete", "source": "manual", "model_sha256": SHA,
              "result": {"model_sha256": SHA, "protocol": "official_10_lap_race",
                         "validation_mode": None, "attempts_required": 3,
                         "competitive_score_valid": True, "runs": service_runs}}
    result = validate_final_evaluation(record, paths, expected_sha256=SHA)
    assert result["passed"] is False


@pytest.mark.parametrize("bad_attempt", [
    None,
    [],
    "not-an-object",
    {**_attempt(laps=10), "collision_count": "unknown"},
    {**_attempt(laps=10), "collision_count": []},
    {**_attempt(laps=10), "collision_count": True},
])
def test_final_evaluation_malformed_attempts_fail_closed(tmp_path: Path, bad_attempt):
    root = tmp_path / "logs" / "rl"
    root.mkdir(parents=True)
    report = root / "attempt.json"
    report.write_text(json.dumps({"model_sha256": SHA, "runs": [bad_attempt]}), encoding="utf-8")
    record = {
        "state": "complete", "source": "manual", "model_sha256": SHA,
        "result": {"model_sha256": SHA, "protocol": "official_10_lap_race", "validation_mode": None,
                   "attempts_required": 3, "competitive_score_valid": True, "runs": []},
    }
    result = validate_final_evaluation(record, [report], expected_sha256=SHA)
    assert result["passed"] is False
    assert result["errors"]


def test_final_evaluation_extreme_lap_time_fails_closed(tmp_path: Path):
    root = tmp_path / "logs" / "rl"
    root.mkdir(parents=True)
    report = root / "attempt.json"
    malformed = _attempt(laps=10)
    malformed["race_lap_times_s"][0] = 10 ** 400
    report.write_text(json.dumps({"model_sha256": SHA, "runs": [malformed]}), encoding="utf-8")
    result = validate_final_evaluation(
        {"result": {"runs": []}}, [report], expected_sha256=SHA,
    )
    assert result["passed"] is False
    assert any("invalid lap timing" in error for error in result["errors"])


def test_model_paths_are_confined_to_campaign_logs_and_round_trip_alias(tmp_path: Path):
    logs = tmp_path / "logs" / "rl"
    model = logs / "official_run_abc" / "checkpoint.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint")
    alias = _model_alias(model, tmp_path)
    assert alias == "/runs/rl/official_run_abc/checkpoint.zip"
    assert _resolve_model_path(alias, tmp_path) == model.resolve()
    assert _resolve_model_path(str(tmp_path / "outside.zip"), tmp_path) is None


def test_final_evaluation_rejects_duplicate_or_mismatched_report_binding(tmp_path: Path):
    root = tmp_path / "logs" / "rl"
    root.mkdir(parents=True)
    paths = []
    for index in range(3):
        path = root / f"attempt-{index}.json"
        path.write_text(json.dumps({"model_sha256": SHA, "runs": [_attempt(laps=10, run_id=f"run-{index}")]}), encoding="utf-8")
        paths.append(path)
    record = {
        "state": "complete", "source": "manual", "model_sha256": SHA,
        "result": {
            "model_sha256": SHA, "protocol": "official_10_lap_race", "validation_mode": None,
            "attempts_required": 3, "competitive_score_valid": True,
            "runs": [
                {"evaluation_report": str(paths[0]), "run_id": "run-0", "warmup_laps": 1,
                 "race_laps_completed": 10, "collision_count": 0, "valid_for_comparison": True},
                {"evaluation_report": str(paths[1]), "run_id": "wrong-run", "warmup_laps": 1,
                 "race_laps_completed": 10, "collision_count": 0, "valid_for_comparison": True},
                {"evaluation_report": str(paths[1]), "run_id": "run-1", "warmup_laps": 1,
                 "race_laps_completed": 10, "collision_count": 0, "valid_for_comparison": True},
            ],
        },
    }
    result = validate_final_evaluation(record, paths, expected_sha256=SHA)
    assert result["passed"] is False
    assert any("run id" in error for error in result["errors"])
    assert any("not bound" in error for error in result["errors"])


def test_phase_tick_stops_active_baseline_before_launching_pace_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeAPI:
        def __init__(self):
            self.posts = []

        def get(self, route):
            if route == "/train/runs":
                return {"runs": [{
                    "run_id": "baseline", "kind": "training", "state": "running",
                    "config": {"time_cost_per_simulated_second": 0.0},
                }]}
            if route == "/api/evaluations/history":
                return []
            if route == "/api/replay/status":
                return {"state": "stopped"}
            if route == "/health":
                return {"status": "ok"}
            raise AssertionError(route)

        def post(self, route, body=None):
            self.posts.append((route, body))
            return {"run_id": "new"}

    fake = FakeAPI()
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=fake,
    )
    state = {"campaign_id": runner_module.CAMPAIGN_ID, "status": "active", "events": []}
    monkeypatch.setattr(runner_module, "load_campaign_state", lambda _path: state)
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: {})
    monkeypatch.setattr(runner_module, "build_campaign_decision_plan", lambda *_args, **_kwargs: {
        "phase": "pace_experiments", "elapsed_hours": 2.1, "branch_candidates": [{"experiment_id": "time_cost_3"}],
    })
    result = runner.tick()
    assert result["action"] == "stop_baseline_before_pace_branch"
    assert fake.posts == [("/train/runs/baseline/stop", None)]


def test_active_evaluation_history_prevents_idle_launch():
    assert runner_module.CampaignPhaseRunner._active_replay({
        "replay": {"state": "stopped"},
        "evaluation_history": [{"state": "running"}],
    }) is True


@pytest.mark.parametrize("terminal_state", ["complete", "incomplete"])
def test_final_replay_terminal_without_history_is_recorded_as_failed(tmp_path: Path, terminal_state: str):
    class FakeAPI:
        pass

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=FakeAPI(),
    )
    runner.output_dir.mkdir(parents=True)
    runner.events_path.write_text(json.dumps({
        "event": "finalist_evaluation_requested", "evaluation_id": "eval-1",
        "checkpoint_sha256": SHA, "model_path": "/runs/rl/finalist.zip",
    }) + "\n", encoding="utf-8")
    result = runner._pending_final_evaluation({
        "replay": {"state": terminal_state, "evaluation_id": "eval-1"},
        "evaluation_history": [],
    })
    assert result["pending"] is False
    assert result["result"]["passed"] is False
    assert "no durable evaluation history" in result["result"]["errors"][0]


def test_finalist_manifest_is_created_only_after_training_and_evaluation_are_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeAPI:
        pass

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=FakeAPI(),
    )
    state = {"campaign_id": runner_module.CAMPAIGN_ID}
    evidence = {"campaign_id": runner_module.CAMPAIGN_ID, "probe_evidence": [{
        "probe": {"median_scored_lap_s": 50.0}, "parent_run_id": "parent-1",
        "report_path": str(tmp_path / "logs" / "rl" / "official_run_child-1" / "evaluation.json"),
        "model_path": str(tmp_path / "logs" / "rl" / "official_run_parent-1" / "checkpoints" / "candidate.zip"),
        "report_sha256": "b" * 64, "model_sha256": "a" * 64, "evaluation_id": "eval-1",
    }]}
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: evidence)
    monkeypatch.setattr(runner_module, "build_finalist_manifest", lambda *_args, **_kwargs: {
        "campaign_id": runner_module.CAMPAIGN_ID,
        "finalists": [{"checkpoint_sha256": "a" * 64}],
    })
    runner.output_dir.mkdir(parents=True)
    active = {"runs": [{"state": "running"}], "replay": {"state": "stopped"}, "evaluation_history": []}
    assert runner._create_finalist_manifest_if_ready(state, active) == {
        "action": "wait_for_quiescence_before_finalist_manifest",
    }
    assert not runner.finalists_manifest.exists()
    idle = {"runs": [], "replay": {"state": "complete"}, "evaluation_history": []}
    result = runner._create_finalist_manifest_if_ready(state, idle)
    assert result["action"] == "finalist_manifest_created"
    assert runner.finalists_manifest.exists()
    assert json.loads(runner.finalists_manifest.read_text(encoding="utf-8"))["finalists"][0]["checkpoint_sha256"] == "a" * 64


def test_finalist_manifest_existing_file_is_not_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeAPI:
        pass

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=FakeAPI(),
    )
    runner.finalists_manifest.parent.mkdir(parents=True)
    runner.finalists_manifest.write_text('{"original":true}', encoding="utf-8")
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: (_ for _ in ()).throw(AssertionError("must not regenerate")))
    result = runner._create_finalist_manifest_if_ready(
        {"campaign_id": runner_module.CAMPAIGN_ID}, {"runs": [], "replay": {"state": "complete"}, "evaluation_history": []},
    )
    assert result is None
    assert runner.finalists_manifest.read_text(encoding="utf-8") == '{"original":true}'


def test_unresolved_branch_launch_intent_blocks_duplicate_post(tmp_path: Path):
    class FakeAPI:
        def get(self, route):
            raise AssertionError(route)

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=FakeAPI(),
    )
    runner.output_dir.mkdir(parents=True)
    runner.events_path.write_text(json.dumps({
        "event": "candidate_launch_intent", "experiment_id": "time_cost_3",
        "phase": "pace_experiments", "config_sha256": "c" * 64,
        "resume_path": "/runs/rl/parent.zip", "timestamp_utc": "2026-10-12T20:00:00+00:00",
    }) + "\n", encoding="utf-8")
    assert runner._unresolved_candidate_intent([])["reconciled"] is False
    matched_config = {
        "resume": "/runs/rl/parent.zip", "campaign_experiment_id": "time_cost_3",
        "campaign_experiment_phase": "pace_experiments",
        "campaign_experiment_config_sha256": "c" * 64,
    }
    mismatched = [{"run_id": "run-2", "config": {
        **matched_config, "campaign_experiment_config_sha256": "d" * 64,
    }}]
    assert runner._unresolved_candidate_intent(mismatched)["reconciled"] is False
    matching = [{"run_id": "run-1", "config": matched_config}]
    assert runner._unresolved_candidate_intent(matching)["reconciled"] is True


@pytest.mark.parametrize("ledger", ['{"event":"candidate_launch_intent"', '[]', '\n'])
def test_corrupt_or_malformed_action_ledger_fails_closed(tmp_path: Path, ledger: str, monkeypatch: pytest.MonkeyPatch):
    class FakeAPI:
        def get(self, route):
            raise AssertionError(f"API must not be called with corrupt ledger: {route}")

    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=FakeAPI(),
    )
    runner.output_dir.mkdir(parents=True)
    runner.events_path.write_text(ledger, encoding="utf-8")
    monkeypatch.setattr(runner_module, "load_campaign_state", lambda _path: {
        "campaign_id": runner_module.CAMPAIGN_ID, "status": "active", "events": [],
    })
    with pytest.raises(RuntimeError, match="action ledger"):
        runner.tick()


def test_candidate_launch_requires_exact_cadence_auth_and_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeAPI:
        def __init__(self):
            self.posts = []

        def post(self, route, body=None):
            self.posts.append((route, body))
            return {"run_id": "branch-1"}

    parent = tmp_path / "logs" / "rl" / "parent.zip"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"checkpoint")
    sha = "b" * 64
    settings = {
        "resume": "/runs/rl/parent.zip", "exploration_profile": "bc_campaign", "n_envs": 1,
        "checkpoint_every": 10000, "automatic_evaluation_enabled": True,
        "automatic_evaluation_mode": "training_probe", "automatic_evaluation_trigger": "steps",
        "automatic_evaluation_interval": 50000,
        "time_cost_per_simulated_second": runner_module.TIME_COST_VARIANTS["time_cost_3_calibrated_retry"],
    }
    fake = FakeAPI()
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=fake,
    )
    monkeypatch.setattr(runner_module, "verify_ppo_lineage_checkpoint",
                        lambda _path, verification_context=None: {"checkpoint_sha256": sha})
    monkeypatch.setattr(runner_module, "declare_campaign_experiment", lambda *_args, **_kwargs: {
        "config_sha256": "c" * 64, "config": settings,
    })
    candidate = {
        "experiment_id": "time_cost_3_calibrated_retry", "phase": "pace_experiments",
        "parent_checkpoint_sha256": sha, "config_sha256": "c" * 64,
        "request_settings": settings, "declaration_evidence": {},
    }
    result = runner._launch_candidate(candidate, {})
    assert result["run_id"] == "branch-1"
    assert len(fake.posts) == 1
    assert fake.posts[0][0] == "/train/runs"
