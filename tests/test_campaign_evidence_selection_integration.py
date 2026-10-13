from __future__ import annotations

import hashlib
import json
from pathlib import Path

import src.layer4.hub.campaign_phase_runner as runner_module
from src.layer3.bc_artifact import PPOVerificationContext
from src.layer4.hub.campaign_deadline import (
    CAMPAIGN_ID,
    TIME_COST_VARIANTS,
    campaign_effective_config_sha256,
)


EXPERIMENT_ID = "time_cost_3_calibrated_retry"
COEFFICIENT = TIME_COST_VARIANTS[EXPERIMENT_ID]


def _lineage(run_id: str, checkpoint: Path, *, parent_sha: str | None = None) -> dict:
    value = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "num_timesteps": 200704,
        "lineage": {
            "run_id": run_id,
            "trigger_step": 200000,
            "bc_ancestor_checkpoint_sha256": "d" * 64,
        },
    }
    if parent_sha is not None:
        value["parent_checkpoint_sha256"] = parent_sha
    return value


def _write_manager(run_dir: Path, run_id: str, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manager_config.json").write_text(json.dumps({
        "run_id": run_id, "config": config,
    }), encoding="utf-8")


def _declared_parent_config(parent_checkpoint: Path) -> tuple[dict, dict]:
    config = {
        "resume": str(parent_checkpoint),
        "time_cost_per_simulated_second": COEFFICIENT,
        "campaign_experiment_id": EXPERIMENT_ID,
        "campaign_experiment_phase": "pace_experiments",
        "campaign_experiment_config_sha256": "0" * 64,
    }
    digest = campaign_effective_config_sha256(config)
    config["campaign_experiment_config_sha256"] = digest
    return config, {
        "event": "campaign_experiment_declared",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": digest,
        "declared_config": config,
    }


def test_collect_evidence_selects_qualified_candidate_and_excludes_one_rollout_seed(
    tmp_path: Path, monkeypatch,
):
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=object(),
    )
    state = {"campaign_id": CAMPAIGN_ID, "status": "active", "events": []}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(runner_module, "load_campaign_state", lambda _path: state)
    monkeypatch.setattr(runner_module, "campaign_clock_status", lambda _state: {"phase": "pace_experiments"})

    checkpoint_dir = tmp_path / "logs" / "rl" / "official_run_campaign-run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    seed_path = checkpoint_dir / "seed.zip"
    seed_path.write_bytes(b"seed-checkpoint")
    qualified_path = checkpoint_dir / "qualified.zip"
    qualified_path.write_bytes(b"two-updates-checkpoint")
    config = {"time_cost_per_simulated_second": 0.0}
    snapshot = {
        "runs": [{"run_id": "campaign-run", "display_name": "Campaign continuation", "config": config}],
        "evaluation_history": [], "telemetry": {}, "replay": {},
    }
    monkeypatch.setattr(runner, "_report_evidence", lambda *_args: [
        {
            "evaluation_id": "seed-eval", "parent_run_id": "campaign-run",
            "model_path": str(seed_path), "model_sha256": "b" * 64,
            "report_sha256": "c" * 64, "probe": {"median_scored_lap_s": 32.0},
            # The report is readiness-valid and syntactically passing, but a
            # single optimizer update is insufficient for candidate use.
            "trend_validation": {
                "outcome": "pass", "completed_rollouts_since_branch": 1,
            },
            "reward_experiment_config_sha256": "",
        },
        {
            "evaluation_id": "qualified-eval", "parent_run_id": "campaign-run",
            "model_path": str(qualified_path), "model_sha256": "e" * 64,
            "report_sha256": "f" * 64, "probe": {"median_scored_lap_s": 34.11},
            "trend_validation": {
                "outcome": "pass", "completed_rollouts_since_branch": 2,
            },
            "reward_experiment_config_sha256": "",
        },
    ])

    evidence = runner.collect_evidence(snapshot)
    assert evidence["passing_candidate_count"] == 1
    assert evidence["selected_passing_checkpoint_sha256"] == "e" * 64
    assert evidence["selected_passing_train_request"]["resume"] == runner_module._model_alias(
        qualified_path, tmp_path,
    )


def test_reward_ancestry_accepts_authenticated_declared_parent_and_rejects_missing_declaration(
    tmp_path: Path, monkeypatch,
):
    parent_checkpoint = tmp_path / "logs" / "rl" / "official_run_parent" / "checkpoints" / "source.zip"
    parent_checkpoint.parent.mkdir(parents=True)
    parent_checkpoint.write_bytes(b"authenticated BC-derived parent")
    checkpoint = tmp_path / "logs" / "rl" / "official_run_retry" / "checkpoints" / "retry.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authenticated PPO descendant")
    run_id = "a1b2c3d4e5f6"
    config, event = _declared_parent_config(parent_checkpoint)
    run_dir = tmp_path / "logs" / "rl" / f"official_run_{run_id}"
    _write_manager(run_dir, run_id, config)
    (run_dir / "config.json").write_text(json.dumps({
        "training_only_reward": {"time_cost_per_simulated_second": COEFFICIENT},
    }), encoding="utf-8")
    parent_verified = _lineage("parent", parent_checkpoint)
    verified = _lineage(run_id, checkpoint, parent_sha=parent_verified["checkpoint_sha256"])
    monkeypatch.setattr(
        runner_module, "verify_ppo_lineage_checkpoint",
        lambda path, **_kw: parent_verified if Path(path) == parent_checkpoint else verified,
    )

    accepted = runner_module._campaign_reward_ancestry_link(
        {"events": [event]}, run_id=run_id,
        run_index={run_id: {"run_id": run_id, "config": config}},
        repo_root=tmp_path, verification_context=PPOVerificationContext(),
        checkpoint=checkpoint, verified_checkpoint=verified,
    )
    assert accepted == event["config_sha256"]

    unrelated_same_reward = runner_module._campaign_reward_ancestry_link(
        {"events": []}, run_id=run_id,
        run_index={run_id: {"run_id": run_id, "config": config}},
        repo_root=tmp_path, verification_context=PPOVerificationContext(),
        checkpoint=checkpoint, verified_checkpoint=verified,
    )
    assert unrelated_same_reward is None


def test_reward_ancestry_rejects_child_sidecar_parent_sha_mismatch(
    tmp_path: Path, monkeypatch,
):
    parent_id, child_id = "111111111111", "222222222222"
    parent_checkpoint = tmp_path / "parent.zip"
    child_checkpoint = tmp_path / "child.zip"
    parent_checkpoint.write_bytes(b"calibrated parent")
    child_checkpoint.write_bytes(b"resumed child")
    parent_config, event = _declared_parent_config(parent_checkpoint)
    child_config = {
        "resume": str(parent_checkpoint),
        "time_cost_per_simulated_second": COEFFICIENT,
        "campaign_experiment_id": "continue_best",
        "campaign_experiment_phase": "best_validated_trend",
    }
    _write_manager(tmp_path / "logs" / "rl" / f"official_run_{parent_id}", parent_id, parent_config)
    _write_manager(tmp_path / "logs" / "rl" / f"official_run_{child_id}", child_id, child_config)
    for run_id in (parent_id, child_id):
        run_dir = tmp_path / "logs" / "rl" / f"official_run_{run_id}"
        (run_dir / "config.json").write_text(json.dumps({
            "training_only_reward": {"time_cost_per_simulated_second": COEFFICIENT},
        }), encoding="utf-8")
    parent_verified = _lineage(parent_id, parent_checkpoint)
    child_verified = _lineage(child_id, child_checkpoint, parent_sha="9" * 64)
    monkeypatch.setattr(
        runner_module, "verify_ppo_lineage_checkpoint",
        lambda path, **_kw: parent_verified if Path(path) == parent_checkpoint else child_verified,
    )

    result = runner_module._campaign_reward_ancestry_link(
        {"events": [event]}, run_id=child_id,
        run_index={
            parent_id: {"run_id": parent_id, "config": parent_config},
            child_id: {"run_id": child_id, "config": child_config},
        },
        repo_root=tmp_path, verification_context=PPOVerificationContext(),
        checkpoint=child_checkpoint, verified_checkpoint=child_verified,
    )
    assert result is None


def test_persisted_report_row_flows_through_collector_into_campaign_trend(
    tmp_path: Path, monkeypatch,
):
    """Exercise the production row schema from service history through trend."""
    from src.layer4.hub.campaign_trend import derive_campaign_trend_evidence

    parent_id = "111111111111"
    source_id = "222222222222"
    evaluation_id = "3333333333333333"
    evaluation_run_id = "444444444444"
    parent_dir = tmp_path / "logs" / "rl" / f"official_run_{parent_id}"
    source_dir = tmp_path / "logs" / "rl" / f"official_run_{source_id}"
    evaluation_dir = tmp_path / "logs" / "rl" / f"official_run_{evaluation_run_id}"
    child_path = parent_dir / "checkpoints" / "automatic_evaluation_000001_steps_250000.zip"
    source_path = source_dir / "checkpoints" / "source.zip"
    report_path = evaluation_dir / "evaluation.json"
    for path in (child_path, source_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    child_path.write_bytes(b"post-update authenticated child")
    source_path.write_bytes(b"authenticated source checkpoint")
    child_sha = hashlib.sha256(child_path.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

    request = {
        "resume": f"/runs/rl/official_run_{source_id}/checkpoints/source.zip",
        "n_steps": 2048,
        "automatic_evaluation_enabled": True,
        "automatic_evaluation_trigger": "steps",
        "automatic_evaluation_interval": 50000,
        "automatic_evaluation_mode": "training_probe",
        "time_cost_per_simulated_second": COEFFICIENT,
        "campaign_experiment_id": EXPERIMENT_ID,
        "campaign_experiment_phase": "pace_experiments",
        "campaign_experiment_config_sha256": "a" * 64,
    }
    parent_verified = {
        "checkpoint_sha256": source_sha,
        "num_timesteps": 200704,
        "lineage": {"run_id": source_id},
    }
    child_verified = {
        "checkpoint_sha256": child_sha,
        "parent_checkpoint_sha256": source_sha,
        "num_timesteps": 250880,
        "lineage": {"run_id": parent_id, "trigger_step": 250000},
    }
    monkeypatch.setattr(runner_module, "_campaign_run_reward_is_eligible", lambda *_a, **_k: True)
    monkeypatch.setattr(
        runner_module,
        "verify_ppo_lineage_checkpoint",
        lambda path, **_kwargs: parent_verified if Path(path).resolve() == source_path.resolve() else child_verified,
    )
    ancestry_calls = []

    def authenticated_reward_link(*_args, **kwargs):
        ancestry_calls.append(kwargs["verified_checkpoint"].get("checkpoint_sha256"))
        return "b" * 64

    monkeypatch.setattr(runner_module, "_campaign_reward_ancestry_link", authenticated_reward_link)
    monkeypatch.setattr(runner_module, "_resume_parent_run_id", lambda _config: source_id)
    monkeypatch.setattr(runner_module, "_valid_completed_rollout_steps", lambda *_a, **_k: {202752, 204800})

    report_attempt = {
        "race_lap_times_s": [20.0, 20.1, 20.2], "warmup_lap_times_s": [21.0],
        "collision_count": 0, "race_collisions": 0,
        "race_laps_completed": 3, "race_laps_required": 3,
        "stop_reason": "race_complete", "frontier_projection_valid": True,
        "camera_sample_count": 300, "camera_fresh_sample_count": 300,
        "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
    }
    report = {
        "model_sha256": child_sha,
        "protocol": "official_training_probe_1_plus_3",
        "validation_mode": "training_probe",
        "runs": [report_attempt],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service_attempt = {
        "run_id": evaluation_run_id, "score_status": "valid", "readiness_passed": True,
        "warmup_laps": 1, "race_laps_completed": 3, "race_laps_required": 3,
        "collision_count": 0, "race_collisions": 0,
        "camera_sample_count": 300, "camera_fresh_sample_count": 300,
        "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
        "evaluation_report": str(report_path),
    }
    history_record = {
        "evaluation_id": evaluation_id, "parent_run_id": parent_id,
        "validation_mode": "training_probe", "state": "complete",
            "created_utc": "2026-10-13T10:00:00Z",
        "model_path": str(child_path), "model_sha256": child_sha,
        "trigger_step": 250000,
        "result": {"readiness_passed": True, "runs": [service_attempt]},
    }
    (parent_dir / "ppo_optimizer_audit.jsonl").write_text(
        "\n".join(json.dumps({
            "num_timesteps": step, "metrics_finite": True, "gradients_finite": True,
            "parameters_finite": True, "optimizer_state_finite": True,
        }) for step in (202752, 204800)),
        encoding="utf-8",
    )
    snapshot = {
        "runs": [
            {"run_id": parent_id, "display_name": "Campaign parent", "config": request},
            {"run_id": source_id, "display_name": "Campaign source", "config": {"time_cost_per_simulated_second": COEFFICIENT}},
        ],
        "evaluation_history": [history_record],
    }
    state = {"events": []}
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "unused-state.json", repo_root=tmp_path, api=object(),
    )

    rows = runner._report_evidence(snapshot, state, PPOVerificationContext())
    assert len(rows) == 1
    row = rows[0]
    assert row["validation_mode"] == "training_probe"
    assert row["report_validation_mode"] == "training_probe"
    assert row["model_sha256"] == child_sha and row["model_lineage_verified"] is True
    assert row["trend_validation"]["completed_rollouts_since_branch"] == 2
    assert row["trend_validation"]["collision_count"] == 0
    assert row["reward_experiment_config_sha256"] == "b" * 64
    assert ancestry_calls == [child_sha]
    trend = derive_campaign_trend_evidence(rows)
    assert trend["trend_probe_count"] == 1
    assert trend["best_trend_checkpoint_sha256"] == child_sha
