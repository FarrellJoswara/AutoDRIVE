from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.layer4.hub.campaign_phase_runner as runner_module
from src.layer3.bc_artifact import PPOVerificationContext
from src.layer4.hub.campaign_deadline import CAMPAIGN_ID


ROLLOUT = 2048


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_fixture(root: Path, *, name: str, parent_step: int, checkpoint_step: int,
                   trigger_step: int, audit_steps: list[int]) -> dict:
    id_map = {
        "seed": ("111111111111", "aaaaaaaaaaaa", "333333333333"),
        "qualified": ("222222222222", "bbbbbbbbbbbb", "444444444444"),
        "badproof": ("555555555555", "cccccccccccc", "666666666666"),
    }
    run_id, source_id, evaluation_run_id = id_map[name]
    evaluation_id = f"{name}-evaluation"
    run_dir = root / "logs" / "rl" / f"official_run_{run_id}"
    source_path = root / "logs" / "rl" / f"official_run_{source_id}" / "checkpoints" / "source.zip"
    checkpoint = run_dir / "checkpoints" / f"automatic_evaluation_steps_{trigger_step}.zip"
    report_path = root / "logs" / "rl" / f"official_run_{evaluation_run_id}" / "evaluation.json"
    for path in (source_path, checkpoint, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{name}:{path.name}".encode())
    source_sha = _digest(source_path)
    checkpoint_sha = _digest(checkpoint)
    config = {
        "resume": f"/runs/rl/official_run_{source_id}/checkpoints/source.zip",
        "n_steps": ROLLOUT,
        "automatic_evaluation_enabled": True,
        "automatic_evaluation_trigger": "steps",
        "automatic_evaluation_interval": 50000 if name != "seed" else ROLLOUT,
        "automatic_evaluation_mode": "training_probe",
        "time_cost_per_simulated_second": 0.01,
    }
    attempt = {
        "race_lap_times_s": [18.0, 18.1, 18.2],
        "warmup_lap_times_s": [18.5],
        "stop_reason": "race_complete",
        "complete": True,
        "score_status": "valid",
        "collision_count": 0,
        "race_collisions": 0,
        "race_laps_completed": 3,
        "race_laps_required": 3,
        "frontier_projection_valid": True,
        "route_projection_valid": True,
        "camera_sample_count": 100,
        "camera_fresh_sample_count": 100,
        "camera_stale_sample_count": 0,
        "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
    }
    report = {
        "model_sha256": checkpoint_sha,
        "protocol": "official_training_probe_1_plus_3",
        "validation_mode": "training_probe",
        "runs": [attempt],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service_attempt = {
        "run_id": evaluation_run_id,
        "score_status": "valid",
        "readiness_passed": True,
        "warmup_laps": 1,
        "race_laps_completed": 3,
        "race_laps_required": 3,
        "collision_count": 0,
        "race_collisions": 0,
        "camera_sample_count": 100,
        "camera_fresh_sample_count": 100,
        "camera_stale_sample_count": 0,
        "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
        "evaluation_report": str(report_path),
    }
    record = {
        "evaluation_id": evaluation_id,
        "parent_run_id": run_id,
        "state": "complete",
        "validation_mode": "training_probe",
        "model_path": str(checkpoint),
        "model_sha256": checkpoint_sha,
        "trigger_step": trigger_step,
        "result": {"readiness_passed": True, "runs": [service_attempt]},
    }
    (run_dir / "evaluation_history.jsonl").write_text(json.dumps({
        "evaluation_id": evaluation_id,
        "state": "complete",
        "latest_result": {
            "source": "automatic",
            "validation_mode": "training_probe",
            "protocol": "official_training_probe_1_plus_3",
            "readiness_passed": True,
            "model_sha256": checkpoint_sha,
            "completed_attempts": 1,
            "attempts_required": 1,
            "runs": [service_attempt],
        },
    }) + "\n", encoding="utf-8")
    (run_dir / "ppo_optimizer_audit.jsonl").write_text(
        "\n".join(json.dumps({
            "num_timesteps": step,
            "metrics_finite": True,
            "gradients_finite": True,
            "parameters_finite": True,
            "optimizer_state_finite": True,
        }) for step in audit_steps),
        encoding="utf-8",
    )
    verified_source = {
        "checkpoint_sha256": source_sha,
        "num_timesteps": parent_step,
        "lineage": {"run_id": source_id},
    }
    verified_child = {
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": source_sha,
        "num_timesteps": checkpoint_step,
        "lineage": {"run_id": run_id, "trigger_step": trigger_step},
    }
    return {
        "run_id": run_id,
        "source_id": source_id,
        "source_path": source_path,
        "checkpoint": checkpoint,
        "report_path": report_path,
        "record": record,
        "config": config,
        "verified_source": verified_source,
        "verified_child": verified_child,
        "checkpoint_sha": checkpoint_sha,
        "evaluation_id": evaluation_id,
    }


def _install_collector_fakes(monkeypatch, fixtures: list[dict], state_path: Path, state: dict) -> None:
    monkeypatch.setattr(runner_module, "load_campaign_state", lambda _path: state)
    monkeypatch.setattr(runner_module, "campaign_clock_status", lambda _state: {
        "phase": "final_validation", "remaining_seconds": 3600,
    })
    monkeypatch.setattr(runner_module, "_campaign_run_reward_is_eligible", lambda *_a, **_k: True)
    monkeypatch.setattr(runner_module, "_campaign_reward_ancestry_link", lambda *_a, **_k: "d" * 64)
    by_path = {}
    for fixture in fixtures:
        by_path[fixture["source_path"].resolve()] = fixture["verified_source"]
        by_path[fixture["checkpoint"].resolve()] = fixture["verified_child"]

    def verify(path, **_kwargs):
        return by_path.get(Path(path).resolve())

    monkeypatch.setattr(runner_module, "verify_ppo_lineage_checkpoint", verify)


def _snapshot(fixtures: list[dict]) -> dict:
    runs = []
    history = []
    for fixture in fixtures:
        runs.append({
            "run_id": fixture["run_id"],
            "display_name": "Campaign probe branch",
            "state": "complete",
            "config": fixture["config"],
        })
        runs.append({
            "run_id": fixture["source_id"],
            "display_name": "Campaign authenticated parent",
            "state": "complete",
            "config": {},
        })
        history.append(fixture["record"])
    return {"runs": runs, "evaluation_history": history, "replay": {"state": "complete"}, "telemetry": {}}


def test_real_collector_excludes_one_rollout_seed_and_sends_qualified_probe_to_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    seed = _probe_fixture(
        tmp_path, name="seed", parent_step=651264, checkpoint_step=653312,
        trigger_step=653312, audit_steps=[],
    )
    qualified = _probe_fixture(
        tmp_path, name="qualified", parent_step=800768, checkpoint_step=851968,
        trigger_step=850000, audit_steps=[802816, 804864],
    )
    state = {"campaign_id": CAMPAIGN_ID, "status": "active", "events": []}
    state_path = tmp_path / "campaign.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_collector_fakes(monkeypatch, [seed, qualified], state_path, state)
    runner = runner_module.CampaignPhaseRunner(state_path=state_path, repo_root=tmp_path, api=object())
    captured = {}

    def fake_builder(_state, evidence, *, repo_root):
        captured.update(evidence)
        return {"campaign_id": CAMPAIGN_ID, "finalists": [{"checkpoint_sha256": qualified["checkpoint_sha"]}]}

    monkeypatch.setattr(runner_module, "build_finalist_manifest", fake_builder)
    result = runner._create_finalist_manifest_if_ready(state, _snapshot([seed, qualified]))

    assert result["action"] == "finalist_manifest_created"
    assert [row["checkpoint_sha256"] for row in captured["candidates"]] == [qualified["checkpoint_sha"]]
    assert len(captured["excluded_seed_candidates"]) == 1
    excluded = captured["excluded_seed_candidates"][0]
    assert excluded["evaluation_id"] == seed["evaluation_id"]
    assert excluded["checkpoint_sha256"] == seed["checkpoint_sha"]
    assert excluded["parent_run_id"] == seed["run_id"]
    assert excluded["parent_global_timestep"] == 651264
    assert excluded["checkpoint_global_timestep"] == 653312
    assert excluded["post_parent_training_transitions"] == ROLLOUT
    assert excluded["post_parent_rollout_audit_steps"] == []
    assert excluded["post_parent_rollout_audit_count"] == 0
    assert excluded["exclusion_reason"] == "insufficient_post_parent_training"
    expected_history_path = tmp_path / "logs" / "rl" / f"official_run_{seed['run_id']}" / "evaluation_history.jsonl"
    assert Path(excluded["evaluation_history_path"]) == expected_history_path
    assert expected_history_path.is_file()


@pytest.mark.parametrize("damage", ["unverified_lineage", "missing_report"])
def test_real_collector_does_not_downgrade_claimed_pass_with_missing_or_tampered_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
):
    fixture = _probe_fixture(
        tmp_path, name="badproof", parent_step=800768, checkpoint_step=851968,
        trigger_step=850000, audit_steps=[802816, 804864],
    )
    state = {"campaign_id": CAMPAIGN_ID, "status": "active", "events": []}
    state_path = tmp_path / "campaign.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_collector_fakes(monkeypatch, [fixture], state_path, state)
    if damage == "unverified_lineage":
        real_verify = runner_module.verify_ppo_lineage_checkpoint

        def verify(path, **kwargs):
            if Path(path).resolve() == fixture["checkpoint"].resolve():
                return None
            return real_verify(path, **kwargs)

        monkeypatch.setattr(runner_module, "verify_ppo_lineage_checkpoint", verify)
    else:
        fixture["report_path"].unlink()
        fixture["record"]["result"]["runs"][0]["evaluation_report"] = str(fixture["report_path"])
    runner = runner_module.CampaignPhaseRunner(state_path=state_path, repo_root=tmp_path, api=object())

    with pytest.raises(RuntimeError, match="claims passing probe"):
        runner._create_finalist_manifest_if_ready(state, _snapshot([fixture]))
    assert not runner.finalists_manifest.exists()
