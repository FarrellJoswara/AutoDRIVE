import hashlib
import json
import os
from pathlib import Path

import pytest

import src.layer4.hub.campaign_finalists as finalists_module
from src.layer4.hub.campaign_deadline import CAMPAIGN_ID, CAMPAIGN_PLAN
from src.layer4.hub.campaign_finalists import build_finalist_manifest
from src.layer4.hub.campaign_finalists import _atomic_create, _verified_campaign_ancestry


def _candidate(repo: Path, *, evaluation_id: str, run_id: str, laps: list[float]) -> dict:
    run_root = repo / "logs" / "rl" / f"official_run_{run_id}"
    run_root.mkdir(parents=True)
    checkpoint = repo / "logs" / "rl" / f"official_run_parent_{run_id}" / "checkpoints" / "probe.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint {evaluation_id}".encode())
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    lineage_path = checkpoint.with_name(f"{checkpoint.stem}.lineage.json")
    lineage_path.write_text(json.dumps({
        "checkpoint_sha256": checkpoint_sha,
        "bc_ancestor_checkpoint": {"sha256": "b" * 64},
    }), encoding="utf-8")
    training_config = {
        "runtime": "official IROS 2026 API and simulator images",
        "observation_profile": "official_sensors_camera",
        "ppo": {"checkpoint_interval_scope": "global_step_buckets", "effective": {"n_envs": 1}},
        "automatic_evaluation": {
            "enabled": True, "mode": "training_probe",
            "interval_scope": "global_step_buckets", "interval": 50000,
        },
        "ppo_exploration": {"profile": "bc_campaign"},
        "training_only_reward": {"time_cost_per_simulated_second": 0.0},
    }
    (checkpoint.parent.parent / "config.json").write_text(json.dumps(training_config), encoding="utf-8")
    (checkpoint.parent.parent / "ppo_optimizer_audit.jsonl").write_text("\n".join(
        json.dumps({"num_timesteps": step, "metrics_finite": True,
                    "gradients_finite": True, "parameters_finite": True,
                    "optimizer_state_finite": True})
        for step in (18000, 20000)
    ) + "\n", encoding="utf-8")
    report_path = run_root / "evaluation.json"
    report = {
        "model_sha256": checkpoint_sha,
        "validation_mode": "training_probe",
        "protocol": "official_training_probe_1_plus_3",
        "competitive_score_valid": False,
        "noncompetitive_diagnostic": True,
        "official_base_image": "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete",
        "simulator_image": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete",
        "competition": "RoboRacer Sim Racing League @ IROS 2026",
        "observation_profile": "official_sensors_camera",
        "steering_mode": "normal",
        "throttle_mode": "bidirectional",
        "deterministic": True,
        "runs": [{
            "complete": True,
            "stop_reason": "race_complete",
            "score_status": "valid",
            "race_laps_completed": 3,
            "race_laps_required": 3,
            "race_lap_times_s": laps,
            "warmup_lap_times_s": [max(laps) + 1],
            "race_collisions": 0,
            "warmup_collisions": 0,
            "valid_for_comparison": False,
            "noncompetitive_diagnostic": True,
            "frontier_projection_valid": True,
            "route_projection_valid": True,
            "camera_sample_count": 20,
            "camera_fresh_sample_count": 20,
            "camera_stale_sample_count": 0,
            "camera_missing_sample_count": 0,
            "camera_freshness_unknown_count": 0,
        }],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    history_path = repo / "logs" / "rl" / f"official_run_parent_{run_id}" / "evaluation_history.jsonl"
    history_path.write_text(json.dumps({
        "state": "complete",
        "evaluation_id": evaluation_id,
        "latest_result": {
            "source": "automatic",
            "validation_mode": "training_probe",
            "protocol": "official_training_probe_1_plus_3",
            "readiness_passed": True,
            "model_sha256": checkpoint_sha,
            "completed_attempts": 1,
            "attempts_required": 1,
            "runs": [{
                "run_id": run_id,
                "evaluation_report": f"/runs/rl/official_run_{run_id}/evaluation.json",
                "score_status": "valid",
                "warmup_laps": 1,
                "race_laps_completed": 3,
                "race_laps_required": 3,
                "collision_count": 0,
                "race_collisions": 0,
                "readiness_passed": True,
            }],
        },
    }), encoding="utf-8")
    return {
        "model_path": f"/runs/rl/official_run_parent_{run_id}/checkpoints/probe.zip",
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_history_path": f"/runs/rl/official_run_parent_{run_id}/evaluation_history.jsonl",
        "evaluation_id": evaluation_id,
        "probe_report_path": f"/runs/rl/official_run_{run_id}/evaluation.json",
        "probe_report_sha256": report_sha,
    }


def _state() -> dict:
    return {"campaign_id": CAMPAIGN_ID, "status": "active", "campaign_plan": CAMPAIGN_PLAN}


def _mock_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(finalists_module, "verify_training_probe_checkpoint", lambda path, **kwargs: {
        "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    })
    def fake_lineage(path, **kwargs):
        path = Path(path)
        if path.name == "base.zip":
            return {"num_timesteps": 15000}
        return {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "effective_config_sha256": hashlib.sha256((path.parent.parent / "config.json").read_bytes()).hexdigest(),
            "effective_config": str(path.parent.parent / "config.json"),
            "parent_checkpoint": str(path.parent / "base.zip"),
            "num_timesteps": 20000,
            "lineage": {"run_id": path.parent.parent.name.removeprefix("official_run_")},
        }
    monkeypatch.setattr(finalists_module, "verify_ppo_lineage_checkpoint", fake_lineage)


def test_finalist_builder_shortlists_two_fastest_authenticated_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _mock_verifiers(monkeypatch)
    monkeypatch.setattr(finalists_module, "campaign_clock_status", lambda *a, **k: {
        "phase": "final_validation", "remaining_seconds": 1000,
    })
    candidates = [
        _candidate(tmp_path, evaluation_id="eval-slow", run_id="slow", laps=[70.0, 71.0, 72.0]),
        _candidate(tmp_path, evaluation_id="eval-fast", run_id="fast", laps=[40.0, 41.0, 42.0]),
        _candidate(tmp_path, evaluation_id="eval-mid", run_id="mid", laps=[50.0, 51.0, 52.0]),
    ]
    manifest = build_finalist_manifest(
        _state(), {"campaign_id": CAMPAIGN_ID, "candidates": candidates}, repo_root=tmp_path,
    )
    assert len(manifest["finalists"]) == 2
    assert [row["evaluation_id"] for row in manifest["finalists"]] == ["eval-fast", "eval-mid"]
    assert manifest["final_validation"]["scored_laps_per_attempt"] == 10
    for row in manifest["finalists"]:
        sidecar = Path(row["lineage_sidecar_path"])
        assert row["lineage_sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        assert row["effective_config_sha256"] == row["training_config_sha256"]
    assert "nonranking" in manifest["selection_policy"]


def test_finalist_builder_fails_closed_outside_final_window_or_on_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _mock_verifiers(monkeypatch)
    candidate = _candidate(tmp_path, evaluation_id="eval-one", run_id="one", laps=[40.0, 41.0, 42.0])
    evidence = {"campaign_id": CAMPAIGN_ID, "candidates": [candidate]}
    monkeypatch.setattr(finalists_module, "campaign_clock_status", lambda *a, **k: {
        "phase": "training_window", "remaining_seconds": 2000,
    })
    with pytest.raises(ValueError, match="9h-to-12h"):
        build_finalist_manifest(_state(), evidence, repo_root=tmp_path)

    monkeypatch.setattr(finalists_module, "campaign_clock_status", lambda *a, **k: {
        "phase": "final_validation", "remaining_seconds": 1000,
    })
    Path(candidate["probe_report_path"].replace("/runs/rl/", str(tmp_path / "logs" / "rl") + "/")).write_text("{}")
    with pytest.raises(ValueError, match="report hash changed"):
        build_finalist_manifest(_state(), evidence, repo_root=tmp_path)


def test_finalist_builder_seals_all_candidate_dependencies_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _mock_verifiers(monkeypatch)
    monkeypatch.setattr(finalists_module, "campaign_clock_status", lambda *a, **k: {
        "phase": "final_validation", "remaining_seconds": 1000,
    })
    candidate = _candidate(tmp_path, evaluation_id="eval-seal", run_id="seal", laps=[40.0, 41.0, 42.0])
    config_path = tmp_path / "logs" / "rl" / "official_run_parent_seal" / "config.json"
    original = config_path.read_bytes()
    original_stat = config_path.stat()

    original_jsonl = finalists_module._jsonl
    def read_then_tamper_history(path, verification_context=None):
        rows = original_jsonl(path, verification_context)
        if path.name == "evaluation_history.jsonl":
            changed = original.replace(b"official", b"Official")
            assert len(changed) == len(original)
            config_path.write_bytes(changed)
            os.utime(config_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return rows
    monkeypatch.setattr(finalists_module, "_jsonl", read_then_tamper_history)
    with pytest.raises(RuntimeError, match="changed during verification"):
        build_finalist_manifest(
            _state(), {"campaign_id": CAMPAIGN_ID, "candidates": [candidate]}, repo_root=tmp_path,
        )


def test_finalist_manifest_publish_is_atomic_and_never_overwrites(tmp_path: Path):
    target = tmp_path / "campaign" / "finalists.json"
    payload = {"campaign_id": CAMPAIGN_ID, "finalists": []}
    digest = _atomic_create(target, payload)
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        _atomic_create(target, {"campaign_id": CAMPAIGN_ID, "finalists": [{"bad": True}]})
    assert json.loads(target.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("field,value", [
    ("race_collisions", False),
    ("warmup_collisions", False),
    ("camera_fresh_sample_count", True),
    ("camera_stale_sample_count", False),
    ("camera_missing_sample_count", False),
    ("camera_freshness_unknown_count", False),
])
def test_finalist_builder_rejects_boolean_report_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: bool,
):
    _mock_verifiers(monkeypatch)
    monkeypatch.setattr(finalists_module, "campaign_clock_status", lambda *a, **k: {
        "phase": "final_validation", "remaining_seconds": 1000,
    })
    candidate = _candidate(tmp_path, evaluation_id="eval-report", run_id="report", laps=[40.0, 41.0, 42.0])
    report_path = tmp_path / "logs" / "rl" / "official_run_report" / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runs"][0][field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    candidate["probe_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=r"camera|zero-collision|1\+3"):
        build_finalist_manifest(_state(), {"campaign_id": CAMPAIGN_ID, "candidates": [candidate]}, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["collision_count", "race_collisions"])
def test_finalist_builder_rejects_boolean_service_collision_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
):
    _mock_verifiers(monkeypatch)
    monkeypatch.setattr(finalists_module, "campaign_clock_status", lambda *a, **k: {
        "phase": "final_validation", "remaining_seconds": 1000,
    })
    candidate = _candidate(tmp_path, evaluation_id="eval-service", run_id="service", laps=[40.0, 41.0, 42.0])
    history_path = tmp_path / "logs" / "rl" / "official_run_parent_service" / "evaluation_history.jsonl"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["latest_result"]["runs"][0][field] = False
    history_path.write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="service attempt"):
        build_finalist_manifest(_state(), {"campaign_id": CAMPAIGN_ID, "candidates": [candidate]}, repo_root=tmp_path)


def test_finalist_ancestry_rejects_invalidated_run_even_with_zero_cost_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    child = tmp_path / "logs" / "rl" / "official_run_child" / "checkpoints" / "child.zip"
    parent = tmp_path / "logs" / "rl" / "official_run_1bd61c474512" / "checkpoints" / "parent.zip"
    child.parent.mkdir(parents=True)
    parent.parent.mkdir(parents=True)
    child.write_bytes(b"child")
    parent.write_bytes(b"parent")
    (parent.parent / "parent.lineage.json").write_text(json.dumps({
        "kind": "official_ppo_training_checkpoint",
    }), encoding="utf-8")
    child_config = child.parent.parent / "config.json"
    parent_config = parent.parent.parent / "config.json"
    child_config.write_text(json.dumps({"training_only_reward": {"time_cost_per_simulated_second": 0.0}}), encoding="utf-8")
    parent_config.write_text(json.dumps({"training_only_reward": {"time_cost_per_simulated_second": 5.0}}), encoding="utf-8")

    def fake_verify(path, **kwargs):
        path = Path(path).resolve()
        if path == child.resolve():
            return {"lineage": {"run_id": "child"}, "effective_config": str(child_config),
                    "parent_checkpoint": str(parent), "num_timesteps": 20000}
        if path == parent.resolve():
            return {"lineage": {"run_id": "1bd61c474512"}, "effective_config": str(parent_config),
                    "parent_checkpoint": str(tmp_path / "bc.zip"), "num_timesteps": 15000}
        return None

    monkeypatch.setattr(finalists_module, "verify_ppo_lineage_checkpoint", fake_verify)
    with pytest.raises(ValueError, match="invalidated literal per-second"):
        _verified_campaign_ancestry(child)
