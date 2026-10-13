from __future__ import annotations

import json
from pathlib import Path

import src.layer4.hub.campaign_phase_runner as runner


PARENT_SHA = "b" * 64
CHILD_SHA = "c" * 64


def _audit(step: int) -> dict:
    return {
        "num_timesteps": step,
        "metrics_finite": True,
        "gradients_finite": True,
        "parameters_finite": True,
        "optimizer_state_finite": True,
    }


def test_global_trigger_bucket_binds_later_actual_checkpoint_and_audits(tmp_path: Path, monkeypatch):
    parent_step, trigger_bucket, actual_snapshot_step = 151_552, 200_000, 200_704
    parent = tmp_path / "logs" / "rl" / "parent.zip"
    child = tmp_path / "logs" / "rl" / "child.zip"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    run_dir = tmp_path / "logs" / "rl" / "official_run_branch"
    run_dir.mkdir(parents=True)
    audit_rows = [
        _audit(153_600),
        _audit(155_648),
        _audit(actual_snapshot_step),
        _audit(actual_snapshot_step),  # duplicate audit record is one optimizer update
        _audit(actual_snapshot_step + 2_048),  # later update is outside this frozen checkpoint
    ]
    (run_dir / "ppo_optimizer_audit.jsonl").write_text(
        "\n".join(json.dumps(row) for row in audit_rows), encoding="utf-8",
    )

    verified_parent = {"checkpoint_sha256": PARENT_SHA, "num_timesteps": parent_step}
    verified_child = {
        "checkpoint_sha256": CHILD_SHA,
        "parent_checkpoint_sha256": PARENT_SHA,
        "num_timesteps": actual_snapshot_step,
        "lineage": {"trigger_step": trigger_bucket},
    }
    monkeypatch.setattr(
        runner, "verify_ppo_lineage_checkpoint",
        lambda path: verified_parent if Path(path) == parent else verified_child,
    )
    assert runner._checkpoint_matches_trigger_bucket(verified_child, trigger_bucket)
    assert not runner._checkpoint_matches_trigger_bucket(verified_child, trigger_bucket + 1)
    assert runner._valid_completed_rollout_steps(
        audit_rows, parent_step=parent_step, trigger_step=actual_snapshot_step,
    ) == {153_600, 155_648, actual_snapshot_step}

    train_row = {
        "run_id": "branch",
        "output_dir": str(run_dir),
        "config": {"resume": str(parent)},
    }
    record = {
        "state": "complete",
        "validation_mode": "training_probe",
        "model_path": str(child),
        "model_sha256": CHILD_SHA,
        "trigger_step": trigger_bucket,
        "attempts": [{
            "stop_reason": "race_complete",
            "frontier_projection_valid": True,
            "camera_sample_count": 20,
            "camera_stale_sample_count": 0,
            "camera_missing_sample_count": 0,
            "camera_freshness_unknown_count": 0,
        }],
    }
    assert runner._pace_probe_is_screening_complete(record, train_row, tmp_path)

    # Repeated audit lines at one completed step must not fake two rollouts.
    (run_dir / "ppo_optimizer_audit.jsonl").write_text(
        "\n".join(json.dumps(_audit(155_648)) for _ in range(3)), encoding="utf-8",
    )
    assert not runner._pace_probe_is_screening_complete(record, train_row, tmp_path)


def test_global_probe_rejects_parent_plus_one_seed_and_bad_trigger_metadata(tmp_path: Path, monkeypatch):
    parent_step, trigger_bucket = 151_552, 200_000
    parent = tmp_path / "logs" / "rl" / "parent.zip"
    child = tmp_path / "logs" / "rl" / "child.zip"
    parent.parent.mkdir(parents=True)
    parent.write_bytes(b"parent")
    child.write_bytes(b"child")
    run_dir = tmp_path / "logs" / "rl" / "official_run_branch"
    run_dir.mkdir(parents=True)
    (run_dir / "ppo_optimizer_audit.jsonl").write_text(
        json.dumps(_audit(parent_step + 2_048)) + "\n", encoding="utf-8",
    )
    verified_parent = {"checkpoint_sha256": PARENT_SHA, "num_timesteps": parent_step}
    verified_seed = {
        "checkpoint_sha256": CHILD_SHA,
        "parent_checkpoint_sha256": PARENT_SHA,
        "num_timesteps": parent_step + 1,
        "lineage": {"trigger_step": trigger_bucket},
    }
    monkeypatch.setattr(
        runner, "verify_ppo_lineage_checkpoint",
        lambda path: verified_parent if Path(path) == parent else verified_seed,
    )
    train_row = {"run_id": "branch", "output_dir": str(run_dir), "config": {"resume": str(parent)}}
    record = {
        "state": "complete", "validation_mode": "training_probe",
        "model_path": str(child), "model_sha256": CHILD_SHA,
        "trigger_step": trigger_bucket,
        "attempts": [{
            "stop_reason": "race_complete", "frontier_projection_valid": True,
            "camera_sample_count": 20, "camera_stale_sample_count": 0,
            "camera_missing_sample_count": 0, "camera_freshness_unknown_count": 0,
        }],
    }
    assert not runner._pace_probe_is_screening_complete(record, train_row, tmp_path)

    verified_seed["num_timesteps"] = 200_704
    verified_seed["lineage"]["trigger_step"] = trigger_bucket + 1
    assert not runner._checkpoint_matches_trigger_bucket(verified_seed, trigger_bucket)
    assert not runner._pace_probe_is_screening_complete(record, train_row, tmp_path)
