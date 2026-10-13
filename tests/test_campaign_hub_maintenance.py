import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.layer4.hub import campaign_deadline as deadline


START = datetime(2026, 10, 12, 15, tzinfo=timezone.utc)
BOOT = "linux-boot-test"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _campaign(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "updated_policy_probe": {"status": "PASS", "useful": True, "checkpoint_sha256": HASH_A},
        "single_environment_queue_interruption": {"status": "PASS", "passed": True},
    }), encoding="utf-8")
    state_path = tmp_path / "active_campaign.json"
    deadline.initialize_campaign(state_path, gate, campaign_id="maintenance-test",
                                 utc_now=START, monotonic_ns=1000, boot_id=BOOT)
    deadline.amend_campaign_deadline_policy(
        state_path, authorization="user_override", reason="train until readiness",
        utc_now=START, monotonic_ns=1000,
    )
    return state_path


def _identity(pid, code_hash):
    return {
        "pid": pid, "process_start_identity": f"pid-{pid}-start", "boot_id": "test-boot",
        "executable": str(Path(sys.executable).resolve()), "argv": [sys.executable, "hub"],
        "source_sha256": code_hash,
    }


def test_hub_maintenance_is_checkpoint_bound_single_resume_and_clock_preserving(tmp_path, monkeypatch):
    state_path = _campaign(tmp_path)
    repo = tmp_path / "repo"
    checkpoint = repo / "logs/rl/official_run_123456abcdef/checkpoints/post.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    effective_config = {
        "learning_rate": 3e-5, "n_envs": 1, "n_steps": 2048,
        "time_cost_per_simulated_second": 0.0,
    }
    config_path = checkpoint.parent.parent / "config.json"
    config_path.write_text(json.dumps(effective_config), encoding="utf-8")
    (checkpoint.parent.parent / "manager_config.json").write_text(json.dumps({
        "run_id": "123456abcdef", "config": effective_config,
    }), encoding="utf-8")
    config_hash = HASH_B
    contract_hash = deadline.campaign_training_contract_sha256(effective_config)
    first_checkpoint = repo / "logs/rl/official_run_abcdef123456/checkpoints/update.zip"
    first_checkpoint.parent.mkdir(parents=True)
    first_checkpoint.write_bytes(b"update")
    first_sha = hashlib.sha256(first_checkpoint.read_bytes()).hexdigest()

    def fake_verify(path, **_kwargs):
        resolved = str(path)
        if resolved.endswith("post.zip"):
            return {
                "checkpoint_sha256": checkpoint_sha,
                "effective_config_sha256": config_hash,
                "effective_config": str(config_path),
                "num_timesteps": 450560,
                "parent_checkpoint_sha256": HASH_C,
                "lineage": {"post_optimizer_update": True},
            }
        return {
            "checkpoint_sha256": first_sha,
            "effective_config_sha256": HASH_C,
            "num_timesteps": 452608,
            "parent_checkpoint_sha256": checkpoint_sha,
            "lineage": {"post_optimizer_update": True},
        }

    monkeypatch.setattr(deadline, "_campaign_relative_path", lambda path: path.relative_to(repo).as_posix())
    monkeypatch.setattr(deadline, "_resolve_campaign_resume_path", lambda _value: checkpoint)
    monkeypatch.setattr("src.layer3.bc_artifact.sha256_file", lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr("src.layer3.bc_artifact.verify_ppo_lineage_checkpoint", fake_verify)

    original = deadline.load_campaign_state(state_path)
    original_clock = deadline._immutable_clock_fields(original)
    old_hub = _identity(10, HASH_A)
    old_hub["source_sha256"] = None
    old_hub["source_attestation"] = "unavailable_legacy_process"
    new_hub = _identity(11, HASH_B)
    maintenance_id = "1" * 32
    started = deadline.begin_campaign_hub_maintenance(
        state_path, maintenance_id=maintenance_id, trainer_run_id="123456abcdef",
        checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_sha,
        global_timestep=450560, effective_config_sha256=config_hash,
        training_contract_sha256=contract_hash, old_hub_identity=old_hub,
        reason="load amended campaign policy",
    )
    assert deadline._immutable_clock_fields(started) == original_clock
    assert len(started["events"]) == len(original["events"]) + 1
    assert started["hub_maintenance"]["intent"]["old_hub_identity"] == old_hub
    assert not deadline.campaign_admission_decision(started, "training")[0]
    assert not deadline.campaign_admission_decision(started, "evaluation")[0]

    def try_second_start(candidate_id):
        return deadline.begin_campaign_hub_maintenance(
            state_path, maintenance_id=candidate_id, trainer_run_id="123456abcdef",
            checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_sha,
            global_timestep=450560, effective_config_sha256=config_hash,
            training_contract_sha256=contract_hash, old_hub_identity=old_hub,
            reason="attempt overlapping handoff",
        )

    with pytest.raises(RuntimeError, match="another campaign hub maintenance"):
        try_second_start("3" * 32)

    resume_config = {**effective_config, "resume": "/runs/rl/official_run_123456abcdef/checkpoints/post.zip"}
    resume_hash = deadline.campaign_effective_config_sha256(resume_config)
    authorized = deadline.authorize_campaign_hub_maintenance_resume(
        state_path, maintenance_id, new_hub_identity=new_hub,
        resume_config=resume_config, resume_config_sha256=resume_hash,
        training_contract_sha256=contract_hash,
    )
    prior_authorization = authorized["hub_maintenance"]["resume_authorization"]
    with pytest.raises(RuntimeError, match="another campaign hub maintenance"):
        try_second_start("4" * 32)
    prior_authorization_sha = deadline._maintenance_authorization_sha256(prior_authorization)
    replacement_hub = _identity(12, HASH_C)
    superseded = deadline.supersede_campaign_hub_maintenance_resume_authorization(
        state_path, maintenance_id, prior_authorization_sha256=prior_authorization_sha,
        new_hub_identity=replacement_hub,
    )
    supersession_event = superseded["events"][-1]
    assert supersession_event["event"] == "campaign_hub_maintenance_resume_authorization_superseded"
    assert supersession_event["prior_authorization_sha256"] == prior_authorization_sha
    assert supersession_event["prior_new_hub_identity"] == new_hub
    current_authorization = superseded["hub_maintenance"]["resume_authorization"]
    assert deadline._maintenance_authorization_binding(current_authorization) == deadline._maintenance_authorization_binding(prior_authorization)
    assert current_authorization["new_hub_identity"] == replacement_hub
    assert deadline.supersede_campaign_hub_maintenance_resume_authorization(
        state_path, maintenance_id, prior_authorization_sha256=prior_authorization_sha,
        new_hub_identity=replacement_hub,
    ) == superseded
    with pytest.raises(RuntimeError, match="prior authorization SHA"):
        deadline.supersede_campaign_hub_maintenance_resume_authorization(
            state_path, maintenance_id, prior_authorization_sha256=HASH_A,
            new_hub_identity=_identity(13, HASH_A),
        )

    context = {"normalized_config": resume_config, "new_hub_identity": replacement_hub}
    assert deadline.campaign_admission_decision(superseded, "training", maintenance_resume=context)[0]
    assert not deadline.campaign_admission_decision(superseded, "evaluation", maintenance_resume=context)[0]
    with pytest.raises(ValueError, match="authorized maintenance resume"):
        deadline.claim_campaign_hub_maintenance_resume(
            state_path, kind="training", normalized_config={**resume_config, "resume": "/runs/other.zip"},
            new_hub_identity=replacement_hub, resumed_run_id="abcdef123456",
        )
    claimed = deadline.claim_campaign_hub_maintenance_resume(
        state_path, kind="training", normalized_config=resume_config, new_hub_identity=replacement_hub,
        resumed_run_id="abcdef123456",
    )
    assert claimed["hub_maintenance"]["resume_claimed"] is True
    assert not deadline.campaign_admission_decision(claimed, "training", maintenance_resume=context)[0]
    with pytest.raises(RuntimeError, match="another campaign hub maintenance"):
        try_second_start("5" * 32)
    with pytest.raises(RuntimeError, match="claimed maintenance resume"):
        deadline.supersede_campaign_hub_maintenance_resume_authorization(
            state_path, maintenance_id,
            prior_authorization_sha256=deadline._maintenance_authorization_sha256(current_authorization),
            new_hub_identity=_identity(13, HASH_A),
        )

    evidence = {
        "operator_stop_completed": True, "neutral_command_published": True,
        "transport_acknowledged": False, "transport_fresh_after_resume": True,
        "fresh_optimizer_constructed": True, "global_timestep_inherited": True,
        "rollout_buffer_reset": True, "loaded_checkpoint_sha256": checkpoint_sha,
        "loaded_global_timestep": 450560, "resume_effective_config_sha256": HASH_C,
        "resume_training_contract_sha256": contract_hash,
        "first_postupdate_checkpoint_path": str(first_checkpoint),
        "first_postupdate_checkpoint_sha256": first_sha,
        "first_postupdate_global_timestep": 452608, "first_postupdate_audit_finite": True,
    }
    completed = deadline.complete_campaign_hub_maintenance(
        state_path, maintenance_id=maintenance_id, resumed_run_id="abcdef123456",
        new_hub_identity=replacement_hub, resume_evidence=evidence,
    )
    assert deadline._immutable_clock_fields(completed) == original_clock
    assert completed["hub_maintenance"]["completed"]["event"] == "campaign_hub_maintenance_completed"
    assert deadline.campaign_admission_decision(completed, "training")[0]
    assert deadline.complete_campaign_hub_maintenance(
        state_path, maintenance_id=maintenance_id, resumed_run_id="abcdef123456",
        new_hub_identity=replacement_hub, resume_evidence=evidence,
    ) == completed
    assert deadline.begin_campaign_hub_maintenance(
        state_path, maintenance_id=maintenance_id, trainer_run_id="123456abcdef",
        checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_sha,
        global_timestep=450560, effective_config_sha256=config_hash,
        training_contract_sha256=contract_hash, old_hub_identity=old_hub,
        reason="load amended campaign policy",
    ) == completed

    next_maintenance_id = "6" * 32
    second_started = deadline.begin_campaign_hub_maintenance(
        state_path, maintenance_id=next_maintenance_id, trainer_run_id="123456abcdef",
        checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_sha,
        global_timestep=450560, effective_config_sha256=config_hash,
        training_contract_sha256=contract_hash, old_hub_identity=replacement_hub,
        reason="recover after a later infrastructure timeout",
    )
    assert second_started["hub_maintenance"]["maintenance_id"] == next_maintenance_id
    assert len(second_started["hub_maintenance_history"]) == 1
    archived = second_started["hub_maintenance_history"][0]
    assert archived["maintenance_id"] == maintenance_id
    assert archived["maintenance_record"] == completed["hub_maintenance"]
    assert archived["archive_event"]["event"] == "campaign_hub_maintenance_archived"
    assert archived["archive_event"]["next_maintenance_id"] == next_maintenance_id
    assert deadline.campaign_hub_maintenance_status(second_started) == second_started["hub_maintenance"]
    assert deadline._immutable_clock_fields(second_started) == original_clock
    tampered_history = json.loads(json.dumps(second_started))
    tampered_history["hub_maintenance_history"][0]["maintenance_record"]["intent"]["reason"] = "rewritten"
    with pytest.raises(ValueError, match="archive event is missing or mismatched"):
        deadline.campaign_hub_maintenance_status(tampered_history)


def test_hub_maintenance_rejects_contract_drift_and_wrong_identity(tmp_path, monkeypatch):
    state_path = _campaign(tmp_path)
    repo = tmp_path / "repo"
    checkpoint = repo / "logs/rl/official_run_123456abcdef/checkpoints/post.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = {"learning_rate": 3e-5, "n_envs": 1, "n_steps": 2048}
    config_path = checkpoint.parent.parent / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (checkpoint.parent.parent / "manager_config.json").write_text(json.dumps({
        "run_id": "123456abcdef", "config": config,
    }), encoding="utf-8")
    config_hash = HASH_B
    contract_hash = deadline.campaign_training_contract_sha256(config)
    monkeypatch.setattr(deadline, "_campaign_relative_path", lambda path: path.relative_to(repo).as_posix())
    monkeypatch.setattr(deadline, "_resolve_campaign_resume_path", lambda _value: checkpoint)
    monkeypatch.setattr("src.layer3.bc_artifact.sha256_file", lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr("src.layer3.bc_artifact.verify_ppo_lineage_checkpoint", lambda *_a, **_k: {
        "checkpoint_sha256": checkpoint_sha, "effective_config_sha256": config_hash,
        "effective_config": str(config_path), "num_timesteps": 100,
        "lineage": {"post_optimizer_update": True},
    })
    deadline.begin_campaign_hub_maintenance(
        state_path, maintenance_id="2" * 32, trainer_run_id="123456abcdef",
        checkpoint_path=checkpoint, checkpoint_sha256=checkpoint_sha, global_timestep=100,
        effective_config_sha256=config_hash, training_contract_sha256=contract_hash,
        old_hub_identity=_identity(10, HASH_A), reason="reload",
    )
    resume = {**config, "resume": "/runs/rl/official_run_123456abcdef/checkpoints/post.zip"}
    altered = {**resume, "learning_rate": 1e-4}
    with pytest.raises(ValueError, match="training contract"):
        deadline.authorize_campaign_hub_maintenance_resume(
            state_path, "2" * 32, new_hub_identity=_identity(11, HASH_B),
            resume_config=altered, resume_config_sha256=deadline.campaign_effective_config_sha256(altered),
            training_contract_sha256=contract_hash,
        )


def test_legacy_old_hub_identity_requires_explicit_unavailable_attestation():
    legacy = _identity(12, HASH_A)
    legacy["source_sha256"] = None
    legacy["source_attestation"] = "unavailable_legacy_process"
    deadline._validate_hub_identity(
        legacy, "old_hub_identity", allow_legacy_source_unavailable=True
    )

    for invalid in (
        {**legacy, "source_attestation": "unknown"},
        {key: value for key, value in legacy.items() if key != "source_attestation"},
        {**legacy, "source_sha256": "not-a-hash"},
    ):
        with pytest.raises(ValueError, match="source_sha256"):
            deadline._validate_hub_identity(
                invalid, "old_hub_identity", allow_legacy_source_unavailable=True
            )

    with pytest.raises(ValueError, match="source_sha256"):
        deadline._validate_hub_identity(legacy, "new_hub_identity")
