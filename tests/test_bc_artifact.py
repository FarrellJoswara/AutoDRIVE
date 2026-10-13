import hashlib
import json
import os
from pathlib import Path

from src.layer3 import bc_artifact


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _aliases(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if bc_artifact.os.name == "nt":
        windows = str(resolved)
        drive = resolved.drive[0].lower()
        wsl = "/mnt/" + drive + "/" + "/".join(resolved.parts[1:])
        return {"host": windows, "windows": windows, "wsl": wsl}
    return {"host": str(resolved), "windows": "", "wsl": str(resolved)}


def _ready_parent(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict[str, str]]:
    run_dir = tmp_path / "parent-run"
    parent = run_dir / "behavior_cloned_policy.zip"
    parent_sha = _write(parent, b"test BC weights")
    parent_hashes = {}
    parent_paths = {}
    for name in ("source_checkpoint", "dataset", "source_summary", "source_config", "teacher_report"):
        source = tmp_path / "sources" / f"{name}.bin"
        parent_paths[name] = source
        parent_hashes[name] = _write(source, name.encode())
    dataset = parent_paths["dataset"]
    dataset_sha = parent_hashes["dataset"]
    pinned = {
        "run_id": "test-run",
        "checkpoint_sha256": parent_hashes["source_checkpoint"],
        "dataset_sha256": parent_hashes["dataset"],
        "summary_sha256": parent_hashes["source_summary"],
        "config_sha256": parent_hashes["source_config"],
        "teacher_report_sha256": parent_hashes["teacher_report"],
    }
    monkeypatch.setattr(bc_artifact, "OFFICIAL_BC_SOURCE", pinned)
    config_path = run_dir / "config.json"
    config = {
        "runtime": "official test runtime",
        "ppo": {"effective": {"learning_rate": 0.00001}},
        "observation_profile": "official_sensors_camera",
        "actions": {"throttle_mode": "bidirectional"},
        "behavior_cloning": {
            "status": "ready", "usable": True,
            "checkpoint_sha256": parent_sha,
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_hash = bc_artifact.sha256_file(config_path)
    summary = {
        "status": "ready",
        "usable": True,
        "readiness": {"ready": True},
        "checkpoint": _aliases(parent),
        "checkpoint_sha256": parent_sha,
        "provenance": {
            "kind": "rebuilt_from_preserved_demonstrations",
            "source_run_id": "test-run",
            "source_legacy_bytes_verified": True,
            "checkpoint_paths": _aliases(parent),
            "parents": {
                name: {"path": _aliases(path), "sha256": parent_hashes[name]}
                for name, path in parent_paths.items()
            },
            "fitted_intermediate_checkpoint": None,
            "artifact_config": {
                "path": _aliases(config_path), "sha256": config_hash,
            },
        },
    }
    (run_dir / "behavior_cloning_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return parent, dataset, {"parent_sha": parent_sha, "dataset_sha": dataset_sha}


def test_path_alias_record_requires_all_spellings_to_resolve_identically(tmp_path):
    target = tmp_path / "checkpoint.zip"
    aliases = _aliases(target)
    assert bc_artifact._resolve_path_record(aliases) == target.resolve()
    aliases["wsl"] = str(tmp_path / "different.zip")
    assert bc_artifact._resolve_path_record(aliases) is None


def test_actor_parameter_comparison_cache_is_content_and_code_bound(tmp_path, monkeypatch):
    parent = tmp_path / "parent.zip"
    checkpoint = tmp_path / "checkpoint.zip"
    _write(parent, b"parent archive bytes")
    _write(checkpoint, b"checkpoint archive bytes")
    calls = []

    def compare(parent_path, checkpoint_path):
        calls.append((parent_path.read_bytes(), checkpoint_path.read_bytes()))
        return {
            "compared_parameter_count": 1,
            "changed_parameter_keys": ["action_net.weight"],
            "changed_parameter_count": 1,
            "log_std_changed": False,
            "max_abs_delta": 0.25,
            "actor_changed": True,
        }

    bc_artifact._actor_parameter_change_cached.cache_clear()
    monkeypatch.setattr(bc_artifact, "_actor_parameter_change_uncached", compare)
    code_identity = {"value": "1" * 64}
    monkeypatch.setattr(bc_artifact, "_actor_comparison_code_identity", lambda: code_identity["value"])
    first = bc_artifact._actor_parameter_change(parent, checkpoint)
    second = bc_artifact._actor_parameter_change(parent, checkpoint)
    assert first == second
    assert first is not second
    assert len(calls) == 1
    first["changed_parameter_keys"].append("caller-mutation")
    assert "caller-mutation" not in bc_artifact._actor_parameter_change(parent, checkpoint)["changed_parameter_keys"]
    assert len(calls) == 1

    code_identity["value"] = "2" * 64
    bc_artifact._actor_parameter_change(parent, checkpoint)
    assert len(calls) == 2
    monkeypatch.setattr(bc_artifact, "_actor_comparison_runtime_identity", lambda: "other-runtime")
    bc_artifact._actor_parameter_change(parent, checkpoint)
    assert len(calls) == 3

    checkpoint.write_bytes(b"modified checkpoint archive bytes")
    third = bc_artifact._actor_parameter_change(parent, checkpoint)
    assert third == second
    assert len(calls) == 4

    parent.write_bytes(b"modified parent archive bytes")
    bc_artifact._actor_parameter_change(parent, checkpoint)
    assert len(calls) == 5


def test_rebuilt_bc_verifier_checks_config_and_parent_bytes(tmp_path, monkeypatch):
    parent, _, hashes = _ready_parent(tmp_path, monkeypatch)
    verified = bc_artifact.verify_bc_checkpoint(parent)
    assert verified is not None
    assert verified["checkpoint_sha256"] == hashes["parent_sha"]

    summary_path = parent.parent / "behavior_cloning_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parent_record = summary["provenance"]["parents"]["dataset"]
    tampered = Path(parent_record["path"]["host"])
    tampered.write_bytes(b"changed dataset")
    assert bc_artifact.verify_bc_checkpoint(parent) is None


def test_warmstart_verifier_requires_exact_noise_and_dataset_parent(tmp_path, monkeypatch):
    parent, dataset, hashes = _ready_parent(tmp_path, monkeypatch)
    candidate = parent.parent / "ppo_warmstart_std_001_005.zip"
    candidate_hash = _write(candidate, b"warmstart weights")
    metadata = {
        "kind": "ppo_warmstart_action_std_derivative",
        "checkpoint_paths": _aliases(candidate),
        "checkpoint_sha256": candidate_hash,
        "parent_checkpoint": {"path": _aliases(parent), "sha256": hashes["parent_sha"]},
        "parent_dataset": {"path": _aliases(dataset), "sha256": hashes["dataset_sha"]},
        "action_std": [0.01, 0.05],
        "source_action_std": [0.2, 0.2],
        "changed_policy_keys": ["log_std"],
        "deterministic_action_max_abs_delta": 0.0,
        "parent_is_verified_rebuild": True,
    }
    sidecar = candidate.parent / "ppo_warmstart_provenance.json"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert bc_artifact.verify_ppo_warmstart(candidate) is not None

    metadata["action_std"] = [0.2, 0.2]
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    assert bc_artifact.verify_ppo_warmstart(candidate) is None


def test_training_probe_snapshot_authenticates_config_and_bc_lineage(tmp_path, monkeypatch):
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv

    from src.layer3.official_train import _training_resume_lineage
    from src.layer3.official_policy import resume_ppo_checkpoint
    from src.layer3.train import make_model

    class ActionRewardEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                "lidar": gym.spaces.Box(0.0, 1.0, shape=(1081,), dtype=np.float32),
                "camera": gym.spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-5.0, 5.0, shape=(9,), dtype=np.float32),
            })
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), float(action[0]), False, False, {}

    parent, _, _ = _ready_parent(tmp_path, monkeypatch)
    env = DummyVecEnv([ActionRewardEnv])
    try:
        source = make_model(
            env, device="cpu", seed=0, tensorboard_log=None,
            learning_rate=1e-3, n_epochs=1, n_steps=8, batch_size=8,
            policy_architecture="lidar_camera_cnn",
        )
        source.save(str(parent.with_suffix("")))
        parent_hash = bc_artifact.sha256_file(parent)
        config_path = parent.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["behavior_cloning"]["checkpoint_sha256"] = parent_hash
        config_path.write_text(json.dumps(config), encoding="utf-8")
        summary_path = parent.parent / "behavior_cloning_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["checkpoint_sha256"] = parent_hash
        summary["provenance"]["artifact_config"]["sha256"] = bc_artifact.sha256_file(config_path)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        parent_lineage = _training_resume_lineage(parent)
        assert parent_lineage is not None

        model = make_model(
            env, device="cpu", seed=1, tensorboard_log=None,
            learning_rate=1e-3, n_epochs=2, n_steps=8, batch_size=8,
            policy_architecture="lidar_camera_cnn", ent_coef=0.0,
        )
        model.set_parameters(str(parent), exact_match=True)
        before = {
            key: value.detach().clone()
            for key, value in model.policy.state_dict().items()
            if "value_net" not in key and key != "log_std"
        }
        model.learn(total_timesteps=8, progress_bar=False)
        after = model.policy.state_dict()
        changed = [key for key, value in before.items() if not value.equal(after[key])]
        assert changed, "the offline PPO optimizer update must change actor weights"

        run = tmp_path / "probe-run"
        checkpoint = run / "checkpoints" / "automatic_evaluation_000001_steps_8.zip"
        checkpoint.parent.mkdir(parents=True)
        model.save(str(checkpoint.with_suffix("")))
        config_path = run / "config.json"
        config_path.write_text(json.dumps({
            "runtime": "official test runtime",
            "ppo": {"effective": {"learning_rate": 0.001, "n_steps": 8}},
            "automatic_evaluation_mode": "training_probe",
            "automatic_evaluation": {
                "enabled": True, "mode": "training_probe", "attempts_per_snapshot": 1,
            },
            "training_probe": {
                "enabled": True, "ranking": False,
                "simulation_timeout_s": 240.0, "wall_timeout_s": 240.0,
                "frontier_stagnation_s": 5.0,
                "lap_counts_by_environment": [1, 3],
                "stop_on_first_collision_including_warmup": True,
            },
        }), encoding="utf-8")
        context = {
            "python": "3.12.3", "platform": "linux",
            "packages": {"stable-baselines3": "2.8.0", "torch": "2.7.1"},
            "code_sha256": {"official_train.py": "b" * 64},
        }
        bc_artifact.write_ppo_lineage_sidecar(
            checkpoint,
            parent_lineage=parent_lineage,
            effective_config=config_path,
            optimizer_updates=int(model._n_updates),
            num_timesteps=int(model.num_timesteps),
            training_context=context,
            run_id="probe-run",
            kind="official_training_probe",
            extra={"optimizer_updates_at_trigger": 0, "trigger_step": 8},
        )
        lineage_path = checkpoint.with_name(f"{checkpoint.stem}.lineage.json")
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        assert lineage["actor_change"]["actor_changed"] is True
        assert lineage["actor_change"]["changed_parameter_keys"]

        verified = bc_artifact.verify_ppo_lineage_checkpoint(checkpoint)
        assert verified is not None
        assert verified["bc_ancestor_checkpoint_sha256"] == parent_hash
        assert verified["num_timesteps"] == 8
        verified_probe = bc_artifact.verify_training_probe_checkpoint(checkpoint)
        assert verified_probe is not None
        assert verified_probe["checkpoint_sha256"] == bc_artifact.sha256_file(checkpoint)

        resumed = resume_ppo_checkpoint(
            checkpoint,
            env=env,
            observation_profile="official_sensors_camera",
            learning_rate=1e-3,
            n_steps=8,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            seed=2,
            device="cpu",
            batch_size=8,
            ent_coef=0.0,
        )
        assert len(resumed.policy.optimizer.state) == 0
        resumed_before = {
            key: value.detach().clone()
            for key, value in resumed.policy.state_dict().items()
            if "value_net" not in key and key != "log_std"
        }
        resumed.learn(total_timesteps=8, progress_bar=False)
        resumed_after = resumed.policy.state_dict()
        assert any(
            not value.equal(resumed_after[key])
            for key, value in resumed_before.items()
        )
        assert len(resumed.policy.optimizer.state) > 0
        continuation = tmp_path / "continuation" / "checkpoints" / "post_update.zip"
        continuation.parent.mkdir(parents=True)
        resumed.save(str(continuation.with_suffix("")))
        continuation_config = continuation.parent.parent / "config.json"
        continuation_config.write_text(json.dumps({
            "ppo": {"effective": {"learning_rate": 0.001, "n_steps": 8}}
        }), encoding="utf-8")
        resumed_parent_lineage = _training_resume_lineage(checkpoint)
        assert resumed_parent_lineage is not None
        bc_artifact.write_ppo_lineage_sidecar(
            continuation,
            parent_lineage=resumed_parent_lineage,
            effective_config=continuation_config,
            optimizer_updates=int(resumed._n_updates),
            num_timesteps=int(resumed.num_timesteps),
            training_context=context,
            run_id="continuation",
        )
        continued_verification = bc_artifact.verify_ppo_lineage_checkpoint(continuation)
        assert continued_verification is not None
        assert continued_verification["bc_ancestor_checkpoint_sha256"] == parent_hash

        verification_context = bc_artifact.PPOVerificationContext()
        assert bc_artifact.verify_training_probe_checkpoint(
            checkpoint, verification_context=verification_context
        ) is not None
        reads_before_repeat = verification_context.hash_reads
        assert bc_artifact.verify_ppo_lineage_checkpoint(
            checkpoint, verification_context=verification_context
        ) is not None
        assert verification_context.memo_hits >= 1
        assert verification_context.hash_reads == reads_before_repeat

        # A same-size edit with restored mtime must still fail the pass seal.
        original_config = config_path.read_bytes()
        original_stat = config_path.stat()
        changed_config = original_config.replace(b"0.001", b"0.002")
        assert len(changed_config) == len(original_config)
        config_path.write_bytes(changed_config)
        os.utime(config_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        assert verification_context.seal() is False
        assert bc_artifact.verify_training_probe_checkpoint(checkpoint) is None
        config_path.write_bytes(original_config)

        lineage["bc_ancestor_checkpoint"]["sha256"] = "0" * 64
        lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
        assert bc_artifact.verify_ppo_lineage_checkpoint(checkpoint) is None
    finally:
        env.close()
