"""Rebuild a BC policy from the preserved official teacher dataset offline.

This command never creates or connects to a simulator environment. It accepts
only the pinned clean source run, validates array headers and parent hashes
before model allocation, and writes every artifact to a fresh output folder.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer3.bc_artifact import (
    OFFICIAL_BC_SOURCE,
    checkpoint_path_aliases,
    resolve_checkpoint_path,
    sha256_file,
)


EXPECTED_ARRAYS = {
    "lidar": ((48_862, 1081), np.dtype("float32")),
    "camera": ((48_862, 90, 160, 3), np.dtype("uint8")),
    "state": ((48_862, 9), np.dtype("float32")),
    "actions": ((48_862, 2), np.dtype("float32")),
}

SOURCE_RUNTIME = {
    "python": "3.12.3",
    "numpy": "1.26.4",
    "torch": "2.7.1+cu126",
    "stable_baselines3": "2.8.0",
    "gymnasium": "1.2.3",
    "cloudpickle": "3.1.2",
}

LEGACY_EVALUATION = {
    "checkpoint_sha256": "b395bc91c3d7b42b7ada87f11916af82ced6093fcf51d9a22913c3eba812ea7c",
    "run_id": "da2a678ec87c",
    "transitions": 4000,
    "outcome": "separate historical BC artifact; collision failures; not a parent of this rebuild",
}


def _load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def _read_npz_headers(dataset_path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    try:
        with zipfile.ZipFile(dataset_path) as archive:
            names = archive.namelist()
            expected_names = {f"{key}.npy" for key in EXPECTED_ARRAYS}
            if set(names) != expected_names or len(names) != len(expected_names):
                raise ValueError(
                    f"demonstration archive members must be {sorted(expected_names)}, got {sorted(names)}"
                )
            for member in names:
                with archive.open(member) as stream:
                    npy_version = np.lib.format.read_magic(stream)
                    if npy_version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif npy_version in {(2, 0), (3, 0)}:
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"unsupported NPY header version {npy_version} in {member}")
                key = member[:-4]
                expected_shape, expected_dtype = EXPECTED_ARRAYS[key]
                if tuple(shape) != expected_shape:
                    raise ValueError(f"{key} shape must be {expected_shape}, got {tuple(shape)}")
                if np.dtype(dtype) != expected_dtype:
                    raise ValueError(f"{key} dtype must be {expected_dtype}, got {dtype}")
                if fortran_order:
                    raise ValueError(f"{key} must use C order")
                headers[key] = (tuple(shape), np.dtype(dtype))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid demonstration NPZ archive: {dataset_path}") from exc
    return headers


def _source_provenance(source_run: Path, dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_run = f"official_run_{OFFICIAL_BC_SOURCE['run_id']}"
    if source_run.name != expected_run:
        raise ValueError(f"source run must be {expected_run}, got {source_run.name}")

    checkpoint_path = source_run / "behavior_cloned_policy.zip"
    summary_path = source_run / "behavior_cloning_summary.json"
    config_path = source_run / "config.json"
    teacher_report_path = ROOT / "competition/iros2026/results/official_teacher_validation_1627565e7893.json"
    expected_dataset_path = source_run / "behavior_cloning_demonstrations.npz"
    if dataset_path != expected_dataset_path.resolve():
        raise ValueError(f"dataset must be the preserved source archive: {expected_dataset_path}")

    hashes = {
        "source_checkpoint": _verify_hash(checkpoint_path, OFFICIAL_BC_SOURCE["checkpoint_sha256"], "source checkpoint"),
        "dataset": _verify_hash(dataset_path, OFFICIAL_BC_SOURCE["dataset_sha256"], "demonstration archive"),
        "source_summary": _verify_hash(summary_path, OFFICIAL_BC_SOURCE["summary_sha256"], "legacy BC summary"),
        "source_config": _verify_hash(config_path, OFFICIAL_BC_SOURCE["config_sha256"], "source config"),
        "teacher_report": _verify_hash(teacher_report_path, OFFICIAL_BC_SOURCE["teacher_report_sha256"], "teacher validation report"),
    }

    source_summary = _load_json(summary_path)
    source_config = _load_json(config_path)
    teacher_report = _load_json(teacher_report_path)
    reported_checkpoint = source_summary.get("checkpoint")
    if not isinstance(reported_checkpoint, str) or resolve_checkpoint_path(reported_checkpoint) != checkpoint_path.resolve():
        raise ValueError("legacy summary checkpoint path does not resolve to the pinned source checkpoint")

    config_bc = source_config.get("behavior_cloning")
    config_summary = config_bc.get("summary") if isinstance(config_bc, dict) else None
    source_collection = source_summary.get("collection")
    report_validation = teacher_report.get("validation")
    report_bc = teacher_report.get("behavior_cloning")
    if not all(isinstance(item, dict) for item in (config_bc, config_summary, source_collection, report_validation, report_bc)):
        raise ValueError("source config, BC summary, collection, or teacher report has an unexpected schema")

    transitions = int(source_summary.get("transitions", -1))
    counts = (
        transitions,
        int(source_collection.get("transitions", -1)),
        int(config_summary.get("transitions", -1)),
        int(report_bc.get("transitions", -1)),
    )
    if len(set(counts)) != 1 or counts[0] != EXPECTED_ARRAYS["actions"][0][0]:
        raise ValueError(f"dataset transition counts disagree across records: {counts}")
    safety_checks = {
        "report_status_verified_success": teacher_report.get("status") == "verified_success",
        "official_lap_target_met": int(report_validation.get("actual_official_laps", 0)) >= int(report_validation.get("required_laps", 10)) == 10,
        "episode_won": report_validation.get("episode_won") is True,
        "zero_collisions": int(report_validation.get("collision_count", -1)) == 0,
        "no_stagnation": report_validation.get("stagnation_detected") is False,
        "one_clean_successful_episode": (
            int(report_validation.get("completed_episodes", -1)) == 1
            and int(report_validation.get("successful_episodes", -1)) == 1
        ),
        "no_failed_source_episodes": (
            int(source_collection.get("completed_episodes", -1))
            == int(source_collection.get("successful_episodes", -2))
        ),
        "source_summary_ready": source_summary.get("status") == "ready" and source_summary.get("usable") is True,
    }
    if not all(safety_checks.values()):
        raise ValueError(f"source teacher readiness gate failed: {safety_checks}")
    return source_summary, source_config, teacher_report, {
        "paths": {
            "source_checkpoint": checkpoint_path,
            "dataset": dataset_path,
            "source_summary": summary_path,
            "source_config": config_path,
            "teacher_report": teacher_report_path,
        },
        "hashes": hashes,
        "safety_checks": safety_checks,
        "transitions": transitions,
        "config_summary": config_summary,
    }


def _fit_parameters(config_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    recorded_params = {
        "epochs": config_summary.get("epochs"),
        "batch_size": config_summary.get("batch_size"),
        "learning_rate": config_summary.get("learning_rate"),
    }
    requested = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    plan_defaults = {"epochs": 8, "batch_size": 256, "learning_rate": 1e-3}
    for name in ("epochs", "batch_size", "learning_rate"):
        recorded = recorded_params[name]
        if requested[name] is not None:
            selected[name] = {"value": requested[name], "origin": "explicit_rebuild_plan"}
        elif recorded is not None:
            selected[name] = {"value": recorded, "origin": "source_config.behavior_cloning.summary"}
        else:
            selected[name] = {
                "value": plan_defaults[name],
                "origin": "plan_supplied_missing_from_legacy_source",
            }
    selected["epochs"]["value"] = int(selected["epochs"]["value"])
    selected["batch_size"]["value"] = int(selected["batch_size"]["value"])
    selected["learning_rate"]["value"] = float(selected["learning_rate"]["value"])
    if selected["epochs"]["value"] < 1 or selected["batch_size"]["value"] < 1 or selected["learning_rate"]["value"] <= 0:
        raise ValueError("BC rebuild epochs, batch size, and learning rate must be positive")
    return selected


def _validate_loaded_dataset(dataset: dict[str, np.ndarray]) -> int:
    if set(dataset) != set(EXPECTED_ARRAYS):
        raise ValueError(f"loaded archive keys differ from the checked headers: {sorted(dataset)}")
    for key, (shape, dtype) in EXPECTED_ARRAYS.items():
        value = dataset[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"loaded {key} changed shape/dtype: {value.shape}/{value.dtype}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"dataset {key} contains NaN or infinite values")
    if dataset["lidar"].min() < 0.0 or dataset["lidar"].max() > 1.0:
        raise ValueError("normalized LiDAR values must be within [0,1]")
    if np.max(np.abs(dataset["actions"])) > 1.0 + 1e-6:
        raise ValueError("teacher action labels must be within [-1,1]")
    return int(len(dataset["actions"]))


def _runtime_record() -> dict[str, Any]:
    actual = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "stable_baselines3": version("stable-baselines3"),
        "gymnasium": version("gymnasium"),
        "cloudpickle": version("cloudpickle"),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in SOURCE_RUNTIME.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"rebuild runtime does not match checkpoint-embedded package versions: {mismatches}")
    return {
        "python": actual["python"],
        "packages": {key: value for key, value in actual.items() if key != "python"},
        "executable": sys.executable,
        "platform": platform.platform(),
        "source_checkpoint_embedded_versions_match": True,
        "byte_identical_runtime_claimed": False,
    }


def _architecture_record(model: Any) -> dict[str, Any]:
    policy = model.policy
    spaces = model.observation_space.spaces
    return {
        "architecture": "lidar_camera_cnn",
        "policy_class": type(policy).__name__,
        "features_extractor_class": type(policy.features_extractor).__name__,
        "extractor_kwargs": {
            "features_dim": int(policy.features_extractor.features_dim),
            "lidar_pooling": bool(getattr(policy.features_extractor, "lidar_pooling", True)),
            "camera_normalization": "RGB uint8 converted to float and divided by 255 once in extractor",
        },
        "actor_head": type(policy.action_net).__name__,
        "critic_head": type(policy.value_net).__name__,
        "actor_critic_mlp": type(policy.mlp_extractor).__name__,
        "observation_shapes": {key: list(space.shape) for key, space in spaces.items()},
        "action_shape": list(model.action_space.shape),
        "normalize_images": bool(policy.normalize_images),
    }


def _deterministic_action(model: Any, observations: dict[str, np.ndarray]) -> np.ndarray:
    actions, _ = model.predict(observations, deterministic=True)
    return np.asarray(actions, dtype=np.float32)


def _validation_loss(model: Any, dataset: dict[str, np.ndarray], batch_size: int) -> float:
    import torch
    import torch.nn.functional as F

    count = len(dataset["actions"])
    validation_count = max(1, int(round(count * 0.1)))
    indices = np.arange(count - validation_count, count, dtype=np.int64)
    policy = model.policy
    losses = []
    policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = np.asarray(indices[start:start + batch_size], dtype=np.int64)
            observations, _ = policy.obs_to_tensor({
                key: value[batch] for key, value in dataset.items() if key != "actions"
            })
            actions = torch.as_tensor(dataset["actions"][batch], device=policy.device)
            predicted = policy.get_distribution(observations).distribution.mean
            losses.append(float(F.smooth_l1_loss(predicted, actions).item()))
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--fitted-checkpoint", type=Path,
        help="package an already fitted BC checkpoint without repeating its training",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    source_run = args.source_run.resolve()
    output = args.output.resolve()
    fitted_checkpoint = args.fitted_checkpoint.resolve() if args.fitted_checkpoint else None
    if output.exists():
        raise FileExistsError(f"refusing to reuse an existing output path: {output}")
    if output == source_run or output in source_run.parents or source_run in output.parents:
        raise ValueError("rebuild output and source run must be separate paths")
    if fitted_checkpoint is not None and (not fitted_checkpoint.is_file() or fitted_checkpoint.suffix != ".zip"):
        raise ValueError(f"fitted checkpoint must be an existing .zip archive: {fitted_checkpoint}")
    if fitted_checkpoint is not None and output in fitted_checkpoint.parents:
        raise ValueError("output must not contain the fitted checkpoint being authenticated")

    source_summary, source_config, teacher_report, source = _source_provenance(source_run, dataset_path)
    headers = _read_npz_headers(dataset_path)
    fit = _fit_parameters(source["config_summary"], args)
    runtime = _runtime_record()
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the selected environment")
    with np.load(dataset_path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in sorted(EXPECTED_ARRAYS)}
    count = _validate_loaded_dataset(dataset)
    if count != source["transitions"]:
        raise ValueError("loaded dataset transition count changed after header validation")

    from stable_baselines3.common.vec_env import DummyVecEnv

    from scripts.initialize_camera_champion import _ObservationEnv
    from src.layer3.behavior_cloning import behavior_clone_actor, initialize_behavior_cloned_action_std, save_behavior_cloned_checkpoint
    from src.layer3.train import make_model
    from stable_baselines3 import PPO

    seed = int(args.seed)
    environment = DummyVecEnv([_ObservationEnv])
    try:
        fitted_hash = sha256_file(fitted_checkpoint) if fitted_checkpoint is not None else None
        if fitted_checkpoint is None:
            model = make_model(
                environment,
                device=device,
                seed=seed,
                tensorboard_log=None,
                learning_rate=1e-5,
                n_epochs=4,
                n_steps=2048,
                gamma=0.9995,
                gae_lambda=0.95,
                target_kl=0.01,
                policy_architecture="lidar_camera_cnn",
            )
            result = behavior_clone_actor(
                model,
                dataset,
                epochs=fit["epochs"]["value"],
                batch_size=fit["batch_size"]["value"],
                learning_rate=fit["learning_rate"]["value"],
                seed=seed,
            )
            result["initial_action_std"] = initialize_behavior_cloned_action_std(model)
        else:
            model = PPO.load(str(fitted_checkpoint), env=environment, device=device)
            fitted_std = model.policy.log_std.detach().exp().cpu().numpy()
            if not np.allclose(fitted_std, [0.2, 0.2], atol=1e-7, rtol=1e-6):
                raise ValueError(f"fitted checkpoint does not have the expected BC standard deviation: {fitted_std}")
            result = {
                "transitions": float(count),
                "initial_validation_loss": None,
                "final_validation_loss": _validation_loss(model, dataset, fit["batch_size"]["value"]),
                "mean_training_loss": None,
                "epochs": float(fit["epochs"]["value"]),
                "stopped_early": False,
                "initial_action_std": float(np.mean(fitted_std)),
                "validation_loss_origin": "post_fit_checkpoint_recomputed_during_artifact_finalization",
            }
        architecture = _architecture_record(model)

        # A fixed seeded batch proves changing exploration noise leaves the
        # deterministic actor output unchanged.
        rng = np.random.default_rng(seed)
        validation_observations = {
            "lidar": rng.uniform(0.0, 1.0, (16, 1081)).astype(np.float32),
            "camera": rng.integers(0, 256, (16, 90, 160, 3), dtype=np.uint8),
            "state": rng.uniform(-1.0, 1.0, (16, 9)).astype(np.float32),
        }
        deterministic_before = _deterministic_action(model, validation_observations)
        source_checkpoint = output / "behavior_cloned_policy.zip"
        output.mkdir(parents=True, exist_ok=False)
        if fitted_checkpoint is None:
            saved = save_behavior_cloned_checkpoint(model, output)
        else:
            saved = source_checkpoint
            shutil.copy2(fitted_checkpoint, saved)
        source_checkpoint_hash = sha256_file(saved)
        if saved != source_checkpoint:
            raise RuntimeError(f"unexpected rebuilt checkpoint path: {saved}")
        from stable_baselines3 import PPO
        reloaded_source = PPO.load(str(saved), env=environment, device=device)
        source_reload_std = reloaded_source.policy.log_std.detach().exp().cpu().numpy()
        source_reload_delta = float(np.max(np.abs(
            _deterministic_action(reloaded_source, validation_observations) - deterministic_before
        )))
        if not np.allclose(source_reload_std, [0.2, 0.2], atol=1e-7, rtol=1e-6) or source_reload_delta > 1e-7:
            raise RuntimeError(
                "saved BC source checkpoint failed standard-deviation or deterministic-action "
                f"reload verification: std={source_reload_std.tolist()}, delta={source_reload_delta}"
            )

        provenance = {
            "kind": "rebuilt_from_preserved_demonstrations",
            "source_run_id": OFFICIAL_BC_SOURCE["run_id"],
            "source_legacy_bytes_verified": True,
            "fitted_intermediate_checkpoint": None if fitted_checkpoint is None else {
                "path": checkpoint_path_aliases(fitted_checkpoint),
                "sha256": fitted_hash,
                "training_completed_before_reload_check_failed": True,
            },
            "checkpoint_paths": checkpoint_path_aliases(source_checkpoint),
            "parents": {
                name: {"path": checkpoint_path_aliases(path), "sha256": source["hashes"][name]}
                for name, path in source["paths"].items()
            },
            "historical_hash_resolution": LEGACY_EVALUATION,
            "legacy_parameter_sources": fit,
            "policy_architecture": architecture,
            "ppo_initialization": {
                "learning_rate": 1e-5,
                "n_steps": 2048,
                "n_epochs": 4,
                "gamma": 0.9995,
                "gae_lambda": 0.95,
                "target_kl": 0.01,
                "batch_size": 64,
                "entropy_coefficient": 0.0,
                "value_coefficient": 0.5,
                "max_gradient_norm": 0.5,
                "device": device,
            },
            "seed": seed,
            "dataset_headers": {
                key: {"shape": list(shape), "dtype": str(dtype)}
                for key, (shape, dtype) in headers.items()
            },
            "safety_checks": source["safety_checks"],
            "software": runtime,
            "source_checkpoint_embedded_software": {
                "runtime": "Linux WSL2",
                "versions": SOURCE_RUNTIME,
                "rebuild_executable": sys.executable,
                "versions_match": True,
                "byte_identical_runtime_claimed": False,
            },
            "implementation_sha256": {
                "behavior_cloning.py": sha256_file(ROOT / "src/layer3/behavior_cloning.py"),
                "train.py": sha256_file(ROOT / "src/layer3/train.py"),
                "extractors.py": sha256_file(ROOT / "src/layer3/extractors.py"),
                "rebuild_bc_from_demonstrations.py": sha256_file(Path(__file__).resolve()),
                "bc_artifact.py": sha256_file(ROOT / "src/layer3/bc_artifact.py"),
            },
        }
        result.update({
            "status": "ready",
            "usable": True,
            "readiness": {
                "ready": True,
                "reason": None,
                "source": "pinned_verified_clean_official_teacher_report",
                "successful_episodes": int(teacher_report["validation"]["successful_episodes"]),
                "completed_episodes": int(teacher_report["validation"]["completed_episodes"]),
                "failed_episodes": int(source_summary.get("readiness", {}).get("failed_episodes", 0)),
                "collision_count": int(teacher_report["validation"]["collision_count"]),
                "stagnation_detected": bool(teacher_report["validation"]["stagnation_detected"]),
                "actual_official_laps": int(teacher_report["validation"]["actual_official_laps"]),
            },
            "checkpoint": str(saved),
            "checkpoint_sha256": source_checkpoint_hash,
            "provenance": provenance,
            "training_parameters": fit,
            "policy_architecture": architecture,
        })
        # BC checkpoints need the official observation and action mapping when
        # Mission Control discovers them for a simulator diagnostic. Carry the
        # pinned source semantics forward while replacing its stale run-local
        # checkpoint pointer with this rebuilt artifact.
        artifact_config = json.loads(json.dumps(source_config))
        artifact_config["behavior_cloning"] = {
            "enabled": True,
            "only": True,
            "teacher": "verified official centerline demonstrations; offline rebuild",
            "checkpoint": source_checkpoint.name,
            "status": "ready",
            "usable": True,
            "readiness": result["readiness"],
            "checkpoint_sha256": source_checkpoint_hash,
            "training_parameters": fit,
            "policy_architecture": architecture,
            "privileged_teacher_state_saved": False,
        }
        artifact_config["rebuild_source"] = {
            "run_id": OFFICIAL_BC_SOURCE["run_id"],
            "source_config_sha256": source["hashes"]["source_config"],
            "dataset_sha256": source["hashes"]["dataset"],
        }
        artifact_config_path = output / "config.json"
        artifact_config_path.write_text(json.dumps(artifact_config, indent=2), encoding="utf-8")
        artifact_config_sha = sha256_file(artifact_config_path)
        provenance["artifact_config"] = {
            "path": checkpoint_path_aliases(artifact_config_path),
            "sha256": artifact_config_sha,
        }
        result["artifact_config_sha256"] = artifact_config_sha
        (output / "behavior_cloning_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        # Build a separate PPO warm-start candidate. Its policy tensors must
        # differ only in log_std and its deterministic outputs must be stable.
        import torch
        with torch.no_grad():
            source_state = {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}
            model.policy.log_std.copy_(torch.log(torch.tensor([0.01, 0.05], device=model.policy.log_std.device)))
        deterministic_after = _deterministic_action(model, validation_observations)
        delta = float(np.max(np.abs(deterministic_after - deterministic_before)))
        candidate_state = model.policy.state_dict()
        changed_keys = [
            key for key, value in candidate_state.items()
            if not torch.equal(source_state[key], value.detach().cpu())
        ]
        if changed_keys != ["log_std"] or delta > 1e-7:
            raise RuntimeError(f"exploration-only candidate changed actor behavior: keys={changed_keys}, deterministic_delta={delta}")
        candidate_path = output / "ppo_warmstart_std_001_005.zip"
        model.save(str(candidate_path.with_suffix("")))
        candidate_hash = sha256_file(candidate_path)
        provenance_record = {
            "kind": "ppo_warmstart_action_std_derivative",
            "checkpoint_paths": checkpoint_path_aliases(candidate_path),
            "checkpoint_sha256": candidate_hash,
            "parent_checkpoint": {
                "path": checkpoint_path_aliases(saved),
                "sha256": source_checkpoint_hash,
            },
            "parent_dataset": {
                "path": checkpoint_path_aliases(dataset_path),
                "sha256": source["hashes"]["dataset"],
            },
            "action_std": [0.01, 0.05],
            "source_action_std": [0.2, 0.2],
            "changed_policy_keys": changed_keys,
            "deterministic_action_max_abs_delta": delta,
            "deterministic_observation_seed": seed,
            "deterministic_observation_count": 16,
            "parent_is_verified_rebuild": True,
        }
        (output / "ppo_warmstart_provenance.json").write_text(json.dumps(provenance_record, indent=2), encoding="utf-8")
        reloaded_candidate = PPO.load(str(candidate_path), env=environment, device=device)
        candidate_std = reloaded_candidate.policy.log_std.detach().exp().cpu().numpy()
        reloaded_delta = float(np.max(np.abs(
            _deterministic_action(reloaded_candidate, validation_observations) - deterministic_before
        )))
        reloaded_state = reloaded_candidate.policy.state_dict()
        reload_changed_keys = [
            key for key, value in reloaded_state.items()
            if not torch.equal(source_state[key], value.detach().cpu())
        ]
        if reload_changed_keys != ["log_std"] or reloaded_delta > 1e-7 or not np.allclose(candidate_std, [0.01, 0.05], atol=1e-7, rtol=1e-6):
            raise RuntimeError("saved PPO warm-start candidate failed reload/tensor/action checks")

        readiness = {
            "schema_version": 1,
            "status": "rebuilt_bc_ready_student_evaluation_pending",
            "created_for": "Offline BC rebuild from a hash-pinned clean official teacher archive; simulator evaluation remains required.",
            "source": {
                "run_id": OFFICIAL_BC_SOURCE["run_id"],
                "checkpoint": checkpoint_path_aliases(source["paths"]["source_checkpoint"]),
                "checkpoint_sha256": source["hashes"]["source_checkpoint"],
                "dataset": checkpoint_path_aliases(dataset_path),
                "dataset_sha256": source["hashes"]["dataset"],
                "teacher_validation_report": checkpoint_path_aliases(source["paths"]["teacher_report"]),
                "teacher_validation_sha256": source["hashes"]["teacher_report"],
                "teacher_facts": source["safety_checks"],
            },
            "artifact_config": {
                "path": checkpoint_path_aliases(artifact_config_path),
                "sha256": artifact_config_sha,
            },
            "rebuilt_candidate": {
                "checkpoint": checkpoint_path_aliases(saved),
                "checkpoint_sha256": source_checkpoint_hash,
                "training_parameters": fit,
                "initial_action_std": [0.2, 0.2],
                "source_reload_deterministic_action_max_abs_delta": source_reload_delta,
                "policy_architecture": architecture,
                "initial_validation_loss": result["initial_validation_loss"],
                "final_validation_loss": result["final_validation_loss"],
            },
            "ppo_warmstart_candidate": provenance_record,
            "historical_hash_resolution": LEGACY_EVALUATION,
            "required_result_before_ppo": "Review a fresh official BC student drive evaluation and approve a separate PPO smoke run.",
            "limitations": [
                "The legacy batch size and learning rate were absent from the source records; this rebuild records them as plan-supplied settings.",
                "The matching WSL package versions were used; byte-for-byte identity of the historical runtime is not claimed.",
                "A clean privileged teacher episode and offline imitation loss do not establish student driving safety.",
            ],
        }
        (output / "bc_readiness_manifest.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps({
            "checkpoint": str(saved),
            "checkpoint_sha256": source_checkpoint_hash,
            "ppo_warmstart_candidate": str(candidate_path),
            "ppo_warmstart_candidate_sha256": candidate_hash,
            "dataset_sha256": source["hashes"]["dataset"],
            "transitions": count,
            "training_parameters": fit,
            "initial_validation_loss": result["initial_validation_loss"],
            "final_validation_loss": result["final_validation_loss"],
            "initial_action_std": result["initial_action_std"],
            "candidate_action_std": [0.01, 0.05],
            "changed_policy_keys": changed_keys,
            "deterministic_action_max_abs_delta": delta,
            "reloaded_candidate_deterministic_action_max_abs_delta": reloaded_delta,
        }, indent=2))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
