"""Offline reconstruction evidence for a declared official PPO resume.

This script does not create or step a simulator.  It authenticates the exact
parent archive, invokes the same clean-optimizer PPO resume helper used by the
trainer, initializes the BC-campaign exploration schedule and training
callbacks, then samples the policy on a deterministic subset of the recorded
demonstrations.  The report explicitly labels this as reconstruction evidence;
it is not a recording of a past live process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.layer2.spaces import make_action_space, make_observation_space
from src.layer3.bc_artifact import verify_ppo_lineage_checkpoint
from src.layer3.official_policy import load_ppo_checkpoint, resume_ppo_checkpoint
from src.layer3.official_train import (
    _make_training_exploration_schedule,
    _resolve_exploration_settings,
    _training_context_metadata,
)
from src.layer3.ppo_audit import CooperativePauseCallback, OperatorStopCallback, PostOptimizerAuditCallback
from stable_baselines3.common.utils import obs_as_tensor


class _OfflineExplorationCallback(BaseCallback):
    """Match the trainer's automatic-evaluation exploration lifecycle."""

    def __init__(self, run_dir: Path, resolved: dict[str, Any]) -> None:
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.resolved = resolved
        self.exploration = None

    def _on_training_start(self) -> None:
        self.exploration = _make_training_exploration_schedule(
            self.model,
            self.run_dir,
            initial_std=self.resolved["initial_std"],
            std_min=self.resolved["std_min"],
            std_max=self.resolved["std_max"],
        )

    def _on_rollout_start(self) -> None:
        if self.exploration is not None:
            self.exploration.on_rollout_start()

    def _on_step(self) -> bool:
        return True


class _RecordedObservationEnv(gym.Env):
    """Space-only env; its methods never communicate with a simulator."""

    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        self.observation_space = make_observation_space(include_camera=True)
        self.action_space = make_action_space()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        observation = {
            "camera": np.zeros((90, 160, 3), dtype=np.uint8),
            "lidar": np.ones((1081,), dtype=np.float32),
            "state": np.zeros((9,), dtype=np.float32),
        }
        return observation, {}

    def step(self, action):
        del action
        raise RuntimeError("offline reconstruction must never step an environment")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(flat)),
        "p01": float(np.percentile(flat, 1)),
        "median": float(np.median(flat)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
    }


def run_evidence(
    *, parent: Path, config_path: Path, dataset: Path, output: Path,
    expected_parent_sha256: str, expected_dataset_sha256: str,
) -> dict[str, Any]:
    for path in (parent, config_path, dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent_sha = _sha256(parent)
    dataset_sha = _sha256(dataset)
    config_sha = _sha256(config_path)
    if parent_sha != expected_parent_sha256:
        raise ValueError("parent checkpoint hash does not match the declared source")
    if dataset_sha != expected_dataset_sha256:
        raise ValueError("demonstration archive hash does not match the declared source")

    lineage = verify_ppo_lineage_checkpoint(parent)
    if not lineage or lineage.get("checkpoint_sha256") != parent_sha:
        raise ValueError("parent PPO lineage did not authenticate")
    parent_record = lineage.get("lineage", {}).get("parent_checkpoint") or {}
    branch_config = json.loads(config_path.read_text(encoding="utf-8"))
    ppo = branch_config["ppo"]
    explore = branch_config["ppo_exploration"]
    if branch_config.get("resume") is None:
        raise ValueError("branch config must record a resume source")
    if Path(branch_config["resume"]).name != parent.name:
        raise ValueError("branch config resume path does not name the declared parent")
    if explore.get("resume_checkpoint_sha256") != parent_sha:
        raise ValueError("branch exploration metadata does not bind the parent hash")
    if int(ppo["requested"]["n_steps"]) != 2048:
        raise ValueError("unexpected rollout size")
    expected_effective = {
        "learning_rate": 1e-5, "n_steps": 2048, "n_epochs": 4,
        "gamma": 0.9995, "gae_lambda": 0.95, "target_kl": 0.01,
        "batch_size": 64, "clip_range": 0.1, "ent_coef": 0.0,
        "vf_coef": 0.5, "max_grad_norm": 0.5,
    }
    for key, expected in expected_effective.items():
        if not math.isclose(float(ppo["effective"][key]), expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"unexpected effective PPO setting {key}")

    arrays = np.load(dataset, allow_pickle=False)
    required = {"camera", "lidar", "state", "actions"}
    if set(arrays.files) != required:
        raise ValueError(f"unexpected demonstration keys: {arrays.files}")
    total = int(arrays["actions"].shape[0])
    indices = np.linspace(0, total - 1, num=256, dtype=np.int64)
    observations = {
        "camera": np.transpose(
            np.asarray(arrays["camera"][indices], dtype=np.uint8), (0, 3, 1, 2)
        ),
        "lidar": np.asarray(arrays["lidar"][indices], dtype=np.float32),
        "state": np.asarray(arrays["state"][indices], dtype=np.float32),
    }
    if observations["camera"].shape != (256, 3, 90, 160):
        raise ValueError("authentic camera observations have an unexpected shape")
    if observations["lidar"].shape != (256, 1081) or observations["state"].shape != (256, 9):
        raise ValueError("authentic sensor/state observations have an unexpected shape")
    if not all(np.isfinite(value).all() for key, value in observations.items() if key != "camera"):
        raise ValueError("selected authentic numeric observations are nonfinite")
    rows_digest = hashlib.sha256()
    for key in ("camera", "lidar", "state"):
        rows_digest.update(key.encode("ascii") + b"\0")
        rows_digest.update(observations[key].tobytes(order="C"))

    seed = 20261013
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    env = VecTransposeImage(DummyVecEnv([_RecordedObservationEnv]))
    try:
        model = resume_ppo_checkpoint(
            parent,
            env=env,
            observation_profile="official_sensors_camera",
            learning_rate=expected_effective["learning_rate"],
            n_steps=expected_effective["n_steps"],
            n_epochs=expected_effective["n_epochs"],
            gamma=expected_effective["gamma"],
            gae_lambda=expected_effective["gae_lambda"],
            target_kl=expected_effective["target_kl"],
            seed=seed,
            device="cpu",
            batch_size=expected_effective["batch_size"],
            clip_range=expected_effective["clip_range"],
            ent_coef=expected_effective["ent_coef"],
            vf_coef=expected_effective["vf_coef"],
            max_grad_norm=expected_effective["max_grad_norm"],
        )
        if int(model.num_timesteps) != int(lineage["lineage"]["num_timesteps"]):
            raise ValueError("resumed model timestep differs from authenticated parent")
        pre_profile_state = {
            key: value.detach().clone()
            for key, value in model.policy.state_dict().items()
        }
        # Independently load the serialized policy through the production
        # inference loader, then compare it to the fresh-optimizer resume.
        parent_model = load_ppo_checkpoint(
            parent, device="cpu", observation_profile="official_sensors_camera", env=env,
            learning_rate=expected_effective["learning_rate"],
            clip_range=expected_effective["clip_range"],
        )
        source_state = parent_model.policy.state_dict()
        if set(source_state) != set(pre_profile_state):
            raise ValueError("resume initialization changed the policy tensor key set")
        source_tensor_match = all(
            torch.equal(source_state[key].detach().cpu(), pre_profile_state[key].detach().cpu())
            for key in source_state
        )
        if not source_tensor_match:
            raise ValueError("fresh PPO initialization did not copy exact parent policy tensors")
        parent_actor_sha = _tensor_fingerprint(source_state)
        pre_profile_actor_sha = _tensor_fingerprint(pre_profile_state)
        parent_std = [float(value) for value in model.policy.log_std.detach().exp().cpu()]
        resolved = _resolve_exploration_settings(
            requested_profile="bc_campaign",
            overrides={
                "initial_throttle": None, "initial_steering": None,
                "min_throttle": None, "min_steering": None,
                "max_throttle": None, "max_steering": None,
            },
            saved=None,
            verified_bc_resume=True,
            current_std=tuple(parent_std),
        )
        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.as_tensor(
                resolved["initial_std"], dtype=model.policy.log_std.dtype,
                device=model.policy.log_std.device,
            )))
        with tempfile.TemporaryDirectory(prefix="aicar-ppo-init-") as temporary_dir:
            callback_dir = Path(temporary_dir)
            audit = PostOptimizerAuditCallback(
                callback_dir,
                save_every_transitions=int(branch_config["optimizer_audit"]["save_every_transitions"]),
                global_step_cadence=True,
                lineage={"parent_lineage": lineage},
            )
            stop = OperatorStopCallback(Event())
            exploration_callback = _OfflineExplorationCallback(callback_dir, resolved)
            pause = CooperativePauseCallback(
                callback_dir, "offline-reconstruction", Event(), audit,
                resume_request_wait_timeout_s=9000,
                resume_freshness_timeout_s=120,
            )
            callback_list = CallbackList([audit, stop, exploration_callback, pause])
            callback_list.init_callback(model)
            callback_list.on_training_start({}, {})
            callback_list.on_rollout_start()
            callback_classes = [type(item).__name__ for item in callback_list.callbacks]
        initialized_state = model.policy.state_dict()
        non_log_std_match = all(
            torch.equal(source_state[key].detach().cpu(), initialized_state[key].detach().cpu())
            for key in source_state if key != "log_std"
        )
        if not non_log_std_match:
            raise ValueError("exploration initialization changed non-log_std actor/policy tensors")

        policy_obs = obs_as_tensor(observations, model.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(policy_obs).distribution
            raw = dist.sample((64,)).detach().cpu().numpy()
        clipped = np.clip(raw, -1.0, 1.0)
        throttle_raw = raw[..., 0]
        steering_raw = raw[..., 1]
        measured_std = [float(v) for v in model.policy.log_std.detach().exp().cpu()]
        stats = {
            "draw_count": int(raw.shape[0] * raw.shape[1]),
            "reverse_throttle_fraction_raw": float(np.mean(throttle_raw < 0.0)),
            "raw_throttle_outside_action_bounds_fraction": float(np.mean(np.abs(throttle_raw) > 1.0)),
            "raw_throttle_magnitude_over_0_5_fraction": float(np.mean(np.abs(throttle_raw) > 0.5)),
            "clipped_action_fraction_any_dimension": float(np.mean(np.any(np.abs(raw) > 1.0, axis=-1))),
            "steering_clipping_fraction": float(np.mean(np.abs(steering_raw) > 1.0)),
            "raw_throttle": _percentiles(throttle_raw),
            "applied_clipped_throttle": _percentiles(clipped[..., 0]),
            "raw_steering": _percentiles(steering_raw),
            "applied_clipped_steering": _percentiles(clipped[..., 1]),
        }
        if not np.isfinite(raw).all():
            raise ValueError("stochastic action samples are nonfinite")
        report = {
            "schema_version": 1,
            "evidence_kind": "offline_initialization_reconstruction",
            "live_runtime_capture": False,
            "simulator_started_or_stepped": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parent_checkpoint": {"path": str(parent.resolve()), "sha256": parent_sha},
            "parent_lineage_direct_parent_sha256": (
                parent_record.get("sha256") if isinstance(parent_record, dict) else None
            ),
            "authenticated_bc_ancestor_sha256": lineage["lineage"]["bc_ancestor_checkpoint"]["sha256"],
            "branch_config": {"path": str(config_path.resolve()), "sha256": config_sha},
            "dataset": {
                "path": str(dataset.resolve()), "sha256": dataset_sha,
                "transitions": total, "keys": sorted(arrays.files),
                "observation_sampling": "256 deterministic evenly spaced row indices from 0 through N-1",
                "selected_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "selected_observation_bytes_sha256": rows_digest.hexdigest(),
            },
            "initialization": {
                "code_and_package_context": _training_context_metadata(),
                "device": "cpu (deliberate offline reconstruction; avoids competing with live CUDA learner)",
                "seed": seed,
                "fresh_optimizer_state_entries": len(model.policy.optimizer.state),
                "optimizer_step_count_before_learning": 0,
                "parent_policy_tensor_sha256": parent_actor_sha,
                "resume_policy_tensor_sha256_before_exploration_profile": pre_profile_actor_sha,
                "initialized_policy_tensor_sha256": _tensor_fingerprint(model.policy.state_dict()),
                "parent_std_before_profile": parent_std,
                "requested_profile": "bc_campaign",
                "resolved_profile": resolved,
                "post_schedule_std": measured_std,
                "model_num_timesteps_inherited": int(model.num_timesteps),
                "n_envs": int(model.n_envs), "n_steps": int(model.n_steps),
                "n_epochs": int(model.n_epochs), "batch_size": int(model.batch_size),
                "learning_rate": float(model.lr_schedule(1.0)),
                "target_kl": model.target_kl,
                "clip_range": float(model.clip_range(1.0)),
                "ent_coef": float(model.ent_coef), "vf_coef": float(model.vf_coef),
                "max_grad_norm": float(model.max_grad_norm),
                "normalize_images": bool(model.policy.normalize_images),
                "callback_classes_initialized": callback_classes,
                "callback_lifecycle_invoked_in_order": ["on_training_start", "on_rollout_start"],
                "callback_lifecycle_env_steps": 0,
                "cooperative_pause_wait_seconds": pause.resume_request_wait_timeout_s,
                "cooperative_freshness_wait_seconds": pause.resume_freshness_timeout_s,
                "resume_policy_exactly_matches_parent_before_exploration_profile": source_tensor_match,
                "all_non_log_std_tensors_equal_parent_after_profile_clamp": non_log_std_match,
                "log_std_changed_by_requested_initial_profile": not torch.equal(
                    source_state["log_std"].detach().cpu(), initialized_state["log_std"].detach().cpu()
                ),
                "optimizer_updates_performed": 0,
            },
            "stochastic_action_distribution": stats,
            "warning": "Reconstructed offline using the exact saved parent weights/config and authentic observations. It does not prove actions/statistics from the historical live startup process.",
        }
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded + b"\n")
        report["report_sha256"] = hashlib.sha256(encoded + b"\n").hexdigest()
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    args = parser.parse_args()
    report = run_evidence(
        parent=args.parent, config_path=args.config, dataset=args.dataset,
        output=args.output, expected_parent_sha256=args.parent_sha256,
        expected_dataset_sha256=args.dataset_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
