"""Read-only stochastic action preflight for the authenticated BC smoke warmstart.

This constructs a fresh camera PPO learner and initializes the same bounded
exploration schedule and post-update audit callback used by official training.
It samples the policy on preserved teacher observations without calling learn()
or connecting to a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(flattened.size),
        "mean": float(flattened.mean()),
        "std": float(flattened.std()),
        "min": float(flattened.min()),
        "p01": float(np.quantile(flattened, 0.01)),
        "p05": float(np.quantile(flattened, 0.05)),
        "p50": float(np.quantile(flattened, 0.50)),
        "p95": float(np.quantile(flattened, 0.95)),
        "p99": float(np.quantile(flattened, 0.99)),
        "max": float(flattened.max()),
    }


def build_report(
    *, checkpoint: Path, dataset_path: Path, output_dir: Path,
    sample_count: int = 256, draws_per_observation: int = 64, seed: int = 20261012,
) -> dict[str, Any]:
    from scripts.initialize_camera_champion import _ObservationEnv
    from src.layer3.bc_artifact import verify_ppo_warmstart
    from src.layer3.official_exploration import OfficialExplorationSchedule
    from src.layer3.official_policy import resume_ppo_checkpoint
    from src.layer3.ppo_audit import PostOptimizerAuditCallback
    from src.layer3.official_train import _official_ppo_config, _verify_official_ppo_config

    checkpoint = Path(checkpoint).resolve()
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse preflight output directory: {output_dir}")
    if not checkpoint.is_file() or not dataset_path.is_file():
        raise FileNotFoundError("warmstart checkpoint and preserved dataset must both exist")
    verified = verify_ppo_warmstart(checkpoint)
    if not verified:
        raise RuntimeError("warmstart checkpoint failed strict BC/dataset provenance verification")
    if int(sample_count) < 1 or int(draws_per_observation) < 2:
        raise ValueError("sample_count must be positive and draws_per_observation must be >= 2")

    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {"camera", "lidar", "state", "actions"}
        if set(archive.files) != required:
            raise ValueError(f"demonstration keys differ from expected {sorted(required)}")
        count = len(archive["actions"])
        if count < sample_count:
            raise ValueError(f"dataset has {count} rows, fewer than requested {sample_count}")
        indices = np.linspace(0, count - 1, num=sample_count, dtype=np.int64)
        observations = {
            "camera": archive["camera"][indices].copy(),
            "lidar": archive["lidar"][indices].copy(),
            "state": archive["state"][indices].copy(),
        }

    # This is the prescribed smoke learner setup. resume_ppo_checkpoint builds
    # a fresh optimizer and transfers only policy tensors and global timesteps.
    smoke = {
        "n_envs": 1,
        "n_steps": 2048,
        "batch_size": 64,
        "learning_rate": 1e-5,
        "n_epochs": 4,
        "gamma": 0.9995,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "exploration_profile": "bc_smoke",
        "action_std_by_action": [0.01, 0.05],
        "total_transition_cap": 8192,
        "wall_time_cap_seconds": 600,
        "checkpoint_after_optimizer_update_every_transitions": 2048,
    }
    # Match unchanged official reward defaults; this report does not start them.
    reward_defaults = {
        "time_cost_per_simulated_second": 0.0,
        "collision_penalty_magnitude": 200.0,
        "collision_reward_percent": 0.0,
        "lap_completion_reward": 0.0,
        "failed_episode_penalty": 200.0,
        "backward_speed_penalty_scale": 10.0,
        "episode_failure_reward_percent": 0.0,
        "frontier_stagnation_s": 5.0,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    env = DummyVecEnv([_ObservationEnv])
    try:
        model = resume_ppo_checkpoint(
            checkpoint,
            env=env,
            observation_profile="official_sensors_camera",
            learning_rate=smoke["learning_rate"],
            n_steps=smoke["n_steps"],
            n_epochs=smoke["n_epochs"],
            gamma=smoke["gamma"],
            gae_lambda=smoke["gae_lambda"],
            target_kl=smoke["target_kl"],
            seed=0,
            device="cpu",
            tensorboard_log=None,
            batch_size=smoke["batch_size"],
            clip_range=smoke["clip_range"],
            ent_coef=smoke["ent_coef"],
            vf_coef=smoke["vf_coef"],
            max_grad_norm=smoke["max_grad_norm"],
        )
        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.tensor([0.01, 0.05])))
        exploration = OfficialExplorationSchedule(
            model, output_dir, std_min=(0.01, 0.05), std_max=(0.01, 0.05)
        )
        audit = PostOptimizerAuditCallback(output_dir, save_every_transitions=2048)
        audit.init_callback(model)
        audit.on_training_start({}, {})
        exploration.on_rollout_start()

        effective_std = model.policy.log_std.detach().exp().cpu().numpy()
        if not np.allclose(effective_std, [0.01, 0.05], atol=1e-7, rtol=1e-6):
            raise RuntimeError(f"initialized exploration std differs from smoke preset: {effective_std}")
        observations_tensor, _ = model.policy.obs_to_tensor(observations)
        with torch.no_grad():
            distribution = model.policy.get_distribution(observations_tensor).distribution
            means = distribution.mean
            stds = distribution.stddev
            generator = torch.Generator(device=means.device).manual_seed(int(seed))
            noise = torch.randn(
                (sample_count, draws_per_observation, 2),
                dtype=means.dtype,
                device=means.device,
                generator=generator,
            )
            raw = means[:, None, :] + noise * stds[:, None, :]
        raw = raw.cpu().numpy()
        # Canonical competition action mapping from the verified BC config.
        source_config = json.loads((checkpoint.parent / "config.json").read_text(encoding="utf-8"))
        action_config = source_config["actions"]
        from src.layer2.spaces import transform_policy_action
        ppo_requested = {
            "n_steps": 2048,
            "learning_rate": 1e-5,
            "n_epochs": 4,
            "gamma": 0.9995,
            "gae_lambda": 0.95,
            "target_kl": 0.01,
            "batch_size": 64,
            "clip_range": 0.1,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
        effective_ppo = _verify_official_ppo_config(
            model, ppo_requested, observation_profile="official_sensors_camera"
        )
        optimizer_state_entries = len(model.policy.optimizer.state_dict()["state"])
        if optimizer_state_entries != 0:
            raise RuntimeError(
                "fresh smoke PPO unexpectedly contains optimizer moments before learn()"
            )

        def action_metrics(raw_values: np.ndarray) -> dict[str, Any]:
            clipped_values = np.clip(raw_values, -1.0, 1.0)
            mapped_values = np.empty_like(clipped_values.reshape(-1, 2))
            for i, action in enumerate(clipped_values.reshape(-1, 2)):
                mapped_values[i] = transform_policy_action(
                    action,
                    throttle_mode=action_config["throttle_mode"],
                    negative_throttle_mode=action_config["negative_throttle_mode"],
                    steering_mode=action_config["steering_mode"],
                    steering_action_scale=float(action_config["steering_action_scale"]),
                    straight_throttle_gain=float(action_config["straight_throttle_gain"]),
                    straight_throttle_steering_threshold=float(
                        action_config["straight_throttle_steering_threshold"]
                    ),
                )
            mapped_values = mapped_values.reshape(raw_values.shape)
            return {
                "reverse_throttle_fraction_raw": float(np.mean(raw_values[..., 0] < 0.0)),
                "reverse_throttle_fraction_after_action_clip": float(
                    np.mean(clipped_values[..., 0] < 0.0)
                ),
                "positive_over_throttle_fraction_raw": float(np.mean(raw_values[..., 0] > 1.0)),
                "negative_throttle_overflow_fraction_raw": float(np.mean(raw_values[..., 0] < -1.0)),
                "throttle_overflow_fraction_raw_abs_gt_1": float(
                    np.mean(np.abs(raw_values[..., 0]) > 1.0)
                ),
                "steering_overflow_fraction_raw_abs_gt_1": float(
                    np.mean(np.abs(raw_values[..., 1]) > 1.0)
                ),
                "raw_throttle": _distribution_summary(raw_values[..., 0]),
                "clipped_throttle": _distribution_summary(clipped_values[..., 0]),
                "mapped_throttle": _distribution_summary(mapped_values[..., 0]),
                "raw_steering": _distribution_summary(raw_values[..., 1]),
                "clipped_steering": _distribution_summary(clipped_values[..., 1]),
                "mapped_steering": _distribution_summary(mapped_values[..., 1]),
            }
        smoke_effective = {
            **smoke,
            "action_std_after_schedule_init": [float(value) for value in effective_std],
            "resume_checkpoint_sha256": _sha256(checkpoint),
            "resume_optimizer_state_reused": False,
            "resume_optimizer_state_entry_count": optimizer_state_entries,
            "global_timesteps_transferred": int(model.num_timesteps),
            "resume_architecture": type(model.policy.features_extractor).__name__,
            "initialized_callbacks": [
                "OfficialExplorationSchedule.on_rollout_start",
                "PostOptimizerAuditCallback.on_training_start",
            ],
        }
        import gymnasium
        import stable_baselines3

        return {
            "schema_version": 1,
            "status": "offline_stochastic_preflight_only",
            "created_for": "Inspect smoke-profile action distribution on preserved demonstration observations; no learning or simulator access.",
            "candidate_gate": "awaiting deterministic official BC student driving evaluation; this report is not a driving-safety approval",
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint), "verification": verified},
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
                "transitions": int(count),
                "sample_count": int(sample_count),
                "sample_selection": "evenly spaced row indices from first through final transition",
                "sample_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "draws_per_observation": int(draws_per_observation),
                "seed": int(seed),
            },
            "smoke_effective_config": smoke_effective,
            "effective_ppo_after_fresh_resume": effective_ppo,
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "gymnasium": gymnasium.__version__,
                "device": "cpu",
            },
            "reward_defaults_unchanged": reward_defaults,
            "action_mapping": action_config,
            "stochastic_action_diagnostics": {
                **action_metrics(raw),
                "sampled_values_finite": bool(np.isfinite(raw).all()),
                "model_forward_mean_finite": bool(torch.isfinite(means).all().item()),
                "dataset_sections": {
                    "first_third": action_metrics(raw[indices < count / 3]),
                    "middle_third": action_metrics(raw[(indices >= count / 3) & (indices < 2 * count / 3)]),
                    "final_third": action_metrics(raw[indices >= 2 * count / 3]),
                },
            },
            "post_update_save_semantics": {
                "checkpoint_every_transitions": 2048,
                "save_callback": "PostOptimizerAuditCallback saves at optimizer-update boundaries and preserves prior post-update saves when collection stops",
                "maximum_configured_transitions": 8192,
                "maximum_configured_wall_seconds": 600,
                "stop_boundary_note": "Operator, plateau, and trainer wall-time stops return false at the next environment-step callback. An interrupted partial rollout is not optimized; the last post-update checkpoint remains intact and official_policy_latest is saved from the current policy. A separate supervisor owns any hard external wall-clock deadline.",
            },
            "resume_optimizer_semantics": "resume_ppo_checkpoint constructs a fresh PPO optimizer from the smoke settings, loads only policy tensors, and transfers checkpoint num_timesteps; optimizer moments, schedules, and rollout buffer are not resumed.",
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--draws-per-observation", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    report = build_report(
        checkpoint=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sample_count=args.samples,
        draws_per_observation=args.draws_per_observation,
        seed=args.seed,
    )
    report_path = args.output_dir / "stochastic_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    ledger = {
        "schema_version": 1,
        "status": "student_driving_gate_pending",
        "candidate_checkpoint_sha256": report["checkpoint"]["sha256"],
        "source_demonstration_dataset_sha256": report["dataset"]["sha256"],
        "evidence": {
            "teacher_source": "PASS: provenance pins source checkpoint, source config, demonstration dataset and clean official teacher validation report; route 45.80209158004757 m, scene SHA a5d11e...e014b674.",
            "offline_rebuild": "PASS: eight epochs completed; 48,862 transitions; validation loss recomputed as 0.00011903037211595802; batch size 256 and lr 0.001 recorded plan-supplied because legacy source omitted them; initial loss unavailable.",
            "warmstart_authentication": "PASS: exact log_std-only derivative; [.01,.05], deterministic action parity delta 0, strict parent BC/dataset hashes verified.",
            "post_initialization_action_probe": report["stochastic_action_diagnostics"],
            "bc_student_official_diagnostic": "NOT RUN: candidate still requires the deterministic official 1 warmup + 10 scored lap drive gate.",
            "ppo_smoke": "NOT STARTED: gated on BC student driving pass.",
            "ppo_campaign": "NOT STARTED.",
        },
        "preserved_failed_or_superseded_artifacts": [
            {
                "path": "logs/rl/bc_rebuild_seed0_20261012",
                "status": "preserved completed fit; packaging reload comparison failed only under an over-tight CUDA-versus-CPU tolerance; same checkpoint bytes were carried into the authenticated bundle and same-device checks passed",
                "checkpoint_sha256": report["checkpoint"]["sha256"],
            },
            {
                "path": "logs/rl/bc_rebuild_seed0_20261012_finalized",
                "status": "superseded incomplete packaging; missing config and nested parent path aliases made independent verification fail; do not use",
            },
            {
                "historical_checkpoint_sha256": "b395bc91c3d7b42b7ada87f11916af82ced6093fcf51d9a22913c3eba812ea7c",
                "historical_run_id": "da2a678ec87c",
                "status": "separate 4,000-transition BC artifact with collision failures; not a parent of the rebuilt candidate",
            },
            {
                "path": "logs/rl/bc_smoke_preflight_20261012",
                "status": "preflight reporter attempt failed on an output-array indexing error before saving any report; no learning or simulator access; preserved as failed diagnostic attempt",
            },
        ],
        "report_path": str(report_path.resolve()),
        "limitation": "Offline action distributions and teacher success do not certify student driving safety; no PPO result is claimed.",
    }
    ledger_path = args.output_dir / "readiness_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, allow_nan=False), encoding="utf-8")
    print(report_path)
    print(json.dumps(report["stochastic_action_diagnostics"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
