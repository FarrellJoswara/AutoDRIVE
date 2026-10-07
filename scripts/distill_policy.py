"""Initialize a sensor-only PPO actor from a simulator-state PPO teacher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer3.behavior_cloning import (
    behavior_clone_actor,
    collect_policy_demonstrations,
)
from src.layer3.train import _resolve_device, make_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000,
                        help="sensor/action collection steps per environment")
    parser.add_argument("--map-id", default="porto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.n_envs < 1 or args.steps < 1:
        parser.error("--n-envs and --steps must be positive")
    if not args.teacher.is_file():
        parser.error(f"teacher checkpoint does not exist: {args.teacher}")

    from stable_baselines3 import PPO
    from src.layer3.envs import make_vec_env

    args.out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    env = make_vec_env(
        args.n_envs,
        seed=args.seed,
        headless=True,
        auto_launch=False,
        observation_profile="official_sensors",
        map_id=args.map_id,
        frame_skip=1,
        action_interval_s=0.025,
        steering_action_scale=0.946,
        straight_throttle_gain=1.025,
        straight_throttle_steering_threshold=0.15,
        throttle_mode="bidirectional",
        laps_per_episode=0,
        frontier_stagnation_seconds=10.0,
        terminate_on_collision=True,
    )
    try:
        teacher = PPO.load(str(args.teacher), device=device)
        dataset_path = args.out / "sensor_demonstrations.npz"
        dataset, collection = collect_policy_demonstrations(
            env, teacher, steps=args.steps, output_path=dataset_path
        )
        student = make_model(
            env,
            device=device,
            seed=args.seed,
            tensorboard_log=None,
            policy_architecture="lidar_cnn",
        )
        cloning = behavior_clone_actor(
            student,
            dataset,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            learning_rate=args.bc_learning_rate,
            seed=args.seed,
        )
        model_path = args.out / "sensor_student"
        student.save(str(model_path))
        report = {
            "teacher_checkpoint": str(args.teacher.resolve()),
            "student_checkpoint": str(model_path.with_suffix(".zip")),
            "dataset": str(dataset_path),
            "map_id": args.map_id,
            "observation_profile": "official_sensors",
            "teacher_observation_profile": "simulator",
            "n_envs": args.n_envs,
            "steps_per_environment": args.steps,
            "collection": collection,
            "behavior_cloning": cloning,
            "runtime_policy_inputs": ["lidar", "state"],
            "privileged_teacher_inputs_used_during_collection": True,
            "privileged_inputs_saved_to_dataset": False,
            "privileged_inputs_used_at_policy_runtime": False,
        }
        (args.out / "distillation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
