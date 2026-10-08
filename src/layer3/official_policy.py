"""Competition policy runner using only permitted ROS 2 inputs and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np


class StraightLineDiagnostic:
    """Fixed throttle pulse for validating the official simulator control path."""

    def __init__(self, throttle: float = 0.35) -> None:
        if not 0.0 < throttle <= 1.0:
            raise ValueError("diagnostic throttle must be in (0, 1]")
        self.throttle = float(throttle)

    def predict(self, observation: Any, deterministic: bool = True):
        del observation, deterministic
        return np.asarray([self.throttle, 0.0], dtype=np.float32), None


def load_policy(
    model_path: Path, *, device: str = "cpu", controller: str = "ppo",
    observation_profile: str = "official_sensors_camera",
) -> Any:
    if controller == "lidar_gap":
        from src.layer3.lidar_gap_policy import load_gap_policy

        return load_gap_policy()
    if controller == "straight":
        return StraightLineDiagnostic()
    if controller != "ppo":
        raise ValueError(f"Unknown official controller: {controller}")
    return load_ppo_checkpoint(
        model_path,
        device=device,
        observation_profile=observation_profile,
    )


def load_ppo_checkpoint(
    model_path: Path,
    *,
    device: str = "cpu",
    observation_profile: str = "official_sensors_camera",
    env: Any = None,
    learning_rate: float = 1e-5,
    clip_range: float = 0.2,
) -> Any:
    """Load PPO weights across the custom/official NumPy and SB3 versions.

    Checkpoints may contain pickled spaces, rollout observations, and schedule
    classes from newer NumPy/SB3 versions than the official API image provides.
    Those cached objects are not needed to run the policy or resume PPO: bind
    current Layer 2 spaces and start the next rollout with fresh state instead.
    """
    from stable_baselines3 import PPO

    from src.layer2.spaces import make_action_space, make_observation_space

    checkpoint = Path(model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    if observation_profile not in (
        "official_sensors", "official_sensors_history", "official_sensors_camera"
    ):
        raise ValueError(f"unsupported official observation profile: {observation_profile}")

    observation_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)
    if observation_space is None:
        observation_space = make_observation_space(
            lidar_history_frames=4 if observation_profile == "official_sensors_history" else 1,
            include_camera=observation_profile == "official_sensors_camera",
        )
    if action_space is None:
        action_space = make_action_space()

    constant_lr = float(learning_rate)
    constant_clip = float(clip_range)
    custom_objects = {
        "observation_space": observation_space,
        "action_space": action_space,
        # The saved rollout is intentionally discarded when loading a new env.
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
        # Newer SB3 serializes schedule wrapper classes that do not exist in
        # the pinned official image. The trainer supplies the intended values.
        "clip_range": lambda _: constant_clip,
        "lr_schedule": lambda _: constant_lr,
    }
    return PPO.load(
        str(checkpoint), env=env, device=device, custom_objects=custom_objects
    )


def resume_ppo_checkpoint(
    model_path: Path,
    *,
    env: Any,
    observation_profile: str,
    learning_rate: float,
    n_steps: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    seed: int,
    device: str = "cpu",
    tensorboard_log: Optional[str] = None,
) -> Any:
    """Rebuild PPO from authoritative settings and copy only checkpoint weights.

    The checkpoint's serialized optimizer, schedules, and rollout buffer are
    intentionally ignored. Only policy parameters and the global timestep are
    transferred into a newly constructed PPO instance.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    from src.layer3.train import make_model

    checkpoint = Path(model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    if observation_profile not in (
        "official_sensors", "official_sensors_history", "official_sensors_camera"
    ):
        raise ValueError(f"unsupported official observation profile: {observation_profile}")

    architecture = {
        "official_sensors_history": "temporal_lidar_cnn",
        "official_sensors_camera": "lidar_camera_cnn",
    }.get(observation_profile, "lidar_cnn")
    constant_lr = float(learning_rate)
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "ep_info_buffer": None,
        "ep_success_buffer": None,
        # Replace legacy/newer serialized schedule objects while unpickling,
        # although schedules from the checkpoint are never used for training.
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: constant_lr,
    }
    data, saved_parameters, _ = load_from_zip_file(
        str(checkpoint), device="cpu", custom_objects=custom_objects
    )
    if not isinstance(data, dict) or not isinstance(saved_parameters, dict):
        raise RuntimeError(f"Checkpoint has an unsupported SB3 archive format: {checkpoint}")
    if "policy" not in saved_parameters:
        raise RuntimeError(f"Checkpoint does not contain PPO policy parameters: {checkpoint}")
    if "num_timesteps" not in data:
        raise RuntimeError(f"Checkpoint does not contain PPO timestep metadata: {checkpoint}")

    fresh_model = make_model(
        env,
        device=device,
        seed=int(seed),
        tensorboard_log=tensorboard_log,
        learning_rate=constant_lr,
        n_epochs=int(n_epochs),
        n_steps=int(n_steps),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        policy_architecture=architecture,
    )
    fresh_model.policy.load_state_dict(saved_parameters["policy"], strict=True)
    fresh_model.num_timesteps = int(data["num_timesteps"])
    return fresh_model


def run_policy(
    model_path: Path,
    *,
    device: str = "cpu",
    timeout_s: float = 5.0,
    negative_throttle_mode: str = "allow",
    steering_mode: str = "normal",
    throttle_mode: str = "bidirectional",
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
    controller: str = "ppo",
    observation_profile: str = "official_sensors_camera",
) -> None:
    """Drive continuously; no reset, odometry, race counters, or map data."""
    from src.layer1.ros2_racer import RacerRos2
    from src.layer2.official_race_env import OfficialObservationBuilder
    from src.layer2.spaces import transform_policy_action

    if observation_profile not in (
        "official_sensors", "official_sensors_history", "official_sensors_camera"
    ):
        raise ValueError("official runner supports official sensor observation profiles only")
    model = load_policy(
        model_path, device=device, controller=controller,
        observation_profile=observation_profile,
    )
    racer = RacerRos2(
        timeout_s=timeout_s,
        include_race_metrics=False,
        require_camera=observation_profile == "official_sensors_camera",
    )
    observation_builder = OfficialObservationBuilder(
        observation_profile=observation_profile
    )
    previous_throttle = 0.0
    previous_steering = 0.0
    try:
        snap = racer.wait_until_ready()
        observation = observation_builder.reset(snap)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            command = np.asarray(action, dtype=np.float32).reshape(2)
            previous_throttle, previous_steering = transform_policy_action(
                command,
                throttle_mode=throttle_mode,
                negative_throttle_mode=negative_throttle_mode,
                steering_mode=steering_mode,
                steering_action_scale=steering_action_scale,
                straight_throttle_gain=straight_throttle_gain,
                straight_throttle_steering_threshold=straight_throttle_steering_threshold,
            )
            snap = racer.step(previous_throttle, previous_steering)
            elapsed_s = float(racer.last_control_interval_s)
            if elapsed_s <= 0.0:
                elapsed_s = float(racer.last_step_duration_s)
            observation = observation_builder.observe(
                snap,
                previous_throttle,
                previous_steering,
                elapsed_s=elapsed_s,
            )
    except KeyboardInterrupt:
        pass
    finally:
        # Stopping the process leaves the simulator's current actuator command
        # untouched; the next competition run starts in a fresh simulator.
        racer.kill()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/policy.zip"))
    parser.add_argument("--controller", choices=("ppo", "lidar_gap", "straight"), default="ppo")
    parser.add_argument(
        "--observation-profile",
        choices=("official_sensors", "official_sensors_history", "official_sensors_camera"),
        default="official_sensors_camera",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--negative-throttle-mode",
        choices=("allow", "zero", "positive_magnitude"),
        default="allow",
        help="How to map negative policy throttle actions to the actuator",
    )
    parser.add_argument(
        "--steering-mode",
        choices=("normal", "invert"),
        default="normal",
        help="Map policy steering into the official actuator direction",
    )
    parser.add_argument(
        "--throttle-mode",
        choices=("bidirectional", "forward_only"),
        default="bidirectional",
        help="Map normalized policy throttle into signed or forward-only actuator range",
    )
    parser.add_argument("--steering-action-scale", type=float, default=1.0)
    parser.add_argument("--straight-throttle-gain", type=float, default=1.0)
    parser.add_argument(
        "--straight-throttle-steering-threshold", type=float, default=0.15
    )
    args = parser.parse_args(argv)
    run_policy(
        args.model,
        device=args.device,
        timeout_s=args.timeout_s,
        negative_throttle_mode=args.negative_throttle_mode,
        steering_mode=args.steering_mode,
        throttle_mode=args.throttle_mode,
        steering_action_scale=args.steering_action_scale,
        straight_throttle_gain=args.straight_throttle_gain,
        straight_throttle_steering_threshold=args.straight_throttle_steering_threshold,
        controller=args.controller,
        observation_profile=args.observation_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
