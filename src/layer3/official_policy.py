"""Competition policy runner using only permitted ROS 2 inputs and outputs."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_policy(
    model_path: Path, *, device: str = "cpu", controller: str = "ppo",
    observation_profile: str = "official_sensors",
) -> Any:
    if controller == "lidar_gap":
        from src.layer3.lidar_gap_policy import load_gap_policy

        return load_gap_policy()
    if controller != "ppo":
        raise ValueError(f"Unknown official controller: {controller}")
    """Load a saved PPO policy using the current canonical Layer 2 spaces."""
    from stable_baselines3 import PPO

    from src.layer2.spaces import make_action_space, make_observation_space

    checkpoint = Path(model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    custom_objects = {
        "observation_space": make_observation_space(
            lidar_history_frames=4 if observation_profile == "official_sensors_history" else 1
        ),
        "action_space": make_action_space(),
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "clip_range": lambda _: 0.2,
        "lr_schedule": lambda _: 1e-5,
    }
    return PPO.load(str(checkpoint), device=device, custom_objects=custom_objects)


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
    observation_profile: str = "official_sensors",
) -> None:
    """Drive continuously; no reset, odometry, race counters, or map data."""
    from src.layer1.ros2_racer import RacerRos2
    from src.layer2.spaces import (
        snapshot_to_obs,
        transform_policy_action,
    )

    if observation_profile not in ("official_sensors", "official_sensors_history"):
        raise ValueError("official runner supports official sensor observation profiles only")
    model = load_policy(
        model_path, device=device, controller=controller,
        observation_profile=observation_profile,
    )
    racer = RacerRos2(timeout_s=timeout_s, include_race_metrics=False)
    previous_throttle = 0.0
    previous_steering = 0.0
    try:
        snap = racer.wait_until_ready()
        previous_scans: deque[np.ndarray] = deque(maxlen=3)
        observation = snapshot_to_obs(snap, previous_throttle, previous_steering)
        if observation_profile == "official_sensors_history":
            previous_scans.extend([observation["lidar"].copy()] * 3)
            observation = snapshot_to_obs(
                snap, previous_throttle, previous_steering,
                lidar_history=list(previous_scans),
            )
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
            if observation_profile == "official_sensors_history":
                current_scan = snapshot_to_obs(snap, previous_throttle, previous_steering)["lidar"]
                observation = snapshot_to_obs(
                    snap, previous_throttle, previous_steering,
                    lidar_history=list(previous_scans),
                )
                previous_scans.append(current_scan.copy())
            else:
                observation = snapshot_to_obs(snap, previous_throttle, previous_steering)
    except KeyboardInterrupt:
        pass
    finally:
        # Stopping the process leaves the simulator's current actuator command
        # untouched; the next competition run starts in a fresh simulator.
        racer.kill()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/policy.zip"))
    parser.add_argument("--controller", choices=("ppo", "lidar_gap"), default="ppo")
    parser.add_argument(
        "--observation-profile",
        choices=("official_sensors", "official_sensors_history"),
        default="official_sensors",
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
