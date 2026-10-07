"""Competition policy runner using only permitted ROS 2 inputs and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_policy(model_path: Path, *, device: str = "cpu") -> Any:
    """Load a saved PPO policy using the current canonical Layer 2 spaces."""
    from stable_baselines3 import PPO

    from src.layer2.spaces import make_action_space, make_observation_space

    checkpoint = Path(model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    custom_objects = {
        "observation_space": make_observation_space(),
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
) -> None:
    """Drive continuously; no reset, odometry, race counters, or map data."""
    from src.layer1.ros2_racer import RacerRos2
    from src.layer2.spaces import map_throttle_action, snapshot_to_obs

    model = load_policy(model_path, device=device)
    racer = RacerRos2(timeout_s=timeout_s, include_race_metrics=False)
    previous_throttle = 0.0
    previous_steering = 0.0
    try:
        snap = racer.wait_until_ready()
        observation = snapshot_to_obs(snap, previous_throttle, previous_steering)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            command = np.asarray(action, dtype=np.float32).reshape(2)
            previous_throttle = map_throttle_action(
                float(command[0]), negative_throttle_mode
            )
            previous_steering = float(np.clip(command[1], -1.0, 1.0))
            snap = racer.step(previous_throttle, previous_steering)
            observation = snapshot_to_obs(
                snap, previous_throttle, previous_steering
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--negative-throttle-mode",
        choices=("allow", "zero", "positive_magnitude"),
        default="allow",
        help="How to map negative policy throttle actions to the actuator",
    )
    args = parser.parse_args(argv)
    run_policy(
        args.model,
        device=args.device,
        timeout_s=args.timeout_s,
        negative_throttle_mode=args.negative_throttle_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
