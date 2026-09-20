"""Env factories for PPO: single env or SubprocVecEnv of N AutoDriveEnvs."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.layer2 import AutoDriveEnv, RewardConfig


def build_env(
    *,
    port: int,
    seed: int = 0,
    headless: bool = True,
    auto_launch: bool = True,
    connect_timeout: float = 60.0,
    frame_skip: int = 1,
    max_episode_steps: int = 0,
    stagnation_speed_threshold: float = 0.15,
    stagnation_steps: int = 200,
    forward_scale: float = 1.0,
    collision_penalty: float = 0.0,
    slip_penalty: float = 0.0,
    steer_jerk_penalty: float = 0.0,
) -> AutoDriveEnv:
    """Construct one AutoDriveEnv (top-level for Windows pickling / partial)."""
    env = AutoDriveEnv(
        port=port,
        headless=headless,
        auto_launch=auto_launch,
        connect_timeout=connect_timeout,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        stagnation_speed_threshold=stagnation_speed_threshold,
        stagnation_steps=stagnation_steps,
        reward_config=RewardConfig(
            forward_scale=forward_scale,
            collision_penalty=collision_penalty,
            slip_penalty=slip_penalty,
            steer_jerk_penalty=steer_jerk_penalty,
        ),
    )
    env.reset(seed=seed)
    return env


def make_env(
    port: int,
    seed: int = 0,
    **env_kwargs: Any,
) -> Callable[[], AutoDriveEnv]:
    """Return a zero-arg factory suitable for DummyVecEnv / SubprocVecEnv."""
    return partial(build_env, port=port, seed=seed, **env_kwargs)


def make_vec_env(
    n_envs: int,
    *,
    base_port: int = 4567,
    seed: int = 0,
    **env_kwargs: Any,
) -> VecEnv:
    """
    Build a vectorized env.

    n_envs == 1 → DummyVecEnv (same process).
    n_envs >= 2 → SubprocVecEnv (one process per AutoDriveEnv / Unity).
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    env_fns = [
        make_env(port=base_port + i, seed=seed + i, **env_kwargs) for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    # spawn (not forkserver): gevent WSGI in each worker must bind its own hub;
    # forkserver inherits a broken hub and Racer ports never listen.
    return SubprocVecEnv(env_fns, start_method="spawn")


def env_kwargs_from_args(args: Any) -> Dict[str, Any]:
    """Map argparse namespace fields used by train/play into build_env kwargs."""
    keys = (
        "headless",
        "auto_launch",
        "connect_timeout",
        "frame_skip",
        "max_episode_steps",
        "stagnation_speed_threshold",
        "stagnation_steps",
        "forward_scale",
        "collision_penalty",
        "slip_penalty",
        "steer_jerk_penalty",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if hasattr(args, key):
            out[key] = getattr(args, key)
    return out
