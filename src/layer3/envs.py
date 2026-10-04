"""Env factories for PPO: single env or SubprocVecEnv of N AutoDriveEnvs."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.layer2 import AutoDriveEnv, RewardConfig

BRIDGE_BASE_PORT = 4567


def build_env(
    *,
    port: int,
    seed: int = 0,
    headless: bool = True,
    auto_launch: bool = True,
    connect_timeout: float = 60.0,
    frame_skip: int = 4,
    max_episode_steps: int = 0,
    stagnation_speed_threshold: float = 0.15,
    stagnation_steps: int = 50,
    map_id: str = "none",
    laps_per_episode: int = 0,
    frontier_stagnation_seconds: float = 5.0,
    terminate_on_collision: bool = True,
    route_progress_scale: float = 10.0,
    time_penalty_per_second: float = 5.0,
    forward_scale: float = 0.0,
    backward_speed_penalty_scale: float = 1.0,
    collision_penalty_magnitude: float = 100.0,
    collision_reward_percent: float = 100.0,
    episode_failure_penalty_magnitude: float = 100.0,
    episode_failure_reward_percent: float = 100.0,
    slip_penalty: float = 0.2,
    steer_jerk_penalty: float = 0.05,
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
        map_id=map_id,
        laps_per_episode=laps_per_episode,
        frontier_stagnation_seconds=frontier_stagnation_seconds,
        terminate_on_collision=terminate_on_collision,
        reward_config=RewardConfig(
            forward_scale=forward_scale,
            backward_speed_penalty_scale=backward_speed_penalty_scale,
            route_progress_scale=route_progress_scale,
            time_penalty_per_second=time_penalty_per_second,
            collision_penalty_magnitude=collision_penalty_magnitude,
            collision_reward_percent=collision_reward_percent,
            episode_failure_penalty_magnitude=episode_failure_penalty_magnitude,
            episode_failure_reward_percent=episode_failure_reward_percent,
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
    seed: int = 0,
    port_start: int = BRIDGE_BASE_PORT,
    **env_kwargs: Any,
) -> VecEnv:
    """
    Build a vectorized env.

    n_envs == 1 → DummyVecEnv (same process).
    n_envs >= 2 → SubprocVecEnv (one process per AutoDriveEnv / Unity).
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    if n_envs > 16:
        raise ValueError("n_envs must be <= 16 (the simulator bridge range)")

    env_fns = [
        make_env(port=port_start + i, seed=seed + i, **env_kwargs)
        for i in range(n_envs)
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
        "map_id",
        "laps_per_episode",
        "frontier_stagnation_seconds",
        "terminate_on_collision",
        "route_progress_scale",
        "time_penalty_per_second",
        "forward_scale",
        "backward_speed_penalty_scale",
        "collision_penalty_magnitude",
        "collision_reward_percent",
        "episode_failure_penalty_magnitude",
        "episode_failure_reward_percent",
        "slip_penalty",
        "steer_jerk_penalty",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if hasattr(args, key):
            out[key] = getattr(args, key)
    return out
