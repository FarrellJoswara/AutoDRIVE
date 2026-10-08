"""One official ROS domain per simulator, with one shared PPO learner."""
from functools import partial
import os


def build_official_env(domain_id: int, **kwargs):
    # Called inside a fresh spawned process, before rclpy initializes.
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    from stable_baselines3.common.monitor import Monitor
    from src.layer2.official_race_env import OfficialRaceEnv
    return Monitor(OfficialRaceEnv(**kwargs))


def make_official_vec_env(n_envs: int, *, seed: int = 0, **kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    if not 1 <= n_envs <= 8:
        raise ValueError("official n_envs must be between 1 and 8")
    factories = [partial(build_official_env, index, **kwargs) for index in range(n_envs)]
    env = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    env.seed(seed)
    return env


def close_official_vec_env(env):
    """Bound shutdown even when a simulator or environment worker has died."""
    if env is None:
        return
    processes = getattr(env, "processes", None)
    if processes is None:
        env.close()
        return
    # Gracefully close workers that are not stuck waiting for a sensor frame.
    for remote in env.remotes:
        try:
            remote.send(("close", None))
        except (EOFError, BrokenPipeError, OSError):
            pass
    import time
    deadline = time.monotonic() + 6.0
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
    for remote in env.remotes:
        remote.close()
    env.closed = True
