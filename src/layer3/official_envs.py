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
    # Native ROS runtimes may reserve a domain other than zero. Keep the
    # historical 0..n_envs-1 allocation by default, while allowing an
    # explicitly configured base domain to match a preflighted runtime.
    try:
        domain_base = int(os.environ.get("AICAR_ROS_DOMAIN_ID_BASE", "0"))
    except ValueError as exc:
        raise ValueError("AICAR_ROS_DOMAIN_ID_BASE must be an integer") from exc
    if domain_base < 0 or domain_base + n_envs - 1 > 232:
        raise ValueError("official ROS domain range must stay within 0..232")
    factories = [
        partial(build_official_env, domain_base + index, **kwargs)
        for index in range(n_envs)
    ]
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
