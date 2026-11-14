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


def wait_for_official_sensor_readiness(env, *, timeout_s: float = 180.0, poll_s: float = 0.25):
    """Wait for every simulator domain to deliver fresh required callbacks.

    Subprocess workers are polled against the deadline before responses are
    collected, so a dead or wedged domain cannot hang training startup.
    """
    import time

    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("readiness timeouts must be positive")
    deadline = time.monotonic() + timeout_s
    last = []
    while time.monotonic() < deadline:
        if callable(getattr(env, "env_method_async", None)) and getattr(env, "remotes", None):
            env.env_method_async("get_sensor_readiness")
            remaining = max(0.0, deadline - time.monotonic())
            responses_ready = True
            for remote in env.remotes:
                if not remote.poll(max(0.0, deadline - time.monotonic())):
                    responses_ready = False
                    break
            if not responses_ready:
                raise TimeoutError(
                    "official sensor readiness RPC timed out; last per-domain diagnostics: "
                    + __import__("json").dumps(last, sort_keys=True)
                )
            last = env.env_method_wait()
        else:
            last = env.env_method("get_sensor_readiness")
        if last and all(bool(item.get("ready")) for item in last):
            return last
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        "official sensor readiness timed out; per-domain callback diagnostics: "
        + __import__("json").dumps(last, sort_keys=True)
    )
