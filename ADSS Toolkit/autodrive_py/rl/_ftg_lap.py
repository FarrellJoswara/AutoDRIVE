"""Scratch: can FTG complete a real lap given enough time?"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.ftg import FollowTheGap  # noqa: E402
from rl.racing_env import RacingEnv  # noqa: E402

RL = Path(__file__).resolve().parent
TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 400.0
THROTTLE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.66
MAPS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["map0"]

for mid in MAPS:
    env = RacingEnv(map_yaml=RL / f"maps/{mid}/{mid}.yaml", n_lidar=180, seed=0,
                    timeout_s=TIMEOUT)
    pol = FollowTheGap(n_lidar=180, max_throttle=THROTTLE)
    env.reset(seed=0)
    cols = 0
    lap = None
    t0 = time.time()
    steps = 0
    while True:
        a = pol.act(env._raw_scan)
        obs, rew, term, trunc, info = env.step(a)
        steps += 1
        if info.get("collision"):
            cols += 1
        if info.get("lap_time") is not None:
            lap = info["lap_time"]
        if term or trunc:
            break
    print(f"{mid}: lap={lap if lap is None else round(lap, 2)} "
          f"tag={info['crash_tag']} cols={cols} "
          f"progress={env.progress_frac * 100:.1f}% "
          f"len={env._track_length:.0f}m steps={steps} wall={time.time() - t0:.0f}s")
