"""Scratch: diagnose why FTG crashes early on the map pack."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.contracts import LIDAR_MAX_M  # noqa: E402
from rl.ftg import FollowTheGap  # noqa: E402
from rl.racing_env import RacingEnv  # noqa: E402

RL = Path(__file__).resolve().parent
env = RacingEnv(map_yaml=RL / "maps/map0/map0.yaml", n_lidar=180, seed=0)
pol = FollowTheGap(n_lidar=180)
env.reset(seed=0)

print(f"track_len={env._track_length:.1f} m  v_max={env.v_max} m/s  timeout={env.timeout_s} s")
print(f"min possible lap = {env._track_length / env.v_max:.1f} s\n")
print(f"{'step':>4} {'thr':>5} {'steer':>6} {'tgt_beam':>8} {'frac_at_max':>11} {'front_m':>8} {'minr':>6}")
step = 0
while True:
    raw = env._raw_scan
    a = pol.act(raw)
    thr, st = float(a[0]), float(a[1])
    r = raw.copy()
    at_max = float(np.mean(r >= LIDAR_MAX_M - 1e-3))
    front = float(r[len(r) // 2])
    if step % 15 == 0 or step < 5:
        steer_val = np.linspace(-1, 1, 11)[int(st)]
        print(f"{step:>4} {thr:>5.0f} {steer_val:>6.1f} {'-':>8} {at_max:>11.2f} {front:>8.2f} {float(np.min(r)):>6.2f}")
    obs, rew, term, trunc, info = env.step(a)
    step += 1
    if term or trunc:
        print(f"\nended step={step} t={step*env.dt:.2f}s tag={info['crash_tag']} "
              f"collision={info['collision']} progress={info['progress_frac']*100:.2f}% of lap")
        break
    if step > 1300:
        print("\nno termination in 1300 steps")
        break
