"""Live Layer 2 smoke: launch headless AutoDriveEnv, reset, step, close."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env import AutoDriveEnv, RewardConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer 2 AutoDriveEnv smoke test")
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--connect-timeout", type=float, default=90.0)
    args = parser.parse_args()

    print("launching AutoDriveEnv (headless)...")
    env = AutoDriveEnv(
        headless=True,
        port=args.port,
        frame_skip=1,
        max_episode_steps=max(50, args.steps + 10),
        stagnation_steps=200,
        connect_timeout=args.connect_timeout,
        reward_config=RewardConfig(forward_scale=1.0, collision_penalty=0.0),
    )
    try:
        obs, info = env.reset()
        print("reset ok")
        print(
            "  lidar",
            obs["lidar"].shape,
            float(obs["lidar"].min()),
            float(obs["lidar"].max()),
        )
        print("  state", obs["state"].shape, obs["state"])
        print("  info keys", sorted(info.keys()))

        total_r = 0.0
        terminated = truncated = False
        n = 0
        t0 = time.time()
        while n < args.steps and not (terminated or truncated):
            action = env.action_space.sample()
            action[0] = 0.6  # bias throttle forward to confirm motion
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            n += 1
            if n == 1 or n % 10 == 0:
                print(
                    f"  step {n}: v_long={info['v_long']:.3f} "
                    f"reward={float(reward):.3f} idle={info['idle_steps']} "
                    f"coll_evt={info['collision_event']}"
                )

        dt = time.time() - t0
        print(
            f"done steps={n} total_reward={total_r:.3f} "
            f"terminated={terminated} truncated={truncated} "
            f"reason={info.get('truncate_reason')} elapsed={dt:.1f}s"
        )
        if n < 1:
            print("Layer 2 smoke: FAIL (no steps)")
            return 1
        print("Layer 2 smoke: PASS")
        return 0
    finally:
        env.close()
        print("closed")


if __name__ == "__main__":
    raise SystemExit(main())
