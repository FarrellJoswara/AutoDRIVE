"""Gymnasium / Stable-Baselines3 compliance check for AutoDriveEnv.

Requires a working AutoDRIVE binary (Windows .exe or Linux .x86_64 in Docker).

Usage:
  python -m tests.check_gym_env
  python -m tests.check_gym_env --simulator "/app/simulator/AutoDRIVE Simulator.x86_64"
  python -m tests.check_gym_env --port 4567
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="check_env for AutoDriveEnv")
    parser.add_argument(
        "--simulator",
        type=Path,
        default=None,
        help="Path to AutoDRIVE binary (default: auto-detect / AICAR_SIMULATOR_PATH)",
    )
    parser.add_argument("--port", type=int, default=4567)
    parser.add_argument("--connect-timeout", type=float, default=60.0)
    args = parser.parse_args()

    from stable_baselines3.common.env_checker import check_env

    from src.env import AutoDriveEnv

    env = AutoDriveEnv(
        simulator_path=args.simulator,
        port=args.port,
        auto_launch=True,
        headless=True,
        connect_timeout=args.connect_timeout,
    )
    try:
        check_env(env, warn=True)
        print("check_env: OK")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
