"""Set the initial Gaussian action standard deviation of an SB3 PPO archive."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.initialize_camera_champion import _ObservationEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--std", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 < args.std <= 1.0:
        raise ValueError("--std must be in (0, 1]")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    env = DummyVecEnv([_ObservationEnv])
    model = PPO.load(str(args.source), env=env, device="cpu")
    with torch.no_grad():
        model.policy.log_std.fill_(math.log(args.std))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    verified = PPO.load(str(args.output), env=env, device="cpu")
    actual_std = float(verified.policy.log_std.detach().exp().mean().cpu())
    if not math.isclose(actual_std, args.std, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError(f"saved action standard deviation is {actual_std}, expected {args.std}")
    print(f"Saved {args.output} with verified action standard deviation {actual_std:.4f}")


if __name__ == "__main__":
    main()
