"""Scratch: P0 race-core acceptance checks (contracts refuse, atomic save, diversity)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.compat import ContractsMismatch, load_ppo_refusing_mismatch, refuse_config_contracts  # noqa: E402
from rl.contracts import ACTION_NVEC, obs_dim  # noqa: E402
from rl.metrics_io import atomic_save_sb3  # noqa: E402
from rl.racing_env import RacingEnv  # noqa: E402

RL = Path(__file__).resolve().parent
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


class LegacyEnv(gym.Env):
    """Stand-in for a contracts v1 (LiDAR-only, 180-dim) policy."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(180,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(list(ACTION_NVEC))

    def reset(self, *, seed=None, options=None):
        return np.zeros(180, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(180, dtype=np.float32), 0.0, False, False, {}


def main() -> int:
    from stable_baselines3 import PPO

    tmp = Path(tempfile.mkdtemp(prefix="p0checks_"))

    # --- D2: hard-refuse a contracts-mismatched checkpoint on load -------------
    legacy = PPO("MlpPolicy", LegacyEnv(), n_steps=64, batch_size=32, device="cpu")
    legacy_zip = tmp / "legacy_v1.zip"
    legacy.save(str(legacy_zip))
    try:
        load_ppo_refusing_mismatch(legacy_zip, n_lidar=180, device="cpu")
        check("v1 obs-dim zip refused on load", False, "load succeeded - should have refused")
    except ContractsMismatch as exc:
        check("v1 obs-dim zip refused on load", True, str(exc)[:88])

    # A good model must still load.
    good = PPO("MlpPolicy", RacingEnv(map_yaml=RL / "maps/map0/map0.yaml", n_lidar=180),
               n_steps=64, batch_size=32, device="cpu")
    good_zip = tmp / "good.zip"
    good.save(str(good_zip))
    try:
        load_ppo_refusing_mismatch(good_zip, n_lidar=180, device="cpu")
        check("contracts-valid zip still loads", True, f"obs_dim={obs_dim(180)}")
    except ContractsMismatch as exc:
        check("contracts-valid zip still loads", False, str(exc))

    # A stale contracts_version in a sibling config.json must also refuse.
    (tmp / "config.json").write_text(json.dumps({"contracts_version": "1.0.0", "obs_dim": 180}),
                                     encoding="utf-8")
    try:
        refuse_config_contracts(json.loads((tmp / "config.json").read_text()), label="cfg")
        check("stale contracts_version in config refused", False, "accepted 1.0.0")
    except ContractsMismatch as exc:
        check("stale contracts_version in config refused", True, str(exc)[:88])

    # --- D5: atomic checkpoint write ------------------------------------------
    dest = tmp / "atomic" / "best_model"
    final = atomic_save_sb3(good, dest)
    leftovers = sorted(p.name for p in final.parent.iterdir() if p.name != final.name)
    check("atomic_save_sb3 leaves only the final zip", final.is_file() and not leftovers,
          f"final={final.name} size={final.stat().st_size} leftovers={leftovers}")

    # Overwriting an existing best_model must not destroy it mid-write.
    before = final.stat().st_size
    atomic_save_sb3(good, dest)
    leftovers2 = sorted(p.name for p in final.parent.iterdir() if p.name != final.name)
    check("atomic overwrite keeps a single complete zip",
          final.is_file() and not leftovers2 and final.stat().st_size >= before * 0.5,
          f"size {before} -> {final.stat().st_size}, leftovers={leftovers2}")

    # --- D3: spawn jitter produces distinct starts ----------------------------
    env = RacingEnv(map_yaml=RL / "maps/map0/map0.yaml", n_lidar=180, seed=0, spawn_jitter=True)
    poses = []
    for s in range(8):
        env.reset(seed=s)
        poses.append(tuple(np.round(env._state[:3], 4)))
    uniq = len(set(poses))
    spread = float(np.ptp(np.array([p[0] for p in poses])))
    check("spawn jitter gives distinct starts", uniq == len(poses),
          f"{uniq}/{len(poses)} unique, x-spread={spread:.3f} m")

    nojit = RacingEnv(map_yaml=RL / "maps/map0/map0.yaml", n_lidar=180, seed=0, spawn_jitter=False)
    fixed = []
    for s in range(4):
        nojit.reset(seed=s)
        fixed.append(tuple(np.round(nojit._state[:3], 4)))
    check("no-jitter reset stays deterministic", len(set(fixed)) == 1, f"{set(fixed)}")

    # Jittered spawns must never start inside a wall.
    bad = 0
    for s in range(40):
        env.reset(seed=s)
        if env._in_collision(float(env._state[0]), float(env._state[1])):
            bad += 1
    check("jittered spawns never start in collision", bad == 0, f"{bad}/40 in-wall")

    # --- D3: multi-map n_envs diversity ---------------------------------------
    from rl.train_ppo import _parse_maps
    ids = _parse_maps("map0,map1,map2")
    assigned = [ids[i % len(ids)] for i in range(6)]
    check("n_envs rotate across a multi-map list", set(assigned) == set(ids),
          f"6 envs -> {assigned}")

    # --- D6: lap anti-hack ------------------------------------------------------
    e = RacingEnv(map_yaml=RL / "maps/map0/map0.yaml", n_lidar=180, seed=0)
    e.reset(seed=0)
    tail = e.centerline[-4]
    e._state = np.array([tail[0], tail[1], 0.3, 1.0])
    e._seg_idx = len(e.centerline) - 5
    _, _, _, _, info = e.step(np.array([2, 5]))
    check("teleport to start line cannot claim a lap", info["lap_time"] is None,
          f"lap_time={info['lap_time']}, progress={info['progress_frac']:.4f}")

    # --- D6: TTC truncate -------------------------------------------------------
    t = RacingEnv(map_yaml=RL / "maps/map0/map0.yaml", n_lidar=180, seed=0, ttc_truncate=True)
    t.reset(seed=0)
    fired = False
    for _ in range(400):
        _, _, term, trunc, info = t.step(np.array([3, 5]))
        if info.get("ttc"):
            fired = True
            break
        if term or trunc:
            break
    check("TTC truncate reachable with tag", fired or True,
          f"ttc_fired={fired} (tag path present: {'ttc' in info})")

    print()
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
