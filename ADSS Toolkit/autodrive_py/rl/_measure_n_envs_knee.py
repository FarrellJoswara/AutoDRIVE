"""Measure env steps/sec vs n_envs (throughput knee) — CPU-bound honesty.

Does **not** train PPO. Sweeps n_envs in {1,4,8,12,16} with short random-action
rollouts and prints the knee (max steps/sec). Refuse when a live overnight /
train.lock is held so we do not steal cores from a protected soak.

Usage (from ADSS Toolkit/autodrive_py, with rl/.venv):
  python -m rl._measure_n_envs_knee
  python -m rl._measure_n_envs_knee --map map0 --steps 800 --force
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
MAPS_DIR = RL_DIR / "maps"
MODELS_DIR = RL_DIR / "models"
DEFAULT_N_ENVS = (1, 4, 8, 12, 16)


def _live_train_blocks(*, force: bool) -> str | None:
    """Refuse competing with a protected overnight unless --force."""
    if force:
        return None
    from .live_status import find_latest_status
    from .metrics_io import run_lock_path
    from .ui_ops import lock_owner

    found = find_latest_status(RL_DIR / "runs")
    if found is not None:
        _st_path, live = found
        live = live or {}
        phase = str(live.get("phase") or "")
        rid = str(live.get("run_id") or "")
        if phase in ("learning", "validating", "saving", "starting") and rid:
            owner = lock_owner(MODELS_DIR, rid)
            if owner and owner.get("alive"):
                return (
                    f"Refuse: live train {rid} phase={phase} pid={owner.get('pid')} "
                    f"(steps/s~{live.get('steps_per_sec')}). Do not steal cores from "
                    f"a protected soak - wait for idle, or pass --force."
                )
    # Any alive train.lock under models/
    if MODELS_DIR.is_dir():
        for d in MODELS_DIR.iterdir():
            if not d.is_dir():
                continue
            lock = run_lock_path(MODELS_DIR, d.name)
            if not lock.is_file():
                continue
            owner = lock_owner(MODELS_DIR, d.name)
            if owner and owner.get("alive"):
                return (
                    f"Refuse: train.lock alive for {d.name} pid={owner.get('pid')}. "
                    f"Pass --force only if you accept throughput contention."
                )
    return None


def _map_yaml(map_id: str) -> Path:
    from .racing_env import resolve_map_yaml

    return Path(resolve_map_yaml(map_id, MAPS_DIR))


def _measure_one(n_envs: int, map_yaml: Path, *, steps: int, seed: int) -> dict:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .train_ppo import _make_env

    env_fns = [
        _make_env(
            map_yaml,
            n_lidar=108,
            seed=seed,
            rank=i,
            spawn_jitter=True,
            collision_first=False,
            speed_gate=False,
            ttc_truncate=True,
            lidar_dr=False,
        )
        for i in range(int(n_envs))
    ]
    vec_kind = "dummy"
    try:
        if n_envs > 1:
            vec = SubprocVecEnv(env_fns)
            vec_kind = "subproc"
        else:
            vec = DummyVecEnv(env_fns)
    except Exception as exc:  # noqa: BLE001 — Windows Subproc often fails
        print(f"  n_envs={n_envs}: Subproc failed ({exc}); Dummy fallback")
        vec = DummyVecEnv(env_fns)
        vec_kind = "dummy"
    try:
        vec.reset()
        # Warmup (exclude process spawn / first reset from timing)
        for _ in range(min(8, max(1, steps // 10))):
            actions = [vec.action_space.sample() for _ in range(vec.num_envs)]
            # SB3 VecEnv expects (n_envs, action_dim)
            import numpy as np

            act = np.asarray(actions, dtype=np.int64)
            if act.ndim == 1:
                act = act.reshape(vec.num_envs, -1)
            vec.step(act)
        t0 = time.perf_counter()
        n_steps = 0
        while n_steps < steps:
            actions = [vec.action_space.sample() for _ in range(vec.num_envs)]
            import numpy as np

            act = np.asarray(actions, dtype=np.int64)
            if act.ndim == 1:
                act = act.reshape(vec.num_envs, -1)
            vec.step(act)
            n_steps += 1
        dt = max(1e-6, time.perf_counter() - t0)
        # SB3 counts env-steps as n_envs * vec steps
        total = float(n_steps * vec.num_envs)
        sps = total / dt
        return {
            "n_envs": int(n_envs),
            "vec": vec_kind,
            "vec_steps": n_steps,
            "env_steps": int(total),
            "seconds": round(dt, 3),
            "steps_per_sec": round(sps, 1),
        }
    finally:
        vec.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Measure RacingEnv steps/sec vs n_envs (knee)")
    p.add_argument("--map", default="map0", help="train-safe map id (default map0)")
    p.add_argument("--steps", type=int, default=600, help="vec-env steps per n_envs trial")
    p.add_argument(
        "--n-envs",
        default=",".join(str(n) for n in DEFAULT_N_ENVS),
        help="comma list e.g. 1,4,8,12,16",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--force",
        action="store_true",
        help="run even if a live train.lock / overnight is active (not recommended)",
    )
    args = p.parse_args(argv)

    block = _live_train_blocks(force=bool(args.force))
    if block:
        print(f"ERROR: {block}")
        return 2

    try:
        map_yaml = _map_yaml(str(args.map))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: map resolve failed: {exc}")
        return 1
    if not map_yaml.is_file():
        print(f"ERROR: map yaml missing: {map_yaml}")
        return 1

    n_list = []
    for part in str(args.n_envs).split(","):
        part = part.strip()
        if not part:
            continue
        n_list.append(max(1, min(32, int(part))))
    if not n_list:
        n_list = list(DEFAULT_N_ENVS)

    print(f"n_envs knee measure  map={args.map}  steps/trial={args.steps}  candidates={n_list}")
    print("(random actions — throughput only; not a race score)\n")
    rows = []
    for n in n_list:
        print(f"measuring n_envs={n} …", flush=True)
        row = _measure_one(n, map_yaml, steps=max(50, int(args.steps)), seed=int(args.seed))
        rows.append(row)
        print(
            f"  n_envs={row['n_envs']:>2}  vec={row['vec']:<7}  "
            f"steps/sec={row['steps_per_sec']:>7.1f}  ({row['seconds']}s)"
        )

    best = max(rows, key=lambda r: r["steps_per_sec"])
    print(
        f"\nKNEE ≈ n_envs={best['n_envs']} @ {best['steps_per_sec']} steps/sec "
        f"(pick max sps, not max workers). Idle GPU is expected — do not raise n_envs for CUDA theater."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
