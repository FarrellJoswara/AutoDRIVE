"""Watch a policy on a top-down map (separate process — zero train overhead).

Modes
-----
- Default: roll out FTG or a fixed PPO zip in a local Gym env + OpenCV.
- ``--follow``: lag-behind trail — read ``live_status.json`` for train metrics and
  drive a *twin* car with the latest saved weights. Does **not** stream train frames;
  training stays headless.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .ftg import FollowTheGap
from .live_status import find_latest_status, read_live_status, resolve_latest_weights
from .racing_env import RacingEnv, resolve_map_yaml
from .viewer import MapViewer


def _overlay_from_status(status: dict | None, twin: dict) -> dict:
    m = dict(twin)
    if not status:
        return m
    m["train_timesteps"] = status.get("timesteps")
    m["train_ep_rew"] = status.get("ep_rew_mean")
    m["train_episodes"] = status.get("episode_count_estimate")
    m["train_collisions"] = status.get("collisions_estimate")
    m["run_id"] = status.get("run_id")
    return m


def _run_standalone(args) -> int:
    maps_root = Path(__file__).resolve().parent / "maps"
    map_yaml = resolve_map_yaml(args.map, maps_root)
    env = RacingEnv(map_yaml=map_yaml, seed=args.seed)
    viewer = MapViewer(env, show_beams=not args.no_beams)

    model = None
    ftg = None
    if args.policy == "ppo":
        from stable_baselines3 import PPO

        model_path = Path(args.model) if args.model else None
        if model_path is None or not model_path.exists():
            zips = sorted((Path(__file__).resolve().parent / "models").glob("*/best_model.zip"))
            if not zips:
                print("ERROR: no PPO model found; pass --model path/to/best_model.zip")
                return 2
            model_path = zips[-1]
        print(f"loading {model_path}")
        model = PPO.load(str(model_path), device="auto")
    else:
        ftg = FollowTheGap()

    print("Viewer: q or Esc to quit. Independent of training (use TensorBoard for curves).")
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_return = 0.0
        collisions = 0
        while not done:
            if args.policy == "ftg":
                action = ftg.act(env._raw_scan)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_return += float(reward)
            if info.get("collision"):
                collisions += 1
            if step % max(1, args.every) == 0:
                metrics = {
                    "speed": info.get("speed", env._state[3] if env._state is not None else 0.0),
                    "step": step,
                    "episode": ep,
                    "ep_return": ep_return,
                    "collisions": collisions,
                }
                if not viewer.show(metrics=metrics, wait_ms=1):
                    viewer.close()
                    return 0
            done = terminated or truncated
        print(
            f"episode={ep} steps={step} return={ep_return:.1f} "
            f"collision={info.get('collision')} lap_time={info.get('lap_time')}"
        )
    viewer.close()
    return 0


def _run_follow(args) -> int:
    """Lag-behind twin: train stats from JSON + local rollout of latest weights."""
    from stable_baselines3 import PPO

    rl_root = Path(__file__).resolve().parent
    runs_root = rl_root / "runs"
    models_root = rl_root / "models"
    maps_root = rl_root / "maps"
    map_yaml = resolve_map_yaml(args.map, maps_root)

    status_path: Path | None = None
    status: dict | None = None
    if args.run_id:
        status_path = runs_root / args.run_id / "live_status.json"
        status = read_live_status(status_path)
        if status is None:
            print(f"Waiting for {status_path} ...")
        else:
            found = find_latest_status(runs_root)
            if found:
                status_path, status = found
                print(f"Following {status_path}")
            else:
                print(f"Waiting for any {runs_root}/*/live_status.json ...")

    env = RacingEnv(map_yaml=map_yaml, seed=args.seed)
    viewer = MapViewer(env, show_beams=not args.no_beams, window="F1TENTH RL (follow)")

    model = None
    loaded_path: Path | None = None
    loaded_mtime = -1.0
    ep = 0
    print(
        "Follow mode: twin car uses latest checkpoint; overlay shows train live_status. "
        "q/Esc quit. Training is NOT slowed by this window."
    )

    poll_s = max(0.2, float(args.poll))
    while True:
        # Refresh status file handle / contents
        if status_path is None:
            found = find_latest_status(runs_root)
            if found:
                status_path, status = found
                print(f"Following {status_path}")
        else:
            status = read_live_status(status_path) or status

        run_id = (status or {}).get("run_id") or args.run_id
        weights = None
        if run_id:
            weights = resolve_latest_weights(models_root / str(run_id), status)
        if weights is None and status_path is not None:
            # Status exists but weights not yet written — keep waiting
            twin = {"step": 0, "episode": ep, "ep_return": 0.0, "collisions": 0, "speed": 0.0}
            if not viewer.show(metrics=_overlay_from_status(status, twin), wait_ms=max(1, int(poll_s * 1000))):
                viewer.close()
                return 0
            time.sleep(0.05)
            continue

        if weights is None:
            twin = {"step": 0, "episode": ep, "ep_return": 0.0, "collisions": 0, "speed": 0.0}
            if not viewer.show(metrics=_overlay_from_status(status, twin), wait_ms=max(1, int(poll_s * 1000))):
                viewer.close()
                return 0
            time.sleep(0.05)
            continue

        mtime = weights.stat().st_mtime
        if model is None or weights != loaded_path or mtime > loaded_mtime:
            print(f"loading twin weights {weights}")
            model = PPO.load(str(weights), device="auto")
            loaded_path = weights
            loaded_mtime = mtime

        obs, info = env.reset()
        done = False
        step = 0
        ep_return = 0.0
        collisions = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            ep_return += float(reward)
            if info.get("collision"):
                collisions += 1

            # Light reload of status while twin drives
            if step % 20 == 0 and status_path is not None:
                status = read_live_status(status_path) or status
                # Hot-swap if a newer zip appeared
                if run_id:
                    newer = resolve_latest_weights(models_root / str(run_id), status)
                    if newer is not None and newer.stat().st_mtime > loaded_mtime:
                        break  # restart episode with new brain

            if step % max(1, args.every) == 0:
                twin = {
                    "speed": info.get("speed", env._state[3] if env._state is not None else 0.0),
                    "step": step,
                    "episode": ep,
                    "ep_return": ep_return,
                    "collisions": collisions,
                }
                if not viewer.show(
                    metrics=_overlay_from_status(status, twin),
                    wait_ms=1,
                ):
                    viewer.close()
                    return 0
            done = terminated or truncated

        print(
            f"twin ep={ep} steps={step} return={ep_return:.1f} "
            f"train_ts={(status or {}).get('timesteps')} weights={loaded_path.name if loaded_path else None}"
        )
        ep += 1
        if args.episodes > 0 and ep >= args.episodes:
            break

    viewer.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal live viewer (map + car + metrics). Not used during training."
    )
    parser.add_argument("--map", type=str, default="map0")
    parser.add_argument("--policy", choices=["ftg", "ppo"], default="ftg")
    parser.add_argument("--model", type=str, default="", help="SB3 zip for --policy ppo")
    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help="Redraw every N steps (higher = lighter; try 5-10 if laggy)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes then exit (follow: 0 = run forever until q)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-beams",
        action="store_true",
        help="Hide sparse LiDAR rays (map + car + numbers only)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Lag-behind trail: live_status.json overlay + twin rollout of latest_model/checkpoint",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Follow a specific run (default: newest live_status.json under rl/runs/)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Seconds between status polls while waiting for weights (follow mode)",
    )
    args = parser.parse_args(argv)

    if args.follow:
        if args.episodes == 5:
            # Sensible follow default: keep going until user quits
            args.episodes = 0
        return _run_follow(args)
    return _run_standalone(args)


if __name__ == "__main__":
    raise SystemExit(main())
