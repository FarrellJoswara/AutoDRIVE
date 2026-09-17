"""Watch a policy on a top-down map (separate process — zero train overhead)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ftg import FollowTheGap
from .racing_env import RacingEnv, resolve_map_yaml
from .viewer import MapViewer


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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-beams",
        action="store_true",
        help="Hide sparse LiDAR rays (map + car + numbers only)",
    )
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
