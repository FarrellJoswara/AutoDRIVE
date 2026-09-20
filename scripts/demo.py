"""Live demos and checks for Layer 1 / Layer 2 / Layer 3.

Examples:
  python scripts/demo.py layer1
  python scripts/demo.py layer1 --headless --racers 1 --duration 10
  python scripts/demo.py layer2 --steps 40
  python scripts/demo.py check-env
  python scripts/demo.py train --n-envs 1 --timesteps 10000
  python scripts/demo.py play --model logs/rl/.../final_model.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _cmd_layer1(args: argparse.Namespace) -> int:
    from src.layer1.track import RaceTrack

    num_racers = args.racers if args.racers is not None else (1 if args.headless else 2)
    if num_racers < 1:
        print("ERROR: --racers must be >= 1")
        return 1

    sim_path = (ROOT / "simulator" / "windows" / "AutoDRIVE Simulator.exe").resolve()
    if not sim_path.exists():
        print(f"ERROR: Simulator not found at {sim_path}")
        return 1

    mode = "Headless (-batchmode -nographics)" if args.headless else "Headed Visual GUI"
    ports = ", ".join(str(args.base_port + i) for i in range(num_racers))

    print("\n" + "=" * 75)
    print("      AUTODRIVE LAYER 1 DEMO")
    print("=" * 75)
    print(f"Simulator : {sim_path}")
    print(f"Fleet     : {num_racers} @ port(s) {ports}")
    print(f"Mode      : {mode}")
    print(f"Duration  : {args.duration:.0f}s")
    print("=" * 75 + "\n")

    track = RaceTrack(
        num_racers=num_racers,
        base_port=args.base_port,
        simulator_path=sim_path,
        auto_launch=True,
        headless=args.headless,
    )

    print(f"--> Servers listening on {ports}")
    if args.headless:
        print("--> Headless sim launched (CLI auto-connect)\n")
    else:
        print("--> Simulator window(s) launched")
        print("    For each window: set Port as listed below, then click Connect:")
        for i in range(num_racers):
            print(f"      Instance {i + 1} -> port {args.base_port + i}")
        print()

    timeout = 60.0 if args.headless else 120.0
    print(f"Waiting for connect (timeout {int(timeout)}s)...")
    connected: set[int] = set()
    start_wait = time.time()
    last_print = -1

    while time.time() - start_wait < timeout:
        for r_id in range(num_racers):
            if r_id not in connected and track.racers[r_id].is_connected:
                connected.add(r_id)
                print(f"--> Connected racer {r_id} ({len(connected)}/{num_racers})")
        if len(connected) == num_racers:
            break
        elapsed = int(time.time() - start_wait)
        if elapsed % 10 == 0 and elapsed != last_print:
            print(f"    ... {int(timeout - elapsed)}s remaining")
            last_print = elapsed
        time.sleep(0.5)

    if not connected:
        print("[!] No simulator connected. Exiting.")
        track.kill_all()
        return 1

    if len(connected) < num_racers:
        missing = sorted(set(range(num_racers)) - connected)
        print(f"[!] Missing racers {missing}; continuing with {sorted(connected)}")

    active = sorted(connected)
    print(f"--> Driving forward (throttle=1, steering=0) for {args.duration:.0f}s\n")

    run_start = time.time()
    tick = 0
    max_speed = {r_id: 0.0 for r_id in active}
    last_pos = {r_id: None for r_id in active}

    try:
        while time.time() - run_start < args.duration:
            tick += 1
            telemetry = track.step_all({r_id: (1, 0) for r_id in active})

            for r_id in active:
                snap = telemetry.get(r_id)
                if snap:
                    max_speed[r_id] = max(max_speed[r_id], snap.true_speed)
                    last_pos[r_id] = snap.position

            if tick % 5 == 0:
                parts = []
                for r_id in active:
                    snap = telemetry.get(r_id)
                    if snap:
                        parts.append(
                            f"R{r_id}: pos=({snap.position[0]:5.2f},{snap.position[2]:5.2f}) "
                            f"spd={snap.true_speed:4.1f} steps={track.racers[r_id].step_counter}"
                        )
                print(f"[{tick:04d} | {time.time() - run_start:4.1f}s] " + " | ".join(parts))

            time.sleep(0.025)

        print("\nSummary:")
        ok = True
        for r_id in active:
            steps = track.racers[r_id].step_counter
            print(
                f"  Racer {r_id}: steps={steps}, max_speed={max_speed[r_id]:.2f}, "
                f"last_pos={last_pos[r_id]}"
            )
            if steps < 10 or max_speed[r_id] < 0.05:
                ok = False
        print("PASS" if ok else "PARTIAL")

        out_dir = ROOT / "logs" / "trajectories"
        for r_id, path in track.save_all_trajectories(out_dir).items():
            print(f"  CSV racer {r_id}: {path}")
        return 0 if ok else 2
    finally:
        track.kill_all()
        print("Done.")


def _cmd_layer2(args: argparse.Namespace) -> int:
    from src.layer2 import AutoDriveEnv, RewardConfig

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
            action[0] = 0.6
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


def _cmd_check_env(args: argparse.Namespace) -> int:
    from stable_baselines3.common.env_checker import check_env

    from src.layer2 import AutoDriveEnv

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


def _cmd_train(args: argparse.Namespace) -> int:
    from src.layer3.train import main as train_main

    # Rebuild argv for train.main from remaining namespace fields.
    argv = [
        "--n-envs",
        str(args.n_envs),
        "--base-port",
        str(args.base_port),
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--connect-timeout",
        str(args.connect_timeout),
        "--forward-scale",
        str(args.forward_scale),
        "--collision-penalty",
        str(args.collision_penalty),
    ]
    if args.out is not None:
        argv.extend(["--out", str(args.out)])
    if args.resume is not None:
        argv.extend(["--resume", str(args.resume)])
    if args.headless:
        argv.append("--headless")
    else:
        argv.append("--no-headless")
    if args.auto_launch:
        argv.append("--auto-launch")
    else:
        argv.append("--no-auto-launch")
    return int(train_main(argv))


def _cmd_play(args: argparse.Namespace) -> int:
    from src.layer3.play import main as play_main

    argv = [
        "--model",
        str(args.model),
        "--port",
        str(args.port),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--connect-timeout",
        str(args.connect_timeout),
    ]
    if args.headless:
        argv.append("--headless")
    else:
        argv.append("--no-headless")
    return int(play_main(argv))


def main() -> int:
    parser = argparse.ArgumentParser(description="AiCar live demos / checks")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("layer1", help="Live Layer 1 RaceTrack demo")
    p1.add_argument(
        "--racers",
        type=int,
        default=None,
        help="Fleet size (default: 1 if --headless else 2)",
    )
    p1.add_argument("--headless", action="store_true", help="batchmode / nographics")
    p1.add_argument("--duration", type=float, default=15.0, help="Drive seconds")
    p1.add_argument("--base-port", type=int, default=4567)
    p1.set_defaults(func=_cmd_layer1)

    p2 = sub.add_parser("layer2", help="Live Layer 2 AutoDriveEnv smoke")
    p2.add_argument("--port", type=int, default=4567)
    p2.add_argument("--steps", type=int, default=40)
    p2.add_argument("--connect-timeout", type=float, default=90.0)
    p2.set_defaults(func=_cmd_layer2)

    pc = sub.add_parser("check-env", help="SB3 Gymnasium check_env (needs torch/SB3)")
    pc.add_argument("--simulator", type=Path, default=None)
    pc.add_argument("--port", type=int, default=4567)
    pc.add_argument("--connect-timeout", type=float, default=60.0)
    pc.set_defaults(func=_cmd_check_env)

    pt = sub.add_parser("train", help="Layer 3 PPO train")
    pt.add_argument("--n-envs", type=int, default=1)
    pt.add_argument("--base-port", type=int, default=4567)
    pt.add_argument("--timesteps", type=int, default=10_000)
    pt.add_argument("--out", type=Path, default=None)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--device", type=str, default="auto")
    pt.add_argument("--resume", type=Path, default=None)
    pt.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    pt.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    pt.add_argument("--connect-timeout", type=float, default=90.0)
    pt.add_argument("--forward-scale", type=float, default=1.0)
    pt.add_argument("--collision-penalty", type=float, default=0.0)
    pt.set_defaults(func=_cmd_train)

    pp = sub.add_parser("play", help="Layer 3 PPO play (no learning)")
    pp.add_argument("--model", type=Path, required=True)
    pp.add_argument("--port", type=int, default=4567)
    pp.add_argument("--steps", type=int, default=500)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--device", type=str, default="auto")
    pp.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    pp.add_argument("--connect-timeout", type=float, default=90.0)
    pp.set_defaults(func=_cmd_play)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
