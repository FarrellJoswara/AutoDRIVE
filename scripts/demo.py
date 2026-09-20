"""Layer 1 live demo against AutoDRIVE Simulator (headed or headless).

Examples:
  python scripts/demo.py
  python scripts/demo.py --headless
  python scripts/demo.py --racers 2 --duration 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.racer.track import RaceTrack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDRIVE Layer 1 live demo")
    parser.add_argument(
        "--racers",
        type=int,
        default=None,
        help="Number of simulator instances (default: 1 if --headless else 2)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch with -batchmode -nographics (auto-connect via -ip/-port)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Drive duration in seconds (default: 15)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=4567,
        help="First Socket.IO port (default: 4567)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_racers = args.racers if args.racers is not None else (1 if args.headless else 2)
    if num_racers < 1:
        print("ERROR: --racers must be >= 1")
        return

    sim_path = (ROOT / "simulator" / "windows" / "AutoDRIVE Simulator.exe").resolve()
    if not sim_path.exists():
        print(f"ERROR: Simulator not found at {sim_path}")
        return

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
        return

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
            print(f"  Racer {r_id}: steps={steps}, max_speed={max_speed[r_id]:.2f}, last_pos={last_pos[r_id]}")
            if steps < 10 or max_speed[r_id] < 0.05:
                ok = False
        print("PASS" if ok else "PARTIAL")

        out_dir = ROOT / "logs" / "trajectories"
        for r_id, path in track.save_all_trajectories(out_dir).items():
            print(f"  CSV racer {r_id}: {path}")

    finally:
        track.kill_all()
        print("Done.")


if __name__ == "__main__":
    main()
