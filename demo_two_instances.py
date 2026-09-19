"""Layer 1 Live Multi-Instance Demo: 2 Visual Racers Driving Forward.

Launches 2 visual AutoDRIVE Simulator instances side-by-side with DirectX 12 hardware
acceleration on your GPU, manages both in turn-based lockstep, drives them forward,
prints live telemetry, and exports trajectory CSV logs.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.racer.track import RaceTrack


def main():
    sim_path = Path("simulator/windows/AutoDRIVE Simulator.exe").resolve()
    if not sim_path.exists():
        print(f"ERROR: Simulator not found at {sim_path}")
        return

    print("\n" + "=" * 75)
    print("      AUTODRIVE SIM-RACING: LAYER 1 MULTI-INSTANCE LIVE DEMO")
    print("=" * 75)
    print(f"Simulator Executable : {sim_path}")
    print("Fleet Size           : 2 Vehicles (Racer 0 @ port 4567, Racer 1 @ port 4568)")
    print("Render Mode          : Headed Visual GUI (DirectX 12 / NVIDIA RTX 3060)")
    print("=" * 75 + "\n")

    # 1. Initialize fleet of 2 Racers
    track = RaceTrack(
        num_racers=2,
        base_port=4567,
        simulator_path=sim_path,
        auto_launch=True,
        headless=False,
    )

    print("--> [1/3] Both Racer servers started (ports 4567 and 4568).")
    print("--> [2/3] Both 3D Simulator windows launched on your desktop!\n")
    print("=" * 75)
    print("  ACTION REQUIRED IN SIMULATOR WINDOWS:")
    print("  In each simulator window on your screen:")
    print("  - Instance 1: Verify Port is 4567 and click 'Connect'")
    print("  - Instance 2: Change Port to 4568 and click 'Connect'")
    print("=" * 75 + "\n")

    # Wait for both racers to establish connection
    timeout = 120.0
    print(f"Waiting for simulator(s) to connect (timeout {int(timeout)}s)...")
    connected_racers = set()
    start_wait = time.time()
    last_print = 0

    while time.time() - start_wait < timeout:
        for r_id in [0, 1]:
            if r_id not in connected_racers and track.racers[r_id].is_connected:
                connected_racers.add(r_id)
                print(f"\n--> [CONNECTED] Racer {r_id} connected successfully! ({len(connected_racers)}/2 connected)")
        if len(connected_racers) == 2:
            break
        elapsed = int(time.time() - start_wait)
        if elapsed % 10 == 0 and elapsed != last_print:
            print(f"    ... waiting for 'Connect' button clicks in simulator windows ({int(timeout - elapsed)}s remaining)")
            last_print = elapsed
        time.sleep(1.0)

    if len(connected_racers) == 0:
        print("\n[!] No simulator connected within timeout. Terminating.")
        track.kill_all()
        return

    active_racers = sorted(list(connected_racers))
    print(f"\n--> [3/3] ACTIVE FLEET ({len(active_racers)} car{'s' if len(active_racers) > 1 else ''}): DRIVING FORWARD!")
    print("    Throttle = 0.6 | Steering = 0.0 (Straight)\n")

    # 2. Drive forward for 15 seconds (turn-based lockstep)
    run_start = time.time()
    tick = 0

    try:
        while time.time() - run_start < 15.0:
            tick += 1
            # Step all active racers
            actions = {r_id: (0.6, 0.0) for r_id in active_racers}
            telemetry = track.step_all(actions)

            # Print telemetry report every 5 ticks (~8 Hz)
            if tick % 5 == 0:
                reports = []
                for r_id in active_racers:
                    snap = telemetry.get(r_id)
                    if snap:
                        reports.append(
                            f"Racer {r_id}: Pos=({snap.position[0]:5.2f}, {snap.position[2]:5.2f}) | "
                            f"Speed={snap.true_speed:4.1f} m/s | "
                            f"Yaw={snap.heading_yaw:4.2f} rad"
                        )
                print(f"[Tick {tick:04d} | {time.time() - run_start:4.1f}s]  " + "  ||  ".join(reports))

            time.sleep(0.025)

        print("\n" + "=" * 75)
        print("--> SUCCESS: 15 seconds of live multi-instance driving completed!")
        print("=" * 75)

        # 3. Export batch CSV trajectories
        out_dir = Path("logs/trajectories")
        exported = track.save_all_trajectories(out_dir)
        for r_id, csv_path in exported.items():
            print(f"    Saved trajectory for Racer {r_id} -> {csv_path}")

    finally:
        print("\n--> Cleaning up and terminating simulator processes...")
        track.kill_all()
        print("Done.")


if __name__ == "__main__":
    main()
