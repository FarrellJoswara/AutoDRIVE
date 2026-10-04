"""Record an isolated fixed-step Unity Development Player to a .raw profile."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from test_simulator_build_equivalence import run_one


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-path", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=4960)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--interval", type=float, default=0.086)
    parser.add_argument("--frames", type=int, default=600)
    args = parser.parse_args()

    output = args.raw_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    original_popen = subprocess.Popen
    profiled_launches = 0

    def profiled_popen(command, *popen_args, **popen_kwargs):
        nonlocal profiled_launches
        if isinstance(command, (list, tuple)) and command and Path(str(command[0])).resolve() == args.simulator_path.resolve():
            command = list(command) + [
                "-profiler-log-file", str(output),
                "-profiler-capture-frame-count", str(args.frames),
                "-profiler-maxusedmemory", "67108864",
            ]
            profiled_launches += 1
        return original_popen(command, *popen_args, **popen_kwargs)

    subprocess.Popen = profiled_popen
    try:
        run = run_one(
            label="cpu-timeline", simulator=args.simulator_path.resolve(),
            port=args.port, repeat=0, seconds=args.seconds,
            interval=args.interval, map_id="porto", seed=1729,
        )
    finally:
        subprocess.Popen = original_popen

    if profiled_launches != 1:
        raise RuntimeError(f"expected one profiled player launch, got {profiled_launches}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Unity did not write a profile capture to {output}")
    summary = {
        "label": run["label"],
        "simulated_seconds": run["simulated_seconds"],
        "actions": run["action_count"],
        "physics_ticks": run["physics_ticks_delta"],
        "wall_seconds": run["wall_seconds"],
        "unity_cpu_seconds": run["unity_cpu_seconds"],
        "raw_profile": str(output),
        "raw_profile_bytes": output.stat().st_size,
        "requested_profile_frames": args.frames,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
