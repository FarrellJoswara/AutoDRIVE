"""Measure camera omission against Unity's actual simulated time per Bridge.

Run only against an isolated diagnostic player that emits ``AICAR Probe Fixed
Time`` and ``AICAR Probe Frame``. This does not start RL training or save models.
It preserves Racer's normal asynchronous command exchange, so measured deltas
are response-to-response intervals, not a claimed action acknowledgement.

The player must support AICAR_DISABLE_CAMERA_STREAM. Both modes use the same
binary, actions, ports, and number of concurrent processes. No CPU limit, frame
cap, physics configuration, or sensor setting is introduced by this harness.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.layer1.racer import Racer
from src.layer1.telemetry import TelemetrySnapshot


_parse_snapshot = TelemetrySnapshot.from_raw_dict


def _parse_probe_snapshot(cls, raw, step_id=0):
    snap = _parse_snapshot(raw, step_id=step_id)
    # Attach to this exact snapshot before Racer publishes it to the caller.
    snap.probe_fixed_time = float(raw.get("AICAR Probe Fixed Time", "nan"))
    snap.probe_frame = int(raw.get("AICAR Probe Frame", -1))
    snap.probe_camera_keys = tuple(sorted(k for k in raw if "Camera Image" in k))
    return snap


def _cpu_seconds(pid: int) -> float:
    # Unity is a direct child in Linux/Docker, rather than a separate container.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _player_hashes(executable: Path):
    candidates = [executable, executable.parent / "GameAssembly.so", executable.parent / "UnityPlayer.so"]
    for data_name in ("AutoDRIVE Simulator_Data", "Data"):
        candidates.append(executable.parent / data_name / "globalgamemanagers")
    hashes = {}
    for path in candidates:
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def run_trial(args, camera_off: bool, repeat: int):
    previous_flag = os.environ.get("AICAR_DISABLE_CAMERA_STREAM")
    if camera_off:
        os.environ["AICAR_DISABLE_CAMERA_STREAM"] = "1"
    else:
        os.environ.pop("AICAR_DISABLE_CAMERA_STREAM", None)
    racers = []
    try:
        for i in range(args.n_envs):
            racers.append(Racer(
                racer_id=i, port=args.base_port + i,
                simulator_path=args.simulator, auto_launch=True, headless=True,
                enable_logging=False, step_timeout=15.0,
            ))
        deadline = time.monotonic() + 90.0
        while not all(r.is_connected for r in racers):
            if time.monotonic() >= deadline:
                raise TimeoutError("Probe players did not connect within 90 seconds")
            time.sleep(0.05)
        with ThreadPoolExecutor(max_workers=args.n_envs) as pool:
            list(pool.map(lambda r: r.reset(), racers))
            previous = list(pool.map(lambda r: r.step(0.0, 0.0), racers))
            if not all(math.isfinite(s.probe_fixed_time) for s in previous):
                raise ValueError("Player lacks diagnostic fixed-time telemetry; use a probe build")
            traces = [[] for _ in racers]
            pids = [r._sim_process.pid for r in racers]
            cpu_before = sum(_cpu_seconds(pid) for pid in pids)
            start = time.perf_counter()
            for k in range(args.steps):
                command = (0.20, 0.12 * math.sin(k / 35.0))
                current = list(pool.map(lambda r: r.step(*command), racers))
                for i, (old, new) in enumerate(zip(previous, current)):
                    traces[i].append({
                        "fixed_time_delta_s": new.probe_fixed_time - old.probe_fixed_time,
                        "frame_delta": new.probe_frame - old.probe_frame,
                        "bridge_id_delta": new.step_id - old.step_id,
                        "camera_keys": new.probe_camera_keys,
                    })
                previous = current
            wall = time.perf_counter() - start
            cpu = sum(_cpu_seconds(pid) for pid in pids) - cpu_before
        result = {
            "camera_off": camera_off, "repeat": repeat, "n_envs": args.n_envs,
            "steps_each": args.steps, "wall_s": wall,
            "aggregate_steps_per_s": args.n_envs * args.steps / wall,
            "unity_cpu_s": cpu, "unity_cores_used": cpu / wall,
            "per_environment": [],
        }
        for trace in traces:
            result["per_environment"].append({
                "fixed_time_per_returned_step_s": _distribution([t["fixed_time_delta_s"] for t in trace]),
                "fixed_time_per_bridge_mean_s": sum(t["fixed_time_delta_s"] for t in trace) / sum(t["bridge_id_delta"] for t in trace),
                "frames_per_returned_step": _distribution([t["frame_delta"] for t in trace]),
                "bridge_ids_per_step": _distribution([t["bridge_id_delta"] for t in trace]),
                "observed_camera_keys": sorted({key for t in trace for key in t["camera_keys"]}),
            })
        return result
    finally:
        for racer in racers:
            racer.kill()
        if previous_flag is None:
            os.environ.pop("AICAR_DISABLE_CAMERA_STREAM", None)
        else:
            os.environ["AICAR_DISABLE_CAMERA_STREAM"] = previous_flag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=4580)
    args = parser.parse_args()
    if os.name != "posix" or not args.simulator.is_file():
        parser.error("Use Linux/Docker with an existing isolated diagnostic player")
    if min(args.n_envs, args.steps, args.repeats) < 1:
        parser.error("n-envs, steps, and repeats must be positive")
    hashes = _player_hashes(args.simulator)
    results = []
    original_parser = TelemetrySnapshot.__dict__["from_raw_dict"]
    TelemetrySnapshot.from_raw_dict = classmethod(_parse_probe_snapshot)
    try:
        for repeat in range(1, args.repeats + 1):
            modes = (False, True) if repeat % 2 else (True, False)
            for camera_off in modes:
                result = run_trial(args, camera_off, repeat)
                results.append(result)
                print(json.dumps(result), flush=True)
    finally:
        TelemetrySnapshot.from_raw_dict = original_parser
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "simulator": str(args.simulator),
        "player_sha256": hashes,
        "measurement": "Actual Unity fixed-time delta between Racer responses; camera modes share one binary",
        "workload": "Independent Unity children; one Python process submits concurrent Racer.step calls with a thread pool; no policy inference",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
