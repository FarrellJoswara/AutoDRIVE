"""Compare fixed-step baseline and LiDAR scratch-buffer candidate."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/app")

from src.layer1.racer import Racer
from src.layer2.autodrive_env import AutoDriveEnv

def cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

def run(exe: Path, label: str, port: int, envs: int, actions: int, interval: float):
    os.environ["AICAR_MAP_ID"] = "porto"
    racers = []
    try:
        for i in range(envs):
            racers.append(Racer(racer_id=i, port=port+i, simulator_path=exe,
                                auto_launch=True, headless=True, enable_logging=False,
                                step_timeout=15.0, action_interval_s=interval))
        deadline = time.monotonic() + 90
        while not all(r.is_connected for r in racers):
            if time.monotonic() > deadline: raise TimeoutError("simulators did not connect")
            time.sleep(.05)
        def one(i):
            r = racers[i]
            r.reset()
            rows = []
            for k in range(actions):
                s = r.step(.22 if (k // 20) % 2 == 0 else .12,
                           .08 if (k // 30) % 2 == 0 else -.08)
                rows.append({"sim": r.simulation_time(), "ticks": r.physics_ticks,
                             "position": np.asarray(s.position, dtype=np.float64),
                             "orientation": np.asarray(s.orientation_quat, dtype=np.float64),
                             "velocity": np.asarray(s.linear_velocity, dtype=np.float64),
                             "angular": np.asarray(s.angular_velocity, dtype=np.float64),
                             "accel": np.asarray(s.linear_acceleration, dtype=np.float64),
                             "speed": float(s.true_speed), "collision": bool(s.collision),
                             "collisions": int(s.collision_count),
                             "lidar": np.asarray(s.lidar_ranges, dtype=np.float64)})
            return rows
        with concurrent.futures.ThreadPoolExecutor(max_workers=envs) as pool:
            list(pool.map(lambda r: r.reset(), racers))
            pids = [r._sim_process.pid for r in racers]
            c0 = sum(cpu_seconds(pid) for pid in pids); w0 = time.perf_counter()
            traces = list(pool.map(one, range(envs)))
            wall = time.perf_counter() - w0; cpu = sum(cpu_seconds(pid) for pid in pids) - c0
        sim = [float(x[-1]["sim"] - x[0]["sim"] + interval) for x in traces]
        ticks = [[int(x["ticks"]) for x in rows] for rows in traces]
        return {"label": label, "envs": envs, "actions": actions, "interval": interval,
                "simulated_seconds": sim, "physics_ticks_per_action": ticks,
                "wall_seconds": wall, "unity_cpu_seconds": cpu,
                "unity_cpu_seconds_per_simulated_second": cpu / sum(sim),
                "simulated_seconds_per_wall_second": sum(sim) / wall,
                "traces": traces}
    finally:
        for r in racers:
            try: r.kill()
            except Exception: pass

def safe(value):
    if isinstance(value, dict): return {k: safe(v) for k,v in value.items() if k != "traces"}
    if isinstance(value, list): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    return value

def main():
    p=argparse.ArgumentParser(); p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=4940); p.add_argument("--envs", type=int, default=4)
    p.add_argument("--actions", type=int, default=48); p.add_argument("--interval", type=float, default=.086)
    a=p.parse_args(); runs=[]
    for rep in range(2):
        order=[("baseline",a.baseline),("candidate",a.candidate)] if rep%2==0 else [("candidate",a.candidate),("baseline",a.baseline)]
        for j,(label,exe) in enumerate(order):
            run_result=run(exe,label,a.port,a.envs,a.actions,a.interval)
            print(json.dumps(safe(run_result)), flush=True); runs.append(run_result)
    base, cand = next((x, y) for x, y in zip(runs[::2], runs[1::2]) if x["label"] == "baseline")
    fields = ["position", "orientation", "velocity", "angular", "accel", "speed", "collision", "collisions", "lidar"]
    behavior = {}
    for field in fields:
        per_env = []
        exact = []
        for a, b in zip(base["traces"], cand["traces"]):
            av = np.asarray([row[field] for row in a], dtype=np.float64)
            bv = np.asarray([row[field] for row in b], dtype=np.float64)
            finite = np.isfinite(av) & np.isfinite(bv)
            per_env.append(float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else 0.0)
            exact.append(bool(np.array_equal(av, bv, equal_nan=True)))
        behavior[field] = {"max_abs_error_max": max(per_env), "exact_all_envs": all(exact)}
    report={"benchmark":"fixed action protocol LiDAR scratch-buffer A/B",
            "conditions":["same map, interval, actions, env count, physics timestep"],
            "behavior_first_pair": behavior, "runs":[safe(x) for x in runs]}
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
if __name__ == "__main__": main()
