#!/usr/bin/env python3
"""Bounded native Linux practice smoke through the Mission Control manager.

Without flags, inspect an existing ROS sensor stream. --launch-preflight
briefly runs the attested official player and bridge under their launcher.
--run performs that sensor check before creating a bounded CUDA PPO practice
run through the manager. No training starts before fresh ROS sensors pass.
--replay uses normal competitive Replay and monitors observed policy steps
and elapsed time before calling the existing Stop endpoint.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pathlib
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE = pathlib.Path(os.environ.get("AICAR_NATIVE_STATE", ROOT / ".native-linux")).resolve()
RUNTIME = pathlib.Path(os.environ.get("AICAR_NATIVE_RUNTIME", STATE / "runtime")).resolve()
EXPECTED_DIGEST = "sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76"


def require_runtime_identity() -> dict:
    path = RUNTIME / "IMAGE-IDENTITY.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"official runtime identity is unavailable: {path}: {exc}") from exc
    required = {
        "repository": "autodriveecosystem/autodrive_roboracer_sim",
        "tag": "2026-iros-practice",
        "digest": EXPECTED_DIGEST,
        "architecture": "linux/amd64",
        "manifest_verified": True,
    }
    mismatches = [f"{key}={payload.get(key)!r} (expected {value!r})"
                  for key, value in required.items() if payload.get(key) != value]
    if not payload.get("rootfs_tree_digest"):
        mismatches.append("rootfs_tree_digest is missing")
    for key in ("simulator_sha256", "simulator_path"):
        if not payload.get(key):
            mismatches.append(f"{key} is missing")
    if mismatches:
        raise RuntimeError("official runtime identity is not complete: " + "; ".join(mismatches))
    rootfs = RUNTIME / "rootfs"
    simulator = rootfs / str(payload["simulator_path"])
    if not simulator.is_file() or not os.access(simulator, os.X_OK):
        raise RuntimeError(f"attested simulator is absent or not executable: {simulator}")
    return payload


def collect_ros_samples(timeout_s: float, max_age_s: float) -> dict:
    """Require recent nonblack camera, finite LiDAR and finite pose ROS data."""
    try:
        import rclpy
        from geometry_msgs.msg import Point
        from sensor_msgs.msg import Image, LaserScan
        from std_msgs.msg import Float32
    except Exception as exc:
        raise RuntimeError(f"ROS Python message imports failed: {type(exc).__name__}: {exc}") from exc

    topics = {
        "camera": os.environ.get("AICAR_SMOKE_CAMERA_TOPIC", "/autodrive/roboracer_1/front_camera"),
        "lidar": os.environ.get("AICAR_SMOKE_LIDAR_TOPIC", "/autodrive/roboracer_1/lidar"),
        "pose": os.environ.get("AICAR_SMOKE_POSE_TOPIC", "/autodrive/roboracer_1/ips"),
    }
    state: dict[str, dict] = {}
    history: dict[str, list[dict]] = {key: [] for key in topics}
    rclpy.init(args=None)
    node = rclpy.create_node("aicar_native_smoke_sensor_gate")
    throttle_pub = node.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", 10)
    steering_pub = node.create_publisher(Float32, "/autodrive/roboracer_1/steering_command", 10)

    def received(kind: str, msg, values: dict) -> None:
        row = dict(values, at=time.monotonic(), received_unix_ns=time.time_ns(),
                   received_monotonic_ns=time.monotonic_ns())
        if hasattr(msg, "header"):
            stamp = msg.header.stamp
            row["ros_stamp_ns"] = stamp.sec * 1_000_000_000 + stamp.nanosec
            row["header_age_at_receive_s"] = (node.get_clock().now().nanoseconds - row["ros_stamp_ns"]) / 1e9
        else:
            row["stamp_note"] = "geometry_msgs/Point has no header; freshness uses live volatile ROS receipt"
        state[kind] = row
        history[kind].append(row)
        history[kind] = history[kind][-5:]

    def camera(msg: Image) -> None:
        data = bytes(msg.data)
        nonblack = any(data[index] > 8 for index in range(0, len(data), max(1, len(data) // 4096)))
        received("camera", msg, {"width": msg.width, "height": msg.height,
                 "encoding": msg.encoding, "nonblack": nonblack,
                 "sha256": hashlib.sha256(data).hexdigest()})

    def lidar(msg: LaserScan) -> None:
        finite = [v for v in msg.ranges if math.isfinite(v) and msg.range_min <= v <= msg.range_max]
        received("lidar", msg, {"valid_ranges": len(finite), "range_count": len(msg.ranges)})

    def pose(msg: Point) -> None:
        received("pose", msg, {"x": float(msg.x), "y": float(msg.y), "z": float(msg.z),
                              "finite": all(math.isfinite(v) for v in (msg.x, msg.y, msg.z))})

    node.create_subscription(Image, topics["camera"], camera, 10)
    node.create_subscription(LaserScan, topics["lidar"], lidar, 10)
    node.create_subscription(Point, topics["pose"], pose, 10)
    def send_neutral() -> None:
        throttle_pub.publish(Float32(data=0.0))
        steering_pub.publish(Float32(data=0.0))
    node.create_timer(0.1, send_neutral)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            valid = (
                "camera" in state and state["camera"]["nonblack"]
                and state["camera"]["width"] > 0 and state["camera"]["height"] > 0
                and "lidar" in state and state["lidar"]["valid_ranges"] > 0
                and "pose" in state and state["pose"]["finite"]
                and all(now - row["at"] <= max_age_s for row in state.values())
                and all(len(rows) >= 2 for rows in history.values())
                and all(history[kind][-1]["ros_stamp_ns"] > history[kind][0]["ros_stamp_ns"] > 0
                        for kind in ("camera", "lidar"))
                and all(-0.5 <= state[kind]["header_age_at_receive_s"] <= max_age_s
                        for kind in ("camera", "lidar"))
            )
            if valid:
                return {"topics": topics, "samples": {
                    name: {key: value for key, value in row.items() if key != "at"}
                    for name, row in state.items()}, "max_sample_age_s": max_age_s,
                    "sample_history": {kind: [dict(
                        {key: value for key, value in row.items() if key != "at"},
                        receipt_age_at_gate_s=now-row["at"])
                        for row in rows] for kind, rows in history.items()}}
        raise RuntimeError(f"fresh nonblack camera, valid LiDAR, and finite pose not all received; latest={json.dumps(state)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _free_port(start: int, end: int, *, display_safe: bool = False) -> int:
    for port in range(start, end):
        if display_safe:
            display = 100 + ((port - 49152) % 30000)
            if pathlib.Path(f"/tmp/.X11-unix/X{display}").exists():
                continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"no free native preflight port in {start}:{end}")


def launch_preflight(max_age_s: float, sensor_timeout_s: float) -> dict:
    """Run the official bridge, relay and player briefly, always cleaning up."""
    import subprocess
    import uuid

    from src.layer4.hub.native_linux_official_backend import NativeLinuxOfficialBackend

    hub = os.environ.get("HUB_URL", "http://127.0.0.1:8090").rstrip("/")
    runs = request_json(hub + "/train/runs").get("runs", [])
    active = [row for row in runs if row.get("state") in {"starting", "running", "stopping"}]
    if active:
        raise RuntimeError("manager already has active training runs; refusing a second simulator: " +
                           ", ".join(str(row.get("run_id")) for row in active))
    lock_path = RUNTIME / ".native-runtime-smoke.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another native runtime smoke owns {lock_path}") from exc
        # Also detect the manager's durable player manifests: the integration
        # probe must not share its X display or official simulator instance.
        for manifest in (ROOT / "logs" / "rl").glob("official_run_*/linux_simulator_processes.json"):
            try:
                entries = json.loads(manifest.read_text()).get("processes", [])
            except (OSError, ValueError, TypeError):
                continue
            if entries:
                raise RuntimeError(f"manager simulator manifest is present; refusing display overlap: {manifest}")

        backend = NativeLinuxOfficialBackend()
        backend.validate_linux_runtime({"track_profile": "practice"})
        run_id = "preflight-" + uuid.uuid4().hex[:12]
        run_dir = ROOT / "logs" / "rl" / f"official_run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        sim_port = _free_port(49152, 65535, display_safe=True)
        relay_port = sim_port
        domain = backend._allocate_ros_domain()
        devkit_port = backend._allocate_devkit_port()
        env = os.environ.copy()
        env.update({"ROS_DOMAIN_ID": str(domain), "AICAR_ROS_DOMAIN_ID_BASE": str(domain),
                    "AICAR_RUN_BACKEND": "native_linux", "PYTHONNOUSERSITE": "1"})
        records: list[dict] = []
        connection = [{"environment": 0, "host": "127.0.0.1", "port": sim_port,
                       "ros_domain_id": domain}]
        try:
            backend._start_bridge_pair(records, run_id, 0, env, run_dir,
                                       relay_port=relay_port, ros_domain=domain,
                                       devkit_port=devkit_port)
            backend.start_simulators(run_id=run_id, connections=connection,
                                     hub_url=hub, config={"track_profile": "practice"})
            old_domain = os.environ.get("ROS_DOMAIN_ID")
            os.environ["ROS_DOMAIN_ID"] = str(domain)
            try:
                result = collect_ros_samples(sensor_timeout_s, max_age_s)
            finally:
                if old_domain is None:
                    os.environ.pop("ROS_DOMAIN_ID", None)
                else:
                    os.environ["ROS_DOMAIN_ID"] = old_domain
            result.update({"ros_domain_id": domain, "relay_port": relay_port,
                           "devkit_port": devkit_port, "simulator_port": sim_port,
                           "neutral_commands_published": True, "preflight_run_id": run_id})
            return result
        finally:
            backend.stop_simulators(run_id)
            for record in reversed(records):
                process = record.get("process")
                if process is None:
                    continue
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=8)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                try:
                    record["log"].close()
                except (KeyError, OSError):
                    pass
            backend._reserved_ros_domains.discard(domain)
            backend._reserved_devkit_ports.discard(devkit_port)
    finally:
        lock.close()


def request_json(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"hub {method} {url} returned HTTP {exc.code}: {body}") from exc


def run_replay_smoke(args, safe_neutral: float) -> int:
    """Use normal Replay plus its existing Stop API; never change BC gates."""
    from src.layer4.hub.linux_official_simulators import validate_linux_simulator_paths

    runtime, _, _ = validate_linux_simulator_paths({"track_profile": "competition"})
    identity = json.loads((runtime / "IMAGE-IDENTITY.json").read_text())
    if identity.get("tag") != "2026-iros-compete" or identity.get("manifest_verified") is not True:
        raise RuntimeError("normal replay is held until the competition selector resolves a verified 2026-iros-compete runtime")
    model = args.replay.resolve()
    try:
        model_id = model.relative_to((ROOT / "logs" / "rl").resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("replay checkpoint must be inside logs/rl") from exc
    if not model.is_file():
        raise RuntimeError(f"replay checkpoint is missing: {model}")
    hub = args.hub.rstrip("/")
    before = request_json(hub + "/api/replay/status")
    if before.get("state") in {"starting", "running", "queued", "stopping"}:
        raise RuntimeError("another replay is active; refusing to stop or replace it")
    request = {"model_id": model_id, "map_id": "none", "device": "cpu",
               "trace_steps": min(args.steps, 500)}
    result = {"purpose": "bounded normal competitive infrastructure smoke; no valid race score claim",
              "request": request, "runtime_identity": identity,
              "checkpoint_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "monitor_step_limit": args.steps, "monitor_wall_limit_s": args.wall_seconds,
              "trace_steps_is_only_trace_cap": True, "manual_gui_connect": False,
              "configured_environment_fallback_s": safe_neutral, "snapshots": []}
    args.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.state_file.write_text(json.dumps(result, indent=2) + "\n")

    save()
    started = time.monotonic()
    accepted = False
    steps_by_run: dict[str, int] = {}
    try:
        result["start_response"] = request_json(hub + "/api/replay/start", "POST", request, timeout=10)
        accepted = True
        save()
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= args.wall_seconds:
                result["stop_trigger"] = "monitor_wall_limit"
                break
            status = request_json(hub + "/api/replay/status", timeout=3)
            run_id = status.get("run_id")
            row = {"elapsed_s": elapsed, "status": status}
            if run_id:
                telemetry = request_json(hub + f"/train/runs/{run_id}/telemetry", timeout=3)
                fleet = telemetry.get("last_fleet") or telemetry.get("fleet") or {}
                steps_by_run[run_id] = max(steps_by_run.get(run_id, 0), int(fleet.get("step") or 0))
                row["telemetry"] = telemetry
                try:
                    import io
                    from PIL import Image
                    with urllib.request.urlopen(hub + f"/train/runs/{run_id}/camera/0", timeout=2) as response:
                        frame = response.read()
                    with Image.open(io.BytesIO(frame)) as image:
                        rgb = image.convert("RGB")
                        row["camera"] = {"size": list(rgb.size), "nonblack": max(v[1] for v in rgb.getextrema()) > 8,
                                         "sha256": hashlib.sha256(frame).hexdigest()}
                    frame_path = args.state_file.parent / f"{args.state_file.stem}-{run_id}-camera.jpg"
                    if not frame_path.exists():
                        frame_path.write_bytes(frame)
                except (OSError, ValueError) as exc:
                    row["camera_not_ready"] = str(exc)
            result["snapshots"].append(row)
            result["observed_steps_by_run"] = steps_by_run
            save()
            if sum(steps_by_run.values()) >= args.steps:
                result["stop_trigger"] = "observed_step_limit"
                break
            if status.get("state") in {"completed", "failed", "stopped"}:
                result["stop_trigger"] = "job_terminal_before_monitor_limit"
                break
            time.sleep(min(0.5, max(0.01, args.wall_seconds - (time.monotonic() - started))))
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if accepted:
            result["stop_requested_after_s"] = time.monotonic() - started
            try:
                result["stop_response"] = request_json(hub + "/api/replay/stop", "POST", {}, timeout=180)
            except Exception as exc:
                result["stop_error"] = f"{type(exc).__name__}: {exc}"
                save()
                raise
            save()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = request_json(hub + "/api/replay/status", timeout=3)
                result["final_status"] = status
                if status.get("state") in {"completed", "failed", "stopped", "idle"}:
                    break
                time.sleep(0.5)
            for run_id in steps_by_run:
                report = ROOT / "logs" / "rl" / f"official_run_{run_id}" / "evaluation.json"
                if report.is_file():
                    result.setdefault("raw_evaluation_reports", {})[run_id] = json.loads(report.read_text())
        result["finished_after_s"] = time.monotonic() - started
        save()
    if not any(steps_by_run.values()):
        raise RuntimeError("replay never exposed a completed policy/sensor step; see unique report")
    if not any(row.get("camera", {}).get("nonblack") for row in result["snapshots"]):
        raise RuntimeError("replay had no observed nonblack camera frame; see unique report")
    if result.get("final_status", {}).get("state") not in {"completed", "failed", "stopped", "idle"}:
        raise RuntimeError("Replay Stop did not reach a terminal state")
    print(json.dumps({"report": str(args.state_file), "stop_trigger": result.get("stop_trigger"),
                      "observed_steps_by_run": steps_by_run, "valid_score_claim": False}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default=os.environ.get("HUB_URL", "http://127.0.0.1:8090"))
    parser.add_argument("--run", action="store_true", help="create and run the bounded manager-owned practice smoke")
    parser.add_argument("--replay", type=pathlib.Path, help="normal competitive replay checkpoint; requires verified competition runtime and uses existing Replay Stop")
    parser.add_argument("--wall-seconds", type=float, default=90, help="replay monitor wall limit before requesting existing Replay Stop")
    parser.add_argument("--launch-preflight", action="store_true",
                        help="launch the attested official player + ROS bridge briefly for fresh sensor checks")
    parser.add_argument("--steps", type=int, default=2048, help="bounded PPO steps (must be a multiple of 64)")
    parser.add_argument("--resume", type=pathlib.Path, help="resume a saved checkpoint inside the persistent logs directory; --steps is the additional budget")
    parser.add_argument("--stop-after-rollouts", type=int, help="request manager Stop after this many newly saved 64-step rollouts, retaining the larger configured budget")
    parser.add_argument("--sensor-timeout", type=float, default=30)
    parser.add_argument("--max-age", type=float, default=1)
    parser.add_argument("--state-file", type=pathlib.Path)
    args = parser.parse_args()
    if args.replay is not None and (args.run or args.resume is not None or args.launch_preflight):
        parser.error("--replay cannot be combined with training or a separate preflight")
    if args.wall_seconds <= 0:
        parser.error("--wall-seconds must be positive")
    if args.state_file is None:
        args.state_file = RUNTIME / f"native-smoke-{time.time_ns()}-{os.getpid()}.json"
    if args.state_file.exists():
        parser.error(f"refusing to overwrite prior evidence: {args.state_file}")
    safe_neutral = float(os.environ.get("AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S", "0.20"))
    if not 0 <= safe_neutral < 0.25:
        parser.error("safe neutral fallback must be >=0 and <0.25 seconds")
    if args.steps < 64 or args.steps % 64:
        parser.error("--steps must be >=64 and a multiple of 64")
    if args.stop_after_rollouts is not None and not 1 <= args.stop_after_rollouts <= args.steps // 64:
        parser.error("--stop-after-rollouts must fit within the configured --steps budget")
    starting_steps = 0
    if args.resume is not None:
        args.resume = args.resume.resolve()
        if not args.resume.is_relative_to((ROOT / "logs").resolve()):
            parser.error("--resume must be inside the persistent logs directory")
        with zipfile.ZipFile(args.resume) as archive:
            starting_steps = int(json.loads(archive.read("data"))["num_timesteps"])
    if not os.environ.get("AICAR_NATIVE_PYTHON"):
        raise RuntimeError("source scripts/native-linux-env.sh first; AICAR_NATIVE_PYTHON is unset")
    if pathlib.Path(sys.executable).resolve() != pathlib.Path(os.environ["AICAR_NATIVE_PYTHON"]).resolve():
        raise RuntimeError("invoke this script with $AICAR_NATIVE_PYTHON")
    if args.replay is not None:
        return run_replay_smoke(args, safe_neutral)
    identity = require_runtime_identity()
    hub = args.hub.rstrip("/")
    os.environ["HUB_URL"] = hub
    health = request_json(hub + "/health")
    result = {"runtime_digest": identity["digest"], "hub_health": health,
              "training_started": False, "started_unix_ns": time.time_ns(),
              "report_path": str(args.state_file.resolve()),
              "python": sys.executable, "ros_setup": os.environ.get("AICAR_NATIVE_ROS_SETUP"),
              "ros_workspace": os.environ.get("AICAR_NATIVE_ROS_WORKSPACE"),
              "safe_neutral_environment": os.environ.get("AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S"),
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ("src/layer1/ros2_racer.py", "src/layer2/official_race_env.py",
                                             "scripts/native-linux-runner.py", "scripts/native-linux-smoke.py")},
              "requested_additional_steps": args.steps, "starting_steps": starting_steps,
              "stop_after_rollouts": args.stop_after_rollouts,
              "safe_neutral_fallback_s": safe_neutral, "manual_gui_connect": False,
              "resume": str(args.resume) if args.resume is not None else None}
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.state_file.write_text(json.dumps(result, indent=2) + "\n")
        if result.get("run_id"):
            (ROOT / "logs" / "rl" / f"official_run_{result['run_id']}" / "native_smoke_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        result["ros_preflight"] = (launch_preflight(args.max_age, args.sensor_timeout)
                                   if args.launch_preflight or args.run
                                   else collect_ros_samples(args.sensor_timeout, args.max_age))
    except Exception as exc:
        result["preflight_error"] = f"{type(exc).__name__}: {exc}"
        save()
        print(f"Failure evidence: {args.state_file}", file=sys.stderr)
        raise
    save()
    if args.run:
        created = request_json(hub + "/train/runs", "POST", {
            "display_name": "Native Linux bounded practice smoke",
            "config": {"simulator_backend": "linux_external", "track_profile": "practice",
                       "observation_profile": "official_sensors_camera", "n_envs": 1,
                       "total_timesteps": args.steps, "n_steps": 64, "device": "cuda",
                       "safe_neutral_fallback_s": safe_neutral,
                       "automatic_evaluation_enabled": False, "behavior_cloning_enabled": False,
                       "behavior_cloning_only": False, "stop_on_plateau": False,
                       "checkpoint_every": 64, "max_duration_hours": 0.25,
                       "resume": str(args.resume) if args.resume is not None else None}
        }, timeout=300)
        run_id = created["run_id"]
        result.update(training_started=True, run_id=run_id, creation=created)
        save()
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                run = request_json(hub + f"/train/runs/{run_id}")
                result["latest_run"] = run
                save()
                print(f"run={run_id} state={run.get('state')}", flush=True)
                if run.get("state") not in {"starting", "running", "stopping"}:
                    if run.get("state") == "failed":
                        raise RuntimeError(f"manager smoke failed: {run}")
                    break
                if args.stop_after_rollouts is not None and not result.get("active_stop_requested"):
                    threshold = starting_steps + args.stop_after_rollouts * 64
                    checkpoints_dir = ROOT / "logs" / "rl" / f"official_run_{run_id}" / "checkpoints"
                    saved_steps = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints_dir.glob("post_optimizer_update_*_steps_*.zip")]
                    if saved_steps and max(saved_steps) >= threshold:
                        result["active_stop_requested"] = {"checkpoint_steps": max(saved_steps), "threshold": threshold}
                        save()
                        result["active_stop_result"] = request_json(hub + f"/train/runs/{run_id}/stop", "POST", {}, timeout=180)
                        save()
                time.sleep(1 if args.stop_after_rollouts is not None else 5)
            else:
                raise TimeoutError("15 minute smoke deadline exceeded")
        finally:
            result["stop_result"] = request_json(hub + f"/train/runs/{run_id}/stop", "POST", {}, timeout=180)
            result["checkpoints"] = [str(p) for p in (ROOT / "logs" / "rl" / f"official_run_{run_id}").rglob("*.zip")]
            save()
        verified = []
        for checkpoint in result["checkpoints"]:
            with zipfile.ZipFile(checkpoint) as archive:
                data = json.loads(archive.read("data"))
            verified.append({"path": checkpoint, "bytes": pathlib.Path(checkpoint).stat().st_size,
                             "num_timesteps": data.get("num_timesteps"),
                             "n_updates": data.get("_n_updates")})
        result["checkpoint_metadata"] = verified
        save()
        expected_steps = starting_steps + (args.stop_after_rollouts * 64 if args.stop_after_rollouts is not None else args.steps)
        if not any(int(row.get("num_timesteps") or 0) >= expected_steps for row in verified):
            raise RuntimeError("no saved checkpoint reached the requested smoke step count")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"native-linux-smoke: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
