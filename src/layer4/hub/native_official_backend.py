"""Native WSL process backend for official Windows AutoDRIVE training.

The simulator remains the unmodified Windows GUI build. Each environment gets
its own ROS domain, official Devkit bridge, and Layer 1 relay; one native PPO
process owns the shared learner.
"""

from __future__ import annotations

import os
import json
import signal
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from src.layer4.settings import ROOT


class _RecoveredProcess:
    """Small Popen-compatible handle for a child adopted after hub restart."""

    def __init__(self, pid: int):
        self.pid = int(pid)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        try:
            state = Path(f"/proc/{self.pid}/stat").read_text().split()[2]
            if state != "Z":
                return None
        except (OSError, IndexError):
            pass
        self.returncode = self.returncode or 0
        return self.returncode

    def terminate(self) -> None:
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("recovered process", timeout)
            time.sleep(0.05)
        return int(self.returncode or 0)


def _signal_process_group(process: Any, sig: int) -> None:
    try:
        os.killpg(int(process.pid), sig)
    except (ProcessLookupError, PermissionError):
        if sig == signal.SIGTERM:
            try:
                process.terminate()
            except (ProcessLookupError, OSError):
                pass


class NativeOfficialBackend:
    """Drop-in lifecycle backend used by OfficialRunManager without Docker."""

    def __init__(self, *, ros_workspace: Path | None = None, startup_timeout_s: float = 45.0):
        self.ros_workspace = Path(ros_workspace or Path.home() / "autodrive_ws")
        self.startup_timeout_s = float(startup_timeout_s)
        self.simulator_host = subprocess.run(
            ["hostname", "-I"], check=True, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", timeout=10,
        ).stdout.split()[0]
        self._lock = threading.RLock()
        # Native ROS bridges share the host network namespace. Serialize their
        # startup so concurrent runs cannot pick the same free port/domain.
        self._startup_lock = threading.Lock()
        self._groups: dict[str, dict[str, Any]] = {}
        self._reserved_ros_domains: set[int] = set()
        self._reserved_devkit_ports: set[int] = set()
        self._devkit_port_start = 4567
        self._networks: dict[str, dict[str, Any]] = {}
        self._windows_launchers: dict[str, subprocess.Popen] = {}
        self._recover_groups()

    def _recover_groups(self) -> None:
        for manifest in (ROOT / "logs" / "rl").glob("official_run_*/native_group_*.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                records = []
                for item in payload["records"]:
                    log_path = Path(item["log"])
                    records.append({
                        "role": item["role"],
                        "process": _RecoveredProcess(int(item["pid"])),
                        "log": log_path.open("ab", buffering=0) if log_path.exists() else open(os.devnull, "ab"),
                    })
                group = {**payload, "records": records, "manifest": manifest}
                self._groups[group["Id"]] = group
                run_id = group["Labels"].get("aicar.run_id")
                if run_id:
                    network_id = f"native-{run_id}"
                    self._networks[network_id] = {
                        "Id": network_id, "Name": f"aicar-run-{run_id}",
                        "Labels": {"aicar.run_id": run_id},
                    }
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _native_policy_command(self, command: list[str]) -> list[str]:
        """Run the policy process with the same ROS setup as its bridge pair."""
        setup = (
            "set +u; source /opt/ros/jazzy/setup.bash; "
            f"source {shlex.quote(str(self.ros_workspace / 'install' / 'setup.bash'))}; "
            'set -u; exec "$@"'
        )
        return ["bash", "-lc", setup, "aicar-native-policy", *command]

    @staticmethod
    def _persist_group(group: dict[str, Any]) -> None:
        run_id = group["Labels"]["aicar.run_id"]
        role = group["Labels"]["aicar.role"]
        run_dir = ROOT / "logs" / "rl" / f"official_run_{run_id}"
        manifest = run_dir / f"native_group_{role}.json"
        payload = {key: group[key] for key in ("Id", "Names", "Labels")}
        for key in ("ros_domains", "devkit_ports"):
            if key in group:
                payload[key] = group[key]
        payload["records"] = [
            {"role": item["role"], "pid": item["process"].pid, "log": str(Path(item["log"].name))}
            for item in group["records"]
        ]
        temp = manifest.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temp, manifest)
        group["manifest"] = manifest

    @staticmethod
    def _labels(run_id: str, role: str, metadata: dict[str, str]) -> dict[str, str]:
        return {**metadata, "aicar.run_id": run_id, "aicar.role": role}

    def find_host_path_for_container_path(self, path: Path) -> str:
        return str(Path(path).resolve())

    def create_official_run_network(self, run_id: str) -> str:
        network_id = f"native-{run_id}"
        with self._lock:
            self._networks[network_id] = {
                "Id": network_id,
                "Name": f"aicar-run-{run_id}",
                "Labels": {"aicar.run_id": run_id},
            }
        return network_id

    def remove_official_run_network(self, network_id: str) -> None:
        with self._lock:
            self._networks.pop(network_id, None)

    @staticmethod
    def _training_command(env: dict[str, str]) -> list[str]:
        """Translate the manager's validated environment contract to CLI args."""
        def value(name: str, default: str) -> str:
            return env.get(name, default)

        command = [
            sys.executable, "-m", "src.layer3.official_train",
            "--out", value("AICAR_TRAIN_OUT", "/runs/official"),
            "--n-envs", value("AICAR_TRAIN_N_ENVS", "1"),
            "--total-timesteps", value("AICAR_TRAIN_TIMESTEPS", "1000000"),
            "--stop-on-plateau", value("AICAR_TRAIN_STOP_ON_PLATEAU", "1"),
            "--plateau-min-timesteps", value("AICAR_TRAIN_PLATEAU_MIN_TIMESTEPS", "100000"),
            "--plateau-window-episodes", value("AICAR_TRAIN_PLATEAU_WINDOW_EPISODES", "5"),
            "--plateau-patience", value("AICAR_TRAIN_PLATEAU_PATIENCE", "5"),
            "--plateau-min-improvement-pct", value("AICAR_TRAIN_PLATEAU_MIN_IMPROVEMENT_PCT", "1.0"),
            "--max-duration-hours", value("AICAR_TRAIN_MAX_DURATION_HOURS", "0"),
            "--seed", value("AICAR_TRAIN_SEED", "0"),
            "--device", value("AICAR_PPO_DEVICE", "cpu"),
            "--checkpoint-every", value("AICAR_TRAIN_CHECKPOINT_EVERY", "10000"),
            "--automatic-evaluation-enabled", value("AICAR_AUTO_EVALUATION_ENABLED", "1"),
            "--automatic-evaluation-trigger", value("AICAR_AUTO_EVALUATION_TRIGGER", "steps"),
            "--automatic-evaluation-interval", value("AICAR_AUTO_EVALUATION_INTERVAL", "50000"),
            "--n-steps", value("AICAR_PPO_N_STEPS", "2048"),
            "--learning-rate", value("AICAR_PPO_LEARNING_RATE", "0.0001"),
            "--n-epochs", value("AICAR_PPO_N_EPOCHS", "4"),
            "--gamma", value("AICAR_PPO_GAMMA", "0.9995"),
            "--gae-lambda", value("AICAR_PPO_GAE_LAMBDA", "0.95"),
            "--timeout-s", value("AICAR_SENSOR_TIMEOUT_S", "180"),
            "--training-timeout-s", value("AICAR_TRAIN_WATCHDOG_S", "600"),
            "--race-laps", value("AICAR_TRAIN_RACE_LAPS", "10"),
            "--warmup-laps", value("AICAR_TRAIN_WARMUP_LAPS", "0"),
            "--time-cost-per-simulated-second", value("AICAR_TRAIN_TIME_COST", "5"),
            "--frontier-stagnation-s", value("AICAR_TRAIN_FRONTIER_STAGNATION_S", "10"),
            "--collision-penalty-magnitude", value("AICAR_TRAIN_COLLISION_PENALTY_MAGNITUDE", "100"),
            "--collision-reward-percent", value("AICAR_TRAIN_COLLISION_REWARD_PERCENT", "50"),
            "--failed-episode-penalty", value("AICAR_TRAIN_FAILURE_PENALTY", "100"),
            "--backward-speed-penalty-scale", value("AICAR_TRAIN_BACKWARD_SPEED_PENALTY_SCALE", "10"),
            "--episode-failure-reward-percent", value("AICAR_TRAIN_FAILURE_REWARD_PERCENT", "50"),
            "--observation-profile", value("AICAR_OBSERVATION_PROFILE", "official_sensors_camera"),
            "--steering-action-scale", value("AICAR_STEERING_ACTION_SCALE", "1.0"),
            "--straight-throttle-gain", value("AICAR_STRAIGHT_THROTTLE_GAIN", "1.0"),
            "--straight-throttle-steering-threshold", value("AICAR_STRAIGHT_THROTTLE_STEERING_THRESHOLD", "0.15"),
            "--negative-throttle-mode", value("AICAR_NEGATIVE_THROTTLE_MODE", "allow"),
            "--steering-mode", value("AICAR_STEERING_MODE", "normal"),
            "--throttle-mode", value("AICAR_THROTTLE_MODE", "bidirectional"),
        ]
        resume = value("AICAR_TRAIN_RESUME", "")
        if resume:
            command.extend(["--resume", resume])
        return command

    @staticmethod
    def _evaluation_command(env: dict[str, str]) -> list[str]:
        return [
            sys.executable, "-m", "src.layer3.official_evaluate",
            "--model", env.get("AICAR_MODEL_PATH", ""),
            "--controller", env.get("AICAR_CONTROLLER", "ppo"),
            "--attempts", env.get("AICAR_EVALUATION_ATTEMPTS", "1"),
            "--device", "cpu",
            "--timeout-s", env.get("AICAR_SENSOR_TIMEOUT_S", "180"),
            "--wall-timeout-s", env.get("AICAR_RACE_WALL_TIMEOUT_S", "1800"),
            "--max-steps", env.get("AICAR_RACE_STEP_GUARD", "150000"),
            "--observation-profile", env.get("AICAR_OBSERVATION_PROFILE", "official_sensors_camera"),
            "--negative-throttle-mode", env.get("AICAR_NEGATIVE_THROTTLE_MODE", "allow"),
            "--steering-mode", env.get("AICAR_STEERING_MODE", "normal"),
            "--throttle-mode", env.get("AICAR_THROTTLE_MODE", "bidirectional"),
            "--steering-action-scale", env.get("AICAR_STEERING_ACTION_SCALE", "1.0"),
            "--straight-throttle-gain", env.get("AICAR_STRAIGHT_THROTTLE_GAIN", "1.0"),
            "--straight-throttle-steering-threshold", env.get("AICAR_STRAIGHT_THROTTLE_STEERING_THRESHOLD", "0.15"),
            "--out", env.get("AICAR_EVALUATION_OUTPUT", "/tmp/aicar-iros-evaluation.json"),
            "--trace-steps", env.get("AICAR_EVALUATION_TRACE_STEPS", "0"),
        ]

    def create_official_run_container(
        self,
        *,
        run_id: str,
        role: str,
        image: str,
        network_name: str,
        env: list[str],
        metadata_labels: dict[str, str],
        published_ports: dict[str, int] | None = None,
        **_ignored: Any,
    ) -> str:
        child_env = os.environ.copy()
        child_env.update(dict(item.split("=", 1) for item in env if "=" in item))
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["AICAR_SIMULATOR_BACKEND"] = "windows_external"
        run_dir = Path(metadata_labels["aicar.output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        allocated_domain: int | None = None
        allocated_port: int | None = None
        try:
            with self._startup_lock:
                if role == "api" or role.startswith("api-"):
                    index = 0 if role == "api" else int(role.rsplit("-", 1)[1])
                    relay_port = int((published_ports or {}).get("4568", 49152 + index))
                    allocated_domain = self._allocate_ros_domain()
                    allocated_port = self._allocate_devkit_port()
                    # The bridge and relay share the allocated domain, and the
                    # policy/evaluator process must join that same domain to
                    # receive their ROS topics. Previously only the bridge
                    # pair inherited this value, leaving evaluation clients on
                    # the default domain where no simulator topics exist.
                    child_env["ROS_DOMAIN_ID"] = str(allocated_domain)
                    self._start_bridge_pair(
                        records, run_id, index, child_env, run_dir,
                        relay_port=relay_port, ros_domain=allocated_domain,
                        devkit_port=allocated_port,
                    )
                    if role == "api":
                        self._localize_runtime_paths(child_env)
                        hub_url = child_env.get("HUB_URL", "")
                        if "host.docker.internal" in hub_url:
                            child_env["HUB_URL"] = os.environ.get("AICAR_HUB_PUBLIC_URL", "http://127.0.0.1:8091")
                        train_log = (run_dir / "hub_train.log").open("ab", buffering=0)
                        mode = child_env.get("AICAR_MODE", "train").strip().lower()
                        command = (
                            self._evaluation_command(child_env)
                            if mode == "evaluate"
                            else self._training_command(child_env)
                        )
                        process = subprocess.Popen(
                            self._native_policy_command(command),
                            cwd=ROOT,
                            env=child_env,
                            stdin=subprocess.DEVNULL,
                            stdout=train_log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        records.append({"role": "trainer", "process": process, "log": train_log})
                else:
                    raise ValueError(f"unsupported native official role: {role}")
        except Exception:
            if allocated_domain is not None:
                self._reserved_ros_domains.discard(allocated_domain)
            if allocated_port is not None:
                self._reserved_devkit_ports.discard(allocated_port)
            for record in records:
                process = record["process"]
                _signal_process_group(process, signal.SIGTERM)
                try:
                    record["log"].close()
                except OSError:
                    pass
            raise

        identity = f"native-{run_id}-{role}-{uuid.uuid4().hex[:8]}"
        group = {
            "Id": identity,
            "Names": [f"/aicar-run-{run_id}-{role}"],
            "Labels": self._labels(run_id, role, metadata_labels),
            "State": "running",
            "records": records,
        }
        if allocated_domain is not None:
            group["ros_domains"] = [allocated_domain]
        if allocated_port is not None:
            group["devkit_ports"] = [allocated_port]
        with self._lock:
            self._groups[identity] = group
            if allocated_domain is not None:
                self._reserved_ros_domains.discard(allocated_domain)
            if allocated_port is not None:
                self._reserved_devkit_ports.discard(allocated_port)
            self._persist_group(group)
        return identity

    def _allocate_ros_domain(self) -> int:
        with self._lock:
            used = set()
            for group in self._groups.values():
                used.update(group.get("ros_domains", ()))
                if "ros_domains" not in group:
                    for record in group.get("records", ()):
                        role = str(record.get("role", ""))
                        if role.startswith("devkit-"):
                            try:
                                used.add(int(role.rsplit("-", 1)[1]))
                            except ValueError:
                                continue
            used.update(self._reserved_ros_domains)
            for domain in range(233):
                if domain not in used:
                    self._reserved_ros_domains.add(domain)
                    return domain
        raise RuntimeError("no free ROS domain IDs remain for native simulator bridges")

    def _allocate_devkit_port(self) -> int:
        with self._lock:
            used = {port for group in self._groups.values() for port in group.get("devkit_ports", ())}
            for group in self._groups.values():
                if "devkit_ports" not in group:
                    used.update(self._devkit_ports_from_records(group.get("records", ())))
            used.update(self._reserved_devkit_ports)
            for port in range(self._devkit_port_start, 60000):
                if port in used:
                    continue
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    try:
                        probe.bind(("0.0.0.0", port))
                    except OSError:
                        continue
                self._reserved_devkit_ports.add(port)
                return port
        raise RuntimeError("no free TCP ports remain for native Devkit bridges")

    @staticmethod
    def _devkit_ports_from_records(records: Any) -> set[int]:
        ports: set[int] = set()
        for record in records:
            role = str(record.get("role", ""))
            if not role.startswith("devkit-"):
                continue
            # Older manifests did not persist port assignments; reserve the
            # legacy per-environment range while those processes are adopted.
            try:
                ports.add(4567 + int(role.rsplit("-", 1)[1]))
            except ValueError:
                pass
        return ports

    @staticmethod
    def _local_logs_path(value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/runs/"):
            return str((ROOT / "logs" / normalized[len("/runs/"):]).resolve())
        if normalized.startswith("/app/logs/"):
            return str((ROOT / "logs" / normalized[len("/app/logs/"):]).resolve())
        return value

    @classmethod
    def _localize_runtime_paths(cls, env: dict[str, str]) -> None:
        """Translate all container-style persistent file paths for WSL processes."""
        for key in (
            "AICAR_TRAIN_OUT", "AICAR_EVALUATION_OUTPUT", "AICAR_TRAIN_RESUME",
            "AICAR_MODEL_PATH",
        ):
            if key in env:
                env[key] = cls._local_logs_path(env[key])

    def _start_bridge_pair(
        self, records: list[dict[str, Any]], run_id: str, index: int,
        env: dict[str, str], run_dir: Path, *, relay_port: int,
        ros_domain: int, devkit_port: int,
    ) -> None:
        domain = str(ros_domain)
        bridge_env = env.copy()
        bridge_env.update({
            "ROS_DOMAIN_ID": domain,
            "AICAR_DEVKIT_PORT": str(devkit_port),
            "PYTHONNOUSERSITE": "1",
        })
        bridge_log = (run_dir / f"devkit_env{index}.log").open("ab", buffering=0)
        setup = (
            "set +u; source /opt/ros/jazzy/setup.bash; "
            f"source '{self.ros_workspace}/install/setup.bash'; set -u; "
            "exec ros2 launch autodrive_roboracer bringup_headless.launch.py"
        )
        devkit = subprocess.Popen(
            ["bash", "-lc", setup], cwd=ROOT, env=bridge_env,
            stdin=subprocess.DEVNULL, stdout=bridge_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        records.append({"role": f"devkit-{index}", "process": devkit, "log": bridge_log})

        relay_env = env.copy()
        relay_env["ROS_DOMAIN_ID"] = domain
        relay_env["PYTHONNOUSERSITE"] = "1"
        relay_log = (run_dir / f"relay_env{index}.log").open("ab", buffering=0)
        relay = subprocess.Popen(
            [sys.executable, "-m", "src.layer1.official_bridge_relay",
             "--host", "0.0.0.0", "--port", str(relay_port),
             "--devkit-url", f"http://127.0.0.1:{devkit_port}"],
            cwd=ROOT, env=relay_env, stdin=subprocess.DEVNULL,
            stdout=relay_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        records.append({"role": f"relay-{index}", "process": relay, "log": relay_log})

        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            for record in records:
                process = record["process"]
                if process.poll() is not None:
                    try:
                        detail = Path(record["log"].name).read_text(encoding="utf-8", errors="replace")[-3000:]
                    except OSError:
                        detail = ""
                    raise RuntimeError(
                        f"{record['role']} exited during startup with code {process.returncode}: {detail}"
                    )
            try:
                with socket.create_connection(("127.0.0.1", relay_port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError(f"native official bridge for environment {index} did not listen on port {relay_port}")

    def start_simulators(self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str) -> None:
        """Open one official Windows simulator GUI for each ready native relay."""
        if not connections:
            raise ValueError("no simulator connection ports were allocated")
        powershell = "powershell.exe"
        simulator_exe = os.environ.get("AICAR_OFFICIAL_SIMULATOR_EXE", "").strip()
        if not simulator_exe:
            configured_path = ROOT / "logs" / "layer4" / "official_simulator_path.txt"
            try:
                simulator_exe = configured_path.read_text(encoding="utf-8-sig").strip()
            except OSError:
                simulator_exe = ""
        if not simulator_exe:
            discover = (
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
                "$p=Get-Process -Name 'AutoDRIVE Simulator' | "
                "Select-Object -First 1 -ExpandProperty Path; "
                "if ($p) { $p }"
            )
            found = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", discover],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=15,
            )
            simulator_exe = found.stdout.strip()
            if simulator_exe:
                configured_path = ROOT / "logs" / "layer4" / "official_simulator_path.txt"
                configured_path.parent.mkdir(parents=True, exist_ok=True)
                configured_path.write_text(simulator_exe, encoding="utf-8")
        if not simulator_exe:
            raise RuntimeError(
                "Set AICAR_OFFICIAL_SIMULATOR_EXE to the official Windows AutoDRIVE Simulator.exe path "
                "or leave one official simulator process open so its executable path can be detected."
            )
        script_wsl = ROOT / "scripts" / "start_official_windows_simulators.ps1"
        script_windows = subprocess.run(
            ["wslpath", "-w", str(script_wsl)], check=True,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", timeout=10,
        ).stdout.strip()
        bridge_ip = self.simulator_host
        run_dir = self._run_dir_for_id(run_id)
        launcher_log = (run_dir / "windows_simulators.log").open("ab", buffering=0)
        pid_file = run_dir / "windows_simulator_pids.txt"
        pid_file_windows = subprocess.run(
            ["wslpath", "-w", str(pid_file)], check=True,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", timeout=10,
        ).stdout.strip()
        ports_file = run_dir / "windows_simulator_ports.csv"
        ports_file.write_text(",".join(str(int(item["port"])) for item in connections), encoding="ascii")
        ports_file_windows = subprocess.run(
            ["wslpath", "-w", str(ports_file)], check=True,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", timeout=10,
        ).stdout.strip()
        pause_file = run_dir / "evaluation_pause.flag"
        pause_file_windows = subprocess.run(
            ["wslpath", "-w", str(pause_file)], check=True,
            stdout=subprocess.PIPE, text=True, encoding="utf-8", timeout=10,
        ).stdout.strip()
        arguments = [
            powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", script_windows, "-SimulatorExePath", simulator_exe,
            "-RunId", run_id, "-HubUrl", hub_url, "-BridgeIpAddress", bridge_ip,
            "-PidFile", pid_file_windows, "-PortsFile", ports_file_windows,
            "-PauseFile", pause_file_windows,
        ]
        process = subprocess.Popen(
            arguments, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=launcher_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        with self._lock:
            self._windows_launchers[run_id] = process

        ready_marker = "All official simulator environments are connected."
        deadline = time.monotonic() + max(240.0, 120.0 * len(connections))
        log_path = Path(launcher_log.name)
        launcher_ready = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"Windows simulator launcher exited ({process.returncode}): {excerpt}")
            if ready_marker in log_path.read_text(encoding="utf-8", errors="replace"):
                launcher_ready = True
                break
            time.sleep(0.5)
        launcher_log.close()
        if not launcher_ready:
            raise TimeoutError("Windows simulator windows did not all connect before the startup deadline")

        # A successful TCP connect only proves the port is open; it does not
        # prove that the Unity Socket.IO client completed its handshake or is
        # publishing camera-bearing frames. Require all four independent
        # relay health reports before declaring run startup complete.
        self._wait_for_simulator_streams(connections, run_dir)

    def _read_simulator_health(self, port: int) -> dict[str, Any] | None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as response:
                health = json.loads(response.read())
            return health if isinstance(health, dict) else None
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def _wait_for_simulator_streams(
        self,
        connections: list[dict[str, Any]],
        run_dir: Path,
        *,
        minimum_frames: dict[int, tuple[int, int]] | None = None,
    ) -> dict[int, dict[str, Any]]:
        deadline = time.monotonic() + max(45.0, self.startup_timeout_s)
        last: dict[int, dict[str, Any] | str] = {}
        while time.monotonic() < deadline:
            all_ready = True
            for item in connections:
                index = int(item["environment"])
                port = int(item["port"])
                health = self._read_simulator_health(port)
                if health is None:
                    last[index] = "health endpoint unavailable"
                    all_ready = False
                    continue
                last[index] = health
                minimum_complete, minimum_camera = (minimum_frames or {}).get(index, (1, 1))
                if not (
                    int(health.get("simulator_clients", 0)) >= 1
                    and int(health.get("complete_frames", 0)) >= minimum_complete
                    and int(health.get("camera_frames", 0)) >= minimum_camera
                ):
                    all_ready = False
            if all_ready:
                (run_dir / "simulator_stream_health.json").write_text(
                    json.dumps({"environments": last}, indent=2), encoding="utf-8"
                )
                return {int(index): value for index, value in last.items() if isinstance(value, dict)}
            time.sleep(0.5)
        (run_dir / "simulator_stream_health.json").write_text(
            json.dumps({"environments": last}, indent=2), encoding="utf-8"
        )
        raise TimeoutError(
            "Windows simulator ports accepted TCP connections, but all official "
            f"Socket.IO camera streams did not become ready: {json.dumps(last, sort_keys=True)}"
        )

    def pause_training_for_evaluation(
        self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str,
    ) -> None:
        """Freeze PPO and retire its Windows sims while the official evaluator runs."""
        with self._lock:
            group = next((item for item in self._groups.values()
                          if item.get("Labels", {}).get("aicar.run_id") == run_id
                          and item.get("Labels", {}).get("aicar.role") == "api"), None)
            trainer = next((item for item in (group or {}).get("records", ())
                            if item.get("role") == "trainer"), None)
        if trainer is None:
            raise RuntimeError(f"training process for {run_id} is unavailable for evaluation pause")
        process = trainer["process"]
        if process.poll() is not None:
            raise RuntimeError(f"training process for {run_id} has already exited")

        run_dir = self._run_dir_for_id(run_id)
        pause_file = run_dir / "evaluation_pause.flag"
        if pause_file.exists():
            raise RuntimeError(f"training run {run_id} is already paused for evaluation")
        baselines: dict[int, tuple[int, int]] = {}
        for item in connections:
            index = int(item["environment"])
            health = self._read_simulator_health(int(item["port"])) or {}
            baselines[index] = (
                int(health.get("complete_frames", 0)) + 1,
                int(health.get("camera_frames", 0)) + 1,
            )
        group["evaluation_frame_baselines"] = baselines

        os.killpg(int(process.pid), getattr(signal, "SIGSTOP", 19))
        try:
            pause_file.write_text(json.dumps({
                "frames": {
                    str(index): {"complete": values[0] - 1, "camera": values[1] - 1}
                    for index, values in baselines.items()
                }
            }), encoding="utf-8")
            pid_file = run_dir / "windows_simulator_pids.txt"
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    remaining = pid_file.read_text(encoding="ascii").strip()
                except OSError:
                    remaining = ""
                if not remaining:
                    return
                time.sleep(0.2)
            raise TimeoutError(f"Windows training simulators for {run_id} did not pause")
        except Exception:
            pause_file.unlink(missing_ok=True)
            os.killpg(int(process.pid), getattr(signal, "SIGCONT", 18))
            raise

    def resume_training_after_evaluation(
        self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str,
    ) -> None:
        with self._lock:
            group = next((item for item in self._groups.values()
                          if item.get("Labels", {}).get("aicar.run_id") == run_id
                          and item.get("Labels", {}).get("aicar.role") == "api"), None)
            trainer = next((item for item in (group or {}).get("records", ())
                            if item.get("role") == "trainer"), None)
        if trainer is None:
            raise RuntimeError(f"training process for {run_id} is unavailable for evaluation resume")
        process = trainer["process"]
        pause_file = self._run_dir_for_id(run_id) / "evaluation_pause.flag"
        baselines = (group or {}).get("evaluation_frame_baselines", {})
        if not baselines:
            try:
                pause_state = json.loads(pause_file.read_text(encoding="utf-8"))
                baselines = {
                    int(index): (int(counts.get("complete", 0)) + 1, int(counts.get("camera", 0)) + 1)
                    for index, counts in pause_state.get("frames", {}).items()
                }
            except (OSError, ValueError, TypeError, AttributeError):
                baselines = {}
        try:
            pause_file.unlink(missing_ok=True)
            self._wait_for_simulator_streams(
                connections, self._run_dir_for_id(run_id), minimum_frames=baselines,
            )
        finally:
            if group is not None:
                group.pop("evaluation_frame_baselines", None)
            if process.poll() is None:
                os.killpg(int(process.pid), getattr(signal, "SIGCONT", 18))

    def abort_evaluation_pause(self, run_id: str) -> None:
        """Wake a suspended trainer so the normal run shutdown can finish."""
        with self._lock:
            group = next((item for item in self._groups.values()
                          if item.get("Labels", {}).get("aicar.run_id") == run_id
                          and item.get("Labels", {}).get("aicar.role") == "api"), None)
            trainer = next((item for item in (group or {}).get("records", ())
                            if item.get("role") == "trainer"), None)
        if trainer is not None and trainer["process"].poll() is None:
            os.killpg(int(trainer["process"].pid), getattr(signal, "SIGCONT", 18))

    def stop_simulators(self, run_id: str) -> None:
        pid_file = self._run_dir_for_id(run_id) / "windows_simulator_pids.txt"
        try:
            pids = [int(line.strip()) for line in pid_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            return
        if not pids:
            return
        command = ";".join(
            f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue" for pid in pids
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=15,
        )

    @staticmethod
    def _run_dir_for_id(run_id: str) -> Path:
        return ROOT / "logs" / "rl" / f"official_run_{run_id}"

    def official_run_container_logs(self, identity: str) -> str:
        with self._lock:
            group = self._groups.get(identity)
            records = list(group["records"]) if group else []
        chunks = []
        for record in records:
            log = record.get("log")
            try:
                path = Path(log.name)
                if path.is_file():
                    chunks.append(f"\n--- {record['role']} ---\n{path.read_text(encoding='utf-8', errors='replace')}")
            except (AttributeError, OSError):
                continue
        return "".join(chunks)

    def stop_official_run_container(self, identity: str, timeout_s: int = 15) -> None:
        with self._lock:
            group = self._groups.get(identity)
            records = list(group["records"]) if group else []
        if not group:
            return
        # For the primary API role, stop only PPO first so the official bridges
        # remain alive long enough for its graceful checkpoint/shutdown path.
        target = [record for record in records if record["role"] == "trainer"] or records
        for record in target:
            process = record["process"]
            if process.poll() is None:
                # Let PPO handle SIGTERM itself so it can close SubprocVecEnv
                # workers and write its final checkpoint.
                if record["role"] == "trainer":
                    process.terminate()
                else:
                    _signal_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + max(1, timeout_s)
        for record in target:
            process = record["process"]
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=2)

    def remove_official_run_container(self, identity: str) -> None:
        with self._lock:
            group = self._groups.pop(identity, None)
        if not group:
            return
        for record in group["records"]:
            process = record["process"]
            _signal_process_group(process, signal.SIGTERM)
        for record in group["records"]:
            process = record["process"]
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                process.wait(timeout=1)
            # The ROS launcher can exit before its node process; clean the
            # remaining members of every isolated process group as well.
            _signal_process_group(process, signal.SIGKILL)
            try:
                record["log"].close()
            except OSError:
                pass
        manifest = group.get("manifest")
        if manifest:
            try:
                Path(manifest).unlink()
            except OSError:
                pass

    def list_official_run_containers(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            groups = list(self._groups.values())
        result = []
        for group in groups:
            if run_id and group["Labels"].get("aicar.run_id") != run_id:
                continue
            trainer = next((record["process"] for record in group["records"] if record["role"] == "trainer"), None)
            bridge_processes = [record["process"] for record in group["records"] if record["role"] != "trainer"]
            alive = all(process.poll() is None for process in bridge_processes)
            if trainer is not None:
                state = "running" if trainer.poll() is None and alive else "exited"
            else:
                state = "running" if alive else "exited"
            result.append({key: value for key, value in group.items() if key != "records"} | {"State": state})
        return result

    def list_official_run_networks(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            networks = list(self._networks.values())
        return [row for row in networks if not run_id or row["Labels"].get("aicar.run_id") == run_id]
