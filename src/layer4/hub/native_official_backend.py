"""Native WSL process backend for official Windows AutoDRIVE training.

The simulator remains the unmodified Windows build. Each environment gets
its own ROS domain, official Devkit bridge, and Layer 1 relay; one native PPO
process owns the shared learner.
"""

from __future__ import annotations

import os
import json
import hashlib
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
import tempfile
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # Native process ownership is only supported on the WSL/Linux host.
    fcntl = None

from src.layer4.settings import ROOT


def simulator_session_ready(
    health: dict[str, Any], *, minimum_complete_frames: int = 1,
    minimum_camera_frames: int = 1, minimum_devkit_commands: int = 1,
) -> bool:
    """Check readiness within one current SID, never via cumulative totals."""
    sessions = health.get("simulator_sessions")
    if not isinstance(sessions, list):
        return False
    return int(health.get("simulator_clients", 0)) >= 1 and any(
        isinstance(session, dict)
        and isinstance(session.get("sid"), str) and bool(session["sid"])
        and int(session.get("complete_frames", 0)) >= minimum_complete_frames
        and int(session.get("camera_frames", 0)) >= minimum_camera_frames
        and int(session.get("devkit_commands", 0)) >= minimum_devkit_commands
        for session in sessions
    )


def _windows_simulator_mode_args(config: dict[str, Any] | None) -> list[str]:
    """Select presentation mode without changing the official player build."""
    return ["-BatchMode"] if (config or {}).get("simulator_display_mode", "gui") == "offscreen" else []


class _RecoveredProcess:
    """Small Popen-compatible handle for a child adopted after hub restart."""

    def __init__(
        self, pid: int, process_group_id: int | None = None,
        process_starttime_ticks: int | None = None,
    ):
        self.pid = int(pid)
        self.process_group_id = int(process_group_id or pid)
        self.process_starttime_ticks = process_starttime_ticks
        self.returncode: int | None = None

    def poll(self) -> int | None:
        try:
            stat = NativeOfficialBackend._read_process_identity(self.pid)
            if stat is None:
                try:
                    os.kill(self.pid, 0)
                except ProcessLookupError:
                    self.returncode = 0
                    return self.returncode
                except PermissionError:
                    return None
                # Procfs may be temporarily unreadable or malformed. A PID
                # that still exists is not known to have exited.
                return None
            if (self.process_starttime_ticks is not None
                    and stat.get("starttime_ticks") != self.process_starttime_ticks):
                self.returncode = 0
                return self.returncode
            if stat.get("state") not in {"Z", "X"}:
                return None
        except ProcessLookupError:
            self.returncode = 0
            return self.returncode
        except (PermissionError, OSError, ValueError, TypeError):
            # A transient procfs read failure is not evidence that the process
            # exited. Keep it alive until a later poll can establish state.
            return None
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
        self.backend_instance_id = uuid.uuid4().hex
        self.hub_pid = os.getpid()
        # Native ROS bridges share the host network namespace. Serialize their
        # startup so concurrent runs cannot pick the same free port/domain.
        self._startup_lock = threading.Lock()
        self._groups: dict[str, dict[str, Any]] = {}
        self._reserved_ros_domains: set[int] = set()
        self._reserved_devkit_ports: set[int] = set()
        self._devkit_port_start = 4567
        self._networks: dict[str, dict[str, Any]] = {}
        self._windows_launchers: dict[str, subprocess.Popen] = {}
        self._ownership_lock_file = None
        self.is_owner = self._acquire_artifact_root_lock()
        if self.is_owner:
            self._recover_groups()

    def _acquire_artifact_root_lock(self) -> bool:
        """Ensure only one hub process can adopt or mutate native run manifests."""
        if fcntl is None:
            return False
        root_key = hashlib.sha256(str(ROOT.resolve()).encode("utf-8")).hexdigest()[:24]
        lock_path = Path(tempfile.gettempdir()) / f"aicar-native-backend-{root_key}.lock"
        try:
            lock_file = lock_path.open("a+")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            try:
                lock_file.close()
            except (UnboundLocalError, OSError):
                pass
            return False
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(json.dumps({"backend_instance_id": self.backend_instance_id, "hub_pid": self.hub_pid}))
        lock_file.flush()
        self._ownership_lock_file = lock_file
        return True

    def close(self) -> None:
        """Release the exclusive artifact-root owner lock without stopping runs."""
        lock_file = getattr(self, "_ownership_lock_file", None)
        if lock_file is None:
            return
        self._ownership_lock_file = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def _recover_groups(self) -> None:
        for manifest in (ROOT / "logs" / "rl").glob("official_run_*/native_group_*.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                records = []
                for item in payload["records"]:
                    log_path = Path(item["log"])
                    records.append({
                        "role": item["role"],
                        "process": _RecoveredProcess(
                            int(item["pid"]), item.get("process_group_id"),
                            item.get("process_starttime_ticks"),
                        ),
                        "process_group_id": int(item.get("process_group_id", item["pid"])),
                        "process_starttime_ticks": item.get("process_starttime_ticks"),
                        "handle_origin": "recovered_from_manifest",
                        "original_process_handle_class": item.get("process_handle_class"),
                        "launch_parent_pid": item.get("launch_parent_pid"),
                        "launch_command": item.get("launch_command"),
                        "log": log_path.open("ab", buffering=0) if log_path.exists() else open(os.devnull, "ab"),
                    })
                group = {
                    **payload, "records": records, "manifest": manifest,
                    "recovered_by_backend_instance_id": self.backend_instance_id,
                    "recovered_by_hub_pid": self.hub_pid,
                }
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
    def _read_process_identity(pid: int) -> dict[str, Any] | None:
        """Read procfs fields without splitting the parenthesized comm naively."""
        try:
            raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
            close_paren = raw.rfind(")")
            if close_paren < 0:
                return None
            fields = raw[close_paren + 1:].split()
            if len(fields) <= 19:
                return None
            return {
                "state": fields[0],
                "parent_pid": int(fields[1]),
                "process_group_id": int(fields[2]),
                "session_id": int(fields[3]),
                # stat field 22 is index 19 after state (field 3).
                "starttime_ticks": int(fields[19]),
                "command": raw[raw.find("(") + 1:close_paren],
            }
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _process_cmdline(pid: int) -> str | None:
        try:
            raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        except OSError:
            return None
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip() or None

    @staticmethod
    def _persist_group(group: dict[str, Any]) -> None:
        run_id = group["Labels"]["aicar.run_id"]
        role = group["Labels"]["aicar.role"]
        run_dir = ROOT / "logs" / "rl" / f"official_run_{run_id}"
        manifest = run_dir / f"native_group_{role}.json"
        payload = {key: group[key] for key in ("Id", "Names", "Labels")}
        payload.update({
            "backend_instance_id": group.get("backend_instance_id"),
            "hub_pid": group.get("hub_pid"),
            "created_at_unix_s": group.get("created_at_unix_s"),
        })
        for key in ("ros_domains", "devkit_ports"):
            if key in group:
                payload[key] = group[key]
        payload["records"] = [
            {
                "role": item["role"],
                "pid": item["process"].pid,
                "process_group_id": int(item.get("process_group_id", item["process"].pid)),
                "process_starttime_ticks": item.get("process_starttime_ticks"),
                "handle_origin": item.get("handle_origin", "popen"),
                "process_handle_class": item.get(
                    "original_process_handle_class",
                    f"{type(item['process']).__module__}.{type(item['process']).__qualname__}",
                ),
                "launch_parent_pid": item.get("launch_parent_pid"),
                "launch_command": item.get("launch_command"),
                "log": str(Path(item["log"].name)),
            }
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
            sys.executable, "-u", "-m", "src.layer3.official_train",
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
            "--behavior-cloning-enabled", value("AICAR_BEHAVIOR_CLONING_ENABLED", "1"),
            "--behavior-cloning-only", value("AICAR_BEHAVIOR_CLONING_ONLY", "0"),
            "--behavior-cloning-steps-per-env", value("AICAR_BEHAVIOR_CLONING_STEPS_PER_ENV", "4000"),
            "--behavior-cloning-epochs", value("AICAR_BEHAVIOR_CLONING_EPOCHS", "8"),
            "--behavior-cloning-batch-size", value("AICAR_BEHAVIOR_CLONING_BATCH_SIZE", "256"),
            "--behavior-cloning-learning-rate", value("AICAR_BEHAVIOR_CLONING_LEARNING_RATE", "0.001"),
            "--behavior-teacher-max-speed-mps", value("AICAR_BEHAVIOR_TEACHER_MAX_SPEED_MPS", "1.2"),
            "--n-steps", value("AICAR_PPO_N_STEPS", "2048"),
            "--learning-rate", value("AICAR_PPO_LEARNING_RATE", "0.0001"),
            "--n-epochs", value("AICAR_PPO_N_EPOCHS", "4"),
            "--gamma", value("AICAR_PPO_GAMMA", "0.9995"),
            "--gae-lambda", value("AICAR_PPO_GAE_LAMBDA", "0.95"),
            "--timeout-s", value("AICAR_SENSOR_TIMEOUT_S", "180"),
            "--training-timeout-s", value("AICAR_TRAIN_WATCHDOG_S", "600"),
            "--race-laps", value("AICAR_TRAIN_RACE_LAPS", "10"),
            "--warmup-laps", value("AICAR_TRAIN_WARMUP_LAPS", "0"),
            "--time-cost-per-simulated-second", value("AICAR_TRAIN_TIME_COST", "0"),
            "--frontier-stagnation-s", value("AICAR_TRAIN_FRONTIER_STAGNATION_S", "5"),
            "--automatic-evaluation-mode", value("AICAR_AUTO_EVALUATION_MODE", "competitive"),
            "--collision-penalty-magnitude", value("AICAR_TRAIN_COLLISION_PENALTY_MAGNITUDE", "200"),
            "--collision-reward-percent", value("AICAR_TRAIN_COLLISION_REWARD_PERCENT", "0"),
            "--lap-completion-reward", value("AICAR_TRAIN_LAP_COMPLETION_REWARD", "0"),
            "--failed-episode-penalty", value("AICAR_TRAIN_FAILURE_PENALTY", "200"),
            "--backward-speed-penalty-scale", value("AICAR_TRAIN_BACKWARD_SPEED_PENALTY_SCALE", "10"),
            "--episode-failure-reward-percent", value("AICAR_TRAIN_FAILURE_REWARD_PERCENT", "0"),
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
        target_kl = value("AICAR_PPO_TARGET_KL", "").strip()
        if target_kl:
            command.extend(["--target-kl", target_kl])
        exploration_profile = value("AICAR_PPO_EXPLORATION_PROFILE", "").strip()
        if exploration_profile:
            command.extend(["--exploration-profile", exploration_profile])
        for env_name, argument in (
            ("AICAR_PPO_EXPLORATION_INITIAL_STD_THROTTLE", "--exploration-initial-std-throttle"),
            ("AICAR_PPO_EXPLORATION_INITIAL_STD_STEERING", "--exploration-initial-std-steering"),
            ("AICAR_PPO_EXPLORATION_MIN_STD_THROTTLE", "--exploration-min-std-throttle"),
            ("AICAR_PPO_EXPLORATION_MIN_STD_STEERING", "--exploration-min-std-steering"),
            ("AICAR_PPO_EXPLORATION_MAX_STD_THROTTLE", "--exploration-max-std-throttle"),
            ("AICAR_PPO_EXPLORATION_MAX_STD_STEERING", "--exploration-max-std-steering"),
        ):
            configured = value(env_name, "").strip()
            if configured:
                command.extend([argument, configured])
        return command

    @staticmethod
    def _evaluation_command(env: dict[str, str]) -> list[str]:
        command = [
            sys.executable, "-u", "-m", "src.layer3.official_evaluate",
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
        validation_mode = env.get("AICAR_EVALUATION_VALIDATION_MODE")
        if validation_mode in {"bc_diagnostic", "training_probe"}:
            command.extend([
                "--validation-mode", validation_mode,
                "--lap-progress-timeout-s", env.get("AICAR_LAP_PROGRESS_TIMEOUT_S", "240"),
                "--lap-progress-wall-timeout-s", env.get("AICAR_LAP_PROGRESS_WALL_TIMEOUT_S", "240"),
                "--frontier-stall-after-s", env.get("AICAR_FRONTIER_STALL_AFTER_S", "5"),
            ])
        frontier_path = env.get("AICAR_EVALUATION_FRONTIER_PATH")
        if frontier_path:
            command.extend(["--frontier-path", frontier_path])
        if env.get("AICAR_ROUTE_MONITOR_ONLY") == "1":
            command.append("--route-monitor-only")
        return command

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
        child_env["PYTHONFAULTHANDLER"] = "1"
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
                            child_env["HUB_URL"] = os.environ.get("AICAR_HUB_PUBLIC_URL", "http://127.0.0.1:8090")
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
                        records.append({"role": "trainer", "process": process,
                                        "process_group_id": int(process.pid),
                                        "process_starttime_ticks": (
                                            self._read_process_identity(process.pid) or {}
                                        ).get("starttime_ticks"),
                                        "handle_origin": "created_by_popen",
                                        "original_process_handle_class": f"{type(process).__module__}.{type(process).__qualname__}",
                                        "launch_parent_pid": os.getpid(),
                                        "launch_command": list(command),
                                        "log": train_log})
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
            "backend_instance_id": getattr(self, "backend_instance_id", uuid.uuid4().hex),
            "hub_pid": getattr(self, "hub_pid", os.getpid()),
            "created_at_unix_s": time.time(),
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
        records.append({"role": f"devkit-{index}", "process": devkit,
                        "process_group_id": int(devkit.pid),
                        "process_starttime_ticks": (
                            self._read_process_identity(devkit.pid) or {}
                        ).get("starttime_ticks"),
                        "handle_origin": "created_by_popen",
                        "original_process_handle_class": f"{type(devkit).__module__}.{type(devkit).__qualname__}",
                        "launch_parent_pid": os.getpid(),
                        "launch_command": list(devkit.args) if isinstance(devkit.args, list) else str(devkit.args),
                        "log": bridge_log})

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
        records.append({"role": f"relay-{index}", "process": relay,
                        "process_group_id": int(relay.pid),
                        "process_starttime_ticks": (
                            self._read_process_identity(relay.pid) or {}
                        ).get("starttime_ticks"),
                        "handle_origin": "created_by_popen",
                        "original_process_handle_class": f"{type(relay).__module__}.{type(relay).__qualname__}",
                        "launch_parent_pid": os.getpid(),
                        "launch_command": list(relay.args) if isinstance(relay.args, list) else str(relay.args),
                        "log": relay_log})

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

    def start_simulators(
        self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Start official Windows simulators against each ready native relay."""
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
        arguments.extend(_windows_simulator_mode_args(config))
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
                if not simulator_session_ready(
                    health, minimum_complete_frames=minimum_complete,
                    minimum_camera_frames=minimum_camera,
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

    def _wait_for_simulator_sessions_disconnected(
        self, connections: list[dict[str, Any]], *, timeout_s: float = 30.0,
    ) -> None:
        """Require retired simulator SIDs to be gone before evaluating/resuming."""
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        last: dict[int, Any] = {}
        while time.monotonic() < deadline:
            all_disconnected = True
            for item in connections:
                index = int(item["environment"])
                health = self._read_simulator_health(int(item["port"]))
                last[index] = health if health is not None else "health endpoint unavailable"
                if health is None:
                    all_disconnected = False
                    continue
                sessions = health.get("simulator_sessions")
                if int(health.get("simulator_clients", 0)) != 0 or sessions != []:
                    all_disconnected = False
            if all_disconnected:
                return
            time.sleep(0.2)
        raise TimeoutError(
            "Retired Windows simulator sessions did not disconnect before evaluation: "
            f"{json.dumps(last, sort_keys=True)}"
        )

    @staticmethod
    def _watch_step(run_dir: Path) -> int | None:
        try:
            payload = json.loads((run_dir / "watch_latest.json").read_text(encoding="utf-8"))
            value = payload.get("step") if isinstance(payload, dict) else None
            return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        """Publish small cross-process handshakes without exposing partial JSON."""
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _wait_for_training_handshake(
        self, *, run_id: str, process: subprocess.Popen[Any], path: Path,
        request_id: str, phase: str, expected_status: str,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Wait for the live trainer's cooperative callback ACK, never SIGSTOP it."""
        deadline = time.monotonic() + max(
            30.0, float(timeout_s if timeout_s is not None else self.startup_timeout_s)
        )
        last_error: str | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"training process for {run_id} exited before the {phase} handshake "
                    f"(returncode={process.returncode})"
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (isinstance(payload, dict) and payload.get("run_id") == run_id
                        and payload.get("request_id") == request_id):
                    if payload.get("status") != expected_status:
                        raise RuntimeError(
                            f"training process for {run_id} did not complete {phase}: "
                            f"status={payload.get('status')!r}"
                        )
                    return payload
            except RuntimeError:
                raise
            except (OSError, ValueError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.1)
        raise TimeoutError(
            f"training process for {run_id} did not acknowledge {phase} within the "
            f"deadline (request_id={request_id}, last_error={last_error})"
        )

    @staticmethod
    def _verify_pause_checkpoint(run_dir: Path, ack: dict[str, Any]) -> dict[str, Any]:
        """Require a durable post-update checkpoint before retiring a training sim."""
        checkpoint = ack.get("checkpoint")
        if (not isinstance(checkpoint, dict)
                or checkpoint.get("post_optimizer_update") is not True):
            raise RuntimeError("cooperative pause ACK lacks a post-optimizer checkpoint")
        supplied_hash = checkpoint.get("sha256")
        if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
            raise RuntimeError("cooperative pause ACK has no SHA-256 checkpoint digest")
        try:
            checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve(strict=True)
            checkpoint_path.relative_to(run_dir.resolve(strict=True))
            actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cooperative pause checkpoint is not a durable run artifact: {exc}") from exc
        if actual_hash.lower() != supplied_hash.lower():
            raise RuntimeError("cooperative pause checkpoint SHA-256 does not match its ACK")
        try:
            update = int(ack.get("optimizer_updates", 0))
            timesteps = int(ack.get("num_timesteps", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("cooperative pause ACK has invalid trainer counters") from exc
        if update < 1 or timesteps < 1:
            raise RuntimeError("cooperative pause ACK predates the first optimizer update")
        return {
            **checkpoint,
            "path": str(checkpoint_path),
            "sha256": actual_hash,
            "optimizer_updates": update,
            "num_timesteps": timesteps,
        }

    def _wait_for_training_observation_after_resume(
        self, *, run_id: str, run_dir: Path, process: subprocess.Popen[Any],
        previous_step: int | None,
    ) -> int:
        """Wait until the continued learner completes an env step on fresh ROS data."""
        deadline = time.monotonic() + max(30.0, self.startup_timeout_s)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"training process for {run_id} exited before a post-resume observation "
                    f"(returncode={process.returncode})"
                )
            current_step = self._watch_step(run_dir)
            if current_step is not None and (previous_step is None or current_step > previous_step):
                return current_step
            time.sleep(0.1)
        raise TimeoutError(
            f"training process for {run_id} did not publish a newer Watch step after resume "
            f"(previous={previous_step}, current={self._watch_step(run_dir)})"
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
        request_file = run_dir / "evaluation_pause.request.json"
        ack_file = run_dir / "evaluation_pause.ack.json"
        if pause_file.exists() or request_file.exists():
            raise RuntimeError(f"training run {run_id} is already paused for evaluation")
        ack_file.unlink(missing_ok=True)
        (run_dir / "evaluation_resume.request.json").unlink(missing_ok=True)
        (run_dir / "evaluation_resume.ack.json").unlink(missing_ok=True)
        request_id = uuid.uuid4().hex
        request = {
            "run_id": run_id,
            "request_id": request_id,
            "requested_at_unix_s": time.time(),
            "protocol": "cooperative_post_optimizer_rollout_boundary_v1",
        }
        self._write_json_atomic(request_file, request)
        try:
            # The trainer reaches this callback after the current rollout was
            # optimized. It keeps the ROS executor alive while its learner thread
            # waits, avoiding a multi-minute SIGSTOP/DDS lease discontinuity.
            pause_ack = self._wait_for_training_handshake(
                run_id=run_id, process=process, path=ack_file,
                request_id=request_id, phase="pause", expected_status="paused",
                timeout_s=max(240.0, self.startup_timeout_s),
            )
            pause_ack = dict(pause_ack)
            pause_ack["checkpoint"] = self._verify_pause_checkpoint(run_dir, pause_ack)
            baselines: dict[int, tuple[int, int]] = {}
            for item in connections:
                index = int(item["environment"])
                health = self._read_simulator_health(int(item["port"])) or {}
                baselines[index] = (
                    int(health.get("complete_frames", 0)) + 1,
                    int(health.get("camera_frames", 0)) + 1,
                )
            group["evaluation_frame_baselines"] = baselines
            pause_file.write_text(json.dumps({
                "frames": {
                    str(index): {"complete": values[0] - 1, "camera": values[1] - 1}
                    for index, values in baselines.items()
                },
                "run_id": run_id,
                "request_id": request_id,
            }), encoding="utf-8")
            pid_file = run_dir / "windows_simulator_pids.txt"
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    remaining = pid_file.read_text(encoding="ascii").strip()
                except OSError:
                    remaining = ""
                if not remaining:
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError(f"Windows training simulators for {run_id} did not pause")
            self._wait_for_simulator_sessions_disconnected(connections)
            group["evaluation_pause_request_id"] = request_id
            group["evaluation_pause_ack"] = pause_ack
        except Exception:
            pause_file.unlink(missing_ok=True)
            request_file.unlink(missing_ok=True)
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
        run_dir = self._run_dir_for_id(run_id)
        pause_file = run_dir / "evaluation_pause.flag"
        request_file = run_dir / "evaluation_pause.request.json"
        pause_ack_file = run_dir / "evaluation_pause.ack.json"
        resume_request_file = run_dir / "evaluation_resume.request.json"
        resume_ack_file = run_dir / "evaluation_resume.ack.json"
        previous_watch_step = self._watch_step(run_dir)
        pause_ack = (group or {}).get("evaluation_pause_ack")
        if not isinstance(pause_ack, dict):
            try:
                value = json.loads(pause_ack_file.read_text(encoding="utf-8"))
                pause_ack = value if isinstance(value, dict) else None
            except (OSError, ValueError, TypeError):
                pause_ack = None
        request_id = (group or {}).get("evaluation_pause_request_id")
        if not isinstance(request_id, str):
            try:
                state = json.loads(request_file.read_text(encoding="utf-8"))
                request_id = state.get("request_id") if isinstance(state, dict) else None
            except (OSError, ValueError, TypeError):
                request_id = None
        if not isinstance(request_id, str):
            raise RuntimeError(f"training run {run_id} has no cooperative pause request to resume")
        resume_request = {
            "run_id": run_id,
            "request_id": request_id,
            "requested_at_unix_s": time.time(),
            "protocol": "cooperative_post_optimizer_rollout_boundary_v1",
        }
        resume_ack: dict[str, Any] | None = None
        try:
            pause_file.unlink(missing_ok=True)
            self._wait_for_simulator_streams(
                connections, run_dir, minimum_frames={
                    int(item["environment"]): (1, 1) for item in connections
                },
            )
            if process.poll() is not None:
                raise RuntimeError(f"training process for {run_id} exited before resume")
            self._write_json_atomic(resume_request_file, resume_request)
            resume_ack = self._wait_for_training_handshake(
                run_id=run_id, process=process, path=resume_ack_file,
                request_id=request_id, phase="resume", expected_status="resumed",
                timeout_s=max(120.0, self.startup_timeout_s),
            )
            resumed_step = self._wait_for_training_observation_after_resume(
                run_id=run_id, run_dir=run_dir, process=process,
                previous_step=previous_watch_step,
            )
            event = {
                "run_id": run_id,
                "previous_watch_step": previous_watch_step,
                "resumed_watch_step": resumed_step,
                "acknowledged_at_unix_s": time.time(),
                "fresh_simulator_session_required": True,
                "cooperative_pause_ack": pause_ack,
                "resume_ack": resume_ack,
            }
            with (run_dir / "evaluation_resume_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if group is not None:
                group.pop("evaluation_frame_baselines", None)
                group.pop("evaluation_pause_request_id", None)
                group.pop("evaluation_pause_ack", None)
            # If the resume gate failed, still release the cooperative wait so
            # its own bounded ROS-freshness guard can fail safely and durably.
            if process.poll() is None and not resume_request_file.exists():
                self._write_json_atomic(resume_request_file, resume_request)
            request_file.unlink(missing_ok=True)
            pause_ack_file.unlink(missing_ok=True)
            if resume_ack is not None:
                resume_request_file.unlink(missing_ok=True)
                resume_ack_file.unlink(missing_ok=True)

    def abort_evaluation_pause(self, run_id: str) -> None:
        """Unblock simulator retirement; the manager's SIGTERM stop event wakes the callback."""
        run_dir = self._run_dir_for_id(run_id)
        # Keep the trainer request visible until its stop callback observes the
        # manager's SIGTERM event. Removing it could let a callback slip into a
        # fresh env.step while the training simulator is already retired.
        (run_dir / "evaluation_pause.flag").unlink(missing_ok=True)

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

    @staticmethod
    def _read_log_tail(record: dict[str, Any], limit: int = 8192) -> str | None:
        try:
            path = Path(record["log"].name)
            if not path.is_file():
                return None
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - limit))
                return stream.read(limit).decode("utf-8", errors="replace")
        except (AttributeError, KeyError, OSError, TypeError):
            return None

    @staticmethod
    def _process_group_alive(record: dict[str, Any]) -> bool | None:
        raw_pgid = record.get("process_group_id")
        if raw_pgid is None:
            return None
        try:
            os.killpg(int(raw_pgid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except (OSError, TypeError, ValueError):
            return None

    def _capture_terminal_context(self, group: dict[str, Any]) -> dict[str, Any]:
        """Persist the first observed process exit before cleanup mutates roles."""
        run_id = (group.get("Labels") or {}).get("aicar.run_id")
        path = self._run_dir_for_id(str(run_id)) / "native_terminal_context.json" if run_id else None
        if path is not None:
            with self._lock:
                try:
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(saved, dict) and saved.get("run_id") == run_id:
                        return saved
                except (OSError, ValueError, TypeError):
                    pass
        role_records = []
        for record in group.get("records", []):
            process = record.get("process")
            try:
                returncode = process.poll()
            except (AttributeError, OSError):
                returncode = None
            known = not isinstance(process, _RecoveredProcess)
            process_identity = self._read_process_identity(getattr(process, "pid", -1))
            role_records.append({
                "role": str(record.get("role", "unknown")),
                "pid": getattr(process, "pid", None),
                "process_group_id": record.get("process_group_id"),
                "process_starttime_ticks": record.get("process_starttime_ticks"),
                "proc_identity": process_identity,
                "proc_cmdline": self._process_cmdline(getattr(process, "pid", -1)),
                "handle_origin": record.get(
                    "handle_origin",
                    "recovered_from_manifest" if isinstance(process, _RecoveredProcess) else "created_by_popen",
                ),
                "handle_class": f"{type(process).__module__}.{type(process).__qualname__}",
                "launch_parent_pid": record.get("launch_parent_pid"),
                "launch_command": record.get("launch_command"),
                "status": "exited" if returncode is not None else "running",
                "returncode": returncode if known else None,
                "returncode_known": bool(known and returncode is not None),
                "process_group_alive": self._process_group_alive(record),
                "log_tail": self._read_log_tail(record),
            })
        exited_roles = [item["role"] for item in role_records if item["status"] == "exited"]
        context = {
            "captured_at_unix_s": time.time(),
            "capture_reason": "first_observed_process_exit",
            "triggering_roles": exited_roles,
            "run_id": run_id,
            "container_id": group.get("Id"),
            "backend_instance_id": group.get("backend_instance_id"),
            "hub_pid": group.get("hub_pid"),
            "recovered_by_backend_instance_id": group.get("recovered_by_backend_instance_id"),
            "recovered_by_hub_pid": group.get("recovered_by_hub_pid"),
            "group_created_at_unix_s": group.get("created_at_unix_s"),
            "group_manifest_path": str(group.get("manifest")) if group.get("manifest") else None,
            "group_manifest_snapshot": self._read_manifest_snapshot(group.get("manifest")),
            "records": role_records,
        }
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(context, indent=2), encoding="utf-8")
            with self._lock:
                try:
                    if path.exists():
                        context = json.loads(path.read_text(encoding="utf-8"))
                    else:
                        # Hard-link creation is atomic and does not replace an
                        # existing snapshot. If unsupported by the filesystem,
                        # fall back to exclusive creation under the process lock.
                        try:
                            os.link(temp, path)
                        except OSError:
                            try:
                                with path.open("x", encoding="utf-8") as stream:
                                    stream.write(json.dumps(context, indent=2))
                            except FileExistsError:
                                context = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    pass
                finally:
                    try:
                        temp.unlink()
                    except OSError:
                        pass
        return context

    @staticmethod
    def _read_manifest_snapshot(manifest: Any) -> dict[str, Any] | None:
        if not manifest:
            return None
        try:
            payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, TypeError, ValueError):
            return None

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
                manifest_path = Path(manifest)
                terminal_copy = manifest_path.with_name(
                    f"native_terminal_manifest_{group.get('Labels', {}).get('aicar.role', 'unknown')}.json"
                )
                if manifest_path.is_file() and not terminal_copy.exists():
                    try:
                        with terminal_copy.open("x", encoding="utf-8") as stream:
                            stream.write(manifest_path.read_text(encoding="utf-8"))
                    except FileExistsError:
                        pass
                manifest_path.unlink()
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
            terminal = self._capture_terminal_context(group) if state == "exited" else None
            result.append({key: value for key, value in group.items() if key != "records"} | {
                "State": state,
                **({"TerminalContext": terminal} if terminal is not None else {}),
            })
        return result

    def list_official_run_networks(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            networks = list(self._networks.values())
        return [row for row in networks if not run_id or row["Labels"].get("aicar.run_id") == run_id]
