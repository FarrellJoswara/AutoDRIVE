"""Run one Docker official environment's ROS bridge, relay, and policy worker.

The official simulator is started by OfficialRunManager only after this
supervisor reports that the ROS Devkit is connected and the relay is serving.
Each child gets its own log under the run output directory; the supervisor's
stdout remains the Docker lifecycle/readiness channel.
"""

from __future__ import annotations

import json
import argparse
import os
import signal
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


class DockerOfficialRunSupervisor:
    def __init__(self) -> None:
        self.env = os.environ.copy()
        self.run_id = self.env.get("AICAR_RUN_ID", "unknown")
        self.index = int(self.env.get("AICAR_OFFICIAL_ENV_INDEX", "0"))
        self.mode = self.env.get("AICAR_MODE", "train").strip().lower()
        if self.mode not in {"train", "evaluate", "bridge"}:
            raise ValueError(f"unsupported AICAR_MODE: {self.mode}")
        self.domain = int(self.env.get("ROS_DOMAIN_ID", "86"))
        self.devkit_port = int(self.env.get("AICAR_DEVKIT_PORT", "4567"))
        self.relay_port = int(self.env.get("AICAR_RELAY_PORT", "4568"))
        self.workspace = Path(self.env.get("AICAR_AUTODRIVE_WORKSPACE", "/opt/autodrive_ws"))
        self.output_dir = Path(self.env.get("AICAR_TRAIN_OUT", f"/runs/rl/official_run_{self.run_id}"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env["ROS_DOMAIN_ID"] = str(self.domain)
        self.env["AICAR_ROS_DOMAIN_ID_BASE"] = str(self.domain)
        self.children: list[tuple[str, subprocess.Popen[bytes], Any]] = []
        self.stop_requested = threading.Event()

    def _spawn(self, role: str, command: list[str], *, env: dict[str, str]) -> subprocess.Popen[bytes]:
        log_path = self.output_dir / f"{role}_env{self.index}.log"
        log = log_path.open("a+b", buffering=0)
        process = subprocess.Popen(
            command,
            cwd="/app",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.children.append((role, process, log))
        _emit("child_started", role=role, pid=process.pid, log=str(log_path), command=command)
        return process

    def _start_bridge_and_relay(self) -> None:
        if not (Path("/opt/ros/jazzy/setup.bash").is_file() and self.workspace.is_dir()):
            raise RuntimeError("ROS Jazzy or the pinned AutoDRIVE workspace is missing from this API image")
        bridge_env = self.env.copy()
        bridge_env.update({
            "ROS_DOMAIN_ID": str(self.domain),
            "AICAR_DEVKIT_PORT": str(self.devkit_port),
            "PYTHONNOUSERSITE": "1",
            # `ros2 launch` starts the installed bridge script with the system
            # interpreter. Expose the pinned app venv packages (including the
            # official bridge's socketio/gevent imports) to that
            # interpreter while retaining Jazzy's apt-installed modules.
            "PYTHONPATH": os.pathsep.join(filter(None, [
                str(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"),
                self.env.get("PYTHONPATH", ""),
            ])),
        })
        setup = (
            "set +u; source /opt/ros/jazzy/setup.bash; "
            f"source {self.workspace}/install/setup.bash; set -u; "
            "exec ros2 launch autodrive_roboracer bringup_headless.launch.py"
        )
        bridge = self._spawn("devkit", ["bash", "-lc", setup], env=bridge_env)

        relay_env = self.env.copy()
        relay_env.update({"ROS_DOMAIN_ID": str(self.domain), "PYTHONNOUSERSITE": "1"})
        relay = self._spawn(
            "relay",
            [sys.executable, "-u", "-m", "src.layer1.official_bridge_relay",
             "--host", "0.0.0.0", "--port", str(self.relay_port),
             "--devkit-url", f"http://127.0.0.1:{self.devkit_port}"],
            env=relay_env,
        )
        deadline = time.monotonic() + float(self.env.get("AICAR_BRIDGE_STARTUP_TIMEOUT_S", "45"))
        while time.monotonic() < deadline:
            for role, process, _log in self.children:
                if process.poll() is not None:
                    raise RuntimeError(f"{role} exited during startup with code {process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", self.relay_port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError(f"official relay did not listen on port {self.relay_port}")
        # The relay binds only after it has connected to the Devkit upstream.
        _emit("ready", run_id=self.run_id, mode=self.mode, env=self.index,
              ros_domain=self.domain, relay_port=self.relay_port,
              devkit_port=self.devkit_port)

    def _policy_command(self) -> list[str]:
        from src.layer4.hub.native_official_backend import NativeOfficialBackend

        if self.mode == "evaluate":
            command = NativeOfficialBackend._evaluation_command(self.env)
        else:
            command = NativeOfficialBackend._training_command(self.env)
        setup = (
            "set +u; source /opt/ros/jazzy/setup.bash; "
            f"source {self.workspace}/install/setup.bash; set -u; "
            f"exec {shlex.join(command)}"
        )
        return ["bash", "-lc", setup]

    def _failure_context(self, error: BaseException) -> None:
        _emit("startup_or_runtime_error", error=f"{type(error).__name__}: {error}")
        for role, _process, log in self.children:
            try:
                log.flush()
                log.seek(0, os.SEEK_END)
                size = log.tell()
                log.seek(max(0, size - 6000))
                detail = log.read().decode("utf-8", errors="replace")
            except (OSError, AttributeError):
                detail = ""
            _emit("child_log_tail", role=role, tail=detail)

    def run(self) -> int:
        # Resolve the policy command before announcing readiness. Constructing
        # it imports the native backend and its training CLI, which can take a
        # couple of seconds. If deferred until after the simulator is launched,
        # an early Unity exit can cause manager cleanup to terminate this API
        # process before the policy child is even spawned, leaving an empty
        # policy log and no actionable startup error.
        policy_command: list[str] | None = None
        if self.mode != "bridge":
            _emit("policy_command_prepare", mode=self.mode)
            policy_command = self._policy_command()
            _emit("policy_command_prepared", mode=self.mode)
        self._start_bridge_and_relay()
        if self.mode == "bridge":
            return self._wait_for_stop()
        gate_value = self.env.get("AICAR_POLICY_START_GATE", "").strip()
        if gate_value:
            gate = Path(gate_value)
            deadline = time.monotonic() + float(self.env.get("AICAR_POLICY_START_GATE_TIMEOUT_S", "180"))
            _emit("policy_gate_wait", path=str(gate))
            while not gate.is_file() and time.monotonic() < deadline:
                if self.stop_requested.wait(0.1):
                    return 0
                for role, process, _log in self.children:
                    if process.poll() is not None:
                        raise RuntimeError(f"{role} exited before policy gate with code {process.returncode}")
            if not gate.is_file():
                raise TimeoutError(f"policy start gate was not released: {gate}")
            _emit("policy_gate_released", path=str(gate))
        if policy_command is None:
            raise RuntimeError("policy command was not prepared before readiness")
        policy = self._spawn("policy", policy_command, env=self.env)
        while not self.stop_requested.wait(0.1):
            if policy.poll() is not None:
                _emit("policy_exited", run_id=self.run_id, returncode=policy.returncode)
                return int(policy.returncode or 0)
            for role, process, _log in self.children:
                if role != "policy" and process.poll() is not None:
                    raise RuntimeError(f"{role} exited unexpectedly with code {process.returncode}")
        return int(policy.poll() or 0)

    def _wait_for_stop(self) -> int:
        while not self.stop_requested.wait(0.25):
            for role, process, _log in self.children:
                if process.poll() is not None:
                    raise RuntimeError(f"{role} exited unexpectedly with code {process.returncode}")
        return 0

    def shutdown(self) -> None:
        self.stop_requested.set()
        for role, process, _log in reversed(self.children):
            if process.poll() is None:
                _emit("child_stopping", role=role, pid=process.pid)
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            try:
                _log.close()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "bridge"))
    args = parser.parse_args()
    if args.mode:
        os.environ["AICAR_MODE"] = args.mode
    supervisor: DockerOfficialRunSupervisor | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        _emit("shutdown_requested", signal=signum)
        supervisor.stop_requested.set()

    try:
        supervisor = DockerOfficialRunSupervisor()
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        return supervisor.run()
    except Exception as exc:
        if supervisor is not None:
            supervisor._failure_context(exc)
        else:
            _emit("startup_or_runtime_error", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if supervisor is not None:
            supervisor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
