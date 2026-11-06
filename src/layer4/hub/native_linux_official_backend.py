"""Native-process backend for the official Linux AutoDRIVE player."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Any

from src.layer4.hub.linux_official_simulators import (
    start_linux_simulators,
    stop_linux_simulators,
    validate_linux_simulator_paths,
)
from src.layer4.hub.native_official_backend import NativeOfficialBackend
from src.layer4.hub.native_official_backend import simulator_session_ready


class NativeLinuxOfficialBackend(NativeOfficialBackend):
    """Use local ROS/Layer 1 processes and the extracted official Linux player."""

    supports_linux_external = True
    required_simulator_backend = "linux_external"
    relay_bind_host = "127.0.0.1"

    def __init__(self, *, startup_timeout_s: float = 45.0):
        ros_setup = Path(os.environ.get("AICAR_NATIVE_ROS_SETUP", "")).expanduser()
        ros_workspace = Path(os.environ.get("AICAR_NATIVE_ROS_WORKSPACE", "")).expanduser()
        python_executable = Path(os.environ.get(
            "AICAR_NATIVE_PYTHON", sys.executable
        )).expanduser()
        self.native_ros_setup = ros_setup
        self.native_ros_workspace = ros_workspace
        self.native_python = python_executable
        super().__init__(
            ros_workspace=ros_workspace,
            ros_setup_script=ros_setup,
            startup_timeout_s=startup_timeout_s,
        )
        self.simulator_host = "127.0.0.1"
        self._linux_launch_states: dict[str, dict[str, Any]] = {}

    def validate_linux_runtime(self, config: dict[str, Any] | None = None) -> None:
        validate_linux_simulator_paths(config)
        workspace_setup = self.native_ros_workspace / "install" / "setup.bash"
        missing = []
        if not self.native_ros_setup.is_file():
            missing.append(f"AICAR_NATIVE_ROS_SETUP is missing: {self.native_ros_setup}")
        if not workspace_setup.is_file():
            missing.append(f"AICAR_NATIVE_ROS_WORKSPACE must contain {workspace_setup}")
        if not self.native_python.is_file() or not os.access(self.native_python, os.X_OK):
            missing.append(f"native Python environment is unavailable: {self.native_python}")
        if Path(sys.executable).resolve() != self.native_python.resolve():
            missing.append(
                f"start Mission Control with AICAR_NATIVE_PYTHON={self.native_python}; "
                f"current interpreter is {sys.executable}"
            )
        if missing:
            raise RuntimeError("native Linux backend dependencies are not configured: " + "; ".join(missing))
        setup_command = (
            f"source {shlex.quote(str(self.native_ros_setup.resolve()))}; "
            f"source {shlex.quote(str(workspace_setup.resolve()))}; "
            f"{shlex.quote(str(self.native_python.resolve()))} -c 'import rclpy; import socketio; import aiohttp' "
            "&& ros2 pkg prefix autodrive_roboracer"
        )
        checked = subprocess.run(
            ["bash", "-lc", setup_command], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
        if checked.returncode:
            raise RuntimeError(
                "native Linux ROS setup could not import rclpy and autodrive_roboracer: "
                + checked.stdout[-3000:]
            )

    def start_simulators(
        self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Start wrapper-owned Xvfb/player process groups and require camera streams."""
        config = config or {}
        runtime, executable, _wrapper = validate_linux_simulator_paths(config)
        profile = config.get("track_profile", "practice")
        launch_config = {
            "track_profile": profile,
            "simulator_executable": str(executable),
            "native_runtime": str(runtime),
        }
        state = {
            "config": launch_config,
            "connections": [dict(item) for item in connections],
            "hub_url": hub_url,
        }
        if not hasattr(self, "_linux_launch_states"):
            self._linux_launch_states = {}
        self._linux_launch_states[run_id] = state
        run_dir = self._run_dir_for_id(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "linux_simulator_lifecycle.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        start_linux_simulators(run_id=run_id, connections=connections, config=config)
        try:
            self._wait_for_simulator_streams(
                connections, self._run_dir_for_id(run_id)
            )
        except Exception:
            stop_linux_simulators(run_id)
            raise

    @staticmethod
    def stop_simulators(run_id: str) -> None:
        stop_linux_simulators(run_id)

    def _linux_state(self, run_id: str, connections: list[dict[str, Any]], hub_url: str) -> dict[str, Any]:
        state = getattr(self, "_linux_launch_states", {}).get(run_id)
        if state is None:
            path = self._run_dir_for_id(run_id) / "linux_simulator_lifecycle.json"
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Linux simulator lifecycle state is unavailable for {run_id}") from exc
            self._linux_launch_states[run_id] = state
        stored = {int(item["environment"]): int(item["port"]) for item in state["connections"]}
        requested = {int(item["environment"]): int(item["port"]) for item in connections}
        if stored != requested:
            raise RuntimeError("Linux evaluation handoff connections do not match the run's saved simulator ports")
        state["hub_url"] = hub_url
        return state

    def _linux_session_sids(self, connections: list[dict[str, Any]]) -> dict[str, list[str]]:
        result = {}
        for item in connections:
            health = self._read_simulator_health(int(item["port"])) or {}
            sessions = health.get("simulator_sessions")
            result[str(int(item["environment"]))] = [
                str(session["sid"]) for session in sessions or []
                if isinstance(session, dict) and session.get("sid")
            ]
        return result

    def _wait_for_linux_fresh_streams(
        self, *, run_id: str, connections: list[dict[str, Any]], old_sids: dict[str, list[str]],
    ) -> dict[int, dict[str, Any]]:
        deadline = time.monotonic() + max(45.0, self.startup_timeout_s)
        last: dict[int, Any] = {}
        while time.monotonic() < deadline:
            ready = True
            for item in connections:
                index, port = int(item["environment"]), int(item["port"])
                health = self._read_simulator_health(port)
                last[index] = health if health is not None else "health endpoint unavailable"
                sessions = health.get("simulator_sessions") if health else None
                old_for_environment = old_sids.get(str(index), [])
                has_ready_new_sid = isinstance(sessions, list) and any(
                    isinstance(session, dict)
                    and isinstance(session.get("sid"), str)
                    and session["sid"]
                    and session["sid"] not in old_for_environment
                    and int(session.get("complete_frames", 0)) >= 1
                    and int(session.get("camera_frames", 0)) >= 1
                    and int(session.get("devkit_commands", 0)) >= 1
                    for session in sessions
                )
                if (health is None or int(health.get("simulator_clients", 0)) < 1
                        or not has_ready_new_sid):
                    ready = False
            if ready:
                (self._run_dir_for_id(run_id) / "simulator_stream_health.json").write_text(
                    json.dumps({"environments": last}, indent=2), encoding="utf-8"
                )
                return {int(index): value for index, value in last.items() if isinstance(value, dict)}
            time.sleep(0.5)
        raise TimeoutError(
            "restarted native Linux simulator did not establish fresh Socket.IO/camera streams "
            f"with new relay session ids: {json.dumps(last, sort_keys=True)}"
        )

    def pause_training_for_evaluation(
        self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str,
    ) -> None:
        """Use the shared PPO checkpoint handshake, then retire Linux player groups by manifest."""
        state = self._linux_state(run_id, connections, hub_url)
        with self._lock:
            group = next((item for item in self._groups.values()
                          if item.get("Labels", {}).get("aicar.run_id") == run_id
                          and item.get("Labels", {}).get("aicar.role") == "api"), None)
            trainer = next((item for item in (group or {}).get("records", ())
                            if item.get("role") == "trainer"), None)
        if trainer is None or group is None:
            raise RuntimeError(f"training process for {run_id} is unavailable for evaluation pause")
        process = trainer["process"]
        if process.poll() is not None:
            raise RuntimeError(f"training process for {run_id} has already exited")
        run_dir = self._run_dir_for_id(run_id)
        pause_file = run_dir / "evaluation_pause.flag"
        request_file = run_dir / "evaluation_pause.request.json"
        ack_file = run_dir / "evaluation_pause.ack.json"
        resume_request_file = run_dir / "evaluation_resume.request.json"
        if pause_file.exists() or request_file.exists():
            raise RuntimeError(f"training run {run_id} is already paused for evaluation")
        ack_file.unlink(missing_ok=True)
        resume_request_file.unlink(missing_ok=True)
        (run_dir / "evaluation_resume.ack.json").unlink(missing_ok=True)
        request_id = uuid.uuid4().hex
        request = {"run_id": run_id, "request_id": request_id, "requested_at_unix_s": time.time(),
                   "protocol": "cooperative_post_optimizer_rollout_boundary_v1"}
        self._write_json_atomic(request_file, request)
        pause_ack = None
        retired = False
        watch_step_before_pause = self._watch_step(run_dir)
        old_sids = self._linux_session_sids(connections)
        try:
            pause_ack = self._wait_for_training_handshake(
                run_id=run_id, process=process, path=ack_file, request_id=request_id,
                phase="pause", expected_status="paused", timeout_s=max(240.0, self.startup_timeout_s),
            )
            pause_ack = dict(pause_ack)
            pause_ack["checkpoint"] = self._verify_pause_checkpoint(run_dir, pause_ack)
            state.update({"paused": True, "request_id": request_id, "pause_ack": pause_ack,
                          "old_sids": old_sids})
            self._write_json_atomic(run_dir / "linux_simulator_lifecycle.json", state)
            group["evaluation_pause_request_id"] = request_id
            group["evaluation_pause_ack"] = pause_ack
            baselines = {}
            for item in connections:
                index = int(item["environment"])
                health = self._read_simulator_health(int(item["port"])) or {}
                baselines[index] = {
                    "complete": int(health.get("complete_frames", 0)),
                    "camera": int(health.get("camera_frames", 0)),
                }
            pause_file.write_text(json.dumps({"run_id": run_id, "request_id": request_id,
                                             "frames": baselines}), encoding="utf-8")
            # Treat retirement as attempted before entering the helper: an OS
            # error can occur after it has already signalled some groups.
            retired = True
            stop_linux_simulators(run_id)
            self._wait_for_simulator_sessions_disconnected(connections)
            group["linux_evaluation_pause"] = state
        except Exception as pause_error:
            recovery_error = None
            try:
                # If retirement began, restore the selected pinned player before
                # releasing the callback. A failed restart leaves its durable
                # post-update checkpoint and the matching resume request intact.
                if retired:
                    stop_linux_simulators(run_id)
                    start_linux_simulators(run_id=run_id, connections=connections, config=state["config"])
                    self._wait_for_linux_fresh_streams(run_id=run_id, connections=connections, old_sids=old_sids)
                request_file.unlink(missing_ok=True)
                self._write_json_atomic(resume_request_file, request)
                if pause_ack is not None:
                    resume_ack = self._wait_for_training_handshake(
                        run_id=run_id, process=process,
                        path=run_dir / "evaluation_resume.ack.json", request_id=request_id,
                        phase="resume", expected_status="resumed", timeout_s=max(30.0, self.startup_timeout_s),
                    )
                    self._wait_for_training_observation_after_resume(
                        run_id=run_id, run_dir=run_dir, process=process,
                        previous_step=watch_step_before_pause,
                    )
                    pause_file.unlink(missing_ok=True)
                    resume_request_file.unlink(missing_ok=True)
                    (run_dir / "evaluation_resume.ack.json").unlink(missing_ok=True)
                    state.update({"paused": False, "request_id": None, "pause_ack": None, "old_sids": {}})
            except Exception as exc:
                recovery_error = exc
            state.update({"paused": False, "pause_recovery_error": str(recovery_error) if recovery_error else None})
            self._write_json_atomic(run_dir / "linux_simulator_lifecycle.json", state)
            if recovery_error:
                raise RuntimeError(f"Linux training pause failed ({pause_error}); recovery failed ({recovery_error}); checkpoint remains in {run_dir}") from pause_error
            raise

    def resume_training_after_evaluation(
        self, *, run_id: str, connections: list[dict[str, Any]], hub_url: str,
    ) -> None:
        state = self._linux_state(run_id, connections, hub_url)
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
        request_id = state.get("request_id") or (group or {}).get("evaluation_pause_request_id")
        if not isinstance(request_id, str):
            raise RuntimeError(f"training run {run_id} has no cooperative pause request to resume")
        old_sids = state.get("old_sids") or {}
        previous_step = self._watch_step(run_dir)
        resume_request_path = run_dir / "evaluation_resume.request.json"
        pause_file = run_dir / "evaluation_pause.flag"
        resume_request = {"run_id": run_id, "request_id": request_id,
                          "requested_at_unix_s": time.time(),
                          "protocol": "cooperative_post_optimizer_rollout_boundary_v1"}
        resume_ack = None
        try:
            # Retire any stale manifest-owned group before relaunch. This is
            # idempotent and never searches for or kills unrelated processes.
            stop_linux_simulators(run_id)
            self._wait_for_simulator_sessions_disconnected(connections)
            start_linux_simulators(run_id=run_id, connections=connections, config=state["config"])
            self._wait_for_linux_fresh_streams(run_id=run_id, connections=connections, old_sids=old_sids)
            if process.poll() is not None:
                raise RuntimeError(f"training process for {run_id} exited before resume")
            pause_file.unlink(missing_ok=True)
            self._write_json_atomic(resume_request_path, resume_request)
            resume_ack = self._wait_for_training_handshake(
                run_id=run_id, process=process, path=run_dir / "evaluation_resume.ack.json",
                request_id=request_id, phase="resume", expected_status="resumed",
                timeout_s=max(120.0, self.startup_timeout_s),
            )
            resumed_step = self._wait_for_training_observation_after_resume(
                run_id=run_id, run_dir=run_dir, process=process, previous_step=previous_step,
            )
            event = {"run_id": run_id, "previous_watch_step": previous_step,
                     "resumed_watch_step": resumed_step, "acknowledged_at_unix_s": time.time(),
                     "fresh_simulator_session_required": True,
                     "cooperative_pause_ack": state.get("pause_ack"), "resume_ack": resume_ack}
            with (run_dir / "evaluation_resume_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            state.update({"paused": False, "request_id": None, "pause_ack": None, "old_sids": {}})
            self._write_json_atomic(run_dir / "linux_simulator_lifecycle.json", state)
        except Exception as resume_error:
            state["resume_error"] = str(resume_error)
            self._write_json_atomic(run_dir / "linux_simulator_lifecycle.json", state)
            if process.poll() is None and resume_ack is None:
                self._write_json_atomic(resume_request_path, resume_request)
                pause_file.write_text(json.dumps({"run_id": run_id, "request_id": request_id,
                                                  "recovery_required": True}), encoding="utf-8")
            raise RuntimeError(f"Linux training resume failed; post-update checkpoint remains in {run_dir}: {resume_error}") from resume_error
        finally:
            if group is not None:
                group.pop("evaluation_frame_baselines", None)
                group.pop("evaluation_pause_request_id", None)
                group.pop("evaluation_pause_ack", None)
                group.pop("linux_evaluation_pause", None)
            (run_dir / "evaluation_pause.request.json").unlink(missing_ok=True)
            (run_dir / "evaluation_pause.ack.json").unlink(missing_ok=True)
            if resume_ack is not None:
                resume_request_path.unlink(missing_ok=True)
                (run_dir / "evaluation_resume.ack.json").unlink(missing_ok=True)
