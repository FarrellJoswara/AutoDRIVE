"""Racer: Individual AutoDRIVE Vehicle Controller and Simulator Bridge.

Manages Socket.IO server communications via gevent/WSGI (matching the official
AutoDRIVE Devkit bridge), waits for fresh Bridge responses, profiles execution,
and manages the child simulator process lifecycle.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import shlex
import subprocess
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import socketio
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from .telemetry import TelemetrySnapshot, TrajectoryLogger

logger = logging.getLogger(__name__)


def _read_active_map_id() -> Optional[str]:
    """Read simulator/maps/.active_map.json without importing Layer 4."""
    env_dir = os.environ.get("AICAR_MAPS_DIR", "").strip()
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir) / ".active_map.json")
    here = Path(__file__).resolve()
    repo = here.parents[2]
    candidates.append(repo / "simulator" / "maps" / ".active_map.json")
    candidates.append(Path("/app/simulator/maps/.active_map.json"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import json

            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        mid = raw.get("id")
        if isinstance(mid, str) and mid.strip() and mid.strip().lower() != "none":
            return mid.strip()
    return None


def _get_wsl_host_ip() -> str:
    """Retrieve Windows host IP address from WSL /etc/resolv.conf."""
    try:
        res = subprocess.run(["wsl", "cat", "/etc/resolv.conf"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            if line.strip().startswith("nameserver"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    return parts[1]
    except Exception:
        pass
    return "127.0.0.1"


class Racer:
    """Controls a single AutoDRIVE vehicle instance and its simulator connection."""

    def __init__(
        self,
        racer_id: int = 0,
        port: int = 4567,
        simulator_path: Optional[Union[str, Path]] = None,
        auto_launch: bool = False,
        headless: bool = True,
        step_timeout: float = 5.0,
        enable_logging: bool = True,
        action_interval_s: Optional[float] = None,
        simulator_log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.racer_id = racer_id
        self.port = port
        self.simulator_path = Path(simulator_path) if simulator_path else None
        self.auto_launch = auto_launch
        self.headless = headless
        self.simulator_log_path = simulator_log_path
        self.step_timeout = step_timeout
        if action_interval_s is not None and (
            not math.isfinite(float(action_interval_s)) or float(action_interval_s) <= 0.0
        ):
            raise ValueError("action_interval_s must be a finite positive number")
        self.action_interval_s = (
            None if action_interval_s is None else float(action_interval_s)
        )
        self._explicit_mode = self.action_interval_s is not None
        self._explicit_protocol_verified = False
        self._explicit_protocol_error: Optional[str] = None
        self._incompatible_protocol_error: Optional[str] = None
        self._explicit_timeout_error = False
        self._explicit_pending_step_id: Optional[int] = None
        self._explicit_next_step_id = 1
        self._explicit_sim_time: Optional[float] = None
        self.physics_ticks: Optional[int] = None
        self._explicit_expected_physics_ticks: Optional[int] = None
        self._explicit_last_response_id: Optional[int] = None
        self._explicit_last_response_data: Optional[Dict[str, Any]] = None
        self.latest_bridge_keys: frozenset[str] = frozenset()
        self._bridge_condition = threading.Condition()

        # Telemetry & Logger
        self.telemetry: TelemetrySnapshot = TelemetrySnapshot()
        self.logger: TrajectoryLogger = TrajectoryLogger()
        self.enable_logging = enable_logging

        # Control Command Buffers
        self._cmd_throttle: float = 0.0
        self._cmd_steering: float = 0.0
        self._reset_requested: bool = False
        self._reset_command_step: Optional[int] = None

        # Profiler Metrics
        self.step_counter: int = 0
        self.step_latency_ms: float = 0.0
        self.steps_per_second: float = 0.0
        self.real_time_factor: float = 0.0
        self._last_step_time: float = (
            time.monotonic() if self._explicit_mode else time.time()
        )

        # Synchronization Primitives
        self._step_event = threading.Event()
        self._reset_sent_event = threading.Event()
        self._reset_event = threading.Event()
        self._simulation_pause_ack = threading.Event()
        self._simulation_resume_ack = threading.Event()
        self._simulation_pause_lock = threading.Lock()
        self._simulation_paused = False
        self._simulation_pause_started_at: Optional[float] = None
        self._simulation_paused_total_s = 0.0
        self._connected = False
        self._is_alive = True
        self.client_sid: Optional[str] = None

        # Process & Server Handles
        self._sim_process: Optional[subprocess.Popen] = None
        self._wsgi_server: Optional[pywsgi.WSGIServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_ready = threading.Event()

        # Socket.IO gevent server (matches official AutoDRIVE Devkit bridge)
        self.sio = socketio.Server(async_mode="gevent", cors_allowed_origins="*")
        self._register_handlers()
        self._start_server()

        # Auto-launch child process if requested
        if self.auto_launch and self.simulator_path:
            self.launch_simulator()

    def _register_handlers(self) -> None:
        """Register Socket.IO event handlers for the AutoDRIVE simulator."""

        # Unity's Socket.IO client often emits 'Bridge' before completing the
        # formal namespace connect handshake. Without this, events are dropped and
        # is_connected stays False even though TCP/WebSocket is up.
        orig_handle_event = self.sio._handle_event

        def _auto_connect_handle_event(eio_sid, namespace, id, data):
            ns = namespace or "/"
            sid = self.sio.manager.sid_from_eio_sid(eio_sid, ns)
            if sid is None or not self.sio.manager.is_connected(sid, ns):
                self.sio._handle_connect(eio_sid, ns, None)
                sid = self.sio.manager.sid_from_eio_sid(eio_sid, ns)
                self.client_sid = sid
                self._connected = True
                logger.info(
                    f"[Racer {self.racer_id}] Auto-connected Unity client sid={sid} on port {self.port}"
                )
            return orig_handle_event(eio_sid, namespace, id, data)

        self.sio._handle_event = _auto_connect_handle_event

        @self.sio.on("connect")
        def connect(sid, environ):
            self.client_sid = sid
            self._connected = True
            logger.info(f"[Racer {self.racer_id}] Connected to simulator client sid={sid} on port {self.port}")

        @self.sio.on("disconnect")
        def disconnect(sid):
            self._connected = False
            self.client_sid = None
            logger.warning(f"[Racer {self.racer_id}] Disconnected from simulator on port {self.port}")

        @self.sio.on("Bridge")
        def on_bridge(sid, data: Dict[str, Any]):
            """Handle a Unity Bridge response and reply via emit (official protocol)."""
            self._on_bridge(sid, data)

        @self.sio.on("AICAR_SIMULATION_PAUSE_ACK")
        def on_simulation_pause_ack(sid, data=None):
            self._mark_simulation_paused()
            self._simulation_pause_ack.set()

        @self.sio.on("AICAR_SIMULATION_RESUME_ACK")
        def on_simulation_resume_ack(sid, data=None):
            self._mark_simulation_resumed()
            self._simulation_resume_ack.set()

    def _on_bridge(self, sid: str, data: Dict[str, Any]) -> None:
        """Accept one telemetry packet and run legacy or request/response mode."""
        if getattr(self, "_explicit_mode", False):
            self._on_explicit_bridge(sid, data)
            return

        if not data:
            return
        if str(data.get("AICAR Step Protocol", "")) == "1":
            self.client_sid = sid
            self._connected = False
            self._incompatible_protocol_error = (
                "Unity is running the explicit action-step protocol, but this Racer was "
                "created without action_interval_s; configure the same action interval "
                "on the Racer/AutoDriveEnv"
            )
            logger.error("[Racer %s] %s", self.racer_id, self._incompatible_protocol_error)
            return

        if not data:
            return

        self._connected = True
        self.client_sid = sid
        self.step_counter += 1

        # Update Telemetry Snapshot
        self.telemetry = TelemetrySnapshot.from_raw_dict(data, step_id=self.step_counter)

        # Record trajectory step
        if self.enable_logging:
            self.logger.log(self.telemetry)

        # Check if reset was requested
        reset_flag = self._reset_requested
        if self._reset_requested:
            self._reset_requested = False
            # This Bridge reply is the reset command, not an acknowledgment
            # that Unity has applied it. Wait for a later telemetry frame.
            self._reset_command_step = self.step_counter
            self._reset_sent_event.set()
        elif self._reset_command_step is not None:
            if self.step_counter > self._reset_command_step:
                self._reset_command_step = None
                self._reset_event.set()

        # Snapshot current commands before waking up caller in step()
        cmd_th = float(self._cmd_throttle)
        cmd_st = float(self._cmd_steering)

        # Wake up caller waiting in step()
        self._step_event.set()

        # Official AutoDRIVE reply: emit string-valued Bridge payload only (no ACK return)
        try:
            self.sio.emit(
                "Bridge",
                data={
                    "V1 Throttle": str(cmd_th),
                    "V1 Steering": str(cmd_st),
                    "V1 Reset": str(bool(reset_flag)),
                },
                to=sid,
            )
        except Exception as exc:
            logger.warning(f"[Racer {self.racer_id}] Failed to emit Bridge commands: {exc}")

    @staticmethod
    def _metadata_float(data: Dict[str, Any], key: str) -> float:
        try:
            value = float(data[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"explicit Bridge response is missing valid {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"explicit Bridge response has non-finite {key}")
        return value

    @staticmethod
    def _metadata_int(data: Dict[str, Any], key: str) -> int:
        try:
            raw = data[key]
            value = int(raw)
            if str(value) != str(raw).strip():
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"explicit Bridge response is missing valid {key}") from exc
        return value

    def _set_explicit_error(self, message: str) -> None:
        with self._bridge_condition:
            if self._explicit_protocol_error is None:
                self._explicit_protocol_error = message
            self._bridge_condition.notify_all()

    def _on_explicit_bridge(self, sid: str, data: Dict[str, Any]) -> None:
        if not data:
            self._set_explicit_error("explicit Bridge response was empty")
            return
        self._connected = True
        self.client_sid = sid
        self.latest_bridge_keys = frozenset(str(key) for key in data)

        try:
            if str(data.get("AICAR Step Protocol", "")) != "1":
                raise ValueError("explicit Bridge protocol metadata is missing or unsupported")
            advertised_interval = self._metadata_float(data, "AICAR Action Interval")
            sim_time = self._metadata_float(data, "AICAR Sim Time")
            response_id = self._metadata_int(data, "AICAR Step ID")
            physics_ticks = self._metadata_int(data, "AICAR Physics Ticks")
            raw_step_error = data.get("AICAR Step Error")
            step_error = "" if raw_step_error is None else str(raw_step_error).strip()
            if step_error:
                raise RuntimeError(f"Unity rejected explicit step: {step_error}")
            if not math.isclose(
                advertised_interval, float(self.action_interval_s), rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"Unity action interval {advertised_interval:g}s does not match requested "
                    f"{self.action_interval_s:g}s"
                )

            with self._bridge_condition:
                if not self._explicit_protocol_verified:
                    if response_id != 0 or not math.isclose(sim_time, 0.0, abs_tol=1e-9) or physics_ticks != 0:
                        raise ValueError(
                            "initial explicit handshake must use step ID 0, sim time 0, "
                            f"and zero physics ticks (got id={response_id}, time={sim_time:g}, "
                            f"ticks={physics_ticks})"
                        )
                    self._explicit_protocol_verified = True
                else:
                    expected_id = self._explicit_pending_step_id
                    if expected_id is None or response_id != expected_id:
                        if (
                            response_id == self._explicit_last_response_id
                            and self._explicit_last_response_data == self._explicit_payload_without_cameras(data)
                        ):
                            # Socket.IO may redeliver the latest handshake/ACK.
                            # A byte-equivalent transition packet is harmless and
                            # must not advance telemetry or the simulated clock.
                            return
                        if expected_id is None:
                            detail = f"unsolicited explicit Bridge response for step ID {response_id}"
                        elif response_id > expected_id:
                            detail = f"explicit Bridge step ID mismatch: expected {expected_id}, got {response_id}"
                        else:
                            detail = f"stale or out-of-order explicit Bridge step ID {response_id}"
                        raise ValueError(
                            detail if expected_id is None or response_id > expected_id
                            else f"{detail}; expected {expected_id}"
                        )
                    previous_sim_time = self._explicit_sim_time
                    if previous_sim_time is None or not math.isclose(
                        sim_time - previous_sim_time,
                        advertised_interval,
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(
                            f"explicit Unity simulation time delta must equal action interval "
                            f"{advertised_interval:g}s (got "
                            f"{(sim_time - previous_sim_time) if previous_sim_time is not None else float('nan'):g}s)"
                        )
                    if physics_ticks <= 0:
                        raise ValueError(
                            f"explicit response step {response_id} reports no physics ticks"
                        )
                    if self._explicit_expected_physics_ticks is None:
                        self._explicit_expected_physics_ticks = physics_ticks
                    elif physics_ticks != self._explicit_expected_physics_ticks:
                        raise ValueError(
                            "explicit response physics tick count changed: expected "
                            f"{self._explicit_expected_physics_ticks}, got {physics_ticks}"
                        )

                self.step_counter += 1
                self.telemetry = TelemetrySnapshot.from_raw_dict(data, step_id=response_id)
                self._explicit_sim_time = sim_time
                self.physics_ticks = physics_ticks
                self._explicit_last_response_id = response_id
                self._explicit_last_response_data = self._explicit_payload_without_cameras(data)
                if self.enable_logging:
                    self.logger.log(self.telemetry)
                if response_id != 0:
                    self._explicit_pending_step_id = None
                self._bridge_condition.notify_all()
        except Exception as exc:
            self._set_explicit_error(str(exc))

    def _emit_socket_event(self, event: str, *, sid: str) -> None:
        """Emit a simulator control event on the Socket.IO server's gevent loop.

        Gym's subprocess worker calls pause/resume from its normal thread, while
        the Socket.IO server is serviced by a separate gevent loop thread. The
        explicit Bridge action path already marshals sends onto that loop; these
        control events must do the same or Unity may never receive them.
        """
        server = getattr(self, "_wsgi_server", None)
        loop = getattr(server, "loop", None) if server is not None else None
        if loop is None:
            # Test doubles and manually managed clients have no gevent loop.
            self.sio.emit(event, to=sid)
            return

        sent = threading.Event()
        errors = []

        def emit_on_server_loop() -> None:
            try:
                self.sio.emit(event, to=sid)
            except Exception as exc:
                errors.append(exc)
            finally:
                sent.set()

        loop.run_callback_threadsafe(emit_on_server_loop)
        if not sent.wait(timeout=self.step_timeout):
            raise TimeoutError(
                f"[Racer {self.racer_id}] Socket.IO server did not dispatch {event} "
                f"within {self.step_timeout:.1f}s"
            )
        if errors:
            raise RuntimeError(
                f"[Racer {self.racer_id}] Failed to emit {event}: {errors[0]}"
            ) from errors[0]

    @staticmethod
    def _explicit_payload_without_cameras(data: Dict[str, Any]) -> Dict[str, Any]:
        """Copy packet fields used outside optional camera-image transport."""
        return {
            key: value for key, value in data.items()
            if not str(key).endswith(" Camera Image")
        }

    def _wait_for_explicit_handshake(self) -> None:
        deadline = time.monotonic() + self.step_timeout
        with self._bridge_condition:
            while not self._explicit_protocol_verified and self._explicit_protocol_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Unity did not advertise explicit-step protocol on port {self.port} "
                        f"within {self.step_timeout:.1f}s"
                    )
                self._bridge_condition.wait(remaining)
            if self._explicit_protocol_error is not None:
                raise RuntimeError(
                    f"Unity explicit-step handshake failed on port {self.port}: "
                    f"{self._explicit_protocol_error}"
                )

    def _raise_explicit_error(self) -> None:
        if self._explicit_protocol_error is not None:
            message = (
                f"Unity explicit-step protocol failed on port {self.port}: "
                f"{self._explicit_protocol_error}"
            )
            if self._explicit_timeout_error:
                raise TimeoutError(message)
            raise RuntimeError(message)

    def _explicit_request(self, throttle: float, steering: float, reset: bool) -> TelemetrySnapshot:
        if not self._explicit_protocol_verified:
            self._wait_for_explicit_handshake()
        with self._bridge_condition:
            self._raise_explicit_error()
            if self._explicit_pending_step_id is not None:
                raise RuntimeError("an explicit Unity step is already pending")
            step_id = self._explicit_next_step_id
            self._explicit_next_step_id += 1
            self._explicit_pending_step_id = step_id
            sid = self.client_sid
            if not sid:
                self._explicit_pending_step_id = None
                raise RuntimeError("Unity Bridge client disconnected")

        payload = {
            "V1 Throttle": str(float(throttle)),
            "V1 Steering": str(float(steering)),
            "V1 Reset": str(bool(reset)),
            "AICAR Step ID": str(step_id),
        }
        started = time.monotonic()
        try:
            server = getattr(self, "_wsgi_server", None)
            loop = getattr(server, "loop", None) if server is not None else None
            if loop is None:
                # Test doubles and manually managed clients have no gevent
                # server loop; retain the direct emit path for those callers.
                self.sio.emit("Bridge", data=payload, to=sid)
            else:
                def emit_on_server_loop() -> None:
                    try:
                        self.sio.emit("Bridge", data=payload, to=sid)
                    except Exception as exc:
                        self._set_explicit_error(
                            f"failed to send explicit step {step_id}: {exc}"
                        )

                loop.run_callback_threadsafe(partial(emit_on_server_loop))
        except Exception as exc:
            with self._bridge_condition:
                self._explicit_pending_step_id = None
                self._explicit_protocol_error = f"failed to send explicit step {step_id}: {exc}"
                self._explicit_timeout_error = False
                self._bridge_condition.notify_all()
            raise

        deadline = started + self.step_timeout
        with self._bridge_condition:
            while self._explicit_pending_step_id == step_id and self._explicit_protocol_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._explicit_protocol_error = (
                        f"timed out waiting for explicit Bridge response to step ID {step_id}"
                    )
                    self._explicit_timeout_error = True
                    self._explicit_pending_step_id = None
                    self._bridge_condition.notify_all()
                    break
                self._bridge_condition.wait(remaining)
            self._raise_explicit_error()
            if self.telemetry.step_id != step_id:
                raise RuntimeError(
                    f"explicit Bridge response mismatch: expected {step_id}, "
                    f"latest telemetry has step ID {self.telemetry.step_id}"
                )
            return self.telemetry

    def _serve_forever(self) -> None:
        """Create and run the gevent WSGI server inside this thread's hub.

        The server object must be constructed on the same OS thread that runs
        the accept loop; otherwise gevent raises LoopExit and the port never binds
        (common on Windows when the server is built on the main thread).
        """
        app = socketio.WSGIApp(self.sio)
        self._wsgi_server = pywsgi.WSGIServer(
            ("0.0.0.0", self.port),
            app,
            handler_class=WebSocketHandler,
            log=None,
            error_log=None,
        )
        try:
            self._wsgi_server.start()
            self._server_ready.set()
            self._wsgi_server._stop_event.wait()
        except Exception as exc:
            logger.debug(f"[Racer {self.racer_id}] Gevent server exited: {exc}")
        finally:
            self._server_ready.set()  # unblock waiter if start() failed early

    def _start_server(self) -> None:
        """Start the background gevent WSGI server hosting the Socket.IO listener."""
        self._server_ready.clear()
        self._server_thread = threading.Thread(
            target=self._serve_forever,
            name=f"Racer-{self.racer_id}-Gevent-{self.port}",
            daemon=True,
        )
        self._server_thread.start()
        if not self._server_ready.wait(timeout=5.0):
            raise RuntimeError(f"[Racer {self.racer_id}] Gevent server failed to start on port {self.port}")
        # Brief pause to ensure accept loop is active
        time.sleep(0.2)
        logger.info(f"[Racer {self.racer_id}] Listening on 0.0.0.0:{self.port} (gevent/WSGI)")

    def launch_simulator(self) -> None:
        """Launch the standalone Unity simulator subprocess (native or via WSL on Windows)."""
        if not self.simulator_path or not self.simulator_path.exists():
            raise FileNotFoundError(f"Simulator executable not found at: {self.simulator_path}")

        is_windows = platform.system() == "Windows"
        is_linux_elf = self.simulator_path.suffix == ".x86_64"

        flags = []
        if self.headless:
            flags.extend(["-batchmode", "-nographics"])
        elif not is_windows:
            flags.extend(["-force-vulkan"])
        if self.simulator_log_path is not None:
            flags.extend(["-logFile", str(self.simulator_log_path)])

        # Phase 3: pass active map id when set (TrackLoader-enabled player only)
        map_id = os.environ.get("AICAR_MAP_ID", "").strip()
        if not map_id or map_id.lower() == "none":
            map_id = _read_active_map_id() or ""
        if map_id and map_id.lower() != "none":
            flags.extend(["-map-id", map_id])

        if is_windows and is_linux_elf:
            sim_dir = str(self.simulator_path.parent.resolve()).replace("\\", "/")
            drive_letter = sim_dir[0].lower()
            wsl_dir = f"/mnt/{drive_letter}{sim_dir[2:]}"
            sim_name = self.simulator_path.name
            target_ip = _get_wsl_host_ip()
            flags.extend(["-ip", target_ip, "-port", str(self.port)])
            action_interval_env = (
                f"export AICAR_ACTION_INTERVAL_SECONDS='{self.action_interval_s:.17g}' && "
                if self._explicit_mode else ""
            )
            cmd_str = (
                action_interval_env + f"cd {shlex.quote(wsl_dir)} && "
                + shlex.join([f"./{sim_name}", *flags])
            )
            cmd = ["wsl", "bash", "-c", cmd_str]
        else:
            cmd = [str(self.simulator_path.resolve())]
            flags.extend(["-ip", "127.0.0.1", "-port", str(self.port)])
            cmd.extend(flags)

        creationflags = 0
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        if is_windows and not self.headless:
            creationflags = subprocess.CREATE_NEW_CONSOLE
            stdout_dest = None
            stderr_dest = None

        logger.info(f"[Racer {self.racer_id}] Spawning simulator (headless={self.headless}): {' '.join(cmd)}")
        child_env = os.environ.copy()
        if self._explicit_mode:
            child_env["AICAR_ACTION_INTERVAL_SECONDS"] = f"{self.action_interval_s:.17g}"
        else:
            child_env.pop("AICAR_ACTION_INTERVAL_SECONDS", None)
        self._sim_process = subprocess.Popen(
            cmd,
            cwd=str(self.simulator_path.parent.resolve()),
            stdout=stdout_dest,
            stderr=stderr_dest,
            creationflags=creationflags,
            env=child_env,
        )

    def step(self, throttle: float, steering: float) -> TelemetrySnapshot:
        """Set controls and wait for a Bridge response.

        Legacy mode returns the next telemetry packet and uses the historical
        asynchronous command reply. Explicit mode returns only the response
        tagged with this request's step ID and advances the advertised action
        interval before returning.

        Args:
            throttle: Acceleration/braking command in [-1.0, 1.0].
            steering: Steering angle command in [-1.0, 1.0].

        Returns:
            The latest telemetry snapshot received after this call began.
        """
        if not self._is_alive:
            raise RuntimeError(f"[Racer {self.racer_id}] Cannot step a killed/closed racer.")
        self._raise_incompatible_protocol()

        if self._explicit_mode:
            t0 = time.monotonic()
            snap = self._explicit_request(
                max(-1.0, min(1.0, float(throttle))),
                max(-1.0, min(1.0, float(steering))),
                reset=False,
            )
            t1 = time.monotonic()
            self.step_latency_ms = round((t1 - t0) * 1000.0, 2)
            step_dt = t1 - self._last_step_time
            self._last_step_time = t1
            if step_dt > 0:
                self.steps_per_second = round(1.0 / step_dt, 1)
                self.real_time_factor = round(self.steps_per_second * float(self.action_interval_s), 2)
            return snap

        t0 = time.time()
        start_step = self.step_counter
        self._cmd_throttle = float(max(-1.0, min(1.0, throttle)))
        self._cmd_steering = float(max(-1.0, min(1.0, steering)))

        # Wait until a newer Bridge packet arrives (step_counter > start_step).
        end_time = time.time() + self.step_timeout
        while self.step_counter <= start_step and self._is_alive:
            self._raise_incompatible_protocol()
            self._step_event.clear()
            remaining = end_time - time.time()
            if remaining <= 0 or not self._step_event.wait(timeout=max(0.01, remaining)):
                break

        if self.step_counter <= start_step:
            self._raise_incompatible_protocol()
            logger.warning(
                f"[Racer {self.racer_id}] Watchdog timeout ({self.step_timeout}s) waiting for simulator frame "
                f"(last step_counter={self.step_counter}, connected={self._connected})."
            )

        # Update Profiler Metrics
        t1 = time.time()
        dt = t1 - t0
        step_dt = t1 - self._last_step_time
        self._last_step_time = t1

        self.step_latency_ms = round(dt * 1000.0, 2)
        if step_dt > 0:
            self.steps_per_second = round(1.0 / step_dt, 1)
            # Compare Bridge responses against the nominal 40 Hz control rate.
            self.real_time_factor = round(self.steps_per_second / 40.0, 2)

        return self.telemetry

    def reset(self) -> TelemetrySnapshot:
        """Reset the vehicle to the starting grid and reset timers."""
        if not self._is_alive:
            raise RuntimeError(f"[Racer {self.racer_id}] Cannot reset a killed/closed racer.")
        self._raise_incompatible_protocol()

        if self._explicit_mode:
            snap = self._explicit_request(0.0, 0.0, reset=True)
            self._last_step_time = time.monotonic()
            return snap

        # The Bridge reply that carries V1 Reset also carries actuator commands.
        # Clear them first so the previous episode cannot keep driving during reset.
        self._cmd_throttle = 0.0
        self._cmd_steering = 0.0
        self._reset_sent_event.clear()
        self._reset_event.clear()
        self._reset_command_step = None
        self._reset_requested = True

        reset_sent = self._reset_sent_event.wait(timeout=self.step_timeout)
        self._raise_incompatible_protocol()
        if not reset_sent:
            raise TimeoutError(
                f"[Racer {self.racer_id}] No Bridge frame arrived to send reset "
                f"on port {self.port} within {self.step_timeout:.1f}s"
            )
        reset_complete = self._reset_event.wait(timeout=self.step_timeout)
        self._raise_incompatible_protocol()
        if not reset_complete:
            raise TimeoutError(
                f"[Racer {self.racer_id}] Unity did not return a post-reset Bridge "
                f"frame on port {self.port} within {self.step_timeout:.1f}s"
            )
        return self.telemetry

    def set_simulation_paused(self, paused: bool) -> None:
        """Hold/resume Unity physics during PPO optimization updates."""
        paused = bool(paused)
        if not paused:
            self.resume_simulation()
            return
        with self._simulation_pause_lock:
            if paused == self._simulation_paused:
                return
            if not self._connected or not self.client_sid:
                raise RuntimeError(
                    f"[Racer {self.racer_id}] Cannot change simulation pause state "
                    "before the Unity Bridge is connected."
                )

            ack = self._simulation_pause_ack
            ack.clear()
            self._emit_socket_event("AICAR_SIMULATION_PAUSE", sid=self.client_sid)
            if not ack.wait(timeout=self.step_timeout):
                raise TimeoutError(
                    f"[Racer {self.racer_id}] Unity did not acknowledge "
                    "simulation pause within "
                    f"{self.step_timeout:.1f}s. Rebuild the simulator with "
                    "AiCarSimulationGate enabled."
                )
            self._simulation_paused = paused

    def resume_simulation(self) -> None:
        """Force Unity physics to resume, even after reconnecting to a paused player."""
        with self._simulation_pause_lock:
            if not self._connected or not self.client_sid:
                raise RuntimeError(
                    f"[Racer {self.racer_id}] Cannot resume simulation "
                    "before the Unity Bridge is connected."
                )
            ack = self._simulation_resume_ack
            ack.clear()
            self._emit_socket_event("AICAR_SIMULATION_RESUME", sid=self.client_sid)
            if not ack.wait(timeout=self.step_timeout):
                raise TimeoutError(
                    f"[Racer {self.racer_id}] Unity did not acknowledge "
                    f"simulation resume within {self.step_timeout:.1f}s. "
                    "Rebuild the simulator with AiCarSimulationGate enabled."
                )
            self._simulation_paused = False
            self._mark_simulation_resumed()

    def _mark_simulation_paused(self) -> None:
        """Start excluding wall time once Unity confirms physics is held."""
        if self._simulation_pause_started_at is None:
            self._simulation_pause_started_at = time.monotonic()

    def _mark_simulation_resumed(self) -> None:
        """Close the held interval when Unity confirms physics has resumed."""
        started_at = self._simulation_pause_started_at
        if started_at is not None:
            self._simulation_paused_total_s += max(0.0, time.monotonic() - started_at)
            self._simulation_pause_started_at = None

    def simulation_time(self) -> float:
        """Return elapsed training time with PPO update pauses removed."""
        if getattr(self, "_explicit_mode", False):
            return float(self._explicit_sim_time or 0.0)
        now = time.monotonic()
        paused_now = (
            max(0.0, now - self._simulation_pause_started_at)
            if self._simulation_pause_started_at is not None else 0.0
        )
        return now - self._simulation_paused_total_s - paused_now

    def kill(self) -> None:
        """Terminate the Unity child simulator OS process and shut down gevent server."""
        self._is_alive = False

        # 1. Kill Unity child OS process
        if self._sim_process is not None:
            try:
                logger.info(f"[Racer {self.racer_id}] Terminating simulator PID {self._sim_process.pid}...")
                self._sim_process.terminate()
                self._sim_process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, Exception):
                logger.warning(f"[Racer {self.racer_id}] Force killing simulator PID {self._sim_process.pid}...")
                try:
                    self._sim_process.kill()
                except Exception:
                    pass
            finally:
                self._sim_process = None

        # 2. Stop gevent WSGI server (may raise if called cross-thread; safe to ignore)
        server = self._wsgi_server
        self._wsgi_server = None
        if server is not None:
            try:
                # Wake the serve thread's _stop_event without requiring hub affinity
                server._stop_event.set()
            except Exception:
                pass
            try:
                server.stop()
            except Exception as e:
                logger.debug(f"[Racer {self.racer_id}] Server close error: {e}")
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
        self._server_thread = None

        # Release any threads waiting on events
        self._step_event.set()
        self._reset_sent_event.set()
        self._reset_event.set()
        logger.info(f"[Racer {self.racer_id}] Process terminated and port {self.port} released.")

    def save_trajectory(self, file_path: Union[str, Path]) -> str:
        """Export this vehicle's trajectory log to CSV."""
        return self.logger.save_to_csv(file_path)

    @property
    def is_alive(self) -> bool:
        """Return True if this racer is active and not terminated."""
        return self._is_alive

    @property
    def is_connected(self) -> bool:
        """Return True if simulator is actively connected via Socket.IO."""
        self._raise_incompatible_protocol()
        return self._connected

    def _raise_incompatible_protocol(self) -> None:
        message = getattr(self, "_incompatible_protocol_error", None)
        if message:
            raise RuntimeError(message)
