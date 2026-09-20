"""Racer: Individual AutoDRIVE Vehicle Controller and Simulator Bridge.

Manages Socket.IO server communications via gevent/WSGI (matching the official
AutoDRIVE Devkit bridge), turn-based lockstep synchronization, real-time
execution profiling, and child simulator process lifecycle.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import socketio
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from .telemetry import TelemetrySnapshot, TrajectoryLogger

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.racer_id = racer_id
        self.port = port
        self.simulator_path = Path(simulator_path) if simulator_path else None
        self.auto_launch = auto_launch
        self.headless = headless
        self.step_timeout = step_timeout

        # Telemetry & Logger
        self.telemetry: TelemetrySnapshot = TelemetrySnapshot()
        self.logger: TrajectoryLogger = TrajectoryLogger()
        self.enable_logging = enable_logging

        # Control Command Buffers
        self._cmd_throttle: float = 0.0
        self._cmd_steering: float = 0.0
        self._reset_requested: bool = False

        # Profiler Metrics
        self.step_counter: int = 0
        self.step_latency_ms: float = 0.0
        self.steps_per_second: float = 0.0
        self.real_time_factor: float = 0.0
        self._last_step_time: float = time.time()

        # Synchronization Primitives
        self._step_event = threading.Event()
        self._reset_event = threading.Event()
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
            """Callback on each Unity physics tick; reply via emit (official protocol)."""
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

        if is_windows and is_linux_elf:
            sim_dir = str(self.simulator_path.parent.resolve()).replace("\\", "/")
            drive_letter = sim_dir[0].lower()
            wsl_dir = f"/mnt/{drive_letter}{sim_dir[2:]}"
            sim_name = self.simulator_path.name
            target_ip = _get_wsl_host_ip()
            flags.extend(["-ip", target_ip, "-port", str(self.port)])
            cmd_str = f"cd '{wsl_dir}' && ./'{sim_name}' " + " ".join(flags)
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
        self._sim_process = subprocess.Popen(
            cmd,
            cwd=str(self.simulator_path.parent.resolve()),
            stdout=stdout_dest,
            stderr=stderr_dest,
            creationflags=creationflags,
        )

    def step(self, throttle: float, steering: float) -> TelemetrySnapshot:
        """Execute one control step in turn-based lockstep with the simulator.

        Args:
            throttle: Acceleration/braking command in [-1.0, 1.0].
            steering: Steering angle command in [-1.0, 1.0].

        Returns:
            TelemetrySnapshot after simulator advances physics.
        """
        if not self._is_alive:
            raise RuntimeError(f"[Racer {self.racer_id}] Cannot step a killed/closed racer.")

        t0 = time.time()
        start_step = self.step_counter
        self._cmd_throttle = float(max(-1.0, min(1.0, throttle)))
        self._cmd_steering = float(max(-1.0, min(1.0, steering)))

        # Wait until a fresh frame arrives (step_counter > start_step)
        end_time = time.time() + self.step_timeout
        while self.step_counter <= start_step and self._is_alive:
            self._step_event.clear()
            remaining = end_time - time.time()
            if remaining <= 0 or not self._step_event.wait(timeout=max(0.01, remaining)):
                break

        if self.step_counter <= start_step:
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
            # 40 Hz is base Unity physics rate
            self.real_time_factor = round(self.steps_per_second / 40.0, 2)

        return self.telemetry

    def reset(self) -> TelemetrySnapshot:
        """Reset the vehicle to the starting grid and reset timers."""
        if not self._is_alive:
            raise RuntimeError(f"[Racer {self.racer_id}] Cannot reset a killed/closed racer.")

        self._reset_event.clear()
        self._reset_requested = True

        # Wait for reset frame to be acknowledged by Unity
        signaled = self._reset_event.wait(timeout=self.step_timeout)
        if not signaled:
            logger.warning(f"[Racer {self.racer_id}] Timeout waiting for simulator reset acknowledgment.")

        time.sleep(0.05)
        return self.telemetry

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
        return self._connected
