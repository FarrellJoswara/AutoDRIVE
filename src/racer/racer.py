"""Racer: Individual AutoDRIVE Vehicle Controller and Simulator Bridge.

Manages Socket.IO server communications via ASGI/uvicorn, turn-based lockstep
synchronization, real-time execution profiling, and child simulator process lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import socketio
import uvicorn

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
        self._server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None

        # Socket.IO ASGI Server Setup
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self._register_handlers()
        self._start_server()

        # Auto-launch child process if requested
        if self.auto_launch and self.simulator_path:
            self.launch_simulator()

    def _register_handlers(self) -> None:
        """Register Socket.IO event handlers for the AutoDRIVE simulator."""

        # Auto-connect client if event arrives before explicit connect packet (Unity websocket-sharp client)
        orig_handle_event = self.sio._handle_event

        async def _auto_connect_handle_event(eio_sid, namespace, id, data):
            ns = namespace or "/"
            sid = self.sio.manager.sid_from_eio_sid(eio_sid, ns)
            if not self.sio.manager.is_connected(sid, ns):
                await self.sio._handle_connect(eio_sid, ns, None)
                sid = self.sio.manager.sid_from_eio_sid(eio_sid, ns)
                self.client_sid = sid
                self._connected = True
                logger.info(f"[Racer {self.racer_id}] Connected to simulator client sid={sid} on port {self.port}")
            return await orig_handle_event(eio_sid, namespace, id, data)

        self.sio._handle_event = _auto_connect_handle_event

        @self.sio.event
        async def connect(sid, environ):
            self.client_sid = sid
            self._connected = True
            logger.info(f"[Racer {self.racer_id}] Connected to simulator client sid={sid} on port {self.port}")

        @self.sio.event
        async def disconnect(sid):
            self._connected = False
            self.client_sid = None
            logger.warning(f"[Racer {self.racer_id}] Disconnected from simulator on port {self.port}")

        @self.sio.on("Bridge")
        async def on_bridge(sid, data: Dict[str, Any]):
            """Turn-based lockstep callback invoked on each Unity physics tick (40 Hz)."""
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

            # Return driving commands to simulator via explicit Bridge emission
            await self.sio.emit(
                "Bridge",
                data={
                    "V1 Throttle": str(cmd_th),
                    "V1 Steering": str(cmd_st),
                    "V1 Reset": "1" if reset_flag else "0",
                },
                to=sid,
            )

            return {
                "V1 Throttle": cmd_th,
                "V1 Steering": cmd_st,
                "V1 Reset": bool(reset_flag),
            }

    def _start_server(self) -> None:
        """Start the background uvicorn ASGI server hosting the Socket.IO listener."""
        app = socketio.ASGIApp(self.sio)
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            name=f"Racer-{self.racer_id}-Uvicorn-{self.port}",
            daemon=True,
        )
        self._server_thread.start()
        # Brief pause to ensure port is listening
        time.sleep(0.3)
        logger.info(f"[Racer {self.racer_id}] Listening on 0.0.0.0:{self.port} (ASGI/uvicorn)")

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
                f"[Racer {self.racer_id}] Watchdog timeout ({self.step_timeout}s) waiting for simulator frame."
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
        """Terminate the Unity child simulator OS process and shut down ASGI server."""
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

        # 2. Stop uvicorn server
        if self._server is not None:
            try:
                self._server.should_exit = True
            except Exception as e:
                logger.debug(f"[Racer {self.racer_id}] Server close error: {e}")
            finally:
                self._server = None

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
