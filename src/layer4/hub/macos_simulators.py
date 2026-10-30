"""Launch the official macOS AutoDRIVE practice simulator against host relays."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from src.layer4.settings import ROOT

DEFAULT_CANDIDATES = (
    Path.home() / "Downloads" / "autodrive_simulator" / "AutoDRIVE Simulator.app",
    Path("/Applications") / "AutoDRIVE Simulator.app",
    ROOT / "simulator" / "macos" / "AutoDRIVE Simulator.app",
)
PATH_CACHE = ROOT / "logs" / "layer4" / "official_simulator_path.txt"
_LAUNCHERS: dict[str, list[subprocess.Popen]] = {}


def resolve_macos_simulator_binary() -> Path:
    """Return the native AutoDRIVE Simulator executable inside the .app bundle."""
    configured = os.environ.get("AICAR_OFFICIAL_SIMULATOR_APP", "").strip() or os.environ.get(
        "AICAR_OFFICIAL_SIMULATOR_EXE", ""
    ).strip()
    if not configured:
        try:
            configured = PATH_CACHE.read_text(encoding="utf-8").strip()
        except OSError:
            configured = ""
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(DEFAULT_CANDIDATES)
    for candidate in candidates:
        binary = _binary_from_app_or_exe(candidate)
        if binary is not None:
            PATH_CACHE.parent.mkdir(parents=True, exist_ok=True)
            app_path = binary.parents[2] if binary.parent.name == "MacOS" else binary
            PATH_CACHE.write_text(str(app_path), encoding="utf-8")
            return binary
    raise RuntimeError(
        "macOS AutoDRIVE Simulator.app not found. Download the IROS 2026 practice build "
        "and set AICAR_OFFICIAL_SIMULATOR_APP to the .app path "
        "(expected ~/Downloads/autodrive_simulator/AutoDRIVE Simulator.app)."
    )


def _binary_from_app_or_exe(path: Path) -> Path | None:
    if path.is_file() and os.access(path, os.X_OK):
        return path
    if path.suffix == ".app" or path.name.endswith(".app"):
        binary = path / "Contents" / "MacOS" / "AutoDRIVE Simulator"
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
    nested = path / "AutoDRIVE Simulator.app" / "Contents" / "MacOS" / "AutoDRIVE Simulator"
    if nested.is_file() and os.access(nested, os.X_OK):
        return nested
    return None


def _simulator_mode_args(config: dict[str, Any] | None) -> list[str]:
    mode = (config or {}).get("simulator_display_mode", "offscreen")
    if mode == "gui":
        return []
    return ["-batchmode", "-nographics"]


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def _wait_for_relay_ports(connections: list[dict[str, Any]], *, timeout_s: float) -> None:
    """Block until each published relay accepts TCP (current API image has no /healthz)."""
    deadline = time.monotonic() + max(15.0, timeout_s)
    pending = {(str(item.get("host") or "127.0.0.1"), int(item["port"])) for item in connections}
    while time.monotonic() < deadline and pending:
        for endpoint in list(pending):
            if _port_open(endpoint[1], endpoint[0]):
                pending.discard(endpoint)
        if pending:
            time.sleep(0.25)
    if pending:
        raise TimeoutError(
            "Docker API relay did not accept TCP before macOS simulator launch: "
            f"{sorted(f'{host}:{port}' for host, port in pending)}"
        )


def _relay_log_ready(text: str, *, minimum_complete_frames: int = 1) -> dict[str, Any]:
    connected = text.count('"event": "simulator_connected"')
    frames = text.count("first_complete_frame_forwarded")
    if connected < 1 or frames < minimum_complete_frames:
        return {
            "simulator_connected_events": connected,
            "complete_frame_events": frames,
            "ready": False,
        }
    return {
        "simulator_connected_events": connected,
        "complete_frame_events": frames,
        "ready": True,
    }


def _read_api_logs(run_id: str) -> str:
    from src.layer4.hub import docker_control

    chunks: list[str] = []
    for container in docker_control.list_official_run_containers(run_id):
        role = str((container.get("Labels") or {}).get("aicar.role", ""))
        if role != "api" and not role.startswith("api"):
            continue
        try:
            chunks.append(docker_control.official_run_container_logs(str(container["Id"])))
        except Exception as exc:
            chunks.append(f"log_error:{exc}")
    if chunks:
        return "\n".join(chunks)
    try:
        return (ROOT / "logs" / "rl" / f"official_run_{run_id}" / "hub_train.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ""


def start_macos_simulators(
    *,
    run_id: str,
    connections: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    startup_timeout_s: float = 120.0,
) -> None:
    """Start one native simulator process per published relay port."""
    if not connections:
        raise ValueError("no simulator connection ports were allocated")
    binary = resolve_macos_simulator_binary()
    run_dir = ROOT / "logs" / "rl" / f"official_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "macos_simulators.log"
    pid_path = run_dir / "macos_simulator_pids.txt"
    mode_args = _simulator_mode_args(config)
    _wait_for_relay_ports(connections, timeout_s=min(60.0, startup_timeout_s))
    processes: list[subprocess.Popen] = []
    last: dict[str, Any] = {}
    with log_path.open("ab", buffering=0) as log:
        log.write(f"binary={binary}\n".encode())
        for item in connections:
            port = int(item["port"])
            host = str(item.get("host") or "127.0.0.1")
            command = [str(binary), *mode_args, "-ip", host, "-port", str(port)]
            log.write((" ".join(command) + "\n").encode())
            process = subprocess.Popen(
                command,
                cwd=str(binary.parent),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
        pid_path.write_text("\n".join(str(process.pid) for process in processes) + "\n", encoding="ascii")
        _LAUNCHERS[run_id] = processes

        deadline = time.monotonic() + max(startup_timeout_s, 45.0 * len(connections))
        while time.monotonic() < deadline:
            for process in processes:
                code = process.poll()
                if code is not None:
                    excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                    stop_macos_simulators(run_id)
                    raise RuntimeError(
                        f"macOS simulator exited during startup ({code}): {excerpt}"
                    )
            status = _relay_log_ready(_read_api_logs(run_id))
            last = status
            if status.get("ready"):
                (run_dir / "simulator_stream_health.json").write_text(
                    json.dumps({"environments": {"0": last}, "connections": connections}, indent=2),
                    encoding="utf-8",
                )
                log.write(b"All official simulator environments are connected.\n")
                return
            time.sleep(0.5)
    (run_dir / "simulator_stream_health.json").write_text(
        json.dumps({"environments": {"0": last}, "connections": connections}, indent=2),
        encoding="utf-8",
    )
    stop_macos_simulators(run_id)
    raise TimeoutError(
        "macOS simulator did not publish a complete sensor frame to the Docker relay: "
        f"{json.dumps(last, sort_keys=True)}"
    )


def stop_macos_simulators(run_id: str) -> None:
    """Terminate native simulator processes started for one official run."""
    processes = _LAUNCHERS.pop(run_id, [])
    pids: list[int] = [int(process.pid) for process in processes]
    pid_path = ROOT / "logs" / "rl" / f"official_run_{run_id}" / "macos_simulator_pids.txt"
    try:
        for line in pid_path.read_text(encoding="ascii").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except OSError:
        pass
    for pid in sorted(set(pids)):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError, OSError):
                    break
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
    for process in processes:
        try:
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            pass
    try:
        pid_path.unlink()
    except OSError:
        pass
