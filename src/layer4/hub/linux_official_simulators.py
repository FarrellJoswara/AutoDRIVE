"""Run-scoped launcher for the extracted official Linux AutoDRIVE player."""

from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from src.layer4.settings import ROOT

DEFAULT_RUNTIME = ROOT / ".native-linux" / "runtime"
DEFAULT_WRAPPER = ROOT / "scripts" / "native-linux-simulator.sh"
PINNED_IMAGES = {
    "practice": ("2026-iros-practice", "sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76"),
    "competition": ("2026-iros-compete", "sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d"),
}
_PROCESSES: dict[str, list[subprocess.Popen]] = {}


def linux_simulator_paths(config: dict[str, Any] | None = None) -> tuple[Path, Path, Path]:
    """Resolve explicit runtime, simulator, and wrapper paths without probing substitutes."""
    config = config or {}
    profile = config.get("track_profile", "practice")
    if profile not in PINNED_IMAGES:
        raise RuntimeError(f"unsupported native Linux official track profile: {profile!r}")
    default_runtime = DEFAULT_RUNTIME if profile == "practice" else ROOT / ".native-linux" / "competition-runtime"
    runtime_value = config.get("native_runtime")
    if runtime_value is None:
        runtime_value = str(default_runtime) if profile == "competition" else os.environ.get("AICAR_NATIVE_RUNTIME", str(default_runtime))
    runtime = Path(str(runtime_value)).expanduser()
    default_executable = runtime / "rootfs" / "home" / "autodrive_simulator" / "AutoDRIVE Simulator.x86_64"
    executable = Path(
        str(config.get("simulator_executable") or os.environ.get(
            "AICAR_OFFICIAL_SIMULATOR_EXE", str(default_executable)
        ))
    ).expanduser()
    wrapper = Path(os.environ.get("AICAR_OFFICIAL_SIMULATOR_WRAPPER", str(DEFAULT_WRAPPER))).expanduser()
    return runtime.resolve(), executable.resolve(), wrapper.resolve()


def validate_linux_simulator_paths(config: dict[str, Any] | None = None) -> tuple[Path, Path, Path]:
    runtime, executable, wrapper = linux_simulator_paths(config)
    identity = runtime / "IMAGE-IDENTITY.json"
    try:
        payload = json.loads(identity.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"official Linux simulator runtime identity is unavailable: {identity}") from exc
    profile = (config or {}).get("track_profile", "practice")
    expected_tag, expected_digest = PINNED_IMAGES[profile]
    if (payload.get("repository") != "autodriveecosystem/autodrive_roboracer_sim"
            or payload.get("tag") != expected_tag
            or payload.get("digest") != expected_digest
            or payload.get("architecture") != "linux/amd64"
            or payload.get("manifest_verified") is not True
            or not payload.get("rootfs_tree_digest")):
        raise RuntimeError(f"native Linux simulator runtime does not match the pinned official linux/amd64 {profile} image")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"official Linux simulator executable is missing or not executable: {executable}")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise RuntimeError(f"official Linux simulator wrapper is missing or not executable: {wrapper}")
    if executable.name != "AutoDRIVE Simulator.x86_64":
        raise RuntimeError("configured Linux simulator executable is not the official AutoDRIVE player")
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    if payload.get("simulator_sha256") != executable_hash:
        raise RuntimeError("official Linux simulator executable hash does not match its runtime attestation")
    try:
        attested_path = executable.relative_to(runtime / "rootfs").as_posix()
    except ValueError as exc:
        raise RuntimeError("configured Linux simulator executable must be inside its attested official rootfs") from exc
    if payload.get("simulator_path") != attested_path:
        raise RuntimeError("configured Linux simulator executable path does not match its runtime attestation")
    complete_path = runtime / "rootfs" / ".OCI-COMPLETE.json"
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("native Linux OCI completion attestation is unavailable") from exc
    if complete.get("tag", expected_tag) != expected_tag or complete.get("digest") != expected_digest:
        raise RuntimeError("native Linux OCI completion attestation does not match the selected official image")
    return runtime, executable, wrapper


def _proc_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def _proc_group_id(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rfind(")") + 2:].split()
        return int(tail[2])
    except (OSError, ValueError, IndexError):
        return None


def _find_player_in_group(group_id: int, executable: Path) -> tuple[int, int] | None:
    expected = str(executable)
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            if pid == group_id or _proc_group_id(pid) != group_id:
                continue
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            proc_executable = str((proc / "exe").resolve())
            if expected not in command and proc_executable != expected:
                continue
            ticks = _proc_start_ticks(pid)
            if ticks is not None:
                return pid, ticks
        except (OSError, ValueError, PermissionError):
            continue
    return None


def _manifest_path(run_id: str) -> Path:
    return ROOT / "logs" / "rl" / f"official_run_{run_id}" / "linux_simulator_processes.json"


def _write_manifest(run_id: str, rows: list[dict[str, Any]]) -> None:
    path = _manifest_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"run_id": run_id, "processes": rows}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def start_linux_simulators(
    *, run_id: str, connections: list[dict[str, Any]], config: dict[str, Any] | None = None,
) -> None:
    if not connections:
        raise ValueError("no simulator connection ports were allocated")
    runtime, executable, wrapper = validate_linux_simulator_paths(config)
    attestation = json.loads((runtime / "IMAGE-IDENTITY.json").read_text(encoding="utf-8"))
    executable_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    run_dir = _manifest_path(run_id).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "linux_simulators.log").open("ab", buffering=0)
    processes: list[subprocess.Popen] = []
    rows: list[dict[str, Any]] = []
    try:
        for item in connections:
            index, port = int(item["environment"]), int(item["port"])
            host = str(item.get("host") or "127.0.0.1")
            display_number = 100 + ((port - 49152) % 30000)
            display = f":{display_number}"
            command = [str(wrapper)]
            env = os.environ.copy()
            env.update({
                "AICAR_NATIVE_RUNTIME": str(runtime),
                "AICAR_NATIVE_RELEASE": str((config or {}).get("track_profile", "practice")),
                "AICAR_SIMULATOR_PATH": str(executable),
                "AICAR_NATIVE_XVFB_LOG": str(run_dir / f"linux_xvfb_env{index}.log"),
                "BRAIN_HOST": host,
                "PORT": str(port),
                "DISPLAY": display,
            })
            log.write((f"env={index} command={command!r} host={host} port={port} display={display}\n").encode())
            process = subprocess.Popen(
                command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append(process)
            player = None
            player_deadline = time.monotonic() + 35.0
            while time.monotonic() < player_deadline and process.poll() is None:
                player = _find_player_in_group(process.pid, executable)
                if player:
                    break
                time.sleep(0.1)
            if player is None:
                raise RuntimeError(
                    f"official Linux simulator did not start for environment {index}; "
                    "see linux_simulators.log and its Xvfb log"
                )
            rows.append({
                "environment": index, "pid": process.pid,
                "process_group_id": process.pid,
                "process_starttime_ticks": _proc_start_ticks(process.pid),
                "simulator_pid": player[0], "simulator_starttime_ticks": player[1],
                "wrapper": str(wrapper), "executable": str(executable),
                "executable_sha256": executable_hash,
                "simulator_image": f"{attestation['repository']}:{attestation['tag']}",
                "simulator_image_digest": attestation["digest"],
                "port": port, "display": display,
            })
            _PROCESSES[run_id] = processes
            _write_manifest(run_id, rows)
        (run_dir / "linux_simulators.pid").write_text(
            "\n".join(str(p.pid) for p in processes) + "\n", encoding="ascii"
        )
    except Exception:
        stop_linux_simulators(run_id)
        raise
    finally:
        log.close()


def stop_linux_simulators(run_id: str, *, timeout_s: float = 8.0) -> None:
    """Stop only process groups whose leader identity still matches our manifest."""
    path = _manifest_path(run_id)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records = manifest.get("processes", [])
    except (OSError, ValueError, TypeError):
        records = []
    for row in records:
        pid = int(row.get("pid", 0))
        expected_ticks = row.get("process_starttime_ticks")
        actual_ticks = _proc_start_ticks(pid) if pid > 0 else None
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if (actual_ticks is None or expected_ticks is None or actual_ticks != expected_ticks
                or str(row.get("wrapper", "")) not in command):
            continue
        try:
            os.killpg(int(row.get("process_group_id", pid)), signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and _proc_start_ticks(pid) == expected_ticks:
            time.sleep(0.05)
        if _proc_start_ticks(pid) == expected_ticks:
            try:
                os.killpg(int(row.get("process_group_id", pid)), signal.SIGKILL)
            except ProcessLookupError:
                pass
    _PROCESSES.pop(run_id, None)
    path.unlink(missing_ok=True)
    (path.parent / "linux_simulators.pid").unlink(missing_ok=True)
