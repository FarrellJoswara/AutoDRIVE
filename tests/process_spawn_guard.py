"""Fail-closed guard against accidentally starting the official simulator in pytest."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any


_PROTECTED_TEXT_MARKERS = (
    "autodrive simulator.exe",
    "autodrive simulator.x86_64",
    "autodrive_roboracer bringup_headless.launch.py",
    "official_bridge_relay",
    "aicar-native-policy",
    "start_official_windows_simulators",
)
_ROS_COMMAND = re.compile(r"(?i)(?:^|[\s/'\"\\])ros2(?:\.exe)?\s+(?:launch|run)(?:\s|$)")


def _strings(value: Any):
    if isinstance(value, (str, bytes, os.PathLike)):
        try:
            yield os.fsdecode(os.fspath(value))
        except (TypeError, ValueError):
            return
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _strings(item)


def _resolved_executable_markers(command_parts: list[str]) -> list[str]:
    """Catch renamed/symlinked simulator binaries passed as argv tokens."""
    found: list[str] = []
    for part in command_parts:
        candidates = [part]
        try:
            candidates.extend(shlex.split(part, posix=os.name != "nt"))
        except ValueError:
            pass
        for candidate in candidates:
            # Shell command fragments aren't paths.  Check only plausible path
            # tokens and resolve symlinks when they name an existing file.
            if not any(sep in candidate for sep in ("/", "\\")):
                continue
            try:
                resolved = str(Path(candidate).resolve(strict=True)).lower()
            except (OSError, RuntimeError):
                continue
            if Path(resolved).name in {
                "autodrive simulator.exe",
                "autodrive simulator.x86_64",
            } or any(
                component.lower() == "autodrive_simulator"
                for component in Path(resolved).parts
            ):
                found.append(resolved)
    return found


def protected_runtime_reason(event_args: tuple[Any, ...]) -> str | None:
    """Return a reason when a process launch targets protected runtime code."""
    parts = list(_strings(event_args))
    command_text = "\n".join(parts).lower()
    marker = next((item for item in _PROTECTED_TEXT_MARKERS if item in command_text), None)
    if marker:
        return f"protected runtime marker: {marker}"
    if _ROS_COMMAND.search(command_text):
        return "ROS 2 launch/run commands are disabled in pytest"
    resolved = _resolved_executable_markers(parts)
    if resolved:
        return f"resolved simulator executable: {resolved[0]}"
    return None


def reject_protected_runtime_spawn(event: str, args: tuple[Any, ...]) -> None:
    """Reject protected launches through subprocess, shell, or os.spawn APIs."""
    if event not in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.spawn",
        "os.exec",
    }:
        return
    reason = protected_runtime_reason(args)
    if reason:
        raise PermissionError(f"pytest runtime-launch guard blocked process creation ({reason})")
