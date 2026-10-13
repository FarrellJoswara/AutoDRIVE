from __future__ import annotations

import subprocess
import sys
import os

import pytest

from process_spawn_guard import protected_runtime_reason


@pytest.mark.parametrize(
    "command",
    [
        (None, ["/tmp/AutoDRIVE Simulator.x86_64", "-port", "4567"], None, None),
        (None, ["bash", "-lc", "exec ros2 launch autodrive_roboracer bringup_headless.launch.py"], None, None),
        (None, ["python", "-m", "src.layer1.official_bridge_relay"], None, None),
        (None, ["powershell.exe", "-File", "start_official_windows_simulators.ps1"], None, None),
    ],
)
def test_process_classifier_blocks_known_runtime_launch_shapes(command):
    assert protected_runtime_reason(command)


def test_audit_hook_blocks_before_a_nonexistent_simulator_can_execute():
    # The pytest conftest hook runs before the OS tries to resolve this binary;
    # PermissionError proves the tripwire fired before a child could start.
    with pytest.raises(PermissionError, match="pytest runtime-launch guard"):
        subprocess.Popen(["/tmp/AutoDRIVE Simulator.exe"])


@pytest.mark.parametrize(
    "command",
    [
        ["wsl.exe", "bash", "-lc", "exec ros2 launch autodrive_roboracer bringup_headless.launch.py"],
        ["powershell.exe", "-File", r"C:\repo\scripts\start_official_windows_simulators.ps1"],
        [sys.executable, "-m", "src.layer1.official_bridge_relay"],
    ],
)
def test_audit_hook_blocks_real_launcher_command_shapes(command):
    # The audit callback executes before the OS tries to resolve the command.
    with pytest.raises(PermissionError, match="pytest runtime-launch guard"):
        subprocess.Popen(command)


def test_os_system_path_is_guarded_before_shell_execution():
    with pytest.raises(PermissionError, match="pytest runtime-launch guard"):
        os.system("exec ros2 run autodrive_roboracer autodrive_bridge")


def test_audit_hook_allows_harmless_test_processes():
    result = subprocess.run(
        [sys.executable, "-c", "print('harmless')"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "harmless"


def test_resolved_symlink_alias_to_simulator_is_blocked(tmp_path):
    official_binary = tmp_path / "AutoDRIVE Simulator.x86_64"
    official_binary.write_text("unit-test placeholder", encoding="utf-8")
    alias = tmp_path / "innocent-worker"
    alias.symlink_to(official_binary)
    command = (None, [str(alias)], None, None)
    assert protected_runtime_reason(command)
    with pytest.raises(PermissionError, match="pytest runtime-launch guard"):
        subprocess.Popen([str(alias)])
