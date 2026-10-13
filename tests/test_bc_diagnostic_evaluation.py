from src.layer4.hub.native_official_backend import NativeOfficialBackend


def test_native_evaluation_command_keeps_scoring_defaults_unmodified():
    command = NativeOfficialBackend._evaluation_command({})
    assert "--lap-progress-timeout-s" not in command
    assert "--frontier-stall-after-s" not in command


def test_native_evaluation_command_propagates_bc_diagnostic_watchdogs():
    command = NativeOfficialBackend._evaluation_command({
        "AICAR_EVALUATION_VALIDATION_MODE": "bc_diagnostic",
        "AICAR_LAP_PROGRESS_TIMEOUT_S": "240",
        "AICAR_LAP_PROGRESS_WALL_TIMEOUT_S": "240",
        "AICAR_FRONTIER_STALL_AFTER_S": "5",
        "AICAR_EVALUATION_FRONTIER_PATH": "competition/iros2026/tracks/competition/official_centerline.csv",
    })
    assert command[command.index("--validation-mode") + 1] == "bc_diagnostic"
    assert command[command.index("--lap-progress-timeout-s") + 1] == "240"
    assert command[command.index("--lap-progress-wall-timeout-s") + 1] == "240"
    assert command[command.index("--frontier-stall-after-s") + 1] == "5"
    assert command[command.index("--frontier-path") + 1].endswith("official_centerline.csv")


def test_native_evaluation_command_supports_route_monitor_without_competitive_watchdogs():
    command = NativeOfficialBackend._evaluation_command({
        "AICAR_EVALUATION_FRONTIER_PATH": "competition/iros2026/tracks/competition/official_centerline.csv",
    })
    assert "--validation-mode" not in command
    assert "--lap-progress-timeout-s" not in command
    assert "--lap-progress-wall-timeout-s" not in command
    assert "--frontier-stall-after-s" not in command
    assert command[command.index("--frontier-path") + 1].endswith("official_centerline.csv")


def test_native_evaluation_command_marks_nonblocking_route_monitor_only():
    command = NativeOfficialBackend._evaluation_command({
        "AICAR_ROUTE_MONITOR_ONLY": "1",
        "AICAR_EVALUATION_FRONTIER_PATH": "competition/iros2026/tracks/competition/official_centerline.csv",
    })
    assert "--route-monitor-only" in command
    assert "--validation-mode" not in command
    assert "--lap-progress-timeout-s" not in command
    assert "--lap-progress-wall-timeout-s" not in command
    assert "--frontier-stall-after-s" not in command
