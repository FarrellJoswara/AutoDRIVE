import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from src.layer4.hub.native_official_backend import NativeOfficialBackend


def test_practice_startup_barrier_runs_on_allocated_domain_before_launch(tmp_path):
    run_id = "a1b2c3d4e5f6"
    run_dir = tmp_path / f"official_run_{run_id}"
    run_dir.mkdir()
    backend = object.__new__(NativeOfficialBackend)
    backend._lock = threading.RLock()
    backend.ros_workspace = tmp_path / "autodrive_ws"
    backend._groups = {
        "native-run-api": {
            "Labels": {"aicar.run_id": run_id},
            "ros_domains": [86],
        }
    }
    backend._run_dir_for_id = lambda _run_id: run_dir
    invocation = {}

    def complete_barrier(command, **kwargs):
        invocation["command"] = command
        invocation["env"] = kwargs["env"]
        output = run_dir / "neutral_startup_env0.json"
        output.write_text(json.dumps({
            "verified": True,
            "ros_domain_id": 86,
            "zero_commands_sent": 60,
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="neutral barrier ready")

    with patch(
        "src.layer4.hub.native_official_backend.subprocess.run",
        side_effect=complete_barrier,
    ):
        backend._run_prelaunch_neutral_barrier(
            run_id=run_id, run_dir=run_dir, environment_count=1,
        )

    assert invocation["env"]["ROS_DOMAIN_ID"] == "86"
    assert invocation["env"]["AICAR_ROS_DOMAIN_ID_BASE"] == "86"
    assert invocation["command"][0:2] == ["bash", "-lc"]
    assert "native_prelaunch_neutral_barrier.py" in invocation["command"][6]
    assert (run_dir / "neutral_startup_env0.json").is_file()


def test_practice_startup_barrier_fails_closed_without_devkit_command_subscribers(tmp_path):
    import pytest

    run_id = "a1b2c3d4e5f6"
    backend = object.__new__(NativeOfficialBackend)
    backend._lock = threading.RLock()
    backend.ros_workspace = tmp_path / "autodrive_ws"
    backend._groups = {}

    with pytest.raises(RuntimeError, match="identify every native ROS domain"):
        backend._run_prelaunch_neutral_barrier(
            run_id=run_id, run_dir=tmp_path, environment_count=1,
        )
