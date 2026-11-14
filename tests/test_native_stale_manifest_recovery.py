import json
import threading

import pytest

from src.layer4.hub import native_official_backend


def _write_manifest(tmp_path, *, records=None):
    run_dir = tmp_path / "logs" / "rl" / "official_run_oldrun"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("devkit.log", "relay.log"):
        (run_dir / name).touch()
    manifest = run_dir / "native_group_api-5.json"
    manifest.write_text(json.dumps({
        "Id": "native-oldrun-api-5",
        "Names": ["/aicar-run-oldrun-api-5"],
        "Labels": {"aicar.run_id": "oldrun", "aicar.role": "api-5"},
        "backend_instance_id": "old-backend",
        "hub_pid": 1234,
        "created_at_unix_s": 1,
        "ros_domains": [5],
        "devkit_ports": [4572],
        "records": records or [
            {"role": "devkit-5", "pid": 4321, "process_group_id": 4321,
             "process_starttime_ticks": 99, "log": str(run_dir / "devkit.log")},
            {"role": "relay-5", "pid": 4322, "process_group_id": 4322,
             "process_starttime_ticks": 100, "log": str(run_dir / "relay.log")},
        ],
    }))
    return run_dir, manifest


def _backend(tmp_path, monkeypatch, recovered_process):
    monkeypatch.setattr(native_official_backend, "ROOT", tmp_path)
    monkeypatch.setattr(native_official_backend, "_RecoveredProcess", recovered_process)
    backend = native_official_backend.NativeOfficialBackend.__new__(
        native_official_backend.NativeOfficialBackend
    )
    backend._groups = {}
    backend._networks = {}
    backend._lock = threading.RLock()
    backend._reserved_ros_domains = set()
    backend._reserved_devkit_ports = set()
    backend._devkit_port_start = 4567
    backend.backend_instance_id = "test-backend"
    backend.hub_pid = 9999
    return backend


def test_recovery_archives_fully_exited_manifest_and_avoids_future_pid_reuse(tmp_path, monkeypatch):
    _, manifest = _write_manifest(tmp_path)

    class ExitedProcess:
        def __init__(self, *_args):
            pass

        def poll(self):
            return 0

    backend = _backend(tmp_path, monkeypatch, ExitedProcess)
    backend._recover_groups()

    assert backend._groups == {}
    assert not manifest.exists()
    assert (manifest.parent / "native_terminal_manifest_api-5.json").exists()

    class ReusedPidProcess:
        def __init__(self, *_args):
            pass

        def poll(self):
            return None

    backend = _backend(tmp_path, monkeypatch, ReusedPidProcess)
    backend._recover_groups()
    assert backend._groups == {}
    assert backend._allocate_ros_domain(5) == 5


@pytest.mark.parametrize("statuses", [[None, None], [0, None]])
def test_recovery_keeps_live_or_uncertain_group_reserved(tmp_path, monkeypatch, statuses):
    _, manifest = _write_manifest(tmp_path)

    class Process:
        next_status = iter(statuses)

        def __init__(self, *_args):
            self.status = next(self.next_status)

        def poll(self):
            return self.status

    backend = _backend(tmp_path, monkeypatch, Process)
    backend._recover_groups()

    assert "native-oldrun-api-5" in backend._groups
    assert manifest.exists()
    with pytest.raises(RuntimeError, match="ROS domain 5 required.*already reserved"):
        backend._allocate_ros_domain(5)


def test_recovery_closes_logs_when_loading_a_later_record_fails(tmp_path, monkeypatch):
    _, _ = _write_manifest(tmp_path)
    opened = []
    original_open = native_official_backend.Path.open

    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name in {"devkit.log", "relay.log"}:
            opened.append(handle)
        return handle

    monkeypatch.setattr(native_official_backend.Path, "open", track_open)

    class Process:
        count = 0

        def __init__(self, *_args):
            type(self).count += 1
            if type(self).count == 2:
                raise ValueError("malformed later process record")

        def poll(self):
            return None

    backend = _backend(tmp_path, monkeypatch, Process)
    backend._recover_groups()

    assert len(opened) == 2
    assert all(handle.closed for handle in opened)
    assert backend._groups == {}
