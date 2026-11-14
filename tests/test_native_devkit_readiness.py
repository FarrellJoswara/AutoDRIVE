from src.layer4.hub import native_official_backend
from src.layer4.hub.native_official_backend import _devkit_node_is_visible


def test_devkit_readiness_requires_bridge_node_in_assigned_ros_domain():
    assert _devkit_node_is_visible(0, "/autodrive_bridge\n/another_node\n")
    assert not _devkit_node_is_visible(0, "/another_node\n")
    assert not _devkit_node_is_visible(1, "/autodrive_bridge\n")


def test_native_bridge_domain_matches_environment_index(monkeypatch):
    backend = object.__new__(native_official_backend.NativeOfficialBackend)
    backend._lock = native_official_backend.threading.RLock()
    backend._groups = {}
    backend._reserved_ros_domains = set()
    monkeypatch.setenv("AICAR_ROS_DOMAIN_ID_BASE", "0")

    assert backend._allocate_ros_domain(5) == 5
    # A run's base is authoritative; the host process base may belong to a
    # different context and must not reject a valid explicit assignment.
    assert backend._allocate_ros_domain(105) == 105
    backend._groups["other"] = {"ros_domains": [6]}
    assert backend._allocate_ros_domain(4) == 4

    backend._groups["conflict"] = {"ros_domains": [3]}
    try:
        backend._allocate_ros_domain(3)
    except RuntimeError as exc:
        assert "required by this environment" in str(exc)
    else:
        raise AssertionError("a conflicting expected domain must fail instead of silently shifting")


def test_api_worker_uses_run_domain_base_even_when_host_base_differs(tmp_path, monkeypatch):
    backend = object.__new__(native_official_backend.NativeOfficialBackend)
    backend._startup_lock = native_official_backend.threading.RLock()
    backend._lock = native_official_backend.threading.RLock()
    backend._groups = {}
    backend._reserved_ros_domains = set()
    backend._reserved_devkit_ports = set()
    backend._allocate_devkit_port = lambda: 4577
    backend._persist_group = lambda _group: None
    captured = []
    backend._start_bridge_pair = lambda _records, _run_id, _index, _env, _run_dir, **kwargs: captured.append(kwargs)
    monkeypatch.setenv("AICAR_ROS_DOMAIN_ID_BASE", "0")

    backend.create_official_run_container(
        run_id="domain-base-test", role="api-5", image="unused", network_name="unused",
        env=["ROS_DOMAIN_ID=10", "AICAR_ROS_DOMAIN_ID_BASE=10"],
        metadata_labels={"aicar.output_dir": str(tmp_path), "aicar.run_id": "domain-base-test"},
    )

    assert captured[0]["ros_domain"] == 15


def test_partial_startup_reaps_processes_with_term_then_kill(monkeypatch):
    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.alive = True

        def poll(self):
            return None if self.alive else 0

    process = Process(321)
    signals = []

    def signal_group(_process, sig):
        signals.append(sig)
        if sig == native_official_backend.signal.SIGKILL:
            process.alive = False

    monkeypatch.setattr(native_official_backend, "_signal_process_group", signal_group)
    assert native_official_backend._reap_started_processes(
        [{"process": process}], grace_s=0.01
    )
    assert signals == [native_official_backend.signal.SIGTERM,
                       native_official_backend.signal.SIGKILL]
