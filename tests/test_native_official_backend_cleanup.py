import signal
from types import SimpleNamespace

from src.layer4.hub import native_official_backend


def test_owned_launcher_stays_live_until_child_process_group_is_cleaned(monkeypatch):
    child_group_alive = True
    signaled = []

    def fake_killpg(pgid, sig):
        nonlocal child_group_alive
        assert pgid == 4321
        if sig == 0:
            if not child_group_alive:
                raise ProcessLookupError
            return
        signaled.append(sig)
        if sig == signal.SIGTERM:
            child_group_alive = False

    monkeypatch.setattr(native_official_backend.os, "killpg", fake_killpg)
    launcher = SimpleNamespace(
        pid=4321,
        args=["ros2", "launch", "autodrive_roboracer", "bringup_headless.launch.py"],
        poll=lambda: 0,
        wait=lambda timeout=None: 0,
    )
    process = native_official_backend._OwnedProcessGroupPopen(launcher)

    # The launch parent has exited, but its ROS node still occupies the owned
    # session. Cleanup code using the usual Popen poll guard must still signal it.
    assert process.poll() is None
    if process.poll() is None:
        native_official_backend._signal_process_group(process, signal.SIGTERM)
        process.wait(timeout=0.1)

    assert signaled == [signal.SIGTERM]
    assert process.poll() == 0
