from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.layer1.racer import Racer


def _racer_for_launch(path: Path) -> Racer:
    racer = Racer.__new__(Racer)
    racer.simulator_path = path
    racer.headless = True
    racer.simulator_log_path = None
    racer.port = 4567
    racer.racer_id = 0
    racer.action_interval_s = None
    racer._explicit_mode = False
    racer._sim_process = None
    return racer


def test_linux_headless_launch_keeps_rendering_and_uses_xvfb(tmp_path: Path) -> None:
    simulator = tmp_path / "AutoDRIVE Simulator.x86_64"
    simulator.touch()
    racer = _racer_for_launch(simulator)

    with patch("src.layer1.racer.platform.system", return_value="Linux"), patch(
        "src.layer1.racer.subprocess.Popen"
    ) as popen:
        racer.launch_simulator()

    argv = popen.call_args.args[0]
    assert argv[:4] == ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24"]
    assert "-batchmode" in argv
    assert "-nographics" not in argv
    assert argv[-4:] == ["-ip", "127.0.0.1", "-port", "4567"]


def test_windows_headless_launch_does_not_disable_graphics(tmp_path: Path) -> None:
    simulator = tmp_path / "AutoDRIVE Simulator.exe"
    simulator.touch()
    racer = _racer_for_launch(simulator)

    with patch("src.layer1.racer.platform.system", return_value="Windows"), patch(
        "src.layer1.racer.subprocess.Popen"
    ) as popen:
        racer.launch_simulator()

    argv = popen.call_args.args[0]
    assert "-batchmode" in argv
    assert "-nographics" not in argv


def test_custom_docker_entrypoint_keeps_graphics_enabled() -> None:
    entrypoint = Path("docker/entrypoint-sim.sh").read_text(encoding="utf-8")
    assert "Xvfb :99" in entrypoint
    assert 'exec "$SIM_PATH" -batchmode -ip "$BRAIN_IP" -port "$SIM_PORT"' in entrypoint
    assert "-nographics" not in entrypoint
