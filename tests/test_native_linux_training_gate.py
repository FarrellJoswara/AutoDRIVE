from __future__ import annotations

import pytest

from scripts.test_official_run_manager import FakeDocker
from src.layer4.hub.official_run_manager import OfficialRunManager


class FakeNativeLinux(FakeDocker):
    supports_linux_external = True
    required_simulator_backend = "linux_external"
    simulator_host = "127.0.0.1"

    def __init__(self, host_logs):
        super().__init__(host_logs)
        self.validated_profiles = []
        self.started = []

    def validate_linux_runtime(self, config):
        self.validated_profiles.append(config.get("track_profile"))

    def start_simulators(self, **kwargs):
        self.started.append(kwargs)


def _manager(tmp_path):
    backend = FakeNativeLinux(tmp_path)
    manager = OfficialRunManager(
        backend=backend,
        registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "logs" / "rl",
        start_monitor=False,
    )
    return manager, backend


def test_native_linux_training_rejects_competition_profile_before_launch(tmp_path):
    manager, backend = _manager(tmp_path)
    try:
        with pytest.raises(ValueError, match="competition-track training is unavailable"):
            manager.create_run({
                "simulator_backend": "linux_external",
                "track_profile": "competition",
                "automatic_evaluation_enabled": False,
            })

        assert not backend.validated_profiles
        assert not backend.started
        assert not backend.containers
        assert not backend.networks
    finally:
        manager.close()


def test_native_linux_practice_training_remains_available(tmp_path):
    manager, backend = _manager(tmp_path)
    try:
        run = manager.create_run({
            "simulator_backend": "linux_external",
            "track_profile": "practice",
            "automatic_evaluation_enabled": False,
        })

        assert run["simulator_image"].endswith("Linux practice release (pinned extracted runtime)")
        assert backend.validated_profiles == ["practice"]
        assert backend.started[0]["config"]["track_profile"] == "practice"
        manager.stop_run(run["run_id"])
    finally:
        manager.close()
