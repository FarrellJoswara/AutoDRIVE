import sys

import pytest

from src.layer4.hub import native_official_backend
from src.layer4.hub.official_run_manager import OfficialRunManager


@pytest.mark.skipif(sys.platform != "linux", reason="native backend ownership uses the WSL/Linux host")
def test_only_one_native_backend_owns_and_recovers_an_artifact_root(tmp_path, monkeypatch):
    (tmp_path / "logs" / "rl").mkdir(parents=True)
    monkeypatch.setattr(native_official_backend, "ROOT", tmp_path)
    first = native_official_backend.NativeOfficialBackend()
    second = native_official_backend.NativeOfficialBackend()
    try:
        assert first.is_owner is True
        assert second.is_owner is False
        manager = OfficialRunManager(
            backend=second, registry_path=tmp_path / "registry.json",
            runs_root=tmp_path / "logs" / "rl", start_monitor=True,
        )
        try:
            assert manager._monitor is None
            assert manager.list_runs()["runs"] == []
            try:
                manager.create_run({})
            except RuntimeError as exc:
                assert "another Mission Control hub owns" in str(exc)
            else:
                raise AssertionError("non-owner backend must not create runs")
        finally:
            manager.close()
    finally:
        first.close()
