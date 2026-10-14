import hashlib
import asyncio
import importlib
import json

import pytest

from src.layer4.hub.official_models import official_model_metadata
from src.layer4.official_settings import OfficialTrainSettings


def _official_checkpoint(root, relative="run/checkpoints/policy.zip", data=b"model-bytes"):
    path = root / relative
    run_dir = path.parent.parent if path.parent.name in {"checkpoints", "ckpt"} else path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = OfficialTrainSettings()
    config = {
        "runtime": "official IROS 2026 API and simulator images",
        "observation_profile": "official_sensors_camera",
        "simulator_backend": "windows_external",
        "actions": {
            key: getattr(settings, key)
            for key in (
                "throttle_mode", "negative_throttle_mode", "steering_mode",
                "steering_action_scale", "straight_throttle_gain",
                "straight_throttle_steering_threshold",
            )
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_selected_checkpoint_metadata_hashes_only_exact_catalog_member(tmp_path):
    catalog = tmp_path / "logs" / "rl"
    model = _official_checkpoint(catalog)

    metadata = official_model_metadata(catalog, "run/checkpoints/policy.zip")

    assert metadata == {
        "id": "run/checkpoints/policy.zip",
        "sha256": hashlib.sha256(b"model-bytes").hexdigest(),
        "simulator_backend": "windows_external",
        "observation_profile": "official_sensors_camera",
        "throttle_mode": OfficialTrainSettings().throttle_mode,
        "negative_throttle_mode": OfficialTrainSettings().negative_throttle_mode,
        "steering_mode": OfficialTrainSettings().steering_mode,
        "steering_action_scale": OfficialTrainSettings().steering_action_scale,
        "straight_throttle_gain": OfficialTrainSettings().straight_throttle_gain,
        "straight_throttle_steering_threshold": OfficialTrainSettings().straight_throttle_steering_threshold,
    }
    assert model.is_file()


def test_selected_checkpoint_metadata_detects_replaced_bytes_at_same_id(tmp_path):
    catalog = tmp_path / "logs" / "rl"
    model = _official_checkpoint(catalog)
    first = official_model_metadata(catalog, "run/checkpoints/policy.zip")

    model.write_bytes(b"replacement-model-bytes")
    refreshed = official_model_metadata(catalog, "run/checkpoints/policy.zip")

    assert refreshed["id"] == first["id"]
    assert refreshed["sha256"] == hashlib.sha256(b"replacement-model-bytes").hexdigest()
    assert refreshed["sha256"] != first["sha256"]


@pytest.mark.parametrize("model_id", ["../outside.zip", "run/../../outside.zip", "/etc/passwd", "run\\checkpoints\\policy.zip", "run/config.json"])
def test_selected_checkpoint_metadata_rejects_non_catalog_ids(tmp_path, model_id):
    catalog = tmp_path / "logs" / "rl"
    _official_checkpoint(catalog)

    with pytest.raises(ValueError):
        official_model_metadata(catalog, model_id)


def test_selected_checkpoint_metadata_rejects_symlink_escape(tmp_path):
    catalog = tmp_path / "logs" / "rl"
    outside = _official_checkpoint(tmp_path / "outside", "run/checkpoints/policy.zip")
    catalog.mkdir(parents=True)
    run_link = catalog / "linked"
    try:
        run_link.symlink_to(outside.parents[1], target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not available in this environment")

    with pytest.raises(ValueError):
        official_model_metadata(catalog, "linked/checkpoints/policy.zip")


def test_replay_metadata_endpoint_returns_selected_archive_identity(tmp_path, monkeypatch):
    hub_app = importlib.import_module("src.layer4.hub.app")
    _official_checkpoint(tmp_path / "logs" / "rl")
    monkeypatch.setattr(hub_app, "ROOT", tmp_path)

    metadata = asyncio.run(hub_app.replay_model_metadata("run/checkpoints/policy.zip"))

    assert metadata["id"] == "run/checkpoints/policy.zip"
    assert metadata["sha256"] == hashlib.sha256(b"model-bytes").hexdigest()


def test_replay_metadata_endpoint_returns_not_found_for_escape(tmp_path, monkeypatch):
    from fastapi import HTTPException

    hub_app = importlib.import_module("src.layer4.hub.app")
    monkeypatch.setattr(hub_app, "ROOT", tmp_path)

    with pytest.raises(HTTPException) as error:
        asyncio.run(hub_app.replay_model_metadata("../outside.zip"))
    assert error.value.status_code == 404
