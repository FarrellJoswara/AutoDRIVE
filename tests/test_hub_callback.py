import numpy as np

from src.layer3.hub_callback import HubTelemetryCallback


def test_official_watch_uses_ips_ground_plane_and_frontier_reward_fields():
    callback = HubTelemetryCallback(hub_url="http://localhost", run_id="test", runtime="official")
    info = {
        "race_position": (1.25, -2.5, 0.08),
        "position": (1.25, -2.5, 0.08),
        "yaw": 0.3,
        "frontier_line": [[1.0, -2.0], [1.5, -2.0]],
        "frontier_progress_m": 12.0,
        "current_progress_line": [[1.1, -2.1], [1.6, -2.1]],
        "current_progress_m": 11.5,
        "signed_route_delta_m": -0.2,
        "current_route_speed_mps": 0.0,
        "route_projection_valid": True,
        "reward_components": {"route_progress": 0.0, "total": -0.1},
    }

    sample = callback._build_fleet_sample([info], np.asarray([False]), [0])
    car = sample["cars"][0]

    assert car["pose"] == [1.25, -2.5]
    assert car["frontier_line"] == info["frontier_line"]
    assert car["current_progress_line"] == info["current_progress_line"]
    assert car["current_progress_m"] == 11.5
    assert car["signed_route_delta_m"] == -0.2
    assert car["route_projection_valid"] is True
    assert car["reward_components"] == info["reward_components"]


def test_custom_watch_keeps_unity_xz_ground_plane():
    callback = HubTelemetryCallback(hub_url="http://localhost", run_id="test", runtime="custom")
    sample = callback._build_fleet_sample(
        [{"position": (1.25, 0.08, -2.5), "yaw": 0.3}],
        np.asarray([False]),
        [0],
    )

    assert sample["cars"][0]["pose"] == [1.25, -2.5]


def test_official_camera_frame_is_saved_as_bounded_latest_jpeg(tmp_path):
    from PIL import Image

    callback = HubTelemetryCallback(
        hub_url="http://localhost", run_id="test", runtime="official", run_dir=tmp_path
    )
    callback.locals = {"new_obs": {"camera": np.full((2, 3, 90, 160), 127, dtype=np.uint8)}}
    callback._persist_camera_frames()

    paths = sorted((tmp_path / "camera").glob("*.jpg"))
    assert [path.name for path in paths] == ["env-0.jpg", "env-1.jpg"]
    with Image.open(paths[0]) as image:
        assert image.size == (160, 90)
        assert image.mode == "RGB"
    assert not list((tmp_path / "camera").glob("*.tmp"))
