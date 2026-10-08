from types import SimpleNamespace

import numpy as np
import pytest

from src.layer1.ros2_racer import ros_image_to_rgb, ros_messages_to_bridge_payload
from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.official_race_env import OfficialObservationBuilder
from src.layer2.spaces import CAMERA_HEIGHT, CAMERA_WIDTH, make_observation_space


def _vehicle_messages():
    vector = lambda x=0.0, y=0.0, z=0.0: SimpleNamespace(x=x, y=y, z=z)
    return {
        "scan": SimpleNamespace(
            ranges=[5.0] * 1081, scan_time=0.025, range_min=0.06, range_max=10.0
        ),
        "imu": SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=vector(),
            linear_acceleration=vector(),
        ),
        "throttle": SimpleNamespace(data=0.0),
        "steering": SimpleNamespace(data=0.0),
        "left_encoder": SimpleNamespace(position=[0.0]),
        "right_encoder": SimpleNamespace(position=[0.0]),
    }


def test_ros_image_decoder_converts_bgr_and_ignores_row_padding():
    # Each 2-pixel BGR row includes two padding bytes after the pixels.
    message = SimpleNamespace(
        height=2,
        width=2,
        encoding="bgr8",
        step=8,
        data=bytes([0, 0, 255, 0, 255, 0, 99, 99, 255, 0, 0, 255, 255, 255, 99, 99]),
    )

    rgb = ros_image_to_rgb(message)

    assert rgb.dtype == np.uint8
    assert rgb.shape == (2, 2, 3)
    np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
    np.testing.assert_array_equal(rgb[0, 1], [0, 255, 0])
    np.testing.assert_array_equal(rgb[1, 0], [0, 0, 255])
    np.testing.assert_array_equal(rgb[1, 1], [255, 255, 255])


def test_camera_sensor_is_carried_through_layer1_snapshot():
    image = np.full((12, 20, 3), 127, dtype=np.uint8)
    payload = ros_messages_to_bridge_payload(
        **_vehicle_messages(), camera_image_rgb=image
    )

    snap = TelemetrySnapshot.from_raw_dict(payload)

    np.testing.assert_array_equal(snap.camera_image_rgb, image)


def test_layer2_camera_profile_includes_fixed_rgb_image_and_requires_data():
    snap = TelemetrySnapshot(
        lidar_ranges=np.full(1081, 10.0, dtype=np.float32),
        lidar_valid=True,
        camera_image_rgb=np.full((180, 320, 3), 42, dtype=np.uint8),
    )
    builder = OfficialObservationBuilder(observation_profile="official_sensors_camera")
    obs = builder.reset(snap)
    space = make_observation_space(include_camera=True)

    assert obs["camera"].shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
    assert obs["camera"].dtype == np.uint8
    assert space.contains(obs)
    assert np.all(obs["camera"] == 42)

    missing = TelemetrySnapshot(lidar_ranges=np.full(1081, 10.0), lidar_valid=True)
    with pytest.raises(ValueError, match="camera image is required"):
        builder.reset(missing)


def test_camera_policy_extractor_accepts_layer2_observation():
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from src.layer3.extractors import LidarCameraStateExtractor

    observation_space = make_observation_space(include_camera=True)
    extractor = LidarCameraStateExtractor(observation_space)
    batch = {
        "lidar": torch.zeros((2, 1081), dtype=torch.float32),
        "state": torch.zeros((2, 9), dtype=torch.float32),
        "camera": torch.zeros((2, CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=torch.uint8),
    }

    with torch.no_grad():
        features = extractor(batch)

    assert tuple(features.shape) == (2, 256)
    assert isinstance(observation_space, gym.spaces.Dict)
