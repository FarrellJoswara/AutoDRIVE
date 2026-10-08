from types import SimpleNamespace

import numpy as np
import pytest

from src.layer1.ros2_racer import ros_image_to_rgb, ros_messages_to_bridge_payload
from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.official_race_env import OfficialObservationBuilder
from src.layer2.spaces import CAMERA_HEIGHT, CAMERA_WIDTH, make_observation_space
from src.layer2.autodrive_env import AutoDriveEnv
from src.layer2.official_race_env import OfficialRaceEnv
from src.layer4.official_settings import OfficialTrainSettings
from src.layer4.settings import Settings


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


@pytest.mark.parametrize(
    ("profile", "camera_expected"),
    (("simulator", False), ("simulator_camera", True),
     ("official_sensors", False), ("official_sensors_camera", True)),
)
def test_local_environment_camera_profiles_match_space_and_observation(profile, camera_expected):
    snap = TelemetrySnapshot(
        lidar_ranges=np.full(1081, 10.0, dtype=np.float32),
        lidar_valid=True,
        camera_image_rgb=np.full((180, 320, 3), 17, dtype=np.uint8),
    )
    env = AutoDriveEnv.__new__(AutoDriveEnv)
    env.observation_profile = profile
    env.lidar_beams = 1081
    env._lidar_speed_estimator = None
    env._observation_heading_yaw = None
    env._last_lidar_forward_speed_mps = 0.0
    env.observation_space = make_observation_space(include_camera=camera_expected)

    obs = env._policy_observation(snap, 0.0, 0.0, lidar_beams=1081, elapsed_s=0.025)

    assert ("camera" in obs) is camera_expected
    assert env.observation_space.contains(obs)


def test_new_training_defaults_use_camera_for_local_and_official_simulators():
    assert Settings().observation_profile == "simulator_camera"
    assert Settings().policy_architecture == "lidar_camera_cnn"
    assert OfficialTrainSettings().observation_profile == "official_sensors_camera"


def test_official_frontier_stall_is_charged_as_episode_failure():
    snap = TelemetrySnapshot(
        lidar_ranges=np.full(1081, 10.0, dtype=np.float32),
        lidar_valid=True,
    )

    class FakeRacer:
        race_metrics = SimpleNamespace(
            collision_count=0, lap_count=0, last_lap_time=0.0,
            position=(0.0, 0.0, 0.0),
        )
        last_step_duration_s = 0.1
        last_control_interval_s = 0.1
        last_control_wall_interval_s = 0.1
        control_interval_source = "test"
        scan_interval_source = "test"

        def step(self, throttle, steering):
            return snap

    class FakeRoute:
        def update(self, x, z, *, now):
            return {
                "advanced_m": 0.0, "progress_m": 4.0,
                "current_progress_m": 4.0, "current_delta_m": 0.0,
                "current_projection_valid": True,
            }

    class FakeObservationBuilder:
        last_forward_speed_mps = 0.0

        def observe(self, *args, **kwargs):
            return {"test": np.zeros(1, dtype=np.float32)}

    env = OfficialRaceEnv.__new__(OfficialRaceEnv)
    env.throttle_mode = "bidirectional"
    env.negative_throttle_mode = "allow"
    env.steering_mode = "normal"
    env.steering_action_scale = 1.0
    env.straight_throttle_gain = 1.0
    env.straight_throttle_steering_threshold = 0.15
    env.racer = FakeRacer()
    env._episode_steps = 0
    env._last_collision_count = 0
    env._last_lap_count = 0
    env._initial_lap_count = 0
    env._warmup_lap_times_s = []
    env._race_laps_count = 0
    env._race_lap_times_s = []
    env._race_collision_baseline = 0
    env._initial_collision_count = 0
    env._warmup_collision_count = 0
    env._last_snap = snap
    env._observation_builder = FakeObservationBuilder()
    env.race_laps = 10
    env.warmup_laps = 0
    env.training_mode = True
    env._training_elapsed_s = 9.9
    env.training_frontier_stagnation_s = 10.0
    env.training_timeout_s = 600.0
    env.route_progress = FakeRoute()
    env._episode_frontier_distance_m = 4.0
    env._frontier_last_push_s = 0.0
    env._positive_episode_return = 25.0
    env._last_training_steering = 0.0
    env.training_time_cost_per_simulated_second = 5.0
    env.training_collision_penalty_magnitude = 100.0
    env.training_collision_reward_percent = 50.0
    env.training_failure_penalty = 100.0
    env.training_failure_reward_percent = 50.0
    env.training_backward_speed_penalty_scale = 10.0
    env._build_info = lambda *args, **kwargs: {}

    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert terminated and not truncated
    assert info["termination_reason"] == "frontier_stagnation"
    assert info["training_reward_components"]["episode_failure"] == -112.5
    assert reward < -112.5


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
