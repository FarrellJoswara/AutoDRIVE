from types import SimpleNamespace
import threading
import time

import numpy as np

from src.layer1.ros2_racer import (
    RaceMetrics,
    RacerRos2,
    _encoder_forward_speed_mps,
    _next_scan_deadline,
    ros_messages_to_bridge_payload,
)
from src.layer1.official_bridge_relay import (
    DEVKIT_REQUIRED_FIELDS,
    NEUTRAL_COMMAND,
    missing_devkit_fields,
)
from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.official_race_env import OfficialRaceEnv
from src.layer2.spaces import map_steering_action, map_throttle_action
from src.layer3.official_evaluate import (
    action_observation_diagnostics,
    collision_penalty_seconds,
    evaluate_attempt,
    _policy_observation_trace,
    summarize_attempt,
    verify_vehicle_motion,
)


def _vector(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _ros_messages():
    return {
        "scan": SimpleNamespace(
            ranges=list(np.linspace(0.1, 9.0, 1080, dtype=np.float32)),
            scan_time=0.025,
            range_min=0.06,
            range_max=10.0,
        ),
        "imu": SimpleNamespace(
            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
            angular_velocity=_vector(7, 8, 9),
            linear_acceleration=_vector(10, 11, 12),
        ),
        "throttle": SimpleNamespace(data=0.4),
        "steering": SimpleNamespace(data=-0.2),
        "left_encoder": SimpleNamespace(position=[13.0]),
        "right_encoder": SimpleNamespace(position=[14.0]),
    }


def test_ros2_adapter_preserves_official_signals_for_existing_bridge_parser():
    messages = _ros_messages()
    payload = ros_messages_to_bridge_payload(**messages, forward_speed_mps=3.5)

    assert payload["V1 Position"] == "0 0 0"
    assert payload["V1 Linear Velocity"] == "3.5 0 0"
    assert payload["V1 Angular Velocity"] == "7.0 8.0 9.0"
    assert payload["V1 Linear Acceleration"] == "10.0 11.0 12.0"
    assert payload["V1 LIDAR Range Array"] == messages["scan"].ranges
    assert payload["V1 Encoder Angles"] == "13.0 14.0"

    snapshot = TelemetrySnapshot.from_raw_dict(payload)
    assert snapshot.position == (0.0, 0.0, 0.0)
    assert snapshot.v_long == 3.5
    assert snapshot.v_lat == 0.0
    assert snapshot.lidar_ranges.size == 1080
    assert snapshot.lidar_ranges[0] == messages["scan"].ranges[-1]


def test_official_collision_penalties_are_cumulative_steps():
    assert collision_penalty_seconds(0) == 0
    assert collision_penalty_seconds(1) == 10
    assert collision_penalty_seconds(2) == 30
    assert collision_penalty_seconds(3) == 60
    result = summarize_attempt(lap_times_s=[5.0] * 10, race_collisions=2)
    assert result["race_time_s"] == 50.0
    assert result["adjusted_race_time_s"] == 80.0


def test_incomplete_attempts_are_not_scored_as_zero_time():
    no_laps = summarize_attempt(lap_times_s=[], race_collisions=0)
    partial = summarize_attempt(lap_times_s=[6.0, 5.0], race_collisions=1)

    assert no_laps["race_time_s"] is None
    assert no_laps["adjusted_race_time_s"] is None
    assert no_laps["complete"] is False
    assert partial["race_time_s"] == 11.0
    assert partial["adjusted_race_time_s"] is None
    assert partial["complete"] is False


def test_official_trace_records_only_policy_input_observations_compactly():
    trace = _policy_observation_trace({
        "lidar": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "state": np.arange(9, dtype=np.float32),
    })

    assert trace["observation_state"] == list(map(float, range(9)))
    assert trace["lidar_min_left"] == 0.1
    assert trace["lidar_min_front"] == 0.4
    assert trace["lidar_min_right"] == 0.7


def test_motion_preflight_rejects_stationary_ips_and_accepts_displacement():
    stationary = verify_vehicle_motion([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]], 3.0)
    moving = verify_vehicle_motion([[1.0, 0.0, 2.0], [1.3, 0.0, 2.0]], 1.0)

    assert stationary["verified"] is False
    assert stationary["max_displacement_m"] == 0.0
    assert moving["verified"] is True
    assert np.isclose(moving["max_displacement_m"], 0.3)


def test_motion_preflight_waits_until_window_when_position_is_static():
    pending = verify_vehicle_motion([[0.0, 0.0, 0.0]], 1.0)
    unavailable = verify_vehicle_motion([], 3.0)

    assert pending["verified"] is None
    assert unavailable["available"] is False
    assert unavailable["verified"] is None


def test_evaluator_stops_stationary_run_and_marks_it_invalid():
    class Model:
        def predict(self, obs, deterministic=True):
            return np.zeros(2, dtype=np.float32), None

    class Env:
        def reset(self):
            return (
                {"lidar": np.ones(8, dtype=np.float32), "state": np.zeros(9, dtype=np.float32)},
                {"race_position": (1.0, 0.0, 2.0), "race_complete": False},
            )

        def step(self, action):
            return (
                {"lidar": np.ones(8, dtype=np.float32), "state": np.zeros(9, dtype=np.float32)},
                0.0,
                False,
                False,
                {
                    "race_position": (1.0, 0.0, 2.0),
                    "race_complete": False,
                    "control_interval_s": 0.05,
                    "lidar_scan_rate_hz": 40.0,
                    "race_lap_times_s": [],
                    "race_collisions": 0,
                    "throttle_command": 0.0,
                    "steering_command": 0.0,
                },
            )

    result = evaluate_attempt(
        Model(), Env(), wall_timeout_s=2.0, max_steps=1000, attempt_index=1
    )

    assert result["stop_reason"] == "vehicle_motion_not_verified"
    assert result["motion_verification"]["verified"] is False
    assert result["race_time_s"] is None
    assert result["adjusted_race_time_s"] is None
    assert result["score_status"] == "vehicle_motion_not_verified"
    assert result["valid_for_comparison"] is False


def test_evaluation_racer_stores_ips_separately_from_policy_telemetry():
    racer = object.__new__(RacerRos2)
    racer.include_race_metrics = True
    racer._condition = threading.Condition()
    racer._race_metrics = RaceMetrics()
    racer._race_metrics_received = set()
    racer._raw = {}
    racer._raw_receipts = {}
    racer._on_message("position", _vector(1.0, 2.0, 3.0))
    assert racer.race_metrics.position == (1.0, 2.0, 3.0)
    assert "position" in racer._race_metrics_received


class FakeRacer:
    def __init__(self, snapshots):
        self.telemetry = snapshots[0]
        self.snapshots = list(snapshots[1:])
        self.last_step_duration_s = 0.025
        self.ready_calls = 0

    def wait_until_ready(self):
        self.ready_calls += 1
        return self.telemetry

    def wait_for_race_metrics(self):
        return self.telemetry

    @property
    def race_metrics(self):
        return self.telemetry

    def step(self, throttle, steering):
        self.telemetry = self.snapshots.pop(0)
        return self.telemetry

    def kill(self):
        pass


def _snapshot(*, lap, last_lap, collisions):
    snap = TelemetrySnapshot(lidar_ranges=np.full(1081, 10.0, dtype=np.float32))
    snap.lap_count = lap
    snap.last_lap_time = last_lap
    snap.collision_count = collisions
    snap.lidar_valid = True
    return snap


def test_race_env_ignores_warmup_collision_and_records_ten_race_laps():
    snapshots = [_snapshot(lap=1, last_lap=6.0, collisions=1)]
    snapshots.extend(_snapshot(lap=lap, last_lap=5.0, collisions=1) for lap in range(2, 12))
    racer = FakeRacer([_snapshot(lap=0, last_lap=0.0, collisions=0), *snapshots])
    env = OfficialRaceEnv(racer=racer)
    _, info = env.reset()

    _, _, terminated, _, info = env.step(np.asarray([0.5, 0.0], dtype=np.float32))
    assert not terminated
    assert info["warmup_lap_times_s"] == [6.0]
    assert info["race_collisions"] == 0

    for _ in range(9):
        _, _, terminated, _, info = env.step(np.asarray([0.5, 0.0], dtype=np.float32))
        assert not terminated
    _, _, terminated, _, info = env.step(np.asarray([0.5, 0.0], dtype=np.float32))
    assert terminated
    assert info["race_laps_completed"] == 10
    assert info["race_collisions"] == 0
    assert info["race_time_s"] == 50.0


def test_official_env_starts_from_live_spawn_without_reset_command():
    racer = FakeRacer([_snapshot(lap=0, last_lap=0.0, collisions=0)])
    env = OfficialRaceEnv(racer=racer)
    env.reset()
    assert racer.ready_calls == 1


def test_restricted_position_is_reported_as_evaluator_info_not_policy_observation():
    racer = FakeRacer([_snapshot(lap=0, last_lap=0.0, collisions=0)])
    env = OfficialRaceEnv(racer=racer)
    obs, info = env.reset()

    assert info["race_position"] == (0.0, 0.0, 0.0)
    assert set(obs) == {"lidar", "state"}


def test_race_env_counts_only_collisions_after_warmup():
    racer = FakeRacer([
        _snapshot(lap=0, last_lap=0.0, collisions=0),
        _snapshot(lap=1, last_lap=6.0, collisions=2),  # both warm-up
        _snapshot(lap=1, last_lap=6.0, collisions=3),  # first race collision
        _snapshot(lap=2, last_lap=5.0, collisions=3),
    ])
    env = OfficialRaceEnv(racer=racer)
    env.reset()
    _, _, _, _, warmup = env.step(np.zeros(2, dtype=np.float32))
    assert warmup["race_collisions"] == 0
    assert warmup["warmup_collisions"] == 2
    assert warmup["race_collision_baseline"] == 2
    _, _, _, _, collision = env.step(np.zeros(2, dtype=np.float32))
    assert collision["race_collisions"] == 1
    _, _, _, _, after_lap = env.step(np.zeros(2, dtype=np.float32))
    assert after_lap["race_collisions"] == 1
    assert after_lap["race_lap_times_s"] == [5.0]


def test_ros_control_waits_for_scan_after_command_and_declared_period():
    racer = object.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._step_counter = 0
    racer._last_scan_receipt = 0.0
    racer._closed = False
    racer.timeout_s = 0.5
    racer._telemetry = object()
    command_time = time.monotonic()
    earliest_time = command_time + 0.04

    def publish_scan(receipt_time):
        with racer._condition:
            racer._step_counter += 1
            racer._last_scan_receipt = receipt_time
            racer._telemetry = racer._step_counter
            racer._condition.notify_all()

    def send_scans():
        time.sleep(0.01)
        publish_scan(time.monotonic())  # duplicate/early message is ignored
        time.sleep(0.04)
        publish_scan(time.monotonic())

    publisher = threading.Thread(target=send_scans)
    publisher.start()
    started = time.monotonic()
    result = racer._wait_for_control_frame(
        0, command_time=command_time, earliest_time=earliest_time
    )
    elapsed = time.monotonic() - started
    publisher.join()

    assert result == 2
    assert elapsed >= 0.04


def test_ros_control_cadence_is_measured_between_delivered_scans():
    assert np.isclose(_next_scan_deadline(10.0, 9.99, 40.0), 10.015)
    assert np.isclose(_next_scan_deadline(10.03, 9.99, 40.0), 10.03)
    assert np.isclose(_next_scan_deadline(10.0, 0.0, 40.0), 10.0)


def test_encoder_speed_uses_accumulated_wheel_angles_without_aliasing():
    speed = _encoder_forward_speed_mps(
        current=(4.2, 4.2),
        previous=(0.0, 0.0),
        elapsed_s=0.025,
    )
    expected = 4.2 * 0.059 / 0.025
    assert np.isclose(speed, expected)


def test_official_sensor_observation_profile_uses_lidar_motion_not_wheel_rotation():
    from src.layer2.lidar_odometry import ScanMotion
    from src.layer2.autodrive_env import AutoDriveEnv

    class FixedMotionEstimator:
        def update(self, ranges_m, elapsed_s, *, yaw_delta_rad=None):
            assert elapsed_s == 0.1
            return ScanMotion(forward_m=2.5, valid=True)

    env = object.__new__(AutoDriveEnv)
    env.observation_profile = "official_sensors"
    env._lidar_speed_estimator = FixedMotionEstimator()
    env._observation_heading_yaw = None
    snap = _snapshot(lap=0, last_lap=0.0, collisions=0)
    snap.v_long = 99.0  # simulator-only velocity must not reach this profile
    snap.v_lat = 7.0
    snap.encoder_left = 200.0
    snap.encoder_right = 200.0
    obs = env._policy_observation(
        snap, 0.0, 0.0, lidar_beams=1081, elapsed_s=0.1
    )
    expected_speed = 2.5
    assert np.isclose(obs["state"][0], expected_speed / 22.88)
    assert obs["state"][1] == 0.0


def test_bridge_relay_neutralizes_only_the_incomplete_startup_packet():
    complete_packet = {key: "value" for key in DEVKIT_REQUIRED_FIELDS}
    startup_packet = dict(complete_packet)
    startup_packet.pop("V1 LIDAR Range Array")

    assert missing_devkit_fields(startup_packet) == ("V1 LIDAR Range Array",)
    assert missing_devkit_fields(complete_packet) == ()
    assert NEUTRAL_COMMAND == {
        "V1 Throttle": "0.0000",
        "V1 Steering": "0.0000",
        "V1 Reset": "False",
    }


def test_negative_throttle_mapping_modes_are_explicit_and_bounded():
    assert map_throttle_action(-0.8, "allow") == -0.8
    assert map_throttle_action(-0.8, "zero") == 0.0
    assert map_throttle_action(-0.8, "positive_magnitude") == 0.8
    assert map_throttle_action(1.4, "allow") == 1.0
    assert map_throttle_action(-1.4, "positive_magnitude") == 1.0


def test_steering_mapping_modes_are_explicit_and_bounded():
    assert map_steering_action(0.8, "normal") == 0.8
    assert map_steering_action(0.8, "invert") == -0.8
    assert map_steering_action(-1.4, "invert") == 1.0


def test_evaluation_diagnostics_capture_only_policy_io():
    result = action_observation_diagnostics(
        policy_throttle=[-1.0, 0.5],
        policy_steering=[-0.25, 0.25],
        applied_throttle=[0.0, 0.5],
        applied_steering=[-0.25, 0.25],
        observation_states=[[0.1] * 9, [0.3] * 9],
        normalized_lidar_minima=[0.2, 0.4],
    )
    assert result["policy_reverse_fraction"] == 0.5
    assert result["policy_throttle"]["mean"] == -0.25
    assert result["applied_throttle"]["mean"] == 0.25
    assert result["observation_state"]["forward_speed"]["mean"] == 0.2
    assert result["normalized_lidar_minimum"]["min"] == 0.2
    assert "lap" not in result and "collision" not in result
