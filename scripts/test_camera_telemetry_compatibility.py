"""Camera-key compatibility check for the AutoDRIVE Bridge telemetry parser.

This is a parser/observation/reward invariance test only. It does not measure
Unity rendering cost, Bridge cadence, or closed-loop simulator equivalence.
"""

from __future__ import annotations

import base64
import gzip
from dataclasses import fields

import numpy as np
import pytest

from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.rewards import RewardConfig, compute_reward_components
from src.layer2.spaces import snapshot_to_obs


def _compressed_unity_scan() -> str:
    """Encode ranges in the Unity DataCompressor newline/GZip/Base64 format."""
    # Include a finite return, a suppressed near return, and a no-return value.
    # Fill to the canonical live 1081 beams used by the simulator.
    ranges = ["1.25", "0.05", "inf"] + ["4.5"] * 1078
    raw = "\n".join(ranges).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


@pytest.fixture
def raw_bridge_packet() -> dict[str, object]:
    return {
        "V1 Position": "-1.3900 -7.5350 0.0861",
        "V1 Orientation Quaternion": "0.0103 0.0103 -0.7094 0.7047",
        "V1 Orientation Euler Angles": "0.0021 0.0291 4.7058",
        "V1 Linear Velocity": "3.0 0.0 0.2",
        "V1 Angular Velocity": "0.0 0.1 0.55",
        "V1 Linear Acceleration": "0.4 0.2 9.80665",
        "V1 Encoder Angles": "12.5 12.8",
        "V1 Encoder Ticks": "100 102",
        "V1 LIDAR Range Array": _compressed_unity_scan(),
        "V1 LIDAR Scan Rate": "40.0",
        "V1 LIDAR Range Min": "0.06",
        "V1 LIDAR Range Max": "10.0",
        "V1 Throttle": "0.8",
        "V1 Steering": "-0.15",
        "V1 Lap Count": "2",
        "V1 Lap Time": "14.25",
        "V1 Last Lap Time": "14.10",
        "V1 Best Lap Time": "13.90",
        "V1 Collisions": "0",
    }


CAMERA_KEYS = (
    "V1 Front Camera Image",
    "V1 Rear Camera Image",
    "V1 Left Camera Image",
    "V1 Right Camera Image",
)


def _assert_same_snapshot(left: TelemetrySnapshot, right: TelemetrySnapshot) -> None:
    """Compare all parsed snapshot data except the wall-clock timestamp."""
    for field in fields(TelemetrySnapshot):
        name = field.name
        if name in {"timestamp", "lidar_ranges"}:
            continue
        assert getattr(left, name) == getattr(right, name), name
    np.testing.assert_array_equal(left.lidar_ranges, right.lidar_ranges)


def _observation_and_reward(snap: TelemetrySnapshot):
    observation = snapshot_to_obs(snap, prev_throttle=0.25, prev_steering=-0.1)
    reward = compute_reward_components(
        v_long=snap.v_long,
        step_duration_s=1.0 / 40.0,
        frontier_advanced_m=0.125,
        positive_episode_return=1.75,
        collision_event=snap.collision,
        episode_failure=False,
        slip_angle=snap.slip_angle,
        prev_steering=-0.1,
        steering=-0.15,
        cfg=RewardConfig(),
    )
    return observation, reward


@pytest.mark.parametrize("camera_key", CAMERA_KEYS)
def test_each_optional_camera_key_does_not_change_rl_transition_inputs(
    raw_bridge_packet: dict[str, object], camera_key: str
) -> None:
    without_camera = TelemetrySnapshot.from_raw_dict(raw_bridge_packet, step_id=17)
    packet_with_camera = dict(raw_bridge_packet)
    packet_with_camera[camera_key] = "not-decoded-by-the-rl-telemetry-parser"
    with_camera = TelemetrySnapshot.from_raw_dict(packet_with_camera, step_id=17)

    _assert_same_snapshot(without_camera, with_camera)
    assert without_camera.lidar_valid
    assert without_camera.lidar_ranges.shape == (1081,)
    assert np.isinf(without_camera.lidar_ranges).any()

    obs_without, reward_without = _observation_and_reward(without_camera)
    obs_with, reward_with = _observation_and_reward(with_camera)
    assert obs_without.keys() == obs_with.keys() == {"lidar", "state"}
    for key in obs_without:
        np.testing.assert_array_equal(obs_without[key], obs_with[key])
    assert reward_without == reward_with


def test_all_optional_camera_keys_can_be_absent_without_changing_rl_inputs(
    raw_bridge_packet: dict[str, object],
) -> None:
    full_packet = dict(raw_bridge_packet)
    full_packet.update({key: "camera-payload" for key in CAMERA_KEYS})

    with_camera = TelemetrySnapshot.from_raw_dict(full_packet, step_id=23)
    without_camera = TelemetrySnapshot.from_raw_dict(raw_bridge_packet, step_id=23)
    _assert_same_snapshot(with_camera, without_camera)

    obs_with, reward_with = _observation_and_reward(with_camera)
    obs_without, reward_without = _observation_and_reward(without_camera)
    for key in obs_with:
        np.testing.assert_array_equal(obs_with[key], obs_without[key])
    assert reward_with == reward_without

