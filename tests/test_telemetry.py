"""Unit tests for TelemetrySnapshot, TrajectoryLogger, and RaceTrack geometry math."""

import math
import os
import tempfile
import numpy as np
import pytest

from src.racer.telemetry import TelemetrySnapshot, TrajectoryLogger
from src.racer.track import RaceTrack


def test_telemetry_snapshot_parsing():
    """Test parsing of raw AutoDRIVE Bridge data packet."""
    raw_packet = {
        "V1 Position": [10.5, 0.2, -5.3],
        "V1 Orientation": [0.0, 0.7071, 0.0, 0.7071],  # 90 deg yaw
        "V1 Linear Velocity": [3.0, 0.0, 4.0],  # 5.0 m/s
        "V1 Linear Acceleration": [0.0, 0.0, 9.80665],
        "V1 Left Encoder": 12.5,
        "V1 Right Encoder": 12.8,
        "V1 Lidar Scan": [1.5] * 1080,
        "V1 Throttle": 0.8,
        "V1 Steering": -0.15,
        "V1 Lap Count": 2,
        "V1 Lap Time": 14.25,
        "V1 Collision": False,
    }

    snap = TelemetrySnapshot.from_raw_dict(raw_packet, step_id=42)

    assert snap.step_id == 42
    assert snap.position == (10.5, 0.2, -5.3)
    assert pytest.approx(snap.true_speed, 0.001) == 5.0
    assert pytest.approx(snap.encoder_left, 0.001) == 12.5
    assert pytest.approx(snap.encoder_right, 0.001) == 12.8
    assert len(snap.lidar_ranges) == 1080
    assert snap.lidar_ranges[0] == 1.5
    assert snap.throttle == 0.8
    assert snap.steering == -0.15
    assert snap.lap_count == 2
    assert not snap.collision
    # Check yaw: orientation (0, 0.7071, 0, 0.7071) corresponds to 90 deg (pi/2)
    assert pytest.approx(snap.heading_yaw, 0.01) == math.pi / 2


def test_trajectory_logger_csv_export():
    """Test logging snapshots and exporting to CSV."""
    logger = TrajectoryLogger()

    snap1 = TelemetrySnapshot(
        step_id=1,
        position=(1.0, 0.0, 2.0),
        orientation_quat=(0.0, 0.0, 0.0, 1.0),
        true_speed=3.5,
        throttle=0.5,
        steering=0.0,
    )
    snap2 = TelemetrySnapshot(
        step_id=2,
        position=(2.0, 0.0, 4.0),
        orientation_quat=(0.0, 0.0, 0.0, 1.0),
        true_speed=4.0,
        throttle=0.7,
        steering=0.1,
    )

    logger.log(snap1)
    logger.log(snap2)
    assert len(logger) == 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_log.csv")
        saved_file = logger.save_to_csv(csv_path)
        assert os.path.exists(saved_file)

        with open(saved_file, "r") as f:
            lines = f.readlines()
            # Header + 2 rows = 3 lines
            assert len(lines) == 3
            assert "step_id,timestamp,pos_x" in lines[0]


def test_racetrack_geometry_and_checkpoints():
    """Test Frenet progress and checkpoint tracking without spawning servers."""
    # Create an oval/box track
    waypoints = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 10.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ])

    track = RaceTrack(
        num_racers=1,
        base_port=19999,  # High unused port
        num_gates=4,
        waypoints=waypoints,
        auto_launch=False,
    )

    try:
        assert track.track_length > 0.0
        assert len(track.gate_locations) == 4

        # Test Frenet projection at start
        s, d = track.get_frenet_progress(0.0, 0.0)
        assert pytest.approx(s, 0.1) == 0.0
        assert pytest.approx(d, 0.1) == 0.0

        # Test gate passage
        passed, lap = track.update_checkpoints(0, 0.1)
        assert passed  # Gate 0 passed

    finally:
        track.kill_all()
