"""Layer 1 unit + mock Socket.IO tests (no Unity binary required)."""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time

import numpy as np
import pytest
import socketio

from src.racer.telemetry import TelemetrySnapshot, TrajectoryLogger
from src.racer.track import RaceTrack


# ---------------------------------------------------------------------------
# Telemetry / geometry
# ---------------------------------------------------------------------------


def test_telemetry_snapshot_parsing():
    """Parse a raw AutoDRIVE Bridge packet into TelemetrySnapshot."""
    raw_packet = {
        "V1 Position": [10.5, 0.2, -5.3],
        "V1 Orientation": [0.0, 0.7071, 0.0, 0.7071],
        "V1 Linear Velocity": [3.0, 0.0, 4.0],
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
    assert pytest.approx(snap.heading_yaw, 0.01) == math.pi / 2


def test_trajectory_logger_csv_export():
    """Buffer snapshots and export a CSV with header + rows."""
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
            assert len(lines) == 3
            assert "step_id,timestamp,pos_x" in lines[0]


def test_racetrack_geometry_and_checkpoints():
    """Frenet progress and gate tracking without needing a live Unity client."""
    waypoints = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [20.0, 10.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0],
        ]
    )
    track = RaceTrack(
        num_racers=1,
        base_port=19999,
        num_gates=4,
        waypoints=waypoints,
        auto_launch=False,
    )
    try:
        assert track.track_length > 0.0
        assert len(track.gate_locations) == 4
        s, d = track.get_frenet_progress(0.0, 0.0)
        assert pytest.approx(s, 0.1) == 0.0
        assert pytest.approx(d, 0.1) == 0.0
        passed, _lap = track.update_checkpoints(0, 0.1)
        assert passed
    finally:
        track.kill_all()


# ---------------------------------------------------------------------------
# Mock Socket.IO lockstep (1 car)
# ---------------------------------------------------------------------------


def test_lockstep_stepping_and_kill():
    """One mock Unity client: step, command emit, CSV export, kill."""
    port = 18888
    track = RaceTrack(num_racers=1, base_port=port, auto_launch=False)

    client = socketio.Client()
    received_commands = []
    running = True

    @client.on("Bridge")
    def on_bridge_reply(data):
        received_commands.append(data)

    try:
        client.connect(f"http://127.0.0.1:{port}")
        time.sleep(0.2)
        assert client.connected

        def unity_simulation_loop():
            step_num = 1
            while running and client.connected:
                fake_bridge_data = {
                    "V1 Position": [step_num * 0.1, 0.0, step_num * 0.5],
                    "V1 Orientation": [0.0, 0.0, 0.0, 1.0],
                    "V1 Linear Velocity": [0.0, 0.0, 10.0],
                    "V1 Linear Acceleration": [0.0, 0.0, 1.0],
                    "V1 Left Encoder": step_num * 1.5,
                    "V1 Right Encoder": step_num * 1.5,
                    "V1 Lidar Scan": [5.0] * 1080,
                    "V1 Throttle": 0.5,
                    "V1 Steering": 0.0,
                    "V1 Lap Count": 0,
                    "V1 Lap Time": step_num * 0.025,
                    "V1 Collision": False,
                }
                try:
                    client.emit("Bridge", fake_bridge_data)
                    step_num += 1
                except Exception:
                    break
                time.sleep(0.01)

        threading.Thread(target=unity_simulation_loop, daemon=True).start()
        time.sleep(0.05)

        snap = track.step_single(0, throttle=0.85, steering=-0.3)
        assert snap.position[2] > 0.0
        assert snap.true_speed == 10.0

        snap2 = track.step_single(0, throttle=0.5, steering=0.1)
        assert snap2.step_id >= 2

        deadline = time.time() + 2.0
        while time.time() < deadline and not received_commands:
            time.sleep(0.05)
        assert len(received_commands) >= 1
        matching_commands = [
            c
            for c in received_commands
            if c is not None and abs(float(c.get("V1 Throttle", 0.0)) - 0.85) < 0.05
        ]
        assert len(matching_commands) > 0
        assert "V1 Reset" in matching_commands[0]

        racer = track.racers[0]
        assert racer.step_counter >= 2
        assert racer.step_latency_ms >= 0.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            exported = track.save_all_trajectories(tmp_dir)
            assert 0 in exported
            with open(exported[0], "r") as f:
                assert len(f.readlines()) >= 2

        killed = track.kill_racer(0)
        assert killed
        assert not track.racers[0].is_alive
    finally:
        running = False
        try:
            if client.connected:
                client.disconnect()
        except Exception:
            pass
        try:
            track.kill_all()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mock Socket.IO lockstep (2 cars)
# ---------------------------------------------------------------------------


def test_two_instances_lockstep_drive():
    """Two mock Unity clients stepped together via RaceTrack.step_all."""
    base_port = 24567
    track = RaceTrack(num_racers=2, base_port=base_port, auto_launch=False)

    assert len(track.racers) == 2
    assert track.racers[0].port == 24567
    assert track.racers[1].port == 24568

    client0 = socketio.Client()
    client1 = socketio.Client()
    running = True
    received_0 = []
    received_1 = []

    @client0.on("Bridge")
    def on_bridge_0(data):
        received_0.append(data)

    @client1.on("Bridge")
    def on_bridge_1(data):
        received_1.append(data)

    try:
        client0.connect(f"http://127.0.0.1:{base_port}")
        client1.connect(f"http://127.0.0.1:{base_port + 1}")
        assert client0.connected
        assert client1.connected

        def sim_loop(client, car_idx):
            step = 1
            while running and client.connected:
                data = {
                    "V1 Position": f"{car_idx * 2.0} 0.0 {step * 0.5}",
                    "V1 Orientation": "0.0 0.0 0.0 1.0",
                    "V1 Linear Velocity": "0.0 0.0 8.5",
                    "V1 Linear Acceleration": "0.0 0.0 0.5",
                    "V1 Encoder Angles": f"{step * 1.2} {step * 1.2}",
                    "V1 Throttle": "0.6",
                    "V1 Steering": "0.0",
                    "V1 Lap Count": "0",
                    "V1 Lap Time": f"{step * 0.025:.3f}",
                    "V1 Collision": "0",
                }
                try:
                    client.emit("Bridge", data)
                    step += 1
                except Exception:
                    break
                time.sleep(0.01)

        threading.Thread(target=sim_loop, args=(client0, 0), daemon=True).start()
        threading.Thread(target=sim_loop, args=(client1, 1), daemon=True).start()
        time.sleep(0.05)

        results = track.step_all([(0.6, 0.0), (0.6, 0.0)])
        assert 0 in results and 1 in results
        assert results[0].position[2] > 0.0
        assert results[1].position[2] > 0.0
        assert results[0].true_speed == 8.5
        assert results[1].true_speed == 8.5

        results2 = track.step_all([(0.6, 0.0), (0.6, 0.0)])
        assert results2[0].step_id >= 2
        assert results2[1].step_id >= 2
        assert track.racers[0].step_counter >= 2
        assert track.racers[1].step_counter >= 2

        deadline = time.time() + 2.0
        while time.time() < deadline and (not received_0 or not received_1):
            time.sleep(0.05)
        assert len(received_0) >= 1
        assert len(received_1) >= 1
    finally:
        running = False
        try:
            if client0.connected:
                client0.disconnect()
            if client1.connected:
                client1.disconnect()
        except Exception:
            pass
        try:
            track.kill_all()
        except Exception:
            pass
