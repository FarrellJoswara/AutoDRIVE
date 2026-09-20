"""Test Layer 1 Socket.IO server, lockstep stepping, reset, and fleet kill controls."""

import threading
import time
import tempfile
import socketio
from src.racer.track import RaceTrack


def test_lockstep_stepping_and_kill():
    """Test end-to-end Socket.IO lockstep communication using a mock Unity client."""
    port = 18888
    track = RaceTrack(num_racers=1, base_port=port, auto_launch=False)

    client = socketio.Client()
    received_commands = []
    running = True

    @client.on("Bridge")
    def on_bridge_reply(data):
        received_commands.append(data)

    try:
        # 1. Connect mock Unity client to Racer server
        client.connect(f"http://127.0.0.1:{port}")
        time.sleep(0.2)
        assert client.connected

        # 2. Simulate Unity physics loop in background thread
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
                    # Official protocol: emit Bridge telemetry; server replies via emit (not ACK)
                    client.emit("Bridge", fake_bridge_data)
                    step_num += 1
                except Exception:
                    break
                time.sleep(0.01)

        sim_thread = threading.Thread(target=unity_simulation_loop, daemon=True)
        sim_thread.start()

        # Allow initial handshake to establish
        time.sleep(0.05)

        # 3. Step the racer with specific control commands
        snap = track.step_single(0, throttle=0.85, steering=-0.3)
        assert snap.position[2] > 0.0
        assert snap.true_speed == 10.0

        # Step again to verify turn-based responsiveness
        snap2 = track.step_single(0, throttle=0.5, steering=0.1)
        assert snap2.step_id >= 2

        # Verify command emitted back to Unity (string-valued official format)
        deadline = time.time() + 2.0
        while time.time() < deadline and not received_commands:
            time.sleep(0.05)
        assert len(received_commands) >= 1
        matching_commands = [
            c for c in received_commands
            if c is not None and abs(float(c.get("V1 Throttle", 0.0)) - 0.85) < 0.05
        ]
        assert len(matching_commands) > 0
        assert "V1 Reset" in matching_commands[0]

        # 4. Verify Profiler
        racer = track.racers[0]
        assert racer.step_counter >= 2
        assert racer.step_latency_ms >= 0.0

        # 5. Test Batch CSV Export
        with tempfile.TemporaryDirectory() as tmp_dir:
            exported = track.save_all_trajectories(tmp_dir)
            assert 0 in exported
            with open(exported[0], "r") as f:
                lines = f.readlines()
                # Header + recorded steps
                assert len(lines) >= 2

        # 6. Test Kill Racer & Process Termination
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
