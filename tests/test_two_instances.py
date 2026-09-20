import threading
import time
import socketio
from src.racer.track import RaceTrack


def test_two_instances_lockstep_drive():
    """Verify Layer 1 multi-instance orchestration with 2 concurrent racers driving forward."""
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
        # Connect both mock simulator instances
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
                    # Official protocol: emit Bridge telemetry; server replies via emit
                    client.emit("Bridge", data)
                    step += 1
                except Exception:
                    break
                time.sleep(0.01)

        t0 = threading.Thread(target=sim_loop, args=(client0, 0), daemon=True)
        t1 = threading.Thread(target=sim_loop, args=(client1, 1), daemon=True)
        t0.start()
        t1.start()

        time.sleep(0.05)

        # Step both vehicles concurrently: throttle=0.6 (drive forward), steering=0.0 (straight)
        actions = [(0.6, 0.0), (0.6, 0.0)]
        results = track.step_all(actions)

        assert 0 in results
        assert 1 in results

        snap0 = results[0]
        snap1 = results[1]

        # Both cars drove forward (Z coordinate increased)
        assert snap0.position[2] > 0.0
        assert snap1.position[2] > 0.0
        assert snap0.true_speed == 8.5
        assert snap1.true_speed == 8.5

        # Step second time
        results2 = track.step_all([(0.6, 0.0), (0.6, 0.0)])
        assert results2[0].step_id >= 2
        assert results2[1].step_id >= 2

        # Verify profiler on both racers
        assert track.racers[0].step_counter >= 2
        assert track.racers[1].step_counter >= 2

        # Verify command emits reached both mock clients
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
