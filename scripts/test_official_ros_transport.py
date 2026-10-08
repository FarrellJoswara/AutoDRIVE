"""Deterministic lifecycle and freshness tests for official ROS transport."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from src.layer1.ros2_racer import OfficialRosTransportError, RacerRos2, _ros_message_timestamp


def _required_messages():
    return {
        "imu": SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        ),
        "throttle": SimpleNamespace(data=0.0),
        "steering": SimpleNamespace(data=0.0),
        "left_encoder": SimpleNamespace(position=[0.0]),
        "right_encoder": SimpleNamespace(position=[0.0]),
    }


def _scan():
    return SimpleNamespace(
        ranges=[2.0] * 1081,
        scan_time=0.025,
        range_min=0.05,
        range_max=10.0,
    )


def _stamped_scan(seconds):
    message = _scan()
    message.header = SimpleNamespace(stamp=SimpleNamespace(sec=seconds, nanosec=0))
    return message


def _prime(racer: RacerRos2):
    for field, message in _required_messages().items():
        racer._on_message(field, message)


class _FakeRclpy:
    def __init__(self, exception=None):
        self.exception = exception

    def ok(self):
        return True

    def spin_once(self, node, timeout_sec):
        if self.exception:
            raise self.exception
        time.sleep(min(timeout_sec, 0.005))


class OfficialRosTransportTests(unittest.TestCase):
    def make_racer(self, *, timeout_s=1.0, frame_timeout_s=0.05):
        return RacerRos2(node=object(), timeout_s=timeout_s, frame_timeout_s=frame_timeout_s)

    def test_missing_scan_times_out_with_topic_level_diagnostic(self):
        racer = self.make_racer(frame_timeout_s=0.02)
        started = time.monotonic()
        with self.assertRaisesRegex(OfficialRosTransportError, "never received: scan"):
            racer._wait_for_scan_after(0)
        self.assertLess(time.monotonic() - started, 0.2)

    def test_ros_header_timestamp_is_parsed_and_invalid_or_missing_stamp_falls_back(self):
        self.assertEqual(_ros_message_timestamp(_stamped_scan(12)), 12.0)
        self.assertIsNone(_ros_message_timestamp(_scan()))
        self.assertIsNone(_ros_message_timestamp(_stamped_scan(0)))

    def test_sensor_elapsed_uses_ros_clock_when_header_timestamps_advance(self):
        racer = self.make_racer()
        _prime(racer)
        racer._on_message("scan", _stamped_scan(10))
        racer._on_message("scan", _stamped_scan(10.04))
        self.assertAlmostEqual(racer.last_step_duration_s, 0.04)
        self.assertEqual(racer.scan_interval_source, "ros_sensor_stamp")
        self.assertAlmostEqual(racer.simulation_time(), 0.04)

    def test_duplicate_ros_timestamp_does_not_add_receipt_time_to_sim_clock(self):
        racer = self.make_racer()
        _prime(racer)
        racer._on_message("scan", _stamped_scan(10))
        time.sleep(0.01)
        racer._on_message("scan", _stamped_scan(10))
        self.assertEqual(racer.last_step_duration_s, 0.0)
        self.assertEqual(racer.scan_interval_source, "ros_sensor_stamp")
        self.assertEqual(racer.simulation_time(), 0.0)

    def test_sensor_elapsed_falls_back_to_monotonic_receipt_clock(self):
        racer = self.make_racer()
        _prime(racer)
        racer._on_message("scan", _scan())
        time.sleep(0.01)
        racer._on_message("scan", _scan())
        self.assertGreater(racer.last_step_duration_s, 0.005)
        self.assertEqual(racer.scan_interval_source, "receipt_clock")

    def test_startup_readiness_uses_startup_timeout_not_step_timeout(self):
        racer = self.make_racer(timeout_s=0.2, frame_timeout_s=0.02)
        _prime(racer)

        def deliver_startup_scan():
            time.sleep(0.08)
            _prime(racer)
            racer._on_message("scan", _scan())

        publisher = threading.Thread(target=deliver_startup_scan)
        publisher.start()
        started = time.monotonic()
        snapshot = racer.wait_until_ready()
        elapsed = time.monotonic() - started
        publisher.join(timeout=0.2)
        self.assertTrue(snapshot.lidar_valid)
        self.assertGreater(elapsed, racer.frame_timeout_s)

    def test_observation_requires_recent_component_topics(self):
        racer = self.make_racer(frame_timeout_s=0.02)
        _prime(racer)
        racer._raw_receipts["imu"] -= 1.0
        racer._on_message("scan", _scan())
        self.assertEqual(racer.step_counter, 0)
        with self.assertRaisesRegex(OfficialRosTransportError, "stale: .*imu="):
            racer.wait_until_ready()

    def test_transport_recovers_when_all_required_topics_and_scan_arrive(self):
        racer = self.make_racer()
        _prime(racer)
        racer._on_message("scan", _scan())
        snapshot = racer.wait_until_ready()
        self.assertEqual(racer.step_counter, 1)
        self.assertTrue(snapshot.lidar_valid)
        status = racer.transport_status()
        self.assertEqual(status["step_counter"], 1)
        self.assertLess(status["last_scan_age_s"], 0.1)
        self.assertTrue(all(age is not None for age in status["topic_age_s"].values()))

    def test_empty_encoder_uses_zero_speed_fallback_without_blocking_scan(self):
        racer = self.make_racer()
        racer._encoder_speed_mps = 4.2
        _prime(racer)
        racer._on_message("left_encoder", SimpleNamespace(position=[]))
        racer._on_message("scan", _scan())
        snapshot = racer.wait_until_ready()
        self.assertEqual(racer.step_counter, 1)
        self.assertTrue(snapshot.lidar_valid)
        self.assertEqual(racer._encoder_speed_mps, 0.0)
        self.assertIn("left_encoder", racer.transport_status()["topic_errors"])

    def test_control_frame_obeys_scan_cadence_not_duplicate_callback_rate(self):
        racer = self.make_racer(frame_timeout_s=1.0)
        _prime(racer)
        now = time.monotonic()
        racer._telemetry = SimpleNamespace(lidar_scan_rate=20.0)
        racer._last_control_scan_receipt = now
        result = []
        waiter = threading.Thread(target=lambda: result.append(
            racer._wait_for_control_frame(
                0, command_time=now, earliest_time=now + 0.06
            )
        ))
        waiter.start()
        time.sleep(0.01)
        racer._on_message("scan", _scan())
        time.sleep(0.02)
        self.assertTrue(waiter.is_alive(), "pre-cadence scan must not complete the action step")
        time.sleep(0.05)
        _prime(racer)
        racer._on_message("scan", _scan())
        waiter.join(timeout=0.3)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(result), 1)

    def test_shutdown_during_spin_wakes_waiter_without_thread_traceback(self):
        racer = self.make_racer(frame_timeout_s=1.0)
        racer._rclpy = _FakeRclpy(type("ExternalShutdownException", (Exception,), {})())
        racer._thread = threading.Thread(target=racer._spin, daemon=True)
        racer._thread.start()
        with self.assertRaisesRegex(OfficialRosTransportError, "ROS receive thread stopped"):
            racer.wait_until_ready()
        self.assertEqual(racer.transport_status()["spin_state"], "failed")

    def test_close_cancels_wait_immediately(self):
        racer = self.make_racer(frame_timeout_s=1.0)
        errors = []
        waiter = threading.Thread(target=lambda: self._capture_wait_error(racer, errors))
        waiter.start()
        time.sleep(0.01)
        racer.kill()
        waiter.join(timeout=0.2)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("closed", str(errors[0]))

    @staticmethod
    def _capture_wait_error(racer, errors):
        try:
            racer.wait_until_ready()
        except Exception as exc:  # captured for assertions in the owning thread
            errors.append(exc)


if __name__ == "__main__":
    unittest.main()
