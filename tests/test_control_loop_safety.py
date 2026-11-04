import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.layer1.ros2_racer import (
    OfficialRosTransportError,
    OfficialTrainingResetCancelled,
    RaceMetrics,
    RacerRos2,
)
from src.layer3.preview_writer import PreviewWriter
from src.layer3.official_train import _AsyncProgress


def test_stalled_policy_releases_throttle_once_without_changing_steering():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._publishers = {"throttle": Mock(), "steering": Mock()}
    racer._msg_types = {"Float32": SimpleNamespace}
    racer._last_action_time = 10.0
    racer._command_expired = False
    racer.command_expiry_count = 0
    with patch("src.layer1.ros2_racer.time.monotonic", return_value=10.2):
        racer._expire_stale_command()
    racer._publishers["throttle"].publish.assert_not_called()
    with patch("src.layer1.ros2_racer.time.monotonic", return_value=10.3):
        racer._expire_stale_command()
        racer._expire_stale_command()
    assert racer._publishers["throttle"].publish.call_count == 1
    assert racer._publishers["throttle"].publish.call_args.args[0].data == 0.0
    racer._publishers["steering"].publish.assert_not_called()
    assert racer.command_expiry_count == 1
    diagnostic = racer._command_expiry_diagnostic
    assert diagnostic["action_age_s"] == pytest.approx(0.3)
    assert diagnostic["step_counter"] == 0
    assert diagnostic["last_scan_age_s"] is None
    assert diagnostic["topic_age_s"] == {}
    assert diagnostic["last_spin_iteration_s"] == 0.0


def test_opt_in_neutral_fallback_sends_only_zero_commands_once_before_lease_expiry():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._publishers = {"throttle": Mock(), "steering": Mock()}
    racer._msg_types = {"Float32": SimpleNamespace}
    racer._last_action_time = 10.0
    racer._last_policy_action = {"throttle": 0.3, "steering": -0.2}
    racer.safe_neutral_fallback_s = 0.20
    racer._command_expired = False
    racer.command_expiry_count = 0
    racer.neutral_fallback_count = 0
    racer._neutral_fallback_diagnostic = None
    racer._raw = {"throttle": SimpleNamespace(data=0.3), "steering": SimpleNamespace(data=-0.2)}
    racer._raw_receipts = {"throttle": 10.1, "steering": 10.1}
    racer._last_scan_receipt = 10.1
    racer._step_counter = 9

    with patch("src.layer1.ros2_racer.time.monotonic", return_value=10.21):
        racer._expire_stale_command()
        racer._expire_stale_command()

    for actuator in ("throttle", "steering"):
        publisher = racer._publishers[actuator]
        assert publisher.publish.call_count == 1
        assert publisher.publish.call_args.args[0].data == 0.0
    assert racer.neutral_fallback_count == 1
    assert racer.command_expiry_count == 0
    diagnostic = racer._neutral_fallback_diagnostic
    assert diagnostic["action_age_s"] == pytest.approx(0.21)
    assert diagnostic["threshold_s"] == 0.20
    assert diagnostic["reason"] == "policy_action_waiting_for_fresh_sensor_frame"
    assert diagnostic["previous_policy_action"] == {"throttle": 0.3, "steering": -0.2}
    assert diagnostic["active_action"] == {"throttle": 0.0, "steering": 0.0}


def test_fallback_neutralizes_old_action_but_missed_fallback_keeps_hard_expiry():
    def make_racer():
        racer = RacerRos2.__new__(RacerRos2)
        racer._condition = threading.Condition()
        racer._publishers = {"throttle": Mock(), "steering": Mock()}
        racer._msg_types = {"Float32": SimpleNamespace}
        racer._last_action_time = 20.0
        racer._last_policy_action = {"throttle": 0.4, "steering": 0.1}
        racer.safe_neutral_fallback_s = 0.20
        racer._command_expired = False
        racer.command_expiry_count = 0
        racer.neutral_fallback_count = 0
        racer._raw = {}
        racer._raw_receipts = {}
        racer._last_scan_receipt = 20.0
        racer._step_counter = 1
        return racer

    neutralized = make_racer()
    with patch("src.layer1.ros2_racer.time.monotonic", return_value=20.22):
        neutralized._expire_stale_command()
    with patch("src.layer1.ros2_racer.time.monotonic", return_value=20.40):
        neutralized._expire_stale_command()
    assert neutralized.neutral_fallback_count == 1
    assert neutralized.command_expiry_count == 0
    assert neutralized._publishers["throttle"].publish.call_count == 1
    assert neutralized._publishers["steering"].publish.call_count == 1

    missed = make_racer()
    with patch("src.layer1.ros2_racer.time.monotonic", return_value=20.26):
        missed._expire_stale_command()
    assert missed.neutral_fallback_count == 0
    assert missed.command_expiry_count == 1
    assert missed._publishers["throttle"].publish.call_args.args[0].data == 0.0
    assert missed._publishers["steering"].publish.call_args.args[0].data == 0.0
    assert missed._command_expiry_diagnostic["reason"] == (
        "hard_command_lease_exceeded_before_neutral_fallback"
    )


def test_missing_fresh_frame_still_times_out_after_safe_neutral_fallback():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._closed = False
    racer._spin_error = None
    racer._spin_state = "running"
    racer._topic_errors = {}
    racer._raw_receipts = {}
    racer.require_camera = False
    racer._step_counter = 0
    racer._last_scan_receipt = 0.0
    racer.frame_timeout_s = 0.02
    racer._publishers = {"throttle": Mock(), "steering": Mock()}
    racer._msg_types = {"Float32": SimpleNamespace}
    racer._last_action_time = time.monotonic() - 0.21
    racer._last_policy_action = {"throttle": 0.3, "steering": 0.1}
    racer.safe_neutral_fallback_s = 0.20
    racer._command_expired = False
    racer.command_expiry_count = 0
    racer.neutral_fallback_count = 0
    racer._raw = {}
    racer._neutral_fallback_diagnostic = None
    racer._expire_stale_command()
    assert racer.neutral_fallback_count == 1
    with pytest.raises(OfficialRosTransportError, match="Timed out.*official LiDAR frame"):
        racer._wait_for_control_frame(0, command_time=0.0, earliest_time=0.0)


def test_training_reset_metric_baseline_uses_fresh_post_reset_counter_epoch():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._closed = False
    racer._spin_error = None
    racer.frame_timeout_s = 0.5
    racer._race_metric_update_counts = {name: 8 for name in (
        "lap_count", "lap_time", "last_lap_time", "best_lap_time", "collision_count"
    )}
    racer._race_metrics = RaceMetrics(
        lap_count=1, lap_time=118.0, last_lap_time=117.0,
        best_lap_time=117.0, collision_count=0,
    )
    before_reset = dict(racer._race_metric_update_counts)
    racer._check_wait_state = lambda: None
    racer._raw = {}
    racer._raw_receipts = {}
    racer._race_metrics_received = set()
    racer._transport_diagnostic = lambda *args, **kwargs: "test transport"

    def publish_fresh_metrics():
        time.sleep(0.01)
        for field, value in (
            ("lap_count", 0), ("lap_time", 0.0), ("last_lap_time", 0.0),
            ("best_lap_time", 0.0), ("collision_count", 0), ("lap_count", 0),
        ):
            racer._on_message(field, SimpleNamespace(data=value))
            time.sleep(0.005)

    publisher = threading.Thread(target=publish_fresh_metrics)
    publisher.start()
    metrics = racer._wait_for_fresh_reset_metrics(before_reset)
    publisher.join()

    assert metrics.lap_count == 0
    assert racer._race_metric_update_counts["lap_count"] == before_reset["lap_count"] + 2


def test_training_reset_metrics_fail_closed_without_fresh_epoch():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._closed = False
    racer._spin_error = None
    racer.frame_timeout_s = 0.01
    racer._race_metric_update_counts = {name: 4 for name in (
        "lap_count", "lap_time", "last_lap_time", "best_lap_time", "collision_count"
    )}
    racer._race_metrics = RaceMetrics(
        lap_count=1, lap_time=118.0, last_lap_time=117.0,
        best_lap_time=117.0, collision_count=0,
    )
    racer._check_wait_state = lambda: None
    racer._raw = {}
    racer._raw_receipts = {}
    racer._race_metrics_received = set()
    racer._transport_diagnostic = lambda *args, **kwargs: "test transport"
    with pytest.raises(OfficialRosTransportError, match="fresh, stable official race metrics"):
        racer._wait_for_fresh_reset_metrics(dict(racer._race_metric_update_counts))


def test_cooperative_pause_publishes_neutral_and_reports_transport_ack_limit():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._publishers = {"throttle": Mock(), "steering": Mock()}
    racer._msg_types = {"Float32": SimpleNamespace}
    racer._closed = False
    racer._spin_error = None
    racer._spin_state = "running"
    racer._raw_receipts = {}
    racer._last_scan_receipt = 0.0
    racer._step_counter = 17
    racer._topic_errors = {}
    racer._last_action_time = 0.0
    racer._command_expired = False
    racer.command_expiry_count = 0
    racer.frame_timeout_s = 5.0
    racer.require_camera = True
    result = racer.request_pause_neutral()
    assert racer._publishers["throttle"].publish.call_args.args[0].data == 0.0
    assert racer._publishers["steering"].publish.call_args.args[0].data == 0.0
    assert result["published"] is True
    assert result["transport_acknowledged"] is False
    assert result["step_counter"] == 17
    assert result["topic_fresh"] is False
    assert set(result["topic_age_s"]) == {
        "scan", "imu", "throttle", "steering", "left_encoder", "right_encoder", "camera"
    }
    assert racer.command_expiry_count == 1


def test_kill_publishes_both_terminal_zero_commands_once_and_is_idempotent():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._closed = False
    racer._cleanup_complete = False
    racer._publishers = {"throttle": Mock(), "steering": Mock()}
    racer._msg_types = {"Float32": SimpleNamespace}
    racer._thread = None
    racer._owned_node = False
    racer._last_policy_action = {"throttle": 0.4, "steering": -0.3}
    racer._last_action_time = 12.0
    racer._command_expired = False

    racer.kill()
    racer.kill()

    for name in ("throttle", "steering"):
        publisher = racer._publishers[name]
        assert publisher.publish.call_count == 1
        assert publisher.publish.call_args.args[0].data == 0.0
    assert racer._closed is True
    assert racer._cleanup_complete is True
    assert racer._last_policy_action == {"throttle": 0.0, "steering": 0.0}


def test_kill_waits_for_inflight_policy_publish_then_leaves_actuators_neutral():
    racer = RacerRos2.__new__(RacerRos2)
    racer._condition = threading.Condition()
    racer._closed = False
    racer._cleanup_complete = False
    racer._owned_node = False
    racer._thread = None
    racer._publishers = {}
    racer._msg_types = {"Float32": SimpleNamespace}
    racer._last_action_time = 0.0
    racer._last_policy_action = {"throttle": 0.0, "steering": 0.0}
    racer._command_expired = False
    racer.command_expiry_count = 0
    racer.neutral_fallback_count = 0
    racer._step_counter = 0
    racer._last_control_source_time = None
    racer._last_control_scan_receipt = 0.0
    racer._last_scan_source_time = None
    racer._last_scan_receipt = 0.0
    racer._last_control_interval_s = 0.0
    racer._last_control_interval_source = "receipt_clock"
    racer._last_control_wall_interval_s = 0.0
    racer._last_step_duration_s = 0.0
    racer._telemetry = SimpleNamespace(lidar_scan_rate=40.0)
    racer._spin_error = None
    entered, release, killed = threading.Event(), threading.Event(), threading.Event()
    throttle_values, steering_values = [], []

    def publish_throttle(message):
        throttle_values.append(message.data)
        if len(throttle_values) == 1:
            entered.set()
            assert release.wait(1.0)

    racer._publishers = {
        "throttle": SimpleNamespace(publish=publish_throttle),
        "steering": SimpleNamespace(publish=lambda message: steering_values.append(message.data)),
    }
    racer._wait_for_control_frame = lambda *args, **kwargs: object()
    step_thread = threading.Thread(target=lambda: racer.step(0.35, -0.15))
    kill_thread = threading.Thread(target=lambda: (racer.kill(), killed.set()))
    step_thread.start()
    assert entered.wait(1.0)
    kill_thread.start()
    assert not killed.wait(0.03)
    release.set()
    step_thread.join(1.0)
    kill_thread.join(1.0)

    assert not step_thread.is_alive()
    assert not kill_thread.is_alive()
    assert throttle_values == [0.35, 0.0]
    assert steering_values == [-0.15, 0.0]
    assert racer._closed is True


def test_kill_is_safe_when_ros_initialization_did_not_finish():
    racer = RacerRos2.__new__(RacerRos2)
    racer.kill()
    racer.kill()
    assert racer._closed is True
    assert racer._cleanup_complete is True


def test_stop_during_training_reset_deasserts_reset_and_returns_promptly():
    racer = RacerRos2.__new__(RacerRos2)
    racer.allow_training_reset = True
    published = []
    racer._publishers = {
        "training_reset": SimpleNamespace(publish=lambda message: published.append(bool(message.data)))
    }
    racer._msg_types = {"Bool": lambda: SimpleNamespace(data=False)}
    racer.frame_timeout_s = 5.0
    racer.timeout_s = 5.0
    racer._condition = threading.Condition()
    racer._step_counter = 0
    racer._position_update_count = 0
    racer._closed = False
    racer._spin_error = None
    racer.wait_until_ready = lambda **kwargs: object()
    cancelled = threading.Event()
    timer = threading.Timer(0.04, cancelled.set)
    timer.start()
    try:
        with pytest.raises(OfficialTrainingResetCancelled):
            racer.reset_simulation_for_training(cancel_check=cancelled.is_set)
    finally:
        timer.cancel()
        timer.join(1.0)
    assert published == [True, False]


def test_slow_preview_does_not_block_control_or_queue_old_frames(tmp_path):
    writer = PreviewWriter(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def blocked_write(frames):
        entered.set()
        release.wait(2)
    writer._write = blocked_write
    try:
        assert writer.submit(np.zeros((1, 90, 160, 3), dtype=np.uint8))
        assert entered.wait(1)
        assert not writer.submit(np.zeros((1, 90, 160, 3), dtype=np.uint8))
    finally:
        release.set()
        writer._thread.join(2)


def test_preview_writes_real_rgb_image(tmp_path):
    from PIL import Image
    writer = PreviewWriter(tmp_path)
    assert writer.submit(np.full((1, 3, 90, 160), 125, dtype=np.uint8))
    writer._thread.join(2)
    with Image.open(tmp_path / "env-0.jpg") as image:
        assert image.size == (160, 90)
        assert np.asarray(image).mean() > 120


def test_blocked_hub_update_does_not_block_driving_or_queue_stale_updates():
    entered, release = threading.Event(), threading.Event()
    calls = []
    def callback(step):
        calls.append(step)
        entered.set()
        release.wait(2)
    with _AsyncProgress(callback) as publish:
        try:
            publish(1)
            assert entered.wait(1)
            publish(2)
            publish(3)
            assert calls == [1]
        finally:
            release.set()
    assert publish.pending.done()
