import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.layer1.ros2_racer import OfficialTrainingResetCancelled, RacerRos2
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
