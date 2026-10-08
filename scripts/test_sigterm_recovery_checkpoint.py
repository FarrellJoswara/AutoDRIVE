"""Regression tests for preserving PPO state on a cooperative SIGTERM stop."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.layer3.train import RequestedStopCallback


class RequestedStopCheckpointTests(unittest.TestCase):
    def test_stop_saves_recovery_policy_before_ending_learn(self) -> None:
        requested = threading.Event()
        requested.set()
        target = Path("run/ppo_signal_recovery.zip")
        callback = RequestedStopCallback(requested, target)
        callback.model = Mock()

        with patch.object(Path, "mkdir"):
            keep_training = callback._on_step()

        self.assertFalse(keep_training)
        self.assertEqual(callback.stop_reason, "operator_stop")
        callback.model.save.assert_called_once_with(str(target.with_suffix("")))

    def test_normal_step_does_not_write_a_recovery_checkpoint(self) -> None:
        callback = RequestedStopCallback(threading.Event(), Path("run/recovery.zip"))
        callback.model = Mock()

        self.assertTrue(callback._on_step())
        callback.model.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
