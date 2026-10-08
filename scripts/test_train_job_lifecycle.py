"""Regression tests for train-process watcher races."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock

from src.layer4.hub.train_job import TrainJob


class FakeProcess:
    def __init__(self, code: int = 0) -> None:
        self.code = code

    def wait(self) -> int:
        return self.code


class TrainJobLifecycleTests(unittest.TestCase):
    def test_stale_watcher_cannot_overwrite_a_new_run(self) -> None:
        job = TrainJob()
        old_proc = FakeProcess()
        new_proc = FakeProcess()
        settings = Mock()
        log_file = Mock()
        job._proc = new_proc
        job.state = "running"
        job.pid = 456

        job._watch_exit(old_proc, settings, log_file)

        self.assertIs(job._proc, new_proc)
        self.assertEqual(job.state, "running")
        self.assertEqual(job.pid, 456)
        log_file.close.assert_not_called()

    def test_run_stays_stopping_until_simulator_cleanup_finishes(self) -> None:
        job = TrainJob()
        proc = FakeProcess(code=1)
        settings = Mock()
        log_file = Mock()
        entered_cleanup = threading.Event()
        release_cleanup = threading.Event()
        job._proc = proc
        job.state = "stopping"
        job.pid = 123
        job._docker_cleanup_enabled = lambda _: True
        job._cleanup_docker_services = lambda _: (
            entered_cleanup.set(), release_cleanup.wait(timeout=2)
        )

        watcher = threading.Thread(target=job._watch_exit, args=(proc, settings, log_file))
        watcher.start()
        self.assertTrue(entered_cleanup.wait(timeout=1))
        self.assertEqual(job.status()["state"], "stopping")
        self.assertIs(job._proc, proc)

        release_cleanup.set()
        watcher.join(timeout=1)
        self.assertFalse(watcher.is_alive())
        self.assertEqual(job.status()["state"], "exited")
        self.assertIsNone(job._proc)
        self.assertIsNone(job.pid)


if __name__ == "__main__":
    unittest.main()
