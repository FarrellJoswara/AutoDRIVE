"""Regression checks for the performance benchmark's CPU accounting."""
import os

import pytest

from scripts.test_camera_fixed_interval import _proc_cpu_seconds


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="Linux /proc accounting")
def test_cpu_reader_measures_a_live_process():
    value = _proc_cpu_seconds(os.getpid())
    assert isinstance(value, float)
    assert value >= 0


def test_cpu_reader_handles_absent_process():
    assert _proc_cpu_seconds(None) is None
    assert _proc_cpu_seconds(999999999) is None
