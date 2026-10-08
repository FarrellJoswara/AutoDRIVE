"""Layer 2 surfaces official transport failures without reusing observations."""

from __future__ import annotations

import unittest

import numpy as np

from src.layer1.ros2_racer import OfficialRosTransportError
from src.layer2.official_race_env import OfficialRaceEnv


class _FailingRacer:
    last_step_duration_s = 0.0
    last_control_interval_s = 0.0

    def step(self, throttle, steering):
        raise OfficialRosTransportError(
            "Timed out waiting for a fresh official ROS sensor frame; stale: lidar=5.01s"
        )


class OfficialEnvTransportTests(unittest.TestCase):
    def test_step_fails_actionably_on_transport_loss(self):
        env = OfficialRaceEnv(racer=_FailingRacer(), timeout_s=180.0)
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "OfficialRaceEnv.step cannot continue.*fresh observation.*stale: lidar=5.01s",
            ):
                env.step(np.zeros(2, dtype=np.float32))
        finally:
            env.close()

    def test_frame_timeout_is_bounded_by_race_timeout(self):
        env = OfficialRaceEnv(racer=_FailingRacer(), timeout_s=180.0)
        self.assertEqual(env.frame_timeout_s, 5.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
