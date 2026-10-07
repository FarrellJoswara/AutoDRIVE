"""Tests that offline teacher inputs stay outside sensor-only observations."""

from __future__ import annotations

import unittest

import numpy as np

from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.autodrive_env import AutoDriveEnv


class TeacherObservationTests(unittest.TestCase):
    def test_teacher_observation_is_separate_from_official_sensor_profile(self) -> None:
        env = AutoDriveEnv.__new__(AutoDriveEnv)
        snap = TelemetrySnapshot(v_long=2.0, v_lat=1.0)
        env._last_snap = snap
        env._prev_throttle = 0.3
        env._prev_steering = -0.2
        env._observation_encoder_positions = (snap.encoder_left, snap.encoder_right)
        env.observation_profile = "official_sensors"

        teacher_observation = env.get_privileged_teacher_observation()
        student_observation = env._policy_observation(
            snap,
            env._prev_throttle,
            env._prev_steering,
            lidar_beams=1081,
            elapsed_s=0.025,
        )

        self.assertAlmostEqual(float(teacher_observation["state"][1]), 1.0 / 22.88)
        self.assertEqual(float(student_observation["state"][1]), 0.0)
        self.assertAlmostEqual(float(teacher_observation["state"][7]), 0.3)
        self.assertAlmostEqual(float(student_observation["state"][7]), 0.3)
        self.assertTrue(np.array_equal(teacher_observation["lidar"], student_observation["lidar"]))


if __name__ == "__main__":
    unittest.main()
