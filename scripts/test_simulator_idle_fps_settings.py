"""Regression checks for CPU-lean fixed-step idle polling selection."""

import unittest

from src.layer4.settings import Settings


class SimulatorIdleFpsSettingsTests(unittest.TestCase):
    def test_legacy_simulator_keeps_idle_cap_unset(self):
        settings = Settings(simulator_mode="legacy", n_envs=16)
        self.assertEqual(settings.simulator_env(containerized=True)["AICAR_ACTION_IDLE_TARGET_FPS"], "")

    def test_fixed_camera_on_keeps_30_fps_idle_cap(self):
        settings = Settings(simulator_mode="fixed_camera_on", n_envs=16)
        env = settings.simulator_env(containerized=True)
        self.assertEqual(env["AICAR_ACTION_IDLE_TARGET_FPS"], "30")
        self.assertEqual(env["AICAR_DISABLE_CAMERA_STREAM"], "0")

    def test_camera_off_uses_10_fps_only_for_eight_or_more_training_envs(self):
        for count, expected in ((1, "30"), (6, "30"), (7, "30"), (8, "10"), (16, "10")):
            with self.subTest(n_envs=count):
                env = Settings(
                    simulator_mode="fixed_camera_off", n_envs=count,
                ).simulator_env(containerized=True)
                self.assertEqual(env["AICAR_ACTION_IDLE_TARGET_FPS"], expected)
                self.assertAlmostEqual(float(env["AICAR_ACTION_INTERVAL_SECONDS"]), 0.086)
                self.assertEqual(env["AICAR_DISABLE_CAMERA_STREAM"], "1")


if __name__ == "__main__":
    unittest.main()
