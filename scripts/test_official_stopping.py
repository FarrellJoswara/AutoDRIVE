from __future__ import annotations

import unittest

from src.layer3.official_train import EpisodeReturnPlateau
from src.layer4.official_settings import OfficialTrainSettings


class EpisodeReturnPlateauTests(unittest.TestCase):
    def test_waits_for_minimum_steps_and_full_window_then_plateaus(self):
        stop = EpisodeReturnPlateau(
            window_episodes=2,
            patience=2,
            min_timesteps=100,
            min_improvement_pct=1.0,
        )
        self.assertFalse(stop.observe(10, 100))
        self.assertFalse(stop.observe(10, 99))  # full window before the threshold is ignored
        self.assertFalse(stop.observe(10, 100))
        self.assertFalse(stop.observe(10, 100))  # first qualifying window is baseline
        self.assertFalse(stop.observe(10, 100))
        self.assertFalse(stop.observe(10, 100))
        self.assertFalse(stop.observe(10, 100))
        self.assertTrue(stop.observe(10, 100))

    def test_meaningful_gain_resets_stale_window_count(self):
        stop = EpisodeReturnPlateau(
            window_episodes=2,
            patience=2,
            min_timesteps=0,
            min_improvement_pct=1.0,
        )
        for value in (10, 10, 10, 10):
            self.assertFalse(stop.observe(value, 100))
        self.assertEqual(stop.stale_windows, 1)
        self.assertFalse(stop.observe(12, 100))
        self.assertFalse(stop.observe(12, 100))
        self.assertEqual(stop.stale_windows, 0)
        self.assertFalse(stop.observe(12, 100))
        self.assertFalse(stop.observe(12, 100))
        self.assertFalse(stop.observe(12, 100))
        self.assertTrue(stop.observe(12, 100))


class OfficialStopSettingsTests(unittest.TestCase):
    def test_zero_timestep_budget_maps_to_plateau_mode_controls(self):
        settings = OfficialTrainSettings(
            total_timesteps=0,
            stop_on_plateau=True,
            plateau_min_timesteps=0,
            plateau_window_episodes=4,
            plateau_patience=3,
            plateau_min_improvement_pct=0.5,
            max_duration_hours=12,
        )
        env = settings.to_container_env(hub_url="http://brain", run_id="abc")
        self.assertEqual(env["AICAR_TRAIN_TIMESTEPS"], "0")
        self.assertEqual(env["AICAR_TRAIN_STOP_ON_PLATEAU"], "1")
        self.assertEqual(env["AICAR_TRAIN_PLATEAU_WINDOW_EPISODES"], "4")
        self.assertEqual(env["AICAR_TRAIN_PLATEAU_PATIENCE"], "3")
        self.assertEqual(env["AICAR_TRAIN_MAX_DURATION_HOURS"], "12.0")


if __name__ == "__main__":
    unittest.main()
