from __future__ import annotations

import unittest

from src.layer4.official_settings import OfficialTrainSettings


class OfficialTrainSettingsTests(unittest.TestCase):
    def test_official_only_fields_map_to_trainer_container_environment(self):
        settings = OfficialTrainSettings(
            total_timesteps=2048,
            seed=8,
            resume="/runs/champion.zip",
            learning_rate=5e-5,
            n_steps=1024,
            n_epochs=2,
        )
        env = settings.to_container_env(hub_url="http://brain:8090", run_id="abc123")
        self.assertEqual(env["AICAR_MODE"], "train")
        self.assertEqual(env["AICAR_TRAIN_TIMESTEPS"], "2048")
        self.assertEqual(env["AICAR_PPO_LEARNING_RATE"], "5e-05")
        self.assertEqual(env["AICAR_PPO_N_STEPS"], "1024")
        self.assertEqual(env["AICAR_TRAIN_RESUME"], "/runs/champion.zip")
        self.assertEqual(env["AICAR_RUN_ID"], "abc123")
        self.assertEqual(env["AICAR_TRAIN_OUT"], "/runs/rl/official_run_abc123")
        self.assertNotIn("PORT", env)
        self.assertNotIn("AICAR_MAP_ID", env)

    def test_competition_settings_reject_custom_runtime_fields(self):
        with self.assertRaises(ValueError):
            OfficialTrainSettings.model_validate({"n_envs": 4, "frame_skip": 4})


if __name__ == "__main__":
    unittest.main()
