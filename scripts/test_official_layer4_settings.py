from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.layer4.official_settings import OfficialTrainSettings, load_official_train_settings


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
        self.assertEqual(env["AICAR_TRAIN_TIME_COST"], "5.0")
        self.assertEqual(env["AICAR_TRAIN_FRONTIER_STAGNATION_S"], "10.0")
        self.assertEqual(env["AICAR_TRAIN_COLLISION_PENALTY_MAGNITUDE"], "100.0")
        self.assertEqual(env["AICAR_TRAIN_COLLISION_REWARD_PERCENT"], "50.0")
        self.assertEqual(env["AICAR_TRAIN_BACKWARD_SPEED_PENALTY_SCALE"], "10.0")
        self.assertEqual(env["AICAR_TRAIN_FAILURE_REWARD_PERCENT"], "50.0")
        self.assertNotIn("AICAR_TRAIN_LAP_REWARD", env)
        self.assertNotIn("PORT", env)
        self.assertNotIn("AICAR_MAP_ID", env)

    def test_competition_settings_reject_custom_runtime_fields(self):
        with self.assertRaises(ValueError):
            OfficialTrainSettings.model_validate({"n_envs": 4, "frame_skip": 4})

    def test_legacy_reward_settings_migrate_to_frontier_objective(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({
                "n_envs": 4,
                "time_cost_per_simulated_second": 1.0,
                "lap_completion_reward": 100.0,
                "collision_penalty_base": 10.0,
                "failed_episode_penalty": 1000.0,
            }), encoding="utf-8")
            settings = load_official_train_settings(path)
        self.assertEqual(settings.n_envs, 4)
        self.assertEqual(settings.time_cost_per_simulated_second, 5.0)
        self.assertEqual(settings.collision_penalty_magnitude, 100.0)
        self.assertEqual(settings.collision_reward_percent, 50.0)
        self.assertEqual(settings.failed_episode_penalty, 100.0)
        self.assertEqual(settings.warmup_laps, 0)


if __name__ == "__main__":
    unittest.main()
