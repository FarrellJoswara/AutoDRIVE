"""Replay simulators must use the selected run's simulator configuration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.layer3.envs import env_kwargs_from_args
from src.layer3.play import build_arg_parser as build_play_arg_parser
from src.layer3.play import env_kwargs_from_play_args
from src.layer3.train import build_arg_parser
from src.layer4.hub import docker_control
from src.layer4.settings import Settings


class ReplaySettingsTests(unittest.TestCase):
    def test_play_applies_replay_runtime_settings_over_parser_defaults(self) -> None:
        args = build_play_arg_parser().parse_args([
            "--model", "policy.zip",
            "--env-kwargs-json",
            '{"frame_skip":1,"action_interval_s":0.025,"observation_profile":"simulator"}',
        ])

        kwargs = env_kwargs_from_play_args(args)

        self.assertEqual(kwargs["frame_skip"], 1)
        self.assertEqual(kwargs["action_interval_s"], 0.025)
        self.assertEqual(kwargs["observation_profile"], "simulator")

    def test_replay_env_kwargs_match_layer3_training_settings(self) -> None:
        settings = Settings(
            auto_launch=False,
            simulator_mode="fixed_camera_on",
            action_interval_s=0.025,
            frame_skip=1,
            map_id="porto",
            throttle_mode="bidirectional",
            observation_profile="simulator",
            steering_action_scale=0.946,
            straight_throttle_gain=1.05,
            collision_reward_percent=50,
        )
        train_args = build_arg_parser().parse_args(settings.to_train_argv())

        self.assertEqual(
            settings.to_env_kwargs(headless=True, auto_launch=False),
            env_kwargs_from_args(train_args),
        )

    def test_replay_container_replaces_inherited_simulator_mode_values(self) -> None:
        template = {
            "Config": {
                "Env": [
                    "BASE_PORT=4567",
                    "PORT=4567",
                    "AICAR_EXPECTED_SIMS=1",
                    "AICAR_MAP_ID=none",
                    "AICAR_ACTION_INTERVAL_SECONDS=0.05",
                    "AICAR_SIMULATOR_PATH=/app/simulator/old-player",
                    "AICAR_DISABLE_CAMERA_STREAM=0",
                ],
                "Labels": {"com.docker.compose.project": "aicar"},
                "Image": "aicar-sim",
            },
            "HostConfig": {"NetworkMode": "aicar_aicar"},
            "NetworkSettings": {"Networks": {"aicar_aicar": {}}},
        }
        captured = {}

        def request(method, path, **kwargs):
            if method == "GET":
                return template
            if "/containers/create?" in path:
                captured.update(kwargs["body"])
                return {"Id": "replay-container"}
            return None

        with (
            patch.object(docker_control, "_list_compose_containers", side_effect=[[{"Id": "sim-template"}], []]),
            patch.object(docker_control, "_request", side_effect=request),
        ):
            docker_control.create_replay_sim(
                "porto",
                port=4583,
                env_overrides={
                    "AICAR_ACTION_INTERVAL_SECONDS": "0.025",
                    "AICAR_SIMULATOR_PATH": "/app/simulator/fixed-player",
                    "AICAR_DISABLE_CAMERA_STREAM": "0",
                },
            )

        env = captured["Env"]
        self.assertEqual(env.count("AICAR_ACTION_INTERVAL_SECONDS=0.025"), 1)
        self.assertEqual(env.count("AICAR_SIMULATOR_PATH=/app/simulator/fixed-player"), 1)
        self.assertFalse(any(value == "AICAR_ACTION_INTERVAL_SECONDS=0.05" for value in env))
        self.assertIn("PORT=4583", env)
        self.assertIn("BASE_PORT=4583", env)
        self.assertIn("AICAR_MAP_ID=porto", env)

    def test_replay_container_rejects_network_setting_overrides(self) -> None:
        with patch.object(
            docker_control, "_list_compose_containers", return_value=[{"Id": "sim-template"}]
        ), patch.object(docker_control, "_request", return_value={
            "Config": {"Env": [], "Labels": {}},
            "HostConfig": {},
            "NetworkSettings": {"Networks": {}},
        }):
            with self.assertRaisesRegex(ValueError, "reserved network settings"):
                docker_control.create_replay_sim(
                    "porto", env_overrides={"PORT": "4999"}
                )


if __name__ == "__main__":
    unittest.main()
