"""Smoke tests for initializing a sensor-only PPO actor from teacher labels."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from src.layer3.behavior_cloning import (
    assess_teacher_collection,
    behavior_clone_actor,
    collect_policy_demonstrations,
    collect_centerline_demonstrations,
    initialize_behavior_cloned_action_std,
    save_behavior_cloned_checkpoint,
)
from src.layer3.train import make_model
from src.layer3.official_train import _demonstration_progress_publisher
from src.layer2.spaces import LIDAR_BEAMS, STATE_DIM


class _SensorEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(0.0, 1.0, shape=(LIDAR_BEAMS,), dtype=np.float32),
            "state": gym.spaces.Box(-5.0, 5.0, shape=(STATE_DIM,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"lidar": np.ones(LIDAR_BEAMS, np.float32), "state": np.zeros(STATE_DIM, np.float32)}, {}

    def step(self, action):
        obs, info = self.reset()
        return obs, 0.0, False, False, info


class BehaviorCloningTests(unittest.TestCase):
    def test_teacher_trace_keeps_privileged_diagnostics_outside_training_archive(self) -> None:
        info = {
            "position": [1.0, 2.0, 0.0], "yaw": 0.5,
            "current_progress_m": 3.0, "true_speed": 0.2,
            "frontier_speed_mps": 0.1,
            "lidar_range_min": 0.06, "lidar_range_max": 10.0,
        }
        observations = {
            "lidar": np.ones((1, LIDAR_BEAMS), np.float32),
            "state": np.zeros((1, STATE_DIM), np.float32),
        }

        class Env:
            num_envs = 1

            def reset(self):
                return observations

            def env_method(self, name):
                return [info]

            def step(self, actions):
                return observations, np.zeros(1), np.zeros(1, dtype=bool), [info]

        class Expert:
            def action_batch(self, infos, obs):
                return np.asarray([[0.1, -0.2]], np.float32)

        with TemporaryDirectory() as directory:
            archive = Path(directory) / "demonstrations.npz"
            phases = []
            _, summary = collect_centerline_demonstrations(
                Env(), Expert(), steps=2, output_path=archive,
                progress_callback=lambda *values: phases.append(values),
            )

            with np.load(archive) as saved:
                saved_files = set(saved.files)
            trace = [json.loads(line) for line in archive.with_name("teacher_trace.jsonl").read_text().splitlines()]

        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["position_xy_m"], [1.0, 2.0])
        self.assertEqual(summary["latest_teacher_poses"], [{
            "env_id": 0, "pose": [1.0, 2.0], "yaw": 0.5, "speed": 0.2,
        }])
        np.testing.assert_allclose(trace[0]["teacher_action"], [0.1, -0.2])
        self.assertNotIn("position", saved_files)
        self.assertNotIn("yaw", saved_files)
        self.assertEqual(phases[-1][3], [{
            "env_id": 0, "pose": [1.0, 2.0], "yaw": 0.5, "speed": 0.2,
            "route_progress_m": 3.0,
        }])

    def test_teacher_collection_gate_accepts_only_clean_collision_free_episode(self) -> None:
        clean = assess_teacher_collection({
            "completed_episodes": 1,
            "successful_episodes": 1,
            "collision_count": 0,
            "termination_counts": {"lap_complete": 1},
        })
        self.assertTrue(clean["ready"])
        self.assertIsNone(clean["reason"])

        collided = assess_teacher_collection({
            "completed_episodes": 1,
            "successful_episodes": 0,
            "collision_count": 1,
            "termination_counts": {"collision": 1},
        })
        self.assertFalse(collided["ready"])
        self.assertIn("teacher_collection_no_clean_episode", collided["reason"])
        self.assertIn("teacher_collection_collision_detected", collided["reason"])
        self.assertIn("teacher_collection_failed_episode", collided["reason"])

        stalled = assess_teacher_collection({
            "completed_episodes": 0,
            "successful_episodes": 0,
            "stagnation_detected": True,
        })
        self.assertIn("teacher_collection_stalled", stalled["reason"])

    def test_teacher_collection_gate_rejects_missing_clean_episode(self) -> None:
        result = assess_teacher_collection({
            "completed_episodes": 0,
            "successful_episodes": 0,
            "termination_counts": {},
        })
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "teacher_collection_no_clean_episode")

    def test_route_progress_publisher_uses_route_length_meters(self) -> None:
        phases = []
        route = type("Route", (), {"length_m": 31.1})()
        publish = _demonstration_progress_publisher(
            route, lambda phase, **details: phases.append((phase, details))
        )

        teacher_poses = [{"env_id": 0, "pose": [1.25, 2.5], "yaw": 0.5, "speed": 0.2}]
        publish(4, 10, [2.5], teacher_poses)

        self.assertEqual(phases, [(
            "demonstration_collection",
            {
                "completed_steps_per_env": 4,
                "target_steps_per_env": 10,
                "route_progress_m": [2.5],
                "route_length_m": 31.1,
                "teacher_poses": teacher_poses,
            },
        )])

    def test_cloned_policy_uses_safe_action_noise(self) -> None:
        policy = type("Policy", (), {})()
        policy.log_std = torch.nn.Parameter(torch.zeros(2))
        model = type("Model", (), {"policy": policy})()

        applied_std = initialize_behavior_cloned_action_std(model)

        self.assertAlmostEqual(applied_std, 0.2, places=6)
        self.assertTrue(torch.allclose(
            policy.log_std.exp(), torch.full((2,), 0.2), atol=1e-6
        ))

    def test_cloned_checkpoint_is_saved_with_replay_catalog_filename(self) -> None:
        class FakeModel:
            def save(self, path: str) -> None:
                Path(path + ".zip").write_bytes(b"policy")

        with TemporaryDirectory() as directory:
            checkpoint = save_behavior_cloned_checkpoint(FakeModel(), Path(directory))
            self.assertEqual(checkpoint.name, "behavior_cloned_policy.zip")
            self.assertEqual(checkpoint.read_bytes(), b"policy")

    def test_cloning_fits_sensor_observation_to_continuous_action_labels(self) -> None:
        env = DummyVecEnv([_SensorEnv])
        try:
            model = make_model(
                env, device="cpu", seed=12, tensorboard_log=None,
                policy_architecture="lidar_cnn",
            )
            rng = np.random.default_rng(12)
            count = 256
            states = rng.normal(size=(count, STATE_DIM)).astype(np.float32)
            lidar = rng.uniform(0.2, 1.0, size=(count, LIDAR_BEAMS)).astype(np.float32)
            actions = np.column_stack((
                np.tanh(states[:, 0] * 0.4),
                np.tanh(states[:, 1] * 0.4),
            )).astype(np.float32)
            summary = behavior_clone_actor(
                model,
                {"lidar": lidar, "state": states, "actions": actions},
                epochs=8,
                batch_size=64,
                seed=12,
            )
            self.assertEqual(summary["transitions"], count)
            self.assertLess(summary["final_validation_loss"], summary["initial_validation_loss"])
        finally:
            env.close()

    def test_camera_enabled_policy_accepts_uint8_demonstrations_during_cloning(self) -> None:
        class CameraEnv(_SensorEnv):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Dict({
                    "lidar": gym.spaces.Box(0.0, 1.0, shape=(LIDAR_BEAMS,), dtype=np.float32),
                    "state": gym.spaces.Box(-5.0, 5.0, shape=(STATE_DIM,), dtype=np.float32),
                    "camera": gym.spaces.Box(0, 255, shape=(90, 160, 3), dtype=np.uint8),
                })

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return {
                    "lidar": np.ones(LIDAR_BEAMS, np.float32),
                    "state": np.zeros(STATE_DIM, np.float32),
                    "camera": np.full((90, 160, 3), 128, np.uint8),
                }, {}

        env = DummyVecEnv([CameraEnv])
        try:
            model = make_model(
                env, device="cpu", seed=13, tensorboard_log=None,
                policy_architecture="lidar_camera_cnn",
            )
            rng = np.random.default_rng(13)
            count = 64
            state = rng.normal(size=(count, STATE_DIM)).astype(np.float32)
            summary = behavior_clone_actor(
                model,
                {
                    "lidar": rng.uniform(0.2, 1.0, size=(count, LIDAR_BEAMS)).astype(np.float32),
                    "state": state,
                    "camera": rng.integers(0, 256, size=(count, 90, 160, 3), dtype=np.uint8),
                    "actions": np.tanh(state[:, :2]).astype(np.float32),
                },
                epochs=1, batch_size=16, seed=13,
            )
            self.assertEqual(summary["transitions"], count)
            self.assertTrue(np.isfinite(summary["final_validation_loss"]))
            self.assertFalse(model.policy.normalize_images)
        finally:
            env.close()

    def test_policy_collection_persists_only_student_inputs_and_teacher_actions(self) -> None:
        class FakeVecEnv:
            num_envs = 2

            def reset(self):
                return {
                    "lidar": np.full((2, LIDAR_BEAMS), 0.25, np.float32),
                    "state": np.full((2, STATE_DIM), 0.5, np.float32),
                }

            def env_method(self, method):
                self.method = method
                return [
                    {"lidar": np.full(LIDAR_BEAMS, 0.9, np.float32),
                     "state": np.full(STATE_DIM, 0.9, np.float32)}
                    for _ in range(2)
                ]

            def step(self, actions):
                self.actions = actions
                return self.reset(), np.zeros(2), np.zeros(2, dtype=bool), [{}, {}]

        class FakeTeacher:
            def predict(self, observations, deterministic):
                assert deterministic
                assert np.all(observations["state"] == 0.9)
                return np.tile(np.asarray([0.1, -0.2], np.float32), (2, 1)), None

        with TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            dataset, summary = collect_policy_demonstrations(
                FakeVecEnv(), FakeTeacher(), steps=3, output_path=path
            )
            self.assertEqual(summary, {"transitions": 6, "environments": 2})
            self.assertEqual(set(dataset), {"lidar", "state", "actions"})
            self.assertTrue(np.all(dataset["lidar"] == 0.25))
            self.assertTrue(np.all(dataset["state"] == 0.5))
            self.assertTrue(np.allclose(dataset["actions"], [0.1, -0.2]))
            with np.load(path) as saved:
                self.assertEqual(set(saved.files), {"lidar", "state", "actions"})

    def test_dagger_collection_executes_student_but_stores_teacher_labels(self) -> None:
        class FakeVecEnv:
            num_envs = 1

            def reset(self):
                return {
                    "lidar": np.full((1, LIDAR_BEAMS), 0.25, np.float32),
                    "state": np.full((1, STATE_DIM), 0.5, np.float32),
                }

            def env_method(self, method):
                return [{
                    "lidar": np.full(LIDAR_BEAMS, 0.9, np.float32),
                    "state": np.full(STATE_DIM, 0.9, np.float32),
                }]

            def step(self, actions):
                self.executed_actions = actions.copy()
                return self.reset(), np.zeros(1), np.zeros(1, dtype=bool), [{}]

        class FakePolicy:
            def __init__(self, action):
                self.action = np.asarray([action], dtype=np.float32)

            def predict(self, observations, deterministic):
                self.deterministic = deterministic
                return self.action.copy(), None

        env = FakeVecEnv()
        with TemporaryDirectory() as directory:
            dataset, _ = collect_policy_demonstrations(
                env,
                FakePolicy([0.1, -0.2]),
                steps=1,
                output_path=Path(directory) / "dagger.npz",
                behavior_policy=FakePolicy([0.7, 0.6]),
            )
        self.assertTrue(np.allclose(env.executed_actions[0], [0.7, 0.6]))
        self.assertTrue(np.allclose(dataset["actions"][0], [0.1, -0.2]))

    def test_official_route_teacher_collection_keeps_camera_and_excludes_privileged_state(self) -> None:
        from src.layer3.behavior_cloning import collect_centerline_demonstrations

        class FakeVecEnv:
            num_envs = 1

            def reset(self):
                return {
                    "lidar": np.ones((1, LIDAR_BEAMS), np.float32),
                    "state": np.zeros((1, STATE_DIM), np.float32),
                    "camera": np.full((1, 90, 160, 3), 128, np.uint8),
                }

            def env_method(self, method):
                assert method == "get_current_info"
                return [{"position": [1.0, 2.0], "yaw": 0.25, "current_progress_m": 3.0}]

            def step(self, actions):
                assert actions.shape == (1, 2)
                return self.reset(), np.zeros(1), np.zeros(1, dtype=bool), [{}]

        class Expert:
            def action_batch(self, infos, observations=None):
                assert infos[0]["position"] == [1.0, 2.0]
                assert observations is not None and "lidar" in observations
                return np.asarray([[0.2, -0.1]], dtype=np.float32)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "official-demo.npz"
            progress_updates = []
            dataset, summary = collect_centerline_demonstrations(
                FakeVecEnv(), Expert(), steps=2, output_path=path,
                progress_callback=lambda completed, target, stations, poses: progress_updates.append(
                    (completed, target, stations, poses)
                ),
            )
            self.assertEqual(summary["transitions"], 2)
            self.assertEqual(progress_updates[-1][3], [{
                "env_id": 0, "pose": [1.0, 2.0], "yaw": 0.25, "speed": None,
                "route_progress_m": 3.0,
            }])
            self.assertEqual(
                summary["route_direction_diagnostics"],
                [{
                    "env_id": 0,
                    "start_progress_m": 3.0,
                    "end_progress_m": 3.0,
                    "net_progress_m": 0.0,
                    "forward_progress_steps": 0,
                    "backward_progress_steps": 0,
                    "reverse_motion_steps": 0,
                    "negative_throttle_steps": 0,
                }],
            )
            self.assertEqual(set(dataset), {"lidar", "state", "camera", "actions"})
            self.assertEqual(progress_updates[-1][:3], (2, 2, [3.0]))
            with np.load(path) as saved:
                self.assertEqual(set(saved.files), {"lidar", "state", "camera", "actions"})
                self.assertEqual(saved["camera"].dtype, np.uint8)

    def test_official_route_diagnostics_do_not_join_progress_across_auto_reset(self) -> None:
        from src.layer3.behavior_cloning import collect_centerline_demonstrations

        class FakeVecEnv:
            num_envs = 1

            def __init__(self):
                self.progress = 10.0
                self.steps = 0

            def reset(self):
                return {
                    "lidar": np.ones((1, LIDAR_BEAMS), np.float32),
                    "state": np.zeros((1, STATE_DIM), np.float32),
                }

            def env_method(self, method):
                return [{"position": [0.0, 0.0], "yaw": 0.0,
                         "current_progress_m": self.progress}]

            def step(self, actions):
                self.steps += 1
                if self.steps == 1:
                    self.progress = 3.0  # auto-reset state for the next call
                    return self.reset(), np.zeros(1), np.ones(1, dtype=bool), [{
                        "current_progress_m": 12.0,
                        "frontier_progress_m": 12.0,
                        "termination_reason": "collision",
                        "collision_event": True,
                        "episode_won": False,
                        "position": [1.25, -0.5, 0.1],
                        "yaw": 0.75,
                        "true_speed": 0.4,
                        "throttle_command": 0.12,
                        "steering_command": -0.2,
                    }]
                self.progress = 3.2
                return self.reset(), np.zeros(1), np.zeros(1, dtype=bool), [{
                    "current_progress_m": 3.2,
                }]

        class Expert:
            def action_batch(self, infos, observations=None):
                return np.zeros((1, 2), dtype=np.float32)

        with TemporaryDirectory() as directory:
            dataset, summary = collect_centerline_demonstrations(
                FakeVecEnv(), Expert(), steps=2,
                output_path=Path(directory) / "episodes.npz",
            )

        self.assertEqual(summary["transitions"], 1)
        self.assertEqual(dataset["actions"].shape, (1, 2))
        episode = summary["route_episode_diagnostics"][0]
        self.assertEqual(episode["reason"], "collision")
        self.assertEqual(episode["start_progress_m"], 10.0)
        self.assertEqual(episode["end_progress_m"], 12.0)
        self.assertEqual(episode["net_progress_m"], 2.0)
        self.assertEqual(episode["terminal_position_xy_m"], [1.25, -0.5])
        self.assertEqual(episode["terminal_yaw_rad"], 0.75)
        self.assertEqual(episode["terminal_true_speed_mps"], 0.4)
        self.assertEqual(episode["terminal_throttle_command"], 0.12)
        self.assertEqual(episode["terminal_steering_command"], -0.2)
        self.assertIsNone(summary["route_direction_diagnostics"][0]["net_progress_m"])
        trace = summary["collision_trace_examples"][0]
        self.assertEqual(trace["step"], 1)
        self.assertEqual(trace["collision_state"]["position_xy_m"], [1.25, -0.5])
        self.assertEqual(len(trace["preceding_steps"]), 1)
        self.assertEqual(trace["preceding_steps"][0]["teacher_action"], [0.0, 0.0])
        self.assertAlmostEqual(trace["preceding_steps"][0]["front_min_lidar_m"], 10.0)

    def test_teacher_collection_stops_after_five_seconds_without_frontier_progress(self) -> None:
        from src.layer3.behavior_cloning import collect_centerline_demonstrations

        class FakeVecEnv:
            num_envs = 1

            def reset(self):
                return {
                    "lidar": np.ones((1, LIDAR_BEAMS), np.float32),
                    "state": np.zeros((1, STATE_DIM), np.float32),
                }

            def env_method(self, method):
                assert method == "get_current_info"
                return [{
                    "current_progress_m": 3.0,
                    "time_since_push_s": 5.0,
                }]

            def step(self, actions):
                raise AssertionError("stalled teacher must stop before issuing a step")

        class Expert:
            def action_batch(self, infos, observations=None):
                raise AssertionError("stalled teacher must stop before computing an action")

        with TemporaryDirectory() as directory:
            dataset, summary = collect_centerline_demonstrations(
                FakeVecEnv(), Expert(), steps=100,
                output_path=Path(directory) / "stalled.npz",
            )

        self.assertEqual(summary["transitions"], 0)
        self.assertTrue(summary["stagnation_detected"])
        self.assertEqual(summary["stagnation_env_id"], 0)
        gate = assess_teacher_collection(summary)
        self.assertFalse(gate["ready"])
        self.assertIn("teacher_collection_stalled", gate["reason"])
        self.assertEqual(dataset["actions"].shape, (0, 2))

    def test_teacher_collection_stops_after_successful_target_episode(self) -> None:
        class FakeVecEnv:
            num_envs = 1

            def reset(self):
                return {
                    "lidar": np.ones((1, LIDAR_BEAMS), np.float32),
                    "state": np.zeros((1, STATE_DIM), np.float32),
                }

            def env_method(self, method):
                return [{"position": [0.0, 0.0], "yaw": 0.0}]

            def step(self, actions):
                return self.reset(), np.zeros(1), np.ones(1, dtype=bool), [{
                    "episode_won": True,
                    "termination_reason": "training_goal_reached",
                    "current_progress_m": 10.0,
                    "frontier_progress_m": 311.2,
                    "lap_count": 10,
                }]

        class Expert:
            def action_batch(self, infos, observations=None):
                return np.asarray([[0.05, 0.0]], dtype=np.float32)

        with TemporaryDirectory() as directory:
            dataset, summary = collect_centerline_demonstrations(
                FakeVecEnv(), Expert(), steps=100,
                output_path=Path(directory) / "ten-lap-demo.npz",
            )

        self.assertEqual(summary["transitions"], 1)
        self.assertEqual(summary["successful_episodes"], 1)
        self.assertEqual(summary["route_episode_diagnostics"][0]["reason"], "training_goal_reached")
        self.assertEqual(dataset["actions"].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
