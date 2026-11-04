"""Offline acceptance checks for the isolated Porto demonstration path."""
import json

import pytest

from src.layer3.behavior_cloning import assess_teacher_collection, fit_accepted_behavior_cloned_dataset


def _clean_collection():
    return {
        "successful_episodes": 2,
        "completed_episodes": 2,
        "clean_laps_completed": 10,
        "expected_clean_laps": 10,
        "required_clean_episodes": 2,
        "laps_per_episode": 5,
        "collection_stop_reason": "clean_lap_goal",
        "collision_count": 0,
        "stagnation_detected": False,
        "telemetry_fault_count": 0,
        "control_fault_count": 0,
        "camera_render_valid": True,
        "data_integrity_valid": True,
        "route_episode_diagnostics": [
            {"episode_clean": True, "race_laps_completed": 5},
            {"episode_clean": True, "race_laps_completed": 5},
        ],
    }


def test_porto_collection_acceptance_requires_exact_two_episode_lap_goal():
    accepted = assess_teacher_collection(
        _clean_collection(), expected_clean_laps=10,
        required_clean_episodes=2, laps_per_episode=5,
    )
    assert accepted["ready"] is True

    collection = _clean_collection()
    collection["route_episode_diagnostics"][1]["race_laps_completed"] = 4
    rejected = assess_teacher_collection(
        collection, expected_clean_laps=10,
        required_clean_episodes=2, laps_per_episode=5,
    )
    assert rejected["ready"] is False
    assert "teacher_collection_episode_lap_goal_mismatch" in rejected["reason"]


@pytest.mark.parametrize("fault, reason", [
    ("collision_count", "teacher_collection_collision_detected"),
    ("telemetry_fault_count", "teacher_collection_telemetry_fault"),
    ("control_fault_count", "teacher_collection_control_fault"),
    ("camera_render_valid", "teacher_collection_camera_frames_unrendered"),
])
def test_porto_collection_rejects_faulted_attempts(fault, reason):
    collection = _clean_collection()
    collection[fault] = False if fault == "camera_render_valid" else 1
    result = assess_teacher_collection(
        collection, expected_clean_laps=10,
        required_clean_episodes=2, laps_per_episode=5,
    )
    assert result["ready"] is False
    assert reason in result["reason"]


def test_offline_fit_refuses_unaccepted_data_before_loading_policy_runtime(tmp_path):
    dataset = tmp_path / "not_accepted.npz"
    dataset.write_bytes(b"not a dataset")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(json.dumps({"status": "rejected", "readiness": {"ready": False}}))
    with pytest.raises(ValueError, match="refuses a dataset without accepted clean-lap evidence"):
        fit_accepted_behavior_cloned_dataset(dataset, acceptance, tmp_path / "fit")


def test_native_windows_training_invocation_translates_porto_collection_profile():
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    command = NativeOfficialBackend._training_command({
        "AICAR_TRAIN_OUT": "/runs/rl/porto_bc_native",
        "AICAR_TRAIN_N_ENVS": "1",
        "AICAR_TRAIN_TIMESTEPS": "0",
        "AICAR_AUTO_EVALUATION_ENABLED": "0",
        "AICAR_BEHAVIOR_CLONING_COLLECT_ONLY": "1",
        "AICAR_BEHAVIOR_CLONING_CLEAN_LAPS": "10",
        "AICAR_BEHAVIOR_CLONING_REQUIRED_CLEAN_EPISODES": "2",
        "AICAR_BEHAVIOR_CLONING_WALL_TIMEOUT_S": "1800",
        "AICAR_BEHAVIOR_CLONING_STEPS_PER_ENV": "30000",
        "AICAR_TRAIN_RACE_LAPS": "5",
        "AICAR_TRAIN_WARMUP_LAPS": "0",
        "AICAR_OFFICIAL_TRACK_PROFILE": "practice",
        "AICAR_OFFICIAL_SIMULATOR_IMAGE": "autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice",
    })
    assert command[1:3] == ["-u", "-m"]
    args = dict(zip(command[4::2], command[5::2]))
    assert args["--behavior-cloning-collect-only"] == "1"
    assert args["--behavior-cloning-clean-laps"] == "10"
    assert args["--behavior-cloning-required-clean-episodes"] == "2"
    assert args["--race-laps"] == "5"


def test_native_windows_evaluation_invocation_uses_single_porto_attempt():
    from src.layer4.hub.native_official_backend import NativeOfficialBackend

    route = "competition/iros2026/official_centerline.csv"
    command = NativeOfficialBackend._evaluation_command({
        "AICAR_MODEL_PATH": "/runs/rl/porto_bc_native/behavior_cloned_policy.zip",
        "AICAR_CONTROLLER": "ppo",
        "AICAR_EVALUATION_ATTEMPTS": "1",
        "AICAR_EVALUATION_VALIDATION_MODE": "porto_practice_diagnostic",
        "AICAR_EVALUATION_FRONTIER_PATH": route,
        "AICAR_EVALUATION_OUTPUT": "/runs/rl/porto_bc_native/attempt_1.json",
        "AICAR_OFFICIAL_TRACK_PROFILE": "practice",
    })
    assert command[1:3] == ["-u", "-m"]
    args = dict(zip(command[4::2], command[5::2]))
    assert args["--attempts"] == "1"
    assert args["--validation-mode"] == "porto_practice_diagnostic"
    assert args["--frontier-path"] == route
