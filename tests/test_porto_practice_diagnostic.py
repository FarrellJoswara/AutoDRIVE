from pathlib import Path

import pytest

from src.layer3 import official_evaluate


def _practice_route() -> Path:
    return Path(official_evaluate.__file__).resolve().parents[2] / "competition" / "iros2026" / "official_centerline.csv"


def test_verified_porto_route_is_fixed_aligned_practice_asset():
    identity = official_evaluate._verified_porto_practice_route(_practice_route())

    assert identity["track_profile"] == "practice"
    assert identity["route_sha256"] == official_evaluate.OFFICIAL_PORTO_PRACTICE_ROUTE_SHA256
    assert identity["practice_asset_source_image"] == "autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice"


def test_porto_diagnostic_rejects_competition_route():
    competition_route = _practice_route().parent / "tracks" / "competition" / "official_centerline.csv"

    with pytest.raises(ValueError, match="fixed official practice centerline"):
        official_evaluate._verified_porto_practice_route(competition_route)


def test_porto_diagnostic_requires_fresh_single_attempt_and_practice_profile(monkeypatch):
    monkeypatch.setenv("AICAR_OFFICIAL_TRACK_PROFILE", "practice")
    monkeypatch.setenv("AICAR_SIMULATOR_BACKEND", "windows_external")
    monkeypatch.setenv("AICAR_OFFICIAL_SIMULATOR_EXE", r"C:\official\AutoDRIVE Simulator.exe")
    args = {
        "model_path": Path("unused.zip"),
        "validation_mode": "porto_practice_diagnostic",
        "frontier_path": _practice_route(),
        "controller": "lidar_gap",
        "lap_progress_timeout_s": 240,
        "lap_progress_wall_timeout_s": 240,
        "frontier_stall_after_s": 5.0,
    }

    with pytest.raises(ValueError, match="one invocation per fresh simulator attempt"):
        official_evaluate.evaluate(**args, attempts=3)

    monkeypatch.setenv("AICAR_OFFICIAL_TRACK_PROFILE", "competition")
    with pytest.raises(ValueError, match="AICAR_OFFICIAL_TRACK_PROFILE=practice"):
        official_evaluate.evaluate(**args, attempts=1)


def test_native_practice_runtime_uses_verified_windows_scene_without_docker_image(monkeypatch):
    from src.layer3.official_tracks import WINDOWS_SCENES

    monkeypatch.setenv("AICAR_SIMULATOR_BACKEND", "windows_external")
    monkeypatch.setenv("AICAR_OFFICIAL_TRACK_PROFILE", "practice")
    monkeypatch.setenv("AICAR_OFFICIAL_SIMULATOR_EXE", r"C:\official\AutoDRIVE Simulator.exe")
    monkeypatch.delenv("AICAR_OFFICIAL_SIMULATOR_IMAGE", raising=False)
    monkeypatch.setattr(
        "src.layer3.official_tracks.verify_windows_scene",
        lambda route, executable: {
            "scene_path": str(Path(executable).parent / "Data" / "level0"),
            "scene_sha256": WINDOWS_SCENES["practice"],
            "track_profile": "practice",
            "verified": True,
        },
    )

    identity = official_evaluate._verified_porto_practice_runtime(
        official_evaluate._verified_porto_practice_route(_practice_route())
    )
    assert identity["simulator_backend"] == "windows_external"
    assert identity["scene_verification"]["scene_sha256"] == WINDOWS_SCENES["practice"]


def test_native_practice_runtime_rejects_wrong_windows_scene(monkeypatch):
    monkeypatch.setenv("AICAR_SIMULATOR_BACKEND", "windows_external")
    monkeypatch.setenv("AICAR_OFFICIAL_TRACK_PROFILE", "practice")
    monkeypatch.setenv("AICAR_OFFICIAL_SIMULATOR_EXE", r"C:\official\AutoDRIVE Simulator.exe")
    monkeypatch.setattr(
        "src.layer3.official_tracks.verify_windows_scene",
        lambda route, executable: {
            "scene_sha256": "not-practice",
            "track_profile": "practice",
            "verified": True,
        },
    )
    with pytest.raises(ValueError, match="does not match the verified Porto practice track"):
        official_evaluate._verified_porto_practice_runtime(
            official_evaluate._verified_porto_practice_route(_practice_route())
        )


def test_control_gap_gate_uses_monotonic_interval_and_keeps_ros_stamp_anomaly():
    source_interval, wall_interval, actual_gap, stamp_gap = (
        official_evaluate._control_interval_diagnostics({
            "control_interval_s": 2.0,
            "control_interval_source": "ros_sensor_stamp",
            "control_interval_wall_s": 0.052,
        })
    )

    assert source_interval == 2.0
    assert wall_interval == 0.052
    assert actual_gap is False
    assert stamp_gap is True


def test_practice_attempt_requires_fresh_positive_time_for_every_lap():
    run = {
        "stop_reason": "race_complete",
        "race_laps_completed": 3,
        "race_lap_times_s": [95.0, 96.0, 97.0],
        "race_lap_timing_valid": True,
        "race_lap_timing_missing_count": 0,
        "race_collisions": 0,
        "warmup_collisions": 0,
        "raw_collision_count_at_finish": 0,
        "frontier_projection_valid": True,
        "route_projection_invalid_sample_count": 0,
        "control_gap_fault_count": 0,
        "camera_missing_sample_count": 0,
        "camera_stale_sample_count": 0,
        "camera_freshness_unknown_count": 0,
        "motion_verification": {"verified": True},
    }

    assert official_evaluate._porto_practice_attempt_passed(run, 3)
    run["race_lap_timing_valid"] = False
    assert not official_evaluate._porto_practice_attempt_passed(run, 3)


def test_practice_attempt_gate_supports_ten_fresh_timed_laps():
    run = {
        "stop_reason": "race_complete",
        "race_laps_completed": 10,
        "race_lap_times_s": [95.0 + i for i in range(10)],
        "race_lap_timing_valid": True,
        "race_lap_timing_missing_count": 0,
        "race_collisions": 0,
        "warmup_collisions": 0,
        "raw_collision_count_at_finish": 0,
        "frontier_projection_valid": True,
        "route_projection_invalid_sample_count": 0,
        "control_gap_fault_count": 0,
        "camera_missing_sample_count": 0,
        "camera_stale_sample_count": 0,
        "camera_freshness_unknown_count": 0,
        "motion_verification": {"verified": True},
    }

    assert official_evaluate._porto_practice_attempt_passed(run, 10)
    run["race_lap_times_s"][-1] = 0.0
    assert not official_evaluate._porto_practice_attempt_passed(run, 10)


def test_evaluator_rejects_nonpositive_practice_lap_target():
    with pytest.raises(ValueError, match="porto_practice_laps must be at least 1"):
        official_evaluate.evaluate(Path("unused.zip"), controller="lidar_gap", porto_practice_laps=0)
