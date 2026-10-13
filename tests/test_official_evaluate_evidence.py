from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.layer3 import official_evaluate
from src.layer3.official_evaluate import (
    _frontier_fraction,
    _official_xy_position,
    _sha256,
    _valid_camera_frame,
    evaluate_attempt,
)
from src.layer3.preview_writer import PreviewWriter


class _Transport:
    def __init__(self, ages: list[float | None]):
        self.ages = iter(ages)

    def transport_status(self):
        return {"topic_age_s": {"camera": next(self.ages, None)}}


class _Env:
    def __init__(self, camera_ages: list[float | None]):
        self.steps = 0
        self.racer = _Transport(camera_ages)
        self._last_snap = SimpleNamespace(
            lidar_ranges=np.ones(16, dtype=np.float32), heading_yaw=0.25
        )

    @staticmethod
    def _observation():
        return {
            "state": np.zeros(9, dtype=np.float32),
            "lidar": np.ones(24, dtype=np.float32),
            "camera": np.full((8, 12, 3), 127, dtype=np.uint8),
        }

    def reset(self):
        return self._observation(), {
            # Official IPS pose is [x, y, z]; route progress uses the first
            # two coordinates, while the unrelated height must be ignored.
            "race_position": [0.0, 0.0, 7.0],
            "warmup_lap_times_s": [12.0],
            "race_lap_times_s": [],
            "race_collisions": 0,
        }

    def step(self, _action):
        self.steps += 1
        collision = self.steps >= 2
        info = {
            "race_position": [0.3 * self.steps, 0.7 * self.steps, 7.0],
            "race_lap_times_s": [],
            "warmup_lap_times_s": [12.0],
            "race_collisions": 0,
            "raw_collision_count": int(collision),
            "collision_event": collision,
            "race_complete": False,
            "throttle_command": 0.4,
            "steering_command": -0.2,
            "control_interval_s": 0.05,
            "lidar_scan_rate_hz": 40.0,
        }
        return self._observation(), 0.0, False, False, info

    def close(self):
        return None


class _Model:
    last_debug = {"debug_signal": 1.0}

    def predict(self, _observation, *, deterministic):
        assert deterministic is True
        return np.asarray([0.4, -0.2], dtype=np.float32), None


class _Route:
    length_m = 45.80209158004757

    def __init__(self, *, projection_valid: bool = True):
        self.projection_valid = projection_valid
        self.updates = 0
        self.projected_positions = []

    def reset(self, x, y, now=None):
        self.projected_positions.append((float(x), float(y)))
        return {
            "progress_m": 0.0,
            "current_progress_m": 0.0,
            "line": [[0.0, -0.5], [0.0, 0.5]],
            "current_projection_valid": True,
        }

    def update(self, x, y, now=None):
        self.updates += 1
        self.projected_positions.append((float(x), float(y)))
        return {
            "progress_m": float(5 + self.updates),
            "current_progress_m": float(5 + self.updates),
            "line": [[1.0, -0.5], [1.0, 0.5]],
            "advanced_m": 1.0,
            "current_projection_valid": self.projection_valid,
        }

    def lateral_distance(self, _x, _z):
        return 0.1 * max(1, self.updates)


def test_attempt_records_authenticated_route_camera_and_latency_evidence(tmp_path):
    env = _Env([0.1, 0.8])
    route = _Route()
    live_updates = []
    writer = PreviewWriter(tmp_path / "camera")
    writer.__enter__()
    try:
        result = evaluate_attempt(
            _Model(), env,
            wall_timeout_s=30.0,
            max_steps=5,
            attempt_index=1,
            trace_steps=2,
            frontier_tracker=route,
            frontier_stall_after_s=5.0,
            camera_preview_writer=writer,
            on_step=lambda _step, _observation, info: live_updates.append(info),
        )
    finally:
        writer.__exit__(None, None, None)

    assert result["stop_reason"] == "collision_detected"
    assert result["camera_sample_count"] == 2
    assert result["camera_fresh_sample_count"] == 1
    assert result["camera_stale_sample_count"] == 1
    assert result["camera_missing_sample_count"] == 0
    assert result["camera_freshness_unknown_count"] == 0
    assert result["camera_age_s"]["count"] == 2
    assert result["policy_inference_latency_s"]["count"] == 2
    assert result["frontier_progress_m"] == 7.0
    assert result["frontier_progress_fraction"] == 7.0 / (route.length_m * 10.0)
    assert result["frontier_projection_valid"] is True
    assert result["frontier_lateral_offset_m"] == 0.2
    assert route.projected_positions == [(0.0, 0.0), (0.3, 0.7), (0.6, 1.4)]
    assert all(item["route_projection_valid"] is True for item in live_updates)
    assert result["action_trace"][0]["camera_fresh"] is True
    assert result["action_trace"][1]["camera_fresh"] is False
    assert result["action_trace"][0]["policy_inference_latency_s"] >= 0.0
    assert np.isclose(result["action_trace"][0]["policy_throttle"], 0.4)
    assert result["action_trace"][0]["applied_throttle"] == 0.4
    assert (tmp_path / "camera" / "env-0.jpg").is_file()


def test_diagnostic_stops_and_reports_invalid_final_projection():
    result = evaluate_attempt(
        _Model(), _Env([0.1]),
        wall_timeout_s=30.0,
        max_steps=5,
        attempt_index=1,
        frontier_tracker=_Route(projection_valid=False),
        frontier_stall_after_s=5.0,
    )

    assert result["stop_reason"] == "frontier_telemetry_invalid"
    assert result["frontier_projection_valid"] is False


def test_competitive_route_monitor_is_observational_only():
    result = evaluate_attempt(
        _Model(), _Env([0.1, 0.8]), wall_timeout_s=30.0, max_steps=5,
        attempt_index=1, frontier_tracker=_Route(projection_valid=False),
        route_monitor_only=True, frontier_stall_after_s=0.0,
    )
    # The route monitor records bad projections without changing the official
    # evaluator's collision stop/scoring semantics.
    assert result["stop_reason"] == "collision_detected"
    assert result["route_projection_sample_count"] == 3
    assert result["route_projection_valid_sample_count"] == 1
    assert result["route_projection_invalid_sample_count"] == 2
    assert result["score_status"] == "collision_detected"


def test_route_monitor_identity_binds_scene_route_and_native_process(tmp_path, monkeypatch):
    from src.layer3 import official_tracks

    output = tmp_path / "evaluation.json"
    (tmp_path / "windows_simulator_processes.json").write_text(
        "\ufeff" + __import__("json").dumps({
            "run_id": "abc123abc123",
            "records": [{
                "pid": 1234, "port": 49592,
                "executable_path": r"C:\\sim\\AutoDRIVE Simulator.exe",
                "process_creation_utc": "2026-10-13T11:00:00Z",
                "batch_mode": False,
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(official_tracks, "verify_windows_scene", lambda _route, executable: {
        "scene_path": "/sim/Data/level0", "scene_sha256": "a" * 64,
        "track_profile": "competition", "verified": True,
    })
    route = __import__("pathlib").Path(official_evaluate.__file__).resolve().parents[2] / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"

    identity = official_evaluate._route_monitor_identity(
        output_path=output, frontier_path=route, expected_run_id="abc123abc123",
    )

    assert identity["official_scene_sha256"] == "a" * 64
    assert identity["route_sha256"] == _sha256(route)
    assert identity["simulator_identity"]["pid"] == 1234
    assert identity["simulator_identity"]["port"] == 49592
    assert identity["simulator_identity"]["manifest_sha256"] == _sha256(
        tmp_path / "windows_simulator_processes.json"
    )


def test_warmup_inclusive_frontier_target_keeps_probe_fraction_below_complete():
    route = _Route()
    result = evaluate_attempt(
        _Model(), _Env([0.1, 0.8]),
        wall_timeout_s=30.0,
        max_steps=5,
        attempt_index=1,
        race_laps_required=3,
        frontier_progress_target_laps=4,
        frontier_tracker=route,
    )

    assert result["frontier_progress_target_laps"] == 4
    assert result["frontier_progress_target_m"] == route.length_m * 4
    assert result["frontier_progress_fraction"] == 7.0 / (route.length_m * 4)
    assert result["frontier_progress_fraction"] < 1.0


def test_camera_shape_and_ten_lap_fraction_helpers():
    assert _valid_camera_frame({"camera": np.zeros((8, 12, 3), dtype=np.uint8)}).shape == (8, 12, 3)
    assert _valid_camera_frame({"camera": np.zeros((3, 8, 12), dtype=np.uint8)}).shape == (3, 8, 12)
    assert _valid_camera_frame({"camera": np.zeros((8, 12), dtype=np.uint8)}) is None
    assert _frontier_fraction(45.0, 45.0) == 0.1
    # Diagnostic path progress includes its completed warmup traversal.
    assert _frontier_fraction(45.0 * 3, 45.0, target_laps=4) == 0.75
    assert _frontier_fraction(500.0, 45.0) == 1.0
    assert _official_xy_position([1.0, 2.0, 99.0]) == [1.0, 2.0]
    assert _official_xy_position([1.0]) is None


def test_training_probe_attempt_completes_three_scored_laps_without_ranking(monkeypatch):
    class CompleteProbeEnv(_Env):
        def __init__(self):
            super().__init__([])

        def reset(self):
            observation, info = super().reset()
            info["warmup_lap_times_s"] = [12.0]
            return observation, info

        def step(self, _action):
            self.steps += 1
            info = {
                "race_position": [0.3 * self.steps, 0.0, 0.0],
                "race_lap_times_s": [30.0, 31.0, 32.0],
                "warmup_lap_times_s": [12.0],
                "race_collisions": 0,
                "raw_collision_count": 0,
                "race_complete": True,
                "throttle_command": 0.4,
                "steering_command": -0.2,
                "control_interval_s": 0.05,
                "lidar_scan_rate_hz": 40.0,
            }
            return self._observation(), 0.0, True, False, info

    monkeypatch.setattr(
        "src.layer3.official_evaluate.verify_vehicle_motion",
        lambda *_args, **_kwargs: {"verified": True},
    )
    result = evaluate_attempt(
        _Model(), CompleteProbeEnv(), wall_timeout_s=30.0, max_steps=5,
        attempt_index=1, race_laps_required=3, noncompetitive=True,
    )

    assert result["race_laps_required"] == 3
    assert result["complete"] is True
    assert result["score_status"] == "valid"
    assert result["valid_for_comparison"] is False
    assert result["noncompetitive_diagnostic"] is True
    assert result["adjusted_race_time_s"] is None
    assert result["warmup_lap_times_s"] == [12.0]


def test_summarize_attempt_keeps_ten_lap_default_and_accepts_three_lap_probe():
    from src.layer3.official_evaluate import summarize_attempt

    assert summarize_attempt(lap_times_s=[1.0] * 3, race_collisions=0)["complete"] is False
    probe = summarize_attempt(
        lap_times_s=[1.0, 2.0, 3.0], race_collisions=0, race_laps_required=3
    )
    assert probe["complete"] is True
    assert probe["race_laps_required"] == 3


def test_checkpoint_hash_helper(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"policy artifact")

    assert _sha256(checkpoint) == "1c122c364ffab148fd84474ea2e8fca676eb868e4738b5a61010736865bf669f"


def test_evaluate_report_includes_loaded_checkpoint_hash(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"policy artifact")

    class FakeOfficialEnv(_Env):
        def __init__(self, **_kwargs):
            super().__init__([0.1, 0.8])

    published = []

    class FakePublisher:
        def __init__(self, *_args, **_kwargs):
            pass

        def enqueue(self, payload):
            published.append(payload)

        def stop(self):
            pass

    monkeypatch.setattr("src.layer2.official_race_env.OfficialRaceEnv", FakeOfficialEnv)
    monkeypatch.setattr("src.layer3.official_policy.load_policy", lambda *_args, **_kwargs: _Model())
    monkeypatch.setattr("src.layer3.hub_callback._Publisher", FakePublisher)
    monkeypatch.setenv("HUB_URL", "http://127.0.0.1:8090")
    monkeypatch.setenv("AICAR_RUN_ID", "test-run")

    report = official_evaluate.evaluate(
        checkpoint,
        output_path=tmp_path / "evaluation.json",
        wall_timeout_s=30.0,
        max_steps=5,
    )
    capsys.readouterr()

    assert report["model_sha256"] == "1c122c364ffab148fd84474ea2e8fca676eb868e4738b5a61010736865bf669f"
    assert report["runs"][0]["stop_reason"] == "collision_detected"
    assert published[0]["cars"][0]["pose"] == [0.3, 0.7]


def test_route_monitor_entrypoint_initializes_run_id_and_keeps_competitive_scoring(
    tmp_path, monkeypatch, capsys,
):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"policy artifact")

    class FakeOfficialEnv(_Env):
        def __init__(self, **_kwargs):
            super().__init__([0.1, 0.8])

    identity = {
        "official_scene_sha256": "a" * 64,
        "official_scene_path": "/sim/Data/level0",
        "route_sha256": "b" * 64,
        "route_path": "/repo/competition/iros2026/tracks/competition/official_centerline.csv",
        "track_profile": "competition",
        "route_monitor_only": True,
        "simulator_identity": {
            "pid": 1234, "process_creation_utc": "2026-10-13T11:00:00Z",
            "executable_path": r"C:\\sim\\AutoDRIVE Simulator.exe",
            "port": 49592, "manifest_sha256": "c" * 64,
        },
    }
    monkeypatch.setattr("src.layer2.official_race_env.OfficialRaceEnv", FakeOfficialEnv)
    monkeypatch.setattr("src.layer3.official_policy.load_policy", lambda *_args, **_kwargs: _Model())
    monkeypatch.setattr(official_evaluate, "_route_monitor_identity", lambda **_kwargs: identity)
    monkeypatch.setenv("AICAR_RUN_ID", "abc123abc123")
    route = tmp_path / "route.csv"
    route.write_text("0,0\n1,0\n1,1\n", encoding="utf-8")

    report = official_evaluate.evaluate(
        checkpoint, output_path=tmp_path / "evaluation.json", wall_timeout_s=30.0,
        max_steps=5, frontier_path=route, route_monitor_only=True,
    )
    capsys.readouterr()

    assert report["protocol"] == "official_competitive_1_plus_10"
    assert report["validation_mode"] is None
    assert report["route_monitor_only"] is True
    assert report["simulator_identity"] == identity["simulator_identity"]
    assert report["runs"][0]["route_monitor"]["scoring_or_stop_effect"] is False
    assert report["runs"][0]["stop_reason"] == "collision_detected"
