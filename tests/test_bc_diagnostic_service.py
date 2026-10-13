"""Mission Control orchestration checks for noncompetitive BC diagnostics."""
import json
import time
import asyncio
from pathlib import Path

import pytest
from unittest.mock import patch

from scripts.test_official_run_manager import FakeDocker
from src.layer4.hub.official_evaluation_service import (
    OfficialEvaluationService,
    _competitive_route_monitor_options,
    _evaluation_child_settings,
)
from src.layer4.hub.official_replay_job import OfficialReplayJob
from src.layer4.hub.official_run_manager import OfficialRunManager, normalize_official_config
from src.layer4.hub.native_official_backend import NativeOfficialBackend
from src.layer4.official_settings import OfficialTrainSettings


def _checkpoint(root: Path) -> Path:
    folder = root / "logs" / "rl" / "bc-example"
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({
        "runtime": "official IROS 2026 API and simulator images",
        "observation_profile": "official_sensors_camera",
        "actions": {
            "throttle_mode": "bidirectional",
            "negative_throttle_mode": "allow",
            "steering_mode": "normal",
            "steering_action_scale": 1.0,
            "straight_throttle_gain": 1.0,
            "straight_throttle_steering_threshold": 0.15,
        },
    }))
    checkpoint = checkpoints / "behavior_cloned_policy.zip"
    checkpoint.write_bytes(b"fixture checkpoint")
    return checkpoint


@pytest.fixture
def manager(tmp_path):
    backend = FakeDocker(tmp_path)
    result = OfficialRunManager(
        backend=backend,
        registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "logs" / "rl",
        start_monitor=False,
    )
    yield result
    result.close()


def test_diagnostic_settings_reach_child_environment_and_durable_run_config(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    frontier = tmp_path / "official_centerline.csv"
    options = {
        "validation_mode": "bc_diagnostic",
        "lap_progress_timeout_s": 240,
        "lap_progress_wall_timeout_s": 240,
        "frontier_stall_after_s": 5,
        "frontier_path": str(frontier),
    }
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        row = manager.create_run(
            {"resume": str(checkpoint), "n_envs": 1, "target_kl": 0.01},
            kind="evaluation",
            evaluation_options=options,
        )
    api = next(item for item in manager.backend.containers.values()
               if item["Labels"].get("aicar.run_id") == row["run_id"]
               and item["Labels"].get("aicar.role") == "api")
    environment = dict(item.split("=", 1) for item in api["kwargs"]["env"])
    assert environment["AICAR_EVALUATION_VALIDATION_MODE"] == "bc_diagnostic"
    assert environment["AICAR_LAP_PROGRESS_TIMEOUT_S"] == "240.0"
    assert environment["AICAR_LAP_PROGRESS_WALL_TIMEOUT_S"] == "240.0"
    assert environment["AICAR_FRONTIER_STALL_AFTER_S"] == "5.0"
    assert environment["AICAR_EVALUATION_FRONTIER_PATH"] == str(frontier)
    assert environment["AICAR_PPO_TARGET_KL"] == "0.01"
    durable = json.loads((Path(row["output_dir"]) / "manager_config.json").read_text())
    assert durable["evaluation_options"] == options
    assert durable["config"]["target_kl"] == 0.01


def test_competitive_route_monitor_propagates_without_diagnostic_options(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    frontier = tmp_path / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"
    frontier.parent.mkdir(parents=True)
    frontier.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
    options = {"route_monitor_only": True, "frontier_path": str(frontier)}
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        row = manager.create_run(
            {"resume": str(checkpoint), "n_envs": 1},
            kind="evaluation", evaluation_options=options,
        )
    api = next(item for item in manager.backend.containers.values()
               if item["Labels"].get("aicar.run_id") == row["run_id"]
               and item["Labels"].get("aicar.role") == "api")
    environment = dict(item.split("=", 1) for item in api["kwargs"]["env"])
    assert environment["AICAR_EVALUATION_FRONTIER_PATH"] == str(frontier)
    assert environment["AICAR_ROUTE_MONITOR_ONLY"] == "1"
    assert "AICAR_EVALUATION_VALIDATION_MODE" not in environment
    assert "AICAR_LAP_PROGRESS_TIMEOUT_S" not in environment
    assert "AICAR_LAP_PROGRESS_WALL_TIMEOUT_S" not in environment
    assert "AICAR_FRONTIER_STALL_AFTER_S" not in environment
    durable = json.loads((Path(row["output_dir"]) / "manager_config.json").read_text())
    assert durable["evaluation_options"] == options


def test_competitive_evaluation_service_selects_nonblocking_route_monitor():
    options = _competitive_route_monitor_options()
    assert options["route_monitor_only"] is True
    assert options["frontier_path"].endswith("competition/official_centerline.csv")
    assert "validation_mode" not in options
    assert "lap_progress_timeout_s" not in options
    assert "lap_progress_wall_timeout_s" not in options
    assert "frontier_stall_after_s" not in options


def test_competitive_route_monitor_rejects_arbitrary_or_diagnostic_options(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    official = tmp_path / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"
    official.parent.mkdir(parents=True)
    official.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        with pytest.raises(ValueError, match="verified official frontier path"):
            manager.create_run(
                {"resume": str(checkpoint), "n_envs": 1}, kind="evaluation",
                evaluation_options={"route_monitor_only": True, "frontier_path": str(official),
                                    "frontier_stall_after_s": 5},
            )
        arbitrary = tmp_path / "elsewhere.csv"
        arbitrary.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="verified official frontier path"):
            manager.create_run(
                {"resume": str(checkpoint), "n_envs": 1}, kind="evaluation",
                evaluation_options={"route_monitor_only": True, "frontier_path": str(arbitrary)},
            )


def test_optional_target_kl_is_preserved_from_settings_to_cli():
    defaults = OfficialTrainSettings().to_container_env(hub_url="http://hub", run_id="default")
    assert "AICAR_PPO_TARGET_KL" not in defaults
    assert "--target-kl" not in NativeOfficialBackend._training_command({})

    configured = OfficialTrainSettings(target_kl=0.01)
    environment = configured.to_container_env(hub_url="http://hub", run_id="smoke")
    assert environment["AICAR_PPO_TARGET_KL"] == "0.01"
    command = NativeOfficialBackend._training_command(environment)
    assert command[command.index("--target-kl") + 1] == "0.01"
    with pytest.raises(ValueError):
        OfficialTrainSettings(target_kl=0)


@pytest.mark.parametrize(
    ("attempt", "readiness"),
    [
        ({
            "stop_reason": "race_complete", "score_status": "valid",
            "valid_for_comparison": True, "race_time_s": 100.0,
            "adjusted_race_time_s": 101.0, "warmup_lap_times_s": [12.0],
            "race_lap_times_s": [10.0] * 10, "race_collisions": 0,
            "raw_collision_count_at_finish": 0,
            "frontier_progress_m": 100.0, "frontier_progress_fraction": 1.0,
            "frontier_lateral_offset_m": 0.12, "frontier_projection_valid": True,
            "camera_sample_count": 15, "camera_fresh_sample_count": 14,
            "camera_stale_sample_count": 1, "camera_missing_sample_count": 0,
            "camera_age_s": {"count": 15, "min": 0.01, "max": 0.8, "mean": 0.1, "std": 0.05},
            "policy_inference_latency_s": {"count": 15, "min": 0.001, "max": 0.02, "mean": 0.004, "std": 0.002},
        }, True),
        ({
            "stop_reason": "collision_detected", "score_status": "collision_detected",
            "valid_for_comparison": False, "warmup_lap_times_s": [],
            "race_lap_times_s": [], "race_collisions": 0,
            "raw_collision_count_at_finish": 1,
        }, False),
        ({
            "stop_reason": "frontier_telemetry_invalid", "score_status": "incomplete",
            "valid_for_comparison": False, "warmup_lap_times_s": [12.0],
            "race_lap_times_s": [], "race_collisions": 0,
            "raw_collision_count_at_finish": 0,
        }, False),
    ],
)
def test_diagnostic_attempt_counts_once_but_never_scores_or_promotes(
    manager, tmp_path, attempt, readiness,
):
    checkpoint = _checkpoint(tmp_path)
    report_dir = tmp_path / "diagnostic-run"
    report_dir.mkdir()
    row = {"run_id": "diagnostic-run", "output_dir": str(report_dir)}
    created_options = []

    def create_diagnostic_run(*args, **kwargs):
        created_options.append(kwargs["evaluation_options"])
        (report_dir / "evaluation.json").write_text(json.dumps({"runs": [attempt]}))
        return row

    manager.create_run = create_diagnostic_run
    manager.get_run = lambda run_id: {**row, "state": "completed"}
    service = OfficialEvaluationService(
        manager,
        registry_path=tmp_path / "evaluations.json",
        champion_dir=tmp_path / "best_official_model",
        model_root=tmp_path / "logs" / "rl",
    )
    try:
        with patch("src.layer4.hub.official_evaluation_service.verify_bc_checkpoint", return_value={"sha256": "verified"}):
            queued = service.submit(checkpoint, source="manual", validation_mode="bc_diagnostic")
        assert queued["attempts_required"] == 1
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = service.status()
            if status["state"] in {"complete", "incomplete", "failed"}:
                break
            time.sleep(0.01)
        result = service.status()["evaluation"]
        assert result["completed_attempts"] == 1
        assert result["competitive_attempts"] == 0
        assert result["attempts_required"] == 1
        assert result["valid"] is False
        assert result["competitive_score_valid"] is False
        assert result["readiness_passed"] is readiness
        assert result["runs"][0]["valid_for_comparison"] is False
        assert result["collisions"] == int(attempt.get("raw_collision_count_at_finish", 0))
        if readiness:
            assert result["runs"][0]["frontier_progress_fraction"] == 1.0
            assert result["runs"][0]["frontier_projection_valid"] is True
            assert result["runs"][0]["camera_fresh_sample_count"] == 14
            assert result["runs"][0]["policy_inference_latency_s"]["mean"] == 0.004
        assert created_options[0]["validation_mode"] == "bc_diagnostic"
        assert created_options[0]["lap_progress_timeout_s"] == 240
        assert service.champion() is None
    finally:
        service.close()


def test_diagnostic_rejects_checkpoint_without_authenticated_provenance(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    service = OfficialEvaluationService(
        manager,
        registry_path=tmp_path / "evaluations.json",
        champion_dir=tmp_path / "best_official_model",
        model_root=tmp_path / "logs" / "rl",
    )
    try:
        with patch("src.layer4.hub.official_evaluation_service.verify_bc_checkpoint", return_value=None):
            with pytest.raises(ValueError, match="authenticated"):
                service.submit(checkpoint, source="manual", validation_mode="bc_diagnostic")
        assert service.status()["state"] == "idle"
    finally:
        service.close()


def test_replay_facade_forwards_diagnostic_mode(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)

    class RecordingService:
        submitted = None

        def submit(self, model_path, **options):
            self.submitted = (model_path, options)
            return {"state": "queued", "attempts_required": 1}

    service = RecordingService()
    replay = OfficialReplayJob(manager, evaluation_service=service)
    result = replay.start(model_path=checkpoint, validation_mode="bc_diagnostic")
    assert result["attempts_required"] == 1
    assert service.submitted == (
        checkpoint,
        {"source": "manual", "trace_steps": 0, "validation_mode": "bc_diagnostic"},
    )


def test_training_probe_settings_and_evaluator_flags_reach_native_backend():
    defaults = OfficialTrainSettings().to_container_env(hub_url="http://hub", run_id="default")
    assert defaults["AICAR_AUTO_EVALUATION_MODE"] == "competitive"
    assert "AICAR_PPO_EXPLORATION_PROFILE" not in defaults

    settings = OfficialTrainSettings(
        automatic_evaluation_mode="training_probe", exploration_profile="bc_smoke",
        exploration_initial_std_throttle=0.01, exploration_initial_std_steering=0.05,
        exploration_min_std_throttle=0.01, exploration_min_std_steering=0.05,
        exploration_max_std_throttle=0.01, exploration_max_std_steering=0.05,
    )
    environment = settings.to_container_env(hub_url="http://hub", run_id="probe")
    assert environment["AICAR_AUTO_EVALUATION_MODE"] == "training_probe"
    assert environment["AICAR_PPO_EXPLORATION_PROFILE"] == "bc_smoke"
    train_command = NativeOfficialBackend._training_command(environment)
    assert train_command[train_command.index("--automatic-evaluation-mode") + 1] == "training_probe"
    assert train_command[train_command.index("--exploration-profile") + 1] == "bc_smoke"
    assert train_command[train_command.index("--exploration-initial-std-throttle") + 1] == "0.01"
    assert train_command[train_command.index("--exploration-initial-std-steering") + 1] == "0.05"
    assert train_command[train_command.index("--exploration-min-std-throttle") + 1] == "0.01"
    assert train_command[train_command.index("--exploration-min-std-steering") + 1] == "0.05"
    assert train_command[train_command.index("--exploration-max-std-throttle") + 1] == "0.01"
    assert train_command[train_command.index("--exploration-max-std-steering") + 1] == "0.05"

    evaluation_command = NativeOfficialBackend._evaluation_command({
        "AICAR_EVALUATION_VALIDATION_MODE": "training_probe",
        "AICAR_LAP_PROGRESS_TIMEOUT_S": "240",
        "AICAR_LAP_PROGRESS_WALL_TIMEOUT_S": "240",
        "AICAR_FRONTIER_STALL_AFTER_S": "5",
        "AICAR_EVALUATION_FRONTIER_PATH": "official_centerline.csv",
    })
    assert evaluation_command[evaluation_command.index("--validation-mode") + 1] == "training_probe"
    assert evaluation_command[evaluation_command.index("--lap-progress-timeout-s") + 1] == "240"
    assert evaluation_command[evaluation_command.index("--lap-progress-wall-timeout-s") + 1] == "240"
    assert evaluation_command[evaluation_command.index("--frontier-stall-after-s") + 1] == "5"


def test_training_probe_manager_persists_mode_and_enforces_diagnostic_safeguards(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    frontier = tmp_path / "official_centerline.csv"
    options = {
        "validation_mode": "training_probe",
        "lap_progress_timeout_s": 240,
        "lap_progress_wall_timeout_s": 240,
        "frontier_stall_after_s": 5,
        "frontier_path": str(frontier),
    }
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        row = manager.create_run(
            {"resume": str(checkpoint), "n_envs": 1},
            kind="evaluation", evaluation_options=options,
        )
        with pytest.raises(ValueError, match="240 seconds"):
            manager.create_run(
                {"resume": str(checkpoint), "n_envs": 1}, kind="evaluation",
                evaluation_options={**options, "lap_progress_timeout_s": 239},
            )
    api = next(item for item in manager.backend.containers.values()
               if item["Labels"].get("aicar.run_id") == row["run_id"]
               and item["Labels"].get("aicar.role") == "api")
    environment = dict(item.split("=", 1) for item in api["kwargs"]["env"])
    assert environment["AICAR_EVALUATION_VALIDATION_MODE"] == "training_probe"
    durable = json.loads((Path(row["output_dir"]) / "manager_config.json").read_text())
    assert durable["evaluation_options"] == options


def test_training_probe_child_config_normalizes_with_autoeval_disabled(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    child = _evaluation_child_settings(
        {
            "automatic_evaluation_enabled": True,
            "automatic_evaluation_mode": "training_probe",
        },
        str(checkpoint),
        race_laps=3,
    )
    assert child["automatic_evaluation_enabled"] is False
    assert child["automatic_evaluation_mode"] == "competitive"
    assert child["race_laps"] == 3
    options = {
        "validation_mode": "training_probe",
        "lap_progress_timeout_s": 240,
        "lap_progress_wall_timeout_s": 240,
        "frontier_stall_after_s": 5,
        "frontier_path": str(tmp_path / "official_centerline.csv"),
    }
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
        row = manager.create_run(
            child, "Automatic training probe", kind="evaluation",
            evaluation_options=options,
        )
    durable = json.loads((Path(row["output_dir"]) / "manager_config.json").read_text())
    assert durable["config"]["automatic_evaluation_enabled"] is False
    assert durable["config"]["automatic_evaluation_mode"] == "competitive"
    assert durable["config"]["resume"].endswith(
        "/bc-example/checkpoints/behavior_cloned_policy.zip"
    )
    assert durable["evaluation_options"] == options


def test_automatic_training_probe_uses_competitive_nonrecursive_child_settings(
    manager, tmp_path,
):
    checkpoint = _checkpoint(tmp_path)
    report_dir = tmp_path / "auto-probe-run"
    report_dir.mkdir()
    row = {"run_id": "auto-probe-run", "output_dir": str(report_dir)}
    attempt = {
        "stop_reason": "race_complete", "score_status": "valid",
        "valid_for_comparison": True, "race_time_s": 30.0,
        "adjusted_race_time_s": 31.0, "warmup_lap_times_s": [12.0],
        "race_lap_times_s": [10.0] * 3, "race_collisions": 0,
        "raw_collision_count_at_finish": 0, "frontier_projection_valid": True,
    }
    created = []

    def create_probe_run(config, display_name, **kwargs):
        with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path):
            normalized = normalize_official_config(config)
        created.append((normalized, kwargs))
        (report_dir / "evaluation.json").write_text(json.dumps({"runs": [attempt]}))
        return row

    manager.create_run = create_probe_run
    manager.get_run = lambda run_id: {**row, "state": "completed"}
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "evaluations.json",
        champion_dir=tmp_path / "best_official_model", model_root=tmp_path / "logs" / "rl",
    )
    try:
        with patch(
            "src.layer4.hub.official_evaluation_service.verify_training_probe_checkpoint",
            return_value={"verification": "verified"},
        ):
            service.submit(
                checkpoint, source="automatic", parent_run_id="training-run",
                validation_mode="training_probe",
            )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and service.status()["state"] not in {
            "complete", "incomplete", "failed",
        }:
            time.sleep(0.01)
        assert service.status()["state"] == "complete"
        child_config, kwargs = created[0]
        assert child_config["automatic_evaluation_enabled"] is False
        assert child_config["automatic_evaluation_mode"] == "competitive"
        assert kwargs["evaluation_options"]["validation_mode"] == "training_probe"
        assert service.status()["evaluation"]["competitive_score_valid"] is False
        assert service.champion() is None
    finally:
        service.close()


def test_training_probe_run_requires_authenticated_resume_parent(tmp_path):
    with pytest.raises(ValueError, match="requires an authenticated BC"):
        normalize_official_config({"automatic_evaluation_mode": "training_probe"})

    checkpoint = tmp_path / "logs" / "rl" / "bc-bundle" / "behavior_cloned_policy.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified BC fixture")
    authenticated_bc = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "a" * 64,
        "verification": "authenticated_behavior_cloning_fixture",
    }
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path), \
         patch("src.layer4.hub.official_run_manager.verify_bc_checkpoint", return_value=authenticated_bc) as verify:
        config = normalize_official_config({
            "automatic_evaluation_mode": "training_probe",
            "resume": str(checkpoint),
        })
    assert config["automatic_evaluation_mode"] == "training_probe"
    assert config["resume"] == "/runs/rl/bc-bundle/behavior_cloned_policy.zip"
    verify.assert_called_once_with(checkpoint.resolve())


def test_training_probe_accepts_authenticated_ppo_descendant_resume(tmp_path):
    checkpoint = tmp_path / "logs" / "rl" / "ppo-smoke" / "official_policy_latest.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authenticated descendant fixture")
    checkpoint.with_name(f"{checkpoint.stem}.lineage.json").write_text("{}", encoding="utf-8")
    verified = {
        "checkpoint_sha256": "a" * 64,
        "bc_ancestor_checkpoint_sha256": "b" * 64,
        "verification": "ppo_checkpoint_and_authenticated_bc_ancestry",
    }
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path), \
         patch("src.layer4.hub.official_run_manager.verify_bc_checkpoint", return_value=None), \
         patch("src.layer4.hub.official_run_manager.verify_ppo_warmstart", return_value=None), \
         patch("src.layer4.hub.official_run_manager.verify_ppo_lineage_checkpoint", return_value=verified) as verify:
        config = normalize_official_config({
            "automatic_evaluation_mode": "training_probe",
            "resume": str(checkpoint),
        })
    assert config["resume"] == "/runs/rl/ppo-smoke/official_policy_latest.zip"
    verify.assert_called_once_with(checkpoint.resolve())


def test_resume_rejects_present_but_invalid_ppo_lineage(tmp_path):
    checkpoint = tmp_path / "logs" / "rl" / "ppo-smoke" / "official_policy_latest.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_name(f"{checkpoint.stem}.lineage.json").write_text("{}", encoding="utf-8")
    with patch("src.layer4.hub.official_run_manager.ROOT", tmp_path), \
         patch("src.layer4.hub.official_run_manager.verify_ppo_lineage_checkpoint", return_value=None):
        with pytest.raises(ValueError, match="invalid or tampered PPO lineage"):
            normalize_official_config({"resume": str(checkpoint)})


@pytest.mark.parametrize(
    ("attempt", "readiness"),
    [
        ({
            "stop_reason": "race_complete", "score_status": "valid",
            "valid_for_comparison": True, "race_time_s": 30.0,
            "adjusted_race_time_s": 31.0, "warmup_lap_times_s": [12.0],
            "race_lap_times_s": [10.0] * 3, "race_collisions": 0,
            "raw_collision_count_at_finish": 0, "frontier_projection_valid": True,
            "frontier_progress_m": 250.0, "frontier_progress_fraction": 0.3,
        }, True),
        ({
            "stop_reason": "collision_detected", "valid_for_comparison": False,
            "warmup_lap_times_s": [], "race_lap_times_s": [],
            "raw_collision_count_at_finish": 1,
        }, False),
        ({
            "stop_reason": "frontier_telemetry_invalid", "valid_for_comparison": False,
            "warmup_lap_times_s": [12.0], "race_lap_times_s": [],
            "raw_collision_count_at_finish": 0, "frontier_projection_valid": False,
        }, False),
    ],
)
def test_training_probe_authenticates_lineage_and_never_scores_or_promotes(
    manager, tmp_path, attempt, readiness,
):
    checkpoint = _checkpoint(tmp_path)
    report_dir = tmp_path / "probe-run"
    report_dir.mkdir()
    row = {"run_id": "probe-run", "output_dir": str(report_dir)}
    created = []

    def create_probe_run(*args, **kwargs):
        created.append((args, kwargs))
        (report_dir / "evaluation.json").write_text(json.dumps({"runs": [attempt]}))
        return row

    manager.create_run = create_probe_run
    manager.get_run = lambda run_id: {**row, "state": "completed"}
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "evaluations.json",
        champion_dir=tmp_path / "best_official_model", model_root=tmp_path / "logs" / "rl",
    )
    try:
        with patch("src.layer4.hub.official_evaluation_service.verify_training_probe_checkpoint", return_value={"verification": "verified"}):
            queued = service.submit(
                checkpoint, source="automatic", parent_run_id="training-run",
                validation_mode="training_probe",
            )
        assert queued["attempts_required"] == 1
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = service.status()
            if status["state"] in {"complete", "incomplete", "failed"}:
                break
            time.sleep(0.01)
        result = service.status()["evaluation"]
        assert result["protocol"] == "official_training_probe_1_plus_3"
        assert result["race_laps_required"] == 3
        assert result["completed_attempts"] == 1
        assert result["competitive_attempts"] == 0
        assert result["attempts_required"] == 1
        assert result["valid"] is False
        assert result["competitive_score_valid"] is False
        assert result["readiness_passed"] is readiness
        assert result["runs"][0]["valid_for_comparison"] is False
        assert created[0][0][0]["race_laps"] == 3
        options = created[0][1]["evaluation_options"]
        assert options["validation_mode"] == "training_probe"
        assert options["lap_progress_timeout_s"] == 240
        assert options["lap_progress_wall_timeout_s"] == 240
        assert options["frontier_stall_after_s"] == 5
        assert service.champion() is None
    finally:
        service.close()


def test_manual_training_probe_requires_verified_ppo_lineage(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    def parked_worker(self):
        while not self._worker_stop:
            time.sleep(0.01)

    with patch.object(OfficialEvaluationService, "_worker_loop", parked_worker):
        service = OfficialEvaluationService(
            manager, registry_path=tmp_path / "evaluations.json",
            champion_dir=tmp_path / "best_official_model", model_root=tmp_path / "logs" / "rl",
        )
    try:
        with patch("src.layer4.hub.official_evaluation_service.verify_ppo_lineage_checkpoint", return_value={
            "verification": "ppo_checkpoint_and_authenticated_bc_ancestry",
            "bc_ancestor_checkpoint_sha256": "a" * 64,
        }) as verify:
            queued = service.submit(
                checkpoint, source="manual", validation_mode="training_probe"
            )
            verify.assert_called_once_with(checkpoint.resolve())
            assert queued["attempts_required"] == 1
            assert queued["validation_mode"] == "training_probe"
        with patch("src.layer4.hub.official_evaluation_service.verify_ppo_lineage_checkpoint", return_value=None):
            with pytest.raises(ValueError, match="authenticated PPO checkpoint"):
                service.submit(checkpoint, source="manual", validation_mode="training_probe")
        with patch("src.layer4.hub.official_evaluation_service.verify_training_probe_checkpoint", return_value=None):
            with pytest.raises(ValueError, match="authenticated PPO checkpoint"):
                service.submit(
                    checkpoint, source="automatic", parent_run_id="training-run",
                    validation_mode="training_probe",
                )
        assert service.status()["attempts_required"] == 1
    finally:
        service.close()


def test_manual_training_probe_request_is_exposed_by_replay_api(tmp_path, monkeypatch):
    from src.layer4.hub import app as hub_app

    checkpoint = tmp_path / "logs" / "rl" / "ppo-run" / "official_policy_latest.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"updated PPO checkpoint")

    class RequestStub:
        async def json(self):
            return {
                "model_id": "ppo-run/official_policy_latest.zip",
                "map_id": "none", "seed": 0, "device": "cpu",
                "validation_mode": "training_probe",
            }

    class ReplayStub:
        submitted = None

        def start(self, **kwargs):
            self.submitted = kwargs
            return {"state": "queued", "validation_mode": kwargs["validation_mode"]}

    replay = ReplayStub()
    monkeypatch.setattr(hub_app, "ROOT", tmp_path)
    monkeypatch.setattr(hub_app, "replay_job", replay)
    result = asyncio.run(hub_app.replay_start(RequestStub()))
    assert result["validation_mode"] == "training_probe"
    assert replay.submitted["model_path"] == checkpoint.resolve()
    assert replay.submitted["validation_mode"] == "training_probe"


def test_manual_training_probe_runs_one_noncompetitive_1_plus_3_attempt(manager, tmp_path):
    checkpoint = _checkpoint(tmp_path)
    report_dir = tmp_path / "manual-probe-run"
    report_dir.mkdir()
    row = {"run_id": "manual-probe-run", "output_dir": str(report_dir)}
    attempt = {
        "stop_reason": "race_complete", "score_status": "valid",
        "valid_for_comparison": True, "race_time_s": 30.0,
        "adjusted_race_time_s": 31.0, "warmup_lap_times_s": [12.0],
        "race_lap_times_s": [10.0] * 3, "race_collisions": 0,
        "raw_collision_count_at_finish": 0, "frontier_projection_valid": True,
    }
    created = []

    def create_probe_run(config, display_name, **kwargs):
        created.append((config, kwargs))
        (report_dir / "evaluation.json").write_text(json.dumps({"runs": [attempt]}))
        return row

    manager.create_run = create_probe_run
    manager.get_run = lambda run_id: {**row, "state": "completed"}
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "evaluations.json",
        champion_dir=tmp_path / "best_official_model", model_root=tmp_path / "logs" / "rl",
    )
    try:
        with patch("src.layer4.hub.official_evaluation_service.verify_ppo_lineage_checkpoint", return_value={
            "verification": "ppo_checkpoint_and_authenticated_bc_ancestry",
            "bc_ancestor_checkpoint_sha256": "a" * 64,
        }):
            queued = service.submit(checkpoint, source="manual", validation_mode="training_probe")
        assert queued["attempts_required"] == 1
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = service.status()
            if status["state"] in {"complete", "incomplete", "failed"}:
                break
            time.sleep(0.01)
        result = service.status()["evaluation"]
        assert result["protocol"] == "official_training_probe_1_plus_3"
        assert result["readiness_passed"] is True
        assert result["competitive_attempts"] == 0
        assert result["competitive_score_valid"] is False
        assert result["runs"][0]["valid_for_comparison"] is False
        assert created[0][0]["race_laps"] == 3
        assert created[0][1]["evaluation_options"]["validation_mode"] == "training_probe"
        assert created[0][1]["evaluation_options"]["lap_progress_timeout_s"] == 240
        assert created[0][1]["evaluation_options"]["lap_progress_wall_timeout_s"] == 240
        assert created[0][1]["evaluation_options"]["frontier_stall_after_s"] == 5
        assert service.champion() is None
    finally:
        service.close()


@pytest.mark.parametrize(
    ("configured_mode", "expected_validation_mode"),
    [("competitive", None), ("training_probe", "training_probe")],
)
def test_automatic_evaluation_mode_comes_from_persisted_run_config(
    tmp_path, monkeypatch, configured_mode, expected_validation_mode,
):
    from src.layer4.hub import app as hub_app

    run_dir = tmp_path / "official-run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "automatic_evaluation_100_steps.zip"
    checkpoint.write_bytes(b"automatic checkpoint")

    class RequestStub:
        async def json(self):
            # The client cannot select a stronger validation mode than the
            # authenticated mode persisted when training started.
            return {
                "model_path": str(checkpoint), "step": 100, "episode": 0,
                "trigger": "steps", "validation_mode": "bc_diagnostic",
            }

    class RunStub:
        def get_run(self, run_id):
            return {
                "run_id": run_id, "kind": "training", "output_dir": str(run_dir),
                "config": {"automatic_evaluation_mode": configured_mode},
            }

    class ReplayStub:
        submitted = None

        def start_automatic(self, **kwargs):
            self.submitted = kwargs
            return {"state": "queued"}

    replay = ReplayStub()
    monkeypatch.setattr(hub_app, "official_runs", RunStub())
    monkeypatch.setattr(hub_app, "replay_job", replay)
    monkeypatch.setattr(hub_app, "ROOT", tmp_path)
    asyncio.run(hub_app.queue_automatic_evaluation("run-1", RequestStub()))
    assert replay.submitted["validation_mode"] == expected_validation_mode
