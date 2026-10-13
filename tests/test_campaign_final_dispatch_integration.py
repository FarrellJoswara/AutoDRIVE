from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

import src.layer4.hub.campaign_phase_runner as runner_module
from src.layer4.hub.official_evaluation_service import OfficialEvaluationService


SHA = "a" * 64


class RecordingAPI:
    def __init__(self):
        self.posts: list[tuple[str, dict]] = []
        self.health = {
            "status": "ok", "hub_pid": 222, "hub_start_identity": "new-start",
            "source_fingerprint": "build-sha", "source_files_sha256": {"app.py": "f" * 64},
        }

    def get(self, route: str):
        if route == "/api/replay/status":
            return {"state": "idle"}
        if route == "/health":
            return self.health
        raise AssertionError(f"unexpected GET {route}")

    def post(self, route: str, payload: dict):
        self.posts.append((route, payload))
        return {"evaluation_id": "evaluation-final-1"}


def _runner(tmp_path: Path) -> tuple[runner_module.CampaignPhaseRunner, RecordingAPI]:
    api = RecordingAPI()
    runner = runner_module.CampaignPhaseRunner(
        state_path=tmp_path / "state.json", repo_root=tmp_path, api=api,
    )
    runner.output_dir.mkdir(parents=True)
    runner.finalists_manifest.parent.mkdir(parents=True, exist_ok=True)
    runner.finalists_manifest.write_text('{"sealed":true}', encoding="utf-8")
    candidate = tmp_path / "logs" / "rl" / "official_run_candidate" / "checkpoints" / "candidate.zip"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"candidate checkpoint")
    runner._finalist_manifest_rows = lambda: ("b" * 64, [{
        "model_path": str(candidate), "checkpoint_sha256": SHA,
    }])
    return runner, api


@pytest.mark.parametrize(
    ("snapshot", "expected_action"),
    [
        (
            {"runs": [{"run_id": "trainer-1", "state": "running"}],
             "replay": {"state": "idle"}, "evaluation_history": []},
            "stop_training_at_freeze",
        ),
        (
            {"runs": [], "replay": {"state": "running"}, "evaluation_history": []},
            "wait_for_existing_evaluation",
        ),
        (
            {"runs": [], "replay": {"state": "idle"},
             "evaluation_history": [{"evaluation_id": "history-1", "state": "queued"}]},
            "wait_for_evaluation_history_reconciliation",
        ),
    ],
)
def test_final_dispatch_waits_for_training_replay_and_history_quiescence(
    tmp_path: Path, monkeypatch, snapshot: dict, expected_action: str,
):
    runner, api = _runner(tmp_path)
    stopped: list[str] = []
    monkeypatch.setattr(runner, "_stop_run", lambda run_id, *_args: stopped.append(run_id))

    result = runner._final_phase_tick(
        {"training_freeze_at_utc": "2026-10-13T08:26:44Z"}, snapshot, 10_000.0,
    )

    assert result["action"] == expected_action
    assert api.posts == []
    if expected_action == "stop_training_at_freeze":
        assert stopped == ["trainer-1"]


def test_final_dispatch_posts_only_selected_checkpoint_to_standard_manual_replay(
    tmp_path: Path,
):
    runner, api = _runner(tmp_path)
    state = {"training_freeze_at_utc": "2026-10-13T08:26:44Z"}
    idle_snapshot = {"runs": [], "replay": {"state": "idle"}, "evaluation_history": []}

    result = runner._final_phase_tick(state, idle_snapshot, 10_000.0)

    assert result == {
        "action": "finalist_evaluation_started",
        "evaluation_id": "evaluation-final-1",
        "checkpoint_sha256": SHA,
        "manifest_sha256": "b" * 64,
    }
    assert len(api.posts) == 1
    route, payload = api.posts[0]
    assert route == "/api/replay/start"
    assert payload == {
        "model_id": "official_run_candidate/checkpoints/candidate.zip",
        "map_id": "none", "device": "cpu", "trace_steps": 0,
    }
    assert payload.get("validation_mode") is None


def test_standard_manual_replay_queues_three_competitive_attempts_by_default(
    tmp_path: Path, monkeypatch,
):
    class FakeManager:
        def list_runs(self):
            return {"runs": []}

    model_root = tmp_path / "logs" / "rl"
    model_dir = model_root / "official_run_candidate"
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"runtime":"official IROS 2026 runtime",'
        '"observation_profile":"official_sensors_camera",'
        '"actions":{"throttle_mode":"bidirectional",'
        '"negative_throttle_mode":"allow","steering_mode":"normal",'
        '"steering_action_scale":1.0,"straight_throttle_gain":1.0,'
        '"straight_throttle_steering_threshold":0.15}}',
        encoding="utf-8",
    )
    checkpoint = checkpoints / "candidate.zip"
    checkpoint.write_bytes(b"official model bytes")
    monkeypatch.setenv("AICAR_CAMPAIGN_STATE_PATH", str(tmp_path / "inactive-campaign.json"))
    monkeypatch.setattr(
        OfficialEvaluationService, "_run_batch",
        lambda _self, job: job.update(state="incomplete"),
    )
    service = OfficialEvaluationService(
        FakeManager(), registry_path=tmp_path / "evaluations.json",
        champion_dir=tmp_path / "champion", model_root=model_root,
    )
    try:
        queued = service.submit(checkpoint, source="manual", validation_mode=None)
        assert queued["source"] == "manual"
        assert queued["validation_mode"] is None
        assert queued["attempts_required"] == 3
    finally:
        service.close()


def _competitive_record(root: Path, expected_sha: str, evaluation_id: str,
                        run_ids: list[str], lap_values: list[float]) -> tuple[dict, list[Path]]:
    reports: list[Path] = []
    service_attempts = []
    for index, run_id in enumerate(run_ids):
        report_path = root / "logs" / "rl" / f"official_run_{run_id}" / "evaluation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        laps = lap_values[index * 10:(index + 1) * 10]
        identity = {
            "pid": 4100 + index,
            "process_creation_utc": f"2026-10-13T09:00:0{index}Z",
            "executable_path": r"C:\Users\farre\Documents\Github\AiCar\aicar-official-iros-camera\windows\autodrive_simulator\AutoDRIVE Simulator.exe",
            "port": 58000 + index,
            "manifest_sha256": str(index + 1) * 64,
        }
        scene_sha = runner_module.OFFICIAL_COMPETITION_SCENE_SHA256
        route_sha = runner_module.OFFICIAL_COMPETITION_ROUTE_SHA256
        scene_path = r"C:\simulator\Data\level0"
        route_path = r"C:\repo\competition\iros2026\tracks\competition\official_centerline.csv"
        route_monitor = {
            "projection_valid": True,
            "projection_sample_count": 100,
            "projection_valid_sample_count": 100,
            "projection_invalid_sample_count": 0,
            "scoring_or_stop_effect": False,
        }
        report = {
            "model_sha256": expected_sha,
            "validation_mode": None,
            "protocol": "official_competitive_1_plus_10",
            "competitive_score_valid": True,
            "route_monitor_only": True,
            "official_scene_sha256": scene_sha,
            "route_sha256": route_sha,
            "official_scene_path": scene_path,
            "route_path": route_path,
            "track_profile": "competition",
            "simulator_identity": identity,
            "runs": [{
                "run_id": run_id, "stop_reason": "race_complete",
                "complete": True, "score_status": "valid", "valid_for_comparison": True,
                "race_laps_completed": 10, "race_laps_required": 10, "warmup_laps_completed": 1,
                "frontier_projection_valid": True, "route_projection_valid": True,
                "route_monitor_only": True, "route_monitor": route_monitor,
                "official_scene_sha256": scene_sha, "route_sha256": route_sha,
                "official_scene_path": scene_path, "route_path": route_path,
                "track_profile": "competition", "simulator_identity": identity,
                "collision_count": 0,
                "race_collisions": 0, "warmup_collisions": 0,
                "raw_collision_count_at_finish": 0,
                "camera_sample_count": 100, "camera_fresh_sample_count": 100,
                "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
                "camera_freshness_unknown_count": 0,
                "race_lap_times_s": laps, "warmup_lap_times_s": [35.0],
            }],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        reports.append(report_path)
        service_attempts.append({
            "run_id": run_id, "evaluation_report": str(report_path),
            "race_laps_completed": 10, "warmup_laps": 1,
            "collision_count": 0, "race_collisions": 0,
            "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
            "camera_freshness_unknown_count": 0, "valid_for_comparison": True,
            "frontier_projection_valid": True, "route_projection_valid": True,
            "route_monitor_only": True, "route_monitor": route_monitor,
            "official_scene_sha256": scene_sha, "route_sha256": route_sha,
            "official_scene_path": scene_path, "route_path": route_path,
            "track_profile": "competition", "simulator_identity": identity,
        })
    record = {
        "state": "complete", "evaluation_id": evaluation_id, "source": "manual",
        "model_sha256": expected_sha,
        "result": {
            "model_sha256": expected_sha, "validation_mode": None,
            "protocol": "official_10_lap_race", "attempts_required": 3,
            "competitive_score_valid": True, "runs": service_attempts,
        },
    }
    return record, reports


def _final_dispatch_fixture(tmp_path: Path, monkeypatch):
    api = RecordingAPI()
    state_path = tmp_path / "campaign-state.json"
    state = {
        "campaign_id": "campaign-test", "status": "active",
        "started_at_utc": "2026-10-12T23:26:44Z",
        "training_freeze_at_utc": "2026-10-13T08:26:44Z",
        "hard_deadline_at_utc": "2026-10-13T11:26:44Z",
        "start_monotonic_ns": 100, "training_freeze_monotonic_ns": 200,
        "hard_deadline_monotonic_ns": 300, "boot_id": "boot-test",
        "gate_evidence_canonical_sha256": "3" * 64,
        "updated_policy_probe_checkpoint_sha256": "4" * 64,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    runner = runner_module.CampaignPhaseRunner(
        state_path=state_path, repo_root=tmp_path, api=api,
    )
    runner.output_dir.mkdir(parents=True)
    runner.finalists_manifest.parent.mkdir(parents=True, exist_ok=True)
    runner.finalists_manifest.write_text('{"sealed":true}', encoding="utf-8")

    # The first candidate has a slow outlier: its mean is slower than the
    # second candidate, while its median across all 30 scored laps is faster.
    candidate_rows = []
    result_events = []
    for label, sha, laps in (
        ("median-winner", "1" * 64, [33.0] * 29 + [500.0]),
        ("mean-winner", "2" * 64, [34.0] * 30),
    ):
        checkpoint = tmp_path / "logs" / "rl" / f"official_run_{label}" / "checkpoints" / "policy.zip"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(label.encode())
        candidate_rows.append({"model_path": str(checkpoint), "checkpoint_sha256": sha})
        run_ids = [f"{label}-{index}" for index in range(1, 4)]
        record, report_paths = _competitive_record(tmp_path, sha, f"eval-{label}", run_ids, laps)
        result = runner_module.validate_final_evaluation(record, report_paths, expected_sha256=sha)
        assert result["passed"], result["errors"]
        artifact_path = runner.output_dir / "final_results" / f"result_{label}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(result), encoding="utf-8")
        result_events.append({
            "event": "finalist_evaluation_result", "evaluation_id": f"eval-{label}",
            "timestamp_utc": "2026-10-13T09:00:00+00:00", "manifest_sha256": "b" * 64,
            "checkpoint_sha256": sha, "passed": True, "artifact_path": str(artifact_path),
        })
    runner._finalist_manifest_rows = lambda: ("b" * 64, candidate_rows)
    runner._pending_final_evaluation = lambda _snapshot: None
    runner._create_finalist_manifest_if_ready = lambda *_args: None
    runner.events_path.parent.mkdir(parents=True, exist_ok=True)
    runner.events_path.write_text("".join(json.dumps(event) + "\n" for event in result_events), encoding="utf-8")

    winner = runner._winning_finalist("b" * 64)
    assert winner is not None
    assert winner["checkpoint_sha256"] == "1" * 64
    assert winner["scored_lap_count"] == 30

    health_identity = dict(api.health)
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    clean_runtime = {
        "campaign_id": state["campaign_id"], "winner_checkpoint_sha256": winner["checkpoint_sha256"],
        "primary_evaluation_id": winner["primary_evaluation_id"],
        "primary_result_sha256": winner["primary_result_sha256"],
        "training_idle": True, "replay_idle": True, "evaluation_history_idle": True,
        "simulators_retired": True, "health_ok": True, "source_loaded": True,
        "old_hub_pid": 111, "new_hub_pid": api.health["hub_pid"],
        "old_hub_start_identity": "old-start", "new_hub_start_identity": api.health["hub_start_identity"],
        "hub_commandline": "python -m src.layer4.hub.app", "source_fingerprint": api.health["source_fingerprint"],
        "official_scene_sha256": "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674",
        "route_sha256": "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb",
        "campaign_state_sha256": state_sha, "health_identity": health_identity,
        "campaign_clock_identity": {
            key: state.get(key) for key in (
                "campaign_id", "started_at_utc", "training_freeze_at_utc", "hard_deadline_at_utc",
                "start_monotonic_ns", "training_freeze_monotonic_ns", "hard_deadline_monotonic_ns",
                "boot_id", "gate_evidence_canonical_sha256", "updated_policy_probe_checkpoint_sha256",
            )
        },
        "recorded_at_utc": "2026-10-13T09:00:01+00:00",
    }
    runner.clean_reload_runtime.write_text(json.dumps(clean_runtime), encoding="utf-8")
    return runner, api, state


def test_validated_reports_select_median_winner_then_dispatch_one_clean_reload_probe(
    tmp_path: Path, monkeypatch,
):
    runner, api, state = _final_dispatch_fixture(tmp_path, monkeypatch)
    result = runner._final_phase_tick(
        state, {"runs": [], "replay": {"state": "idle"}, "evaluation_history": []}, 10_000.0,
    )

    assert result["action"] == "winner_clean_reload_probe_started"
    assert result["winner_checkpoint_sha256"] == "1" * 64
    assert len(api.posts) == 1
    route, payload = api.posts[0]
    assert route == "/api/replay/start"
    assert payload == {
        "model_id": "official_run_median-winner/checkpoints/policy.zip",
        "map_id": "none", "device": "cpu", "trace_steps": 0,
        "validation_mode": "training_probe",
    }


@pytest.mark.parametrize("collision_field", ["warmup_collisions", "raw_collision_count_at_finish"])
def test_final_evaluation_rejects_collisions_outside_scored_laps(tmp_path: Path, collision_field: str):
    sha = "a" * 64
    record, report_paths = _competitive_record(
        tmp_path, sha, "evaluation-warmup-collision", ["warmup-1", "warmup-2", "warmup-3"],
        [33.0] * 30,
    )
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    report["runs"][0][collision_field] = 1
    # Deliberately simulate an inconsistent service summary: the authoritative
    # attempt report must still disqualify the finalist.
    report_paths[0].write_text(json.dumps(report), encoding="utf-8")

    result = runner_module.validate_final_evaluation(record, report_paths, expected_sha256=sha)
    assert not result["passed"]
    assert any("zero-collision" in error for error in result["errors"])


def test_clean_reload_probe_completion_survives_append_only_campaign_state_change(
    tmp_path: Path, monkeypatch,
):
    runner, api, state = _final_dispatch_fixture(tmp_path, monkeypatch)
    idle = {"runs": [], "replay": {"state": "idle"}, "evaluation_history": []}
    started = runner._final_phase_tick(state, idle, 10_000.0)
    assert started["action"] == "winner_clean_reload_probe_started"
    assert len(api.posts) == 1

    # The fake service updates normal mutable campaign state after accepting
    # the request. Immutable clock identity and winner binding remain stable.
    state_path = runner.state_path
    mutated_state = json.loads(state_path.read_text(encoding="utf-8"))
    mutated_state["revision"] = 2
    mutated_state["events"] = [{"event": "probe_queued", "evaluation_id": "evaluation-final-1"}]
    state_path.write_text(json.dumps(mutated_state), encoding="utf-8")

    run_id = "clean-probe-run"
    report_path = tmp_path / "logs" / "rl" / f"official_run_{run_id}" / "evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    winner_sha = "1" * 64
    report = {
        "model_sha256": winner_sha, "protocol": "official_training_probe_1_plus_3",
        "validation_mode": "training_probe",
        "runs": [{
            "stop_reason": "race_complete", "frontier_projection_valid": True,
            "collision_count": 0, "race_collisions": 0,
            "race_laps_completed": 3, "race_laps_required": 3,
            "camera_sample_count": 10, "camera_fresh_sample_count": 10,
            "camera_stale_sample_count": 0, "camera_missing_sample_count": 0,
            "camera_freshness_unknown_count": 0,
            "race_lap_times_s": [33.0, 33.1, 33.2], "warmup_lap_times_s": [34.0],
        }],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    history = [{
        "evaluation_id": "evaluation-final-1", "state": "complete", "source": "manual",
        "validation_mode": "training_probe", "model_sha256": winner_sha,
        "updated_utc": "2026-10-13T09:01:00+00:00",
        "result": {
            "readiness_passed": True,
            "runs": [{
                "run_id": run_id, "score_status": "valid", "warmup_laps": 1,
                "race_laps_completed": 3, "race_laps_required": 3,
                "collision_count": 0, "race_collisions": 0, "readiness_passed": True,
                "evaluation_report": str(report_path),
            }],
        },
    }]
    completed = runner._final_phase_tick(
        mutated_state,
        {"runs": [], "replay": {"state": "complete", "evaluation_id": "evaluation-final-1"},
         "evaluation_history": history},
        9_000.0,
    )
    assert completed["action"] == "winner_clean_reload_probe_recorded"
    assert completed["competitive_score"] is False
    assert len(api.posts) == 1


@pytest.mark.parametrize("bad_field", ["campaign_state_sha256", "campaign_clock_identity",
                                        "recorded_at_utc", "winner_checkpoint_sha256"])
def test_clean_reload_probe_refuses_mismatched_or_stale_runtime_evidence(
    tmp_path: Path, monkeypatch, bad_field: str,
):
    runner, api, state = _final_dispatch_fixture(tmp_path, monkeypatch)
    payload = json.loads(runner.clean_reload_runtime.read_text(encoding="utf-8"))
    if bad_field == "campaign_state_sha256":
        payload[bad_field] = "malformed"
    elif bad_field == "campaign_clock_identity":
        payload[bad_field] = {**payload[bad_field], "training_freeze_at_utc": "wrong"}
    elif bad_field == "recorded_at_utc":
        payload[bad_field] = "2026-10-13T09:00:00+00:00"
    else:
        payload[bad_field] = "9" * 64
    runner.clean_reload_runtime.write_text(json.dumps(payload), encoding="utf-8")

    result = runner._final_phase_tick(
        state, {"runs": [], "replay": {"state": "idle"}, "evaluation_history": []}, 10_000.0,
    )
    assert result["action"] == "clean_runtime_evidence_mismatch_fail_closed"
    assert api.posts == []


def test_clean_reload_probe_rechecks_live_health_identity(tmp_path: Path, monkeypatch):
    runner, api, state = _final_dispatch_fixture(tmp_path, monkeypatch)
    api.health = {**api.health, "source_fingerprint": "changed-after-recording"}

    result = runner._final_phase_tick(
        state, {"runs": [], "replay": {"state": "idle"}, "evaluation_history": []}, 10_000.0,
    )
    assert result["action"] == "clean_reload_hub_health_invalid"
    assert api.posts == []
