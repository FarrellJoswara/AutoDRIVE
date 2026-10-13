"""Final race validation against real competitive route-monitor report fields."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.layer4.hub.campaign_phase_runner import validate_final_evaluation
from src.layer3 import official_evaluate


MODEL_SHA = "b" * 64
SCENE_SHA = "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674"
ROUTE_SHA = "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"
EXE = r"C:\Users\farre\Documents\Github\AiCar\aicar-official-iros-camera\windows\autodrive_simulator\AutoDRIVE Simulator.exe"


def _attempt(run_id: str) -> dict:
    identity = {
        "pid": 4100 + int(run_id[-1]),
        "process_creation_utc": "2026-10-13T12:00:00Z",
        "executable_path": EXE,
        "port": 58000 + int(run_id[-1]),
        "manifest_sha256": (str(int(run_id[-1])) * 64),
    }
    return {
        "run_id": run_id,
        "complete": True,
        "stop_reason": "race_complete",
        "score_status": "valid",
        "valid_for_comparison": True,
        "race_laps_completed": 10,
        "race_laps_required": 10,
        "warmup_laps_completed": 1,
        "warmup_lap_times_s": [12.1],
        "race_lap_times_s": [18.1 + i / 10 for i in range(10)],
        "collision_count": 0,
        "race_collisions": 0,
        "warmup_collisions": 0,
        "raw_collision_count_at_finish": 0,
        "camera_sample_count": 100,
        "camera_fresh_sample_count": 100,
        "camera_stale_sample_count": 0,
        "camera_missing_sample_count": 0,
        "camera_freshness_unknown_count": 0,
        "frontier_projection_valid": True,
        "route_projection_valid": True,
        "route_monitor_only": True,
        "route_monitor": {
            "projection_valid": True,
            "projection_sample_count": 100,
            "projection_valid_sample_count": 100,
            "projection_invalid_sample_count": 0,
            "scoring_or_stop_effect": False,
        },
        "official_scene_sha256": SCENE_SHA,
        "route_sha256": ROUTE_SHA,
        "official_scene_path": r"C:\sim\Data\level0",
        "route_path": r"C:\repo\competition\iros2026\tracks\competition\official_centerline.csv",
        "track_profile": "competition",
        "simulator_identity": identity,
    }


def _records(root: Path) -> tuple[dict, list[Path]]:
    paths = []
    attempts = []
    for i in range(1, 4):
        run_id = f"ab12345678{i}"
        run_dir = root / "logs" / "rl" / f"official_run_{run_id}"
        run_dir.mkdir(parents=True)
        report_path = run_dir / "evaluation.json"
        attempt = _attempt(run_id)
        report = {
            "model_sha256": MODEL_SHA,
            "validation_mode": None,
            "protocol": "official_competitive_1_plus_10",
            "competitive_score_valid": True,
            "route_monitor_only": True,
            "official_scene_sha256": SCENE_SHA,
            "route_sha256": ROUTE_SHA,
            "official_scene_path": attempt["official_scene_path"],
            "route_path": attempt["route_path"],
            "track_profile": "competition",
            "simulator_identity": copy.deepcopy(attempt["simulator_identity"]),
            "runs": [attempt],
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(report_path)
        attempts.append({
            "run_id": run_id,
            "evaluation_report": str(report_path),
            "race_laps_completed": 10,
            "warmup_laps": 1,
            "collision_count": 0,
            "race_collisions": 0,
            "camera_stale_sample_count": 0,
            "camera_missing_sample_count": 0,
            "camera_freshness_unknown_count": 0,
            "valid_for_comparison": True,
            "frontier_projection_valid": True,
            "route_projection_valid": True,
            "route_monitor": copy.deepcopy(attempt["route_monitor"]),
            "route_monitor_only": True,
            "official_scene_sha256": SCENE_SHA,
            "route_sha256": ROUTE_SHA,
            "official_scene_path": attempt["official_scene_path"],
            "route_path": attempt["route_path"],
            "track_profile": "competition",
            "simulator_identity": copy.deepcopy(attempt["simulator_identity"]),
        })
    result = {
        "model_sha256": MODEL_SHA,
        "protocol": "official_10_lap_race",
        "validation_mode": None,
        "attempts_required": 3,
        "competitive_score_valid": True,
        "runs": attempts,
    }
    record = {
        "state": "complete",
        "evaluation_id": "eval-route-monitor",
        "source": "manual",
        "model_sha256": MODEL_SHA,
        "result": result,
    }
    return record, paths


def test_final_gate_accepts_standard_three_by_one_plus_ten_with_authenticated_monitor_fields(tmp_path):
    record, reports = _records(tmp_path)
    result = validate_final_evaluation(record, reports, expected_sha256=MODEL_SHA)
    assert result["passed"], result["errors"]
    assert all(item["scored_laps"] == 10 for item in result["attempts"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("official_scene_sha256", "0" * 64),
        ("route_sha256", "0" * 64),
        ("official_scene_path", r"C:\other\Data\level0"),
        ("route_path", r"C:\other\official_centerline.csv"),
        ("track_profile", "practice"),
        ("route_monitor_only", False),
    ],
)
def test_final_gate_rejects_wrong_scene_route_or_profile(tmp_path, path, value):
    record, reports = _records(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    report[path] = value
    reports[0].write_text(json.dumps(report), encoding="utf-8")
    result = validate_final_evaluation(record, reports, expected_sha256=MODEL_SHA)
    assert not result["passed"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("projection_valid", False),
        ("projection_sample_count", 0),
        ("projection_invalid_sample_count", 1),
        ("scoring_or_stop_effect", True),
    ],
)
def test_final_gate_rejects_unusable_or_score_affecting_route_monitor(tmp_path, field, value):
    record, reports = _records(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    report["runs"][0]["route_monitor"][field] = value
    reports[0].write_text(json.dumps(report), encoding="utf-8")
    result = validate_final_evaluation(record, reports, expected_sha256=MODEL_SHA)
    assert not result["passed"]


def test_final_gate_rejects_changed_simulator_process_identity(tmp_path):
    record, reports = _records(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    report["runs"][0]["simulator_identity"]["executable_path"] = r"C:\other\simulator.exe"
    reports[0].write_text(json.dumps(report), encoding="utf-8")
    result = validate_final_evaluation(record, reports, expected_sha256=MODEL_SHA)
    assert not result["passed"]


def test_final_gate_rejects_service_report_identity_disagreement(tmp_path):
    record, reports = _records(tmp_path)
    record["result"]["runs"][0]["simulator_identity"]["pid"] += 1
    result = validate_final_evaluation(record, reports, expected_sha256=MODEL_SHA)
    assert not result["passed"]


def test_final_gate_rejects_missing_monitor_identity_for_old_null_projection_report(tmp_path):
    record, reports = _records(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    run = report["runs"][0]
    run["frontier_projection_valid"] = None
    run.pop("route_monitor", None)
    run.pop("route_monitor_only", None)
    run.pop("official_scene_sha256", None)
    run.pop("route_sha256", None)
    run.pop("simulator_identity", None)
    reports[0].write_text(json.dumps(report), encoding="utf-8")
    result = validate_final_evaluation(record, reports, expected_sha256=MODEL_SHA)
    assert not result["passed"]


def test_powershell_bom_manifest_identity_roundtrips_into_competitive_reports_and_gate(
    tmp_path, monkeypatch,
):
    from src.layer3 import official_tracks

    monkeypatch.setattr(
        official_tracks, "verify_windows_scene",
        lambda _route, executable=None: {
            "scene_sha256": SCENE_SHA,
            "scene_path": r"C:\Installed Simulator\Data\level0",
            "track_profile": "competition", "verified": True,
        },
    )
    record, report_paths = _records(tmp_path)
    expected_route = (
        Path(official_evaluate.__file__).resolve().parents[2]
        / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"
    ).resolve()
    raw_hashes = []
    for index, report_path in enumerate(report_paths):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        attempt = report["runs"][0]
        run_id = attempt["run_id"]
        manifest = {
            "run_id": run_id,
            "records": [{
                "pid": 4200 + index,
                "process_creation_utc": f"2026-10-13T12:10:0{index}Z",
                "executable_path": EXE,
                "port": 58100 + index,
                "batch_mode": False,
            }],
        }
        raw = b"\xef\xbb\xbf" + json.dumps(manifest).encode("utf-8")
        manifest_path = report_path.parent / "windows_simulator_processes.json"
        manifest_path.write_bytes(raw)
        raw_hash = hashlib.sha256(raw).hexdigest()
        raw_hashes.append(raw_hash)
        identity = official_evaluate._route_monitor_identity(
            output_path=report_path,
            frontier_path=expected_route,
            expected_run_id=run_id,
        )
        assert identity["simulator_identity"]["manifest_sha256"] == raw_hash
        assert identity["route_sha256"] == ROUTE_SHA
        assert identity["official_scene_sha256"] == SCENE_SHA
        for target in (report, attempt):
            target.update(identity)
            target["route_monitor"] = copy.deepcopy(_attempt(run_id)["route_monitor"])
            target["frontier_projection_valid"] = True
            target["route_projection_valid"] = True
        report_path.write_text(json.dumps(report), encoding="utf-8")
        service_attempt = record["result"]["runs"][index]
        service_attempt.update(identity)
        service_attempt["route_monitor"] = copy.deepcopy(attempt["route_monitor"])
        service_attempt["frontier_projection_valid"] = True
        service_attempt["route_projection_valid"] = True

    checked = validate_final_evaluation(record, report_paths, expected_sha256=MODEL_SHA)
    assert checked["passed"], checked["errors"]
    assert len(set(raw_hashes)) == 3


def test_powershell_manifest_rejects_invalid_utf8_without_replacement(tmp_path, monkeypatch):
    from src.layer3 import official_tracks

    monkeypatch.setattr(
        official_tracks, "verify_windows_scene",
        lambda _route, executable=None: {
            "scene_sha256": SCENE_SHA, "scene_path": r"C:\sim\Data\level0",
            "track_profile": "competition", "verified": True,
        },
    )
    report_path = tmp_path / "official_run_bom123" / "evaluation.json"
    report_path.parent.mkdir(parents=True)
    (report_path.parent / "windows_simulator_processes.json").write_bytes(b"\xef\xbb\xbf{\xff}")
    route = (
        Path(official_evaluate.__file__).resolve().parents[2]
        / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"
    ).resolve()
    with pytest.raises(UnicodeDecodeError):
        official_evaluate._route_monitor_identity(
            output_path=report_path, frontier_path=route, expected_run_id="bom123",
        )
