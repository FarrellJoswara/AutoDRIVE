import hashlib
import json
from pathlib import Path

import pytest

from src.layer4.hub.campaign_closeout import build_failure_index, publish_failure_index


CAMPAIGN = "test-campaign"


def _state():
    return {
        "campaign_id": CAMPAIGN,
        "events": [
            {"event": "campaign_experiment_invalidated", "experiment_id": "literal_cost_bad",
             "config_sha256": "a" * 64, "reason": "incorrect units"},
        ],
    }


def test_failure_index_binds_event_and_evidence_bytes_and_accounting_semantics():
    events = [
        {"event": "tick_error", "timestamp_utc": "t1", "error": "runner error"},
        {"event": "candidate_launch_aborted", "experiment_id": "bad-id", "config_sha256": "b" * 64},
        {"event": "tick", "action": "stop_training_on_runtime_fault", "run_id": "run1",
         "faults": ["sensor_fault"]},
        {"event": "finalist_evaluation_result", "passed": False, "evaluation_id": "final-eval"},
    ]
    ledger = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in events)
    snapshot = json.dumps({
        "probe_evidence": [{
            "evaluation_id": "failed-probe", "parent_run_id": "run2",
            "failure_classification": "infrastructure_failure",
            "report_path": "logs/rl/official_run_run2/evaluation.json",
            "report_sha256": "c" * 64,
        }],
    }, sort_keys=True).encode()
    index = build_failure_index(
        campaign_id=CAMPAIGN, event_ledger_path="actions.jsonl", event_ledger_bytes=ledger,
        evidence_documents=[("snapshot.json", snapshot)], campaign_state=_state(),
    )
    assert index["event_ledger"]["sha256"] == hashlib.sha256(ledger).hexdigest()
    assert index["evidence_documents"][0]["sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert {row["classification"] for row in index["failed_or_interrupted_events"]} >= {
        "runner_error", "candidate_launch_aborted", "failed_or_interrupted_action",
        "failed_competitive_final_evaluation", "infrastructure_failure",
    }
    assert index["invalidated_experiments"][0]["experiment_id"] == "literal_cost_bad"
    assert "not asserted" in index["optimizer_accounting"]["optimizer_step_calls"]
    assert index["ui_visual_review"]["status"] == "open"


@pytest.mark.parametrize("flag", ["final_results_reconciled", "training_idle", "evaluations_idle"])
@pytest.mark.parametrize("invalid", [False, 1, "true", None])
def test_failure_index_publication_requires_exact_true_closeout_flags(
    tmp_path: Path, flag: str, invalid,
):
    ledger = b'{"event":"tick_error","error":"kept"}\n'
    flags = {"final_results_reconciled": True, "training_idle": True, "evaluations_idle": True}
    flags[flag] = invalid
    with pytest.raises(RuntimeError, match="reconciled final results"):
        publish_failure_index(
            output_dir=tmp_path, campaign_id=CAMPAIGN, event_ledger_path="actions.jsonl",
            event_ledger_bytes=ledger, evidence_documents=[], campaign_state=_state(), **flags,
        )
    assert not (tmp_path / "campaign_actions_snapshot.jsonl").exists()


def test_failure_index_exposes_missing_and_supplied_ledger_evidence_references():
    referenced = "logs/rl/official_campaign/phase_runner/evidence/snapshot.json"
    ledger = (json.dumps({"event": "tick", "action": "observe_only", "evidence_path": referenced}) + "\n").encode()
    missing = build_failure_index(
        campaign_id=CAMPAIGN, event_ledger_path="actions.jsonl", event_ledger_bytes=ledger,
        evidence_documents=[], campaign_state=_state(),
    )
    assert missing["evidence_reference_closure_complete"] is False
    assert missing["unresolved_evidence_references"] == [{
        "path": referenced,
        "reason": "referenced by event ledger but not supplied to closeout index",
        "reference_source": "event_ledger:1:evidence_path",
    }]
    raw = b'{"probe_evidence": []}'
    complete = build_failure_index(
        campaign_id=CAMPAIGN, event_ledger_path="actions.jsonl", event_ledger_bytes=ledger,
        evidence_documents=[(referenced, raw)], campaign_state=_state(),
    )
    assert complete["evidence_reference_closure_complete"] is True
    assert complete["unresolved_evidence_references"] == []


def test_failure_index_publication_is_hash_bound_and_create_only(tmp_path: Path):
    ledger = b'{"event":"tick_error","error":"kept"}\n'
    result = publish_failure_index(
        output_dir=tmp_path, campaign_id=CAMPAIGN, event_ledger_path="actions.jsonl",
        event_ledger_bytes=ledger, evidence_documents=[], campaign_state=_state(),
        final_results_reconciled=True, training_idle=True, evaluations_idle=True,
    )
    ledger_path = Path(result["event_ledger_snapshot"])
    index_path = Path(result["failure_index"])
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert ledger_path.read_bytes() == ledger
    assert payload["event_ledger"]["sha256"] == hashlib.sha256(ledger).hexdigest()
    assert payload["event_ledger"]["snapshot_path"] == str(ledger_path)
    with pytest.raises(FileExistsError):
        publish_failure_index(
            output_dir=tmp_path, campaign_id=CAMPAIGN, event_ledger_path="actions.jsonl",
            event_ledger_bytes=ledger, evidence_documents=[], campaign_state=_state(),
            final_results_reconciled=True, training_idle=True, evaluations_idle=True,
        )
