from __future__ import annotations

import copy

import pytest

from src.layer4.hub.campaign_phase_runner import _campaign_evidence_state_signature


def _base_state() -> dict:
    return {
        "campaign_id": "campaign-test",
        "status": "active",
        "started_at_utc": "2026-10-12T23:26:44Z",
        "training_freeze_at_utc": "2026-10-13T08:26:44Z",
        "hard_deadline_at_utc": "2026-10-13T11:26:44Z",
        "start_monotonic_ns": 10,
        "training_freeze_monotonic_ns": 20,
        "hard_deadline_monotonic_ns": 30,
        "boot_id": "boot-a",
        "campaign_plan": {"target_kl": 0.01, "exploration_profile": "bc_campaign"},
        "revision": 1,
        "last_observed_at_utc": "2026-10-13T01:00:00Z",
        "last_observed_monotonic_ns": 40,
        "updated_at_utc": "2026-10-13T01:00:00Z",
        "events": [],
    }


def test_only_supervisor_observation_timestamps_and_revision_are_ignored():
    first = _base_state()
    first["events"] = [{
        "event": "supervisor_active_work_observed",
        "active": {"training": ["run-a"], "evaluation": None},
        "timestamp_utc": "2026-10-13T01:00:00Z", "monotonic_ns": 40,
    }]
    second = copy.deepcopy(first)
    second.update({
        "revision": 2,
        "last_observed_at_utc": "2026-10-13T01:00:30Z",
        "last_observed_monotonic_ns": 70,
        "updated_at_utc": "2026-10-13T01:00:30Z",
    })
    second["events"][0].update({"timestamp_utc": "2026-10-13T01:00:30Z", "monotonic_ns": 70})
    assert _campaign_evidence_state_signature(first) == _campaign_evidence_state_signature(second)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("status", "failed_closed"),
        ("started_at_utc", "changed"),
        ("training_freeze_at_utc", "changed"),
        ("hard_deadline_at_utc", "changed"),
        ("start_monotonic_ns", 11),
        ("training_freeze_monotonic_ns", 21),
        ("hard_deadline_monotonic_ns", 31),
        ("boot_id", "boot-b"),
        ("campaign_plan", {"target_kl": 0.02, "exploration_profile": "ordinary"}),
    ],
)
def test_clock_status_and_configuration_changes_are_bound(field: str, changed):
    first = _base_state()
    second = {**first, field: changed}
    assert _campaign_evidence_state_signature(first) != _campaign_evidence_state_signature(second)


@pytest.mark.parametrize(
    "decision_event",
    [
        {"event": "campaign_experiment_declared", "experiment_id": "time_cost_3_calibrated_retry",
         "config_sha256": "a" * 64, "declared_config": {"time_cost_per_simulated_second": 0.027}},
        {"event": "campaign_experiment_invalidated", "experiment_id": "time_cost_3_calibrated_retry",
         "invalidated_config_sha256": "a" * 64, "affected_run_ids": ["run-a"]},
        {"event": "supervisor_action", "action": "stop_training", "run_id": "run-a"},
        {"event": "campaign_terminal", "previous_status": "active", "status": "expired",
         "reason": "hard deadline"},
    ],
)
def test_decision_declaration_invalidation_stop_and_terminal_events_are_bound(decision_event: dict):
    first = _base_state()
    changed = {**first, "events": [decision_event]}
    assert _campaign_evidence_state_signature(first) != _campaign_evidence_state_signature(changed)


def test_active_work_identity_and_api_outage_details_are_bound_but_event_clock_is_not():
    first = _base_state()
    first["events"] = [{
        "event": "supervisor_active_work_observed", "active": {"training": ["run-a"]},
        "timestamp_utc": "t1", "monotonic_ns": 1,
    }, {"event": "supervisor_api_unavailable", "error": "TimeoutError",
        "timestamp_utc": "t1", "monotonic_ns": 1}]
    same_observation = copy.deepcopy(first)
    for event in same_observation["events"]:
        event["timestamp_utc"] = "t2"
        event["monotonic_ns"] = 2
    assert _campaign_evidence_state_signature(first) == _campaign_evidence_state_signature(same_observation)

    changed_active = copy.deepcopy(same_observation)
    changed_active["events"][0]["active"] = {"training": ["run-b"]}
    assert _campaign_evidence_state_signature(first) != _campaign_evidence_state_signature(changed_active)

    changed_outage = copy.deepcopy(same_observation)
    changed_outage["events"][1]["error"] = "ConnectionError"
    assert _campaign_evidence_state_signature(first) != _campaign_evidence_state_signature(changed_outage)
