"""Prepare a hash-bound index of campaign failures for final closeout.

This module has no runtime/API side effects. It snapshots the runner ledger and
referenced evidence only when the caller explicitly invokes publication after
the campaign's final evaluations have been reconciled.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label} contains an empty row at line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is malformed at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} is not an object")
        rows.append(row)
    return rows


def _classify_event(row: dict[str, Any]) -> str | None:
    event = row.get("event")
    if event == "tick_error":
        return "runner_error"
    if event == "candidate_launch_aborted":
        return "candidate_launch_aborted"
    if event == "campaign_experiment_invalidated":
        return "invalidated_experiment"
    if event == "finalist_evaluation_result" and row.get("passed") is False:
        return "failed_competitive_final_evaluation"
    if event == "winner_clean_reload_probe_result" and row.get("passed") is False:
        return "failed_clean_reload_probe"
    action = row.get("action")
    if isinstance(action, str) and (
        "failed" in action or "invalid" in action or "rejected" in action
        or action in {"stop_training_on_runtime_fault", "candidate_launch_reconciliation_required"}
    ):
        return "failed_or_interrupted_action"
    return None


def _failure_row(row: dict[str, Any], source: str, line_number: int) -> dict[str, Any]:
    keys = (
        "timestamp_utc", "event", "action", "phase", "experiment_id", "run_id",
        "evaluation_id", "config_sha256", "parent_checkpoint_sha256", "checkpoint_sha256",
        "manifest_sha256", "report_sha256", "artifact_path", "passed", "stage", "reason",
        "error", "faults", "failure_classification", "terminal_context",
    )
    item = {key: row[key] for key in keys if key in row}
    item.update({"classification": _classify_event(row), "source_path": source,
                 "source_line": line_number})
    return item


def build_failure_index(
    *, campaign_id: str, event_ledger_path: str, event_ledger_bytes: bytes,
    evidence_documents: list[tuple[str, bytes]], campaign_state: dict[str, Any],
    unresolved_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build deterministic failure index over exact ledger/evidence bytes.

    `evidence_documents` contains (logical path, exact bytes) for evidence
    snapshots available to the caller. The index discovers ledger path fields
    and marks any reference that was not supplied as unresolved.
    """
    if not campaign_id or campaign_state.get("campaign_id") != campaign_id:
        raise ValueError("campaign state identity mismatch")
    events = _parse_jsonl(event_ledger_bytes, "campaign event ledger")
    failure_rows = []
    for line_number, row in enumerate(events, start=1):
        if _classify_event(row):
            failure_rows.append(_failure_row(row, event_ledger_path, line_number))

    evidence_index: list[dict[str, Any]] = []
    for logical_path, raw in evidence_documents:
        if not isinstance(raw, bytes):
            raise TypeError("evidence document bytes are required")
        digest = _sha256(raw)
        evidence_index.append({"path": logical_path, "sha256": digest, "size_bytes": len(raw)})
        for line_number, row in enumerate(_parse_jsonl(raw, logical_path), start=1) if logical_path.endswith(".jsonl") else []:
            classification = row.get("failure_classification")
            if classification or row.get("probe") is None and row.get("evaluation_id"):
                failure_rows.append({
                    "classification": classification or "probe_failed_or_missing_evidence",
                    "source_path": logical_path, "source_sha256": digest,
                    "source_line": line_number, "evaluation_id": row.get("evaluation_id"),
                    "parent_run_id": row.get("parent_run_id"),
                    "report_path": row.get("report_path"), "report_sha256": row.get("report_sha256"),
                    "terminal_context": row.get("terminal_context"),
                })
        if logical_path.endswith(".json"):
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"evidence document is invalid JSON: {logical_path}") from exc
            probes = document.get("probe_evidence", []) if isinstance(document, dict) else []
            if not isinstance(probes, list):
                raise ValueError(f"probe_evidence is malformed: {logical_path}")
            for row_number, probe in enumerate(probes, start=1):
                if not isinstance(probe, dict):
                    raise ValueError(f"probe_evidence row is malformed: {logical_path}:{row_number}")
                if probe.get("failure_classification") or probe.get("probe") is None:
                    failure_rows.append({
                        "classification": probe.get("failure_classification") or "probe_failed_or_missing_evidence",
                        "source_path": logical_path, "source_sha256": digest,
                        "source_row": row_number, "evaluation_id": probe.get("evaluation_id"),
                        "parent_run_id": probe.get("parent_run_id"),
                        "report_path": probe.get("report_path"), "report_sha256": probe.get("report_sha256"),
                        "terminal_context": probe.get("terminal_context"),
                    })

    state_events = campaign_state.get("events", [])
    if not isinstance(state_events, list):
        raise ValueError("campaign state events must be a list")
    invalidated: list[dict[str, Any]] = []
    for index, row in enumerate(state_events, start=1):
        if isinstance(row, dict) and row.get("event") == "campaign_experiment_invalidated":
            invalidated.append({
                "state_event_index": index,
                "experiment_id": row.get("experiment_id"),
                "config_sha256": row.get("config_sha256"),
                "reason": row.get("reason"),
                "timestamp_utc": row.get("timestamp_utc"),
            })

    supplied_paths = {path.replace("\\", "/").casefold() for path, _ in evidence_documents}
    unresolved: list[dict[str, Any]] = []
    for row in unresolved_evidence or []:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("unresolved evidence entries require a path")
        unresolved.append({
            "path": row["path"],
            "reason": str(row.get("reason") or "unavailable at closeout"),
            "reference_source": row.get("reference_source"),
        })
    unresolved_paths = {row["path"].replace("\\", "/").casefold() for row in unresolved}
    references: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(events, start=1):
        for field in ("evidence_path", "plan_path", "artifact_path", "report_path"):
            value = row.get(field)
            if isinstance(value, str) and value:
                key = value.replace("\\", "/").casefold()
                references.setdefault(key, {"path": value, "reference_source": f"event_ledger:{line_number}:{field}"})
    for key, reference in references.items():
        if key not in supplied_paths and key not in unresolved_paths:
            unresolved.append({
                **reference,
                "reason": "referenced by event ledger but not supplied to closeout index",
            })

    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_state_canonical_sha256": _sha256(
            json.dumps(campaign_state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ),
        "index_status": "failure_history_snapshot; not a final acceptance decision",
        "event_ledger": {
            "path": event_ledger_path, "sha256": _sha256(event_ledger_bytes),
            "size_bytes": len(event_ledger_bytes), "event_count": len(events),
        },
        "evidence_documents": evidence_index,
        "unresolved_evidence_references": unresolved,
        "evidence_reference_closure_complete": not unresolved,
        "failed_or_interrupted_events": failure_rows,
        "invalidated_experiments": invalidated,
        "optimizer_accounting": {
            "rollout_update_cycles": "count distinct post-optimizer audit num_timesteps rows/checkpoint events",
            "completed_ppo_epochs": "SB3 model._n_updates; record separately from rollout cycles",
            "optimizer_step_calls": "not asserted unless directly instrumented; do not infer from _n_updates",
            "resume_reconstruction": (
                "PPO resume constructs a fresh PPO instance and transfers policy parameters plus global timestep; "
                "serialized optimizer, schedules, and rollout buffer are intentionally ignored. Record each exact "
                "parent checkpoint SHA and fresh optimizer initialization evidence from that run's config/lineage."
            ),
            "required_per_run_fields": [
                "run_id", "parent_checkpoint_sha256", "checkpoint_sha256", "effective_config_sha256",
                "fresh_optimizer_instance_created", "resume_optimizer_state_restored",
                "global_timestep_start", "global_timestep_end", "rollout_update_cycle_count",
                "sb3_n_updates_start", "sb3_n_updates_end",
            ],
        },
        "ui_visual_review": {"status": "open", "evidence_path": None, "reviewed_at_utc": None},
    }


def publish_failure_index(
    *, output_dir: Path, campaign_id: str, event_ledger_path: str,
    event_ledger_bytes: bytes, evidence_documents: list[tuple[str, bytes]],
    campaign_state: dict[str, Any], final_results_reconciled: bool,
    training_idle: bool, evaluations_idle: bool,
    unresolved_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Atomically publish a frozen failure ledger/index; never overwrite.

    Must only be called after the final evaluation history is reconciled and
    both training and evaluation services are idle.
    """
    if not (final_results_reconciled is True and training_idle is True and evaluations_idle is True):
        raise RuntimeError("closeout requires reconciled final results and idle training/evaluation services")
    index = build_failure_index(
        campaign_id=campaign_id, event_ledger_path=event_ledger_path,
        event_ledger_bytes=event_ledger_bytes, evidence_documents=evidence_documents,
        campaign_state=campaign_state, unresolved_evidence=unresolved_evidence,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "campaign_actions_snapshot.jsonl"
    index_path = output_dir / "campaign_failure_index.json"
    encoded_index = (json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    ledger_digest = index["event_ledger"]["sha256"]
    index["event_ledger"]["snapshot_path"] = str(ledger_path)
    encoded_index = (json.dumps(index, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _atomic_create(ledger_path, event_ledger_bytes)
    _atomic_create(index_path, encoded_index)
    return {"event_ledger_snapshot": str(ledger_path), "event_ledger_sha256": ledger_digest,
            "failure_index": str(index_path), "failure_index_sha256": _sha256(encoded_index)}


def _atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
