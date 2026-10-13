"""Build a hash-locked finalists manifest from completed campaign probes.

Training probes are nonranking readiness checks. Their measured medians are
used only to shortlist at most two verified checkpoints for the clean,
competitive final protocol; they never count as final race results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.layer3.bc_artifact import (
    PPOVerificationContext,
    sha256_file,
    verify_ppo_lineage_checkpoint,
    verify_training_probe_checkpoint,
)
from src.layer4.hub.campaign_deadline import (
    CAMPAIGN_ID,
    CAMPAIGN_PLAN,
    INVALIDATED_CAMPAIGN_RUN_IDS,
    INVALIDATED_TIME_COST_VARIANTS,
    TIME_COST_VARIANTS,
    campaign_clock_status,
    campaign_validation_request,
    default_state_path,
    load_campaign_state,
)

_851968_BUNDLE_ARCHIVE_SHA256 = "88f1f47d71ba3c086692623426db7ed72996807db17a42e4b762183510154339"
_851968_BUNDLE_MANIFEST_SHA256 = "e0050f364cdddb339028698a5b3032af6c9c3e75c33cca764949f524a0f98111"
_851968_BUNDLE_RECEIPT_SHA256 = "d6f4ad960c907ba3088aba917ae9d1890e84f0979137385f6d576fc0f30d547a"


def _jsonl(path: Path, verification_context: Optional[PPOVerificationContext] = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if verification_context is None:
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        digest = verification_context.hash_file(path)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"JSONL changed while being read: {path}")
        lines = data.decode("utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"empty history row at {path}:{index}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"history row is not an object at {path}:{index}")
        rows.append(row)
    return rows


def _read_json_document(path: Path, context: PPOVerificationContext) -> Any:
    """Read any JSON value while binding the bytes into a verification pass."""
    digest = context.hash_file(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"JSON changed while being read: {path}")
    return json.loads(data.decode("utf-8-sig"))


def _verified_evidence_bundle(
    row: dict[str, Any], expected_candidate_sha: str, repo_root: Path,
    context: PPOVerificationContext,
) -> Optional[dict[str, Any]]:
    receipt_value = row.get("bundle_receipt_path")
    receipt_sha = row.get("bundle_receipt_sha256")
    if receipt_value is None and receipt_sha is None:
        return None
    receipt_path = _resolve_in_logs(receipt_value, repo_root)
    if (receipt_path is None or not receipt_path.is_file()
            or receipt_sha != _851968_BUNDLE_RECEIPT_SHA256
            or context.hash_file(receipt_path) != _851968_BUNDLE_RECEIPT_SHA256):
        raise ValueError("bundle fallback requires the pinned 851968 receipt path and SHA")
    receipt = _read_json_document(receipt_path, context)
    if (not isinstance(receipt, dict)
            or receipt.get("kind") != "candidate_package_receipt"
            or receipt.get("archive_sha256") != _851968_BUNDLE_ARCHIVE_SHA256
            or receipt.get("manifest_sha256") != _851968_BUNDLE_MANIFEST_SHA256
            or receipt.get("candidate_sha256") != expected_candidate_sha):
        raise ValueError("bundle receipt does not bind the fixed candidate, archive, and manifest")
    archive = _resolve_in_logs(receipt.get("archive_path"), repo_root)
    manifest_path = _resolve_in_logs(receipt.get("manifest_path"), repo_root)
    if (archive is None or not archive.is_file() or manifest_path is None or not manifest_path.is_file()
            or context.hash_file(archive) != _851968_BUNDLE_ARCHIVE_SHA256
            or archive.stat().st_size != receipt.get("archive_size_bytes")
            or context.hash_file(manifest_path) != _851968_BUNDLE_MANIFEST_SHA256):
        raise ValueError("bundle archive or extracted manifest is missing or changed")
    manifest = _read_json_document(manifest_path, context)
    if not isinstance(manifest, dict) or manifest.get("candidate", {}).get("sha256") != expected_candidate_sha:
        raise ValueError("bundle manifest does not bind the expected candidate")
    try:
        with zipfile.ZipFile(archive, "r") as archive_zip:
            embedded = archive_zip.read("bundle_manifest.json")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("bundle archive is unreadable or missing its embedded manifest") from exc
    if (hashlib.sha256(embedded).hexdigest() != _851968_BUNDLE_MANIFEST_SHA256
            or embedded != manifest_path.read_bytes()):
        raise ValueError("embedded bundle manifest differs from its receipt-bound extracted copy")
    included = manifest.get("included_files")
    if not isinstance(included, list):
        raise ValueError("bundle manifest has no indexed payload list")
    entries: dict[str, dict[str, Any]] = {}
    for item in included:
        if not isinstance(item, dict):
            raise ValueError("bundle payload index entry is malformed")
        path = item.get("bundle_path")
        if (not isinstance(path, str) or not path or path.startswith(("/", "\\"))
                or ".." in Path(path).parts or path in entries):
            raise ValueError("bundle payload index contains an unsafe or duplicate path")
        entries[path.replace("\\", "/")] = item
    kind = row.get("kind")
    evidence_member = ("evidence/final851968/competitive_benchmark_851968_terminal_evidence.json"
                       if kind == "competitive"
                       else "evidence/final851968/clean_reload_851968_validation.json"
                       if kind == "clean_reload" else None)
    evidence_entry = entries.get(evidence_member) if evidence_member else None
    if (not isinstance(evidence_entry, dict)
            or evidence_entry.get("sha256") != row.get("evidence_sha256")):
        raise ValueError("bundle does not index the exact existing-validation evidence document")
    with zipfile.ZipFile(archive, "r") as archive_zip:
        evidence_bytes = archive_zip.read(evidence_member)
    if hashlib.sha256(evidence_bytes).hexdigest() != evidence_entry.get("sha256"):
        raise ValueError("bundled existing-validation evidence bytes do not match the bundle index")
    return {"archive": archive, "entries": entries, "receipt_path": receipt_path,
            "receipt_sha256": _851968_BUNDLE_RECEIPT_SHA256,
            "archive_sha256": _851968_BUNDLE_ARCHIVE_SHA256,
            "manifest_sha256": _851968_BUNDLE_MANIFEST_SHA256,
            "evidence_member": evidence_member}


def _bundle_member_bytes(bundle: dict[str, Any], member: str) -> bytes:
    normalized = member.replace("\\", "/")
    entry = bundle["entries"].get(normalized)
    if not isinstance(entry, dict):
        raise ValueError(f"bundle does not index required payload {normalized}")
    try:
        with zipfile.ZipFile(bundle["archive"], "r") as archive_zip:
            data = archive_zip.read(normalized)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"bundle is missing indexed payload {normalized}") from exc
    if (len(data) != entry.get("size_bytes")
            or hashlib.sha256(data).hexdigest() != entry.get("sha256")):
        raise ValueError(f"bundle payload bytes do not match index for {normalized}")
    return data


def _resolve_in_logs(value: Any, repo_root: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("\\", "/")
    root = (repo_root / "logs" / "rl").resolve()
    if text.startswith("/runs/rl/"):
        candidate = root / text.removeprefix("/runs/rl/")
    elif text.startswith("/app/logs/rl/"):
        candidate = root / text.removeprefix("/app/logs/rl/")
    elif "/logs/rl/" in text:
        candidate = root / text.rsplit("/logs/rl/", 1)[1]
    else:
        # Parse the already-normalized spelling so Windows receipt paths
        # remain portable when the verifier runs under WSL/Linux.
        raw = Path(text)
        candidate = raw if raw.is_absolute() else repo_root / raw
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _finite_positive_list(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            and float(item) > 0
            for item in value
        )
    )


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _probe_candidate(
    row: Any, repo_root: Path, verification_context: PPOVerificationContext,
    *, allow_seed_exclusion: bool = False,
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("each finalist candidate must be an object")
    model = _resolve_in_logs(row.get("model_path"), repo_root)
    history_path = _resolve_in_logs(row.get("evaluation_history_path"), repo_root)
    report_path = _resolve_in_logs(row.get("probe_report_path"), repo_root)
    expected_sha = row.get("checkpoint_sha256")
    expected_report_sha = row.get("probe_report_sha256")
    evaluation_id = row.get("evaluation_id")
    if (model is None or history_path is None or report_path is None
            or not model.is_file() or not history_path.is_file() or not report_path.is_file()):
        raise ValueError("candidate checkpoint, history, and report must resolve to existing files inside logs/rl")
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha.lower())):
        raise ValueError("candidate checkpoint_sha256 must be a SHA-256 hex digest")
    if (not isinstance(expected_report_sha, str) or len(expected_report_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_report_sha.lower())):
        raise ValueError("candidate probe_report_sha256 must be a SHA-256 hex digest")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise ValueError("candidate evaluation_id is required")
    actual_sha = verification_context.hash_file(model)
    report_sha = verification_context.hash_file(report_path)
    if actual_sha != expected_sha.lower() or report_sha != expected_report_sha.lower():
        raise ValueError("candidate checkpoint or probe report hash changed")

    training_probe = verify_training_probe_checkpoint(model, verification_context=verification_context)
    lineage = verify_ppo_lineage_checkpoint(model, verification_context=verification_context)
    if (not training_probe or training_probe.get("checkpoint_sha256") != expected_sha.lower()
            or not lineage or lineage.get("checkpoint_sha256") != expected_sha.lower()):
        raise ValueError("candidate failed authenticated training-probe/PPO-lineage verification")
    lineage_sidecar = model.with_name(f"{model.stem}.lineage.json")
    if not lineage_sidecar.is_file():
        raise ValueError("candidate checkpoint is missing its authenticated lineage sidecar")
    lineage_sidecar_sha = verification_context.hash_file(lineage_sidecar)
    lineage_record = verification_context.read_json(lineage_sidecar)
    if (not isinstance(lineage_record, dict)
            or lineage_record.get("checkpoint_sha256") != expected_sha.lower()):
        raise ValueError("candidate lineage sidecar does not bind the selected checkpoint hash")
    bc_ancestor_record = lineage_record.get("bc_ancestor_checkpoint")
    bc_ancestor_sha = (bc_ancestor_record.get("sha256")
                       if isinstance(bc_ancestor_record, dict) else None)
    screening = _require_post_parent_training_evidence(
        model, lineage, verification_context, allow_seed_exclusion=allow_seed_exclusion,
    )
    ancestry_run_ids, ancestry_reward_costs = _verified_campaign_ancestry(model, verification_context)
    training_config_path = model.parent.parent / "config.json"
    if not training_config_path.is_file():
        raise ValueError("candidate training run has no persisted effective config")
    training_config_sha = verification_context.hash_file(training_config_path)
    training_config_bytes = training_config_path.read_bytes()
    if hashlib.sha256(training_config_bytes).hexdigest() != training_config_sha:
        raise ValueError("candidate training config changed while being read")
    if training_config_sha != lineage.get("effective_config_sha256"):
        raise ValueError("candidate training config does not match its authenticated lineage hash")
    training_config = json.loads(training_config_bytes)
    ppo = training_config.get("ppo") if isinstance(training_config, dict) else None
    automatic = training_config.get("automatic_evaluation") if isinstance(training_config, dict) else None
    exploration = training_config.get("ppo_exploration") if isinstance(training_config, dict) else None
    ppo_effective = ppo.get("effective") if isinstance(ppo, dict) else None
    reward = training_config.get("training_only_reward") if isinstance(training_config, dict) else None
    candidate_time_cost = reward.get("time_cost_per_simulated_second") if isinstance(reward, dict) else None
    allowed_costs = {0.0, *TIME_COST_VARIANTS.values()}
    if (not isinstance(training_config, dict)
            or training_config.get("runtime") != "official IROS 2026 API and simulator images"
            or training_config.get("observation_profile") != "official_sensors_camera"
            or not isinstance(ppo, dict)
            or ppo.get("checkpoint_interval_scope") != "global_step_buckets"
            or not isinstance(ppo_effective, dict)
            or ppo_effective.get("n_envs") != 1
            or not isinstance(automatic, dict)
            or automatic.get("enabled") is not True
            or automatic.get("mode") != "training_probe"
            or automatic.get("interval_scope") != "global_step_buckets"
            or automatic.get("interval") != 50000
            or not isinstance(exploration, dict)
            or exploration.get("profile") != "bc_campaign"
            or isinstance(candidate_time_cost, bool)
            or not isinstance(candidate_time_cost, (int, float))
            or float(candidate_time_cost) not in allowed_costs):
        raise ValueError("candidate does not have the fixed official single-env campaign configuration")

    history_matches = [event for event in _jsonl(history_path, verification_context)
                       if event.get("evaluation_id") == evaluation_id]
    if len(history_matches) != 1:
        raise ValueError("candidate evaluation_id must match exactly one history row")
    history = history_matches[0]
    result = history.get("latest_result")
    if (history.get("state") != "complete" or not isinstance(result, dict)
            or result.get("source") != "automatic"
            or result.get("validation_mode") != "training_probe"
            or result.get("protocol") != "official_training_probe_1_plus_3"
            or result.get("readiness_passed") is not True
            or result.get("model_sha256") != expected_sha.lower()
            or not _exact_int(result.get("completed_attempts"), 1)
            or not _exact_int(result.get("attempts_required"), 1)):
        raise ValueError("history row is not a completed, passing, correctly linked training probe")
    service_runs = result.get("runs")
    if not isinstance(service_runs, list) or len(service_runs) != 1 or not isinstance(service_runs[0], dict):
        raise ValueError("training probe history must contain exactly one attempt")
    service_attempt = service_runs[0]
    run_id = service_attempt.get("run_id")
    if (not isinstance(run_id, str) or not run_id
            or service_attempt.get("evaluation_report") is None
            or service_attempt.get("score_status") != "valid"
            or not _exact_int(service_attempt.get("warmup_laps"), 1)
            or not _exact_int(service_attempt.get("race_laps_completed"), 3)
            or not _exact_int(service_attempt.get("race_laps_required"), 3)
            or not _exact_int(service_attempt.get("collision_count"), 0)
            or not _exact_int(service_attempt.get("race_collisions"), 0)
            or service_attempt.get("readiness_passed") is not True):
        raise ValueError("training probe service attempt failed the 1+3 zero-collision readiness gate")
    canonical_report = repo_root / "logs" / "rl" / f"official_run_{run_id}" / "evaluation.json"
    if report_path.resolve() != canonical_report.resolve():
        raise ValueError("candidate report path does not match the history run_id")
    if _resolve_in_logs(service_attempt.get("evaluation_report"), repo_root) != report_path:
        raise ValueError("history does not point to the supplied report")

    report = verification_context.read_json(report_path)
    if (not isinstance(report, dict)
            or report.get("model_sha256") != expected_sha.lower()
            or report.get("validation_mode") != "training_probe"
            or report.get("protocol") != "official_training_probe_1_plus_3"
            or report.get("competitive_score_valid") is not False
            or report.get("noncompetitive_diagnostic") is not True
            or report.get("official_base_image") != "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete"
            or report.get("simulator_image") != "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete"
            or report.get("competition") != "RoboRacer Sim Racing League @ IROS 2026"
            or report.get("observation_profile") != "official_sensors_camera"
            or report.get("steering_mode") != "normal"
            or report.get("throttle_mode") != "bidirectional"
            or report.get("deterministic") is not True):
        raise ValueError("probe report identity or nonranking protocol does not match candidate")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError("probe report must contain exactly one run")
    attempt = runs[0]
    lap_times = attempt.get("race_lap_times_s")
    warmup = attempt.get("warmup_lap_times_s")
    camera_n = attempt.get("camera_sample_count")
    if (attempt.get("complete") is not True
            or attempt.get("stop_reason") != "race_complete"
            or attempt.get("score_status") != "valid"
            or not _exact_int(attempt.get("race_laps_completed"), 3)
            or not _exact_int(attempt.get("race_laps_required"), 3)
            or not _exact_int(attempt.get("race_collisions"), 0)
            or not _exact_int(attempt.get("warmup_collisions"), 0)
            or attempt.get("valid_for_comparison") is not False
            or attempt.get("noncompetitive_diagnostic") is not True
            or attempt.get("frontier_projection_valid") is not True
            or attempt.get("route_projection_valid") is not True
            or not _finite_positive_list(lap_times, 3)
            or not _finite_positive_list(warmup, 1)
            or not isinstance(camera_n, int) or isinstance(camera_n, bool) or camera_n <= 0
            or not _exact_int(attempt.get("camera_fresh_sample_count"), camera_n)
            or not _exact_int(attempt.get("camera_stale_sample_count"), 0)
            or not _exact_int(attempt.get("camera_missing_sample_count"), 0)
            or not _exact_int(attempt.get("camera_freshness_unknown_count"), 0)):
        raise ValueError("probe report failed complete 1+3, fresh-camera, projection, or zero-collision checks")

    median = sorted(float(value) for value in lap_times)[1]
    return {
        "model_path": str(model),
        "checkpoint_sha256": expected_sha.lower(),
        "lineage_sidecar_path": str(lineage_sidecar),
        "lineage_sidecar_sha256": lineage_sidecar_sha,
        "bc_ancestor_checkpoint_sha256": bc_ancestor_sha,
        "effective_config_sha256": lineage.get("effective_config_sha256"),
        "training_config_path": str(training_config_path),
        "training_config_sha256": training_config_sha,
        "probe_report_path": str(report_path),
        "probe_report_sha256": expected_report_sha.lower(),
        "evaluation_id": evaluation_id,
        "probe_run_id": run_id,
        "scored_lap_times_s": [float(value) for value in lap_times],
        "median_scored_lap_s": median,
        "warmup_lap_times_s": [float(value) for value in warmup],
        "collisions": 0,
        "camera_sample_count": camera_n,
        "camera_fresh_sample_count": camera_n,
        "ancestry_run_ids": ancestry_run_ids,
        "ancestry_time_cost_per_simulated_second": ancestry_reward_costs,
        **screening,
        "frontier_projection_valid": True,
        "route_projection_valid": True,
        "shortlist_basis": "fastest passing nonranking training-probe median; competitive status remains unproven",
    }


def _verify_existing_competitive_validation(
    row: Any,
    expected_checkpoint_sha256: str,
    repo_root: Path,
    verification_context: PPOVerificationContext,
) -> dict[str, Any]:
    """Authenticate an already completed standard batch for a finalist SHA.

    Existing evaluation batches may satisfy the final protocol when their
    durable terminal evidence and every raw report/process manifest are still
    present and byte-identical. This prevents a finalist runner from queuing
    duplicate races merely because the batch predates its manifest.
    """
    if not isinstance(row, dict) or row.get("kind") not in {"competitive", "clean_reload"}:
        raise ValueError("existing competitive validation requires a supported kind")
    evidence_path = _resolve_in_logs(row.get("evidence_path"), repo_root)
    evidence_sha = row.get("evidence_sha256")
    if (evidence_path is None or not evidence_path.is_file()
            or not isinstance(evidence_sha, str) or len(evidence_sha) != 64
            or any(char not in "0123456789abcdef" for char in evidence_sha.lower())):
        raise ValueError("existing competitive validation evidence path/hash is malformed")
    if verification_context.hash_file(evidence_path) != evidence_sha.lower():
        raise ValueError("existing competitive validation evidence hash changed")
    evidence = verification_context.read_json(evidence_path)
    if not isinstance(evidence, dict):
        raise ValueError("existing competitive validation evidence is not an object")
    bundle = _verified_evidence_bundle(row, expected_checkpoint_sha256, repo_root, verification_context)
    reports: list[dict[str, Any]] = []
    all_times: list[float] = []
    evidence_timestamp: Optional[str] = None
    clean_reload_fields: dict[str, Any] = {}
    if row["kind"] == "competitive":
        result = evidence.get("result")
        attempts = result.get("attempts") if isinstance(result, dict) else None
        if (evidence.get("source") != "manual" or evidence.get("validation_mode") is not None
                or evidence.get("model_sha256") != expected_checkpoint_sha256
                or evidence.get("source_checkpoint_sha256") != expected_checkpoint_sha256
                or not isinstance(result, dict) or result.get("state") != "complete"
                or result.get("valid") is not True
                or result.get("protocol") != "official_10_lap_race"
                or not _exact_int(result.get("completed_attempts"), 3)
                or not _exact_int(result.get("scored_laps_total"), 30)
                or not _exact_int(result.get("collision_total"), 0)
                or not isinstance(attempts, list) or len(attempts) != 3):
            raise ValueError("existing competitive batch does not authenticate a passing 3x(1+10) result")
        evaluation_id = evidence.get("evaluation_id")
        evidence_timestamp = evidence.get("captured_at_utc")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ValueError("existing competitive batch is missing its evaluation ID")
        if evidence.get("model_sha256") != expected_checkpoint_sha256:
            raise ValueError("competitive evidence model hash does not bind the selected candidate")
    else:
        attempts = evidence.get("attempts")
        if (evidence.get("validator") != "validate_final_evaluation"
                or evidence.get("validator_passed") is not True
                or evidence.get("requested_and_loaded_model_sha256") != expected_checkpoint_sha256
                or not _exact_int(evidence.get("scored_lap_count"), 30)
                or not isinstance(attempts, list) or len(attempts) != 3):
            raise ValueError("clean-reload evidence does not authenticate a passing 3x(1+10) result")
        evaluation_id = evidence.get("evaluation_id")
        history_path = _resolve_in_logs(evidence.get("history_row_path"), repo_root)
        history_sha = evidence.get("history_row_sha256")
        if (not isinstance(evaluation_id, str) or not evaluation_id or history_path is None
                or not history_path.is_file() or not isinstance(history_sha, str)
                or verification_context.hash_file(history_path) != history_sha.lower()):
            raise ValueError("clean-reload history row is missing or changed")
        history_rows = _read_json_document(history_path, verification_context)
        matching_history = ([item for item in history_rows
                             if isinstance(item, dict) and item.get("evaluation_id") == evaluation_id]
                            if isinstance(history_rows, list) else [])
        if (len(matching_history) != 1
                or matching_history[0].get("state") != "complete"
                or matching_history[0].get("model_sha256") != expected_checkpoint_sha256):
            raise ValueError("clean-reload history row does not bind the evaluation and candidate")
        evidence_timestamp = matching_history[0].get("updated_utc")
        hub_identity = evidence.get("hub_identity")
        if (evidence.get("validator_passed") is not True
                or evidence.get("training_idle_at_capture") is not True
                or evidence.get("evaluation_queue_idle_at_capture") is not True
                or not isinstance(hub_identity, dict)
                or not isinstance(hub_identity.get("pid"), int)
                or isinstance(hub_identity.get("pid"), bool)
                or not isinstance(hub_identity.get("process_start_identity"), str)
                or not isinstance(hub_identity.get("source_sha256"), str)
                or len(hub_identity["source_sha256"]) != 64):
            raise ValueError("clean-reload evidence lacks an idle, source-identified hub validation")
        clean_reload_fields = {
            "hub_identity": hub_identity,
            "validator_passed": evidence.get("validator_passed"),
            "training_idle_at_capture": evidence.get("training_idle_at_capture"),
            "evaluation_queue_idle_at_capture": evidence.get("evaluation_queue_idle_at_capture"),
            "requested_and_loaded_model_sha256": evidence.get("requested_and_loaded_model_sha256"),
            "history_row_path": str(history_path),
            "history_row_sha256": history_sha.lower(),
        }

    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("existing validation attempt is malformed")
        run_id = attempt.get("run_id")
        report_path = _resolve_in_logs(attempt.get("report_path"), repo_root)
        report_sha = attempt.get("report_sha256")
        if row["kind"] == "competitive":
            if not isinstance(run_id, str) or attempt.get("run_id") != run_id:
                raise ValueError("competitive attempt is missing its canonical run ID")
            expected_laps = attempt.get("scored_laps")
            if (not _exact_int(expected_laps, 10) or not _exact_int(attempt.get("warmup_laps"), 1)
                    or not _exact_int(attempt.get("race_collisions"), 0)
                    or attempt.get("route_monitor_only") is not True):
                raise ValueError("competitive attempt failed lap/collision/route-monitor checks")
        else:
            if (not isinstance(run_id, str) or not run_id
                    or not _exact_int(attempt.get("scored_laps"), 10)
                    or not _exact_int(attempt.get("warmup_laps"), 1)
                    or not _exact_int(attempt.get("collision_count"), 0)
                    or attempt.get("projection_valid") is not True):
                raise ValueError("clean-reload attempt failed lap/collision/projection checks")
            expected_laps = 10
        if (report_path is None or not isinstance(report_sha, str) or len(report_sha) != 64):
            raise ValueError("existing validation raw report reference/hash is malformed")
        canonical_report = (repo_root / "logs" / "rl" / f"official_run_{run_id}" / "evaluation.json").resolve()
        if report_path != canonical_report:
            raise ValueError("existing validation report does not match its canonical evaluator run path")
        report_member = f"evidence/final851968/evaluation_{run_id}.json"
        if bundle is None and report_path.is_file():
            actual_report_sha = verification_context.hash_file(report_path)
            report = verification_context.read_json(report_path)
            rendered_report_path = str(report_path)
        elif bundle is not None:
            report_bytes = _bundle_member_bytes(bundle, report_member)
            actual_report_sha = hashlib.sha256(report_bytes).hexdigest()
            report = json.loads(report_bytes.decode("utf-8-sig"))
            rendered_report_path = f"bundle:{bundle['archive']}!/{report_member}"
        else:
            raise ValueError("existing validation raw report is missing and no pinned bundle was supplied")
        if actual_report_sha != report_sha.lower():
            raise ValueError("existing validation raw report is missing or changed")
        if (not isinstance(report, dict)
                or report.get("model_sha256") != expected_checkpoint_sha256
                or report.get("competition") != "RoboRacer Sim Racing League @ IROS 2026"
                or report.get("official_base_image") != "autodriveecosystem/autodrive_roboracer_api:2026-iros-compete"
                or report.get("simulator_image") != "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete"
                or report.get("observation_profile") != "official_sensors_camera"
                or report.get("throttle_mode") != "bidirectional"
                or report.get("steering_mode") != "normal"
                or report.get("deterministic") is not True
                or report.get("validation_mode") is not None
                or report.get("competitive_score_valid") is not True
                or report.get("protocol") != "official_competitive_1_plus_10"):
            raise ValueError("existing raw report is not the standard candidate-bound competitive evaluation")
        report_runs = report.get("runs")
        if not isinstance(report_runs, list) or len(report_runs) != 1:
            raise ValueError("existing raw report must contain exactly one run")
        race = report_runs[0]
        laps = race.get("race_lap_times_s")
        warmup = race.get("warmup_lap_times_s")
        camera = race.get("camera_sample_count")
        if (race.get("complete") is not True or race.get("stop_reason") != "race_complete"
                or race.get("valid_for_comparison") is not True
                or not _exact_int(race.get("race_laps_required"), 10)
                or not _exact_int(race.get("race_laps_completed"), 10)
                or not _exact_int(race.get("warmup_collisions"), 0)
                or not _exact_int(race.get("race_collisions"), 0)
                or not _exact_int(race.get("raw_collision_count_at_finish"), 0)
                or not _exact_int(race.get("camera_fresh_sample_count"), camera)
                or not _exact_int(race.get("camera_stale_sample_count"), 0)
                or not _exact_int(race.get("camera_missing_sample_count"), 0)
                or not _exact_int(race.get("camera_freshness_unknown_count"), 0)
                or not isinstance(camera, int) or isinstance(camera, bool) or camera <= 0
                or not _finite_positive_list(laps, expected_laps)
                or not _finite_positive_list(warmup, 1)
                or race.get("frontier_projection_valid") is not True
                or race.get("route_projection_valid") is not True
                or race.get("route_monitor_only") is not True
                or race.get("official_scene_sha256") != "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674"
                or race.get("route_sha256") != "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"):
            raise ValueError("existing raw report failed complete lap, collision, camera, or route/scene checks")
        if row["kind"] == "competitive":
            # The terminal evidence summarizes each report. Require those
            # summary fields to agree with the raw bytes so a detached or
            # partially edited evidence document cannot be reused.
            if (attempt.get("report_sha256", "").lower() != report_sha.lower()
                    or attempt.get("scored_lap_times_s") != laps
                    or attempt.get("race_collisions") != race.get("race_collisions")
                    or (attempt.get("warmup_collisions") is not None
                        and attempt.get("warmup_collisions") != race.get("warmup_collisions"))
                    or attempt.get("camera_sample_count") != camera
                    or attempt.get("camera_fresh_sample_count") != race.get("camera_fresh_sample_count")
                    or attempt.get("route_sha256") != race.get("route_sha256")
                    or attempt.get("scene_sha256") != race.get("official_scene_sha256")):
                raise ValueError("competitive terminal evidence summary differs from its raw report")
        else:
            if (attempt.get("report_sha256", "").lower() != report_sha.lower()
                    or attempt.get("scored_lap_times_s") != laps
                    or attempt.get("collision_count") != race.get("race_collisions")):
                raise ValueError("clean-reload summary differs from its raw report")
        identity = race.get("simulator_identity")
        manifest_sha = identity.get("manifest_sha256") if isinstance(identity, dict) else None
        manifest_path = report_path.parent / "windows_simulator_processes.json"
        manifest_member = f"evidence/final851968/identities/{run_id}/windows_simulator_processes.json"
        if not isinstance(identity, dict):
            raise ValueError("existing report is missing simulator process identity")
        if bundle is None and manifest_path.is_file():
            actual_manifest_sha = verification_context.hash_file(manifest_path)
            process_manifest = _read_json_document(manifest_path, verification_context)
            rendered_manifest_path = str(manifest_path)
        elif bundle is not None:
            manifest_bytes = _bundle_member_bytes(bundle, manifest_member)
            actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            process_manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
            rendered_manifest_path = f"bundle:{bundle['archive']}!/{manifest_member}"
        else:
            raise ValueError("existing simulator process manifest is missing and no pinned bundle was supplied")
        if actual_manifest_sha != manifest_sha:
            raise ValueError("existing report simulator identity does not match its process manifest")
        records = process_manifest.get("records") if isinstance(process_manifest, dict) else None
        if (not isinstance(process_manifest, dict) or process_manifest.get("run_id") != run_id
                or process_manifest.get("simulator_exe_path") != identity.get("executable_path")
                or not isinstance(records, list)
                or not any(isinstance(proc, dict)
                           and proc.get("pid") == identity.get("pid")
                           and proc.get("port") == identity.get("port")
                           and proc.get("process_creation_utc") == identity.get("process_creation_utc")
                           for proc in records)):
            raise ValueError("existing report is not bound to the recorded simulator process")
        all_times.extend(float(value) for value in laps)
        reports.append({"run_id": run_id, "report_path": rendered_report_path,
                        "canonical_report_path": str(canonical_report), "report_sha256": report_sha.lower(),
                        "simulator_manifest_path": rendered_manifest_path,
                        "canonical_simulator_manifest_path": str(manifest_path),
                        "simulator_manifest_sha256": manifest_sha,
                        "scored_lap_times_s": [float(value) for value in laps],
                        "warmup_lap_times_s": [float(value) for value in warmup]})
    if len(all_times) != 30:
        raise ValueError("existing validation batch must contain exactly 30 scored laps")
    return {
        "kind": row["kind"], "evaluation_id": evaluation_id,
        "source_recorded_at_utc": evidence_timestamp,
        "evidence_path": str(evidence_path), "evidence_sha256": evidence_sha.lower(),
        "candidate_checkpoint_sha256": expected_checkpoint_sha256,
        "attempts": reports, "scored_laps": len(all_times),
        "median_scored_lap_s": statistics.median(all_times),
        "reused_existing_validation": True,
        **({"bundle_receipt_path": str(bundle["receipt_path"]),
            "bundle_receipt_sha256": bundle["receipt_sha256"],
            "bundle_archive_sha256": bundle["archive_sha256"],
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "bundle_evidence_member": bundle["evidence_member"]} if bundle else {}),
        **clean_reload_fields,
    }


def validate_manifest_existing_validations(
    manifest_payload: dict[str, Any],
    *,
    repo_root: Path,
    verification_context: PPOVerificationContext,
) -> list[dict[str, Any]]:
    """Re-authenticate reused validation rows before runner reconciliation.

    Callers must seal ``verification_context`` before appending events or
    taking actions based on the returned rows.
    """
    if not isinstance(manifest_payload, dict):
        raise ValueError("finalist manifest must be an object")
    finalists = manifest_payload.get("finalists")
    rows = manifest_payload.get("existing_competitive_validations")
    if not isinstance(finalists, list) or not isinstance(rows, list):
        raise ValueError("finalist manifest is missing candidates or reused validation rows")
    allowed = {item.get("checkpoint_sha256") for item in finalists if isinstance(item, dict)}
    seen: set[tuple[str, str]] = set()
    verified_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest reused validation row must be an object")
        kind = row.get("kind")
        checkpoint_sha = row.get("candidate_checkpoint_sha256")
        if checkpoint_sha not in allowed:
            raise ValueError("reused validation is not bound to a selected finalist")
        key = (str(kind), str(checkpoint_sha))
        if key in seen:
            raise ValueError("manifest contains duplicate reused validation rows")
        seen.add(key)
        input_row = {key: row.get(key) for key in (
            "kind", "evidence_path", "evidence_sha256", "bundle_receipt_path", "bundle_receipt_sha256",
        )}
        verified = _verify_existing_competitive_validation(
            input_row, checkpoint_sha, repo_root.resolve(), verification_context,
        )
        if verified != row:
            raise ValueError("manifest reused validation metadata differs from current authenticated evidence")
        verified_rows.append(verified)
    return verified_rows


def _require_post_parent_training_evidence(
    checkpoint: Path,
    lineage_result: dict[str, Any],
    verification_context: Optional[PPOVerificationContext] = None,
    *, allow_seed_exclusion: bool = False,
) -> dict[str, int]:
    current_steps = lineage_result.get("num_timesteps")
    parent_path_value = lineage_result.get("parent_checkpoint")
    if (isinstance(current_steps, bool) or not isinstance(current_steps, int)
            or not isinstance(parent_path_value, str) or not parent_path_value):
        raise ValueError("candidate checkpoint is missing authenticated global-step parent metadata")
    parent_result = verify_ppo_lineage_checkpoint(
        parent_path_value, verification_context=verification_context
    )
    if not isinstance(parent_result, dict):
        raise ValueError("candidate's immediate parent checkpoint is not authenticated")
    parent_steps = parent_result.get("num_timesteps")
    if isinstance(parent_steps, bool) or not isinstance(parent_steps, int):
        raise ValueError("candidate parent is missing its authenticated global step")
    transitions_after_parent = current_steps - parent_steps
    if transitions_after_parent < 0:
        raise ValueError("candidate checkpoint global step precedes its authenticated parent")
    audit_path = checkpoint.parent.parent / "ppo_optimizer_audit.jsonl"
    if not audit_path.is_file():
        raise ValueError("candidate training run is missing its optimizer audit history")
    audits = _jsonl(audit_path, verification_context)
    eligible = [row for row in audits
                if isinstance(row.get("num_timesteps"), int)
                and not isinstance(row.get("num_timesteps"), bool)
                and parent_steps < row["num_timesteps"] <= current_steps]
    unique_steps = {row["num_timesteps"] for row in eligible}
    if any(row.get("metrics_finite") is not True
           or row.get("gradients_finite") is not True
           or row.get("parameters_finite") is not True
           or row.get("optimizer_state_finite") is not True
           for row in eligible):
        raise ValueError("candidate post-parent optimizer audits contain a nonfinite update")
    eligible_for_comparison = transitions_after_parent >= 4096 and len(unique_steps) >= 2
    if not eligible_for_comparison and not allow_seed_exclusion:
        if transitions_after_parent < 4096:
            raise ValueError("candidate probe snapshot is too close to its parent for policy comparison")
        raise ValueError("candidate probe needs at least two post-parent rollout audit records")
    result: dict[str, Any] = {
        "parent_global_timestep": parent_steps,
        "checkpoint_global_timestep": current_steps,
        "post_parent_training_transitions": transitions_after_parent,
        "post_parent_rollout_audit_count": len(unique_steps),
        "post_parent_rollout_audit_steps": sorted(unique_steps),
        "eligible_for_policy_comparison": eligible_for_comparison,
    }
    if not eligible_for_comparison:
        result["exclusion_reason"] = "insufficient_post_parent_training"
    return result


def _verified_campaign_ancestry(
    checkpoint: Path,
    verification_context: Optional[PPOVerificationContext] = None,
) -> tuple[list[str], list[float]]:
    """Return run IDs and actual persisted reward coefficients along PPO parents.

    The PPO verifier authenticates the complete chain; this walk checks each
    saved config so invalidated runs cannot re-enter through a descendant.
    """
    run_ids: list[str] = []
    reward_costs: list[float] = []
    current = checkpoint.resolve()
    seen: set[Path] = set()
    allowed_costs = {0.0, *TIME_COST_VARIANTS.values()}
    while current not in seen:
        seen.add(current)
        verified = verify_ppo_lineage_checkpoint(
            current, verification_context=verification_context
        )
        if not isinstance(verified, dict):
            raise ValueError("candidate PPO ancestry could not be authenticated")
        metadata = verified.get("lineage")
        if not isinstance(metadata, dict):
            raise ValueError("candidate PPO ancestry is missing lineage metadata")
        run_id = metadata.get("run_id")
        if isinstance(run_id, str) and run_id:
            run_ids.append(run_id)
            if run_id in INVALIDATED_CAMPAIGN_RUN_IDS:
                raise ValueError("candidate descends from an invalidated literal per-second campaign run")
        config_path = Path(str(verified.get("effective_config", "")))
        config = (verification_context.read_json(config_path)
                  if verification_context is not None
                  else json.loads(config_path.read_text(encoding="utf-8")))
        reward = config.get("training_only_reward") if isinstance(config, dict) else None
        value = reward.get("time_cost_per_simulated_second") if isinstance(reward, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) not in allowed_costs:
            raise ValueError("candidate PPO ancestry contains a nonapproved time-cost coefficient")
        reward_costs.append(float(value))
        parent = Path(str(verified.get("parent_checkpoint", ""))).resolve()
        parent_sidecar = parent.with_name(f"{parent.stem}.lineage.json")
        if not parent_sidecar.is_file():
            break
        parent_meta = (verification_context.read_json(parent_sidecar)
                       if verification_context is not None
                       else json.loads(parent_sidecar.read_text(encoding="utf-8")))
        if not isinstance(parent_meta, dict) or parent_meta.get("kind") not in {
            "official_ppo_training_checkpoint", "official_training_probe"
        }:
            break
        current = parent
    else:
        raise ValueError("candidate PPO ancestry contains a cycle")
    return run_ids, reward_costs


def build_finalist_manifest(
    state: dict[str, Any],
    evidence: dict[str, Any],
    *,
    repo_root: Path,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Select up to two fastest actually authenticated passing probes.

    This only creates a shortlist for clean final evaluations. It does not
    turn nonranking training probes into competitive scores.
    """
    if (state.get("campaign_id") != CAMPAIGN_ID or state.get("status") != "active"
            or state.get("campaign_plan") != CAMPAIGN_PLAN):
        raise ValueError("finalist selection requires the unchanged active campaign")
    clock = campaign_clock_status(state, utc_now=utc_now, monotonic_ns=monotonic_ns, boot_id=boot_id)
    deadline_policy = clock.get("deadline_policy", {})
    unbounded = deadline_policy.get("mode") == "unbounded_until_readiness"
    validation_request = campaign_validation_request(state) if unbounded else None
    if unbounded:
        if clock.get("phase") != "final_validation_requested" or validation_request is None:
            raise ValueError("unbounded finalist selection requires an active readiness-backed validation request")
    elif clock.get("phase") != "final_validation" or (clock.get("remaining_seconds") or 0) <= 0:
        raise ValueError("finalist manifest may only be built during the active 9h-to-12h final-validation window")
    if not isinstance(evidence, dict) or evidence.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("candidate evidence campaign_id mismatch")
    rows = evidence.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate evidence must contain at least one actual completed probe")
    raw_candidate_count = len(rows)
    if unbounded:
        bound_sha = validation_request.get("candidate_checkpoint_sha256")
        # The readiness request names one exact candidate. Skip unrelated
        # historical rows before recursive archive hashing, but never hide a
        # malformed row that claims either supported SHA field as the bound
        # candidate.
        rows = [row for row in rows if isinstance(row, dict) and bound_sha in {
            row.get("checkpoint_sha256"), row.get("model_sha256"),
        }]
        if not rows:
            raise ValueError("candidate evidence does not contain the exact readiness-request checkpoint")
    excluded_rows = evidence.get("excluded_seed_candidates", [])
    if not isinstance(excluded_rows, list):
        raise ValueError("excluded_seed_candidates must be a list when supplied")
    verification_context = PPOVerificationContext()
    checked = [_probe_candidate(row, repo_root.resolve(), verification_context) for row in rows]
    if any(candidate.get("eligible_for_policy_comparison") is not True for candidate in checked):
        raise ValueError("eligible finalist candidates must each have two finite post-parent rollouts")
    excluded_checked: list[dict[str, Any]] = []
    for row in excluded_rows:
        if not isinstance(row, dict):
            raise ValueError("each excluded seed candidate must be an object")
        # The phase runner carries the parent training run ID rather than a
        # repeated history path in its exclusion ledger; resolve that alias
        # deterministically, then authenticate the same report/history inputs
        # as an ordinary finalist candidate.
        excluded_input = dict(row)
        parent_run_id = excluded_input.get("parent_run_id")
        if (not excluded_input.get("evaluation_history_path")
                and isinstance(parent_run_id, str) and parent_run_id):
            excluded_input["evaluation_history_path"] = str(
                repo_root.resolve() / "logs" / "rl" / f"official_run_{parent_run_id}" / "evaluation_history.jsonl"
            )
        if not excluded_input.get("checkpoint_sha256") and isinstance(excluded_input.get("model_sha256"), str):
            excluded_input["checkpoint_sha256"] = excluded_input["model_sha256"]
        candidate = _probe_candidate(
            excluded_input, repo_root.resolve(), verification_context, allow_seed_exclusion=True,
        )
        if candidate.get("eligible_for_policy_comparison") is not False:
            raise ValueError("excluded seed candidate is eligible and cannot be listed as an exclusion")
        expected_screening = {
            "parent_global_timestep": candidate["parent_global_timestep"],
            "checkpoint_global_timestep": candidate["checkpoint_global_timestep"],
            "post_parent_training_transitions": candidate["post_parent_training_transitions"],
            "post_parent_rollout_audit_count": candidate["post_parent_rollout_audit_count"],
            "post_parent_rollout_audit_steps": candidate["post_parent_rollout_audit_steps"],
        }
        for key, expected_value in expected_screening.items():
            if key in row and row[key] != expected_value:
                raise ValueError(f"excluded seed {key} does not match authenticated checkpoint/audit evidence")
        if "rollout_size" in row and not _exact_int(row["rollout_size"], 2048):
            raise ValueError("excluded seed rollout_size does not match the fixed single-env campaign")
        if "exclusion_reason" in row and row["exclusion_reason"] != candidate["exclusion_reason"]:
            raise ValueError("excluded seed reason does not match authenticated training evidence")
        excluded_checked.append(candidate)
    seen_evaluations: set[str] = set()
    seen_checkpoints: set[str] = set()
    for candidate in [*checked, *excluded_checked]:
        if candidate["evaluation_id"] in seen_evaluations or candidate["checkpoint_sha256"] in seen_checkpoints:
            raise ValueError("candidate and excluded-seed evidence contains duplicate evaluation or checkpoint identities")
        seen_evaluations.add(candidate["evaluation_id"])
        seen_checkpoints.add(candidate["checkpoint_sha256"])
    by_sha: dict[str, dict[str, Any]] = {}
    for candidate in checked:
        previous = by_sha.get(candidate["checkpoint_sha256"])
        if previous is None or candidate["median_scored_lap_s"] < previous["median_scored_lap_s"]:
            by_sha[candidate["checkpoint_sha256"]] = candidate
    ranked = sorted(by_sha.values(), key=lambda item: (item["median_scored_lap_s"], item["checkpoint_sha256"]))
    if unbounded:
        bound = next((candidate for candidate in by_sha.values()
                      if candidate["checkpoint_sha256"] == validation_request["candidate_checkpoint_sha256"]), None)
        if (bound is None
                or bound["evaluation_id"] != validation_request["evaluation_id"]
                or bound["probe_report_sha256"] != validation_request["probe_report_sha256"]):
            raise ValueError("finalist evidence does not contain the exact candidate/report bound by the readiness request")
        # The readiness request is bound to one exact checkpoint/report. Do
        # not append a second unvalidated probe while reusing batch proofs for
        # that selected candidate.
        finalists = [bound]
    else:
        finalists = ranked[:2]
    existing_rows = evidence.get("existing_competitive_validations", [])
    if not isinstance(existing_rows, list):
        raise ValueError("existing_competitive_validations must be a list when supplied")
    reused_validations: list[dict[str, Any]] = []
    seen_validation_kinds: set[tuple[str, str]] = set()
    finalist_shas = {candidate["checkpoint_sha256"] for candidate in finalists}
    for row in existing_rows:
        if not isinstance(row, dict):
            raise ValueError("each existing validation reference must be an object")
        kind = row.get("kind")
        # Keep the reusable evidence intentionally scoped to the selected
        # finalist(s); unrelated historical batches cannot satisfy a slot.
        evidence_path = _resolve_in_logs(row.get("evidence_path"), repo_root.resolve())
        if evidence_path is None or not evidence_path.is_file():
            raise ValueError("existing validation evidence path must resolve inside logs/rl")
        evidence_doc = verification_context.read_json(evidence_path)
        bound_sha = None
        if isinstance(evidence_doc, dict):
            if kind == "competitive":
                bound_sha = evidence_doc.get("model_sha256")
            elif kind == "clean_reload":
                bound_sha = evidence_doc.get("requested_and_loaded_model_sha256")
        if not isinstance(bound_sha, str) or bound_sha not in finalist_shas:
            raise ValueError("existing validation does not bind one of the selected finalist checkpoints")
        identity = (str(kind), bound_sha)
        if identity in seen_validation_kinds:
            raise ValueError("duplicate existing validation kind for a selected checkpoint")
        seen_validation_kinds.add(identity)
        reused_validations.append(_verify_existing_competitive_validation(
            row, bound_sha, repo_root.resolve(), verification_context,
        ))
    now = (utc_now or datetime.now().astimezone()).isoformat()
    if not verification_context.seal():
        raise RuntimeError("finalist candidate evidence changed during verification")
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "generated_at": now,
        "deadline_policy": deadline_policy,
        "validation_request_id": validation_request.get("request_id") if validation_request else None,
        "readiness_candidate_checkpoint_sha256": (
            validation_request.get("candidate_checkpoint_sha256") if validation_request else None
        ),
        "selection_policy": "At most two unique authenticated checkpoints with the fastest passing 1+3 training-probe medians; probes are nonranking readiness evidence only, and final acceptance requires the standard clean competitive protocol.",
        "final_validation": {
            "attempts_per_finalist": 3,
            "warmup_laps_per_attempt": 1,
            "scored_laps_per_attempt": 10,
            "zero_collisions_required": True,
            "clean_reload_required": True,
        },
        "evidence_candidates_count": raw_candidate_count,
        "authenticated_candidate_count": len(checked),
        "finalists": finalists,
        "existing_competitive_validations": reused_validations,
        "excluded_seed_candidates": [
            {key: candidate[key] for key in (
                "evaluation_id", "probe_run_id", "checkpoint_sha256", "probe_report_sha256",
                "parent_global_timestep", "checkpoint_global_timestep",
                "post_parent_training_transitions", "post_parent_rollout_audit_count",
                "post_parent_rollout_audit_steps", "exclusion_reason",
            )}
            for candidate in excluded_checked
        ],
    }


def _atomic_create(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link creation is atomic and refuses to overwrite an existing
        # manifest, including when two freeze handlers race.
        os.link(temporary, path)
        return hashlib.sha256(encoded).hexdigest()
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=default_state_path())
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--at-utc", help="test/replay timestamp; omitted uses current UTC")
    args = parser.parse_args(argv)
    state = load_campaign_state(args.state)
    if state is None:
        raise FileNotFoundError(f"campaign state not found: {args.state}")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("candidate evidence must be a JSON object")
    at = datetime.fromisoformat(args.at_utc.replace("Z", "+00:00")) if args.at_utc else None
    manifest = build_finalist_manifest(state, evidence, repo_root=args.repo_root, utc_now=at)
    digest = _atomic_create(args.output, manifest)
    print(json.dumps({"output": str(args.output), "sha256": digest, "finalists": manifest["finalists"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
