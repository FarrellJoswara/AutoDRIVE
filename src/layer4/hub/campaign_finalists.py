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
import tempfile
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
        raw = Path(value)
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
    screening = _require_post_parent_training_evidence(model, lineage, verification_context)
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


def _require_post_parent_training_evidence(
    checkpoint: Path,
    lineage_result: dict[str, Any],
    verification_context: Optional[PPOVerificationContext] = None,
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
    if transitions_after_parent < 4096:
        raise ValueError("candidate probe snapshot is too close to its parent for policy comparison")
    audit_path = checkpoint.parent.parent / "ppo_optimizer_audit.jsonl"
    if not audit_path.is_file():
        raise ValueError("candidate training run is missing its optimizer audit history")
    audits = _jsonl(audit_path, verification_context)
    eligible = [row for row in audits
                if isinstance(row.get("num_timesteps"), int)
                and not isinstance(row.get("num_timesteps"), bool)
                and parent_steps < row["num_timesteps"] <= current_steps]
    unique_steps = {row["num_timesteps"] for row in eligible}
    if len(unique_steps) < 2:
        raise ValueError("candidate probe needs at least two post-parent rollout audit records")
    if any(row.get("metrics_finite") is not True
           or row.get("gradients_finite") is not True
           or row.get("parameters_finite") is not True
           or row.get("optimizer_state_finite") is not True
           for row in eligible):
        raise ValueError("candidate post-parent optimizer audits contain a nonfinite update")
    return {
        "parent_global_timestep": parent_steps,
        "checkpoint_global_timestep": current_steps,
        "post_parent_training_transitions": transitions_after_parent,
        "post_parent_rollout_audit_count": len(unique_steps),
    }


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
    verification_context = PPOVerificationContext()
    checked = [_probe_candidate(row, repo_root.resolve(), verification_context) for row in rows]
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
        finalists = [bound]
        finalists.extend(candidate for candidate in ranked
                          if candidate["checkpoint_sha256"] != bound["checkpoint_sha256"])
        finalists = sorted(finalists[:2], key=lambda item: (item["median_scored_lap_s"], item["checkpoint_sha256"]))
    else:
        finalists = ranked[:2]
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
        "evidence_candidates_count": len(checked),
        "finalists": finalists,
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
