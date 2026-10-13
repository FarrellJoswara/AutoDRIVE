"""Persisted clock and admission rules for a bounded official PPO campaign.

The monitor is intentionally separate from the Mission Control process. This
module contains only state-file and clock policy so the API can reject work
even when the monitor has just restarted.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import statistics
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional


SCHEMA_VERSION = 1
CAMPAIGN_HOURS = 12
TRAINING_HOURS = 9
CAMPAIGN_ID = "official-ppo-20261012-bc6144"
CAMPAIGN_STATE_PATH_WINDOWS = (
    "C:/Users/farre/AppData/Local/AiCarCampaigns/"
    f"{CAMPAIGN_ID}/active_campaign.json"
)
CAMPAIGN_STATE_PATH_WSL = (
    "/mnt/c/Users/farre/AppData/Local/AiCarCampaigns/"
    f"{CAMPAIGN_ID}/active_campaign.json"
)
DEFAULT_RELATIVE_PATH = Path("logs/rl/official_campaign/active_campaign.json")
STATE_ENV = "AICAR_CAMPAIGN_STATE_PATH"
CAMPAIGN_PLAN = {
    "learning_rate": 3e-5,
    # The independent supervisor owns campaign termination. A zero SB3 target
    # means run until the 9-hour training freeze rather than a guessed step cap.
    "total_timesteps": 0,
    "stop_on_plateau": False,
    "max_duration_hours": 0.0,
    "behavior_cloning_enabled": False,
    "behavior_cloning_only": False,
    "observation_profile": "official_sensors_camera",
    "simulator_backend": "windows_external",
    "throttle_mode": "bidirectional",
    "negative_throttle_mode": "allow",
    "steering_mode": "normal",
    "target_kl": 0.01,
    "exploration_profile": "bc_campaign",
    "exploration_initial_std": [0.01, 0.05],
    "exploration_min_std": [0.005, 0.02],
    "exploration_max_std": [0.01, 0.05],
    "ppo_settings": {
        "n_envs": 1,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.9995,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": "cuda",
    },
    "automatic_evaluation_enabled": True,
    "automatic_evaluation_mode": "training_probe",
    "automatic_evaluation_interval": 50000,
    "checkpoint_every_transitions": 10000,
    "probe_interval_transitions_approx": 50000,
    "initial_resume_checkpoint": {
        "path_windows": "C:/Users/farre/Documents/Github/AiCar/logs/rl/official_run_b1fc871c2bdb/checkpoints/post_optimizer_update_00000004_steps_000000006144.zip",
        "path_wsl": "/mnt/c/Users/farre/Documents/Github/AiCar/logs/rl/official_run_b1fc871c2bdb/checkpoints/post_optimizer_update_00000004_steps_000000006144.zip",
        "sha256": "638a90cacf4cf1e7c49371af8dcafaea8b8044360627bac06cccd9a5f8a16b73",
        "num_timesteps": 6144,
        "optimizer_updates": 4,
        "parent_checkpoint_sha256": "76205e8c4e617ce9eacfa46e1d02de22b04ec9df207c56d2d78dad23f70e5b63",
        "bc_ancestor_checkpoint_sha256": "6f0e836f2c22484d7f0c1e96196ca3aed030a8da62855796ee01b478e5064fe3",
        "training_probe_evaluation_id": "de48af1d8e3c4971",
        "training_probe_report": "logs/rl/official_run_5060876adcc1/evaluation.json",
        "training_probe_median_scored_lap_s": 105.50460052490234,
        "training_probe_scored_lap_times_s": [105.50460052490234, 105.98040008544922, 105.2177963256836],
        "training_probe_collisions": 0,
        "competitive_ranking": False,
    },
    "single_environment_only": True,
    "training_freeze_hours": 9,
    "hard_deadline_hours": 12,
    "decision_schedule": [
        {"elapsed_hours": [0, 2], "decision": "hold baseline fixed; record telemetry, wall time, and compute cost"},
        {"elapsed_hours": [2, 3], "decision": "review time_cost and compute/time cost; consider only that one variable"},
        {"elapsed_hour": 3, "decision": "TBC telemetry/trend review; non-exhaustive; keep one environment"},
        {"elapsed_hour": 5, "decision": "TBC telemetry/trend review; non-exhaustive; keep one environment"},
        {"elapsed_hours": [5, 9], "decision": "select a branch from best validated trend; change one variable at a time"},
    ],
    "training_probe_protocol": {
        "laps_by_environment": [1, 3],
        "attempts_per_checkpoint": 1,
        "stop_on_first_collision_including_warmup": True,
        "ranking": False,
        "simulation_timeout_seconds": 240,
        "wall_timeout_seconds": 240,
        "frontier_stagnation_seconds": 5,
    },
    "final_validation_protocol": {
        "window_hours": 3,
        "finalist_count": 2,
        "attempts_per_finalist": 3,
        "warmup_laps_per_attempt": 1,
        "scored_laps_per_attempt": 10,
        "zero_collisions_required": True,
    },
    "reward_policy": "preserve existing smoke reward configuration unchanged",
    "reward_settings": {
        "time_cost_per_simulated_second": 0.0,
        "collision_penalty_magnitude": 200.0,
        "collision_reward_percent": 0.0,
        "lap_completion_reward": 0.0,
        "failed_episode_penalty": 200.0,
        "backward_speed_penalty_scale": 10.0,
        "episode_failure_reward_percent": 0.0,
    },
}

# These experiment IDs denote reward penalties calibrated as 3 or 5 points
# over the BC reference lap, not per-second coefficients. The reward function
# consumes simulated seconds, so convert the lap penalty to its coefficient.
BC_TIME_COST_CALIBRATION_LAP_S = 110.255550384521
TIME_COST_3_PER_SIMULATED_SECOND = 3.0 / BC_TIME_COST_CALIBRATION_LAP_S
TIME_COST_5_PER_SIMULATED_SECOND = 5.0 / BC_TIME_COST_CALIBRATION_LAP_S
TIME_COST_VARIANTS = {
    "time_cost_3_calibrated_retry": TIME_COST_3_PER_SIMULATED_SECOND,
    "time_cost_5_calibrated_retry": TIME_COST_5_PER_SIMULATED_SECOND,
}
INVALIDATED_TIME_COST_VARIANTS = {
    "time_cost_3": "declared with 3 reward units per simulated second instead of 3 per BC calibration lap",
    "time_cost_5": "declared with 5 reward units per simulated second instead of 5 per BC calibration lap",
}
INVALIDATED_CAMPAIGN_RUN_IDS = frozenset({"dcda7cefacef", "1bd61c474512"})
CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE = {
    "run_id": "381e8446cc36",
    "experiment_id": "continue_best",
    "stop_reason": "operator_stop",
    "phase_runner_reason": "five_hour_pace_phase_boundary",
    "global_timestep": 204800,
    "checkpoint_path": "logs/rl/official_run_381e8446cc36/checkpoints/post_optimizer_update_00000002_steps_000000204800.zip",
    "checkpoint_sha256": "a44735dee2bc57c9268604e6c439e99b385dc6f9c708b76045e84f18cfbc666f",
    "parent_probe_sha256": "3a51ed5ffea719eb62240f720efcbfc78bcaae11f21e34271093cc3cbe7ebda7",
    "manager_config_sha256": "40a5079486c2266573fe29f1b78cba24b498704f2b6ae11adaf56283d8570511",
    "effective_config_sha256": "700e7709645c1c3c20a0eeff1512b6c995a31639adf450b6e1bd9366d0da79ee",
}

EXPERIMENT_METADATA_FIELDS = frozenset({
    "campaign_experiment_id",
    "campaign_experiment_phase",
    "campaign_experiment_config_sha256",
})
_CAMPAIGN_EXPERIMENTS = {
    "time_cost_3_calibrated_retry": {"phase": "pace_experiments", "hours": (2.0, 5.0), "time_cost": TIME_COST_3_PER_SIMULATED_SECOND},
    "time_cost_5_calibrated_retry": {"phase": "pace_experiments", "hours": (2.0, 5.0), "time_cost": TIME_COST_5_PER_SIMULATED_SECOND},
    "lr_1e4_best_stable": {"phase": "pace_experiments", "hours": (2.0, 5.0)},
    "continue_best": {"phase": "best_validated_trend", "hours": (5.0, 9.0)},
    # A one-time, no-hyperparameter-change recovery for the exact checkpoint
    # cut off by the phase runner's known 5h-boundary bug.
    "continue_best_boundary_recovery": {"phase": "best_validated_trend", "hours": (5.0, 9.0)},
    "regression_recovery": {"phase": "best_validated_trend", "hours": (5.0, 9.0), "learning_rate": 1e-5},
    "plateau_lr_conservative": {"phase": "best_validated_trend", "hours": (5.0, 9.0), "learning_rate": 1e-5},
    "plateau_lr_healthy": {"phase": "best_validated_trend", "hours": (5.0, 9.0), "learning_rate": 1e-4},
}


def default_state_path() -> Path:
    configured = os.environ.get(STATE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / DEFAULT_RELATIVE_PATH).resolve()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"campaign {field} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"campaign {field} is not a valid ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"campaign {field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _windows_boot_id() -> str:
    ticks = int(ctypes.windll.kernel32.GetTickCount64())  # type: ignore[attr-defined]
    estimated_boot_epoch = int(time.time() - ticks / 1000.0)
    # A system clock correction that moves this estimate materially causes a
    # fail-closed restart. Rounding only absorbs ordinary sampling jitter.
    return f"windows-boot-{estimated_boot_epoch // 60 * 60}"


def current_boot_id() -> str:
    if os.name == "nt":
        return _windows_boot_id()
    boot_file = Path("/proc/sys/kernel/random/boot_id")
    try:
        return "linux-boot-" + boot_file.read_text(encoding="ascii").strip()
    except OSError:
        # Unknown boot identity is unsafe for persisted monotonic deadlines.
        return "unknown-boot"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"campaign state at {path} must be a JSON object")
    return value


def effective_campaign_deadline_policy(state: dict[str, Any]) -> dict[str, Any]:
    """Return the effective deadline policy without rewriting original clock data.

    A user-authorized unbounded amendment is represented by a new state object
    and an appended event. The original start, freeze, hard-deadline, and
    monotonic values remain immutable historical evidence.
    """
    amendment = state.get("deadline_policy_amendment")
    if amendment is None:
        return {
            "mode": "fixed_9h_12h",
            "authorized": False,
            "training_freeze_at_utc": state.get("training_freeze_at_utc"),
            "hard_deadline_at_utc": state.get("hard_deadline_at_utc"),
            "amendment": None,
        }
    if not isinstance(amendment, dict):
        raise ValueError("campaign deadline policy amendment must be an object")
    if amendment.get("policy") != "unbounded_until_readiness":
        raise ValueError("campaign deadline amendment has an unsupported policy")
    if amendment.get("authorization") != "user_override":
        raise ValueError("campaign deadline amendment lacks explicit user authorization")
    if not isinstance(amendment.get("reason"), str) or not amendment["reason"].strip():
        raise ValueError("campaign deadline amendment requires its authorization reason")
    amended_at = _parse_utc(amendment.get("authorized_at_utc"), "deadline amendment authorized_at_utc")
    if amendment.get("original_training_freeze_at_utc") != state.get("training_freeze_at_utc"):
        raise ValueError("campaign deadline amendment does not preserve the original training freeze")
    if amendment.get("original_hard_deadline_at_utc") != state.get("hard_deadline_at_utc"):
        raise ValueError("campaign deadline amendment does not preserve the original hard deadline")
    if amendment.get("effective_training_freeze_at_utc") is not None:
        raise ValueError("unbounded campaign amendment must disable the effective training freeze")
    if amendment.get("effective_hard_deadline_at_utc") is not None:
        raise ValueError("unbounded campaign amendment must disable the effective hard deadline")
    prior_revision = amendment.get("prior_revision")
    if isinstance(prior_revision, bool) or not isinstance(prior_revision, int) or prior_revision < 1:
        raise ValueError("campaign deadline amendment has an invalid prior revision")
    prior_hash = amendment.get("prior_state_sha256")
    if not _valid_sha256(prior_hash):
        raise ValueError("campaign deadline amendment has an invalid prior state hash")
    event = amendment.get("event")
    if not isinstance(event, dict) or event.get("event") != "campaign_deadline_policy_amended":
        raise ValueError("campaign deadline amendment is missing its audit event")
    matching_events = [row for row in state.get("events", [])
                       if isinstance(row, dict)
                       and row.get("event") == "campaign_deadline_policy_amended"
                       and row.get("amendment_id") == amendment.get("amendment_id")]
    if len(matching_events) != 1 or matching_events[0] != event:
        raise ValueError("campaign deadline amendment audit event is missing or ambiguous")
    if amendment.get("amendment_id") != event.get("amendment_id"):
        raise ValueError("campaign deadline amendment ID does not match its audit event")
    return {
        "mode": "unbounded_until_readiness",
        "authorized": True,
        "authorized_at_utc": amended_at.isoformat(),
        "training_freeze_at_utc": None,
        "hard_deadline_at_utc": None,
        "original_training_freeze_at_utc": state.get("training_freeze_at_utc"),
        "original_hard_deadline_at_utc": state.get("hard_deadline_at_utc"),
        "amendment": amendment,
    }


def campaign_validation_request(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return an active readiness-backed validation request, if one exists."""
    policy = effective_campaign_deadline_policy(state)
    if policy["mode"] != "unbounded_until_readiness":
        return None
    amendment_id = policy["amendment"]["amendment_id"]
    closed = {
        row.get("request_id") for row in state.get("events", [])
        if isinstance(row, dict) and row.get("event") == "campaign_validation_request_closed"
    }
    for row in reversed(state.get("events", [])):
        if not isinstance(row, dict) or row.get("event") != "campaign_validation_requested":
            continue
        if row.get("request_id") in closed:
            continue
        if (row.get("authorization") != "readiness_gate"
                or row.get("policy") != "unbounded_until_readiness"
                or row.get("amendment_id") != amendment_id
                or not _valid_sha256(row.get("candidate_checkpoint_sha256"))
                or not _valid_sha256(row.get("probe_report_sha256"))
                or not isinstance(row.get("evaluation_id"), str)
                or not row.get("evaluation_id")):
            raise ValueError("campaign validation request is malformed or not bound to the current amendment")
        readiness = row.get("readiness_evidence")
        recommendation = (
            readiness.get("sol_readiness_recommendation")
            if isinstance(readiness, dict) else None
        )
        if (not isinstance(readiness, dict)
                or readiness.get("stable_candidate_verified") is not True
                or not _valid_sha256(readiness.get("source_evidence_sha256"))
                or not _valid_sha256(readiness.get("recent_audit_evidence_sha256"))
                or readiness.get("recent_audits_finite") is not True
                or readiness.get("sensor_health_passed") is not True
                or readiness.get("zero_collision_probes") is not True
                or isinstance(readiness.get("passing_nonregressing_probe_count"), bool)
                or not isinstance(readiness.get("passing_nonregressing_probe_count"), int)
                or readiness["passing_nonregressing_probe_count"] < 2
                or not isinstance(recommendation, dict)
                or recommendation.get("decision") != "request_final_validation"
                or recommendation.get("reviewer") != "sol"
                or not isinstance(recommendation.get("reason"), str)
                or not recommendation["reason"].strip()
                or recommendation.get("candidate_checkpoint_sha256") != row.get("candidate_checkpoint_sha256")
                or not _valid_sha256(recommendation.get("recommendation_evidence_sha256"))):
            raise ValueError("campaign validation request lacks stable candidate readiness evidence")
        return row
    return None


def _maintenance_record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _validate_hub_maintenance_history(state: dict[str, Any]) -> None:
    history = state.get("hub_maintenance_history", [])
    if not isinstance(history, list):
        raise ValueError("campaign hub maintenance history must be a list")
    events = state.get("events", [])
    if not isinstance(events, list):
        raise ValueError("campaign event history must be a list")
    archived_ids: set[str] = set()
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError("campaign hub maintenance history entry is invalid")
        maintenance_id = entry.get("maintenance_id")
        record = entry.get("maintenance_record")
        archive_event = entry.get("archive_event")
        if (not isinstance(maintenance_id, str) or not re.fullmatch(r"[0-9a-f]{32}", maintenance_id)
                or maintenance_id in archived_ids or not isinstance(record, dict)
                or record.get("maintenance_id") != maintenance_id
                or not isinstance(record.get("completed"), dict)
                or not isinstance(archive_event, dict)):
            raise ValueError("campaign hub maintenance history binding is invalid")
        archived_ids.add(maintenance_id)
        record_state = dict(state)
        record_state["hub_maintenance"] = record
        record_state["hub_maintenance_history"] = []
        record_state["events"] = [
            row for row in events if not (
                isinstance(row, dict)
                and row.get("event") == "campaign_hub_maintenance_archived"
            )
        ]
        if campaign_hub_maintenance_status(record_state) is not None:
            raise ValueError("archived campaign hub maintenance is not completed")
        matching = [row for row in events if isinstance(row, dict)
                    and row.get("event") == "campaign_hub_maintenance_archived"
                    and row.get("maintenance_id") == maintenance_id]
        completed = record["completed"]
        if (len(matching) != 1 or matching[0] != archive_event
                or archive_event.get("maintenance_record_sha256") != _maintenance_record_sha256(record)
                or archive_event.get("completed_event_sha256") != _maintenance_record_sha256(completed)
                or archive_event.get("campaign_id") != state.get("campaign_id")):
            raise ValueError("campaign hub maintenance archive event is missing or mismatched")
        completed_positions = [index for index, row in enumerate(events)
                               if isinstance(row, dict)
                               and row.get("event") == "campaign_hub_maintenance_completed"
                               and row.get("maintenance_id") == maintenance_id]
        archive_position = events.index(archive_event)
        next_id = archive_event.get("next_maintenance_id")
        next_starts = [index for index, row in enumerate(events)
                       if isinstance(row, dict)
                       and row.get("event") == "campaign_hub_maintenance_started"
                       and row.get("maintenance_id") == next_id]
        if (len(completed_positions) != 1 or archive_position <= completed_positions[0]
                or not isinstance(next_id, str) or len(next_starts) != 1
                or next_starts[0] <= archive_position):
            raise ValueError("campaign hub maintenance archive event is out of order")
    archive_events = [row for row in events if isinstance(row, dict)
                      and row.get("event") == "campaign_hub_maintenance_archived"]
    if len(archive_events) != len(history):
        raise ValueError("campaign hub maintenance archive history is incomplete")


def campaign_hub_maintenance_status(state: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Return a validated in-progress hub handoff, if one is recorded."""
    if state is None:
        return None
    _validate_hub_maintenance_history(state)
    maintenance = state.get("hub_maintenance")
    if maintenance is None:
        return None
    if not isinstance(maintenance, dict):
        raise ValueError("campaign hub maintenance must be an object")
    intent = maintenance.get("intent")
    if not isinstance(intent, dict) or intent.get("event") != "campaign_hub_maintenance_started":
        raise ValueError("campaign hub maintenance is missing its start event")
    _validate_hub_identity(
        intent.get("old_hub_identity"), "old_hub_identity",
        allow_legacy_source_unavailable=True,
    )
    if maintenance.get("maintenance_id") != intent.get("maintenance_id"):
        raise ValueError("campaign hub maintenance ID does not match its start event")
    if maintenance.get("completed") is not None:
        completed = maintenance["completed"]
        if not isinstance(completed, dict) or completed.get("event") != "campaign_hub_maintenance_completed":
            raise ValueError("campaign hub maintenance completion record is invalid")
        completed_events = [row for row in state.get("events", []) if isinstance(row, dict)
                            and row.get("event") == "campaign_hub_maintenance_completed"
                            and row.get("maintenance_id") == maintenance.get("maintenance_id")]
        if len(completed_events) != 1 or completed_events[0] != completed:
            raise ValueError("campaign hub maintenance completion event is missing or ambiguous")
        return None
    events = state.get("events", [])
    starts = [row for row in events if isinstance(row, dict)
              and row.get("event") == "campaign_hub_maintenance_started"
              and row.get("maintenance_id") == maintenance.get("maintenance_id")]
    finishes = [row for row in events if isinstance(row, dict)
                and row.get("event") == "campaign_hub_maintenance_completed"
                and row.get("maintenance_id") == maintenance.get("maintenance_id")]
    if len(starts) != 1 or starts[0] != intent or len(finishes) > 1:
        raise ValueError("campaign hub maintenance event history is missing, completed, or ambiguous")
    if finishes:
        if maintenance.get("completed") != finishes[0]:
            raise ValueError("campaign hub maintenance completion does not match its audit event")
    if (not _valid_sha256(intent.get("checkpoint_sha256"))
            or not _valid_sha256(intent.get("effective_config_sha256"))
            or not _valid_sha256(intent.get("prior_state_sha256"))):
        raise ValueError("campaign hub maintenance hashes are invalid")
    authorization = maintenance.get("resume_authorization")
    supersessions = [row for row in events if isinstance(row, dict)
                     and row.get("event") == "campaign_hub_maintenance_resume_authorization_superseded"
                     and row.get("maintenance_id") == maintenance.get("maintenance_id")]
    if authorization is not None:
        auth_events = [row for row in events if isinstance(row, dict)
                       and row.get("event") == "campaign_hub_maintenance_resume_authorized"
                       and row.get("maintenance_id") == maintenance.get("maintenance_id")]
        if len(auth_events) != 1:
            raise ValueError("campaign maintenance resume authorization is missing or ambiguous")
        current_authorization = auth_events[0]
        _validate_maintenance_resume_authorization(current_authorization, intent)
        initial_authorization_position = events.index(current_authorization)
        if any(events.index(row) <= initial_authorization_position for row in supersessions):
            raise ValueError("maintenance authorization supersession is out of event order")
        for supersession in supersessions:
            if (supersession.get("campaign_id") != intent.get("campaign_id")
                    or supersession.get("amendment_id") != intent.get("amendment_id")):
                raise ValueError("campaign maintenance authorization supersession binding is invalid")
            _validate_hub_identity(
                supersession.get("prior_new_hub_identity"), "prior_new_hub_identity"
            )
            replacement = supersession.get("replacement_authorization")
            if (supersession.get("prior_authorization_sha256") != _maintenance_authorization_sha256(current_authorization)
                    or supersession.get("prior_new_hub_identity") != current_authorization.get("new_hub_identity")
                    or not isinstance(replacement, dict)):
                raise ValueError("campaign maintenance authorization supersession chain is invalid")
            _validate_maintenance_resume_authorization(replacement, intent)
            if _maintenance_authorization_binding(replacement) != _maintenance_authorization_binding(current_authorization):
                raise ValueError("campaign maintenance authorization supersession changed the checkpoint or config")
            if supersession.get("replacement_authorization") != replacement:
                raise ValueError("campaign maintenance authorization supersession is malformed")
            current_authorization = replacement
        if current_authorization != authorization:
            raise ValueError("campaign maintenance resume authorization history does not match current authority")
        claims = [row for row in events if isinstance(row, dict)
                  and row.get("event") == "campaign_hub_maintenance_resume_claimed"
                  and row.get("maintenance_id") == maintenance.get("maintenance_id")]
        if maintenance.get("resume_claimed") is True:
            if (len(claims) != 1
                    or claims[0].get("resume_config_sha256") != authorization.get("resume_config_sha256")
                    or claims[0].get("new_hub_identity") != authorization.get("new_hub_identity")
                    or not isinstance(claims[0].get("resumed_run_id"), str)
                    or not re.fullmatch(r"[0-9a-f]{12}", claims[0]["resumed_run_id"])):
                raise ValueError("campaign maintenance resume claim is missing or ambiguous")
            claim_position = events.index(claims[0])
            if (claim_position <= initial_authorization_position
                    or any(events.index(row) > claim_position for row in supersessions)):
                raise ValueError("maintenance authorization cannot be superseded after its resume is claimed")
        elif maintenance.get("resume_claimed") is not False or claims:
            raise ValueError("campaign maintenance resume claim state is invalid")
    else:
        if maintenance.get("resume_claimed") is not False:
            raise ValueError("campaign maintenance cannot be claimed without an authorization")
        if supersessions or any(
            isinstance(row, dict)
            and row.get("maintenance_id") == maintenance.get("maintenance_id")
            and row.get("event") in {
                "campaign_hub_maintenance_resume_authorized",
                "campaign_hub_maintenance_resume_claimed",
            }
            for row in events
        ):
            raise ValueError("campaign maintenance authorization history has no current authorization")
    return maintenance


def begin_campaign_hub_maintenance(
    state_path: str | Path,
    *,
    maintenance_id: str,
    trainer_run_id: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    global_timestep: int,
    effective_config_sha256: str,
    training_contract_sha256: str,
    old_hub_identity: dict[str, Any],
    reason: str,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Record an append-only, checkpoint-bound hub maintenance intent.

    This helper performs no process actions. A fresh hub must honor the active
    intent by refusing new work until ``complete_campaign_hub_maintenance``
    records an authenticated resume.
    """
    import re

    from src.layer3.bc_artifact import sha256_file, verify_ppo_lineage_checkpoint

    if not re.fullmatch(r"[0-9a-f]{32}", maintenance_id):
        raise ValueError("maintenance_id must be 32 lowercase hexadecimal characters")
    if not re.fullmatch(r"[0-9a-f]{12}", trainer_run_id):
        raise ValueError("trainer_run_id must be a 12-character lowercase run ID")
    if isinstance(global_timestep, bool) or not isinstance(global_timestep, int) or global_timestep < 0:
        raise ValueError("global_timestep must be a nonnegative integer")
    if (not _valid_sha256(checkpoint_sha256) or not _valid_sha256(effective_config_sha256)
            or not _valid_sha256(training_contract_sha256)):
        raise ValueError("maintenance checkpoint and config hashes must be SHA-256")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("maintenance intent requires a reason")
    _validate_hub_identity(
        old_hub_identity, "old_hub_identity", allow_legacy_source_unavailable=True
    )
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None or state.get("status") != "active":
        raise RuntimeError("hub maintenance requires an active campaign")
    policy = effective_campaign_deadline_policy(state)
    if policy.get("mode") != "unbounded_until_readiness":
        raise RuntimeError("hub maintenance requires the explicit unbounded policy")
    if campaign_validation_request(state) is not None:
        raise RuntimeError("hub maintenance cannot start during final validation")
    existing = state.get("hub_maintenance")
    archive_existing: Optional[dict[str, Any]] = None
    if existing is not None:
        active = campaign_hub_maintenance_status(state)
        if isinstance(existing, dict) and existing.get("maintenance_id") == maintenance_id:
            if existing.get("completed") is not None:
                if _maintenance_intent_matches(
                    existing.get("intent", {}), trainer_run_id, checkpoint_path, checkpoint_sha256,
                    global_timestep, effective_config_sha256, training_contract_sha256,
                    old_hub_identity, reason,
                ):
                    return state
                raise RuntimeError("completed campaign maintenance ID cannot be reused with different intent")
            if active and _maintenance_intent_matches(
                active["intent"], trainer_run_id, checkpoint_path, checkpoint_sha256,
                global_timestep, effective_config_sha256, training_contract_sha256,
                old_hub_identity, reason,
            ):
                return state
            raise RuntimeError("another campaign hub maintenance handoff is already active")
        if isinstance(existing, dict) and existing.get("completed") is not None and active is None:
            archive_existing = existing
        else:
            raise RuntimeError("another campaign hub maintenance handoff is already active")

    prior_ids = {row.get("maintenance_id") for row in state.get("events", [])
                 if isinstance(row, dict)
                 and row.get("event") == "campaign_hub_maintenance_started"}
    if maintenance_id in prior_ids:
        raise RuntimeError("campaign maintenance ID has already been used")

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha256.lower():
        raise ValueError("maintenance checkpoint path does not match the supplied SHA-256")
    expected_run_folder = f"official_run_{trainer_run_id}"
    if expected_run_folder not in checkpoint.parts:
        raise ValueError("maintenance checkpoint is outside the declared trainer run folder")
    lineage = verify_ppo_lineage_checkpoint(checkpoint)
    if (not lineage or lineage.get("checkpoint_sha256") != checkpoint_sha256.lower()
            or lineage.get("effective_config_sha256") != effective_config_sha256.lower()
            or lineage.get("num_timesteps") != global_timestep
            or not isinstance(lineage.get("lineage"), dict)
            or lineage["lineage"].get("post_optimizer_update") is not True):
        raise ValueError("maintenance checkpoint is not an authenticated post-optimizer PPO checkpoint at the declared step/config")
    try:
        manager_config_path = checkpoint.parent.parent / "manager_config.json"
        manager_config_record = json.loads(manager_config_path.read_text(encoding="utf-8"))
        if manager_config_record.get("run_id") != trainer_run_id:
            raise ValueError("manager run id mismatch")
        source_config = manager_config_record.get("config")
        if not isinstance(source_config, dict):
            raise ValueError("manager config missing settings")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("maintenance checkpoint's manager config cannot be read or authenticated") from exc
    if campaign_training_contract_sha256(source_config) != training_contract_sha256.lower():
        raise ValueError("declared training contract does not match the checkpoint effective config")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    original_clock = _immutable_clock_fields(state)
    prior_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if archive_existing is not None:
        completed = archive_existing.get("completed")
        if not isinstance(completed, dict):
            raise RuntimeError("only completed campaign maintenance can be archived")
        archive_event = {
            "timestamp_utc": now.isoformat(), "monotonic_ns": mono,
            "event": "campaign_hub_maintenance_archived",
            "maintenance_id": archive_existing["maintenance_id"],
            "campaign_id": state.get("campaign_id"),
            "maintenance_record_sha256": _maintenance_record_sha256(archive_existing),
            "completed_event_sha256": _maintenance_record_sha256(completed),
            "next_maintenance_id": maintenance_id,
        }
        state.setdefault("hub_maintenance_history", []).append({
            "maintenance_id": archive_existing["maintenance_id"],
            "maintenance_record": archive_existing,
            "archive_event": archive_event,
        })
        state.setdefault("events", []).append(archive_event)
    event = {
        "timestamp_utc": now.isoformat(), "monotonic_ns": mono,
        "event": "campaign_hub_maintenance_started", "maintenance_id": maintenance_id,
        "campaign_id": state.get("campaign_id"),
        "amendment_id": policy["amendment"]["amendment_id"],
        "authorization": "bounded_policy_reload",
        "reason": reason.strip(), "trainer_run_id": trainer_run_id,
        "checkpoint_path": _campaign_relative_path(checkpoint),
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "global_timestep": global_timestep,
        "effective_config_sha256": effective_config_sha256.lower(),
        "training_contract_sha256": training_contract_sha256.lower(),
        "old_hub_identity": dict(old_hub_identity),
        "prior_state_sha256": prior_hash,
        "original_clock_fields": original_clock,
    }
    state["hub_maintenance"] = {
        "maintenance_id": maintenance_id,
        "intent": event,
        "resume_authorization": None,
        "resume_claimed": False,
        "completed": None,
    }
    state.setdefault("events", []).append(event)
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    written = load_campaign_state(path)
    if written is None or campaign_hub_maintenance_status(written) != state["hub_maintenance"]:
        raise RuntimeError("maintenance intent failed its post-write validation")
    if _immutable_clock_fields(written) != original_clock:
        raise RuntimeError("maintenance intent changed immutable campaign clock fields")
    return written


def complete_campaign_hub_maintenance(
    state_path: str | Path,
    *,
    maintenance_id: str,
    resumed_run_id: str,
    new_hub_identity: dict[str, Any],
    resume_evidence: dict[str, Any],
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Close a maintenance intent after the same checkpoint is safely resumed."""
    import re

    if not re.fullmatch(r"[0-9a-f]{12}", resumed_run_id):
        raise ValueError("resumed_run_id must be a 12-character lowercase run ID")
    _validate_hub_identity(new_hub_identity, "new_hub_identity")
    if not isinstance(resume_evidence, dict):
        raise ValueError("resume_evidence must be an object")
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    completed_record = state.get("hub_maintenance") if isinstance(state, dict) else None
    if (isinstance(completed_record, dict)
            and completed_record.get("maintenance_id") == maintenance_id
            and isinstance(completed_record.get("completed"), dict)):
        expected = _maintenance_completion_payload(
            maintenance_id, resumed_run_id, new_hub_identity, resume_evidence,
            completed_record["intent"],
        )
        if _same_maintenance_completion(completed_record["completed"], expected):
            return state
        raise RuntimeError("maintenance ID was already completed with different evidence")
    active = campaign_hub_maintenance_status(state)
    if not active or active.get("maintenance_id") != maintenance_id:
        raise RuntimeError("matching active maintenance intent is required")
    intent = active["intent"]
    completion_existing = active.get("completed")
    if completion_existing is not None:
        expected = _maintenance_completion_payload(
            maintenance_id, resumed_run_id, new_hub_identity, resume_evidence, intent
        )
        if _same_maintenance_completion(completion_existing, expected):
            return state  # idempotent exact retry
        raise RuntimeError("maintenance ID was already completed with different evidence")
    authorization = active.get("resume_authorization")
    if not isinstance(authorization, dict) or active.get("resume_claimed") is not True:
        raise RuntimeError("a claimed one-time maintenance resume is required before completion")
    claimed = next((row for row in state.get("events", []) if isinstance(row, dict)
                    and row.get("event") == "campaign_hub_maintenance_resume_claimed"
                    and row.get("maintenance_id") == maintenance_id), None)
    if not isinstance(claimed, dict) or claimed.get("resumed_run_id") != resumed_run_id:
        raise ValueError("completion run ID does not match the one-time maintenance claim")
    if new_hub_identity != authorization.get("new_hub_identity"):
        raise ValueError("completion hub identity differs from the authorized new hub")
    required_true = (
        "operator_stop_completed", "neutral_command_published", "transport_fresh_after_resume",
        "fresh_optimizer_constructed", "global_timestep_inherited", "rollout_buffer_reset",
    )
    if any(resume_evidence.get(key) is not True for key in required_true):
        raise ValueError("maintenance resume evidence must confirm stop, neutral, fresh ROS, fresh optimizer, inherited steps, and rollout-buffer reset")
    if resume_evidence.get("transport_acknowledged") is not False:
        raise ValueError("transport_acknowledged must be false; the simulator exposes no physical command ACK")
    if resume_evidence.get("loaded_checkpoint_sha256") != intent["checkpoint_sha256"]:
        raise ValueError("resumed trainer did not load the exact maintenance checkpoint")
    if resume_evidence.get("loaded_global_timestep") != intent["global_timestep"]:
        raise ValueError("resumed trainer did not inherit the exact checkpoint global timestep")
    if (not _valid_sha256(resume_evidence.get("resume_effective_config_sha256"))
            or resume_evidence.get("resume_training_contract_sha256") != intent["training_contract_sha256"]):
        raise ValueError("resumed trainer training contract differs from maintenance checkpoint")
    from src.layer3.bc_artifact import verify_ppo_lineage_checkpoint
    first_checkpoint = resume_evidence.get("first_postupdate_checkpoint_path")
    first_checkpoint_sha = resume_evidence.get("first_postupdate_checkpoint_sha256")
    first_step = resume_evidence.get("first_postupdate_global_timestep")
    if (not isinstance(first_checkpoint, str) or not _valid_sha256(first_checkpoint_sha)
            or isinstance(first_step, bool) or not isinstance(first_step, int)
            or first_step <= intent["global_timestep"]):
        raise ValueError("completion requires a later post-update checkpoint from the resumed trainer")
    verified_first = verify_ppo_lineage_checkpoint(first_checkpoint)
    if (not verified_first or verified_first.get("checkpoint_sha256") != first_checkpoint_sha.lower()
            or verified_first.get("num_timesteps") != first_step
            or verified_first.get("effective_config_sha256") != resume_evidence.get("resume_effective_config_sha256")
            or verified_first.get("parent_checkpoint_sha256") != intent["checkpoint_sha256"]
            or not isinstance(verified_first.get("lineage"), dict)
            or verified_first["lineage"].get("post_optimizer_update") is not True):
        raise ValueError("first resumed checkpoint failed lineage, parent, or post-update authentication")
    if resume_evidence.get("first_postupdate_audit_finite") is not True:
        raise ValueError("maintenance completion requires a finite first post-update audit")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    original_clock = _immutable_clock_fields(state)
    completion = _maintenance_completion_payload(
        maintenance_id, resumed_run_id, new_hub_identity, resume_evidence, intent,
        timestamp_utc=now.isoformat(), monotonic_ns=mono,
    )
    state.setdefault("events", []).append(completion)
    state["hub_maintenance"]["completed"] = completion
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    written = load_campaign_state(path)
    if written is None or _immutable_clock_fields(written) != original_clock:
        raise RuntimeError("maintenance completion failed validation or altered original clock fields")
    return written


def _immutable_clock_fields(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "created_at_utc", "started_at_utc", "training_freeze_at_utc", "hard_deadline_at_utc",
        "training_window_seconds", "validation_window_seconds", "hard_window_seconds",
        "start_monotonic_ns", "training_freeze_monotonic_ns", "hard_deadline_monotonic_ns", "boot_id",
    )
    return {key: state.get(key) for key in keys}


def _campaign_relative_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("maintenance checkpoint must be inside the repository") from exc


def _validate_hub_identity(
    identity: dict[str, Any], name: str, *, allow_legacy_source_unavailable: bool = False,
) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{name} must be an object")
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"{name}.pid must be a positive integer")
    if not isinstance(identity.get("process_start_identity"), str) or not identity["process_start_identity"].strip():
        raise ValueError(f"{name}.process_start_identity is required")
    if not isinstance(identity.get("boot_id"), str) or not identity["boot_id"].strip():
        raise ValueError(f"{name}.boot_id is required")
    if not isinstance(identity.get("executable"), str) or not Path(identity["executable"]).is_absolute():
        raise ValueError(f"{name}.executable must be an absolute path")
    argv = identity.get("argv")
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(argument, str) for argument in argv)):
        raise ValueError(f"{name}.argv must be a non-empty string array")
    source_sha = identity.get("source_sha256")
    if _valid_sha256(source_sha):
        return
    if (allow_legacy_source_unavailable
            and source_sha is None
            and identity.get("source_attestation") == "unavailable_legacy_process"):
        return
    raise ValueError(f"{name}.source_sha256 must identify the loaded hub source")


def _maintenance_intent_matches(
    intent: dict[str, Any], trainer_run_id: str, checkpoint_path: str | Path,
    checkpoint_sha256: str, global_timestep: int, effective_config_sha256: str,
    training_contract_sha256: str,
    old_hub_identity: dict[str, Any], reason: str,
) -> bool:
    return all((
        intent.get("trainer_run_id") == trainer_run_id,
        intent.get("checkpoint_path") == _campaign_relative_path(Path(checkpoint_path).resolve()),
        intent.get("checkpoint_sha256") == checkpoint_sha256.lower(),
        intent.get("global_timestep") == global_timestep,
        intent.get("effective_config_sha256") == effective_config_sha256.lower(),
        intent.get("training_contract_sha256") == training_contract_sha256.lower(),
        intent.get("old_hub_identity") == old_hub_identity,
        intent.get("reason") == reason.strip(),
    ))


def _same_maintenance_completion(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _resume_config_matches_intent(config: dict[str, Any], intent: dict[str, Any]) -> bool:
    resume = config.get("resume")
    if not isinstance(resume, str):
        return False
    normalized = resume.replace("\\", "/")
    relative = intent.get("checkpoint_path")
    if normalized.startswith("/runs/"):
        normalized = "logs/" + normalized[len("/runs/"):]
    elif normalized.startswith("/app/logs/"):
        normalized = "logs/" + normalized[len("/app/logs/"):]
    else:
        normalized = normalized.lstrip("/")
    return normalized == relative


def _resolve_campaign_resume_path(resume_value: str) -> Path:
    normalized = resume_value.replace("\\", "/")
    root = Path(__file__).resolve().parents[3]
    if normalized.startswith("/runs/"):
        return (root / "logs" / normalized[len("/runs/"):]).resolve()
    if normalized.startswith("/app/logs/"):
        return (root / "logs" / normalized[len("/app/logs/"):]).resolve()
    candidate = Path(resume_value)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def campaign_training_contract_sha256(config: dict[str, Any]) -> str:
    """Hash effective train settings while excluding run-specific resume/display metadata."""
    if not isinstance(config, dict):
        raise ValueError("campaign training config must be an object")
    excluded = EXPERIMENT_METADATA_FIELDS | {"resume", "display_name"}
    projection = {key: value for key, value in config.items() if key not in excluded}
    return hashlib.sha256(json.dumps(
        projection, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _maintenance_completion_payload(
    maintenance_id: str, resumed_run_id: str, new_hub_identity: dict[str, Any],
    resume_evidence: dict[str, Any], intent: dict[str, Any], *,
    timestamp_utc: Optional[str] = None, monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    row = {
        "event": "campaign_hub_maintenance_completed",
        "maintenance_id": maintenance_id,
        "campaign_id": intent["campaign_id"],
        "amendment_id": intent["amendment_id"],
        "trainer_run_id": intent["trainer_run_id"],
        "resumed_run_id": resumed_run_id,
        "checkpoint_path": intent["checkpoint_path"],
        "checkpoint_sha256": intent["checkpoint_sha256"],
        "global_timestep": intent["global_timestep"],
        "source_effective_config_sha256": intent["effective_config_sha256"],
        "training_contract_sha256": intent["training_contract_sha256"],
        "new_hub_identity": dict(new_hub_identity),
        "resume_evidence": dict(resume_evidence),
        "original_clock_fields": intent["original_clock_fields"],
    }
    if timestamp_utc is not None:
        row["timestamp_utc"] = timestamp_utc
        row["monotonic_ns"] = monotonic_ns
    return row


def authorize_campaign_hub_maintenance_resume(
    state_path: str | Path, maintenance_id: str, *, new_hub_identity: dict[str, Any],
    resume_config: dict[str, Any], resume_config_sha256: str,
    training_contract_sha256: str, utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Bind the sole allowed resume request to this maintenance handoff."""
    import re
    from src.layer3.bc_artifact import sha256_file, verify_ppo_lineage_checkpoint

    _validate_hub_identity(new_hub_identity, "new_hub_identity")
    if not re.fullmatch(r"[0-9a-f]{32}", maintenance_id):
        raise ValueError("maintenance_id must be 32 lowercase hexadecimal characters")
    if not isinstance(resume_config, dict) or not _valid_sha256(resume_config_sha256):
        raise ValueError("resume config and its SHA-256 are required")
    if campaign_effective_config_sha256(resume_config) != resume_config_sha256.lower():
        raise ValueError("resume config does not match its canonical effective-config hash")
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    active = campaign_hub_maintenance_status(state)
    if not active or active.get("maintenance_id") != maintenance_id or active.get("completed") is not None:
        raise RuntimeError("matching incomplete maintenance intent is required")
    intent = active["intent"]
    if (training_contract_sha256.lower() != intent["training_contract_sha256"]
            or campaign_training_contract_sha256(resume_config) != intent["training_contract_sha256"]):
        raise ValueError("resume training contract differs from the maintenance checkpoint contract")
    if resume_config.get("campaign_experiment_id") not in {None, ""}:
        raise ValueError("maintenance resume cannot introduce a campaign experiment")
    resume_value = resume_config.get("resume")
    if not isinstance(resume_value, str) or not resume_value:
        raise ValueError("maintenance resume config must name its resume checkpoint")
    resume_path = _resolve_campaign_resume_path(resume_value)
    if not resume_path.is_file() or sha256_file(resume_path) != intent["checkpoint_sha256"]:
        raise ValueError("maintenance resume config must load the exact declared checkpoint bytes")
    parent_lineage = verify_ppo_lineage_checkpoint(resume_path)
    if (not parent_lineage or parent_lineage.get("num_timesteps") != intent["global_timestep"]
            or parent_lineage.get("effective_config_sha256") != intent["effective_config_sha256"]):
        raise ValueError("maintenance resume checkpoint step/config failed lineage verification")
    prior = active.get("resume_authorization")
    if prior is not None:
        if (prior.get("resume_config_sha256") == resume_config_sha256.lower()
                and prior.get("new_hub_identity") == new_hub_identity):
            return state
        raise RuntimeError("maintenance resume is already authorized with different inputs")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    event = {
        "timestamp_utc": now.isoformat(), "monotonic_ns": mono,
        "event": "campaign_hub_maintenance_resume_authorized",
        "maintenance_id": maintenance_id, "campaign_id": state.get("campaign_id"),
        "amendment_id": intent["amendment_id"],
        "new_hub_identity": dict(new_hub_identity),
        "resume_config_sha256": resume_config_sha256.lower(),
        "training_contract_sha256": training_contract_sha256.lower(),
        "checkpoint_path": intent["checkpoint_path"],
        "checkpoint_sha256": intent["checkpoint_sha256"],
        "global_timestep": intent["global_timestep"],
        "consumed": False,
    }
    state["hub_maintenance"]["resume_authorization"] = event
    state["hub_maintenance"]["resume_claimed"] = False
    state.setdefault("events", []).append(event)
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    return load_campaign_state(path) or state


def supersede_campaign_hub_maintenance_resume_authorization(
    state_path: str | Path, maintenance_id: str, *,
    prior_authorization_sha256: str, new_hub_identity: dict[str, Any],
    utc_now: Optional[datetime] = None, monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Rotate an unclaimed authorization to a newly verified hub identity.

    The prior authorization remains in the event history. The supersession
    event binds its canonical SHA and hub identity to a replacement that must
    preserve the original resume config, training contract, checkpoint, and
    global timestep exactly. This helper does not start or stop any process.
    """
    import re
    from src.layer3.bc_artifact import sha256_file, verify_ppo_lineage_checkpoint

    if not re.fullmatch(r"[0-9a-f]{32}", maintenance_id):
        raise ValueError("maintenance_id must be 32 lowercase hexadecimal characters")
    if not _valid_sha256(prior_authorization_sha256):
        raise ValueError("prior authorization SHA-256 is required")
    _validate_hub_identity(new_hub_identity, "new_hub_identity")
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    active = campaign_hub_maintenance_status(state)
    if not active or active.get("maintenance_id") != maintenance_id or active.get("completed") is not None:
        raise RuntimeError("matching incomplete maintenance intent is required")
    if active.get("resume_claimed") is True:
        raise RuntimeError("a claimed maintenance resume cannot be superseded")
    prior = active.get("resume_authorization")
    if not isinstance(prior, dict) or prior.get("consumed") is not False:
        raise RuntimeError("an unconsumed resume authorization is required")

    prior_hash = _maintenance_authorization_sha256(prior)
    supersessions = [row for row in state.get("events", []) if isinstance(row, dict)
                     and row.get("event") == "campaign_hub_maintenance_resume_authorization_superseded"
                     and row.get("maintenance_id") == maintenance_id]
    if (prior_hash != prior_authorization_sha256.lower()
            and supersessions
            and supersessions[-1].get("prior_authorization_sha256") == prior_authorization_sha256.lower()
            and supersessions[-1].get("replacement_authorization") == prior
            and prior.get("new_hub_identity") == new_hub_identity):
        return state  # exact retry after a successful write
    if prior_hash != prior_authorization_sha256.lower():
        raise RuntimeError("prior authorization SHA does not match the current unconsumed authorization")
    if prior.get("new_hub_identity") == new_hub_identity:
        return state

    intent = active["intent"]
    checkpoint = _resolve_campaign_resume_path(str(intent["checkpoint_path"]))
    if not checkpoint.is_file() or sha256_file(checkpoint) != intent["checkpoint_sha256"]:
        raise ValueError("maintenance checkpoint no longer matches the authorized bytes")
    parent_lineage = verify_ppo_lineage_checkpoint(checkpoint)
    if (not parent_lineage
            or parent_lineage.get("num_timesteps") != intent["global_timestep"]
            or parent_lineage.get("effective_config_sha256") != intent["effective_config_sha256"]):
        raise ValueError("maintenance checkpoint step/config failed lineage verification")

    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    original_clock = _immutable_clock_fields(state)
    replacement = dict(prior)
    replacement.update({
        "timestamp_utc": now.isoformat(),
        "monotonic_ns": mono,
        "new_hub_identity": dict(new_hub_identity),
        "consumed": False,
    })
    event = {
        "timestamp_utc": now.isoformat(),
        "monotonic_ns": mono,
        "event": "campaign_hub_maintenance_resume_authorization_superseded",
        "maintenance_id": maintenance_id,
        "campaign_id": intent["campaign_id"],
        "amendment_id": intent["amendment_id"],
        "prior_authorization_sha256": prior_hash,
        "prior_new_hub_identity": dict(prior["new_hub_identity"]),
        "replacement_authorization": replacement,
    }
    state["hub_maintenance"]["resume_authorization"] = replacement
    state["hub_maintenance"]["resume_claimed"] = False
    state.setdefault("events", []).append(event)
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    written = load_campaign_state(path)
    if (written is None
            or campaign_hub_maintenance_status(written) != written["hub_maintenance"]):
        raise RuntimeError("maintenance authorization supersession failed post-write validation")
    if _immutable_clock_fields(written) != original_clock:
        raise RuntimeError("maintenance authorization supersession changed immutable clock fields")
    return written


def _maintenance_authorization_sha256(authorization: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        authorization, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _maintenance_authorization_binding(authorization: dict[str, Any]) -> dict[str, Any]:
    return {key: authorization.get(key) for key in (
        "maintenance_id", "campaign_id", "amendment_id", "resume_config_sha256",
        "training_contract_sha256", "checkpoint_path", "checkpoint_sha256", "global_timestep",
    )}


def _validate_maintenance_resume_authorization(
    authorization: dict[str, Any], intent: dict[str, Any],
) -> None:
    if (authorization.get("event") != "campaign_hub_maintenance_resume_authorized"
            or authorization.get("maintenance_id") != intent.get("maintenance_id")
            or not _valid_sha256(authorization.get("resume_config_sha256"))
            or authorization.get("training_contract_sha256") != intent.get("training_contract_sha256")
            or authorization.get("checkpoint_path") != intent.get("checkpoint_path")
            or authorization.get("checkpoint_sha256") != intent.get("checkpoint_sha256")
            or authorization.get("global_timestep") != intent.get("global_timestep")
            or authorization.get("consumed") is not False):
        raise ValueError("campaign maintenance resume authorization is invalid")
    _validate_hub_identity(authorization.get("new_hub_identity"), "new_hub_identity")


def claim_campaign_hub_maintenance_resume(
    state_path: str | Path, *, kind: Literal["training", "evaluation"],
    normalized_config: dict[str, Any], new_hub_identity: dict[str, Any],
    resumed_run_id: str,
    utc_now: Optional[datetime] = None, monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Atomically consume the one maintenance resume; reject all other work."""
    if kind != "training":
        raise RuntimeError("only the single declared training resume is allowed during hub maintenance")
    import re
    if not isinstance(resumed_run_id, str) or not re.fullmatch(r"[0-9a-f]{12}", resumed_run_id):
        raise ValueError("resumed_run_id must be a 12-character lowercase run ID")
    _validate_hub_identity(new_hub_identity, "new_hub_identity")
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    active = campaign_hub_maintenance_status(state)
    if not active or active.get("completed") is not None:
        raise RuntimeError("no active hub maintenance resume authorization")
    auth = active.get("resume_authorization")
    if (not isinstance(auth, dict) or active.get("resume_claimed") is True
            or auth.get("consumed") is not False):
        raise RuntimeError("maintenance resume is absent or already consumed")
    if (new_hub_identity != auth.get("new_hub_identity")
            or campaign_effective_config_sha256(normalized_config) != auth.get("resume_config_sha256")
            or normalized_config.get("resume") is None):
        raise ValueError("training request does not match the one authorized maintenance resume")
    resume_value = str(normalized_config["resume"])
    if not _resume_config_matches_intent(normalized_config, active["intent"]):
        raise ValueError("maintenance resume request names a different checkpoint")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    state["hub_maintenance"]["resume_claimed"] = True
    state.setdefault("events", []).append({
        "event": "campaign_hub_maintenance_resume_claimed",
        "maintenance_id": auth["maintenance_id"],
        "resumed_run_id": resumed_run_id,
        "resume_config_sha256": auth["resume_config_sha256"],
        "checkpoint_sha256": auth["checkpoint_sha256"],
        "new_hub_identity": dict(auth["new_hub_identity"]),
        "timestamp_utc": now.isoformat(), "monotonic_ns": mono,
    })
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    return load_campaign_state(path) or state


def request_campaign_final_validation(
    state_path: str | Path,
    *,
    candidate_checkpoint_sha256: str,
    evaluation_id: str,
    probe_report_sha256: str,
    readiness_evidence: dict[str, Any],
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Append a validation request only after measured stability is proven.

    The request does not choose finalists itself. It is only written after an
    explicit Sol readiness recommendation; a pair of probes alone does not
    automatically freeze training. It binds that decision to one candidate
    and opens the existing manifest/competitive workflow.
    """
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None or state.get("status") != "active":
        raise RuntimeError("validation request requires an active campaign")
    policy = effective_campaign_deadline_policy(state)
    if policy["mode"] != "unbounded_until_readiness":
        raise RuntimeError("explicit validation requests are only used after the unbounded user amendment")
    if campaign_validation_request(state) is not None:
        raise RuntimeError("an unclosed campaign validation request already exists")
    if (not _valid_sha256(candidate_checkpoint_sha256)
            or not _valid_sha256(probe_report_sha256)
            or not isinstance(evaluation_id, str) or not evaluation_id):
        raise ValueError("validation request must bind checkpoint/report SHA-256 and evaluation ID")
    if not isinstance(readiness_evidence, dict):
        raise ValueError("validation request readiness evidence must be an object")
    # Reuse the same strict validator as the read path before appending.
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    amendment_id = policy["amendment"]["amendment_id"]
    request_id = uuid.uuid4().hex
    event = {
        "timestamp_utc": now.isoformat(),
        "monotonic_ns": mono,
        "event": "campaign_validation_requested",
        "request_id": request_id,
        "authorization": "readiness_gate",
        "policy": "unbounded_until_readiness",
        "amendment_id": amendment_id,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256.lower(),
        "evaluation_id": evaluation_id,
        "probe_report_sha256": probe_report_sha256.lower(),
        "readiness_evidence": readiness_evidence,
    }
    preview = dict(state)
    preview["events"] = [*state.get("events", []), event]
    if campaign_validation_request(preview) != event:
        raise ValueError("validation request readiness evidence did not pass the explicit stability gate")
    state.setdefault("events", []).append(event)
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    written = load_campaign_state(path) or state
    request = campaign_validation_request(written)
    if request != event:
        raise RuntimeError("new validation request failed its own readiness/authentication check")
    return event


def amend_campaign_deadline_policy(
    state_path: str | Path,
    *,
    authorization: str,
    reason: str,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Append one explicit user authorization for unbounded training.

    This is a state-file mutation helper, not a runtime action. It preserves
    every existing clock field and prior event, and refuses repeated edits.
    """
    if authorization != "user_override":
        raise ValueError("deadline amendment requires authorization='user_override'")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("deadline amendment requires the user's reason")
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None or state.get("status") != "active":
        raise RuntimeError("deadline amendment requires an active campaign")
    if state.get("deadline_policy_amendment") is not None:
        raise ValueError("campaign deadline policy has already been amended")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    old_bytes = path.read_bytes()
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    amendment_id = uuid.uuid4().hex
    event = {
        "timestamp_utc": now.isoformat(),
        "monotonic_ns": mono,
        "event": "campaign_deadline_policy_amended",
        "amendment_id": amendment_id,
        "authorization": "user_override",
        "policy": "unbounded_until_readiness",
        "reason": reason.strip(),
        "prior_state_sha256": old_hash,
        "original_training_freeze_at_utc": state["training_freeze_at_utc"],
        "original_hard_deadline_at_utc": state["hard_deadline_at_utc"],
    }
    amendment = {
        "amendment_id": amendment_id,
        "authorization": "user_override",
        "policy": "unbounded_until_readiness",
        "reason": reason.strip(),
        "authorized_at_utc": now.isoformat(),
        "prior_revision": state.get("revision"),
        "prior_state_sha256": old_hash,
        "original_training_freeze_at_utc": state["training_freeze_at_utc"],
        "original_hard_deadline_at_utc": state["hard_deadline_at_utc"],
        "effective_training_freeze_at_utc": None,
        "effective_hard_deadline_at_utc": None,
        "validation_policy": "explicit_readiness_backed_request",
        "event": event,
    }
    state["deadline_policy_amendment"] = amendment
    state.setdefault("events", []).append(event)
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = mono
    _atomic_write_json(path, state)
    # Validate the written object before returning it to any caller.
    return load_campaign_state(path) or state


def _passed_gate(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if str(record.get("status", "")).strip().casefold() not in {"pass", "passed"}:
        return False
    return record.get("passed") is True or record.get("useful") is True or record.get("readiness_passed") is True


def validate_gate_evidence(evidence: dict[str, Any]) -> tuple[str, str]:
    """Require useful updated-policy probe and single-env queue test evidence."""
    probe = evidence.get("updated_policy_probe")
    if not _passed_gate(probe):
        raise ValueError("campaign clock cannot start: updated_policy_probe must explicitly pass and be useful")
    checkpoint_hash = probe.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError("campaign gate probe must name its checkpoint SHA-256")
    try:
        int(checkpoint_hash, 16)
    except ValueError as exc:
        raise ValueError("campaign gate probe checkpoint hash must be hexadecimal") from exc

    queue = evidence.get("single_environment_queue_interruption")
    if not _passed_gate(queue):
        raise ValueError("campaign clock cannot start: single_environment_queue_interruption must explicitly pass")
    return checkpoint_hash.lower(), hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def initialize_campaign(
    state_path: str | Path,
    gate_path: str | Path,
    *,
    campaign_id: Optional[str] = None,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Start one immutable 12-hour window after both explicit readiness gates pass."""
    path = Path(state_path).resolve()
    if path.exists():
        raise FileExistsError(
            f"campaign state already exists at {path}; refusing to reset or extend a campaign clock"
        )
    gate_file = Path(gate_path).resolve()
    evidence = _read_json(gate_file)
    probe_hash, evidence_hash = validate_gate_evidence(evidence)
    start = (utc_now or _utc_now()).astimezone(timezone.utc)
    monotonic_start = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    boot = boot_id or current_boot_id()
    training_seconds = TRAINING_HOURS * 3600
    total_seconds = CAMPAIGN_HOURS * 3600
    state = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id or f"official-ppo-{uuid.uuid4().hex[:12]}",
        "status": "active",
        "phase": "training_window",
        "created_at_utc": start.isoformat(),
        "started_at_utc": start.isoformat(),
        "training_freeze_at_utc": (start + timedelta(seconds=training_seconds)).isoformat(),
        "hard_deadline_at_utc": (start + timedelta(seconds=total_seconds)).isoformat(),
        "training_window_seconds": training_seconds,
        "validation_window_seconds": total_seconds - training_seconds,
        "hard_window_seconds": total_seconds,
        "start_monotonic_ns": monotonic_start,
        "training_freeze_monotonic_ns": monotonic_start + training_seconds * 1_000_000_000,
        "hard_deadline_monotonic_ns": monotonic_start + total_seconds * 1_000_000_000,
        "boot_id": boot,
        "last_observed_at_utc": start.isoformat(),
        "last_observed_monotonic_ns": monotonic_start,
        "revision": 1,
        "gate_evidence_path": str(gate_file),
        "gate_evidence_sha256": hashlib.sha256(gate_file.read_bytes()).hexdigest(),
        "updated_policy_probe_checkpoint_sha256": probe_hash,
        "gate_evidence_canonical_sha256": evidence_hash,
        "policy": {
            "training_allowed_until": "training_freeze_at_utc",
            "evaluation_allowed_until": "hard_deadline_at_utc",
            "reject_training_at_or_after_freeze": True,
            "reject_evaluation_at_or_after_hard_deadline": True,
            "fail_closed_on_boot_id_change": True,
        },
        "campaign_plan": CAMPAIGN_PLAN,
        "monitor": {
            "poll_interval_seconds": 1,
            "graceful_stop_timeout_seconds": 5,
            "hard_deadline_cleanup_lead_seconds": 5,
        },
        "events": [],
    }
    # Reserve by exclusive creation before replacement so two CLI invocations
    # cannot race and silently move the immutable campaign start time.
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    _atomic_write_json(path, state)
    return state


def load_campaign_state(path: Optional[str | Path] = None) -> Optional[dict[str, Any]]:
    state_path = Path(path).resolve() if path is not None else default_state_path()
    if not state_path.exists():
        return None
    state = _read_json(state_path)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported campaign state schema")
    if state.get("status") not in {"active", "completed", "expired", "failed_closed"}:
        raise ValueError("campaign state has an invalid status")
    for field in ("started_at_utc", "training_freeze_at_utc", "hard_deadline_at_utc"):
        _parse_utc(state.get(field), field)
    for field in ("start_monotonic_ns", "training_freeze_monotonic_ns", "hard_deadline_monotonic_ns"):
        if isinstance(state.get(field), bool) or not isinstance(state.get(field), int):
            raise ValueError(f"campaign {field} must be an integer")
    if not isinstance(state.get("boot_id"), str) or not state["boot_id"]:
        raise ValueError("campaign boot_id is required")
    start = _parse_utc(state.get("started_at_utc"), "started_at_utc")
    freeze = _parse_utc(state.get("training_freeze_at_utc"), "training_freeze_at_utc")
    hard = _parse_utc(state.get("hard_deadline_at_utc"), "hard_deadline_at_utc")
    if state.get("training_window_seconds") != TRAINING_HOURS * 3600:
        raise ValueError("campaign training duration does not match the fixed nine-hour window")
    if state.get("hard_window_seconds") != CAMPAIGN_HOURS * 3600:
        raise ValueError("campaign hard duration does not match the fixed twelve-hour window")
    if state.get("validation_window_seconds") != (CAMPAIGN_HOURS - TRAINING_HOURS) * 3600:
        raise ValueError("campaign validation duration does not match the fixed three-hour window")
    if freeze - start != timedelta(hours=TRAINING_HOURS) or hard - start != timedelta(hours=CAMPAIGN_HOURS):
        raise ValueError("campaign UTC deadlines do not match the fixed 9h/12h windows")
    if (state["training_freeze_monotonic_ns"] - state["start_monotonic_ns"]
            != TRAINING_HOURS * 3600 * 1_000_000_000):
        raise ValueError("campaign monotonic training deadline is invalid")
    if (state["hard_deadline_monotonic_ns"] - state["start_monotonic_ns"]
            != CAMPAIGN_HOURS * 3600 * 1_000_000_000):
        raise ValueError("campaign monotonic hard deadline is invalid")
    if state.get("campaign_plan") != CAMPAIGN_PLAN:
        raise ValueError("campaign plan is missing or differs from the fixed campaign contract")
    effective_campaign_deadline_policy(state)
    return state


def campaign_clock_status(
    state: dict[str, Any],
    *,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Calculate conservative elapsed/remaining time from UTC and monotonic clocks."""
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    mono = int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns)
    boot = boot_id or current_boot_id()
    if state.get("status") != "active":
        return {"phase": state.get("status"), "remaining_seconds": 0.0, "reason": "campaign_not_active"}
    policy = effective_campaign_deadline_policy(state)
    if policy["mode"] == "unbounded_until_readiness":
        start_utc = _parse_utc(state["started_at_utc"], "started_at_utc")
        wall_elapsed = (now - start_utc).total_seconds()
        same_boot = boot == state["boot_id"]
        monotonic_elapsed = (
            (mono - int(state["start_monotonic_ns"])) / 1_000_000_000
            if same_boot else 0.0
        )
        request = campaign_validation_request(state)
        return {
            "phase": "final_validation_requested" if request else "training_unbounded",
            "elapsed_seconds": max(0.0, wall_elapsed, monotonic_elapsed),
            "seconds_until_training_freeze": None,
            "remaining_seconds": None,
            "reason": "explicit_user_override_until_readiness",
            "deadline_policy": policy,
            "validation_request": request,
        }
    if boot != state["boot_id"]:
        return {"phase": "expired", "remaining_seconds": 0.0, "reason": "boot_id_changed"}

    start_utc = _parse_utc(state["started_at_utc"], "started_at_utc")
    freeze_utc = _parse_utc(state["training_freeze_at_utc"], "training_freeze_at_utc")
    hard_utc = _parse_utc(state["hard_deadline_at_utc"], "hard_deadline_at_utc")
    wall_elapsed = (now - start_utc).total_seconds()
    monotonic_elapsed = (mono - int(state["start_monotonic_ns"])) / 1_000_000_000
    elapsed = max(wall_elapsed, monotonic_elapsed)
    to_freeze = min(
        (freeze_utc - now).total_seconds(),
        (int(state["training_freeze_monotonic_ns"]) - mono) / 1_000_000_000,
    )
    remaining = min(
        (hard_utc - now).total_seconds(),
        (int(state["hard_deadline_monotonic_ns"]) - mono) / 1_000_000_000,
    )
    if remaining <= 0:
        phase = "expired"
    elif elapsed >= int(state["training_window_seconds"]):
        phase = "final_validation"
    else:
        phase = "training_window"
    return {
        "phase": phase,
        "elapsed_seconds": max(0.0, elapsed),
        "seconds_until_training_freeze": max(0.0, to_freeze),
        "remaining_seconds": max(0.0, remaining),
        "reason": None,
    }


def campaign_admission_decision(
    state: Optional[dict[str, Any]],
    kind: Literal["training", "evaluation"],
    *,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
    maintenance_resume: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    if kind not in {"training", "evaluation"}:
        raise ValueError("campaign admission kind must be training or evaluation")
    if state is None:
        return True, "no active campaign deadline"
    if state.get("status") != "active":
        return False, f"campaign is {state.get('status')}; new {kind} work is disabled"
    maintenance = campaign_hub_maintenance_status(state)
    if maintenance is not None:
        auth = maintenance.get("resume_authorization")
        context = maintenance_resume if isinstance(maintenance_resume, dict) else {}
        config = context.get("normalized_config")
        identity = context.get("new_hub_identity")
        if (kind == "training" and maintenance.get("resume_claimed") is not True
                and isinstance(auth, dict) and auth.get("consumed") is False
                and isinstance(config, dict) and identity == auth.get("new_hub_identity")
                and campaign_effective_config_sha256(config) == auth.get("resume_config_sha256")
                and campaign_training_contract_sha256(config) == auth.get("training_contract_sha256")
                and _resume_config_matches_intent(config, maintenance["intent"])):
            return True, "campaign allows the exact one-time maintenance resume"
        return False, "campaign hub maintenance is active; only the exact authorized resume may start"
    clock = campaign_clock_status(
        state, utc_now=utc_now, monotonic_ns=monotonic_ns, boot_id=boot_id
    )
    if clock["phase"] == "expired":
        return False, f"campaign hard deadline reached or clock invalid ({clock.get('reason') or 'deadline'})"
    if kind == "training" and clock["phase"] == "final_validation_requested":
        return False, "readiness-backed final validation was requested; training is frozen for validation"
    if kind == "training" and clock["phase"] not in {"training_window", "training_unbounded"}:
        return False, "campaign training window ended; only final validation is allowed"
    return True, f"campaign {clock['phase']} allows {kind}"


def campaign_deadline_status(
    *, state_path: Optional[str | Path] = None,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
    boot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Read-only status payload suitable for a health/status API response."""
    path = Path(state_path).resolve() if state_path is not None else default_state_path()
    if not path.exists():
        return {"configured": False, "status": "not_started", "state_path": str(path)}
    try:
        state = load_campaign_state(path)
        if state is None:
            return {"configured": False, "status": "not_started", "state_path": str(path)}
        clock = campaign_clock_status(
            state, utc_now=utc_now, monotonic_ns=monotonic_ns, boot_id=boot_id
        )
        return {
            "configured": True,
            "campaign_id": state.get("campaign_id"),
            "status": state.get("status"),
            "phase": "hub_maintenance" if campaign_hub_maintenance_status(state) is not None else clock["phase"],
            "started_at_utc": state.get("started_at_utc"),
            "training_freeze_at_utc": clock.get("deadline_policy", {}).get(
                "training_freeze_at_utc", state.get("training_freeze_at_utc")
            ),
            "hard_deadline_at_utc": clock.get("deadline_policy", {}).get(
                "hard_deadline_at_utc", state.get("hard_deadline_at_utc")
            ),
            "remaining_seconds": clock.get("remaining_seconds", 0.0),
            "seconds_until_training_freeze": clock.get("seconds_until_training_freeze", 0.0),
            "deadline_policy": clock.get("deadline_policy", effective_campaign_deadline_policy(state)),
            "validation_request": clock.get("validation_request"),
            "hub_maintenance": campaign_hub_maintenance_status(state),
            "campaign_plan": CAMPAIGN_PLAN,
            "reason": clock.get("reason"),
            "state_path": str(path),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "configured": True,
            "status": "invalid",
            "phase": "fail_closed",
            "remaining_seconds": 0.0,
            "reason": str(exc),
            "state_path": str(path),
        }


def assert_campaign_allows(
    kind: Literal["training", "evaluation"],
    *,
    state_path: Optional[str | Path] = None,
    maintenance_resume: Optional[dict[str, Any]] = None,
) -> None:
    """Raise a conflict-friendly error when campaign policy denies new work."""
    path = Path(state_path).resolve() if state_path is not None else default_state_path()
    if not path.exists():
        return
    try:
        state = load_campaign_state(path)
        allowed, reason = campaign_admission_decision(state, kind, maintenance_resume=maintenance_resume)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"campaign deadline state is invalid; refusing {kind}: {exc}") from exc
    if not allowed:
        raise RuntimeError(reason)


def assert_campaign_training_config(
    config: dict[str, Any],
    *,
    state_path: Optional[str | Path] = None,
) -> None:
    """Validate campaign training requests against the immutable run plan.

    Outside a campaign this is a no-op, preserving ordinary PPO defaults.
    Call after ``assert_campaign_allows('training')`` and before persisting run
    settings so an active campaign cannot silently inherit exploratory defaults.
    """
    path = Path(state_path).resolve() if state_path is not None else default_state_path()
    if not path.exists():
        return
    state = load_campaign_state(path)
    if state is None or state.get("status") != "active":
        raise RuntimeError("campaign training configuration rejected because campaign is not active")

    metadata = {key: config.get(key) for key in EXPERIMENT_METADATA_FIELDS}
    present_metadata = {key for key, value in metadata.items() if value is not None}
    experiment_id: Optional[str] = None
    experiment_phase: Optional[str] = None
    parent_effective: Optional[dict[str, Any]] = None
    if present_metadata:
        if present_metadata != EXPERIMENT_METADATA_FIELDS:
            raise ValueError("campaign experiment metadata must be supplied as a complete id/phase/config-hash tuple")
        experiment_id = metadata["campaign_experiment_id"]
        experiment_phase = metadata["campaign_experiment_phase"]
        if experiment_id in INVALIDATED_TIME_COST_VARIANTS:
            raise ValueError(
                f"campaign experiment {experiment_id} is invalidated: "
                f"{INVALIDATED_TIME_COST_VARIANTS[experiment_id]}"
            )
        config_hash = metadata["campaign_experiment_config_sha256"]
        if not all(isinstance(value, str) and value.strip() for value in (experiment_id, experiment_phase)):
            raise ValueError("campaign experiment id and phase must be non-empty strings")
        if not isinstance(config_hash, str) or len(config_hash) != 64:
            raise ValueError("campaign experiment config hash must be a SHA-256 hex digest")
        try:
            int(config_hash, 16)
        except ValueError as exc:
            raise ValueError("campaign experiment config hash must be hexadecimal") from exc
        config_hash = config_hash.lower()
        if campaign_effective_config_sha256(config) != config_hash:
            raise ValueError("campaign experiment effective-config hash does not match the submitted configuration")
        declaration = next((row for row in reversed(state.get("events", []))
                            if isinstance(row, dict)
                            and row.get("event") == "campaign_experiment_declared"
                            and row.get("experiment_id") == experiment_id), None)
        if declaration is None or declaration.get("phase") != experiment_phase or declaration.get("config_sha256") != config_hash:
            raise ValueError("campaign experiment is not present in the append-only declaration ledger")
        if declaration.get("campaign_id") != state.get("campaign_id"):
            raise ValueError("campaign experiment declaration belongs to another campaign")
        evidence = declaration.get("evidence")
        if experiment_id in TIME_COST_VARIANTS:
            control_parent = evidence.get("baseline_control_parent_sha256") if isinstance(evidence, dict) else None
            if (not isinstance(evidence, dict)
                    or not isinstance(control_parent, str)
                    or control_parent.lower() != declaration.get("parent_checkpoint_sha256")
                    or evidence.get("baseline_control_probe_passed") is not True
                    or not _valid_sha256(evidence.get("baseline_control_probe_report_sha256"))
                    or not _positive_finite(evidence.get("baseline_control_median_scored_lap_s"))):
                raise ValueError("campaign time-cost declaration has incomplete same-parent control evidence")
        elif experiment_id == "lr_1e4_best_stable" and not _lr_experiment_evidence_is_healthy(
            evidence if isinstance(evidence, dict) else {}
        ):
            raise ValueError("campaign LR declaration no longer satisfies the required health evidence")
        elif experiment_id in {"continue_best", "continue_best_boundary_recovery", "regression_recovery", "plateau_lr_conservative", "plateau_lr_healthy"}:
            if not isinstance(evidence, dict):
                raise ValueError("trend campaign declaration lacks evidence")
            parent_effective = _validate_parent_relative_config(config, experiment_id, evidence)
            _validate_trend_experiment_evidence(
                state, experiment_id, evidence, str(declaration.get("parent_checkpoint_sha256", ""))
            )
        if declaration.get("declared_config") != {
            key: value for key, value in config.items() if key not in EXPERIMENT_METADATA_FIELDS
        }:
            raise ValueError("campaign experiment effective configuration differs from its declaration")

    expected: dict[str, Any] = {
        "total_timesteps": CAMPAIGN_PLAN["total_timesteps"],
        "stop_on_plateau": CAMPAIGN_PLAN["stop_on_plateau"],
        "max_duration_hours": CAMPAIGN_PLAN["max_duration_hours"],
        "behavior_cloning_enabled": CAMPAIGN_PLAN["behavior_cloning_enabled"],
        "behavior_cloning_only": CAMPAIGN_PLAN["behavior_cloning_only"],
        "observation_profile": CAMPAIGN_PLAN["observation_profile"],
        "simulator_backend": CAMPAIGN_PLAN["simulator_backend"],
        "throttle_mode": CAMPAIGN_PLAN["throttle_mode"],
        "negative_throttle_mode": CAMPAIGN_PLAN["negative_throttle_mode"],
        "steering_mode": CAMPAIGN_PLAN["steering_mode"],
        "learning_rate": CAMPAIGN_PLAN["learning_rate"],
        "target_kl": CAMPAIGN_PLAN["target_kl"],
        "exploration_profile": CAMPAIGN_PLAN["exploration_profile"],
        "n_envs": CAMPAIGN_PLAN["ppo_settings"]["n_envs"],
        "n_steps": CAMPAIGN_PLAN["ppo_settings"]["n_steps"],
        "batch_size": CAMPAIGN_PLAN["ppo_settings"]["batch_size"],
        "n_epochs": CAMPAIGN_PLAN["ppo_settings"]["n_epochs"],
        "gamma": CAMPAIGN_PLAN["ppo_settings"]["gamma"],
        "gae_lambda": CAMPAIGN_PLAN["ppo_settings"]["gae_lambda"],
        "clip_range": CAMPAIGN_PLAN["ppo_settings"]["clip_range"],
        "ent_coef": CAMPAIGN_PLAN["ppo_settings"]["ent_coef"],
        "vf_coef": CAMPAIGN_PLAN["ppo_settings"]["vf_coef"],
        "max_grad_norm": CAMPAIGN_PLAN["ppo_settings"]["max_grad_norm"],
        "device": CAMPAIGN_PLAN["ppo_settings"]["device"],
        "checkpoint_every": CAMPAIGN_PLAN["checkpoint_every_transitions"],
        "automatic_evaluation_enabled": CAMPAIGN_PLAN["automatic_evaluation_enabled"],
        "automatic_evaluation_mode": CAMPAIGN_PLAN["automatic_evaluation_mode"],
        "automatic_evaluation_interval": CAMPAIGN_PLAN["automatic_evaluation_interval"],
        **CAMPAIGN_PLAN["reward_settings"],
    }
    errors = []
    trainer_owned_ppo_fields = {"batch_size", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"}
    for field, target in expected.items():
        if field in trainer_owned_ppo_fields and field not in config:
            # These values are hard-coded in the trainer's bc_campaign fixed
            # preset and are intentionally not exposed as UI/API settings.
            continue
        actual = config.get(field)
        if experiment_id in {"continue_best", "continue_best_boundary_recovery", "regression_recovery", "plateau_lr_conservative", "plateau_lr_healthy"} and parent_effective is not None:
            target = parent_effective.get(field, target)
            if experiment_id in {"regression_recovery", "plateau_lr_conservative"} and field == "learning_rate":
                target = 1e-5
            elif experiment_id == "plateau_lr_healthy" and field == "learning_rate":
                target = 1e-4
        if experiment_id in TIME_COST_VARIANTS and field == "time_cost_per_simulated_second":
            target = _CAMPAIGN_EXPERIMENTS[experiment_id]["time_cost"]
        elif experiment_id == "lr_1e4_best_stable" and field == "learning_rate":
            target = 1e-4
        if isinstance(target, float):
            try:
                equal = actual is not None and abs(float(actual) - target) <= 1e-12
            except (TypeError, ValueError):
                equal = False
        else:
            equal = actual == target
        if not equal:
            errors.append(f"{field} must be {target!r} during the campaign (received {actual!r})")

    if not isinstance(config.get("resume"), str) or not config["resume"].strip():
        errors.append("resume must name an authenticated prior checkpoint during the campaign")

    exploration_targets = {
        "exploration_initial_std_throttle": 0.01,
        "exploration_initial_std_steering": 0.05,
        "exploration_min_std_throttle": 0.005,
        "exploration_min_std_steering": 0.02,
        "exploration_max_std_throttle": 0.01,
        "exploration_max_std_steering": 0.05,
    }
    for field, target in exploration_targets.items():
        actual = config.get(field)
        if actual is not None:
            try:
                equal = abs(float(actual) - target) <= 1e-12
            except (TypeError, ValueError):
                equal = False
            if not equal:
                errors.append(f"{field} must be {target!r} when explicitly supplied during the campaign")
    if errors:
        raise ValueError("campaign run configuration violates fixed campaign plan: " + "; ".join(errors))


def campaign_effective_config_sha256(config: dict[str, Any]) -> str:
    """Hash normalized run settings while excluding their three declaration tags."""
    canonical = {key: value for key, value in config.items() if key not in EXPERIMENT_METADATA_FIELDS}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parent_effective_config(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise ValueError("trend experiment evidence must be an object")
    parent_config = evidence.get("parent_effective_config")
    expected_hash = evidence.get("parent_effective_config_sha256")
    if not isinstance(parent_config, dict) or not _valid_sha256(expected_hash):
        raise ValueError("trend experiment must include the parent effective config and its SHA-256")
    if campaign_effective_config_sha256(parent_config) != expected_hash.lower():
        raise ValueError("trend experiment parent effective config hash mismatch")
    return parent_config


def _validate_parent_relative_config(
    config: dict[str, Any], experiment_id: str, evidence: dict[str, Any],
) -> dict[str, Any]:
    """Require a 5–9h child to preserve its selected parent's effective settings."""
    parent = {key: value for key, value in _parent_effective_config(evidence).items()
              if key not in EXPERIMENT_METADATA_FIELDS}
    child = {key: value for key, value in config.items() if key not in EXPERIMENT_METADATA_FIELDS}
    if not isinstance(child.get("resume"), str) or not child["resume"].strip():
        raise ValueError("trend experiment must resume an authenticated parent checkpoint")
    parent.pop("resume", None)
    child.pop("resume", None)
    if experiment_id in {"continue_best", "continue_best_boundary_recovery"}:
        expected_rate = parent.get("learning_rate")
    elif experiment_id == "regression_recovery":
        expected_rate = 1e-5
    elif experiment_id == "plateau_lr_conservative":
        if abs(float(parent.get("learning_rate", 0.0)) - 3e-5) > 1e-12:
            raise ValueError("plateau_lr_conservative requires a selected parent at learning_rate=3e-5")
        expected_rate = 1e-5
    elif experiment_id == "plateau_lr_healthy":
        if abs(float(parent.get("learning_rate", 0.0)) - 3e-5) > 1e-12:
            raise ValueError("plateau_lr_healthy requires a selected parent at learning_rate=3e-5")
        expected_rate = 1e-4
    else:
        raise ValueError("unsupported parent-relative campaign experiment")
    if experiment_id == "continue_best_boundary_recovery":
        resume_key = str(config.get("resume", "")).replace("\\", "/").lower()
        expected_suffix = CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["checkpoint_path"].lower().removeprefix("logs/")
        if not resume_key.endswith(expected_suffix):
            raise ValueError("continue_best_boundary_recovery must use the exact saved checkpoint path")
    if experiment_id not in {"continue_best", "continue_best_boundary_recovery"}:
        child["learning_rate"] = expected_rate
        parent["learning_rate"] = expected_rate
    if child != parent:
        changed = sorted(key for key in set(parent) | set(child) if parent.get(key) != child.get(key))
        raise ValueError("trend experiment may change only its authorized learning-rate field; differing fields: "
                         + ", ".join(changed))
    return parent


def _validate_trend_experiment_evidence(
    state: dict[str, Any], experiment_id: str, evidence: dict[str, Any], parent_checkpoint_sha256: str,
) -> None:
    parent = _parent_effective_config(evidence)
    if experiment_id == "continue_best":
        if evidence.get("selected_passing_candidate") is not True or evidence.get("selected_passing_checkpoint_sha256") != parent_checkpoint_sha256:
            raise ValueError("continue_best must preserve the selected passing checkpoint")
    elif experiment_id == "continue_best_boundary_recovery":
        recovery = evidence.get("boundary_recovery")
        if not isinstance(recovery, dict) or recovery != CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE:
            raise ValueError("continue_best_boundary_recovery requires the exact authenticated 5h-boundary recovery evidence")
        if parent_checkpoint_sha256 != CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE["checkpoint_sha256"]:
            raise ValueError("continue_best_boundary_recovery must resume the exact saved 204800 checkpoint")
    elif experiment_id == "regression_recovery":
        regressions = evidence.get("consecutive_regressing_probes")
        if (isinstance(regressions, bool) or not isinstance(regressions, int) or regressions < 2
                or evidence.get("infrastructure_failures_excluded") is not True
                or evidence.get("most_recent_passing_checkpoint_sha256") != parent_checkpoint_sha256):
            raise ValueError("regression_recovery requires two policy regressions and the most recent passing checkpoint")
    else:
        plateau_reviews = evidence.get("consecutive_plateau_reviews")
        scheduled_probes = evidence.get("scheduled_probe_count")
        if (isinstance(plateau_reviews, bool) or not isinstance(plateau_reviews, int) or plateau_reviews < 3
                or isinstance(scheduled_probes, bool) or not isinstance(scheduled_probes, int) or scheduled_probes < 3
                or evidence.get("useful_improvement_count") != 0):
            raise ValueError("plateau LR variant requires three scheduled plateau reviews")
        if abs(float(parent.get("learning_rate", 0.0)) - 3e-5) > 1e-12:
            raise ValueError("plateau LR variants require a selected parent at learning_rate=3e-5")
        if experiment_id == "plateau_lr_healthy":
            if evidence.get("best_reward_parent_sha256") != parent_checkpoint_sha256:
                raise ValueError("plateau_lr_healthy must branch from its identified best-reward parent")
            if not _lr_experiment_evidence_is_healthy(evidence):
                raise ValueError("plateau_lr_healthy lacks the required recent audit and probe health evidence")
    reward_cost = float(parent.get("time_cost_per_simulated_second", 0.0))
    if reward_cost not in {0.0, *TIME_COST_VARIANTS.values()}:
        raise ValueError("trend child must preserve one of the campaign's tested reward settings")
    if reward_cost:
        reward_id = next(key for key, value in TIME_COST_VARIANTS.items() if reward_cost == value)
        reward_hash = evidence.get("reward_experiment_config_sha256")
        event = next((row for row in reversed(state.get("events", []))
                      if isinstance(row, dict) and row.get("event") == "campaign_experiment_declared"
                      and row.get("experiment_id") == reward_id
                      and row.get("config_sha256") == reward_hash), None)
        if event is None:
            raise ValueError("trend child reward setting is not linked to its declared tested reward branch")


def _validate_experiment_phase_window(
    state: dict[str, Any], experiment_id: str, phase: str,
    *, utc_now: Optional[datetime] = None, monotonic_ns: Optional[int] = None,
) -> None:
    if campaign_validation_request(state) is not None:
        raise ValueError("training branch declarations are closed while readiness-backed final validation is active")
    policy = _CAMPAIGN_EXPERIMENTS.get(experiment_id)
    if policy is None or policy["phase"] != phase:
        raise ValueError("campaign experiment id/phase is not an approved one-variable branch")
    clock = campaign_clock_status(state, utc_now=utc_now, monotonic_ns=monotonic_ns,
                                  boot_id=state.get("boot_id"))
    elapsed_h = float(clock.get("elapsed_seconds", 0.0)) / 3600.0
    start_h, end_h = policy["hours"]
    unbounded_best_trend = (
        effective_campaign_deadline_policy(state)["mode"] == "unbounded_until_readiness"
        and phase == "best_validated_trend"
        and start_h <= elapsed_h
    )
    if not (unbounded_best_trend or start_h <= elapsed_h < end_h):
        raise ValueError(f"campaign experiment {experiment_id} is outside its approved {start_h:g}-{end_h:g}h phase")


def declare_campaign_experiment(
    state_path: str | Path,
    *,
    experiment_id: str,
    phase: str,
    config: dict[str, Any],
    parent_checkpoint_sha256: str,
    evidence: dict[str, Any],
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Append an exact-hash declaration for one preapproved campaign branch.

    Declarations cannot move either deadline or alter the frozen base plan.
    The pace_experiments phase permits calibrated per-lap retry variants
    (3 or 5 penalty points per BC reference lap), plus an optional
    learning-rate=1e-4 branch only after the explicit stability evidence below.
    """
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None or state.get("status") != "active":
        raise RuntimeError("campaign experiment declaration requires an active campaign")
    _validate_experiment_phase_window(state, experiment_id, phase, utc_now=utc_now, monotonic_ns=monotonic_ns)
    if any(isinstance(row, dict) and row.get("event") == "campaign_experiment_declared"
           and row.get("experiment_id") == experiment_id for row in state.get("events", [])):
        raise ValueError(f"campaign experiment {experiment_id} is already declared")
    if experiment_id in INVALIDATED_TIME_COST_VARIANTS:
        raise ValueError(f"campaign experiment {experiment_id} is invalidated and cannot be redeclared")
    if experiment_id in TIME_COST_VARIANTS:
        sibling_id = {
            "time_cost_3_calibrated_retry": "time_cost_5_calibrated_retry",
            "time_cost_5_calibrated_retry": "time_cost_3_calibrated_retry",
        }[experiment_id]
        sibling = next((row for row in reversed(state.get("events", []))
                        if isinstance(row, dict)
                        and row.get("event") == "campaign_experiment_declared"
                        and row.get("experiment_id") == sibling_id), None)
        if sibling is not None and sibling.get("parent_checkpoint_sha256") != parent_checkpoint_sha256.lower():
            raise ValueError("calibrated time-cost retry variants must fork from the same authenticated parent checkpoint")
    if not isinstance(parent_checkpoint_sha256, str) or len(parent_checkpoint_sha256) != 64:
        raise ValueError("campaign experiment must identify its parent checkpoint SHA-256")
    try:
        int(parent_checkpoint_sha256, 16)
    except ValueError as exc:
        raise ValueError("campaign experiment parent checkpoint hash must be hexadecimal") from exc
    if not isinstance(evidence, dict):
        raise ValueError("campaign experiment evidence must be a JSON object")
    if experiment_id in TIME_COST_VARIANTS:
        control_parent = evidence.get("baseline_control_parent_sha256")
        if not isinstance(control_parent, str) or control_parent.lower() != parent_checkpoint_sha256.lower():
            raise ValueError("time-cost comparison must bind control and variant to the same parent checkpoint")
        if (evidence.get("baseline_control_probe_passed") is not True
                or not _valid_sha256(evidence.get("baseline_control_probe_report_sha256"))
                or not _positive_finite(evidence.get("baseline_control_median_scored_lap_s"))):
            raise ValueError("time-cost comparison requires a passing zero-cost parent probe")
    if experiment_id == "lr_1e4_best_stable":
        best_parent = evidence.get("best_reward_parent_sha256")
        if (not isinstance(best_parent, str) or best_parent.lower() != parent_checkpoint_sha256.lower()
                or not _lr_experiment_evidence_is_healthy(evidence)):
            raise ValueError("lr_1e4_best_stable lacks the required recent audit and probe health evidence")
    if experiment_id in {"continue_best", "continue_best_boundary_recovery", "regression_recovery", "plateau_lr_conservative", "plateau_lr_healthy"}:
        _validate_parent_relative_config(config, experiment_id, evidence)
        _validate_trend_experiment_evidence(state, experiment_id, evidence, parent_checkpoint_sha256.lower())
    declared = dict(config)
    declared.update({
        "campaign_experiment_id": experiment_id,
        "campaign_experiment_phase": phase,
    })
    config_hash = campaign_effective_config_sha256(declared)
    declared["campaign_experiment_config_sha256"] = config_hash
    # Reuse the exact same fixed-plan comparison that protects API admission.
    _validate_declared_experiment_config(declared, experiment_id, evidence=evidence)
    append_campaign_event(path, {
        "event": "campaign_experiment_declared",
        "campaign_id": state.get("campaign_id"),
        "experiment_id": experiment_id,
        "phase": phase,
        "config_sha256": config_hash,
        "declared_config": {key: value for key, value in declared.items()
                            if key not in EXPERIMENT_METADATA_FIELDS},
        "parent_checkpoint_sha256": parent_checkpoint_sha256.lower(),
        "evidence": evidence,
    }, utc_now=utc_now, monotonic_ns=monotonic_ns)
    return {"experiment_id": experiment_id, "phase": phase,
            "config_sha256": config_hash, "config": declared}


def _validate_declared_experiment_config(
    config: dict[str, Any], experiment_id: str, *, evidence: Optional[dict[str, Any]] = None,
) -> None:
    """Validate only an approved one-field override against the frozen plan."""
    if experiment_id not in _CAMPAIGN_EXPERIMENTS:
        raise ValueError("campaign experiment id is not approved")
    if experiment_id in {"continue_best", "continue_best_boundary_recovery", "regression_recovery", "plateau_lr_conservative", "plateau_lr_healthy"}:
        if not isinstance(evidence, dict):
            raise ValueError("trend experiment requires parent-relative evidence")
        _validate_parent_relative_config(config, experiment_id, evidence)
        return
    variant = {key: value for key, value in config.items() if key not in EXPERIMENT_METADATA_FIELDS}
    if experiment_id in TIME_COST_VARIANTS:
        time_cost = _CAMPAIGN_EXPERIMENTS[experiment_id]["time_cost"]
        if variant.get("time_cost_per_simulated_second") != time_cost:
            raise ValueError(f"{experiment_id} must set only time_cost_per_simulated_second to {time_cost}")
        variant["time_cost_per_simulated_second"] = CAMPAIGN_PLAN["reward_settings"]["time_cost_per_simulated_second"]
    elif experiment_id == "lr_1e4_best_stable":
        if float(variant.get("learning_rate", 0.0)) != 1e-4:
            raise ValueError("lr_1e4_best_stable must set learning_rate to 0.0001")
        variant["learning_rate"] = CAMPAIGN_PLAN["learning_rate"]
    errors = []
    expected = {
        "learning_rate": CAMPAIGN_PLAN["learning_rate"],
        "time_cost_per_simulated_second": CAMPAIGN_PLAN["reward_settings"]["time_cost_per_simulated_second"],
        **CAMPAIGN_PLAN["reward_settings"],
    }
    for key, target in expected.items():
        actual = variant.get(key)
        try:
            equal = actual is not None and abs(float(actual) - float(target)) <= 1e-12
        except (TypeError, ValueError):
            equal = actual == target
        if not equal:
            errors.append(f"{key} must remain {target!r}")
    if errors:
        raise ValueError("campaign experiment changes an unapproved variable: " + "; ".join(errors))


def _lr_experiment_evidence_is_healthy(evidence: dict[str, Any]) -> bool:
    probes = evidence.get("passing_zero_collision_probes")
    probe_medians = evidence.get("passing_zero_collision_probe_medians_s")
    probe_report_hashes = evidence.get("passing_zero_collision_probe_report_sha256")
    audit_hashes = evidence.get("last_three_optimizer_audit_sha256")
    kl = evidence.get("last_three_approx_kl")
    clip = evidence.get("last_three_clip_fraction")
    if (evidence.get("all_finite") is not True or evidence.get("actor_fault") is not False
            or evidence.get("sensor_fault") is not False or evidence.get("nonregressing_probe_medians") is not True
            or evidence.get("best_reward_parent_probe_passed") is not True
            or not _valid_sha256(evidence.get("best_reward_parent_probe_report_sha256"))
            or isinstance(probes, bool) or not isinstance(probes, int) or probes < 2
            or not isinstance(probe_medians, list) or len(probe_medians) < 2
            or not isinstance(probe_report_hashes, list) or len(probe_report_hashes) < 2
            or not all(_valid_sha256(value) for value in probe_report_hashes)
            or not isinstance(audit_hashes, list) or len(audit_hashes) != 3
            or not all(_valid_sha256(value) for value in audit_hashes)
            or not isinstance(kl, list) or len(kl) != 3 or not isinstance(clip, list) or len(clip) != 3):
        return False
    try:
        values = [float(value) for value in kl + clip + probe_medians]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) and value >= 0 for value in values):
        return False
    return (statistics.median(float(value) for value in kl) <= 0.01
            and statistics.median(float(value) for value in clip) <= 0.20
            and float(probe_medians[-1]) <= float(probe_medians[0]))


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def _positive_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0


def record_invalidated_time_cost_experiments(
    state_path: str | Path,
    *,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Append audit events for legacy literal 3/5-per-second declarations.

    Existing declaration rows remain immutable. This only appends a distinct
    invalidation row after checking the recorded coefficient and source hash.
    """
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None or state.get("status") != "active":
        raise RuntimeError("invalidated experiment audit requires the active campaign")
    invalidated: list[dict[str, Any]] = []
    affected_runs = {"time_cost_3": ["dcda7cefacef"], "time_cost_5": ["1bd61c474512"]}
    for experiment_id, reason in INVALIDATED_TIME_COST_VARIANTS.items():
        declaration = next((row for row in reversed(state.get("events", []))
                            if isinstance(row, dict)
                            and row.get("event") == "campaign_experiment_declared"
                            and row.get("experiment_id") == experiment_id), None)
        if declaration is None:
            continue
        declared_config = declaration.get("declared_config")
        expected_literal = 3.0 if experiment_id == "time_cost_3" else 5.0
        if (not isinstance(declared_config, dict)
                or declared_config.get("time_cost_per_simulated_second") != expected_literal
                or not _valid_sha256(declaration.get("config_sha256"))):
            raise ValueError(f"legacy {experiment_id} declaration does not match the known invalid literal config")
        already = any(isinstance(row, dict)
                      and row.get("event") == "campaign_experiment_invalidated"
                      and row.get("experiment_id") == experiment_id
                      and row.get("invalidated_config_sha256") == declaration["config_sha256"]
                      for row in state.get("events", []))
        if already:
            continue
        state = append_campaign_event(path, {
            "event": "campaign_experiment_invalidated",
            "campaign_id": state.get("campaign_id"),
            "experiment_id": experiment_id,
            "invalidated_config_sha256": declaration["config_sha256"],
            "observed_time_cost_per_simulated_second": expected_literal,
            "calibrated_time_cost_per_simulated_second": TIME_COST_VARIANTS[
                f"{experiment_id}_calibrated_retry"
            ],
            "affected_run_ids": affected_runs[experiment_id],
            "reason": reason,
            "history_rewritten": False,
        }, utc_now=utc_now, monotonic_ns=monotonic_ns)
        invalidated.append(state["events"][-1])
    return invalidated


def append_campaign_event(
    state_path: str | Path,
    event: dict[str, Any],
    *,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Durably append supervisor evidence without changing any clock origin."""
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None:
        raise FileNotFoundError(f"campaign state not found: {path}")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    row = {
        "timestamp_utc": now.isoformat(),
        "monotonic_ns": int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns),
        **event,
    }
    state.setdefault("events", []).append(row)
    state["revision"] = int(state.get("revision", 0)) + 1
    state["last_observed_at_utc"] = now.isoformat()
    state["last_observed_monotonic_ns"] = row["monotonic_ns"]
    _atomic_write_json(path, state)
    return state


def record_campaign_status(
    state_path: str | Path,
    status: Literal["completed", "expired", "failed_closed"],
    *,
    reason: str,
    utc_now: Optional[datetime] = None,
    monotonic_ns: Optional[int] = None,
) -> dict[str, Any]:
    """Persist a terminal supervisor state without changing either deadline."""
    if status not in {"completed", "expired", "failed_closed"}:
        raise ValueError("campaign terminal status is invalid")
    path = Path(state_path).resolve()
    state = load_campaign_state(path)
    if state is None:
        raise FileNotFoundError(f"campaign state not found: {path}")
    now = (utc_now or _utc_now()).astimezone(timezone.utc)
    previous = state.get("status")
    if previous != status:
        state["status"] = status
        state["phase"] = status
        state["revision"] = int(state.get("revision", 0)) + 1
        row = {
            "timestamp_utc": now.isoformat(),
            "monotonic_ns": int(time.monotonic_ns() if monotonic_ns is None else monotonic_ns),
            "event": "campaign_terminal",
            "previous_status": previous,
            "status": status,
            "reason": str(reason),
        }
        state.setdefault("events", []).append(row)
        state["last_observed_at_utc"] = now.isoformat()
        state["last_observed_monotonic_ns"] = row["monotonic_ns"]
        _atomic_write_json(path, state)
    return state
