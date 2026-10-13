"""Bounded campaign phase controller using the normal Mission Control API.

The independent campaign supervisor owns deadline enforcement. This process
collects immutable evidence, asks the pure campaign planner for eligible
branches, and executes only exact-hash declared candidates through HTTP. It
never controls simulator processes directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if os.name == "nt":
    import msvcrt
    fcntl = None
else:
    import fcntl
    msvcrt = None

from src.layer3.bc_artifact import PPOVerificationContext, verify_ppo_lineage_checkpoint
from src.layer4.hub.campaign_deadline import (
    CAMPAIGN_ID,
    CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE,
    INVALIDATED_CAMPAIGN_RUN_IDS,
    TIME_COST_VARIANTS,
    campaign_clock_status,
    campaign_effective_config_sha256,
    campaign_validation_request,
    declare_campaign_experiment,
    load_campaign_state,
)
from src.layer4.hub.campaign_experiments import build_campaign_decision_plan
from src.layer4.hub.campaign_finalists import build_finalist_manifest
from src.layer4.hub.campaign_trend import derive_campaign_trend_evidence


TERMINAL_STATES = {"complete", "completed", "incomplete", "failed", "stopped", "cancelled"}
ACTIVE_STATES = {"starting", "running", "stopping"}
INVALID_LEGACY_REWARD_RUN_IDS = frozenset(INVALIDATED_CAMPAIGN_RUN_IDS)
OFFICIAL_COMPETITION_SCENE_SHA256 = "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674"
OFFICIAL_COMPETITION_ROUTE_SHA256 = "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb"


def _is_exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _valid_competitive_route_identity(report: dict[str, Any], attempt: dict[str, Any]) -> bool:
    """Validate the monitor-only route and actual Windows simulator identity."""
    from pathlib import PureWindowsPath

    def route_path_ok(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = value.replace("\\", "/").lower().rstrip("/")
        return normalized.endswith("/competition/official_centerline.csv") and "/iros2026/tracks/" in normalized

    def scene_path_ok(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return value.replace("\\", "/").lower().rstrip("/").endswith("/data/level0")

    def simulator_identity_ok(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        pid = value.get("pid")
        port = value.get("port")
        created = value.get("process_creation_utc")
        exe = value.get("executable_path")
        manifest_sha = value.get("manifest_sha256")
        if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                or isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
                or not isinstance(created, str) or not created.strip()
                or not isinstance(exe, str) or not isinstance(manifest_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None):
            return False
        try:
            datetime.fromisoformat(created.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return False
        parts = tuple(part.casefold() for part in PureWindowsPath(exe).parts)
        expected_suffix = (
            "aicar-official-iros-camera", "windows", "autodrive_simulator",
            "autodrive simulator.exe",
        )
        return len(parts) >= len(expected_suffix) and parts[-len(expected_suffix):] == expected_suffix

    # Keep the expression explicit so booleans cannot masquerade as integer counters.
    def monitor_ok(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        sample_count = value.get("projection_sample_count")
        valid_count = value.get("projection_valid_sample_count")
        invalid_count = value.get("projection_invalid_sample_count")
        return (
            value.get("projection_valid") is True
            and isinstance(sample_count, int) and not isinstance(sample_count, bool) and sample_count > 0
            and isinstance(valid_count, int) and not isinstance(valid_count, bool) and valid_count == sample_count
            and _is_exact_int(invalid_count, 0)
            and value.get("scoring_or_stop_effect") is False
        )

    for obj in (report, attempt):
        if (obj.get("route_monitor_only") is not True
                or obj.get("official_scene_sha256") != OFFICIAL_COMPETITION_SCENE_SHA256
                or obj.get("route_sha256") != OFFICIAL_COMPETITION_ROUTE_SHA256
                or obj.get("track_profile") != "competition"
                or not route_path_ok(obj.get("route_path"))
                or not scene_path_ok(obj.get("official_scene_path"))
                or not simulator_identity_ok(obj.get("simulator_identity"))):
            return False
    identity = report.get("simulator_identity")
    return (
        identity == attempt.get("simulator_identity")
        and report.get("official_scene_sha256") == attempt.get("official_scene_sha256")
        and report.get("route_sha256") == attempt.get("route_sha256")
        and report.get("route_path") == attempt.get("route_path")
        and report.get("official_scene_path") == attempt.get("official_scene_path")
        and report.get("track_profile") == attempt.get("track_profile")
        and attempt.get("route_projection_valid") is True
        and attempt.get("frontier_projection_valid") is True
        and monitor_ok(attempt.get("route_monitor"))
    )


def _campaign_evidence_state_signature(state: dict[str, Any]) -> str:
    """Ignore supervisor timestamps only, but bind durable active-work identities and decisions."""
    stable = dict(state)
    for key in ("last_observed_at_utc", "last_observed_monotonic_ns", "revision", "updated_at_utc"):
        stable.pop(key, None)
    events = stable.get("events")
    if isinstance(events, list):
        normalized_events = []
        for event in events:
            if not isinstance(event, dict):
                normalized_events.append(event)
                continue
            normalized = dict(event)
            if normalized.get("event") in {"supervisor_active_work_observed", "supervisor_api_unavailable"}:
                normalized.pop("timestamp_utc", None)
                normalized.pop("monotonic_ns", None)
            normalized_events.append(normalized)
        stable["events"] = normalized_events
    return hashlib.sha256(_json_bytes(stable)).hexdigest()


def _pace_phase_boundary_stop_allowed(elapsed_hours: float, *, replay_active: bool) -> bool:
    """Do not stop the paused learner until its in-flight probe has finished."""
    return elapsed_hours >= 5.0 and not replay_active


def _valid_completed_rollout_steps(audit_rows: Any, *, parent_step: int, trigger_step: int) -> set[int]:
    """Return distinct finite PPO updates bounded to the evaluated snapshot."""
    if not isinstance(audit_rows, list):
        return set()
    return {
        row["num_timesteps"]
        for row in audit_rows
        if isinstance(row, dict)
        and isinstance(row.get("num_timesteps"), int)
        and not isinstance(row.get("num_timesteps"), bool)
        and parent_step < row["num_timesteps"] <= trigger_step
        and row.get("metrics_finite") is True
        and row.get("gradients_finite") is True
        and row.get("parameters_finite") is True
        and row.get("optimizer_state_finite") is True
    }


def _checkpoint_matches_trigger_bucket(
    verified: Any, trigger_step: int, *, rollout_size: int = 2048,
) -> bool:
    """Bind a cadence bucket to the frozen post-update step that fulfilled it."""
    if not isinstance(verified, dict):
        return False
    actual_step = verified.get("num_timesteps")
    lineage = verified.get("lineage")
    lineage_trigger = lineage.get("trigger_step") if isinstance(lineage, dict) else None
    return bool(
        isinstance(actual_step, int) and not isinstance(actual_step, bool)
        and actual_step >= trigger_step
        and actual_step - trigger_step < rollout_size
        and _is_exact_int(lineage_trigger, trigger_step)
    )


def _declared_reward_branch_hash(state: dict[str, Any], request: Any) -> Optional[str]:
    """Return the declaration hash only when a nonzero reward branch is state-bound."""
    if not isinstance(request, dict):
        return None
    cost = request.get("time_cost_per_simulated_second", 0.0)
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    try:
        cost = float(cost)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(cost):
        return None
    if math.isclose(cost, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return ""
    experiment_id = request.get("campaign_experiment_id")
    expected_cost = TIME_COST_VARIANTS.get(experiment_id)
    declared_hash = request.get("campaign_experiment_config_sha256")
    if expected_cost is None:
        # Trend children inherit reward settings while their experiment ID
        # changes (for example continue_best). Re-link them to the unique
        # campaign declaration that introduced this exact calibrated value.
        matching = next((
            (key, value) for key, value in TIME_COST_VARIANTS.items()
            if math.isclose(cost, value, rel_tol=0.0, abs_tol=1e-15)
        ), None)
        if matching is not None:
            experiment_id, expected_cost = matching
            declared_hash = None
    if (expected_cost is None or not math.isclose(cost, expected_cost, rel_tol=0.0, abs_tol=1e-15)):
        return None
    for event in reversed(state.get("events", [])):
        if not isinstance(event, dict) or event.get("event") != "campaign_experiment_declared":
            continue
        declared = event.get("declared_config")
        event_hash = event.get("config_sha256")
        if (event.get("experiment_id") == experiment_id
                and isinstance(event_hash, str) and re.fullmatch(r"[0-9a-f]{64}", event_hash)
                and (declared_hash is None or event.get("config_sha256") == declared_hash)
                and isinstance(declared, dict)
                and not isinstance(declared.get("time_cost_per_simulated_second"), bool)
                and isinstance(declared.get("time_cost_per_simulated_second"), (int, float))
                and math.isclose(float(declared["time_cost_per_simulated_second"]), expected_cost,
                                 rel_tol=0.0, abs_tol=1e-15)):
            # For inherited continue/recovery settings there is no direct
            # declaration hash on the child request. Return the verified
            # source declaration's hash so the trend helper can bind it.
            return event_hash
    return None


def _campaign_reward_ancestry_link(
    state: dict[str, Any], *, run_id: str, run_index: dict[str, dict[str, Any]],
    repo_root: Path, verification_context: PPOVerificationContext,
    checkpoint: Path, verified_checkpoint: dict[str, Any],
) -> Optional[str]:
    """Bind a nonzero reward coefficient to the exact declared training ancestry.

    A matching numeric coefficient is insufficient: each training run must
    resume the checkpoint named by its authenticated child sidecar, and the
    ancestry must reach the calibrated campaign declaration with its exact
    config hash.
    """
    current_id = str(run_id)
    current_checkpoint = checkpoint
    current_verified: Optional[dict[str, Any]] = verified_checkpoint
    seen: set[str] = set()
    initial_cost: Optional[float] = None

    while current_id and current_id not in seen:
        seen.add(current_id)
        run = run_index.get(current_id)
        run_dir = repo_root / "logs" / "rl" / f"official_run_{current_id}"
        manager = verification_context.read_json(run_dir / "manager_config.json")
        manager_config = manager.get("config") if isinstance(manager, dict) else None
        api_config = run.get("config") if isinstance(run, dict) else None
        config = manager_config if isinstance(manager_config, dict) else api_config
        if not isinstance(config, dict) or (isinstance(manager, dict) and manager.get("run_id") != current_id):
            return None
        if isinstance(manager_config, dict) and isinstance(api_config, dict):
            for key in ("resume", "time_cost_per_simulated_second", "campaign_experiment_id",
                        "campaign_experiment_config_sha256"):
                if manager_config.get(key) != api_config.get(key):
                    return None
        cost_value = config.get("time_cost_per_simulated_second")
        if (not isinstance(cost_value, (int, float)) or isinstance(cost_value, bool)
                or not math.isfinite(float(cost_value))):
            return None
        cost = float(cost_value)
        trainer_config = verification_context.read_json(run_dir / "config.json")
        trainer_reward = trainer_config.get("training_only_reward") if isinstance(trainer_config, dict) else None
        trainer_cost = trainer_reward.get("time_cost_per_simulated_second") if isinstance(trainer_reward, dict) else None
        if (not isinstance(trainer_cost, (int, float)) or isinstance(trainer_cost, bool)
                or not math.isfinite(float(trainer_cost))
                or not math.isclose(float(trainer_cost), cost, rel_tol=0.0, abs_tol=1e-15)):
            return None
        if initial_cost is None:
            initial_cost = cost
        elif not math.isclose(cost, initial_cost, rel_tol=0.0, abs_tol=1e-15):
            return None
        if math.isclose(cost, 0.0, rel_tol=0.0, abs_tol=1e-15):
            return "" if initial_cost == 0.0 else None

        lineage = current_verified.get("lineage") if isinstance(current_verified, dict) else None
        if (not isinstance(lineage, dict) or lineage.get("run_id") != current_id
                or current_verified.get("checkpoint_sha256") != verification_context.hash_file(current_checkpoint)):
            return None

        resume_path = _resolve_model_path(config.get("resume"), repo_root)
        if resume_path is None or not resume_path.is_file():
            return None
        parent_verified = verify_ppo_lineage_checkpoint(
            resume_path, verification_context=verification_context,
        )
        if not isinstance(parent_verified, dict):
            return None
        if current_verified.get("parent_checkpoint_sha256") != parent_verified.get("checkpoint_sha256"):
            return None
        parent_lineage = parent_verified.get("lineage")
        parent_run_id = parent_lineage.get("run_id") if isinstance(parent_lineage, dict) else None
        if not isinstance(parent_run_id, str) or not parent_run_id:
            return None

        experiment_id = config.get("campaign_experiment_id")
        if experiment_id in TIME_COST_VARIANTS:
            declared_hash = _declared_reward_branch_hash(state, config)
            if declared_hash is None:
                return None
            if (config.get("campaign_experiment_config_sha256") != declared_hash
                    or campaign_effective_config_sha256(config) != declared_hash):
                return None
            # The event must describe this exact declared request, including
            # its resume identity and all unchanged training settings.
            matching_event = next((event for event in reversed(state.get("events", []))
                                   if isinstance(event, dict)
                                   and event.get("event") == "campaign_experiment_declared"
                                   and event.get("experiment_id") == experiment_id
                                   and event.get("config_sha256") == declared_hash), None)
            declared_config = matching_event.get("declared_config") if isinstance(matching_event, dict) else None
            if not isinstance(declared_config, dict):
                return None
            if campaign_effective_config_sha256(declared_config) != declared_hash:
                return None
            return declared_hash

        current_id = parent_run_id
        current_checkpoint = resume_path
        current_verified = parent_verified
    return None


def _is_calibrated_pace_branch(config: Any) -> bool:
    if not isinstance(config, dict):
        return False
    experiment_id = config.get("campaign_experiment_id")
    expected = TIME_COST_VARIANTS.get(experiment_id)
    cost = config.get("time_cost_per_simulated_second")
    return bool(
        expected is not None
        and isinstance(cost, (int, float)) and not isinstance(cost, bool)
        and math.isfinite(float(cost))
        and math.isclose(float(cost), expected, rel_tol=0.0, abs_tol=1e-15)
    )


def _resume_parent_run_id(config: Any) -> Optional[str]:
    resume = config.get("resume") if isinstance(config, dict) else None
    if not isinstance(resume, str):
        return None
    match = re.search(r"official_run_([0-9a-f]{12})", resume.casefold())
    return match.group(1) if match else None


def _campaign_run_reward_is_eligible(
    config: Any, *, run_id: Optional[str] = None, run_index: Optional[dict[str, dict[str, Any]]] = None,
    seen: Optional[set[str]] = None,
) -> bool:
    """Exclude historical campaigns whose reward coefficient used old literal units."""
    if run_id in INVALID_LEGACY_REWARD_RUN_IDS:
        return False
    if not isinstance(config, dict):
        return False
    value = config.get("time_cost_per_simulated_second")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return False
    value = float(value)
    reward_ok = math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-15) or any(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15)
        for expected in TIME_COST_VARIANTS.values()
    )
    if not reward_ok:
        return False
    parent_id = _resume_parent_run_id(config)
    if parent_id is None:
        return True
    if parent_id in INVALID_LEGACY_REWARD_RUN_IDS:
        return False
    if run_index is None:
        return True
    seen_ids = set() if seen is None else seen
    if parent_id in seen_ids:
        return False
    parent = run_index.get(parent_id)
    if parent is None:
        # If the resumed run has been pruned from API history, the checkpoint
        # path itself can still identify a known invalid ancestor; otherwise
        # strict PPO lineage validation remains responsible for authentication.
        return True
    seen_ids.add(parent_id)
    parent_config = parent.get("config")
    return _campaign_run_reward_is_eligible(
        parent_config, run_id=parent_id, run_index=run_index, seen=seen_ids,
    )


def _not_older_than(value: Any, reference: Any) -> bool:
    try:
        current = datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
        prior = datetime.fromisoformat(str(reference).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False
    return current >= prior


def _timestamp_age_s(value: Any, *, now: Optional[datetime] = None) -> Optional[float]:
    try:
        observed = datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current.astimezone(timezone.utc) - observed).total_seconds())


def _local_run_liveness(run_dir: Path, run_id: str, *, now: Optional[datetime] = None) -> dict[str, Any]:
    """Read per-run producer artifacts; API/UI telemetry is deliberately not authoritative."""
    sources: list[dict[str, Any]] = []
    for name in ("watch_latest.json", "latest_telemetry.json"):
        path = run_dir / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict) or str(value.get("run_id", "")) != run_id:
            continue
        age = _timestamp_age_s(value.get("ts"), now=now)
        step = value.get("step")
        if age is None or not isinstance(step, int) or isinstance(step, bool) or step < 0:
            continue
        if name == "watch_latest.json":
            cars = value.get("cars")
            car = cars[0] if isinstance(cars, list) and cars and isinstance(cars[0], dict) else None
            pose = car.get("pose") if car else None
            lidar = car.get("lidar") if car else None
            valid = (isinstance(pose, list) and len(pose) >= 2
                     and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in pose[:2])
                     and isinstance(lidar, list) and len(lidar) > 0
                     and car.get("route_projection_valid") is True)
            camera_age = car.get("camera_age_s")
            if car.get("camera_fresh") is False:
                valid = False
            if isinstance(camera_age, (int, float)) and not isinstance(camera_age, bool):
                valid = valid and math.isfinite(float(camera_age)) and float(camera_age) <= 0.5
            is_reset = bool(car and car.get("reset") is True)
        else:
            car = None
            valid = False
            is_reset = False
        sources.append({"source": name, "path": str(path), "age_s": age, "step": step,
                        "timestamp": value.get("ts"), "car": car,
                        "valid_sensor_frame": bool(valid), "reset_in_progress": is_reset})
    sources.sort(key=lambda item: (item["age_s"], item["source"]))
    freshest = sources[0] if sources else None
    watch = next((item for item in sources if item["source"] == "watch_latest.json"), None)
    # latest_telemetry is a trainer metrics heartbeat, not sensor evidence.
    # Sensor health must therefore be established by the official Watch frame.
    sensor_fault = (watch is None or watch["age_s"] > 45.0
                    or (not watch.get("valid_sensor_frame") and not watch.get("reset_in_progress")))
    return {"freshest": freshest, "watch": watch, "sources": sources,
            "sensor_fault": sensor_fault}


def _sensor_fault_from_freshness(
    local: dict[str, Any], *, replay_active: bool, run_state: Optional[str] = None,
    run_age_s: Optional[float] = None, startup_grace_s: float = 120.0,
) -> bool:
    # A live evaluator owns the simulator during a pause, so training telemetry
    # is expected to age while the training environment is cooperatively paused.
    if replay_active:
        return False
    watch = local.get("watch")
    if watch is None and (run_state == "starting" or (run_age_s is not None and run_age_s < startup_grace_s)):
        return False
    return bool(local.get("sensor_fault", True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _immutable_json(directory: Path, prefix: str, payload: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"{prefix}_{stamp}.json"
    encoded = _json_bytes(payload)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


class CampaignAPI:
    def __init__(self, base_url: str, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(self, method: str, route: str, body: Optional[dict[str, Any]] = None) -> Any:
        data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"campaign API {method} {route} failed: {exc}") from exc
        if not payload:
            return {}
        result = json.loads(payload)
        return result

    def get(self, route: str) -> Any:
        return self.request("GET", route)

    def post(self, route: str, body: Optional[dict[str, Any]] = None) -> Any:
        return self.request("POST", route, body or {})


def _resolve_model_path(value: Any, repo_root: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.startswith("/runs/rl/"):
        path = repo_root / "logs" / "rl" / raw.removeprefix("/runs/rl/")
    elif raw.startswith("/app/logs/rl/"):
        path = repo_root / "logs" / "rl" / raw.removeprefix("/app/logs/rl/")
    elif "/logs/rl/" in raw.replace("\\", "/"):
        # Native WSL process records use the mounted Windows workspace path
        # (for example /mnt/c/.../AiCar/logs/rl/...). Resolve only its suffix
        # beneath this runner's own logs/rl root; the containment check below
        # still rejects traversal or a path from another repository.
        normalized = raw.replace("\\", "/")
        prefix, suffix = normalized.split("/logs/rl/", 1)
        if repo_root.name.casefold() not in {part.casefold() for part in prefix.split("/") if part}:
            return None
        path = repo_root / "logs" / "rl" / suffix
    else:
        path = Path(raw)
    try:
        path = path.resolve()
        path.relative_to((repo_root / "logs" / "rl").resolve())
    except (OSError, ValueError):
        return None
    return path


def _model_alias(path: Path, repo_root: Path) -> str:
    root = (repo_root / "logs" / "rl").resolve()
    path = path.resolve()
    relative = path.relative_to(root)
    return "/runs/rl/" + relative.as_posix()


def _report_path(record: dict[str, Any], repo_root: Path) -> Optional[Path]:
    result = record.get("result")
    attempts = result.get("runs") if isinstance(result, dict) else None
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[0], dict):
        return None
    return _resolve_model_path(attempts[0].get("evaluation_report"), repo_root)


def _repo_root_for_run_report(path: Path) -> Optional[Path]:
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name == "rl" and parent.parent.name == "logs":
            return parent.parent.parent
    return None


def validate_training_probe(
    record: dict[str, Any], report: dict[str, Any], expected_sha: str, *,
    report_path: Optional[Path] = None, repo_root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Return measured probe facts only when the whole 1+3 protocol passed."""
    if record.get("state") != "complete" or record.get("validation_mode") != "training_probe":
        return None
    if record.get("model_sha256") != expected_sha or report.get("model_sha256") != expected_sha:
        return None
    result = record.get("result")
    if not isinstance(result, dict) or result.get("readiness_passed") is not True:
        return None
    attempts = result.get("runs")
    runs = report.get("runs")
    if (not isinstance(attempts, list) or len(attempts) != 1
            or not isinstance(attempts[0], dict)
            or not isinstance(runs, list) or len(runs) != 1):
        return None
    service_attempt = attempts[0]
    attempt = runs[0]
    if not isinstance(attempt, dict):
        return None
    # Training probes are explicitly non-ranking: the service deliberately
    # sets valid_for_comparison=false and the report's per-run record omits its
    # run_id. Bind the durable result to the report via the service run_id and
    # the canonical per-run report location instead of the competition flag.
    service_run_id = service_attempt.get("run_id")
    if (not isinstance(service_run_id, str) or not service_run_id
            or service_attempt.get("score_status") != "valid"
            or not _is_exact_int(service_attempt.get("warmup_laps"), 1)
            or not _is_exact_int(service_attempt.get("race_laps_completed"), 3)
            or not _is_exact_int(service_attempt.get("race_laps_required"), 3)
            or not _is_exact_int(service_attempt.get("collision_count"), 0)
            or not _is_exact_int(service_attempt.get("race_collisions"), 0)
            or service_attempt.get("readiness_passed") is not True
            or report_path is None or repo_root is None):
        return None
    recorded_report = _resolve_model_path(service_attempt.get("evaluation_report"), repo_root)
    expected_report = repo_root / "logs" / "rl" / f"official_run_{service_run_id}" / "evaluation.json"
    if (recorded_report is None or recorded_report.resolve() != report_path.resolve()
            or report_path.resolve() != expected_report.resolve()):
        return None
    times = attempt.get("race_lap_times_s")
    warmup = attempt.get("warmup_lap_times_s")
    collision_count = attempt.get("collision_count", 0)
    if (
        report.get("protocol") != "official_training_probe_1_plus_3"
        or report.get("validation_mode") != "training_probe"
        or attempt.get("stop_reason") != "race_complete"
        or attempt.get("frontier_projection_valid") is not True
        or not isinstance(collision_count, int) or isinstance(collision_count, bool) or collision_count != 0
        or not _is_exact_int(attempt.get("race_collisions"), 0)
        or not _is_exact_int(attempt.get("race_laps_completed"), 3)
        or not _is_exact_int(attempt.get("race_laps_required"), 3)
        or not isinstance(times, list) or len(times) != 3
        or not isinstance(warmup, list) or len(warmup) != 1
    ):
        return None
    camera_count = attempt.get("camera_sample_count")
    if (not isinstance(camera_count, int) or isinstance(camera_count, bool) or camera_count <= 0
            or not _is_exact_int(attempt.get("camera_fresh_sample_count"), camera_count)
            or not _is_exact_int(attempt.get("camera_stale_sample_count"), 0)
            or not _is_exact_int(attempt.get("camera_missing_sample_count"), 0)
            or not _is_exact_int(attempt.get("camera_freshness_unknown_count"), 0)):
        return None
    values = [*times, *warmup]
    try:
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            return None
    except (OverflowError, TypeError, ValueError):
        return None
    return {
        "median_scored_lap_s": statistics.median(float(value) for value in times),
        "scored_lap_times_s": [float(value) for value in times],
        "collision_count": 0,
        "projection_valid": True,
        "evaluation_id": record.get("evaluation_id"),
        "model_sha256": expected_sha,
    }


def _pace_probe_is_screening_complete(
    record: dict[str, Any], train_row: dict[str, Any], repo_root: Path,
    verification_context: Optional[PPOVerificationContext] = None,
) -> bool:
    """Ignore resume-seed probes until two new rollout updates are persisted.

    A campaign profile resumed from a global cadence boundary can enqueue one
    snapshot on its first callback. That snapshot is useful diagnostics, but it
    does not measure the candidate policy after training. A pace branch only
    becomes eligible for its screening stop after at least 4096 new global
    transitions, a verified post-update checkpoint, and two finite local PPO
    audit rows beyond its parent checkpoint.
    """
    if record.get("validation_mode") != "training_probe" or record.get("state") not in {
        "complete", "incomplete", "failed", "stopped",
    }:
        return False
    config = train_row.get("config")
    if not isinstance(config, dict):
        return False
    parent_path = _resolve_model_path(config.get("resume"), repo_root)
    snapshot_path = _resolve_model_path(record.get("model_path"), repo_root)
    if parent_path is None or snapshot_path is None or not parent_path.is_file() or not snapshot_path.is_file():
        return False
    try:
        if verification_context is None:
            parent = verify_ppo_lineage_checkpoint(parent_path)
            child = verify_ppo_lineage_checkpoint(snapshot_path)
        else:
            parent = verify_ppo_lineage_checkpoint(parent_path, verification_context=verification_context)
            child = verify_ppo_lineage_checkpoint(snapshot_path, verification_context=verification_context)
        if (not parent or not child
                or child.get("parent_checkpoint_sha256") != parent.get("checkpoint_sha256")):
            return False
        parent_steps = int(parent.get("num_timesteps", -1))
        minimum_steps = parent_steps + 2 * 2048
        child_steps = int(child.get("num_timesteps", -1))
        trigger_step = int(record.get("trigger_step", record.get("snapshot_timesteps", -1)))
        rollout_size = int(config.get("n_steps", 2048))
        if (not _checkpoint_matches_trigger_bucket(child, trigger_step, rollout_size=rollout_size)
                or child_steps < minimum_steps):
            return False
        if record.get("model_sha256") != child.get("checkpoint_sha256"):
            return False
        run_dir = Path(str(train_row.get("output_dir") or (repo_root / "logs" / "rl" / f"official_run_{train_row.get('run_id')}")))
        audit_path = run_dir / "ppo_optimizer_audit.jsonl"
        audit_rows = []
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            audit = json.loads(line)
            step = audit.get("num_timesteps")
            if (isinstance(step, int) and not isinstance(step, bool) and parent_steps < step <= child_steps
                    and audit.get("metrics_finite") is True
                    and audit.get("gradients_finite") is True
                    and audit.get("parameters_finite") is True
                    and audit.get("optimizer_state_finite") is True):
                audit_rows.append(audit)
        audit_steps = _valid_completed_rollout_steps(
            audit_rows, parent_step=parent_steps, trigger_step=child_steps,
        )
        if len(audit_steps) < 2 or max(audit_steps, default=-1) < minimum_steps:
            return False
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(attempts[0], dict):
            result = record.get("result")
            attempts = result.get("runs") if isinstance(result, dict) else None
        if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(attempts[0], dict):
            return False
        attempt = attempts[0]
        if (attempt.get("stop_reason") not in {"race_complete", "collision_event"}
                or attempt.get("frontier_projection_valid") is not True):
            return False
        samples = attempt.get("camera_sample_count")
        if not isinstance(samples, int) or isinstance(samples, bool):
            return False
        if samples <= 0:
            return False
        return all(_is_exact_int(attempt.get(key), 0) for key in (
            "camera_stale_sample_count", "camera_missing_sample_count", "camera_freshness_unknown_count",
        ))
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return False


def validate_final_evaluation(
    record: dict[str, Any], report_paths: list[Path], *, expected_sha256: str,
) -> dict[str, Any]:
    """Fail-closed gate for a standard 3-attempt, 1+10 competition evaluation."""
    errors: list[str] = []
    if record.get("state") != "complete":
        errors.append("evaluation state is not complete")
    if record.get("model_sha256") != expected_sha256:
        errors.append("evaluation history loaded SHA differs from requested checkpoint SHA")
    if record.get("source") != "manual":
        errors.append("finalist evaluation was not queued as a manual fresh-checkpoint evaluation")
    result = record.get("result")
    if not isinstance(result, dict):
        errors.append("evaluation result is missing")
        result = {}
    if result.get("model_sha256") != expected_sha256:
        errors.append("evaluation result loaded SHA differs from requested checkpoint SHA")
    if (result.get("validation_mode") is not None
            or result.get("protocol") != "official_10_lap_race"
            or result.get("competitive_score_valid") is not True):
        errors.append("evaluation did not use the standard competitive race protocol")
    attempts = result.get("runs")
    if not isinstance(attempts, list) or len(attempts) != 3 or len(report_paths) != 3:
        errors.append("exactly three completed attempt reports are required")
        attempts = attempts if isinstance(attempts, list) else []
    checked: list[dict[str, Any]] = []
    bound_paths: set[Path] = set()
    bound_run_ids: set[str] = set()
    for index, path in enumerate(report_paths[:3]):
        resolved_report = path.resolve()
        if resolved_report in bound_paths:
            errors.append(f"attempt report {index + 1} duplicates a prior report")
        bound_paths.add(resolved_report)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"attempt report {index + 1} cannot be read: {exc}")
            continue
        if not isinstance(report, dict) or report.get("model_sha256") != expected_sha256:
            errors.append(f"attempt report {index + 1} loaded SHA mismatch")
            continue
        if (report.get("validation_mode") is not None
                or report.get("protocol") != "official_competitive_1_plus_10"
                or report.get("competitive_score_valid") is not True
                or report.get("route_monitor_only") is not True
                or report.get("official_scene_sha256") != OFFICIAL_COMPETITION_SCENE_SHA256
                or report.get("route_sha256") != OFFICIAL_COMPETITION_ROUTE_SHA256
                or report.get("track_profile") != "competition"):
            errors.append(f"attempt report {index + 1} lacks authenticated competitive route/scene identity")
        report_runs = report.get("runs")
        if not isinstance(report_runs, list) or len(report_runs) != 1:
            errors.append(f"attempt report {index + 1} does not contain exactly one run")
            continue
        attempt = report_runs[0]
        if not isinstance(attempt, dict):
            errors.append(f"attempt report {index + 1} is not an object")
            continue
        report_run_id = attempt.get("run_id")
        if not isinstance(report_run_id, str) or not report_run_id or report_run_id in bound_run_ids:
            errors.append(f"attempt report {index + 1} has a missing or duplicate run id")
        elif report_run_id:
            bound_run_ids.add(report_run_id)
        service_attempt = attempts[index] if index < len(attempts) and isinstance(attempts[index], dict) else None
        if service_attempt is None:
            errors.append(f"service attempt {index + 1} is missing")
        else:
            report_repo_root = _repo_root_for_run_report(path)
            recorded_path = _resolve_model_path(service_attempt.get("evaluation_report"), report_repo_root) if report_repo_root else None
            if recorded_path is None or recorded_path.resolve() != path.resolve():
                errors.append(f"attempt report {index + 1} is not bound to the service history row")
            if service_attempt.get("run_id") != attempt.get("run_id"):
                errors.append(f"attempt report {index + 1} run id differs from service history")
            if (not _is_exact_int(service_attempt.get("race_laps_completed"), 10)
                    or not _is_exact_int(service_attempt.get("warmup_laps"), 1)):
                errors.append(f"service attempt {index + 1} does not record exactly 1+10 laps")
            if (not _is_exact_int(service_attempt.get("collision_count"), 0)
                    or not _is_exact_int(service_attempt.get("race_collisions"), 0)
                    or not _is_exact_int(service_attempt.get("camera_stale_sample_count"), 0)
                    or not _is_exact_int(service_attempt.get("camera_missing_sample_count"), 0)
                    or not _is_exact_int(service_attempt.get("camera_freshness_unknown_count"), 0)
                    or service_attempt.get("valid_for_comparison") is not True):
                errors.append(f"service attempt {index + 1} is not a valid zero-collision result")
            if (not _valid_competitive_route_identity(
                    {key: service_attempt.get(key) for key in (
                        "route_monitor_only", "official_scene_sha256", "route_sha256",
                        "track_profile", "route_path", "official_scene_path", "simulator_identity",
                    )}, service_attempt)
                    or any(service_attempt.get(key) != attempt.get(key) for key in (
                        "route_monitor_only", "route_monitor", "official_scene_sha256",
                        "route_sha256", "track_profile", "route_path", "official_scene_path",
                        "simulator_identity", "route_projection_valid", "frontier_projection_valid",
                    ))):
                errors.append(f"service attempt {index + 1} route/scene/process identity differs from its report")
        lap_times = attempt.get("race_lap_times_s") if isinstance(attempt, dict) else None
        warmup = attempt.get("warmup_lap_times_s") if isinstance(attempt, dict) else None
        collision_count = attempt.get("collision_count")
        valid_collision_count = isinstance(collision_count, int) and not isinstance(collision_count, bool) and collision_count == 0
        final_camera_count = attempt.get("camera_sample_count")
        if (
            attempt.get("stop_reason") != "race_complete"
            or not _valid_competitive_route_identity(report, attempt)
            or not valid_collision_count
            or not _is_exact_int(attempt.get("warmup_collisions"), 0)
            or not _is_exact_int(attempt.get("raw_collision_count_at_finish"), 0)
            or not _is_exact_int(attempt.get("race_collisions"), 0)
            or not _is_exact_int(attempt.get("camera_stale_sample_count"), 0)
            or not _is_exact_int(attempt.get("camera_missing_sample_count"), 0)
            or not _is_exact_int(attempt.get("camera_freshness_unknown_count"), 0)
            or not isinstance(final_camera_count, int) or isinstance(final_camera_count, bool) or final_camera_count <= 0
            or not _is_exact_int(attempt.get("camera_fresh_sample_count"), final_camera_count)
            or not isinstance(lap_times, list) or len(lap_times) != 10
            or not isinstance(warmup, list) or len(warmup) != 1
        ):
            errors.append(f"attempt report {index + 1} failed 1+10, projection, completion, or zero-collision gate")
        measured_laps = []
        if isinstance(lap_times, list):
            measured_laps.extend(lap_times)
        if isinstance(warmup, list):
            measured_laps.extend(warmup)
        try:
            if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(float(value)) or float(value) <= 0
                   for value in measured_laps):
                errors.append(f"attempt report {index + 1} has non-finite or non-positive lap timing")
        except (OverflowError, TypeError, ValueError):
            errors.append(f"attempt report {index + 1} has invalid lap timing")
        scored_race_time = None
        scored_lap_values: list[float] = []
        if isinstance(lap_times, list):
            try:
                scored_values = [float(value) for value in lap_times
                                 if isinstance(value, (int, float)) and not isinstance(value, bool)]
                if (len(scored_values) == 10 and all(math.isfinite(value) and value > 0 for value in scored_values)):
                    scored_race_time = sum(scored_values)
                    scored_lap_values = scored_values
            except (OverflowError, TypeError, ValueError):
                scored_race_time = None
        checked.append({
            "report_path": str(path), "report_sha256": _sha256(path),
            "run_id": attempt.get("run_id") if isinstance(attempt, dict) else None,
            "warmup_laps": len(warmup) if isinstance(warmup, list) else 0,
            "scored_laps": len(lap_times) if isinstance(lap_times, list) else 0,
            "scored_lap_times_s": scored_lap_values,
            "scored_race_time_s": scored_race_time,
            "collision_count": collision_count,
            "projection_valid": attempt.get("frontier_projection_valid"),
        })
    if result.get("attempts_required") != 3 or result.get("competitive_score_valid") is not True:
        errors.append("service did not mark all three attempts as competitive and complete")
    return {
        "passed": not errors,
        "evaluation_id": record.get("evaluation_id"),
        "requested_model_sha256": expected_sha256,
        "loaded_model_sha256": result.get("model_sha256"),
        "attempts": checked,
        "errors": errors,
    }


class CampaignPhaseRunner:
    """One-owner, restart-safe phase driver. Use `tick()` once or in a loop."""

    def __init__(
        self,
        *,
        state_path: Path,
        repo_root: Path,
        api: CampaignAPI,
        poll_seconds: float = 60.0,
        finalists_manifest: Optional[Path] = None,
    ) -> None:
        self.state_path = state_path.resolve()
        self.repo_root = repo_root.resolve()
        self.api = api
        self.poll_seconds = poll_seconds
        self.finalists_manifest = (finalists_manifest or (
            self.repo_root / "logs" / "rl" / "official_campaign" / "finalists_manifest.json"
        )).resolve()
        self.output_dir = self.repo_root / "logs" / "rl" / "official_campaign" / "phase_runner"
        self.clean_reload_runtime = self.output_dir / "clean_reload_runtime.json"
        self.events_path = self.output_dir / "actions.jsonl"
        self.lock_path = self.output_dir / "runner.lock"
        self._lock_stream: Any = None

    def acquire(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock_stream = self.lock_path.open("a+b")
        try:
            if fcntl is not None:
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                self._lock_stream.seek(0)
                if self._lock_stream.read(1):
                    self._lock_stream.seek(0)
                else:
                    self._lock_stream.write(b"\0")
                    self._lock_stream.flush()
                    self._lock_stream.seek(0)
                msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError) as exc:
            self._lock_stream.close()
            self._lock_stream = None
            raise RuntimeError("another campaign phase runner owns the lock") from exc
        self._lock_stream.seek(0)
        self._lock_stream.truncate()
        self._lock_stream.write(f"pid={os.getpid()}\n".encode())
        self._lock_stream.flush()

    def close(self) -> None:
        if self._lock_stream is not None:
            if fcntl is not None:
                fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            else:
                self._lock_stream.seek(0)
                msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_UNLCK, 1)
            self._lock_stream.close()
            self._lock_stream = None

    def _event(self, name: str, **fields: Any) -> None:
        _append_jsonl(self.events_path, {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": name, **fields,
        })

    def _api_snapshot(self) -> dict[str, Any]:
        runs_payload = self.api.get("/train/runs")
        runs = runs_payload.get("runs", []) if isinstance(runs_payload, dict) else []
        history = self.api.get("/api/evaluations/history")
        replay = self.api.get("/api/replay/status")
        health = self.api.get("/health")
        training_runs = [row for row in runs if isinstance(row, dict) and row.get("kind", "training") == "training"]
        telemetry = {}
        for row in training_runs:
            if row.get("state") in ACTIVE_STATES and str(row.get("display_name", "")).startswith("Campaign"):
                telemetry[str(row.get("run_id"))] = self.api.get(f"/train/runs/{row.get('run_id')}/telemetry")
        return {
            "runs": training_runs,
            "telemetry": telemetry,
            "evaluation_history": history if isinstance(history, list) else [],
            "replay": replay if isinstance(replay, dict) else {},
            "health": health if isinstance(health, dict) else {},
        }

    def _active_training(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        return [row for row in snapshot["runs"] if row.get("state") in ACTIVE_STATES]

    @staticmethod
    def _active_replay(snapshot: dict[str, Any]) -> bool:
        if snapshot.get("replay", {}).get("state") in {"queued", "running", "stopping"}:
            return True
        return any(
            isinstance(row, dict) and row.get("state") in {"queued", "running", "stopping"}
            for row in snapshot.get("evaluation_history", [])
        )

    def _report_evidence(
        self, snapshot: dict[str, Any], state: dict[str, Any],
        verification_context: PPOVerificationContext,
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        run_index = {str(row.get("run_id")): row for row in snapshot["runs"]
                     if isinstance(row, dict) and row.get("run_id")}
        campaign_run_ids = {
            str(row.get("run_id")) for row in snapshot["runs"]
            if row.get("run_id") and str(row.get("display_name", "")).startswith("Campaign")
            and _campaign_run_reward_is_eligible(
                row.get("config"), run_id=str(row.get("run_id")), run_index=run_index,
            )
        }
        for record in snapshot["evaluation_history"]:
            if not isinstance(record, dict) or record.get("validation_mode") != "training_probe":
                continue
            if str(record.get("parent_run_id")) not in campaign_run_ids:
                continue
            path = _report_path(record, self.repo_root)
            model = _resolve_model_path(record.get("model_path"), self.repo_root)
            if path is None or model is None or not path.is_file() or not model.is_file():
                evidence.append({"evaluation_id": record.get("evaluation_id"), "classification": "infrastructure_or_missing_evidence"})
                continue
            report = verification_context.read_json(path)
            if report is None:
                evidence.append({"evaluation_id": record.get("evaluation_id"), "classification": "infrastructure_or_missing_evidence"})
                continue
            digest = verification_context.hash_file(model)
            verified = verify_ppo_lineage_checkpoint(model, verification_context=verification_context)
            probe = validate_training_probe(
                record, report, digest, report_path=path, repo_root=self.repo_root,
            ) if verified else None
            trend_validation = None
            parent = run_index.get(str(record.get("parent_run_id")))
            parent_config = parent.get("config") if isinstance(parent, dict) else None
            resume_path = _resolve_model_path(
                parent_config.get("resume") if isinstance(parent_config, dict) else None,
                self.repo_root,
            )
            parent_lineage = (verify_ppo_lineage_checkpoint(resume_path, verification_context=verification_context)
                              if resume_path and resume_path.is_file() else None)
            parent_run_id = str(record.get("parent_run_id"))
            verified_lineage = verified.get("lineage") if isinstance(verified, dict) else None
            parent_lineage_record = parent_lineage.get("lineage") if isinstance(parent_lineage, dict) else None
            child_parent_matches = bool(
                isinstance(verified, dict) and isinstance(parent_lineage, dict)
                and verified.get("parent_checkpoint_sha256") == parent_lineage.get("checkpoint_sha256")
                and isinstance(verified_lineage, dict)
                and verified_lineage.get("run_id") == parent_run_id
                and isinstance(parent_lineage_record, dict)
                and parent_lineage_record.get("run_id") == _resume_parent_run_id(parent_config)
            )
            reward_branch_hash = (
                _campaign_reward_ancestry_link(
                    state, run_id=parent_run_id, run_index=run_index,
                    repo_root=self.repo_root, verification_context=verification_context,
                    checkpoint=model, verified_checkpoint=verified,
                )
                if verified and child_parent_matches else None
            )
            result = record.get("result") if isinstance(record.get("result"), dict) else {}
            service_runs = result.get("runs") if isinstance(result.get("runs"), list) else record.get("attempts")
            report_runs = report.get("runs")
            if (verified and parent_lineage and isinstance(parent_config, dict)
                    and isinstance(service_runs, list) and len(service_runs) == 1
                    and isinstance(service_runs[0], dict)
                    and isinstance(report_runs, list) and len(report_runs) == 1
                    and isinstance(report_runs[0], dict)):
                service_attempt, report_attempt = service_runs[0], report_runs[0]
                try:
                    parent_step = int(parent_lineage.get("num_timesteps", -1))
                    trigger_step = int(record.get("trigger_step", record.get("snapshot_timesteps", -1)))
                    actual_snapshot_step = int(verified.get("num_timesteps", -1))
                    rollout_size = int(parent_config.get("n_steps", 2048))
                    interval = int(parent_config.get("automatic_evaluation_interval", 0))
                    audits_path = self.repo_root / "logs" / "rl" / f"official_run_{record.get('parent_run_id')}" / "ppo_optimizer_audit.jsonl"
                    audit_rows = [json.loads(line) for line in audits_path.read_text(encoding="utf-8").splitlines()
                                  if line.strip()]
                    rollout_steps = _valid_completed_rollout_steps(
                        audit_rows, parent_step=parent_step, trigger_step=actual_snapshot_step,
                    )
                    rollout_count = len(rollout_steps)
                    snapshot_matches_trigger = _checkpoint_matches_trigger_bucket(
                        verified, trigger_step, rollout_size=rollout_size,
                    ) and actual_snapshot_step >= parent_step + 2 * rollout_size
                    scheduled = bool(
                        parent_config.get("automatic_evaluation_enabled") is True
                        and parent_config.get("automatic_evaluation_trigger") == "steps"
                        and parent_config.get("automatic_evaluation_mode") == "training_probe"
                        and interval > 0 and trigger_step >= 0
                        and trigger_step // interval > parent_step // interval
                    )
                    collision_count = service_attempt.get("collision_count")
                    if collision_count is None:
                        collision_count = service_attempt.get("race_collisions")
                    camera_count = service_attempt.get("camera_sample_count")
                    camera_fresh = service_attempt.get("camera_fresh_sample_count")
                    report_camera_count = report_attempt.get("camera_sample_count")
                    sensor_valid = bool(
                        report_attempt.get("frontier_projection_valid") is True
                        and isinstance(camera_count, int) and not isinstance(camera_count, bool) and camera_count > 0
                        and _is_exact_int(camera_fresh, camera_count)
                        and _is_exact_int(service_attempt.get("camera_stale_sample_count"), 0)
                        and _is_exact_int(service_attempt.get("camera_missing_sample_count"), 0)
                        and _is_exact_int(service_attempt.get("camera_freshness_unknown_count"), 0)
                        and isinstance(report_camera_count, int) and not isinstance(report_camera_count, bool)
                        and report_camera_count > 0
                        and _is_exact_int(report_attempt.get("camera_fresh_sample_count"), report_camera_count)
                        and _is_exact_int(report_attempt.get("camera_stale_sample_count"), 0)
                        and _is_exact_int(report_attempt.get("camera_missing_sample_count"), 0)
                        and _is_exact_int(report_attempt.get("camera_freshness_unknown_count"), 0)
                    )
                    report_hash_matches = (
                        report.get("model_sha256") == digest
                        and record.get("model_sha256") == digest
                        and report.get("protocol") == "official_training_probe_1_plus_3"
                        and report.get("validation_mode") == "training_probe"
                        and record.get("validation_mode") == "training_probe"
                        and service_attempt.get("run_id")
                        and isinstance(service_attempt.get("evaluation_report"), str)
                        and Path(service_attempt["evaluation_report"]).resolve()
                        == (self.repo_root / "logs" / "rl" / f"official_run_{service_attempt['run_id']}" / "evaluation.json").resolve()
                        and _is_exact_int(service_attempt.get("race_laps_required"), 3)
                        and _is_exact_int(report_attempt.get("race_laps_required"), 3)
                        and _is_exact_int(service_attempt.get("warmup_laps"), 1)
                    )
                    pass_outcome = probe is not None
                    collision_outcome = (
                        isinstance(collision_count, int) and not isinstance(collision_count, bool)
                        and collision_count > 0
                        and report_attempt.get("stop_reason") in {"collision_detected", "collision_disqualification_limit"}
                        and _is_exact_int(report_attempt.get("collision_count"), collision_count)
                    )
                    if (child_parent_matches and reward_branch_hash is not None
                            and report_hash_matches and snapshot_matches_trigger and sensor_valid
                            and scheduled and rollout_count >= 2
                            and (pass_outcome or collision_outcome)):
                        trend_validation = {
                            "authenticated": True,
                            "scheduled_probe": True,
                            "completed_rollouts_since_branch": rollout_count,
                            "protocol_valid": True,
                            "sensor_valid": True,
                            "loaded_model_sha256": digest,
                            "outcome": "pass" if pass_outcome else "collision",
                            "collision_count": 0 if pass_outcome else int(collision_count),
                            "median_scored_lap_s": probe.get("median_scored_lap_s") if probe else None,
                            "timestamp_utc": record.get("updated_utc") or record.get("created_utc"),
                        }
                except (OSError, ValueError, TypeError, OverflowError):
                    trend_validation = None
            failure_kind = None
            if probe is None:
                attempts = ((record.get("result") or {}).get("runs")
                            if isinstance(record.get("result"), dict) else None)
                attempt = attempts[0] if isinstance(attempts, list) and attempts and isinstance(attempts[0], dict) else {}
                reason = " ".join(str(value or "") for value in (
                    record.get("error"), attempt.get("stop_reason"), attempt.get("score_status"),
                )).casefold()
                infra_markers = ("timeout", "sensor", "stale", "transport", "backend", "simulator",
                                 "container", "missing_result", "inference", "launch", "connection")
                failure_kind = "infrastructure_failure" if any(marker in reason for marker in infra_markers) or record.get("terminal_context") else "policy_or_evaluator_failure_requires_review"
            train_request = None
            if isinstance(parent_config, dict):
                train_request = dict(parent_config)
                train_request["resume"] = _model_alias(model, self.repo_root)
            evidence.append({
                "evaluation_id": record.get("evaluation_id"), "parent_run_id": record.get("parent_run_id"),
                "state": record.get("state"), "model_path": str(model), "model_sha256": digest,
                "model_lineage_verified": bool(verified), "report_path": str(path),
                "report_sha256": verification_context.hash_file(path), "probe": probe,
                # Persist the mode as a canonical evidence field as well as
                # preserving the report's original field. The trend helper
                # consumes evidence records (not raw report JSON), so omitting
                # this validated value silently excluded every real probe.
                "validation_mode": record.get("validation_mode"),
                "report_validation_mode": report.get("validation_mode"),
                "report_protocol": report.get("protocol"),
                "trend_validation": trend_validation,
                "train_request": train_request,
                "terminal_context": record.get("terminal_context"),
                "failure_classification": failure_kind,
                "reward_experiment_config_sha256": reward_branch_hash,
            })
        return evidence

    def _boundary_recovery_candidate(
        self, runs: dict[str, dict[str, Any]], state: dict[str, Any],
        verification_context: PPOVerificationContext,
    ) -> Optional[dict[str, Any]]:
        if any(isinstance(event, dict) and event.get("event") == "campaign_experiment_declared"
               and event.get("experiment_id") == "continue_best_boundary_recovery"
               for event in state.get("events", [])):
            return None
        recovery_spec = CONTINUE_BEST_BOUNDARY_RECOVERY_EVIDENCE
        recovery_run = runs.get(str(recovery_spec["run_id"]))
        if not recovery_run or recovery_run.get("state") not in {"stopped", "completed", "complete"}:
            return None
        recovery_run_dir = self.repo_root / "logs" / "rl" / f"official_run_{recovery_spec['run_id']}"
        manager_path = recovery_run_dir / "manager_config.json"
        config_path = recovery_run_dir / "config.json"
        stop_path = recovery_run_dir / "stop_reason.json"
        checkpoint_path = self.repo_root / recovery_spec["checkpoint_path"]
        try:
            manager_hash = verification_context.hash_file(manager_path)
            effective_hash = verification_context.hash_file(config_path)
            manager = verification_context.read_json(manager_path)
            stop = verification_context.read_json(stop_path)
            if manager is None or stop is None:
                return None
            parent_config = manager.get("config")
            checkpoint_verified = verify_ppo_lineage_checkpoint(
                checkpoint_path, verification_context=verification_context,
            )
            checkpoint_hash = verification_context.hash_file(checkpoint_path)
            checkpoint_valid = bool(
                checkpoint_verified
                and checkpoint_verified.get("checkpoint_sha256") == recovery_spec["checkpoint_sha256"]
                and checkpoint_verified.get("num_timesteps") == recovery_spec["global_timestep"]
            )
            if not (
                manager_hash == recovery_spec["manager_config_sha256"]
                and effective_hash == recovery_spec["effective_config_sha256"]
                and checkpoint_hash == recovery_spec["checkpoint_sha256"]
                and checkpoint_valid
                and stop.get("reason") == "operator_stop"
                and stop.get("num_timesteps") == recovery_spec["global_timestep"]
                and isinstance(parent_config, dict)
                and parent_config.get("campaign_experiment_id") == "continue_best"
                and parent_config.get("campaign_experiment_config_sha256")
            ):
                return None
            request = {key: value for key, value in parent_config.items()
                       if key not in {"campaign_experiment_id", "campaign_experiment_phase",
                                      "campaign_experiment_config_sha256"}}
            request["resume"] = _model_alias(checkpoint_path, self.repo_root)
            reward_hash = next((event.get("config_sha256") for event in reversed(state.get("events", []))
                                if isinstance(event, dict)
                                and event.get("event") == "campaign_experiment_declared"
                                and event.get("experiment_id") == "time_cost_3_calibrated_retry"
                                and isinstance(event.get("declared_config"), dict)
                                and math.isclose(float(event["declared_config"].get(
                                    "time_cost_per_simulated_second", -1)),
                                    float(parent_config.get("time_cost_per_simulated_second", -2)),
                                    rel_tol=0.0, abs_tol=1e-15)), None)
            if not isinstance(reward_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", reward_hash):
                return None
            return {
                "boundary_recovery": recovery_spec,
                **recovery_spec,
                "train_request": request,
                "checkpoint_sha256": recovery_spec["checkpoint_sha256"],
                "parent_effective_config": parent_config,
                "parent_effective_config_sha256": campaign_effective_config_sha256(parent_config),
                "reward_experiment_config_sha256": reward_hash,
            }
        except (OSError, ValueError, TypeError, KeyError):
            # A missing or malformed artifact fails closed: no recovery
            # candidate will be declared or posted.
            return None

    def collect_evidence(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        verification_context = PPOVerificationContext()
        state_bytes = self.state_path.read_bytes()
        state_sha256 = hashlib.sha256(state_bytes).hexdigest()
        state = load_campaign_state(self.state_path)
        if state is None or state.get("campaign_id") != CAMPAIGN_ID:
            raise RuntimeError("campaign state is missing or belongs to another campaign")
        state_signature = _campaign_evidence_state_signature(state)
        state_after_load = load_campaign_state(self.state_path)
        if (state_after_load is None
                or _campaign_evidence_state_signature(state_after_load) != state_signature):
            raise RuntimeError("campaign state changed while its evidence snapshot was being loaded")
        clock = campaign_clock_status(state)
        runs = {str(row.get("run_id")): row for row in snapshot["runs"] if row.get("run_id")}
        boundary_recovery = self._boundary_recovery_candidate(runs, state, verification_context)
        # The exact one-time recovery takes priority over historical trend
        # rescans; those rescans can reread gigabytes of old checkpoints.
        probes = [] if boundary_recovery else self._report_evidence(snapshot, state, verification_context)
        # A strictly clean 1+3 report is useful only after the checkpoint is
        # proven to come from at least two completed updates past its parent.
        # In particular, startup-seed probes must not become parent choices.
        passing = [
            row for row in probes
            if row.get("probe")
            and isinstance(row.get("trend_validation"), dict)
            and row["trend_validation"].get("outcome") == "pass"
            and isinstance(row["trend_validation"].get("completed_rollouts_since_branch"), int)
            and not isinstance(row["trend_validation"].get("completed_rollouts_since_branch"), bool)
            and row["trend_validation"]["completed_rollouts_since_branch"] >= 2
        ]
        candidates: list[dict[str, Any]] = []
        for probe in passing:
            parent = runs.get(str(probe.get("parent_run_id")))
            config = parent.get("config") if parent else None
            if not isinstance(config, dict):
                # The parent may have been pruned from API history; its durable
                # manager manifest remains under logs/rl/official_run_<id>.
                config_path = self.repo_root / "logs" / "rl" / f"official_run_{probe['parent_run_id']}" / "manager_config.json"
                try:
                    manager = json.loads(config_path.read_text(encoding="utf-8"))
                    config = manager.get("config")
                except (OSError, ValueError):
                    config = None
            if not isinstance(config, dict):
                continue
            normalized = dict(config)
            normalized["resume"] = _model_alias(Path(probe["model_path"]), self.repo_root)
            reward_branch_hash = probe.get("reward_experiment_config_sha256")
            if not isinstance(reward_branch_hash, str):
                # A nonzero reward branch without an exact append-only
                # declaration must never be selected as a trend parent.
                continue
            candidates.append({**probe, "train_request": normalized, "effective_config_sha256": campaign_effective_config_sha256(normalized)})
        # Deterministic measured selection: fastest median among strictly
        # passing, authenticated probes; ties are broken by evaluation id.
        candidates.sort(key=lambda row: (row["probe"]["median_scored_lap_s"], str(row.get("evaluation_id"))))
        observed: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": state["campaign_id"], "clock": clock,
            "active_training_ids": [row.get("run_id") for row in self._active_training(snapshot)],
            "replay_state": snapshot.get("replay", {}).get("state"),
            "probe_evidence": probes,
            "passing_candidate_count": len(candidates),
        }
        if boundary_recovery:
            observed["boundary_recovery_candidate"] = boundary_recovery
        declared_ids = {
            str(row.get("experiment_id")) for row in state.get("events", [])
            if isinstance(row, dict) and row.get("event") in {
                "campaign_experiment_declared", "campaign_experiment_invalidated",
            }
        }
        observed.update(derive_campaign_trend_evidence(
            probes, declared_experiment_ids=declared_ids,
        ))
        diagnostics = []
        active_rows = self._active_training(snapshot)
        for run in active_rows:
            if not str(run.get("display_name", "")).startswith("Campaign"):
                continue
            run_id = str(run.get("run_id"))
            raw_dir = run.get("output_dir")
            run_dir = Path(raw_dir) if isinstance(raw_dir, str) and raw_dir else (
                self.repo_root / "logs" / "rl" / f"official_run_{run_id}"
            )
            audit_path = run_dir / "ppo_optimizer_audit.jsonl"
            audit_rows = []
            audit_log_sha256 = verification_context.hash_file(audit_path) if audit_path.is_file() else None
            try:
                raw_lines = audit_path.read_bytes().splitlines()
                for line in raw_lines[-3:]:
                    if line.strip():
                        value = json.loads(line)
                        if isinstance(value, dict):
                            audit_rows.append(value)
            except (OSError, ValueError):
                raw_lines = []
            latest_audit = audit_rows[-1] if audit_rows else {}
            telemetry = snapshot.get("telemetry", {}).get(run_id, {})
            fleet = telemetry.get("fleet") if isinstance(telemetry, dict) else None
            metrics = telemetry.get("last_metrics") if isinstance(telemetry, dict) else None
            telemetry_ts = (metrics or fleet or {}).get("ts") if isinstance(metrics or fleet, dict) else None
            telemetry_age_s = None
            telemetry_age_s = _timestamp_age_s(telemetry_ts)
            local_liveness = _local_run_liveness(run_dir, run_id)
            run_age_s = _timestamp_age_s(run.get("started_at") or run.get("created_at"))
            checkpoint_candidates = []
            checkpoints_dir = run_dir / "checkpoints"
            if checkpoints_dir.is_dir():
                for checkpoint in checkpoints_dir.glob("post_optimizer_update_*_steps_*.zip"):
                    match = re.search(r"_steps_(\d+)\.zip$", checkpoint.name)
                    if match:
                        checkpoint_candidates.append((int(match.group(1)), checkpoint))
            latest_checkpoint = max(checkpoint_candidates, default=(None, None), key=lambda row: row[0])
            raw_car = ((fleet or {}).get("cars") or [None])[0] if isinstance(fleet, dict) else None
            if isinstance(raw_car, dict):
                car = {key: raw_car.get(key) for key in (
                    "env_id", "pose", "yaw", "speed", "collision_count", "frontier_progress_m",
                    "frontier_projection_valid", "frontier_lateral_offset_m", "throttle_command",
                    "steering_command",
                )}
            else:
                car = None
            diag = {
                "run_id": run_id, "run_state": run.get("state"),
                "effective_config_sha256": campaign_effective_config_sha256(run.get("config", {}))
                    if isinstance(run.get("config"), dict) else None,
                "telemetry": {
                    "ts": telemetry_ts, "age_s": telemetry_age_s,
                    "step": (fleet or {}).get("step") if isinstance(fleet, dict) else None,
                    "episode": (fleet or {}).get("episode") if isinstance(fleet, dict) else None,
                    "car": car,
                    "api_delivery_lag": (telemetry_age_s is not None and telemetry_age_s > 45.0
                                         and local_liveness.get("watch") is not None
                                         and local_liveness["watch"]["age_s"] <= 45.0),
                    "local_liveness": local_liveness,
                    "authoritative_source": (local_liveness.get("watch") or {}).get("source"),
                    "authoritative_age_s": (local_liveness.get("watch") or {}).get("age_s"),
                    "authoritative_step": (local_liveness.get("watch") or {}).get("step"),
                    "watch_valid_sensor_frame": (local_liveness.get("watch") or {}).get("valid_sensor_frame"),
                    "watch_reset_in_progress": (local_liveness.get("watch") or {}).get("reset_in_progress"),
                },
                "run_age_s": run_age_s,
                "audit_path": str(audit_path),
                "audit_log_sha256": audit_log_sha256,
                "last_three_audits": audit_rows,
                "last_three_audit_record_sha256": [hashlib.sha256(line).hexdigest() for line in raw_lines[-3:] if line.strip()],
                "latest_saved_checkpoint_hint": str(latest_checkpoint[1]) if latest_checkpoint[1] else None,
                "latest_saved_checkpoint_step_hint": latest_checkpoint[0],
            }
            diagnostics.append(diag)
        observed["active_run_diagnostics"] = diagnostics
        if diagnostics:
            diag = diagnostics[0]
            audits = diag.get("last_three_audits") or [{}]
            observed["nonfinite_update"] = any(
                audit.get(flag) is False for audit in audits
                for flag in ("metrics_finite", "gradients_finite", "parameters_finite", "optimizer_state_finite")
            )
            observed["actor_fault"] = any(audit.get("parameters_finite") is False for audit in audits)
            observed["sensor_fault"] = _sensor_fault_from_freshness(
                diag.get("telemetry", {}).get("local_liveness", {}),
                replay_active=self._active_replay(snapshot),
                run_state=diag.get("run_state"), run_age_s=diag.get("run_age_s"),
            )
            observed["telemetry_delivery_lag"] = bool(diag.get("telemetry", {}).get("api_delivery_lag"))
        if candidates:
            best = candidates[0]
            observed.update({
                "selected_passing_train_request": best["train_request"],
                "selected_passing_checkpoint_sha256": best["model_sha256"],
                "selected_passing_candidate_evidence": {
                    "selected_passing_candidate": True,
                    "selected_passing_checkpoint_sha256": best["model_sha256"],
                    "parent_effective_config": best["train_request"],
                    "parent_effective_config_sha256": best["effective_config_sha256"],
                    **({"reward_experiment_config_sha256": best["reward_experiment_config_sha256"]}
                       if best.get("reward_experiment_config_sha256") else {}),
                },
            })
        controls = [row for row in candidates if abs(float(row["train_request"].get("time_cost_per_simulated_second", 0) or 0)) <= 1e-12]
        if controls:
            control = min(controls, key=lambda row: (row["probe"]["median_scored_lap_s"], str(row.get("evaluation_id"))))
            observed.update({
                "baseline_control_parent_sha256": control["model_sha256"],
                "baseline_control_probe_passed": True,
                "baseline_control_probe_report_sha256": control["report_sha256"],
                "baseline_control_median_scored_lap_s": control["probe"]["median_scored_lap_s"],
                "base_train_request": control["train_request"],
            })
        state_after_collection = load_campaign_state(self.state_path)
        if (state_after_collection is None
                or _campaign_evidence_state_signature(state_after_collection) != state_signature):
            raise RuntimeError("campaign decision state changed during evidence collection; pass discarded")
        if not verification_context.seal():
            raise RuntimeError("campaign evidence dependency changed during collection; pass discarded")
        observed["verification_context_stats"] = verification_context.stats
        observed["campaign_state_sha256"] = state_sha256
        return observed

    def _idle(self, snapshot: dict[str, Any]) -> bool:
        return not self._active_training(snapshot) and not self._active_replay(snapshot)

    def _stop_run(self, run_id: str, reason: str, snapshot: dict[str, Any]) -> None:
        replay = snapshot.get("replay", {})
        if replay.get("state") in {"queued", "running", "stopping"}:
            parent = replay.get("parent_run_id")
            if parent in {None, run_id}:
                self.api.post("/api/replay/stop")
                self._event("replay_stop_requested", run_id=run_id, evaluation_id=replay.get("evaluation_id"), reason=reason)
        self.api.post(f"/train/runs/{run_id}/stop")
        self._event("training_stop_requested", run_id=run_id, reason=reason)

    def _candidate_is_declared(self, state: dict[str, Any], experiment_id: str) -> bool:
        return any(
            isinstance(row, dict) and row.get("event") == "campaign_experiment_declared"
            and row.get("experiment_id") == experiment_id
            for row in state.get("events", [])
        )

    def _candidate_parent_authenticated(self, candidate: dict[str, Any]) -> bool:
        config = candidate.get("request_settings")
        path = _resolve_model_path(config.get("resume") if isinstance(config, dict) else None, self.repo_root)
        if path is None:
            return False
        context = PPOVerificationContext()
        verified = verify_ppo_lineage_checkpoint(path, verification_context=context)
        sealed = context.seal()
        return bool(sealed and verified and verified.get("checkpoint_sha256") == candidate.get("parent_checkpoint_sha256"))

    def _unresolved_candidate_intent(self, runs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        events = self._runner_events()
        launched = {row.get("experiment_id") for row in events if row.get("event") == "candidate_launched"}
        resolved_intent_ids = {row.get("intent_id") for row in events
                               if row.get("event") == "candidate_launch_aborted"
                               and row.get("stage") == "declaration" and row.get("intent_id")}
        resolved_legacy_intents = {
            (row.get("experiment_id"), row.get("config_sha256"), row.get("intent_timestamp_utc"))
            for row in events if row.get("event") == "candidate_launch_aborted"
            and row.get("stage") == "declaration" and not row.get("intent_id")
        }
        for intent in events:
            if (intent.get("event") != "candidate_launch_intent"
                    or intent.get("experiment_id") in launched
                    or (intent.get("intent_id") and intent.get("intent_id") in resolved_intent_ids)
                    or ((intent.get("experiment_id"), intent.get("config_sha256"), intent.get("timestamp_utc"))
                        in resolved_legacy_intents)):
                continue
            experiment_id = str(intent.get("experiment_id", ""))
            matches = []
            for row in runs:
                config = row.get("config") if isinstance(row.get("config"), dict) else {}
                if (
                    config.get("campaign_experiment_id") == experiment_id
                    and config.get("campaign_experiment_phase") == intent.get("phase")
                    and config.get("campaign_experiment_config_sha256") == intent.get("config_sha256")
                    and config.get("resume") == intent.get("resume_path")
                ):
                    matches.append(row)
            if len(matches) == 1:
                self._event("candidate_launched", experiment_id=experiment_id,
                            phase=intent.get("phase"), parent_checkpoint_sha256=intent.get("parent_checkpoint_sha256"),
                            config_sha256=intent.get("config_sha256"), status="reconciled_after_restart",
                            run_id=matches[0].get("run_id"))
                return {"experiment_id": experiment_id, "run_id": matches[0].get("run_id"), "reconciled": True}
            return {"experiment_id": experiment_id, "matches": [row.get("run_id") for row in matches],
                    "reconciled": False, "reason": "launch intent has no unique matching API run; refusing duplicate POST"}
        return None

    def _runner_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        try:
            raw = self.events_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"campaign action ledger cannot be read: {exc}") from exc
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                raise RuntimeError(f"campaign action ledger has an empty line at {line_number}")
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(f"campaign action ledger is corrupt at line {line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"campaign action ledger row {line_number} is not an object")
            rows.append(row)
        return rows

    def _finalist_manifest_rows(self) -> tuple[str, list[dict[str, Any]]]:
        verification_context = PPOVerificationContext()
        digest = verification_context.hash_file(self.finalists_manifest)
        payload = verification_context.read_json(self.finalists_manifest)
        if not isinstance(payload, dict) or payload.get("campaign_id") != CAMPAIGN_ID:
            raise ValueError("finalist manifest campaign_id mismatch")
        state = load_campaign_state(self.state_path)
        if state is None:
            raise ValueError("finalist manifest requires the active campaign state")
        if isinstance(state.get("deadline_policy_amendment"), dict):
            request = campaign_validation_request(state)
            if (request is None
                    or payload.get("validation_request_id") != request.get("request_id")
                    or payload.get("readiness_candidate_checkpoint_sha256")
                    != request.get("candidate_checkpoint_sha256")):
                raise ValueError("finalist manifest is not bound to the active readiness request")
        rows = payload.get("finalists")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
            raise ValueError("finalist manifest must contain one or two explicit finalists")
        clean: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each finalist entry must be an object")
            path = _resolve_model_path(row.get("model_path"), self.repo_root)
            expected = row.get("checkpoint_sha256")
            if path is None or not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("finalist needs an in-logs checkpoint path and SHA-256")
            actual = verification_context.hash_file(path)
            verified = verify_ppo_lineage_checkpoint(path, verification_context=verification_context)
            if actual != expected or not verified or verified.get("checkpoint_sha256") != expected:
                raise ValueError(f"finalist failed independent authenticated PPO lineage: {path}")
            if expected in seen:
                raise ValueError("finalist checkpoint hashes must be unique")
            seen.add(expected)
            clean.append({**row, "model_path": str(path), "checkpoint_sha256": expected})
        if not verification_context.seal():
            raise RuntimeError("finalist manifest or checkpoint lineage changed during verification")
        return digest, clean

    def _create_finalist_manifest_if_ready(
        self, state: dict[str, Any], snapshot: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Publish the immutable shortlist once all training/evaluation work is idle."""
        if self.finalists_manifest.is_file():
            return None
        if not self._idle(snapshot):
            return {"action": "wait_for_quiescence_before_finalist_manifest"}
        evidence = self.collect_evidence(snapshot)
        candidates = []
        for row in evidence.get("probe_evidence", []):
            if not isinstance(row, dict) or not isinstance(row.get("probe"), dict):
                continue
            parent_id = row.get("parent_run_id")
            report_path = row.get("report_path")
            model_path = row.get("model_path")
            report_sha = row.get("report_sha256")
            model_sha = row.get("model_sha256")
            evaluation_id = row.get("evaluation_id")
            if not all(isinstance(item, str) and item for item in (
                parent_id, report_path, model_path, report_sha, model_sha, evaluation_id,
            )):
                continue
            history_path = self.repo_root / "logs" / "rl" / f"official_run_{parent_id}" / "evaluation_history.jsonl"
            candidates.append({
                "model_path": model_path,
                "checkpoint_sha256": model_sha,
                "evaluation_history_path": str(history_path),
                "evaluation_id": evaluation_id,
                "probe_report_path": report_path,
                "probe_report_sha256": report_sha,
            })
        if not candidates:
            return {"action": "waiting_for_authenticated_passing_probe_candidate"}
        input_evidence = {"campaign_id": state.get("campaign_id"), "candidates": candidates}
        evidence_path = _immutable_json(self.output_dir / "evidence", "finalist_candidates", input_evidence)
        manifest = build_finalist_manifest(state, input_evidence, repo_root=self.repo_root)
        encoded = _json_bytes(manifest)
        self.finalists_manifest.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.finalists_manifest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            # A concurrent or recovered runner must validate the immutable file,
            # never replace it with a second selection.
            return {"action": "finalist_manifest_created_concurrently"}
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256(self.finalists_manifest)
        self._event("finalist_manifest_created", manifest_path=str(self.finalists_manifest),
                    manifest_sha256=digest, evidence_path=str(evidence_path),
                    finalist_hashes=[row.get("checkpoint_sha256") for row in manifest["finalists"]])
        return {"action": "finalist_manifest_created", "manifest_path": str(self.finalists_manifest),
                "manifest_sha256": digest, "evidence_path": str(evidence_path),
                "finalist_count": len(manifest["finalists"])}

    def _pending_final_evaluation(self, snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
        events = self._runner_events()
        completed = {row.get("evaluation_id") for row in events
                     if row.get("event") == "finalist_evaluation_result"}
        requested_hashes = {row.get("checkpoint_sha256") for row in events
                            if row.get("event") == "finalist_evaluation_requested"}
        completed_hashes = {row.get("checkpoint_sha256") for row in events
                            if row.get("event") == "finalist_evaluation_result"}
        intents = [row for row in events if row.get("event") == "finalist_evaluation_intent"
                   and row.get("checkpoint_sha256") not in requested_hashes
                   and row.get("checkpoint_sha256") not in completed_hashes]
        if intents:
            intent = intents[-1]
            status = snapshot.get("replay", {})
            match = next((row for row in snapshot.get("evaluation_history", [])
                          if isinstance(row, dict)
                          and row.get("model_sha256") == intent.get("checkpoint_sha256")
                          and row.get("source") == "manual"
                          and row.get("model_path") == intent.get("model_path")
                          and _not_older_than(row.get("updated_utc"), intent.get("timestamp_utc"))), None)
            if match is not None:
                self._event("finalist_evaluation_requested", evaluation_id=match.get("evaluation_id"),
                            manifest_sha256=intent.get("manifest_sha256"), model_path=intent.get("model_path"),
                            checkpoint_sha256=intent.get("checkpoint_sha256"),
                            lineage_verification="reconciled_after_restart")
            elif status.get("model") == intent.get("model_path") and status.get("evaluation_id"):
                self._event("finalist_evaluation_requested", evaluation_id=status.get("evaluation_id"),
                            manifest_sha256=intent.get("manifest_sha256"), model_path=intent.get("model_path"),
                            checkpoint_sha256=intent.get("checkpoint_sha256"),
                            lineage_verification="reconciled_after_restart")
            else:
                return {"pending": True, "unresolved_intent": True, "request": intent}
        requests = [row for row in events if row.get("event") == "finalist_evaluation_requested"
                    and row.get("evaluation_id") not in completed]
        if not requests:
            return None
        status = snapshot.get("replay", {})
        request = next((row for row in reversed(requests) if row.get("evaluation_id") == status.get("evaluation_id")), None)
        if request is None:
            return None
        if status.get("state") in {"queued", "running", "stopping"}:
            return {"pending": True, "request": request}
        record = next((row for row in snapshot.get("evaluation_history", [])
                       if isinstance(row, dict) and row.get("evaluation_id") == request.get("evaluation_id")), None)
        if record is None:
            if status.get("state") in TERMINAL_STATES:
                result = {"passed": False, "evaluation_id": request.get("evaluation_id"),
                          "requested_model_sha256": request.get("checkpoint_sha256"),
                          "errors": ["terminal replay has no durable evaluation history row"]}
            else:
                return {"pending": True, "request": request}
        else:
            result_obj = record.get("result") if isinstance(record.get("result"), dict) else {}
            attempts = result_obj.get("runs") if isinstance(result_obj.get("runs"), list) else []
            report_paths = []
            for attempt in attempts:
                path = _resolve_model_path(attempt.get("evaluation_report") if isinstance(attempt, dict) else None, self.repo_root)
                if path is not None:
                    report_paths.append(path)
            result = validate_final_evaluation(record, report_paths, expected_sha256=request["checkpoint_sha256"])
        result["manifest_sha256"] = request.get("manifest_sha256")
        result["requested_checkpoint_path"] = request.get("model_path")
        path = _immutable_json(self.output_dir / "final_results", f"result_{request.get('evaluation_id')}", result)
        self._event("finalist_evaluation_result", evaluation_id=request.get("evaluation_id"),
                    manifest_sha256=request.get("manifest_sha256"),
                    checkpoint_sha256=request.get("checkpoint_sha256"), passed=result.get("passed"),
                    artifact_path=str(path), errors=result.get("errors", []))
        return {"pending": False, "result": result, "artifact_path": str(path)}

    def _winning_finalist(self, manifest_sha: str) -> Optional[dict[str, Any]]:
        """Select the fastest passing competitive result for this exact manifest."""
        try:
            _, finalists = self._finalist_manifest_rows()
        except (OSError, ValueError, RuntimeError):
            return None
        allowed = {row["checkpoint_sha256"]: row for row in finalists}
        winners: list[dict[str, Any]] = []
        for event in self._runner_events():
            if (event.get("event") != "finalist_evaluation_result"
                    or event.get("manifest_sha256") != manifest_sha
                    or event.get("passed") is not True
                    or event.get("checkpoint_sha256") not in allowed):
                continue
            artifact = Path(str(event.get("artifact_path", "")))
            if not artifact.is_absolute():
                artifact = self.repo_root / artifact
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            expected = event["checkpoint_sha256"]
            if (payload.get("passed") is not True
                    or payload.get("requested_model_sha256") != expected
                    or payload.get("loaded_model_sha256") != expected
                    or not isinstance(payload.get("attempts"), list)
                    or len(payload["attempts"]) != 3):
                continue
            lap_rows = [row.get("scored_lap_times_s") for row in payload["attempts"]
                        if isinstance(row, dict)]
            if (len(lap_rows) != 3 or any(not isinstance(values, list) or len(values) != 10
                    or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                           or not math.isfinite(float(value)) or value <= 0 for value in values)
                    for values in lap_rows)):
                continue
            all_laps = [float(value) for values in lap_rows for value in values]
            finalist = dict(allowed[expected])
            finalist.update(
                checkpoint_sha256=expected,
                primary_evaluation_id=event.get("evaluation_id"),
                primary_result_timestamp_utc=event.get("timestamp_utc"),
                primary_result_path=str(artifact),
                primary_result_sha256=_sha256(artifact),
                median_competitive_lap_s=statistics.median(all_laps),
                mean_competitive_lap_s=statistics.fmean(all_laps),
                competitive_lap_population_stdev_s=statistics.pstdev(all_laps),
                min_competitive_lap_s=min(all_laps),
                max_competitive_lap_s=max(all_laps),
                scored_lap_count=len(all_laps),
                competitive_attempts=3,
            )
            winners.append(finalist)
        if not winners:
            return None
        return min(winners, key=lambda row: (row["median_competitive_lap_s"], row["checkpoint_sha256"]))

    def _clean_reload_runtime_valid(self, payload: Any, winner: dict[str, Any], state: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        required_truths = (
            "training_idle", "replay_idle", "evaluation_history_idle", "simulators_retired",
            "health_ok", "source_loaded",
        )
        if any(payload.get(name) is not True for name in required_truths):
            return False
        old_pid, new_pid = payload.get("old_hub_pid"), payload.get("new_hub_pid")
        if (not isinstance(old_pid, int) or isinstance(old_pid, bool) or old_pid <= 0
                or not isinstance(new_pid, int) or isinstance(new_pid, bool) or new_pid <= 0
                or old_pid == new_pid):
            return False
        if (payload.get("campaign_id") != state.get("campaign_id")
                or payload.get("winner_checkpoint_sha256") != winner.get("checkpoint_sha256")
                or payload.get("primary_evaluation_id") != winner.get("primary_evaluation_id")
                or payload.get("primary_result_sha256") != winner.get("primary_result_sha256")):
            return False
        for key in ("old_hub_start_identity", "new_hub_start_identity", "hub_commandline",
                    "source_fingerprint", "official_scene_sha256", "route_sha256", "recorded_at_utc"):
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                return False
        if "src.layer4.hub.app" not in payload["hub_commandline"]:
            return False
        if payload.get("official_scene_sha256") != "a5d11e435857e2ead32ec0eb065dbbd011c41fa80d1bf8a187177014e014b674":
            return False
        if payload.get("route_sha256") != "62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb":
            return False
        immutable_clock_fields = (
            "campaign_id", "started_at_utc", "training_freeze_at_utc", "hard_deadline_at_utc",
            "start_monotonic_ns", "training_freeze_monotonic_ns", "hard_deadline_monotonic_ns",
            "boot_id", "gate_evidence_canonical_sha256", "updated_policy_probe_checkpoint_sha256",
        )
        expected_clock = {key: state.get(key) for key in immutable_clock_fields}
        if (not isinstance(payload.get("campaign_state_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", payload["campaign_state_sha256"])
                or payload.get("campaign_clock_identity") != expected_clock):
            return False
        health_identity = payload.get("health_identity")
        if not isinstance(health_identity, dict):
            return False
        if (health_identity.get("hub_pid") != new_pid
                or health_identity.get("hub_start_identity") != payload.get("new_hub_start_identity")
                or health_identity.get("source_fingerprint") != payload.get("source_fingerprint")
                or not isinstance(health_identity.get("source_files_sha256"), dict)
                or not health_identity["source_files_sha256"]):
            return False
        try:
            evidence_at = datetime.fromisoformat(payload["recorded_at_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
            result_at = datetime.fromisoformat(str(winner.get("primary_result_timestamp_utc")).replace("Z", "+00:00")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            return False
        if evidence_at <= result_at:
            return False
        return True

    def _pending_clean_reload_probe(self, snapshot: dict[str, Any], winner: dict[str, Any]) -> Optional[dict[str, Any]]:
        events = self._runner_events()
        winner_sha = winner["checkpoint_sha256"]
        completed = {row.get("evaluation_id") for row in events
                     if row.get("event") == "winner_clean_reload_probe_result"}
        requests = [row for row in events if row.get("event") == "winner_clean_reload_probe_requested"
                    and row.get("winner_checkpoint_sha256") == winner_sha
                    and row.get("evaluation_id") not in completed]
        intents = [row for row in events if row.get("event") == "winner_clean_reload_probe_intent"
                   and row.get("winner_checkpoint_sha256") == winner_sha
                   and not any(req.get("intent_id") == row.get("intent_id") for req in requests)]
        if intents:
            intent = intents[-1]
            status = snapshot.get("replay", {})
            match = next((row for row in snapshot.get("evaluation_history", [])
                          if isinstance(row, dict) and row.get("source") == "manual"
                          and row.get("validation_mode") == "training_probe"
                          and row.get("model_sha256") == winner_sha
                          and _not_older_than(row.get("updated_utc"), intent.get("timestamp_utc"))), None)
            if match:
                self._event("winner_clean_reload_probe_requested", intent_id=intent.get("intent_id"),
                            evaluation_id=match.get("evaluation_id"), winner_checkpoint_sha256=winner_sha,
                            model_path=winner.get("model_path"), clean_runtime_sha256=intent.get("clean_runtime_sha256"),
                            lineage_verification="reconciled_after_restart")
            elif status.get("model") == intent.get("model_path") and status.get("evaluation_id"):
                self._event("winner_clean_reload_probe_requested", intent_id=intent.get("intent_id"),
                            evaluation_id=status.get("evaluation_id"), winner_checkpoint_sha256=winner_sha,
                            model_path=winner.get("model_path"), clean_runtime_sha256=intent.get("clean_runtime_sha256"),
                            lineage_verification="reconciled_after_restart")
            else:
                return {"pending": True, "unresolved_intent": True}
        requests = [row for row in events if row.get("event") == "winner_clean_reload_probe_requested"
                    and row.get("winner_checkpoint_sha256") == winner_sha
                    and row.get("evaluation_id") not in completed]
        status = snapshot.get("replay", {})
        request = next((row for row in reversed(requests) if row.get("evaluation_id") == status.get("evaluation_id")), None)
        if request is None:
            return None
        if status.get("state") in {"queued", "running", "stopping"}:
            return {"pending": True, "request": request}
        record = next((row for row in snapshot.get("evaluation_history", [])
                       if isinstance(row, dict) and row.get("evaluation_id") == request.get("evaluation_id")), None)
        if record is None:
            if status.get("state") not in TERMINAL_STATES:
                return {"pending": True, "request": request}
            result = {"passed": False, "errors": ["terminal clean-reload replay has no durable history row"]}
        else:
            result_obj = record.get("result") if isinstance(record.get("result"), dict) else {}
            report_paths = [_resolve_model_path(row.get("evaluation_report"), self.repo_root)
                            for row in result_obj.get("runs", []) if isinstance(row, dict)]
            report_path = next((path for path in report_paths if path is not None), None)
            try:
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else None
            except (OSError, json.JSONDecodeError):
                report = None
            result = (validate_training_probe(record, report, winner_sha,
                                              report_path=report_path, repo_root=self.repo_root)
                      if isinstance(report, dict) and report_path is not None else None)
            if result is None:
                result = {"passed": False, "errors": ["clean-reload probe failed strict 1+3 validation"]}
        result.update({"evaluation_id": request.get("evaluation_id"),
                       "winner_checkpoint_sha256": winner_sha,
                       "competitive_score": False,
                       "clean_runtime_sha256": request.get("clean_runtime_sha256")})
        path = _immutable_json(self.output_dir / "clean_reload_results", f"result_{request.get('evaluation_id')}", result)
        self._event("winner_clean_reload_probe_result", evaluation_id=request.get("evaluation_id"),
                    winner_checkpoint_sha256=winner_sha, passed=result.get("passed"), artifact_path=str(path),
                    errors=result.get("errors", []))
        return {"pending": False, "result": result, "artifact_path": str(path)}

    def _final_phase_tick(self, state: dict[str, Any], snapshot: dict[str, Any], remaining_s: float) -> dict[str, Any]:
        readiness_requested = (
            isinstance(state.get("deadline_policy_amendment"), dict)
            and campaign_validation_request(state) is not None
        )
        replay = snapshot.get("replay", {})
        pending = self._pending_final_evaluation(snapshot)
        if pending and pending.get("pending"):
            if pending.get("unresolved_intent"):
                return {"action": "finalist_launch_intent_unresolved_no_retry",
                        "checkpoint_sha256": pending["request"].get("checkpoint_sha256")}
            return {"action": "monitor_finalist_evaluation", "evaluation_id": pending["request"].get("evaluation_id")}
        if pending and not pending.get("pending"):
            return {"action": "finalist_result_recorded", "passed": pending["result"].get("passed"),
                    "artifact_path": pending["artifact_path"]}
        active = self._active_training(snapshot)
        if active:
            for row in active:
                if row.get("state") != "stopping":
                    self._stop_run(
                        str(row["run_id"]),
                        "readiness_backed_final_validation" if readiness_requested else "campaign_training_freeze",
                        snapshot,
                    )
            return {"action": "stop_training_at_freeze" if any(row.get("state") != "stopping" for row in active)
                    else "wait_for_training_freeze_stop", "run_ids": [row.get("run_id") for row in active]}
        if replay.get("state") in {"queued", "running", "stopping"}:
            return {"action": "wait_for_existing_evaluation", "evaluation_id": replay.get("evaluation_id")}
        if self._active_replay(snapshot):
            return {"action": "wait_for_evaluation_history_reconciliation"}
        manifest_action = self._create_finalist_manifest_if_ready(state, snapshot)
        if manifest_action is not None:
            return manifest_action
        manifest_sha, finalists = self._finalist_manifest_rows()
        requested = {row.get("checkpoint_sha256") for row in self._runner_events()
                     if row.get("event") in {"finalist_evaluation_intent", "finalist_evaluation_requested", "finalist_evaluation_result"}}
        finalist = next((row for row in finalists if row["checkpoint_sha256"] not in requested), None)
        if finalist is None:
            winner = self._winning_finalist(manifest_sha)
            if winner is None:
                return {"action": "all_explicit_finalists_evaluated_no_passing_winner",
                        "manifest_sha256": manifest_sha}
            clean_events = [row for row in self._runner_events()
                            if row.get("event") in {"winner_clean_reload_probe_intent",
                                                     "winner_clean_reload_probe_requested",
                                                     "winner_clean_reload_probe_result"}
                            and row.get("winner_checkpoint_sha256") == winner["checkpoint_sha256"]]
            pending_clean = self._pending_clean_reload_probe(snapshot, winner)
            if pending_clean and pending_clean.get("pending"):
                return {"action": "winner_clean_reload_probe_intent_unresolved_no_retry"
                        if pending_clean.get("unresolved_intent") else "monitor_winner_clean_reload_probe",
                        "winner_checkpoint_sha256": winner["checkpoint_sha256"],
                        "evaluation_id": (pending_clean.get("request") or {}).get("evaluation_id")}
            if pending_clean and not pending_clean.get("pending"):
                return {"action": "winner_clean_reload_probe_recorded",
                        "passed": pending_clean["result"].get("passed"),
                        "competitive_score": False, "artifact_path": pending_clean["artifact_path"]}
            if clean_events:
                return {"action": "winner_clean_reload_probe_already_attempted_no_retry",
                        "winner_checkpoint_sha256": winner["checkpoint_sha256"]}
            if not readiness_requested and remaining_s < 900:
                return {"action": "winner_clean_reload_window_too_short_fail_closed",
                        "remaining_seconds": remaining_s}
            if not self._idle(snapshot) or self._active_replay(snapshot):
                return {"action": "wait_for_quiescence_before_clean_reload_probe",
                        "winner_checkpoint_sha256": winner["checkpoint_sha256"]}
            try:
                clean_payload = json.loads(self.clean_reload_runtime.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {"action": "await_manual_clean_runtime_reload_evidence",
                        "winner_checkpoint_sha256": winner["checkpoint_sha256"],
                        "required_evidence_path": str(self.clean_reload_runtime)}
            if not self._clean_reload_runtime_valid(clean_payload, winner, state):
                return {"action": "clean_runtime_evidence_mismatch_fail_closed",
                        "winner_checkpoint_sha256": winner["checkpoint_sha256"],
                        "evidence_path": str(self.clean_reload_runtime)}
            try:
                health = self.api.get("/health")
            except Exception as exc:
                return {"action": "clean_reload_hub_health_unavailable",
                        "error": f"{type(exc).__name__}: {exc}"}
            health_fields = ("hub_pid", "hub_start_identity", "source_fingerprint", "source_files_sha256")
            if (not isinstance(health, dict) or health.get("status") not in {"ok", "healthy"}
                    or any(health.get(key) != clean_payload.get("health_identity", {}).get(key)
                           for key in health_fields)
                    or health.get("hub_pid") != clean_payload.get("new_hub_pid")):
                return {"action": "clean_reload_hub_health_invalid", "health": health}
            alias = _model_alias(Path(winner["model_path"]), self.repo_root)
            model_id = alias.removeprefix("/runs/rl/")
            runtime_sha = _sha256(self.clean_reload_runtime)
            intent_id = uuid.uuid4().hex
            self._event("winner_clean_reload_probe_intent", intent_id=intent_id,
                        winner_checkpoint_sha256=winner["checkpoint_sha256"],
                        model_path=winner["model_path"], primary_evaluation_id=winner["primary_evaluation_id"],
                        clean_runtime_path=str(self.clean_reload_runtime), clean_runtime_sha256=runtime_sha,
                        old_hub_pid=clean_payload["old_hub_pid"], new_hub_pid=clean_payload["new_hub_pid"],
                        primary_median_competitive_lap_s=winner["median_competitive_lap_s"])
            response = self.api.post("/api/replay/start", {
                "model_id": model_id, "map_id": "none", "device": "cpu", "trace_steps": 0,
                "validation_mode": "training_probe",
            })
            evaluation_id = response.get("evaluation_id") if isinstance(response, dict) else None
            if not isinstance(evaluation_id, str) or not evaluation_id:
                raise RuntimeError("clean-reload training_probe API returned no evaluation_id")
            self._event("winner_clean_reload_probe_requested", intent_id=intent_id,
                        evaluation_id=evaluation_id, winner_checkpoint_sha256=winner["checkpoint_sha256"],
                        model_path=winner["model_path"], clean_runtime_sha256=runtime_sha,
                        lineage_verification="verify_ppo_lineage_checkpoint",
                        competitive_score=False)
            return {"action": "winner_clean_reload_probe_started", "evaluation_id": evaluation_id,
                    "winner_checkpoint_sha256": winner["checkpoint_sha256"],
                    "clean_runtime_sha256": runtime_sha, "competitive_score": False}
        if not readiness_requested and remaining_s < 4200:
            return {"action": "finalist_window_too_short_fail_closed", "remaining_seconds": remaining_s}
        if not state.get("training_freeze_at_utc"):
            raise RuntimeError("campaign state has no immutable training freeze")
        alias = _model_alias(Path(finalist["model_path"]), self.repo_root)
        model_id = alias.removeprefix("/runs/rl/")
        # Verify idle one last time before the normal replay API request.
        check = self.api.get("/api/replay/status")
        if check.get("state") in {"queued", "running", "stopping"}:
            return {"action": "wait_for_replay_idle_before_finalist"}
        self._event("finalist_evaluation_intent", manifest_sha256=manifest_sha,
                    model_path=finalist["model_path"], checkpoint_sha256=finalist["checkpoint_sha256"])
        response = self.api.post("/api/replay/start", {
            "model_id": model_id, "map_id": "none", "device": "cpu", "trace_steps": 0,
        })
        evaluation_id = response.get("evaluation_id") if isinstance(response, dict) else None
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise RuntimeError("standard final replay API returned no evaluation_id")
        self._event("finalist_evaluation_requested", evaluation_id=evaluation_id,
                    manifest_sha256=manifest_sha, model_path=finalist["model_path"],
                    checkpoint_sha256=finalist["checkpoint_sha256"],
                    lineage_verification="verify_ppo_lineage_checkpoint")
        return {"action": "finalist_evaluation_started", "evaluation_id": evaluation_id,
                "checkpoint_sha256": finalist["checkpoint_sha256"], "manifest_sha256": manifest_sha}

    def _launch_candidate(self, candidate: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        experiment_id = str(candidate["experiment_id"])
        if not self._candidate_parent_authenticated(candidate):
            raise RuntimeError(f"candidate {experiment_id} parent failed independent PPO lineage/hash verification")
        request_settings = candidate.get("request_settings")
        if not isinstance(request_settings, dict):
            raise RuntimeError("planner candidate has no complete normalized request settings")
        if experiment_id in TIME_COST_VARIANTS:
            configured_cost = request_settings.get("time_cost_per_simulated_second")
            expected_cost = TIME_COST_VARIANTS[experiment_id]
            if (not isinstance(configured_cost, (int, float)) or isinstance(configured_cost, bool)
                    or not math.isclose(float(configured_cost), expected_cost, rel_tol=0.0, abs_tol=1e-15)):
                raise RuntimeError(
                    f"{experiment_id} reward coefficient mismatch: expected {expected_cost:.15g}, got {configured_cost!r}"
                )
        if (
            request_settings.get("exploration_profile") != "bc_campaign"
            or request_settings.get("n_envs") != 1
            or request_settings.get("checkpoint_every") != 10000
            or request_settings.get("automatic_evaluation_enabled") is not True
            or request_settings.get("automatic_evaluation_mode") != "training_probe"
            or request_settings.get("automatic_evaluation_trigger") != "steps"
            or request_settings.get("automatic_evaluation_interval") != 50000
        ):
            raise RuntimeError("campaign candidate does not preserve single-env global 10k/50k bc_campaign cadence")
        intent = {
            "experiment_id": experiment_id, "phase": candidate["phase"],
            "parent_checkpoint_sha256": candidate["parent_checkpoint_sha256"],
            "config_sha256": candidate["config_sha256"], "resume_path": request_settings.get("resume"),
            "status": "launch_intent", "intent_id": uuid.uuid4().hex,
        }
        self._event("candidate_launch_intent", **intent)
        try:
            declaration = declare_campaign_experiment(
                self.state_path,
                experiment_id=experiment_id,
                phase=str(candidate["phase"]),
                config=dict(candidate["request_settings"]),
                parent_checkpoint_sha256=str(candidate["parent_checkpoint_sha256"]),
                evidence=dict(candidate["declaration_evidence"]),
            )
        except Exception as exc:
            # Declaration validation runs before POST, so this exact failure
            # is safe to resolve and retry after a code/evidence correction.
            self._event("candidate_launch_aborted", **intent,
                        stage="declaration", error=f"{type(exc).__name__}: {exc}")
            raise
        if declaration.get("config_sha256") != candidate.get("config_sha256"):
            raise RuntimeError("planner/declaration effective config hashes differ; run was not posted")
        self._event("candidate_declared", **{**intent, "config_sha256": declaration["config_sha256"]})
        result = self.api.post("/train/runs", {
            "display_name": f"Campaign {experiment_id} parent {candidate['parent_checkpoint_sha256'][:12]}",
            "config": declaration["config"],
        })
        run_id = result.get("run_id") if isinstance(result, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("campaign train POST returned no run_id")
        self._event("candidate_launched", **{**intent, "status": "started", "run_id": run_id})
        return result

    def tick(self) -> dict[str, Any]:
        state = load_campaign_state(self.state_path)
        if state is None or state.get("campaign_id") != CAMPAIGN_ID or state.get("status") != "active":
            raise RuntimeError("phase runner refuses missing, inactive, or mismatched campaign state")
        # Validate the durable intent ledger before API reads or any action.
        # Corruption must never erase a previous POST intent and permit a retry.
        self._runner_events()
        snapshot = self._api_snapshot()
        evidence = self.collect_evidence(snapshot)
        evidence_path = _immutable_json(self.output_dir / "evidence", "snapshot", evidence)
        plan = build_campaign_decision_plan(state, evidence=evidence)
        plan_path = _immutable_json(self.output_dir / "plans", "decision", plan)
        active = self._active_training(snapshot)
        active_replay = snapshot.get("replay", {})
        elapsed_h = float(plan.get("elapsed_hours", 0.0))
        phase = str(plan.get("phase"))
        current_state = load_campaign_state(self.state_path) or state
        unbounded_training = isinstance(current_state.get("deadline_policy_amendment"), dict)
        declared = {
            row.get("experiment_id") for row in current_state.get("events", [])
            if isinstance(row, dict) and row.get("event") == "campaign_experiment_declared"
        }
        candidates = [row for row in plan.get("branch_candidates", [])
                     if isinstance(row, dict) and row.get("experiment_id") not in declared]
        outcome: dict[str, Any] = {
            "phase": phase, "elapsed_hours": elapsed_h,
            "plan_path": str(plan_path), "evidence_path": str(evidence_path),
            "active_training_ids": [row.get("run_id") for row in active],
            "replay_state": active_replay.get("state"), "action": "observe_only",
        }

        if plan.get("faults") and active:
            for row in active:
                if row.get("state") != "stopping":
                    self._stop_run(str(row["run_id"]), "campaign_runtime_fault:" + ",".join(plan["faults"]), snapshot)
            outcome.update(action="stop_training_on_runtime_fault", faults=plan["faults"])
            self._event("tick", **outcome)
            return outcome

        if phase == "final_validation":
            final_result = self._final_phase_tick(
                current_state, snapshot, float(plan.get("seconds_until_hard_deadline") or 0.0),
            )
            outcome.update(final_result)
            self._event("tick", **outcome)
            return outcome
        if phase == "hard_deadline":
            outcome["action"] = "deadline_supervisor_owns_shutdown"
            self._event("tick", **outcome)
            return outcome

        # Finish a same-parent pace screening after its first completed probe,
        # or at the 5h boundary; this never waits past the campaign clock.
        if active and elapsed_h >= 5.0 and not unbounded_training:
            for row in active:
                cfg = row.get("config") or {}
                if row.get("state") != "stopping" and _is_calibrated_pace_branch(cfg):
                    if not _pace_phase_boundary_stop_allowed(elapsed_h, replay_active=self._active_replay(snapshot)):
                        outcome.update(action="wait_for_active_probe_at_phase_boundary", run_id=row["run_id"])
                        self._event("tick", **outcome)
                        return outcome
                    self._stop_run(str(row["run_id"]), "five_hour_pace_phase_boundary", snapshot)
                    outcome.update(action="stop_pace_branch_at_phase_boundary", run_id=row["run_id"])
                    return outcome
        for row in active:
            if phase == "pace_experiments" and _is_calibrated_pace_branch(row.get("config")):
                screening_context = PPOVerificationContext()
                seen_probe = any(
                    item.get("parent_run_id") == row.get("run_id")
                    and _pace_probe_is_screening_complete(
                        item, row, self.repo_root, verification_context=screening_context,
                    )
                    for item in snapshot["evaluation_history"] if isinstance(item, dict)
                )
                if not screening_context.seal():
                    raise RuntimeError("pace screening evidence changed during verification")
                if seen_probe and not self._active_replay(snapshot):
                    if row.get("state") != "stopping":
                        self._stop_run(str(row["run_id"]), "first_scheduled_probe_completed", snapshot)
                        outcome.update(action="stop_pace_branch_after_first_probe", run_id=row["run_id"])
                        return outcome

        # Only launch from a clean runtime boundary. During phase 0–2 the
        # baseline continues untouched. If no passing control exists at 2h,
        # evidence remains TBC and the existing run is allowed to continue.
        if not self._idle(snapshot):
            if phase == "pace_experiments" and candidates and len(active) == 1 and not self._active_replay(snapshot):
                current = active[0]
                current_cost = float((current.get("config") or {}).get("time_cost_per_simulated_second", 0) or 0)
                if current_cost == 0.0:
                    self._stop_run(str(current["run_id"]), "baseline_complete_before_pace_branch", snapshot)
                    outcome.update(action="stop_baseline_before_pace_branch", run_id=current["run_id"])
                    self._event("tick", **outcome)
                    return outcome
            return outcome
        unresolved = self._unresolved_candidate_intent(snapshot["runs"])
        if unresolved is not None:
            outcome.update(action="candidate_launch_reconciliation_required", intent=unresolved)
            self._event("tick", **outcome)
            return outcome
        state = current_state
        if phase == "pace_experiments" and candidates:
            # The baseline must be stopped before the first variant. Do not
            # interrupt an in-progress baseline probe; `_idle` enforces this.
            result = self._launch_candidate(candidates[0], state)
            outcome.update(action="launched_declared_candidate", experiment_id=candidates[0]["experiment_id"], run_id=result.get("run_id"))
        elif phase == "best_validated_trend" and candidates:
            result = self._launch_candidate(candidates[0], state)
            outcome.update(action="launched_declared_candidate", experiment_id=candidates[0]["experiment_id"], run_id=result.get("run_id"))
        self._event("tick", **outcome)
        return outcome

    def run(self, *, once: bool = False) -> None:
        self.acquire()
        try:
            while True:
                try:
                    result = self.tick()
                    print(json.dumps(result, sort_keys=True), flush=True)
                except Exception as exc:
                    self._event("tick_error", error=f"{type(exc).__name__}: {exc}")
                    print(json.dumps({"error": f"{type(exc).__name__}: {exc}",
                                      "traceback": traceback.format_exc()}), flush=True)
                if once:
                    return
                time.sleep(self.poll_seconds)
        finally:
            self.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--hub-url", default="http://127.0.0.1:8090")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--finalists-manifest", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not 1.0 <= args.poll_interval <= 3600.0:
        parser.error("--poll-interval must be in [1, 3600] seconds")
    runner = CampaignPhaseRunner(
        state_path=args.state, repo_root=args.repo_root,
        api=CampaignAPI(args.hub_url), poll_seconds=args.poll_interval,
        finalists_manifest=args.finalists_manifest,
    )
    runner.run(once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
