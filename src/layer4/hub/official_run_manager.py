"""Persistent, isolated manager for official AutoDRIVE training runs."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.layer4.settings import ROOT
from src.layer4.hub import docker_control
from src.layer4.hub.campaign_deadline import (
    assert_campaign_allows,
    assert_campaign_training_config,
    campaign_admission_decision,
    campaign_hub_maintenance_status,
    claim_campaign_hub_maintenance_resume,
    default_state_path,
    load_campaign_state,
)
from src.layer4.official_settings import OfficialTrainSettings, validate_bc_teacher_action_mapping
from src.layer3.bc_artifact import (
    OFFICIAL_BC_SOURCE,
    verify_bc_checkpoint,
    verify_ppo_lineage_checkpoint,
    verify_ppo_warmstart,
)


API_IMAGE = os.environ.get("AICAR_OFFICIAL_API_IMAGE", "aicar-iros2026")
LEGACY_SIM_IMAGE_OVERRIDE = os.environ.get("AICAR_OFFICIAL_SIM_IMAGE")
TRAIN_SIM_IMAGE = os.environ.get(
    "AICAR_OFFICIAL_TRAIN_SIM_IMAGE",
    LEGACY_SIM_IMAGE_OVERRIDE
    or "autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice",
)
EVALUATION_SIM_IMAGE = os.environ.get(
    "AICAR_OFFICIAL_EVALUATION_SIM_IMAGE",
    LEGACY_SIM_IMAGE_OVERRIDE
    or "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete",
)
GPU_API_IMAGE = os.environ.get("AICAR_OFFICIAL_API_GPU_IMAGE", "aicar-iros2026:cuda")
RUNS_ROOT = ROOT / "logs" / "rl"
REGISTRY_PATH = ROOT / "logs" / "layer4" / "official_runs.json"
TRAIN_DEFAULTS: Dict[str, Any] = OfficialTrainSettings().model_dump(mode="python")


def is_exact_campaign_maintenance_resume(
    normalized_config: Dict[str, Any], new_hub_identity: Any
) -> bool:
    """Return true only for the currently authorized one-shot maintenance resume.

    Maintenance may intentionally resume a calibrated experiment config that
    differs from the campaign's frozen baseline plan. The separate maintenance
    admission contract binds its complete normalized config, identity, and
    training contract; this predicate permits that exact request to skip only
    the baseline-plan equality check. The manager still consumes the one-shot
    claim at the locked run-creation boundary.
    """
    if not isinstance(normalized_config, dict) or not isinstance(new_hub_identity, dict):
        return False
    state = load_campaign_state(default_state_path())
    if campaign_hub_maintenance_status(state) is None:
        return False
    allowed, _reason = campaign_admission_decision(
        state,
        "training",
        maintenance_resume={
            "normalized_config": normalized_config,
            "new_hub_identity": new_hub_identity,
        },
    )
    return allowed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stop_reason_from_sentinel(path: Path) -> Optional[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    reason = payload.get("reason") if isinstance(payload, dict) else None
    return str(reason) if reason else None


def _resume_effective_settings(
    resume: Any, *, explicit_fields: set[str]
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Read authenticated parent settings for fields omitted by a resume request."""
    if not isinstance(resume, str) or not resume.strip():
        return {}, None
    logs_root = (ROOT / "logs").resolve()
    resume_value = resume.replace("\\", "/")
    if resume_value.startswith("/runs/"):
        resume_path = logs_root / Path(resume_value[len("/runs/"):])
    elif resume_value.startswith("/app/logs/"):
        resume_path = logs_root / Path(resume_value[len("/app/logs/"):])
    else:
        candidate = Path(resume)
        resume_path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    resume_path = resume_path.resolve()
    try:
        resume_path.relative_to(logs_root)
    except ValueError:
        return {}, None
    lineage = verify_ppo_lineage_checkpoint(resume_path)
    if not lineage:
        return {}, None
    config_path = lineage.get("effective_config")
    if not isinstance(config_path, str):
        return {}, lineage
    try:
        saved = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}, lineage
    if not isinstance(saved, dict):
        return {}, lineage

    inherited: Dict[str, Any] = {}
    ppo = saved.get("ppo") if isinstance(saved.get("ppo"), dict) else {}
    episode = saved.get("episode") if isinstance(saved.get("episode"), dict) else {}
    actions = saved.get("actions") if isinstance(saved.get("actions"), dict) else {}
    exploration = saved.get("ppo_exploration") if isinstance(saved.get("ppo_exploration"), dict) else {}
    field_map = {
        "n_envs": "n_envs",
        "n_steps": "n_steps",
        "learning_rate": "learning_rate",
        "n_epochs": "n_epochs",
        "gamma": "gamma",
        "gae_lambda": "gae_lambda",
        "target_kl": "target_kl",
        "checkpoint_every": "post_optimizer_save_every_transitions",
        "stop_on_plateau": "stop_on_plateau",
        "plateau_min_timesteps": "plateau_min_timesteps",
        "plateau_window_episodes": "plateau_window_episodes",
        "plateau_patience": "plateau_patience",
        "plateau_min_improvement_pct": "plateau_min_improvement_pct",
        "max_duration_hours": "max_duration_hours",
    }
    for public, stored in field_map.items():
        if public not in explicit_fields and stored in ppo:
            inherited[public] = ppo[stored]
    for public, source in (
        ("device", "device"),
        ("seed", "seed"),
        ("automatic_evaluation_enabled", "enabled"),
        ("automatic_evaluation_trigger", "trigger"),
        ("automatic_evaluation_interval", "interval"),
        ("automatic_evaluation_mode", "mode"),
        ("behavior_cloning_enabled", "enabled"),
        ("observation_profile", "observation_profile"),
        ("simulator_backend", "simulator_backend"),
    ):
        if public in explicit_fields:
            continue
        section = (
            saved.get("automatic_evaluation", {}) if public.startswith("automatic_evaluation_")
            else saved.get("behavior_cloning", {}) if public == "behavior_cloning_enabled"
            else saved
        )
        if isinstance(section, dict) and source in section:
            inherited[public] = section[source]
    for public, source in (("training_timeout_s", "training_watchdog_s"), ("frontier_stagnation_s", "frontier_stagnation_s")):
        if public not in explicit_fields and source in episode:
            inherited[public] = episode[source]
    for public in (
        "throttle_mode", "negative_throttle_mode", "steering_mode",
        "steering_action_scale", "straight_throttle_gain", "straight_throttle_steering_threshold",
    ):
        source = "throttle_mode" if public == "throttle_mode" else public
        if public not in explicit_fields and source in actions:
            inherited[public] = actions[source]
    if "exploration_profile" not in explicit_fields and exploration.get("profile"):
        inherited["exploration_profile"] = exploration["profile"]
    for public, list_key, index in (
        ("exploration_initial_std_throttle", "starting_action_std", 0),
        ("exploration_initial_std_steering", "starting_action_std", 1),
        ("exploration_min_std_throttle", "minimum_action_std_by_action", 0),
        ("exploration_min_std_steering", "minimum_action_std_by_action", 1),
        ("exploration_max_std_throttle", "maximum_action_std_by_action", 0),
        ("exploration_max_std_steering", "maximum_action_std_by_action", 1),
    ):
        values = exploration.get(list_key)
        # Mission Control uses null to mean "inherit saved exploration".
        if public not in explicit_fields and isinstance(values, list) and len(values) > index:
            inherited[public] = values[index]
    return inherited, lineage


def normalize_official_config(
    raw: Dict[str, Any], *, explicit_fields: Optional[set[str]] = None
) -> Dict[str, Any]:
    """Validate settings, inheriting authenticated PPO values omitted on resume."""
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    unknown = set(raw) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported official training settings: {sorted(unknown)}")
    supplied = set(raw) if explicit_fields is None else set(explicit_fields)
    if not supplied.issubset(set(TRAIN_DEFAULTS)):
        raise ValueError("explicit_fields contains unsupported official training settings")
    inheritable_nulls = {
        "exploration_profile", "exploration_initial_std_throttle", "exploration_initial_std_steering",
        "exploration_min_std_throttle", "exploration_min_std_steering",
        "exploration_max_std_throttle", "exploration_max_std_steering",
    }
    inheritance_exclusions = {
        field for field in supplied if not (field in inheritable_nulls and raw.get(field) is None)
    }
    merged, inherited_lineage = _resume_effective_settings(
        raw.get("resume"), explicit_fields=inheritance_exclusions
    )
    for field in (
        "exploration_profile", "exploration_initial_std_throttle", "exploration_initial_std_steering",
        "exploration_min_std_throttle", "exploration_min_std_steering",
        "exploration_max_std_throttle", "exploration_max_std_steering",
    ):
        if raw.get(field) is None and field in merged:
            # Explicit null remains the UI's documented "inherit" value for exploration.
            continue
        if field in raw:
            merged[field] = raw[field]
    for field, value in raw.items():
        if field not in {
            "exploration_profile", "exploration_initial_std_throttle", "exploration_initial_std_steering",
            "exploration_min_std_throttle", "exploration_min_std_steering",
            "exploration_max_std_throttle", "exploration_max_std_steering",
        }:
            merged[field] = value
    result = OfficialTrainSettings.model_validate(merged).model_dump(mode="python")
    validate_bc_teacher_action_mapping(
        behavior_cloning_enabled=result["behavior_cloning_enabled"],
        resume=result["resume"],
        throttle_mode=result["throttle_mode"],
        negative_throttle_mode=result["negative_throttle_mode"],
        steering_mode=result["steering_mode"],
        steering_action_scale=result["steering_action_scale"],
        straight_throttle_gain=result["straight_throttle_gain"],
    )
    if result["behavior_cloning_only"] and not result["behavior_cloning_enabled"]:
        raise ValueError("BC-only runs require official behavior cloning to be enabled")
    if result["behavior_cloning_only"] and result["resume"] is not None:
        raise ValueError("BC-only runs must start from a new policy, not a resumed checkpoint")
    if result["behavior_cloning_only"] and result["automatic_evaluation_enabled"]:
        raise ValueError("disable automatic PPO evaluation for a BC-only run; evaluate its saved checkpoint from Replay")
    if result["automatic_evaluation_mode"] == "training_probe" and not result["automatic_evaluation_enabled"]:
        raise ValueError("training_probe mode requires automatic evaluation to be enabled")
    if result["simulator_backend"] == "windows_external":
        # Keep API-side campaign hashes identical to the settings ultimately
        # launched by the manager. The native Windows GUI needs extra startup
        # time; this was historically applied only inside create_run().
        result["timeout_s"] = max(180.0, float(result["timeout_s"]))
    resume = result["resume"]
    if resume is not None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("resume must be a non-empty checkpoint path or null")
    if resume is not None:
        logs_root = (ROOT / "logs").resolve()
        resume_value = resume.replace("\\", "/")
        if resume_value.startswith("/runs/"):
            relative = Path(resume_value[len("/runs/"):])
            resume_path = logs_root / relative
        elif resume_value.startswith("/app/logs/"):
            relative = Path(resume_value[len("/app/logs/"):])
            resume_path = logs_root / relative
        else:
            candidate = Path(resume)
            resume_path = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
        resume_path = resume_path.resolve()
        try:
            relative = resume_path.relative_to(logs_root)
        except ValueError as exc:
            raise ValueError("resume checkpoint must be inside the persistent logs directory") from exc
        if not resume_path.is_file():
            raise ValueError(f"resume checkpoint does not exist: {resume_path}")
        lineage_path = resume_path.with_name(f"{resume_path.stem}.lineage.json")
        verified_lineage = None
        if lineage_path.is_file():
            verified_lineage = inherited_lineage or verify_ppo_lineage_checkpoint(resume_path)
            if not verified_lineage:
                raise ValueError("resume checkpoint has invalid or tampered PPO lineage")
        if result["automatic_evaluation_mode"] == "training_probe":
            verified_parent = verify_bc_checkpoint(resume_path)
            if not verified_parent:
                verified_parent = verify_bc_checkpoint(
                    resume_path,
                    expected_sha256=OFFICIAL_BC_SOURCE["checkpoint_sha256"],
                    allow_legacy=True,
                )
            if not verified_parent:
                verified_parent = verify_ppo_warmstart(resume_path)
            if not verified_parent:
                verified_parent = verified_lineage or verify_ppo_lineage_checkpoint(resume_path)
            if not verified_parent:
                raise ValueError(
                    "training_probe mode requires a resume checkpoint authenticated as BC, a verified BC-derived warm start, or a verified PPO descendant"
                )
        result["resume"] = f"/runs/{relative.as_posix()}"
    elif result["automatic_evaluation_mode"] == "training_probe":
        raise ValueError("training_probe mode requires an authenticated BC or BC-derived resume checkpoint")
    return result


def _safe_display_name(value: Optional[str]) -> str:
    if value is not None and not isinstance(value, str):
        raise ValueError("display_name must be text")
    name = (value or "official-training").strip()
    if not name or len(name) > 100:
        raise ValueError("display_name must contain 1 to 100 characters")
    return name


def _official_simulator_command(api_alias: str) -> str:
    """Launch the official player headlessly while keeping camera rendering on.

    AutoDRIVE's camera-capable Linux headless mode uses Xvfb. ``-nographics``
    explicitly disables all rendering, which leaves the required camera input
    unavailable to the official camera observation profile.
    """
    if api_alias not in {"api", *(f"api-{index}" for index in range(1, 8))}:
        raise ValueError("invalid official API network alias")
    return (
        'xvfb-run -a -s "-screen 0 1280x720x24" '
        f'"./AutoDRIVE Simulator.x86_64" -ip {api_alias} -port 4568'
    )


class OfficialRunManager:
    """Manage independent official API+simulator container pairs.

    Docker is injected as a module-like backend for tests. The hub's logs bind
    mount is resolved to its Docker-host path before a run directory is mounted
    into the official API image, making checkpoints survive hub/container restarts.
    """

    def __init__(
        self,
        *,
        max_concurrent_runs: Optional[int] = None,
        backend: Any = docker_control,
        registry_path: Path = REGISTRY_PATH,
        runs_root: Path = RUNS_ROOT,
        poll_interval_s: float = 2.0,
        start_monitor: bool = True,
    ) -> None:
        configured = max_concurrent_runs
        if configured is None:
            configured = int(os.environ.get("AICAR_MAX_TRAIN_JOBS", "2"))
        if configured < 1 or configured > 32:
            raise ValueError("AICAR_MAX_TRAIN_JOBS must be between 1 and 32")
        self.max_concurrent_runs = int(configured)
        self.backend = backend
        self.registry_path = Path(registry_path)
        self.runs_root = Path(runs_root)
        self.poll_interval_s = max(0.25, float(poll_interval_s))
        self._lock = threading.RLock()
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._creating: set[str] = set()
        self._stopping: set[str] = set()
        self._evaluation_paused: set[str] = set()
        self._backend_owner = getattr(self.backend, "is_owner", True)
        if self._backend_owner:
            self._load_registry()
            self.reconcile()
        self._monitor_stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None
        if start_monitor and self._backend_owner:
            self._monitor = threading.Thread(
                target=self._monitor_loop, name="official-run-manager", daemon=True
            )
            self._monitor.start()

    def close(self) -> None:
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=2.0)
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()

    def _load_registry(self) -> None:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            rows = data.get("runs", []) if isinstance(data, dict) else []
            for row in rows:
                if isinstance(row, dict) and row.get("run_id"):
                    # Docker-era registries stored mount paths such as
                    # /app/logs/rl/... . Resolve those records back to this
                    # checkout's persistent logs so native-hub telemetry and
                    # camera endpoints remain readable after migration.
                    raw_output = str(row.get("output_dir") or "")
                    for prefix in ("/app/logs/rl/", "/runs/rl/"):
                        if raw_output.startswith(prefix):
                            row["output_dir"] = str(self.runs_root / raw_output[len(prefix):])
                            if row.get("log_path"):
                                row["log_path"] = str(Path(row["output_dir"]) / "hub_train.log")
                            break
                    self._runs[str(row["run_id"])] = row
        except (OSError, ValueError, TypeError):
            self._runs = {}

    def _persist(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Fleet frames and current metrics are hot, potentially large values;
        # keep them out of the lifecycle registry. Fleet is separately
        # checkpointed by the official Layer 3 callback in watch_latest.json.
        ephemeral = {
            "latest_fleet", "latest_metrics", "latest_telemetry",
            "training_phase", "last_evaluator_live", "last_evaluation",
            "updated_at",
        }
        persisted_runs = [
            {key: value for key, value in row.items() if key not in ephemeral}
            for row in self._runs.values()
        ]
        # WSL DrvFS can host more than one hub process during a restart. A
        # shared fixed `.tmp` name lets one writer replace another writer's
        # staging file, causing ENOENT at os.replace and turning an otherwise
        # successful simulator startup into a failed run. Keep each staging
        # file unique and on the same filesystem as the destination.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.registry_path.name}.", suffix=".tmp",
            dir=self.registry_path.parent,
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"version": 1, "runs": persisted_runs}, indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.registry_path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self.poll_interval_s):
            try:
                self.reconcile()
            except Exception:
                # Keep the manager thread alive; current run state remains in
                # the registry and Docker labels allow a later reconciliation.
                continue

    def _summary(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if row.get("latest_telemetry") is None:
            try:
                fleet = json.loads(
                    (Path(row["output_dir"]) / "watch_latest.json").read_text(encoding="utf-8")
                )
                if isinstance(fleet, dict) and fleet.get("run_id") == row.get("run_id"):
                    row["latest_fleet"] = fleet
                    row["latest_telemetry"] = fleet
                    row["updated_at"] = fleet.get("ts")
            except (KeyError, OSError, ValueError):
                pass
        if row.get("state") in {"completed", "stopped"} and row.get("kind") != "evaluation":
            try:
                result = json.loads((Path(row["output_dir"]) / "stop_reason.json").read_text(encoding="utf-8"))
                steps = int(result["num_timesteps"])
                row["latest_metrics"] = {**(row.get("latest_metrics") or {}), "step": steps}
            except (KeyError, OSError, ValueError, TypeError):
                pass
        keys = (
            "run_id", "display_name", "kind", "simulator_image", "state", "created_at", "started_at",
            "finished_at", "config", "output_dir", "log_path", "error",
            "exit_code", "stop_reason", "cleanup_error", "network_name",
            "terminal_context",
            "api_container_name", "sim_container_name", "latest_fleet",
            "latest_metrics", "latest_telemetry", "map_id", "worker_container_ids",
            "simulator_backend", "simulator_connections",
        )
        summary = {key: row.get(key) for key in keys}
        summary["pid"] = None
        summary["containers"] = [
            name for name in (row.get("api_container_name"), row.get("sim_container_name"))
            if name
        ]
        summary["containers"].extend(f"aicar-run-{row['run_id']}-{role}" for role in (row.get("worker_container_ids") or {}))
        return summary

    def list_runs(self) -> Dict[str, Any]:
        with self._lock:
            rows = sorted(self._runs.values(), key=lambda item: item.get("created_at", ""), reverse=True)
            return {
                "runs": [self._summary(row) for row in rows],
                "max_concurrent_runs": self.max_concurrent_runs,
            }

    def create_run(
        self,
        official_config: Dict[str, Any],
        display_name: Optional[str] = None,
        *,
        kind: str = "training",
        evaluation_trace_steps: int = 0,
        evaluation_options: Dict[str, Any] | None = None,
        maintenance_resume: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if not self._backend_owner:
            raise RuntimeError("another Mission Control hub owns this native artifact root")
        if kind not in {"training", "evaluation"}:
            raise ValueError("unsupported official run kind")
        # Final shared admission boundary, including queued evaluations that
        # reach the manager after the campaign deadline advances.
        supplied_identity = (
            maintenance_resume.get("new_hub_identity")
            if isinstance(maintenance_resume, dict) else None
        )
        if isinstance(evaluation_trace_steps, bool) or not isinstance(evaluation_trace_steps, int):
            raise ValueError("evaluation_trace_steps must be an integer")
        if not 0 <= evaluation_trace_steps <= 500:
            raise ValueError("evaluation_trace_steps must be between 0 and 500")
        if kind != "evaluation" and evaluation_trace_steps:
            raise ValueError("evaluation traces are only supported for evaluation runs")
        evaluation_options = dict(evaluation_options or {})
        if evaluation_options and kind != "evaluation":
            raise ValueError("evaluation options are only supported for evaluation runs")
        if evaluation_options:
            validation_mode = evaluation_options.get("validation_mode")
            route_monitor_only = evaluation_options.get("route_monitor_only") is True
            if route_monitor_only:
                expected_frontier = (ROOT / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv").resolve()
                supplied_frontier = Path(str(evaluation_options.get("frontier_path", ""))).resolve()
                if (set(evaluation_options) != {"route_monitor_only", "frontier_path"}
                        or validation_mode is not None
                        or supplied_frontier != expected_frontier
                        or not supplied_frontier.is_file()):
                    raise ValueError("competitive route monitoring requires only the verified official frontier path")
            else:
                if validation_mode not in {"bc_diagnostic", "training_probe"}:
                    raise ValueError("unsupported evaluation validation mode")
                if (float(evaluation_options.get("lap_progress_timeout_s", 0)) != 240
                        or float(evaluation_options.get("lap_progress_wall_timeout_s", 0)) != 240):
                    raise ValueError("diagnostic lap progress timeouts must both be 240 seconds")
                if float(evaluation_options.get("frontier_stall_after_s", 0)) != 5 or not evaluation_options.get("frontier_path"):
                    raise ValueError("diagnostic requires a frontier path and five-second frontier stall timeout")
        config = normalize_official_config(official_config)
        # Older API callers do not send simulator_backend; keep their explicit
        # legacy behavior. The Mission Control form always sends its selected
        # backend, whose new-run default is the graphics-enabled Windows path.
        if "simulator_backend" not in official_config:
            config["simulator_backend"] = "docker"
            # normalize_official_config applies the Windows startup floor for
            # API/hash consistency; restore the historical Docker default for
            # legacy direct manager callers that omitted their backend.
            config["timeout_s"] = float(official_config.get("timeout_s", TRAIN_DEFAULTS["timeout_s"]))
        admission_context = (
            {"normalized_config": config, "new_hub_identity": supplied_identity}
            if isinstance(supplied_identity, dict) else None
        )
        # Recheck using the final normalized effective config. For ordinary
        # runs there is no maintenance context, preserving existing behavior.
        assert_campaign_allows(kind, maintenance_resume=admission_context)
        if kind == "training":
            # Validate the effective backend/defaults after legacy normalization
            # so an omitted backend cannot satisfy the plan then switch to Docker.
            if not is_exact_campaign_maintenance_resume(config, supplied_identity):
                assert_campaign_training_config(config)
        if kind == "evaluation" and (not config.get("resume") or config["n_envs"] != 1):
            raise ValueError("evaluation requires a checkpoint and exactly one fresh simulator")
        display = _safe_display_name(display_name)
        with self._lock:
            self._refresh_locked()
            active = [row for row in self._runs.values() if row.get("state") in {"starting", "running", "stopping"}]
            active_training = [row for row in active if row.get("kind", "training") != "evaluation"]
            if kind == "training" and len(active_training) >= self.max_concurrent_runs:
                raise RuntimeError(f"maximum concurrent official runs reached ({self.max_concurrent_runs})")
            if kind == "evaluation" and any(row.get("kind") == "evaluation" for row in active):
                raise RuntimeError("another official evaluation simulator is already active")
            # Allocate the durable run identity before consuming the one-shot
            # maintenance authorization so a timed-out API caller can reconcile
            # the accepted resume from the campaign event log.
            run_id = uuid.uuid4().hex[:12]
            campaign_state = load_campaign_state(default_state_path())
            if campaign_hub_maintenance_status(campaign_state) is not None:
                if admission_context is None:
                    raise RuntimeError("campaign hub maintenance is active; a verified maintenance resume is required")
                # Consume the single authorized resume while holding the run
                # manager lock, immediately before registry/run creation.
                claim_campaign_hub_maintenance_resume(
                    default_state_path(), kind=kind, normalized_config=config,
                    new_hub_identity=admission_context["new_hub_identity"],
                    resumed_run_id=run_id,
                )
            created = _utc_now()
            run_dir = self.runs_root / f"official_run_{run_id}"
            simulator_connections = None
            if config["simulator_backend"] == "windows_external":
                external_base_port = self._external_simulator_port_base(run_id, config["n_envs"])
                simulator_connections = [
                    {"environment": index, "host": getattr(self.backend, "simulator_host", "127.0.0.1"), "port": external_base_port + index,
                     "ros_domain_id": index}
                    for index in range(config["n_envs"])
                ]
            row: Dict[str, Any] = {
                "run_id": run_id,
                "display_name": display,
                "kind": kind,
                "simulator_image": (
                    "official AutoDRIVE IROS 2026 Windows simulator"
                    if config.get("simulator_backend", "docker") == "windows_external"
                    else (EVALUATION_SIM_IMAGE if kind == "evaluation" else TRAIN_SIM_IMAGE)
                ),
                "simulator_backend": config.get("simulator_backend", "docker"),
                "evaluation_trace_steps": evaluation_trace_steps,
                "evaluation_options": evaluation_options,
                "state": "starting",
                "created_at": created,
                "started_at": None,
                "finished_at": None,
                "config": config,
                "map_id": "none",
                "output_dir": str(run_dir),
                "log_path": str(run_dir / "hub_train.log"),
                "error": None,
                "exit_code": None,
                "stop_reason": None,
                "cleanup_error": None,
                "network_id": None,
                "network_name": f"aicar-run-{run_id}",
                "api_container_id": None,
                "api_container_name": f"aicar-run-{run_id}-api",
                "sim_container_id": None,
                "sim_container_name": f"aicar-run-{run_id}-sim",
                "simulator_connections": simulator_connections,
            }
            self._runs[run_id] = row
            self._creating.add(run_id)
            self._persist()
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "hub_train.log").touch()
            (run_dir / "manager_config.json").write_text(json.dumps({
                "run_id": run_id, "display_name": display, "runtime": "official",
                "config": config, "simulator_image": row["simulator_image"],
                "evaluation_trace_steps": evaluation_trace_steps,
                "evaluation_options": evaluation_options,
                "created_at": created,
            }, indent=2), encoding="utf-8")
            host_logs = self.backend.find_host_path_for_container_path(ROOT / "logs")
            network_id = self.backend.create_official_run_network(run_id)
            with self._lock:
                row["network_id"] = network_id
                if row.get("state") != "starting":
                    raise RuntimeError("run was stopped while its network was starting")
                self._persist()
            env = self._api_environment(config, run_id)
            external_simulators = config.get("simulator_backend", "docker") == "windows_external"
            external_base_port = row["simulator_connections"][0]["port"] if external_simulators else None
            if external_simulators:
                env = [value for value in env if not value.startswith("AICAR_OFFICIAL_SIMULATOR_IMAGE=")]
                env.append("AICAR_OFFICIAL_SIMULATOR_IMAGE=official AutoDRIVE IROS 2026 Windows simulator")
            if kind == "evaluation":
                env = [value for value in env if not value.startswith("AICAR_MODE=")]
                env.extend([
                    "AICAR_MODE=evaluate", "AICAR_KEEP_BRIDGE=0",
                    "AICAR_EVALUATION_ATTEMPTS=1", "AICAR_RACE_WALL_TIMEOUT_S=1800",
                    f"AICAR_MODEL_PATH={config['resume']}",
                    f"AICAR_EVALUATION_OUTPUT=/runs/rl/official_run_{run_id}/evaluation.json",
                ])
                if evaluation_trace_steps:
                    env.append(f"AICAR_EVALUATION_TRACE_STEPS={evaluation_trace_steps}")
                if evaluation_options:
                    if evaluation_options.get("route_monitor_only") is True:
                        env.extend([
                            "AICAR_ROUTE_MONITOR_ONLY=1",
                            f"AICAR_EVALUATION_FRONTIER_PATH={evaluation_options['frontier_path']}",
                        ])
                    else:
                        env.extend([
                            f"AICAR_EVALUATION_VALIDATION_MODE={evaluation_options['validation_mode']}",
                            f"AICAR_LAP_PROGRESS_TIMEOUT_S={float(evaluation_options['lap_progress_timeout_s'])}",
                            f"AICAR_LAP_PROGRESS_WALL_TIMEOUT_S={float(evaluation_options['lap_progress_wall_timeout_s'])}",
                            f"AICAR_FRONTIER_STALL_AFTER_S={float(evaluation_options['frontier_stall_after_s'])}",
                            f"AICAR_EVALUATION_FRONTIER_PATH={evaluation_options['frontier_path']}",
                        ])
            api_id = self.backend.create_official_run_container(
                run_id=run_id, role="api",
                image=GPU_API_IMAGE if config["device"] == "cuda" else API_IMAGE,
                network_name=row["network_name"], env=env,
                binds=[f"{host_logs}:/runs"],
                extra_hosts=["host.docker.internal:host-gateway"],
                metadata_labels=self._container_metadata(row),
                gpu=config["device"] == "cuda",
                published_ports={"4568": external_base_port} if external_simulators else None,
            )
            with self._lock:
                row["api_container_id"] = api_id
                if row.get("state") != "starting":
                    raise RuntimeError("run was stopped while its API container was starting")
                row["started_at"] = _utc_now()
                self._persist()
            for index in range(1, config["n_envs"]):
                worker_roles = ((f"api-{index}", API_IMAGE),) if external_simulators else (
                    (f"api-{index}", API_IMAGE), (f"sim-{index}", row["simulator_image"])
                )
                for role, image in worker_roles:
                    with self._lock:
                        if row.get("state") != "starting":
                            raise RuntimeError("run was stopped while workers were starting")
                    worker_id = self.backend.create_official_run_container(
                        run_id=run_id, role=role, image=image,
                        network_name=row["network_name"],
                        env=[f"ROS_DOMAIN_ID={index}", "AICAR_MODE=bridge"] if role.startswith("api") else [],
                        entrypoint=None if role.startswith("api") else ["/bin/bash", "-lc"],
                        command=None if role.startswith("api") else [
                            _official_simulator_command(f"api-{index}")
                        ],
                        metadata_labels=self._container_metadata(row),
                        gpu=role.startswith("sim"),
                        published_ports={"4568": external_base_port + index} if external_simulators and role.startswith("api") else None,
                    )
                    with self._lock:
                        row.setdefault("worker_container_ids", {})[role] = worker_id
                        self._persist()
            sim_id = None
            if not external_simulators:
                sim_id = self.backend.create_official_run_container(
                    run_id=run_id, role="sim", image=row["simulator_image"],
                    network_name=row["network_name"],
                    entrypoint=["/bin/bash", "-lc"],
                    command=[_official_simulator_command("api")],
                    metadata_labels=self._container_metadata(row),
                    gpu=True,
                )
            if external_simulators and hasattr(self.backend, "start_simulators"):
                self.backend.start_simulators(
                    run_id=run_id,
                    connections=simulator_connections or [],
                    hub_url=os.environ.get("AICAR_HUB_PUBLIC_URL", "http://127.0.0.1:8090"),
                    config=row["config"],
                )
            with self._lock:
                row["sim_container_id"] = sim_id
                if row.get("state") != "starting":
                    raise RuntimeError("run was stopped while containers were starting")
                row["state"] = "running"
                self._persist()
            return self._summary(row)
        except Exception as exc:
            self._cleanup_row_resources(row)
            with self._lock:
                if row.get("state") != "stopped":
                    row["state"] = "failed"
                    row["finished_at"] = _utc_now()
                    row["error"] = str(exc)
                self._persist()
            raise
        finally:
            with self._lock:
                self._creating.discard(run_id)

    @staticmethod
    def _api_environment(config: Dict[str, Any], run_id: str) -> List[str]:
        env = OfficialTrainSettings.model_validate(config).to_container_env(
            hub_url=os.environ.get("AICAR_HUB_PUBLIC_URL", "http://host.docker.internal:8090"),
            run_id=run_id,
        )
        env["AICAR_CONTROLLER"] = "ppo"
        env["AICAR_SIMULATOR_BACKEND"] = config.get("simulator_backend", "docker")
        env["AICAR_OFFICIAL_SIMULATOR_IMAGE"] = TRAIN_SIM_IMAGE
        return [f"{key}={value}" for key, value in env.items()]

    def _external_simulator_port_base(self, run_id: str, n_envs: int) -> int:
        # Reserve one unique, consecutive, localhost-only block per active run.
        # Each slot is eight ports wide (the maximum supported env count),
        # within the valid dynamic/private TCP port range.
        first_port = 49152
        block_count = (65536 - first_port) // 8
        reserved_blocks = set()
        for row in self._runs.values():
            if row.get("state") not in {"starting", "running", "stopping"}:
                continue
            for connection in row.get("simulator_connections") or []:
                try:
                    reserved_blocks.add((int(connection["port"]) - first_port) // 8)
                except (KeyError, TypeError, ValueError):
                    continue
        start_block = int(run_id[:6], 16) % block_count
        for offset in range(block_count):
            block = (start_block + offset) % block_count
            if block in reserved_blocks:
                continue
            base_port = first_port + block * 8
            if base_port + n_envs - 1 <= 65535:
                return base_port
        raise RuntimeError("no unused localhost port block is available for external simulators")

    @staticmethod
    def _container_metadata(row: Dict[str, Any]) -> Dict[str, str]:
        return {
            "aicar.display_name": row["display_name"],
            "aicar.kind": row.get("kind", "training"),
            "aicar.simulator_image": row["simulator_image"],
            "aicar.output_dir": row["output_dir"],
            "aicar.config": json.dumps(row["config"], separators=(",", ":")),
            "aicar.created_at": row["created_at"],
        }

    def _cleanup_row_resources(self, row: Dict[str, Any], *, stopped_ids: Optional[set[str]] = None) -> None:
        cleanup_errors: List[str] = []
        row["cleanup_error"] = None
        self._capture_terminal_context(row)
        resources = [(key, row.get(key)) for key in ("api_container_id", "sim_container_id")]
        resources.extend(list((row.get("worker_container_ids") or {}).items()))
        for key, container_id in resources:
            if container_id:
                try:
                    logs = self.backend.official_run_container_logs(container_id)
                    if logs:
                        log_path = Path(row["log_path"])
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        source_logs = getattr(self.backend, "_groups", {}).get(container_id, {}).get("records", [])
                        source_paths = [
                            Path(record["log"].name) for record in source_logs
                            if getattr(record.get("log"), "name", None)
                        ]
                        target_path = log_path.resolve()
                        if not any(path.resolve() == target_path for path in source_paths):
                            with log_path.open("a", encoding="utf-8") as output:
                                output.write(logs)
                                if not logs.endswith("\n"):
                                    output.write("\n")
                except Exception as exc:
                    cleanup_errors.append(f"log capture {container_id}: {exc}")
                try:
                    if container_id not in (stopped_ids or set()):
                        self.backend.stop_official_run_container(container_id, timeout_s=60 if key == "api_container_id" else 5)
                        self._capture_terminal_context(row)
                except Exception as exc:
                    cleanup_errors.append(f"stop {container_id}: {exc}")
                try:
                    self.backend.remove_official_run_container(container_id)
                    if key in {"api_container_id", "sim_container_id"}:
                        row[key] = None
                    else:
                        row["worker_container_ids"].pop(key, None)
                except Exception as exc:
                    cleanup_errors.append(f"remove {container_id}: {exc}")
        stop_simulators = getattr(self.backend, "stop_simulators", None)
        if stop_simulators:
            try:
                stop_simulators(str(row.get("run_id", "")))
            except Exception as exc:
                cleanup_errors.append(f"stop Windows simulators: {exc}")
        network_id = row.get("network_id")
        if network_id:
            try:
                self.backend.remove_official_run_network(network_id)
            except Exception as exc:
                cleanup_errors.append(f"remove network {network_id}: {exc}")
            else:
                row["network_id"] = None
        if cleanup_errors:
            row["cleanup_error"] = "; ".join(cleanup_errors)

    def _capture_terminal_context(self, row: Dict[str, Any]) -> None:
        """Copy native process evidence to the registry before backend cleanup."""
        if isinstance(row.get("terminal_context"), dict):
            return
        inventory = getattr(self.backend, "list_official_run_containers", None)
        if inventory is None:
            return
        try:
            items = inventory(str(row.get("run_id", "")))
        except Exception:
            return
        api = next((item for item in items
                    if (item.get("Labels") or {}).get("aicar.role") == "api"
                    and item.get("State") == "exited"), None)
        context = (api or {}).get("TerminalContext")
        if not isinstance(context, dict):
            return
        row["terminal_context"] = context
        trainer = next((record for record in context.get("records", [])
                        if record.get("role") == "trainer"), None)
        if trainer and trainer.get("returncode_known"):
            row["exit_code"] = trainer.get("returncode")

    def _refresh_locked(self) -> None:
        """Reserved for synchronous caller-side state reconciliation."""
        # Docker list snapshots also expose exited containers and are the
        # authoritative source when a monitor callback was delayed.
        self.reconcile()

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._runs.get(run_id)
            if row is None:
                raise KeyError(run_id)
            return self._summary(row)

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._runs.get(run_id)
            if row is None:
                raise KeyError(run_id)
            if run_id in self._stopping:
                return self._summary(row)
            if row.get("state") not in {"starting", "running", "stopping"}:
                return self._summary(row)
            self._stopping.add(run_id)
            row["state"] = "stopping"
            row["stop_reason"] = "user_requested"
            self._persist()
            was_evaluation_paused = run_id in self._evaluation_paused
            self._evaluation_paused.discard(run_id)
            abort_pause = getattr(self.backend, "abort_evaluation_pause", None)
            ids = [row.get("api_container_id")]
        if was_evaluation_paused and abort_pause:
            try:
                abort_pause(run_id)
            except Exception:
                pass
            # The API container owns the trainer. Send it SIGTERM while the
            # simulator is still publishing sensors so OfficialRaceEnv.step
            # can return, the stop callback can exit cleanly, and the trainer
            # can save its final checkpoint before the simulator is stopped.
        errors = []
        stopped_ids = set()
        for container_id in ids:
            if container_id:
                try:
                    self.backend.stop_official_run_container(container_id)
                    stopped_ids.add(container_id)
                except Exception as exc:
                    errors.append(str(exc))
        try:
            self._cleanup_row_resources(row, stopped_ids=stopped_ids)
            with self._lock:
                row["state"] = "stopped" if not errors else "failed"
                row["error"] = "; ".join(errors) or row.get("error")
                if errors:
                    row["stop_reason"] = "stop_error"
                row["finished_at"] = _utc_now()
                self._persist()
                return self._summary(row)
        finally:
            with self._lock:
                self._stopping.discard(run_id)

    def pause_active_training_for_evaluation(self) -> list[str]:
        """Free simulator capacity so official evaluations measure the model alone."""
        with self._lock:
            rows = [dict(row) for row in self._runs.values()
                    if row.get("kind", "training") == "training"
                    and row.get("state") == "running"]
        if not rows:
            return []
        pause = getattr(self.backend, "pause_training_for_evaluation", None)
        if not callable(pause):
            raise RuntimeError("official evaluation cannot run reliably while training is active: backend cannot pause training")

        paused: list[str] = []
        hub_url = os.environ.get("AICAR_HUB_PUBLIC_URL", "http://127.0.0.1:8090")
        try:
            for row in rows:
                run_id = str(row["run_id"])
                if row.get("simulator_backend", row.get("config", {}).get("simulator_backend")) != "windows_external":
                    raise RuntimeError(f"cannot isolate evaluation from active training run {run_id}")
                pause(
                    run_id=run_id,
                    connections=list(row.get("simulator_connections") or []),
                    hub_url=hub_url,
                )
                with self._lock:
                    self._evaluation_paused.add(run_id)
                paused.append(run_id)
            return paused
        except Exception:
            self.resume_training_after_evaluation(paused)
            raise

    def resume_training_after_evaluation(self, run_ids: list[str]) -> None:
        resume = getattr(self.backend, "resume_training_after_evaluation", None)
        errors: list[str] = []
        for run_id in reversed(run_ids):
            with self._lock:
                row = self._runs.get(run_id)
                still_running = row is not None and row.get("state") == "running"
                paused = run_id in self._evaluation_paused
            if not paused:
                continue
            try:
                if still_running:
                    if not callable(resume):
                        raise RuntimeError("backend cannot resume a paused training run")
                    resume(
                        run_id=run_id,
                        connections=list(row.get("simulator_connections") or []),
                        hub_url=os.environ.get("AICAR_HUB_PUBLIC_URL", "http://127.0.0.1:8090"),
                    )
            except Exception as exc:
                errors.append(f"{run_id}: {exc}")
            finally:
                with self._lock:
                    self._evaluation_paused.discard(run_id)
        if errors:
            raise RuntimeError("could not resume training after official evaluation: " + "; ".join(errors))

    def recover_paused_training_after_hub_restart(self) -> None:
        """Restore an interrupted evaluation's parent learner after hub recovery."""
        with self._lock:
            rows = [dict(row) for row in self._runs.values()
                    if row.get("kind", "training") == "training"
                    and row.get("state") == "running"
                    and row.get("simulator_backend", row.get("config", {}).get("simulator_backend")) == "windows_external"]
        paused = []
        for row in rows:
            pause_file = Path(row.get("output_dir", "")) / "evaluation_pause.flag"
            if pause_file.is_file():
                run_id = str(row["run_id"])
                with self._lock:
                    self._evaluation_paused.add(run_id)
                paused.append(run_id)
        if paused:
            self.resume_training_after_evaluation(paused)

    def telemetry(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._runs.get(run_id)
            if row is None:
                raise KeyError(run_id)
            run_dir = Path(row["output_dir"])
        fleet = None
        try:
            candidate = json.loads((run_dir / "watch_latest.json").read_text(encoding="utf-8"))
            if isinstance(candidate, dict) and candidate.get("run_id") == run_id:
                fleet = candidate
        except (OSError, ValueError):
            pass
        with self._lock:
            current = self._runs[run_id]
            if fleet is not None:
                current["latest_fleet"] = fleet
                current["latest_telemetry"] = fleet
                current["updated_at"] = fleet.get("ts")
            latest_fleet = fleet or current.get("latest_fleet")
            return {
                "run_id": run_id,
                "state": current.get("state"),
                "updated_at": current.get("updated_at"),
                "fleet": latest_fleet,
                "last_fleet": latest_fleet,
                "last_metrics": current.get("latest_metrics"),
                "last_train_phase": current.get("training_phase"),
                "last_evaluator_live": current.get("last_evaluator_live"),
                "last_evaluation": current.get("last_evaluation"),
                "map_id": current.get("map_id"),
            }

    def ingest_telemetry(self, payload: Dict[str, Any]) -> bool:
        """Cache a run-tagged official `/telemetry` event for its GET stream."""
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if not run_id:
            return False
        with self._lock:
            row = self._runs.get(str(run_id))
            if row is None:
                return False
            kind = payload.get("kind", "metrics")
            if kind == "fleet":
                row["latest_fleet"] = payload
                row["latest_telemetry"] = payload
                row["updated_at"] = payload.get("ts")
            elif kind == "metrics":
                row["latest_metrics"] = payload
                row["updated_at"] = payload.get("ts")
            elif kind == "phase":
                row["training_phase"] = payload
                row["updated_at"] = payload.get("ts")
            else:
                return False
            return True

    def ingest_evaluator_live(self, payload: Dict[str, Any]) -> bool:
        """Cache a run-tagged evaluator pose event for the run telemetry route."""
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if not run_id:
            return False
        with self._lock:
            row = self._runs.get(str(run_id))
            if row is None:
                return False
            row["last_evaluator_live"] = payload
            row["updated_at"] = payload.get("ts")
            return True

    def reconcile(self, run_id: Optional[str] = None) -> None:
        """Recover run records and lifecycle from persistent registry + labels."""
        if not self._backend_owner:
            return
        try:
            containers = self.backend.list_official_run_containers(run_id)
        except Exception:
            return
        try:
            networks = self.backend.list_official_run_networks(run_id)
        except Exception:
            # A failed inventory is not proof the network disappeared.
            return
        networks_by_run = {
            str((item.get("Labels") or {}).get("aicar.run_id")): item
            for item in networks if (item.get("Labels") or {}).get("aicar.run_id")
        }
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for item in containers:
            labels = item.get("Labels") or {}
            rid = labels.get("aicar.run_id")
            role = labels.get("aicar.role")
            if rid and (role in {"api", "sim"} or str(role).startswith(("api-", "sim-"))):
                grouped.setdefault(str(rid), {})[str(role)] = item
        cleanup_rows: List[Dict[str, Any]] = []
        with self._lock:
            for rid, roles in grouped.items():
                row = self._runs.get(rid)
                if row is None:
                    names = {role: (item.get("Names") or [""])[0].lstrip("/") for role, item in roles.items()}
                    row = {
                        "run_id": rid,
                        "display_name": (roles.get("api", {}).get("Labels") or {}).get("aicar.display_name", rid),
                        "state": "unknown",
                        "kind": (roles.get("api", {}).get("Labels") or {}).get("aicar.kind", "training"),
                        "created_at": (roles.get("api", {}).get("Labels") or {}).get("aicar.created_at", _utc_now()),
                        "started_at": None, "finished_at": None,
                        "config": {},
                        "output_dir": (roles.get("api", {}).get("Labels") or {}).get(
                            "aicar.output_dir", str(self.runs_root / f"official_run_{rid}")
                        ),
                        "log_path": None, "error": None, "exit_code": None,
                        "network_id": (networks_by_run.get(rid) or {}).get("Id"),
                        "network_name": (networks_by_run.get(rid) or {}).get("Name", f"aicar-run-{rid}"),
                        "api_container_id": None, "api_container_name": names.get("api"),
                        "sim_container_id": None, "sim_container_name": names.get("sim"),
                    }
                    labels = roles.get("api", {}).get("Labels") or {}
                    try:
                        row["config"] = json.loads(labels.get("aicar.config", "{}"))
                    except (TypeError, ValueError):
                        row["config"] = {}
                    row["log_path"] = str(Path(row["output_dir"]) / "hub_train.log")
                    self._runs[rid] = row
                if rid in self._creating or rid in self._stopping:
                    continue
                network = networks_by_run.get(rid)
                if network:
                    row["network_id"] = network.get("Id")
                    row["network_name"] = network.get("Name", row.get("network_name"))
                for role, item in roles.items():
                    if role in {"api", "sim"}:
                        row[f"{role}_container_id"] = item.get("Id")
                    else:
                        row.setdefault("worker_container_ids", {})[role] = item.get("Id")
                try:
                    api = roles.get("api")
                    sim = roles.get("sim")
                    api_state = (api or {}).get("State", "")
                    sim_state = (sim or {}).get("State", "")
                    sentinel = Path(row.get("output_dir", "")) / ("evaluation.json" if row.get("kind") == "evaluation" else "stop_reason.json")
                    external_simulators = row.get("simulator_backend", row.get("config", {}).get("simulator_backend", "docker")) == "windows_external"
                    expected_roles = ({"api"} if external_simulators else {"api", "sim"}) | {
                        f"{role}-{index}" for index in range(1, int(row.get("config", {}).get("n_envs", 1)))
                        for role in (("api",) if external_simulators else ("api", "sim"))
                    }
                    all_running = all(roles.get(role, {}).get("State") == "running" for role in expected_roles)
                    if all_running:
                        if network is None:
                            row["state"] = "failed"
                            row["finished_at"] = row.get("finished_at") or _utc_now()
                            row["error"] = row.get("error") or "isolated Docker network is missing"
                            cleanup_rows.append(row)
                        elif row.get("state") == "stopping":
                            row["state"] = "stopped"
                            row["finished_at"] = row.get("finished_at") or _utc_now()
                            cleanup_rows.append(row)
                        elif row.get("state") not in {"stopped", "failed", "completed"}:
                            sentinel = Path(row.get("output_dir", "")) / ("evaluation.json" if row.get("kind") == "evaluation" else "stop_reason.json")
                            sentinel_reason = _stop_reason_from_sentinel(sentinel)
                            rejected_bc = str(sentinel_reason or "").startswith("behavior_cloning_rejected")
                            row["state"] = (
                                "failed" if rejected_bc else "completed"
                            ) if sentinel.is_file() else "running"
                            row["started_at"] = row.get("started_at") or _utc_now()
                        if row["state"] in {"completed", "failed", "stopped"}:
                            row["finished_at"] = row.get("finished_at") or _utc_now()
                            try:
                                row["stop_reason"] = json.loads(sentinel.read_text(encoding="utf-8")).get("reason", "evaluation_complete" if row.get("kind") == "evaluation" else "training_complete")
                            except (OSError, ValueError, AttributeError):
                                row["stop_reason"] = row.get("stop_reason") or ("training_complete" if row["state"] == "completed" else "container_not_running")
                            cleanup_rows.append(row)
                    elif row.get("state") == "stopping":
                        row["state"] = "stopped"
                        row["finished_at"] = row.get("finished_at") or _utc_now()
                        cleanup_rows.append(row)
                    elif api_state == "exited" or (not external_simulators and sim_state == "exited"):
                        terminal_context = (api or {}).get("TerminalContext")
                        if terminal_context is None:
                            terminal_path = Path(row.get("output_dir", "")) / "native_terminal_context.json"
                            try:
                                terminal_context = json.loads(terminal_path.read_text(encoding="utf-8"))
                            except (OSError, ValueError, TypeError):
                                terminal_context = None
                        if isinstance(terminal_context, dict):
                            row["terminal_context"] = terminal_context
                            if not row["terminal_context"].get("triggering_roles"):
                                row["terminal_context"]["triggering_roles"] = [
                                    record.get("role") for record in row["terminal_context"].get("records", [])
                                    if record.get("status") == "exited"
                                ]
                            trainer = next((record for record in terminal_context.get("records", [])
                                            if record.get("role") == "trainer"), None)
                            if trainer and trainer.get("returncode_known"):
                                row["exit_code"] = trainer.get("returncode")
                        sentinel = Path(row.get("output_dir", "")) / ("evaluation.json" if row.get("kind") == "evaluation" else "stop_reason.json")
                        sentinel_reason = _stop_reason_from_sentinel(sentinel)
                        rejected_bc = str(sentinel_reason or "").startswith("behavior_cloning_rejected")
                        row["state"] = (
                            "failed" if rejected_bc else "completed"
                        ) if sentinel.is_file() else "failed"
                        row["finished_at"] = row.get("finished_at") or _utc_now()
                        for item in (api, sim):
                            if item and item.get("State") == "exited" and row.get("exit_code") is None:
                                try:
                                    inspect = getattr(self.backend, "inspect_container", None)
                                    if inspect:
                                        row["exit_code"] = inspect(item["Id"]).get("State", {}).get("ExitCode")
                                except Exception:
                                    pass
                                break
                        if row["state"] == "completed":
                            try:
                                row["stop_reason"] = json.loads(sentinel.read_text(encoding="utf-8")).get("reason", "evaluation_complete" if row.get("kind") == "evaluation" else "training_complete")
                            except (OSError, ValueError, AttributeError):
                                row["stop_reason"] = row.get("stop_reason") or ("training_complete" if row["state"] == "completed" else "container_not_running")
                        if row["state"] == "failed":
                            row["stop_reason"] = sentinel_reason or "container_exit"
                            if rejected_bc:
                                row["error"] = row.get("error") or str(sentinel_reason)
                            else:
                                exit_detail = f" (trainer exit code {row['exit_code']})" if row.get("exit_code") is not None else ""
                                row["error"] = row.get("error") or f"official API or simulator container exited{exit_detail}"
                        cleanup_rows.append(row)
                    elif row.get("state") not in {"stopped", "failed", "completed"}:
                        row["state"] = "failed"
                        row["finished_at"] = row.get("finished_at") or _utc_now()
                        row["stop_reason"] = "container_not_running"
                        row["error"] = row.get("error") or "official run has an incomplete or non-running container pair"
                        cleanup_rows.append(row)
                except Exception:
                    continue
            for rid, network in networks_by_run.items():
                if rid in grouped or rid in self._creating or rid in self._stopping:
                    continue
                row = self._runs.get(rid)
                if row is None:
                    row = {
                        "run_id": rid,
                        "display_name": rid,
                        "state": "failed",
                        "created_at": _utc_now(),
                        "started_at": None,
                        "finished_at": _utc_now(),
                        "config": {},
                        "output_dir": str(self.runs_root / f"official_run_{rid}"),
                        "log_path": str(self.runs_root / f"official_run_{rid}" / "hub_train.log"),
                        "error": "orphaned official-run network recovered after hub restart",
                        "exit_code": None,
                        "network_id": network.get("Id"),
                        "network_name": network.get("Name", f"aicar-run-{rid}"),
                        "api_container_id": None,
                        "api_container_name": None,
                        "sim_container_id": None,
                        "sim_container_name": None,
                    }
                    self._runs[rid] = row
                    cleanup_rows.append(row)
                elif row.get("state") in {"starting", "running", "stopping"}:
                    row["state"] = "failed"
                    row["finished_at"] = row.get("finished_at") or _utc_now()
                    row["stop_reason"] = "orphaned_network"
                    row["error"] = row.get("error") or "run containers were not found; orphaned network recovered"
                    row["network_id"] = network.get("Id")
                    cleanup_rows.append(row)
            for rid, row in self._runs.items():
                if (run_id is None or rid == run_id) and rid not in grouped and rid not in self._creating and rid not in self._stopping and row.get("state") in {
                    "starting", "running", "stopping"
                }:
                    row["state"] = "stopped" if row.get("state") == "stopping" else "failed"
                    row["finished_at"] = row.get("finished_at") or _utc_now()
                    if row["state"] == "failed":
                        row["error"] = row.get("error") or "official run containers were not found during reconciliation"
                    cleanup_rows.append(row)
            self._persist()
        seen = set()
        for row in cleanup_rows:
            if row["run_id"] in seen:
                continue
            seen.add(row["run_id"])
            self._cleanup_row_resources(row)
        if cleanup_rows:
            with self._lock:
                self._persist()
