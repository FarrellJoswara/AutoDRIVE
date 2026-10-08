"""Persistent, isolated manager for official AutoDRIVE training runs."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.layer4.settings import ROOT
from src.layer4.hub import docker_control
from src.layer4.official_settings import OfficialTrainSettings


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_official_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the official trainer's public settings and reject unknowns."""
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    unknown = set(raw) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"unsupported official training settings: {sorted(unknown)}")
    result = OfficialTrainSettings.model_validate(raw).model_dump(mode="python")
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
        result["resume"] = f"/runs/{relative.as_posix()}"
    return result


def _safe_display_name(value: Optional[str]) -> str:
    if value is not None and not isinstance(value, str):
        raise ValueError("display_name must be text")
    name = (value or "official-training").strip()
    if not name or len(name) > 100:
        raise ValueError("display_name must contain 1 to 100 characters")
    return name


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
        self._load_registry()
        self.reconcile()
        self._monitor_stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None
        if start_monitor:
            self._monitor = threading.Thread(
                target=self._monitor_loop, name="official-run-manager", daemon=True
            )
            self._monitor.start()

    def close(self) -> None:
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=2.0)

    def _load_registry(self) -> None:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            rows = data.get("runs", []) if isinstance(data, dict) else []
            for row in rows:
                if isinstance(row, dict) and row.get("run_id"):
                    self._runs[str(row["run_id"])] = row
        except (OSError, ValueError, TypeError):
            self._runs = {}

    def _persist(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
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
        tmp.write_text(
            json.dumps({"version": 1, "runs": persisted_runs}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.registry_path)

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
            "api_container_name", "sim_container_name", "latest_fleet",
            "latest_metrics", "latest_telemetry", "map_id", "worker_container_ids",
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
    ) -> Dict[str, Any]:
        if kind not in {"training", "evaluation"}:
            raise ValueError("unsupported official run kind")
        if isinstance(evaluation_trace_steps, bool) or not isinstance(evaluation_trace_steps, int):
            raise ValueError("evaluation_trace_steps must be an integer")
        if not 0 <= evaluation_trace_steps <= 500:
            raise ValueError("evaluation_trace_steps must be between 0 and 500")
        if kind != "evaluation" and evaluation_trace_steps:
            raise ValueError("evaluation traces are only supported for evaluation runs")
        config = normalize_official_config(official_config)
        if kind == "evaluation" and (not config.get("resume") or config["n_envs"] != 1):
            raise ValueError("evaluation requires a checkpoint and exactly one fresh simulator")
        display = _safe_display_name(display_name)
        with self._lock:
            self._refresh_locked()
            active = [row for row in self._runs.values() if row.get("state") in {"starting", "running", "stopping"}]
            if len(active) >= self.max_concurrent_runs:
                raise RuntimeError(f"maximum concurrent official runs reached ({self.max_concurrent_runs})")
            run_id = uuid.uuid4().hex[:12]
            created = _utc_now()
            run_dir = self.runs_root / f"official_run_{run_id}"
            row: Dict[str, Any] = {
                "run_id": run_id,
                "display_name": display,
                "kind": kind,
                "simulator_image": EVALUATION_SIM_IMAGE if kind == "evaluation" else TRAIN_SIM_IMAGE,
                "evaluation_trace_steps": evaluation_trace_steps,
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
            }
            self._runs[run_id] = row
            self._creating.add(run_id)
            self._persist()
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "hub_train.log").touch()
            (run_dir / "manager_config.json").write_text(json.dumps({
                "run_id": run_id, "display_name": display, "runtime": "official",
                "config": config, "simulator_image": row["simulator_image"], "created_at": created,
            }, indent=2), encoding="utf-8")
            host_logs = self.backend.find_host_path_for_container_path(ROOT / "logs")
            network_id = self.backend.create_official_run_network(run_id)
            with self._lock:
                row["network_id"] = network_id
                if row.get("state") != "starting":
                    raise RuntimeError("run was stopped while its network was starting")
                self._persist()
            env = self._api_environment(config, run_id)
            if kind == "evaluation":
                env = [value for value in env if not value.startswith("AICAR_MODE=")]
                env.extend([
                    "AICAR_MODE=evaluate", "AICAR_KEEP_BRIDGE=0",
                    f"AICAR_MODEL_PATH={config['resume']}",
                    f"AICAR_EVALUATION_OUTPUT=/runs/rl/official_run_{run_id}/evaluation.json",
                ])
                if evaluation_trace_steps:
                    env.append(f"AICAR_EVALUATION_TRACE_STEPS={evaluation_trace_steps}")
            api_id = self.backend.create_official_run_container(
                run_id=run_id, role="api",
                image=GPU_API_IMAGE if config["device"] == "cuda" else API_IMAGE,
                network_name=row["network_name"], env=env,
                binds=[f"{host_logs}:/runs"],
                extra_hosts=["host.docker.internal:host-gateway"],
                metadata_labels=self._container_metadata(row),
                gpu=config["device"] == "cuda",
            )
            with self._lock:
                row["api_container_id"] = api_id
                if row.get("state") != "starting":
                    raise RuntimeError("run was stopped while its API container was starting")
                row["started_at"] = _utc_now()
                self._persist()
            for index in range(1, config["n_envs"]):
                for role, image in ((f"api-{index}", API_IMAGE), (f"sim-{index}", row["simulator_image"])):
                    with self._lock:
                        if row.get("state") != "starting":
                            raise RuntimeError("run was stopped while workers were starting")
                    worker_id = self.backend.create_official_run_container(
                        run_id=run_id, role=role, image=image,
                        network_name=row["network_name"],
                        env=[f"ROS_DOMAIN_ID={index}", "AICAR_MODE=bridge"] if role.startswith("api") else [],
                        entrypoint=None if role.startswith("api") else ["/bin/bash", "-lc"],
                        command=None if role.startswith("api") else [
                            f'"./AutoDRIVE Simulator.x86_64" -batchmode -nographics -ip api-{index} -port 4568'
                        ],
                        metadata_labels=self._container_metadata(row),
                    )
                    with self._lock:
                        row.setdefault("worker_container_ids", {})[role] = worker_id
                        self._persist()
            sim_id = self.backend.create_official_run_container(
                run_id=run_id, role="sim", image=row["simulator_image"],
                network_name=row["network_name"],
                entrypoint=["/bin/bash", "-lc"],
                command=['"./AutoDRIVE Simulator.x86_64" -batchmode -nographics -ip api -port 4568'],
                metadata_labels=self._container_metadata(row),
            )
            with self._lock:
                row["sim_container_id"] = sim_id
                if row.get("state") != "starting":
                    raise RuntimeError("run was stopped while containers were starting")
                row["state"] = "running"
                self._persist()
            return self._summary(row)
        except Exception as exc:
            self._cleanup_row_resources(row, stopped_ids=stopped_ids)
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
        return [f"{key}={value}" for key, value in env.items()]

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
        resources = [(key, row.get(key)) for key in ("api_container_id", "sim_container_id")]
        resources.extend(list((row.get("worker_container_ids") or {}).items()))
        for key, container_id in resources:
            if container_id:
                try:
                    logs = self.backend.official_run_container_logs(container_id)
                    if logs:
                        log_path = Path(row["log_path"])
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a", encoding="utf-8") as output:
                            output.write(logs)
                            if not logs.endswith("\n"):
                                output.write("\n")
                except Exception as exc:
                    cleanup_errors.append(f"log capture {container_id}: {exc}")
                try:
                    if container_id not in (stopped_ids or set()):
                        self.backend.stop_official_run_container(container_id, timeout_s=60 if key == "api_container_id" else 5)
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
            # The API container owns the trainer. Send it SIGTERM while the
            # simulator is still publishing sensors so OfficialRaceEnv.step
            # can return, the stop callback can exit cleanly, and the trainer
            # can save its final checkpoint before the simulator is stopped.
            ids = [row.get("api_container_id")]
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
                    expected_roles = {"api", "sim"} | {
                        f"{role}-{index}" for index in range(1, int(row.get("config", {}).get("n_envs", 1)))
                        for role in ("api", "sim")
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
                            row["state"] = "completed" if sentinel.is_file() else "running"
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
                    elif api_state == "exited" or sim_state == "exited":
                        sentinel = Path(row.get("output_dir", "")) / ("evaluation.json" if row.get("kind") == "evaluation" else "stop_reason.json")
                        row["state"] = "completed" if sentinel.is_file() else "failed"
                        row["finished_at"] = row.get("finished_at") or _utc_now()
                        for item in (api, sim):
                            if item and item.get("State") == "exited":
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
                            row["stop_reason"] = "container_exit"
                            row["error"] = row.get("error") or "official API or simulator container exited"
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
                if rid in grouped or rid in self._creating:
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
                if (run_id is None or rid == run_id) and rid not in grouped and rid not in self._creating and row.get("state") in {
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
