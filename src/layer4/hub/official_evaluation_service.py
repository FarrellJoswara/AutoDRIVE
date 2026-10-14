"""Serialized official race evaluations and a durable cross-run champion."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from src.layer4.hub.official_models import checkpoint_settings
from src.layer4.hub.campaign_deadline import assert_campaign_allows
from src.layer4.settings import ROOT
from src.layer3.bc_artifact import (
    verify_bc_checkpoint,
    verify_ppo_lineage_checkpoint,
    verify_training_probe_checkpoint,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluation_child_settings(
    settings: dict[str, Any], model_path: str, *, race_laps: int,
) -> dict[str, Any]:
    """Build a fresh evaluation run config from a trainer/checkpoint config.

    The validation protocol is carried separately in ``evaluation_options``.
    An evaluation child never launches its own automatic evaluations, so it
    must not inherit the parent's ``training_probe`` auto-evaluation mode.
    """
    return {
        **settings,
        "resume": model_path,
        "n_envs": 1,
        "warmup_laps": 1,
        "race_laps": race_laps,
        "total_timesteps": 0,
        "automatic_evaluation_enabled": False,
        "automatic_evaluation_mode": "competitive",
    }


def _competitive_route_monitor_options() -> dict[str, Any]:
    """Attach route telemetry to competitive reports without changing scoring."""
    return {
        "route_monitor_only": True,
        "frontier_path": str(ROOT / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempt_collision_count(attempt: dict[str, Any]) -> int:
    """Count impacts even when the official race counter has not started.

    The evaluator stops immediately on a collision, including during the
    ignored warm-up lap. In that case ``race_collisions`` is legitimately zero
    while ``raw_collision_count_at_finish`` records the impact. Mission
    Control's batch total should still report the collision that ended the run.
    """
    race_count = int(attempt.get("race_collisions") or 0)
    raw_count = int(attempt.get("raw_collision_count_at_finish") or 0)
    return max(race_count, raw_count)


class OfficialEvaluationService:
    """Runs automatic and manual full-race evaluations through one queue."""

    ATTEMPTS = 3
    REGISTRY_PATH = ROOT / "logs" / "layer4" / "official_evaluations.json"
    CHAMPION_DIR = ROOT / "logs" / "rl" / "best_official_model"

    def __init__(
        self, manager: Any, *, registry_path: Path | None = None,
        champion_dir: Path | None = None, model_root: Path | None = None,
    ) -> None:
        self.manager = manager
        self.registry_path = Path(registry_path or self.REGISTRY_PATH)
        self.champion_dir = Path(champion_dir or self.CHAMPION_DIR)
        self.model_root = Path(model_root or ROOT / "logs" / "rl").resolve()
        self._condition = threading.Condition()
        self._queue: deque[dict[str, Any]] = deque()
        self._current: dict[str, Any] | None = None
        self._latest: dict[str, Any] | None = None
        self._deferred_auto: dict[str, dict[str, Any]] = {}
        self._history: list[dict[str, Any]] = []
        self._best: dict[str, Any] | None = None
        self._restart_queue: list[dict[str, Any]] = []
        self._worker_stop = False
        self._load_registry()
        self._queue.extend(self._restart_queue)
        self._restart_queue.clear()
        self._stop_orphan_evaluations()
        recover_paused_training = getattr(manager, "recover_paused_training_after_hub_restart", None)
        if callable(recover_paused_training):
            recover_paused_training()
        self._persist()
        self._worker = threading.Thread(target=self._worker_loop, name="official-evaluation-queue", daemon=True)
        self._worker.start()

    def _load_registry(self) -> None:
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            model_root = self.model_root

            def belongs_to_model_store(row: Any) -> bool:
                if not isinstance(row, dict):
                    return False
                try:
                    raw_model = str(row.get("model_path", ""))
                    if raw_model.startswith("best_official_model/"):
                        model = (self.champion_dir / "model.zip").resolve(strict=True)
                        model.relative_to(self.champion_dir.resolve())
                    else:
                        model = Path(raw_model).resolve(strict=True)
                        model.relative_to(model_root)
                    return model.is_file() and model.suffix.lower() == ".zip"
                except (OSError, ValueError):
                    return False

            def is_history_record(row: Any) -> bool:
                # Keep completed scores useful even if a user later prunes the
                # evaluated checkpoint. Failed rows need a real persistent
                # source model so temporary test artifacts cannot leak in.
                return belongs_to_model_store(row) or (
                    isinstance(row, dict)
                    and row.get("state") in {"complete", "incomplete", "stopped"}
                    and isinstance(row.get("result"), dict)
                )

            # Registries may outlive temporary test directories or be moved
            # between hosts. Never expose or retain entries whose checkpoint
            # is outside this workspace's persistent model store.
            self._history = [row for row in payload.get("history", []) if is_history_record(row)]
            self._latest = self._history[-1] if self._history else None
            best = payload.get("best")
            self._best = best if belongs_to_model_store(best) else None
            self._restart_queue = [
                row for row in payload.get("queued", [])
                if belongs_to_model_store(row)
            ]
            active = payload.get("active")
            if belongs_to_model_store(active) and not any(
                row.get("evaluation_id") == active.get("evaluation_id") for row in self._history
            ):
                active["state"] = "interrupted"
                active["error"] = "Mission Control restarted during this evaluation; its partial attempts were not promoted."
                active["updated_utc"] = _utc_now()
                self._history.append(active)
        except (OSError, ValueError, TypeError):
            self._history, self._best, self._restart_queue, self._latest = [], None, [], None

    def _stop_orphan_evaluations(self) -> None:
        # A restarted hub cannot safely resume a half-finished race. Stop its
        # simulator and keep the durable champion/history intact.
        try:
            rows = self.manager.list_runs().get("runs", [])
        except Exception:
            return
        for row in rows:
            if row.get("kind") == "evaluation" and row.get("state") in {"starting", "running", "stopping"}:
                try:
                    self.manager.stop_run(str(row["run_id"]))
                except Exception:
                    pass

    def _persist(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1, "updated_utc": _utc_now(), "best": self._best,
            "history": self._history[-500:], "active": self._current,
            "queued": [*self._queue, *self._deferred_auto.values()],
        }
        temp = self.registry_path.with_name(f".{self.registry_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            temp.replace(self.registry_path)
        finally:
            temp.unlink(missing_ok=True)

    def _publish_parent_status(self, job: dict[str, Any]) -> None:
        parent_id = job.get("parent_run_id")
        if not parent_id:
            return
        try:
            run = self.manager.get_run(parent_id)
            output = Path(run["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            status = {
                "state": job["state"],
                "protocol": "official_10_lap_race",
                "snapshot_timesteps": job.get("trigger_step"),
                "trigger_episode": job.get("trigger_episode"),
                "trigger": job.get("trigger"),
                "attempts_completed": len(job.get("attempts", [])),
                "attempts_required": self.ATTEMPTS,
                "latest_result": job.get("result"),
                "error": job.get("error"),
                "updated_utc": job.get("updated_utc"),
            }
            target = output / "evaluation_status.json"
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(json.dumps(status, indent=2), encoding="utf-8")
            temp.replace(target)
            history_path = output / "evaluation_history.jsonl"
            latest_id = job.get("evaluation_id")
            rows = []
            if history_path.exists():
                for line in history_path.read_text(encoding="utf-8").splitlines():
                    try:
                        value = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(value, dict) and value.get("evaluation_id") != latest_id:
                        rows.append(value)
            rows.append({**status, "evaluation_id": latest_id})
            history_path.write_text("".join(json.dumps(value) + "\n" for value in rows[-200:]), encoding="utf-8")
        except Exception:
            pass

    def submit(
        self,
        model_path: Path,
        *,
        source: str,
        parent_run_id: str | None = None,
        trigger: str | None = None,
        trigger_step: int | None = None,
        trigger_episode: int | None = None,
        trace_steps: int = 0,
        validation_mode: str | None = None,
    ) -> dict[str, Any]:
        if source not in {"automatic", "manual"}:
            raise ValueError("evaluation source must be automatic or manual")
        # Reject new queued evaluations at the hard deadline, not only when
        # their simulator run eventually reaches the manager.
        assert_campaign_allows("evaluation")
        model_path = Path(model_path).resolve()
        if not model_path.is_file() or model_path.suffix.lower() != ".zip":
            raise ValueError("evaluation checkpoint must be an existing .zip file")
        if isinstance(trace_steps, bool) or not 0 <= trace_steps <= 500:
            raise ValueError("trace_steps must be between 0 and 500")
        if validation_mode not in {None, "bc_diagnostic", "training_probe"}:
            raise ValueError("validation_mode must be bc_diagnostic or training_probe when provided")
        if validation_mode == "training_probe" and source == "automatic" and not parent_run_id:
            raise ValueError("automatic training_probe requires its parent training run")
        if validation_mode == "bc_diagnostic":
            verified = verify_bc_checkpoint(model_path)
            if verified is None:
                raise ValueError(
                    "BC diagnostic requires a checkpoint authenticated by its adjacent ready summary, canonical path, and SHA-256"
                )
        elif validation_mode == "training_probe":
            if source == "automatic":
                verified = verify_training_probe_checkpoint(model_path)
            else:
                # Manual probes are used for explicitly evaluating a completed
                # PPO update when its source training run had auto-evaluation
                # disabled. Require the generic immutable PPO lineage proof,
                # which binds the checkpoint to its authenticated BC ancestor.
                verified = verify_ppo_lineage_checkpoint(model_path)
            if verified is None:
                raise ValueError(
                    "training probe requires an authenticated PPO checkpoint with an approved BC parent"
                )
        settings = checkpoint_settings(model_path)
        digest = _sha256(model_path)
        job = {
            "evaluation_id": uuid.uuid4().hex[:16], "source": source,
            "parent_run_id": parent_run_id, "trigger": trigger,
            "trigger_step": trigger_step, "trigger_episode": trigger_episode,
            "model_path": str(model_path), "model_sha256": digest,
            "settings": settings, "trace_steps": trace_steps,
            "validation_mode": validation_mode,
            "attempts_required": 1 if validation_mode in {"bc_diagnostic", "training_probe"} else self.ATTEMPTS,
            "state": "queued", "attempts": [], "current_run_id": None,
            "created_utc": _utc_now(), "updated_utc": _utc_now(), "error": None,
        }
        with self._condition:
            if source == "automatic" and parent_run_id:
                if self._current and self._current.get("source") == "automatic" and self._current.get("parent_run_id") == parent_run_id:
                    self._deferred_auto[parent_run_id] = job
                    self._persist()
                    self._publish_parent_status(job)
                    return self._public_status(job)
                # Keep only the newest pending snapshot for a training run.
                for index, queued in enumerate(self._queue):
                    if queued.get("source") == "automatic" and queued.get("parent_run_id") == parent_run_id:
                        self._queue[index] = job
                        self._persist()
                        self._condition.notify_all()
                        self._publish_parent_status(job)
                        return self._public_status(job)
            self._queue.append(job)
            self._persist()
            self._condition.notify_all()
            self._publish_parent_status(job)
        return self._public_status(job)

    @staticmethod
    def _public_status(job: dict[str, Any] | None) -> dict[str, Any]:
        if not job:
            return {"state": "idle", "run_id": None, "evaluation": None, "error": None}
        return {
            "state": job["state"],
            "evaluation_id": job.get("evaluation_id"), "source": job.get("source"),
            "validation_mode": job.get("validation_mode"),
            "parent_run_id": job.get("parent_run_id"), "trigger": job.get("trigger"),
            "trigger_step": job.get("trigger_step"), "trigger_episode": job.get("trigger_episode"),
            "attempts_completed": len(job.get("attempts", [])), "attempts_required": job.get("attempts_required", OfficialEvaluationService.ATTEMPTS),
            "run_id": job.get("current_run_id") or job.get("last_run_id"),
            "model": job.get("model_path"), "evaluation": job.get("result"),
            "error": job.get("error"), "updated_utc": job.get("updated_utc"),
        }

    def status(self) -> dict[str, Any]:
        with self._condition:
            job = self._current or (self._queue[0] if self._queue else self._latest)
            result = self._public_status(job)
        run_id = result.get("run_id")
        if run_id:
            try:
                result["last_fleet"] = self.manager.telemetry(run_id).get("fleet")
                row = self.manager.get_run(run_id)
                result["simulator_backend"] = row.get("simulator_backend")
                result["simulator_connections"] = row.get("simulator_connections", [])
                result["log_path"] = row.get("log_path")
            except Exception:
                pass
        return result

    def champion(self) -> dict[str, Any] | None:
        with self._condition:
            return dict(self._best) if self._best else None

    def history(self) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(row) for row in reversed(self._history)]

    def stop(self) -> dict[str, Any]:
        with self._condition:
            job = self._current
            if job is None:
                if self._queue:
                    queued = self._queue.popleft()
                    queued["state"] = "stopped"
                    queued["error"] = "cancelled before evaluation started"
                    queued["updated_utc"] = _utc_now()
                    self._latest = queued
                    self._history.append({key: value for key, value in queued.items() if key != "settings"})
                    self._persist()
                return self.status()
            run_id = job.get("current_run_id")
        if run_id:
            try:
                self.manager.stop_run(run_id)
            except Exception as exc:
                job["error"] = str(exc)
        job["stop_requested"] = True
        return self.status()

    def close(self) -> None:
        self.stop()
        with self._condition:
            self._worker_stop = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._worker_stop:
                    self._condition.wait()
                if self._worker_stop:
                    return
                job = self._queue.popleft()
                self._current = job
                job["state"] = "running"
                job["updated_utc"] = _utc_now()
                self._persist()
            self._publish_parent_status(job)
            paused_training_runs: list[str] = []
            try:
                pause_training = getattr(self.manager, "pause_active_training_for_evaluation", None)
                if callable(pause_training):
                    paused_training_runs = pause_training()
                self._run_batch(job)
            except Exception as exc:
                if job.get("current_run_id"):
                    try:
                        self.manager.stop_run(str(job["current_run_id"]))
                    except Exception:
                        pass
                job["state"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if paused_training_runs:
                    try:
                        self.manager.resume_training_after_evaluation(paused_training_runs)
                    except Exception as exc:
                        job["resume_error"] = f"{type(exc).__name__}: {exc}"
                        if job.get("error"):
                            job["error"] += f"; training resume failed: {exc}"
                        else:
                            job["error"] = f"training resume failed: {exc}"
                job["current_run_id"] = None
                job["updated_utc"] = _utc_now()
                record = {key: value for key, value in job.items() if key not in {"settings", "stop_requested"}}
                with self._condition:
                    self._history.append(record)
                    self._latest = job
                    self._current = None
                    if self._deferred_auto:
                        self._queue.extend(self._deferred_auto.values())
                        self._deferred_auto.clear()
                    self._persist()
                    self._condition.notify_all()
                self._publish_parent_status(job)

    def _run_batch(self, job: dict[str, Any]) -> None:
        probe = job.get("validation_mode") == "training_probe"
        noncompetitive = job.get("validation_mode") in {"bc_diagnostic", "training_probe"}
        race_laps = 3 if probe else 10
        config = _evaluation_child_settings(
            job["settings"], job["model_path"], race_laps=race_laps,
        )
        scores: list[float] = []
        for attempt_index in range(1, self.ATTEMPTS + 1):
            attempts_required = int(job.get("attempts_required", self.ATTEMPTS))
            if attempt_index > attempts_required:
                break
            if job.get("stop_requested") or self._worker_stop:
                job["state"] = "stopped"
                break
            if job.get("validation_mode") == "bc_diagnostic":
                run_name = f"BC diagnostic evaluation {attempt_index}/1"
            elif probe:
                run_name = f"Training probe {attempt_index}/1"
            else:
                run_name = f"{job['source'].title()} evaluation {attempt_index}/3"
            row = self.manager.create_run(
                config, run_name,
                kind="evaluation", evaluation_trace_steps=job["trace_steps"],
                evaluation_options=(
                    {"validation_mode": job["validation_mode"], "lap_progress_timeout_s": 240,
                     "lap_progress_wall_timeout_s": 240, "frontier_stall_after_s": 5,
                     "frontier_path": str(ROOT / "competition" / "iros2026" / "tracks" / "competition" / "official_centerline.csv")}
                    if noncompetitive else _competitive_route_monitor_options()
                ),
            )
            job["current_run_id"] = row["run_id"]
            job["last_run_id"] = row["run_id"]
            job["updated_utc"] = _utc_now()
            self._publish_parent_status(job)
            deadline = time.monotonic() + 2100.0
            while time.monotonic() < deadline:
                if job.get("stop_requested") or self._worker_stop:
                    try:
                        self.manager.stop_run(row["run_id"])
                    finally:
                        job["state"] = "stopped"
                    break
                try:
                    self.manager.reconcile(row["run_id"])
                except Exception:
                    pass
                current = self.manager.get_run(row["run_id"])
                if current.get("state") in {"completed", "failed", "stopped"}:
                    break
                time.sleep(0.5)
            else:
                self.manager.stop_run(row["run_id"])
                current = self.manager.get_run(row["run_id"])
            output_dir = Path(current.get("output_dir", row.get("output_dir", "")))
            try:
                report = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
                attempt = (report.get("runs") or [{}])[0]
            except (OSError, ValueError, TypeError, IndexError):
                report, attempt = {}, {"score_status": "missing_result", "valid_for_comparison": False}
            attempt_record = {
                "attempt": attempt_index, "run_id": row["run_id"],
                "score_status": attempt.get("score_status", "incomplete"),
                "valid_for_comparison": attempt.get("valid_for_comparison") is True,
                "adjusted_race_time_s": attempt.get("adjusted_race_time_s"),
                "race_time_s": attempt.get("race_time_s"),
                "race_lap_times_s": attempt.get("race_lap_times_s", []),
                "warmup_lap_times_s": attempt.get("warmup_lap_times_s", []),
                "race_collisions": attempt.get("race_collisions"),
                "warmup_collisions": attempt.get("warmup_collisions"),
                "raw_collision_count_at_finish": attempt.get("raw_collision_count_at_finish"),
                "collision_count": _attempt_collision_count(attempt),
                "stop_reason": attempt.get("stop_reason"),
                "wall_duration_s": attempt.get("wall_duration_s"),
                "frontier_progress_m": attempt.get("frontier_progress_m"),
                "frontier_progress_fraction": attempt.get("frontier_progress_fraction"),
                "frontier_lateral_offset_m": attempt.get("frontier_lateral_offset_m"),
                "frontier_projection_valid": attempt.get("frontier_projection_valid"),
                "route_projection_valid": attempt.get("route_projection_valid"),
                "route_monitor": attempt.get("route_monitor"),
                "official_scene_sha256": attempt.get("official_scene_sha256"),
                "route_sha256": attempt.get("route_sha256"),
                "official_scene_path": attempt.get("official_scene_path"),
                "route_path": attempt.get("route_path"),
                "track_profile": attempt.get("track_profile"),
                "route_monitor_only": attempt.get("route_monitor_only"),
                "simulator_identity": attempt.get("simulator_identity"),
                "camera_sample_count": attempt.get("camera_sample_count"),
                "camera_fresh_sample_count": attempt.get("camera_fresh_sample_count"),
                "camera_stale_sample_count": attempt.get("camera_stale_sample_count"),
                "camera_missing_sample_count": attempt.get("camera_missing_sample_count"),
                "camera_freshness_unknown_count": attempt.get("camera_freshness_unknown_count"),
                "camera_age_s": attempt.get("camera_age_s"),
                "policy_inference_latency_s": attempt.get("policy_inference_latency_s"),
                "evaluation_report": str(output_dir / "evaluation.json") if report else None,
                "warmup_laps": int(attempt.get("warmup_laps_completed", len(attempt.get("warmup_lap_times_s", []))) or 0),
                "race_laps_completed": len(attempt.get("race_lap_times_s", [])),
                "race_laps_required": race_laps,
            }
            if noncompetitive:
                attempt_record["valid_for_comparison"] = False
                attempt_record["readiness_passed"] = (
                    attempt_record["stop_reason"] == "race_complete"
                    and attempt_record["warmup_laps"] >= 1
                    and attempt_record["race_laps_completed"] >= race_laps
                    and attempt_record["collision_count"] == 0
                    and attempt_record["frontier_projection_valid"] is True
                )
            job["attempts"].append(attempt_record)
            job["current_run_id"] = None
            job["updated_utc"] = _utc_now()
            self._publish_parent_status(job)
            if attempt_record["valid_for_comparison"] and isinstance(attempt_record["adjusted_race_time_s"], (int, float)):
                scores.append(float(attempt_record["adjusted_race_time_s"]))
            if job["state"] == "stopped":
                break

        attempts_required = int(job.get("attempts_required", self.ATTEMPTS))
        valid_batch = len(scores) == attempts_required and len(job["attempts"]) == attempts_required
        diagnostic = job.get("validation_mode") in {"bc_diagnostic", "training_probe"}
        protocol = "official_training_probe_1_plus_3" if probe else (
            "official_bc_diagnostic_1_plus_10" if job.get("validation_mode") == "bc_diagnostic"
            else "official_10_lap_race"
        )
        readiness_passed = diagnostic and len(job["attempts"]) == 1 and job["attempts"][0].get("readiness_passed") is True
        raw_times = [float(item["race_time_s"]) for item in job["attempts"]
                     if item.get("valid_for_comparison") and isinstance(item.get("race_time_s"), (int, float))]
        result = {
            "protocol": protocol,
            # Diagnostic runs are deliberately excluded from competitive
            # scores, but they still count as completed driving attempts.
            "completed_attempts": len(job["attempts"]) if diagnostic else len(scores),
            "competitive_attempts": len(scores), "attempts_required": attempts_required,
            "valid": valid_batch and not diagnostic, "readiness_passed": readiness_passed,
            "validation_mode": job.get("validation_mode"),
            "race_laps_required": race_laps,
            "competitive_score_valid": not diagnostic and valid_batch,
            "median_adjusted_race_time_s": median(scores) if valid_batch and not diagnostic else None,
            "best_adjusted_race_time_s": min(scores) if scores else None,
            "best_10_lap_time_s": median(raw_times) if len(raw_times) == self.ATTEMPTS else None,
            "laps_observed": min((len(item.get("race_lap_times_s", [])) for item in job["attempts"]), default=0),
            "collisions": sum(int(item.get("collision_count") or 0) for item in job["attempts"]),
            "failed_episodes": 0, "evaluation_runs": len(job["attempts"]),
            "runs": job["attempts"], "model_sha256": job["model_sha256"],
            "source": job["source"], "parent_run_id": job.get("parent_run_id"),
        }
        job["result"] = result
        if job["state"] != "stopped":
            job["state"] = "complete" if valid_batch or readiness_passed else "incomplete"
        if valid_batch and not diagnostic:
            self._consider_champion(job, result)

    def _consider_champion(self, job: dict[str, Any], result: dict[str, Any]) -> None:
        score = float(result["median_adjusted_race_time_s"])
        with self._condition:
            current = self._best
            if current and score >= float(current.get("median_adjusted_race_time_s", float("inf"))):
                return
            source = Path(job["model_path"])
            config = source.parent.parent / "config.json" if source.parent.name in {"checkpoints", "ckpt"} else source.parent / "config.json"
            self.champion_dir.mkdir(parents=True, exist_ok=True)
            temp_model = self.champion_dir / f".model.{uuid.uuid4().hex}.zip"
            temp_config = self.champion_dir / f".config.{uuid.uuid4().hex}.json"
            shutil.copy2(source, temp_model)
            shutil.copy2(config, temp_config)
            temp_model.replace(self.champion_dir / "model.zip")
            temp_config.replace(self.champion_dir / "config.json")
            self._best = {
                "model_path": "best_official_model/model.zip",
                "model_sha256": job["model_sha256"],
                "median_adjusted_race_time_s": score,
                "attempts": self.ATTEMPTS, "source": job["source"],
                "parent_run_id": job.get("parent_run_id"),
                "trigger_step": job.get("trigger_step"), "trigger_episode": job.get("trigger_episode"),
                "evaluation_id": job["evaluation_id"], "updated_utc": _utc_now(),
            }
            result["promoted_to_best"] = True
            self._persist()
