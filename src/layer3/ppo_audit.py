"""Finite-update audit and checkpointing for official PPO training."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from threading import Event
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from stable_baselines3.common.callbacks import BaseCallback


class OperatorStopCallback(BaseCallback):
    """Stop collection at the next environment step after an operator request."""

    def __init__(self, stop_requested: Event, *, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.stop_requested = stop_requested
        self.stop_reason: Optional[str] = None

    def _on_step(self) -> bool:
        if self.stop_requested.is_set():
            self.stop_reason = "operator_stop"
            return False
        return True


class PostOptimizerAuditCallback(BaseCallback):
    """Record PPO health only after SB3 completes an optimizer update.

    Checkpoints are written at the requested transition cadence, but only at a
    rollout boundary after the previous optimization pass has completed. A
    final post-update snapshot is written when training ends.
    """

    _METRIC_KEYS = (
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
    )

    def __init__(
        self,
        output_dir: Path,
        *,
        save_every_transitions: int,
        global_step_cadence: bool = False,
        lineage: Optional[dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_dir = Path(output_dir)
        self.save_every_transitions = max(1, int(save_every_transitions))
        self.global_step_cadence = bool(global_step_cadence)
        self.lineage = lineage
        self.audit_path = self.output_dir / "ppo_optimizer_audit.jsonl"
        self.last_update = 0
        self.last_saved_steps = 0
        self.failed_reason: Optional[str] = None
        self.lineage_warning: Optional[str] = None

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.last_update = int(getattr(self.model, "_n_updates", 0))
        self.last_saved_steps = int(self.model.num_timesteps)
        self.next_global_save_step = (
            (self.last_saved_steps // self.save_every_transitions) + 1
        ) * self.save_every_transitions

    @staticmethod
    def _all_tensor_values_finite(value: Any) -> bool:
        import torch

        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all().item())
        if isinstance(value, dict):
            return all(PostOptimizerAuditCallback._all_tensor_values_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(PostOptimizerAuditCallback._all_tensor_values_finite(item) for item in value)
        if isinstance(value, (float, int)):
            return math.isfinite(float(value))
        return True

    def _audit_latest_update(self, *, force_save: bool = False) -> None:
        import torch

        update = int(getattr(self.model, "_n_updates", 0))
        if update <= self.last_update:
            if force_save and update > 0 and self.last_saved_steps < int(self.model.num_timesteps):
                self._save_checkpoint(update)
            return
        metrics = {}
        for key in self._METRIC_KEYS:
            value = self.model.logger.name_to_value.get(key)
            if value is None:
                metrics[key] = None
                continue
            numeric = float(value)
            metrics[key] = numeric
        metric_values_finite = all(
            value is None or math.isfinite(value) for value in metrics.values()
        )
        gradients = [parameter.grad for parameter in self.model.policy.parameters() if parameter.grad is not None]
        gradients_finite = (
            all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
            if gradients else None
        )
        parameters_finite = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in self.model.policy.parameters()
        )
        optimizer_finite = self._all_tensor_values_finite(self.model.policy.optimizer.state_dict())
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "num_timesteps": int(self.model.num_timesteps),
            "optimizer_updates": update,
            "metrics": metrics,
            "metrics_finite": metric_values_finite,
            "gradient_tensor_count": len(gradients),
            "gradients_finite": gradients_finite,
            "parameters_finite": parameters_finite,
            "optimizer_state_finite": optimizer_finite,
        }
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if not metric_values_finite or gradients_finite is False or not parameters_finite or not optimizer_finite:
            self.failed_reason = f"non-finite PPO training state after optimizer update {update}"
            if self.verbose:
                print(self.failed_reason, flush=True)
        current_steps = int(self.model.num_timesteps)
        if self.global_step_cadence:
            due = current_steps >= self.next_global_save_step
        else:
            due = current_steps - self.last_saved_steps >= self.save_every_transitions
        if due or force_save:
            self._save_checkpoint(update)
            if due and self.global_step_cadence:
                while self.next_global_save_step <= current_steps:
                    self.next_global_save_step += self.save_every_transitions
        self.last_update = update

    def ensure_post_optimizer_checkpoint(self) -> Path:
        """Flush a checkpoint for the latest completed optimizer update."""
        update = int(getattr(self.model, "_n_updates", 0))
        if update <= 0:
            raise RuntimeError("cannot pause before the first completed PPO optimizer update")
        self._audit_latest_update(force_save=True)
        target = self.output_dir / "checkpoints" / (
            f"post_optimizer_update_{update:08d}_steps_{int(self.model.num_timesteps):012d}.zip"
        )
        if not target.is_file():
            raise RuntimeError(f"post-optimizer pause checkpoint was not persisted: {target}")
        return target

    def _save_checkpoint(self, update: int) -> Path:
        steps = int(self.model.num_timesteps)
        target = self.output_dir / "checkpoints" / f"post_optimizer_update_{update:08d}_steps_{steps:012d}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target.with_suffix("")))
        if self.lineage is not None:
            from src.layer3.bc_artifact import write_ppo_lineage_sidecar

            try:
                write_ppo_lineage_sidecar(
                    target,
                    parent_lineage=self.lineage["parent_lineage"],
                    effective_config=self.lineage["effective_config"],
                    optimizer_updates=update,
                    num_timesteps=steps,
                    training_context=self.lineage["training_context"],
                    run_id=self.lineage["run_id"],
                )
                self.lineage_warning = None
            except (OSError, ValueError) as exc:
                self.lineage_warning = f"checkpoint saved without authenticated PPO lineage: {exc}"
                if self.verbose:
                    print(self.lineage_warning, flush=True)
        self.last_saved_steps = steps
        return target

    def _on_rollout_start(self) -> None:
        self._audit_latest_update()

    def _on_step(self) -> bool:
        return self.failed_reason is None

    def _on_training_end(self) -> None:
        self._audit_latest_update(force_save=True)


class CooperativePauseInterrupted(RuntimeError):
    """The operator stopped training while it was parked at a safe boundary."""


class CooperativePauseCallback(BaseCallback):
    """Park PPO between optimizer updates while preserving the live ROS worker.

    The callback is placed after ``PostOptimizerAuditCallback``. SB3 invokes
    ``_on_rollout_start`` after the preceding ``model.train()`` and before the
    next policy action, so the previous rollout is already optimized and the
    next rollout buffer is empty.
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        stop_requested: Event,
        audit_callback: PostOptimizerAuditCallback,
        *,
        pause_boundary_timeout_s: float = 240.0,
        resume_request_wait_timeout_s: float = 9000.0,
        resume_freshness_timeout_s: float = 120.0,
        poll_s: float = 0.2,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.stop_requested = stop_requested
        self.audit_callback = audit_callback
        self.pause_boundary_timeout_s = float(pause_boundary_timeout_s)
        self.resume_request_wait_timeout_s = float(resume_request_wait_timeout_s)
        self.resume_freshness_timeout_s = float(resume_freshness_timeout_s)
        self.poll_s = float(poll_s)
        if (not self.run_id or self.pause_boundary_timeout_s <= 0
                or self.resume_request_wait_timeout_s <= 0 or self.resume_freshness_timeout_s <= 0
                or self.poll_s <= 0):
            raise ValueError("cooperative pause requires run id and positive bounds")
        self.pause_request_path = self.run_dir / "evaluation_pause.request.json"
        self.pause_ack_path = self.run_dir / "evaluation_pause.ack.json"
        self.resume_request_path = self.run_dir / "evaluation_resume.request.json"
        self.resume_ack_path = self.run_dir / "evaluation_resume.ack.json"

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    @staticmethod
    def _read_request(path: Path, run_id: str) -> Optional[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (not isinstance(payload, dict) or payload.get("run_id") != run_id
                or not isinstance(payload.get("request_id"), str)
                or not payload["request_id"].strip()):
            return None
        return payload

    def _env_method(self, method: str, *args: Any) -> Any:
        if int(self.training_env.num_envs) != 1:
            raise RuntimeError("cooperative evaluation pause requires a single environment")
        results = self.training_env.env_method(method, *args)
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError(f"environment did not return one {method} result")
        return results[0]

    @staticmethod
    def _transport_is_fresh(status: Any) -> bool:
        if not isinstance(status, dict) or status.get("spin_state") != "running":
            return False
        if status.get("topic_fresh") is not True:
            return False
        required = status.get("required_topics")
        ages = status.get("topic_age_s")
        if not isinstance(required, list) or not required or not isinstance(ages, dict):
            return False
        for name in required:
            age = ages.get(name)
            if (not isinstance(age, (int, float)) or not math.isfinite(float(age))
                    or float(age) < 0 or float(age) > (0.5 if name == "camera" else 5.0)):
                return False
        return "camera" in required

    def _ack(self, path: Path, request_id: str, status: str, **details: Any) -> None:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "request_id": request_id,
            "status": status,
            "num_timesteps": int(getattr(self.model, "num_timesteps", 0)),
            "optimizer_updates": int(getattr(self.model, "_n_updates", 0)),
            "recorded_at_unix_s": time.time(),
            **details,
        }
        self._atomic_json(path, payload)

    def _wait_for_resume(self, request_id: str) -> dict[str, Any]:
        # The evaluator may need several minutes to complete its bounded laps
        # and restart the simulator. Its request wait has its own deadline;
        # sensor freshness gets a separate window beginning only after the
        # matching resume request arrives.
        request_deadline = time.monotonic() + self.resume_request_wait_timeout_s
        while True:
            if self.stop_requested.is_set():
                self._ack(self.resume_ack_path, request_id, "cancelled_by_stop")
                raise CooperativePauseInterrupted("operator stop interrupted cooperative pause")
            request = self._read_request(self.resume_request_path, self.run_id)
            if request is not None and request.get("request_id") == request_id:
                break
            if time.monotonic() >= request_deadline:
                self._ack(self.resume_ack_path, request_id, "timeout",
                          timeout_phase="waiting_for_resume_request")
                raise TimeoutError("cooperative pause timed out waiting for evaluator resume request")
            time.sleep(self.poll_s)

        freshness_deadline = time.monotonic() + self.resume_freshness_timeout_s
        while True:
            if self.stop_requested.is_set():
                self._ack(self.resume_ack_path, request_id, "cancelled_by_stop")
                raise CooperativePauseInterrupted("operator stop interrupted cooperative pause")
            transport = self._env_method("get_transport_status")
            if self._transport_is_fresh(transport):
                return transport
            if time.monotonic() >= freshness_deadline:
                self._ack(self.resume_ack_path, request_id, "timeout",
                          timeout_phase="waiting_for_fresh_transport",
                          resume_transport_status=transport)
                raise TimeoutError("resumed official telemetry did not become fresh before its deadline")
            time.sleep(self.poll_s)

    def _on_rollout_start(self) -> None:
        request = self._read_request(self.pause_request_path, self.run_id)
        if request is None:
            return
        request_id = request["request_id"]
        requested_at = request.get("requested_at_unix_s")
        if (isinstance(requested_at, bool) or not isinstance(requested_at, (int, float))
                or not math.isfinite(float(requested_at))
                or time.time() - float(requested_at) > self.pause_boundary_timeout_s
                or float(requested_at) > time.time() + 5.0):
            self._ack(self.pause_ack_path, request_id, "timeout",
                      reason="invalid_or_expired_pause_request_timestamp")
            raise TimeoutError("cooperative pause request expired before a safe optimizer boundary")
        if self.stop_requested.is_set():
            self._ack(self.pause_ack_path, request_id, "cancelled_by_stop")
            raise CooperativePauseInterrupted("operator stop preceded cooperative pause")

        neutral = self._env_method("request_pause_neutral")
        if (not isinstance(neutral, dict) or neutral.get("published") is not True
                or neutral.get("command") != {"throttle": 0.0, "steering": 0.0}
                or not self._transport_is_fresh(neutral)):
            self._ack(self.pause_ack_path, request_id, "neutral_command_failed",
                      neutral_command=neutral)
            raise RuntimeError("cooperative pause could not verify a neutral actuator publish")
        checkpoint = self.audit_callback.ensure_post_optimizer_checkpoint()
        pause_payload = {
            "neutral_command": neutral,
            "transport_status": {
                key: neutral.get(key) for key in (
                    "step_counter", "spin_state", "topic_age_s", "topic_fresh", "required_topics",
                )
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "post_optimizer_update": True,
            },
        }
        self._ack(self.pause_ack_path, request_id, "paused", **pause_payload)
        transport = self._wait_for_resume(request_id)

        if self.stop_requested.is_set():
            self._ack(self.resume_ack_path, request_id, "cancelled_by_stop")
            raise CooperativePauseInterrupted("operator stop interrupted cooperative pause before reset")

        # The evaluator retired/restarted the simulator. End the old episode as
        # an infrastructure discontinuity and seed the next PPO rollout from a
        # fresh reset observation, without attributing a policy failure/reward.
        # The reset path can wait for scan and IPS messages. Give the concrete
        # single-process training env the same stop Event so an operator stop
        # can interrupt those waits and still deassert simulator reset safely.
        self._env_method("set_pause_cancel_event", self.stop_requested)
        try:
            fresh_observation = self.training_env.reset()
        except Exception as exc:
            if self.stop_requested.is_set():
                self._ack(
                    self.resume_ack_path,
                    request_id,
                    "cancelled_by_stop",
                    reset_completed=False,
                    reset_error=f"{type(exc).__name__}: {exc}",
                    transport_status=transport,
                )
                raise CooperativePauseInterrupted(
                    "operator stop interrupted cooperative pause during reset"
                ) from exc
            raise
        if self.stop_requested.is_set():
            self._ack(self.resume_ack_path, request_id, "cancelled_by_stop",
                      reset_completed=True, transport_status=transport)
            raise CooperativePauseInterrupted("operator stop interrupted cooperative pause during reset")
        self.model._last_obs = fresh_observation
        import numpy as np

        self.model._last_episode_starts = np.ones(int(self.training_env.num_envs), dtype=bool)
        self._ack(
            self.resume_ack_path,
            request_id,
            "resumed",
            reset_completed=True,
            transport_status=transport,
            episode_boundary={"kind": "infrastructure_simulator_restart", "policy_failure": False},
        )

    def _on_step(self) -> bool:
        return True
