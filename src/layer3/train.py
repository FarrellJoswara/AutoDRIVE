"""PPO training entrypoint for Layer 3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Practical first-policy PPO defaults for continuous lidar control.
# Short/medium runs should get frequent updates + some exploration.
_PPO_LR = 3e-4
_PPO_N_STEPS = 1024
_PPO_BATCH_SIZE = 64
_PPO_N_EPOCHS = 8
_PPO_GAMMA = 0.99
_PPO_GAE_LAMBDA = 0.95
_PPO_CLIP_RANGE = 0.2
_PPO_ENT_COEF = 0.01
_PPO_VF_COEF = 0.5
_PPO_MAX_GRAD_NORM = 0.5


def aggregate_evaluation_attempts(
    attempts: list[Dict[str, Any]], timesteps: int, selection_metric: str,
) -> Dict[str, Any]:
    """Summarize repeated runs; the median is the checkpoint-selection signal."""
    if not attempts:
        raise ValueError("at least one evaluation attempt is required")
    if selection_metric not in {"frontier_speed", "reward_per_simulated_second", "total_reward", "ten_lap_time"}:
        raise ValueError(f"unsupported evaluation metric: {selection_metric}")
    metric_fields = (
        "simulated_seconds", "frontier_distance_m", "frontier_speed_mps",
        "reward_per_simulated_second", "total_reward", "laps_observed",
    )
    aggregate: Dict[str, Any] = {
        "timesteps": int(timesteps),
        "evaluation_runs": len(attempts),
        "successful_attempts": sum(int(run.get("laps_observed", 0)) >= 10 for run in attempts),
        "attempts": attempts,
    }
    for field in metric_fields:
        values = [float(run[field]) for run in attempts if field in run and math.isfinite(float(run[field]))]
        if values:
            aggregate[field] = statistics.median(values)
            aggregate[f"{field}_mean"] = statistics.mean(values)
            aggregate[f"{field}_best"] = min(values) if field == "simulated_seconds" else max(values)
    for field in ("collisions", "failed_episodes"):
        aggregate[field] = sum(int(run.get(field, 0)) for run in attempts)
    lap_count = max((len(run.get("lap_times_s", [])) for run in attempts), default=0)
    aggregate["lap_times_s"] = []
    for lap_index in range(lap_count):
        values = [
            float(run["lap_times_s"][lap_index])
            for run in attempts
            if len(run.get("lap_times_s", [])) > lap_index
            and math.isfinite(float(run["lap_times_s"][lap_index]))
        ]
        aggregate["lap_times_s"].append(statistics.median(values))
    ten_lap_times = [
        float(run["best_10_lap_time_s"])
        for run in attempts
        if run.get("best_10_lap_time_s") is not None
        and math.isfinite(float(run["best_10_lap_time_s"]))
    ]
    aggregate["best_10_lap_time_s"] = statistics.median(ten_lap_times) if ten_lap_times else None
    aggregate["collision_rate"] = sum(int(run.get("collisions", 0) > 0) for run in attempts) / len(attempts)
    aggregate["termination_reasons"] = {}
    for run in attempts:
        for reason, count in run.get("termination_reasons", {}).items():
            aggregate["termination_reasons"][reason] = aggregate["termination_reasons"].get(reason, 0) + int(count)
    if selection_metric == "ten_lap_time":
        aggregate["selection_score"] = (
            statistics.median(ten_lap_times) if ten_lap_times else None
        )
        aggregate["selection_score_mean"] = (
            statistics.mean(ten_lap_times) if ten_lap_times else None
        )
        aggregate["selection_score_best"] = min(ten_lap_times) if ten_lap_times else None
        return aggregate

    score_field = {
        "frontier_speed": "frontier_speed_mps",
        "reward_per_simulated_second": "reward_per_simulated_second",
        "total_reward": "total_reward",
    }[selection_metric]
    scores = [float(run[score_field]) for run in attempts]
    aggregate["selection_score"] = statistics.median(scores)
    aggregate["selection_score_mean"] = statistics.mean(scores)
    aggregate["selection_score_best"] = max(scores)
    return aggregate


def evaluation_improved(
    *,
    metric: str,
    score: Optional[float],
    successful_attempts: int,
    best_score: Optional[float],
    best_successful_attempts: Optional[int],
    min_improvement_pct: float,
) -> bool:
    """Compare candidates, prioritizing ten-lap completion count before time."""
    if metric == "ten_lap_time":
        # A snapshot with no completed 10-lap attempt has no valid lap-time
        # score and must never become the initial champion. In particular,
        # do not promote an all-crash first evaluation or reduce exploration.
        if successful_attempts <= 0 or score is None:
            return False
        if best_successful_attempts is None:
            return True
        if successful_attempts != best_successful_attempts:
            return successful_attempts > best_successful_attempts
        if best_score is None:
            return True
        pct = (best_score - score) / max(abs(best_score), 1e-6) * 100.0
        return score < best_score and pct >= min_improvement_pct
    if score is None:
        return False
    if best_score is None:
        return True
    pct = (score - best_score) / max(abs(best_score), 1e-6) * 100.0
    return score > best_score and pct >= min_improvement_pct


def _device_arg(value: str) -> str:
    v = value.lower().strip()
    if v not in {"auto", "cpu", "cuda"}:
        raise argparse.ArgumentTypeError("device must be auto|cpu|cuda")
    return v


def _resolve_device(requested: str) -> str:
    """Prefer CUDA. Fall back to CPU only when CUDA truly unavailable."""
    import torch

    cuda_ok = torch.cuda.is_available()
    if requested == "cpu":
        print("WARN: --device cpu forced; GPU preferred for PPO updates")
        return "cpu"
    if requested in {"auto", "cuda"}:
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            print(f"device: cuda ({name})")
            return "cuda"
        if requested == "cuda":
            raise RuntimeError(
                "CUDA was requested (--device cuda) but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch (see requirements / LAYER3.md). "
                "Refusing to silently train on CPU."
            )
        print(
            "WARN: CUDA unavailable — falling back to CPU (last resort). "
            "Install torch+cu12x for the RTX GPU."
        )
        return "cpu"
    return requested


def make_model(
    vec_env, *, device: str, seed: int, tensorboard_log: Optional[str],
    learning_rate: float = _PPO_LR, n_epochs: int = _PPO_N_EPOCHS,
    n_steps: int = _PPO_N_STEPS,
    gamma: float = _PPO_GAMMA,
    gae_lambda: float = _PPO_GAE_LAMBDA,
    policy_architecture: str = "lidar_camera_cnn",
):
    from stable_baselines3 import PPO

    from src.layer3.extractors import policy_kwargs_for_architecture

    policy_kwargs = policy_kwargs_for_architecture(policy_architecture)
    return PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=_PPO_BATCH_SIZE,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=_PPO_CLIP_RANGE,
        ent_coef=_PPO_ENT_COEF,
        vf_coef=_PPO_VF_COEF,
        max_grad_norm=_PPO_MAX_GRAD_NORM,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
    )


class RunStopCallback(BaseCallback):
    """Enforce only the optional wall-clock limit; evaluation owns plateau stop."""

    def __init__(self, *, max_duration_seconds: float = 0.0) -> None:
        super().__init__(verbose=0)
        self.max_duration_seconds = max(0.0, float(max_duration_seconds))
        self._started_at = 0.0
        self.stop_reason: Optional[str] = None

    def _on_training_start(self) -> None:
        self._started_at = time.monotonic()

    def _on_step(self) -> bool:
        if self.max_duration_seconds > 0 and time.monotonic() - self._started_at >= self.max_duration_seconds:
            self.stop_reason = "max_duration"
            return False
        return True


class RequestedStopCallback(BaseCallback):
    """Checkpoint and stop at the next completed step after SIGTERM."""

    def __init__(self, requested: threading.Event, recovery_path: Path) -> None:
        super().__init__(verbose=0)
        self.requested = requested
        self.recovery_path = Path(recovery_path)
        self.stop_reason: Optional[str] = None

    def _on_step(self) -> bool:
        if self.requested.is_set():
            self.stop_reason = "operator_stop"
            try:
                self.recovery_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.recovery_path.with_suffix("")))
                print(
                    f"SIGTERM stop requested; saved recovery policy to {self.recovery_path}",
                    flush=True,
                )
            except Exception as exc:
                print(f"SIGTERM recovery checkpoint failed: {exc}", flush=True)
            return False
        return True


class EvaluationCallback(BaseCallback):
    """Asynchronously evaluate snapshots until failure, frontier stall, or ten laps."""

    def __init__(
        self, evaluation_env, *, out_dir: Path, every_timesteps: int,
        frame_skip: int, selection_metric: str, plateau_min_timesteps: int,
        plateau_patience: int, min_improvement_pct: float,
        runs_per_snapshot: int,
        exploration_std_min: float, exploration_std_max: float,
        exploration_improvement_scale: float, exploration_plateau_scale: float,
        live_pose_publisher: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.evaluation_env = evaluation_env
        self.live_pose_publisher = live_pose_publisher
        self.out_dir = Path(out_dir)
        self.every_timesteps = max(1, int(every_timesteps))
        self.frame_skip = max(1, int(frame_skip))
        self.runs_per_snapshot = max(1, int(runs_per_snapshot))
        if selection_metric not in {"frontier_speed", "reward_per_simulated_second", "total_reward", "ten_lap_time"}:
            raise ValueError(f"unsupported evaluation metric: {selection_metric}")
        self.selection_metric = selection_metric
        self.plateau_min_timesteps = max(0, int(plateau_min_timesteps))
        self.plateau_patience = max(1, int(plateau_patience))
        self.min_improvement_pct = max(0.0, float(min_improvement_pct))
        self.std_min = float(exploration_std_min)
        self.std_max = float(exploration_std_max)
        self.improvement_scale = float(exploration_improvement_scale)
        self.plateau_scale = float(exploration_plateau_scale)
        self.next_eval = self.every_timesteps
        self.best_score: Optional[float] = None
        self.best_successful_attempts: Optional[int] = None
        self.stale_evaluations = 0
        self.stop_reason: Optional[str] = None
        self.history_path = self.out_dir / "evaluation_history.jsonl"
        self.status_path = self.out_dir / "evaluation_status.json"
        self.latest: Dict[str, Any] = {}
        self._last_adaptation = "initial"
        self._target_log_std: Optional[float] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-evaluation")
        self._future: Optional[Future] = None
        self._candidate_path: Optional[Path] = None
        self._evaluation_requested = False
        self._evaluation_index = 0
        self._executor_shutdown = False
        self._live_pose: Optional[Dict[str, Any]] = None
        self._active_snapshot_steps: Optional[int] = None
        self._last_live_status_write = 0.0
        self._last_live_pose_publish = 0.0
        self._live_result: Optional[Dict[str, Any]] = None

    def _write_status(self, state: str, snapshot_timesteps: Optional[int] = None) -> None:
        latest = self.latest or None
        status = {
            "state": state,
            "snapshot_timesteps": snapshot_timesteps,
            "selection_metric": self.selection_metric,
            "evaluation_runs_per_snapshot": self.runs_per_snapshot,
            "best_selection_score": self.best_score,
            "stale_evaluations": self.stale_evaluations,
            "plateau_patience": self.plateau_patience,
            "latest_result": latest,
            "live_result": self._live_result,
            "evaluator_car": self._live_pose,
            "stop_reason": self.stop_reason,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    def _on_training_start(self) -> None:
        self._write_status("waiting")

    def _policy_std(self) -> Any:
        return self.model.policy.log_std

    def _on_rollout_start(self) -> None:
        import torch
        with torch.no_grad():
            log_std = self._policy_std()
            if self._target_log_std is None:
                self._target_log_std = float(log_std.detach().mean().cpu())
            self._target_log_std = min(
                math.log(self.std_max),
                max(math.log(self.std_min), self._target_log_std),
            )
            log_std.fill_(self._target_log_std)

    def _adjust_exploration(self, scale: float) -> None:
        import torch
        if self._target_log_std is None:
            self._target_log_std = float(self._policy_std().detach().mean().cpu())
        self._target_log_std += math.log(scale)
        self._target_log_std = min(
            math.log(self.std_max),
            max(math.log(self.std_min), self._target_log_std),
        )
        with torch.no_grad():
            self._policy_std().fill_(self._target_log_std)

    def _evaluate_snapshot(self, policy: Any, timesteps: int, attempt_index: int) -> Dict[str, Any]:
        from src.layer3.evaluate import evaluate_until_episode_end

        class SnapshotModel:
            def predict(self, obs, *, deterministic=True):
                return policy.predict(obs, deterministic=deterministic)

        return evaluate_until_episode_end(
            SnapshotModel(),
            self.evaluation_env,
            frame_skip=self.frame_skip,
            timesteps=timesteps,
            lap_target=10,
            on_step=lambda info: self._on_evaluation_step(info, attempt_index),
        )

    def _evaluate_snapshot_batch(self, policy: Any, timesteps: int) -> Dict[str, Any]:
        attempts = [
            self._evaluate_snapshot(policy, timesteps, index)
            for index in range(1, self.runs_per_snapshot + 1)
        ]
        return aggregate_evaluation_attempts(attempts, timesteps, self.selection_metric)

    def _on_evaluation_step(self, info: Dict[str, Any], attempt_index: int) -> None:
        live_result = info.get("_evaluation_metrics")
        if isinstance(live_result, dict):
            self._live_result = {**{
                key: live_result[key]
                for key in (
                    "frontier_distance_m",
                    "frontier_speed_mps",
                    "simulated_seconds",
                    "laps_observed",
                    "lap_times_s",
                    "lap_diagnostics",
                    "best_10_lap_time_s",
                    "reward_per_simulated_second",
                    "total_reward",
                    "collisions",
                )
                if key in live_result
            }, "attempt_index": attempt_index, "attempt_count": self.runs_per_snapshot}

        # Publish evaluator metrics independently of pose telemetry. Position
        # or yaw can be absent on a valid simulator step; that must not freeze
        # the live score in evaluation_status.json.
        now = time.monotonic()
        if now - self._last_live_status_write >= 0.25:
            self._last_live_status_write = now
            self._write_status("running", snapshot_timesteps=self._active_snapshot_steps)

        position = info.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return
        try:
            pose = [float(position[0]), float(position[2])]
            yaw = float(info["yaw"])
            speed = float(info.get("true_speed", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            return
        if not all(math.isfinite(value) for value in (*pose, yaw, speed)):
            return
        self._live_pose = {"pose": pose, "yaw": yaw, "speed": speed}
        now = time.monotonic()
        if self.live_pose_publisher is not None and now - self._last_live_pose_publish >= 1.0 / 15.0:
            self._last_live_pose_publish = now
            self.live_pose_publisher({
                **self._live_pose,
                "run_id": self.out_dir.name,
                "snapshot_timesteps": self._active_snapshot_steps,
            })

    def _launch_evaluation(self) -> None:
        if self._future is not None or self._executor_shutdown:
            return
        # Snapshot at the rollout boundary, when PPO is not updating weights.
        # Inference uses a CPU copy, so the evaluator never reads weights being
        # changed by the training process.
        snapshot = copy.deepcopy(self.model.policy).to("cpu")
        snapshot.set_training_mode(False)
        evaluation_timestep = int(self.num_timesteps)
        self._active_snapshot_steps = evaluation_timestep
        self._live_pose = None
        self._live_result = None
        self._last_live_status_write = 0.0
        self._last_live_pose_publish = 0.0
        self._evaluation_index += 1
        candidate_base = self.out_dir / f"evaluation_candidate_{self._evaluation_index}"
        self.model.save(str(candidate_base))
        self._candidate_path = Path(f"{candidate_base}.zip")
        self._write_status("running", snapshot_timesteps=evaluation_timestep)
        self._future = self._executor.submit(
            self._evaluate_snapshot_batch, snapshot, evaluation_timestep,
        )
        print(
            f"evaluation started: snapshot_steps={evaluation_timestep} "
            "(training continues in parallel)",
            flush=True,
        )

    def _collect_evaluation(self) -> None:
        if self._future is None or not self._future.done():
            return
        future, candidate = self._future, self._candidate_path
        self._future = None
        self._candidate_path = None
        try:
            result = future.result()
        except Exception as exc:
            if candidate is not None:
                candidate.unlink(missing_ok=True)
            raise RuntimeError("deterministic policy evaluation failed") from exc

        score_value = result.get("selection_score")
        score = float(score_value) if score_value is not None else None
        successful_attempts = int(result.get("successful_attempts", 0))
        improved = evaluation_improved(
            metric=self.selection_metric,
            score=score,
            successful_attempts=successful_attempts,
            best_score=self.best_score,
            best_successful_attempts=self.best_successful_attempts,
            min_improvement_pct=self.min_improvement_pct,
        )
        if improved:
            self.best_score = score
            self.best_successful_attempts = successful_attempts
            self.stale_evaluations = 0
            best_path = self.out_dir / "best_evaluated_model.zip"
            if candidate is None or not candidate.is_file():
                raise RuntimeError("evaluation snapshot checkpoint is missing")
            os.replace(candidate, best_path)
            self._adjust_exploration(self.improvement_scale)
            self._last_adaptation = "reduced_after_improvement"
        else:
            if candidate is not None:
                candidate.unlink(missing_ok=True)
            if self.num_timesteps >= self.plateau_min_timesteps:
                self.stale_evaluations += 1
            if self.stale_evaluations > 0 and self.stale_evaluations % 3 == 0:
                self._adjust_exploration(self.plateau_scale)
                self._last_adaptation = "increased_after_three_stale_evaluations"
        result.update({
            "selection_score": score,
            "selection_score_median": score,
            "selection_metric": self.selection_metric,
            "best_selection_score": self.best_score,
            "improved": improved,
            "stale_evaluations": self.stale_evaluations,
            "exploration_std": self._policy_std().detach().exp().cpu().tolist(),
            "exploration_adaptation": self._last_adaptation,
        })
        self.latest = result
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        print(
            "evaluation: "
            f"snapshot_steps={result['timesteps']} frontier={result['frontier_distance_m']:.1f}m "
            f"pace={result['frontier_speed_mps']:.3f}m/s "
            f"selection={self.selection_metric}:{score if score is not None else 'n/a'} "
            f"crashes={result['collisions']} laps={result['laps_observed']} "
            f"stale={self.stale_evaluations}", flush=True,
        )
        if (self.num_timesteps >= self.plateau_min_timesteps
                and self.stale_evaluations >= self.plateau_patience):
            self.stop_reason = "evaluation_plateau"
            self.latest["stop_reason"] = self.stop_reason
        self._live_pose = None
        self._live_result = None
        self._active_snapshot_steps = None
        self._write_status("complete", snapshot_timesteps=int(result["timesteps"]))

    def _on_rollout_end(self) -> None:
        if self.num_timesteps >= self.next_eval:
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.every_timesteps
            self._evaluation_requested = True

        self._collect_evaluation()
        if self.stop_reason is None and self._evaluation_requested and self._future is None:
            self._evaluation_requested = False
            self._launch_evaluation()

    def _on_step(self) -> bool:
        return self.stop_reason is None

    def _on_training_end(self) -> None:
        # A finite training stop can occur while the separate evaluator is still
        # driving. Join it before train() closes the evaluation simulator.
        if not self._executor_shutdown:
            self._executor.shutdown(wait=True)
            self._executor_shutdown = True
        self._collect_evaluation()


class SimulatorPauseCallback(BaseCallback):
    """Freeze Unity between PPO rollouts so cached actions cannot keep driving."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._paused = False

    def _on_training_start(self) -> None:
        # Fail before collecting a rollout if this player was not rebuilt with
        # the lockstep gate; otherwise stale controls could still drive physics.
        self.training_env.env_method("set_simulation_paused", True)
        self._paused = True
        self.training_env.env_method("set_simulation_paused", False)
        self._paused = False

    def _on_rollout_end(self) -> None:
        self.training_env.env_method("set_simulation_paused", True)
        self._paused = True

    def _on_rollout_start(self) -> None:
        if self._paused:
            self.training_env.env_method("set_simulation_paused", False)
            self._paused = False

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # SB3 finishes on a rollout boundary, where Unity is paused for PPO
        # optimization. Explicitly resume even when this callback's local flag
        # was lost or the final stop callback ended the rollout early.
        self.training_env.env_method("resume_simulation")
        self._paused = False


def train(
    *,
    n_envs: int = 1,
    timesteps: int = 0,
    out_dir: Path,
    seed: int = 0,
    device: str = "auto",
    resume: Optional[Path] = None,
    laps_per_episode: int = 10,
    env_kwargs: Optional[Dict[str, Any]] = None,
    max_duration_seconds: float = 0.0,
    plateau_min_timesteps: int = 100_000,
    plateau_patience: int = 5,
    plateau_min_improvement_pct: float = 1.0,
    evaluation_every_timesteps: int = 50_000,
    evaluation_runs_per_snapshot: int = 3,
    evaluation_metric: str = "frontier_speed",
    exploration_std_min: float = 0.2,
    exploration_std_max: float = 0.8,
    exploration_improvement_scale: float = 0.9,
    exploration_plateau_scale: float = 1.1,
    learning_rate: float = _PPO_LR,
    n_epochs: int = _PPO_N_EPOCHS,
    n_steps: int = _PPO_N_STEPS,
    gamma: float = _PPO_GAMMA,
    gae_lambda: float = _PPO_GAE_LAMBDA,
    policy_architecture: str = "lidar_cnn",
) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from src.layer3.envs import make_vec_env

    env_kwargs = dict(env_kwargs or {})
    observation_profile = env_kwargs.get("observation_profile")
    expected_architecture = {
        "official_sensors_history": "temporal_lidar_cnn",
        "official_sensors_camera": "lidar_camera_cnn",
        "simulator_camera": "lidar_camera_cnn",
    }.get(observation_profile, "lidar_cnn")
    if expected_architecture != policy_architecture:
        raise ValueError(
            f"observation profile {observation_profile!r} requires "
            f"policy architecture {expected_architecture!r}, got {policy_architecture!r}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    tb_dir = out_dir / "tb"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    print(
        f"train: n_envs={n_envs} timesteps={'unbounded' if timesteps == 0 else timesteps} "
        f"device={resolved_device} out={out_dir}"
    )
    print(
        f"ppo: lr={learning_rate} n_steps={n_steps} batch={_PPO_BATCH_SIZE} "
        f"n_epochs={n_epochs} gamma={gamma} gae_lambda={gae_lambda} ent_coef={_PPO_ENT_COEF} "
        f"clip={_PPO_CLIP_RANGE}"
    )
    if env_kwargs:
        print(f"env: {env_kwargs}")

    # A successful lap target ends this car's episode; the overall PPO job
    # continues with a fresh episode. Zero disables the optional target.
    env_kwargs["laps_per_episode"] = max(0, int(laps_per_episode))
    stop_requested = threading.Event()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    try:
        vec_env = make_vec_env(n_envs, seed=seed, **env_kwargs)
    except BaseException:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        raise
    evaluation_env = None
    try:
        if resume is not None:
            print(f"resuming from {resume}")
            model = PPO.load(
                str(resume), env=vec_env, device=resolved_device,
                custom_objects={"n_steps": int(n_steps)},
            )
            extractor_name = type(model.policy.features_extractor).__name__
            checkpoint_architecture = {
                "LidarStateExtractor": "lidar_cnn",
                "PooledLidarStateExtractor": "lidar_cnn_pooled",
                "TemporalLidarStateExtractor": "temporal_lidar_cnn",
                "LidarCameraStateExtractor": "lidar_camera_cnn",
            }.get(extractor_name)
            if checkpoint_architecture != policy_architecture:
                raise ValueError(
                    "Cannot change policy architecture while resuming: "
                    f"checkpoint uses {checkpoint_architecture or extractor_name}, "
                    f"requested {policy_architecture}. Start a fresh run for an architecture change."
                )
            # A resumed archive carries its old optimizer schedule; apply the
            # explicitly selected run settings to both the PPO attributes and
            # optimizer before the first update.
            from stable_baselines3.common.utils import ConstantSchedule

            model.learning_rate = float(learning_rate)
            model.lr_schedule = ConstantSchedule(float(learning_rate))
            model.n_epochs = int(n_epochs)
            model.gamma = float(gamma)
            model.gae_lambda = float(gae_lambda)
            for group in model.policy.optimizer.param_groups:
                group["lr"] = float(learning_rate)
            # Checkpoints retain the source run's TensorBoard folder. Bind the
            # resumed policy to this run's logger so diagnostics stay isolated.
            from stable_baselines3.common.logger import configure

            model.set_logger(
                configure(folder=str(tb_dir), format_strings=["stdout", "csv", "tensorboard"])
            )
        else:
            model = make_model(
                vec_env,
                device=resolved_device,
                seed=seed,
                tensorboard_log=str(tb_dir),
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                policy_architecture=policy_architecture,
            )

        config = {
            "n_envs": n_envs,
            "bridge_ports": list(range(4567, 4567 + n_envs)),
            "evaluation_port": 4567 + n_envs,
            "timesteps": timesteps,
            "stopping": {
                "max_duration_seconds": max(0.0, float(max_duration_seconds)),
                "plateau_min_timesteps": max(0, int(plateau_min_timesteps)),
                "plateau_patience": max(1, int(plateau_patience)),
                "plateau_min_improvement_pct": max(0.0, float(plateau_min_improvement_pct)),
                "evaluation_every_timesteps": max(1, int(evaluation_every_timesteps)),
                "evaluation_runs_per_snapshot": max(1, int(evaluation_runs_per_snapshot)),
                "evaluation_metric": evaluation_metric,
                "evaluation_runs_in_parallel": True,
                "evaluation_stop_conditions": ["collision", "frontier_stall", "10_laps"],
                "plateau_metric": evaluation_metric,
                "exploration": {
                    "std_min": float(exploration_std_min),
                    "std_max": float(exploration_std_max),
                    "scale_after_improvement": float(exploration_improvement_scale),
                    "scale_after_three_stale_evaluations": float(exploration_plateau_scale),
                },
            },
            "learning_strategy": {
                "lap_termination": env_kwargs["laps_per_episode"] > 0,
                "laps_per_episode": env_kwargs["laps_per_episode"],
            },
            "seed": seed,
            "device": resolved_device,
            "env_kwargs": env_kwargs,
            "ppo": {
                "learning_rate": float(learning_rate),
                "n_steps": int(n_steps),
                "batch_size": _PPO_BATCH_SIZE,
                "n_epochs": int(n_epochs),
                "gamma": float(gamma),
                "gae_lambda": float(gae_lambda),
                "clip_range": _PPO_CLIP_RANGE,
                "ent_coef": _PPO_ENT_COEF,
                "vf_coef": _PPO_VF_COEF,
                "max_grad_norm": _PPO_MAX_GRAD_NORM,
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "features_dim": 256,
                "policy_architecture": policy_architecture,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resume": str(resume) if resume else None,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        # Keep regular recovery points even when the maximum step count is off.
        save_freq = (
            max(1, min(10_000, timesteps // max(n_envs, 1)))
            if timesteps > 0 else max(1, 10_000 // max(n_envs, 1))
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(ckpt_dir),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        stop_cb = RunStopCallback(max_duration_seconds=max_duration_seconds)
        evaluation_env_kwargs = dict(env_kwargs)
        evaluation_env_kwargs["laps_per_episode"] = 10
        # Evaluation is bounded by an actual failure/stall or ten laps, never
        # by a step/time cap or the training collision-toggle preference.
        evaluation_env_kwargs["max_episode_steps"] = 0
        evaluation_env_kwargs["terminate_on_collision"] = True
        evaluation_env_kwargs["frontier_stagnation_seconds"] = max(
            0.1, float(evaluation_env_kwargs.get("frontier_stagnation_seconds", 10.0))
        )
        evaluation_env = make_vec_env(
            1, seed=seed + 50_000, port_start=4567 + n_envs,
            **evaluation_env_kwargs,
        )
        from src.layer3.hub_callback import maybe_hub_callback

        hub_cb = maybe_hub_callback(run_id=out_dir.name, run_dir=out_dir)
        evaluation_cb = EvaluationCallback(
            evaluation_env,
            out_dir=out_dir,
            every_timesteps=evaluation_every_timesteps,
            runs_per_snapshot=evaluation_runs_per_snapshot,
            frame_skip=int(env_kwargs.get("frame_skip", 4)),
            selection_metric=evaluation_metric,
            plateau_min_timesteps=plateau_min_timesteps,
            plateau_patience=plateau_patience,
            min_improvement_pct=plateau_min_improvement_pct,
            exploration_std_min=exploration_std_min,
            exploration_std_max=exploration_std_max,
            exploration_improvement_scale=exploration_improvement_scale,
            exploration_plateau_scale=exploration_plateau_scale,
            live_pose_publisher=hub_cb.publish_evaluator_live if hub_cb is not None else None,
        )
        simulator_pause_cb = SimulatorPauseCallback()
        requested_stop_cb = RequestedStopCallback(
            stop_requested, out_dir / "ppo_signal_recovery.zip"
        )
        callbacks = [simulator_pause_cb, evaluation_cb, checkpoint_cb, stop_cb]
        if hub_cb is not None:
            callbacks.append(hub_cb)
            print(f"hub telemetry: HUB_URL set → publishing to hub (run_id={out_dir.name})")
        callbacks.append(requested_stop_cb)

        model.learn(
            # SB3 requires a finite target; plateau stopping ends unbounded runs.
            total_timesteps=int(timesteps) if timesteps > 0 else sys.maxsize,
            callback=callbacks,
            progress_bar=False,
        )
        ppo_timesteps = int(model.num_timesteps)
        last_model_path = out_dir / "ppo_last_model"
        model.save(str(last_model_path))
        final_policy_source = "ppo_last_model"
        selected_path = out_dir / "best_evaluated_model.zip"
        if selected_path.is_file():
            model = PPO.load(str(selected_path), env=vec_env, device=resolved_device)
            final_policy_source = "best_evaluated_model"
        stop_record = {
            "reason": (
                requested_stop_cb.stop_reason
                or evaluation_cb.stop_reason
                or stop_cb.stop_reason
                or "timestep_limit"
            ),
            "num_timesteps": ppo_timesteps,
            "max_duration_seconds": max(0.0, float(max_duration_seconds)),
            "plateau": evaluation_cb.latest,
            "evaluations_path": str(evaluation_cb.history_path),
            "final_policy_source": final_policy_source,
        }
        (out_dir / "stop_reason.json").write_text(
            json.dumps(stop_record, indent=2), encoding="utf-8"
        )
        print(f"training stopped: {stop_record['reason']} at {ppo_timesteps} timesteps")
        final_path = out_dir / "final_model"
        model.save(str(final_path))
        print(f"saved {final_path}.zip")
        return Path(str(final_path) + ".zip")
    finally:
        try:
            if evaluation_env is not None:
                evaluation_env.close()
        finally:
            try:
                vec_env.close()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm_handler)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 3 PPO train")
    p.add_argument("--n-envs", type=int, default=1, help="1=DummyVecEnv; >=2=SubprocVecEnv")
    p.add_argument(
        "--timesteps",
        type=int,
        default=0,
        help="Maximum total env steps; 0 disables the cap and stops on evaluation plateau",
    )
    p.add_argument(
        "--max-duration-seconds", type=float, default=0.0,
        help="Optional wall-clock run limit; 0 disables it",
    )
    p.add_argument("--plateau-min-timesteps", type=int, default=100_000)
    p.add_argument("--plateau-patience", type=int, default=5)
    p.add_argument("--plateau-min-improvement-pct", type=float, default=1.0)
    p.add_argument("--evaluation-every-timesteps", type=int, default=50_000)
    p.add_argument("--evaluation-runs-per-snapshot", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=_PPO_LR)
    p.add_argument(
        "--policy-architecture",
        choices=("lidar_cnn", "lidar_cnn_pooled", "temporal_lidar_cnn", "lidar_camera_cnn"),
        default="lidar_camera_cnn",
        help="Legacy flattened LiDAR CNN or sector-pooled LiDAR CNN",
    )
    p.add_argument("--n-steps", type=int, default=_PPO_N_STEPS)
    p.add_argument("--n-epochs", type=int, default=_PPO_N_EPOCHS)
    p.add_argument(
        "--gamma", type=float, default=_PPO_GAMMA,
        help="PPO discount factor; larger values retain credit over longer driving horizons",
    )
    p.add_argument(
        "--gae-lambda", type=float, default=_PPO_GAE_LAMBDA,
        help="PPO GAE trace factor; larger values propagate delayed outcomes farther back",
    )
    p.add_argument(
        "--evaluation-metric",
        choices=("frontier_speed", "reward_per_simulated_second", "total_reward", "ten_lap_time"),
        default="total_reward",
    )
    p.add_argument("--exploration-std-min", type=float, default=0.2)
    p.add_argument("--exploration-std-max", type=float, default=0.8)
    p.add_argument("--exploration-improvement-scale", type=float, default=0.9)
    p.add_argument("--exploration-plateau-scale", type=float, default=1.1)
    p.add_argument("--out", type=Path, default=None, help="Run directory under logs/rl/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=_device_arg, default="auto")
    p.add_argument("--resume", type=Path, default=None, help="Optional PPO .zip to continue")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-timeout", type=float, default=90.0)
    # ~10 Hz decisions at 40 Hz physics — easier credit assignment than 40 Hz.
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument(
        "--action-interval-s", type=float, default=None,
        help="Opt-in simulated seconds per Bridge action; requires fixed-step player",
    )
    p.add_argument(
        "--steering-action-scale", type=float, default=1.0,
        help="Scale the normalized steering command sent to Unity (0..1)",
    )
    p.add_argument(
        "--straight-throttle-gain", type=float, default=1.0,
        help="Multiply positive throttle by this factor when executed steering is below its threshold (1 disables)",
    )
    p.add_argument(
        "--straight-throttle-steering-threshold", type=float, default=0.15,
        help="Apply straight-throttle gain only when absolute executed steering is below this threshold (0..1)",
    )
    p.add_argument(
        "--observation-profile",
        choices=("simulator", "simulator_camera", "official_sensors", "official_sensors_history", "official_sensors_camera"),
        default="simulator_camera",
        help="Use full simulator telemetry or match the official allowed sensor inputs",
    )
    p.add_argument(
        "--throttle-mode",
        choices=("bidirectional", "forward_only"),
        default="bidirectional",
        help="Map normalized PPO throttle to signed throttle or forward-only [0, 1]",
    )
    # Hard cap so stuck-but-wiggling episodes still reset (~100 s @ frame_skip=4).
    p.add_argument("--max-episode-steps", type=int, default=0)
    p.add_argument("--stagnation-speed-threshold", type=float, default=0.15)
    # ~5 s idle @ frame_skip=4 before truncate.
    p.add_argument("--stagnation-steps", type=int, default=50)
    p.add_argument(
        "--terminate-on-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--map-id", type=str, default="none")
    p.add_argument("--laps-per-episode", type=int, default=10,
                   help="Successful laps before resetting an episode; 0 disables")
    p.add_argument("--frontier-stagnation-seconds", type=float, default=10.0)
    p.add_argument("--forward-scale", type=float, default=0.0)
    p.add_argument("--backward-speed-penalty-scale", type=float, default=1.0)
    p.add_argument("--route-progress-scale", type=float, default=10.0)
    p.add_argument(
        "--frontier-pace-target-mps", type=float, default=6.0,
        help="Average frontier pace at which the pace multiplier reaches its cap",
    )
    p.add_argument(
        "--frontier-pace-bonus-strength", type=float, default=1.0,
        help="Quadratic bonus strength; 1 gives a 2x cap, 2 gives a 3x cap",
    )
    p.add_argument(
        "--frontier-pace-source", choices=("episode_average", "current_push"),
        default="episode_average",
        help="Measure pace bonus from episode-average frontier speed or the current frontier advance",
    )
    p.add_argument("--time-penalty-per-second", type=float, default=5.0)
    p.add_argument("--collision-penalty-magnitude", type=float, default=100.0)
    p.add_argument("--collision-reward-percent", type=float, default=100.0)
    p.add_argument("--episode-failure-penalty-magnitude", type=float, default=100.0)
    p.add_argument("--episode-failure-reward-percent", type=float, default=100.0)
    p.add_argument("--slip-penalty", type=float, default=0.2)
    p.add_argument("--steer-jerk-penalty", type=float, default=0.05)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not math.isfinite(args.steering_action_scale) or not 0.0 <= args.steering_action_scale <= 1.0:
        print("ERROR: --steering-action-scale must be finite and in [0, 1]")
        return 1
    if not math.isfinite(args.straight_throttle_gain) or not 1.0 <= args.straight_throttle_gain <= 2.0:
        print("ERROR: --straight-throttle-gain must be finite and in [1, 2]")
        return 1
    if (
        not math.isfinite(args.straight_throttle_steering_threshold)
        or not 0.0 <= args.straight_throttle_steering_threshold <= 1.0
    ):
        print("ERROR: --straight-throttle-steering-threshold must be finite and in [0, 1]")
        return 1
    if args.n_envs < 1:
        print("ERROR: --n-envs must be >= 1")
        return 1
    if args.n_envs > 16:
        print("ERROR: --n-envs must be <= 16 (the simulator bridge range)")
        return 1

    from src.layer3.envs import env_kwargs_from_args

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or (ROOT / "logs" / "rl" / f"ppo_{stamp}")
    zip_path = train(
        n_envs=args.n_envs,
        timesteps=args.timesteps,
        out_dir=out,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        laps_per_episode=args.laps_per_episode,
        env_kwargs=env_kwargs_from_args(args),
        max_duration_seconds=args.max_duration_seconds,
        plateau_min_timesteps=args.plateau_min_timesteps,
        plateau_patience=args.plateau_patience,
        plateau_min_improvement_pct=args.plateau_min_improvement_pct,
        evaluation_every_timesteps=args.evaluation_every_timesteps,
        evaluation_runs_per_snapshot=args.evaluation_runs_per_snapshot,
        evaluation_metric=args.evaluation_metric,
        exploration_std_min=args.exploration_std_min,
        exploration_std_max=args.exploration_std_max,
        exploration_improvement_scale=args.exploration_improvement_scale,
        exploration_plateau_scale=args.exploration_plateau_scale,
        learning_rate=args.learning_rate,
        policy_architecture=args.policy_architecture,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    print(f"done: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
