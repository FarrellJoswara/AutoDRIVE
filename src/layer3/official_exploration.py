"""Evaluation-driven PPO exploration schedule for official race training."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch


class OfficialExplorationSchedule:
    """Adapt Gaussian action noise after completed official race batches.

    This follows the fastest historical champion's rule: reduce standard
    deviation by 10% after a better valid ten-lap score; increase it by 10%
    after three evaluation snapshots without a valid improvement.
    """

    def __init__(
        self,
        model: Any,
        run_dir: Path,
        *,
        std_min: float = 0.2,
        std_max: float = 0.8,
        improvement_scale: float = 0.9,
        stale_scale: float = 1.1,
        stale_patience: int = 3,
        poll_interval_steps: int = 128,
    ) -> None:
        if not (0 < std_min <= std_max):
            raise ValueError("exploration std bounds must be positive and ordered")
        if improvement_scale <= 0 or improvement_scale > 1:
            raise ValueError("improvement_scale must be in (0, 1]")
        if stale_scale < 1 or stale_patience < 1 or poll_interval_steps < 1:
            raise ValueError("stale scale, patience, and poll interval are invalid")
        self.model = model
        self.run_dir = Path(run_dir)
        self.status_path = self.run_dir / "evaluation_status.json"
        self.history_path = self.run_dir / "exploration_adaptation.jsonl"
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        self.improvement_scale = float(improvement_scale)
        self.stale_scale = float(stale_scale)
        self.stale_patience = int(stale_patience)
        self.poll_interval_steps = int(poll_interval_steps)
        self.target_log_std = float(model.policy.log_std.detach().mean().cpu())
        self.best_valid_time_s: float | None = None
        self.stale_evaluations = 0
        self.last_status_key: tuple[str, int] | None = None
        self.last_poll_steps = -1

    def _clamp(self) -> None:
        self.target_log_std = min(
            math.log(self.std_max), max(math.log(self.std_min), self.target_log_std)
        )
        with torch.no_grad():
            self.model.policy.log_std.fill_(self.target_log_std)

    def on_rollout_start(self) -> None:
        self._clamp()

    def observe(self, timesteps: int, *, force: bool = False) -> dict[str, Any] | None:
        steps = int(timesteps)
        if not force and steps - self.last_poll_steps < self.poll_interval_steps:
            return None
        self.last_poll_steps = steps
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if status.get("state") not in {"complete", "incomplete"}:
            return None
        evaluation_id = str(status.get("evaluation_id") or "")
        snapshot_steps = int(status.get("snapshot_timesteps") or 0)
        key = (evaluation_id, snapshot_steps)
        if key == self.last_status_key:
            return None
        self.last_status_key = key
        result = status.get("latest_result") or {}
        score = result.get("median_adjusted_race_time_s")
        valid = bool(result.get("valid")) and isinstance(score, (int, float))
        improved = valid and (
            self.best_valid_time_s is None or float(score) < self.best_valid_time_s
        )
        if improved:
            self.best_valid_time_s = float(score)
            self.stale_evaluations = 0
            self.target_log_std += math.log(self.improvement_scale)
            reason = "better_valid_ten_lap_score"
        else:
            self.stale_evaluations += 1
            reason = "stale_official_evaluation"
            if self.stale_evaluations >= self.stale_patience:
                self.target_log_std += math.log(self.stale_scale)
                reason = "increased_exploration_after_stale_evaluations"
                self.stale_evaluations = 0
        self._clamp()
        record = {
            "evaluation_id": evaluation_id,
            "snapshot_timesteps": snapshot_steps,
            "valid": valid,
            "score_s": float(score) if valid else None,
            "best_valid_time_s": self.best_valid_time_s,
            "stale_evaluations": self.stale_evaluations,
            "reason": reason,
            "action_std": float(self.model.policy.log_std.detach().exp().mean().cpu()),
            "observed_at": time.time(),
        }
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        return record
