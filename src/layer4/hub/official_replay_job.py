"""Mission Control facade for manual official full-race evaluation."""

from pathlib import Path
from typing import Any

from src.layer4.hub.official_evaluation_service import OfficialEvaluationService


class OfficialReplayJob:
    """Keep the existing Replay API while using the shared evaluator queue."""

    def __init__(self, manager: Any, on_status=None, evaluation_service: OfficialEvaluationService | None = None):
        self.manager = manager
        self.on_status = on_status
        self.evaluation_service = evaluation_service or OfficialEvaluationService(manager)
        self._last_config: dict[str, Any] | None = None
        try:
            latest = next((row for row in manager.list_runs().get("runs", []) if row.get("kind") == "evaluation"), None)
            resume = latest.get("config", {}).get("resume") if latest else None
            if isinstance(resume, str) and resume.startswith("/runs/rl/"):
                self._last_config = {"model_path": manager.runs_root / resume.removeprefix("/runs/rl/"), "trace_steps": 0}
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        status = self.evaluation_service.status()
        result = status.get("evaluation")
        return {
            **status,
            "pid": None, "argv": [], "started_at": status.get("updated_utc"),
            "exit_code": None, "map_id": "none", "seed": None,
            "evaluation": result,
        }

    def start(self, *, model_path, trace_steps=0, validation_mode=None, **_ignored):
        path = Path(model_path)
        self._last_config = {"model_path": path, "trace_steps": trace_steps, "validation_mode": validation_mode}
        result = self.evaluation_service.submit(
            path, source="manual", trace_steps=trace_steps, validation_mode=validation_mode
        )
        if self.on_status:
            self.on_status(self.status())
        return {**result, "pid": None, "argv": [], "map_id": "none", "seed": None}

    def start_automatic(self, *, model_path: Path, run_id: str, trigger: str, step: int | None, episode: int | None,
                        validation_mode: str | None = None):
        result = self.evaluation_service.submit(
            model_path, source="automatic", parent_run_id=run_id,
            trigger=trigger, trigger_step=step, trigger_episode=episode,
            validation_mode=validation_mode,
        )
        if self.on_status:
            self.on_status(self.status())
        return result

    def stop(self):
        result = self.evaluation_service.stop()
        if self.on_status:
            self.on_status(self.status())
        return result

    def reset(self):
        if self._last_config is None:
            raise RuntimeError("Select a checkpoint and start an evaluation first")
        self.stop()
        return self.start(**self._last_config)
