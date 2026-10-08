"""Replay a saved policy as a fresh official evaluation attempt."""
import json
import threading
from pathlib import Path
from src.layer4.hub.official_models import checkpoint_settings


class OfficialReplayJob:
    def __init__(self, manager, on_status=None):
        self.manager = manager
        self.on_status = on_status
        self._lock = threading.RLock()
        self._starting = False
        self._last_config = None
        runs = manager.list_runs()["runs"]
        self.run_id = next((r["run_id"] for r in runs if r.get("kind") == "evaluation"), None)
        if self.run_id:
            resume = manager.get_run(self.run_id).get("config", {}).get("resume")
            if isinstance(resume, str) and resume.startswith("/runs/"):
                self._last_config = {"model_path": manager.runs_root.parent / resume[len("/runs/"):]}

    def status(self):
        row = self.manager.get_run(self.run_id) if self.run_id else {}
        mapped = {"completed": "exited", "stopped": "exited", "failed": "error"}
        report = None
        if row.get("output_dir"):
            try:
                report = json.loads((Path(row["output_dir"]) / "evaluation.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {
            "state": "starting" if self._starting else mapped.get(row.get("state"), row.get("state", "idle")),
            "pid": None, "argv": [], "started_at": row.get("started_at"),
            "exit_code": row.get("exit_code"), "error": row.get("error"),
            "log_path": row.get("log_path"), "model": row.get("config", {}).get("resume"),
            "map_id": "none", "seed": None, "run_id": self.run_id,
            "evaluation": report,
            "last_fleet": self.manager.telemetry(self.run_id).get("fleet") if self.run_id else None,
        }

    def start(self, *, model_path, trace_steps=0, **_ignored):
        if isinstance(trace_steps, bool) or not isinstance(trace_steps, int) or not 0 <= trace_steps <= 500:
            raise ValueError("trace_steps must be an integer between 0 and 500")
        with self._lock:
            if self.status()["state"] in {"starting", "running", "stopping"}:
                raise RuntimeError("An official replay is already active")
            config = {**checkpoint_settings(Path(model_path)), "resume": str(model_path), "n_envs": 1}
            self._starting = True
        try:
            row = self.manager.create_run(
                config,
                f"Evaluation: {Path(model_path).name}",
                kind="evaluation",
                evaluation_trace_steps=trace_steps,
            )
            self.run_id = row["run_id"]
            self._last_config = {"model_path": model_path, "trace_steps": trace_steps}
        finally:
            self._starting = False
        result = self.status()
        if self.on_status:
            self.on_status(result)
        return result

    def stop(self):
        if self.run_id:
            self.manager.stop_run(self.run_id)
        result = self.status()
        if self.on_status:
            self.on_status(result)
        return result

    def reset(self):
        if self._last_config is None:
            raise RuntimeError("Select a checkpoint and start an evaluation first")
        self.stop()
        return self.start(**self._last_config)
