import json
import threading
import time
from pathlib import Path

import pytest

from src.layer4.hub.official_evaluation_service import OfficialEvaluationService


class PauseResumeManager:
    def __init__(self):
        self.calls = []

    def list_runs(self):
        return {"runs": []}

    def recover_paused_training_after_hub_restart(self):
        pass

    def pause_active_training_for_evaluation(self):
        self.calls.append("pause")
        return ["training-run"]

    def resume_training_after_evaluation(self, run_ids):
        self.calls.append(("resume", list(run_ids)))


def _checkpoint(tmp_path: Path) -> Path:
    run = tmp_path / "logs" / "rl" / "source"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    settings = {
        "runtime": "official IROS 2026 runtime",
        "observation_profile": "official_sensors_camera",
        "actions": {
            "throttle_mode": "bidirectional", "negative_throttle_mode": "allow",
            "steering_mode": "normal", "steering_action_scale": 1.0,
            "straight_throttle_gain": 1.0, "straight_throttle_steering_threshold": 0.15,
        },
    }
    (run / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    checkpoint = checkpoints / "policy.zip"
    checkpoint.write_bytes(b"fixture checkpoint")
    return checkpoint


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "evaluation worker did not reach expected state"


def test_training_resumes_when_evaluator_worker_raises(tmp_path):
    manager = PauseResumeManager()
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "registry.json",
        champion_dir=tmp_path / "champion", model_root=tmp_path / "logs" / "rl",
    )
    started = threading.Event()

    def fail_evaluation(job):
        started.set()
        raise RuntimeError("simulated evaluator failure")

    service._run_batch = fail_evaluation
    try:
        service.submit(_checkpoint(tmp_path), source="manual")
        assert started.wait(2.0)
        _wait_until(lambda: service.status()["state"] == "failed")
        assert manager.calls == ["pause", ("resume", ["training-run"])]
        assert "simulated evaluator failure" in service.status()["error"]
    finally:
        service.close()


def test_operator_stop_during_evaluation_resumes_training(tmp_path):
    manager = PauseResumeManager()
    service = OfficialEvaluationService(
        manager, registry_path=tmp_path / "registry.json",
        champion_dir=tmp_path / "champion", model_root=tmp_path / "logs" / "rl",
    )
    entered = threading.Event()

    def wait_for_operator_stop(job):
        entered.set()
        while not job.get("stop_requested"):
            time.sleep(0.005)
        job["state"] = "stopped"

    service._run_batch = wait_for_operator_stop
    try:
        service.submit(_checkpoint(tmp_path), source="manual")
        assert entered.wait(2.0)
        service.stop()
        _wait_until(lambda: service.status()["state"] == "stopped")
        assert manager.calls == ["pause", ("resume", ["training-run"])]
    finally:
        service.close()
