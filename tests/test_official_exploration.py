from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch

from src.layer3.official_exploration import OfficialExplorationSchedule


def _write_evaluation(run_dir, index, *, valid=False, score=None):
    (run_dir / "evaluation_status.json").write_text(json.dumps({
        "state": "complete" if valid else "incomplete",
        "evaluation_id": f"evaluation-{index}",
        "snapshot_timesteps": index * 20_000,
        "latest_result": {
            "valid": valid,
            "median_adjusted_race_time_s": score,
        },
    }), encoding="utf-8")


def _std(policy):
    return float(policy.log_std.detach().exp().mean())


def test_official_exploration_increases_after_three_stale_evaluations(tmp_path):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.full((2,), math.log(0.2))))
    schedule = OfficialExplorationSchedule(SimpleNamespace(policy=policy), tmp_path)

    for index in range(1, 4):
        _write_evaluation(tmp_path, index)
        schedule.observe(index, force=True)
        if index < 3:
            assert math.isclose(_std(policy), 0.2, rel_tol=1e-6)

    assert math.isclose(_std(policy), 0.22, rel_tol=1e-6)
    record = json.loads((tmp_path / "exploration_adaptation.jsonl").read_text().splitlines()[-1])
    assert record["reason"] == "increased_exploration_after_stale_evaluations"
    assert record["stale_evaluations"] == 0


def test_official_exploration_reduces_noise_for_better_valid_scores(tmp_path):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.full((2,), math.log(0.4))))
    schedule = OfficialExplorationSchedule(SimpleNamespace(policy=policy), tmp_path)

    _write_evaluation(tmp_path, 1, valid=True, score=80.0)
    schedule.observe(1, force=True)
    assert math.isclose(_std(policy), 0.36, rel_tol=1e-6)

    _write_evaluation(tmp_path, 2, valid=True, score=70.0)
    schedule.observe(2, force=True)
    assert math.isclose(_std(policy), 0.324, rel_tol=1e-6)


def test_official_exploration_ignores_in_progress_and_duplicate_evaluations(tmp_path):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.full((2,), math.log(0.3))))
    schedule = OfficialExplorationSchedule(SimpleNamespace(policy=policy), tmp_path)
    status_path = tmp_path / "evaluation_status.json"
    status_path.write_text(json.dumps({"state": "running", "evaluation_id": "same"}))
    assert schedule.observe(1, force=True) is None
    _write_evaluation(tmp_path, 1)
    assert schedule.observe(2, force=True) is not None
    assert schedule.observe(3, force=True) is None
    assert schedule.stale_evaluations == 1
    assert math.isclose(_std(policy), 0.3, rel_tol=1e-6)
