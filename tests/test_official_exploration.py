from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.layer3.official_exploration import OfficialExplorationSchedule


def _write_evaluation(run_dir, index, *, valid=False, score=None, validation_mode=None):
    (run_dir / "evaluation_status.json").write_text(json.dumps({
        "state": "complete" if valid else "incomplete",
        "evaluation_id": f"evaluation-{index}",
        "snapshot_timesteps": index * 20_000,
        "validation_mode": validation_mode,
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


def test_per_action_bounds_preserve_safe_bc_throttle_and_steering_noise(tmp_path):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.full((2,), math.log(0.2))))
    with torch.no_grad():
        policy.log_std.copy_(torch.tensor([math.log(0.01), math.log(0.05)]))
    schedule = OfficialExplorationSchedule(
        SimpleNamespace(policy=policy),
        tmp_path,
        std_min=(0.01, 0.05),
        std_max=(0.8, 0.8),
    )
    schedule.on_rollout_start()

    std = policy.log_std.detach().exp().tolist()
    assert math.isclose(std[0], 0.01, rel_tol=1e-6)
    assert math.isclose(std[1], 0.05, rel_tol=1e-6)


def test_per_action_exploration_adapts_and_respects_bounds(tmp_path):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.log(torch.tensor([0.01, 0.05]))))
    schedule = OfficialExplorationSchedule(
        SimpleNamespace(policy=policy),
        tmp_path,
        std_min=(0.01, 0.05),
        std_max=(0.8, 0.8),
    )

    for index in range(1, 4):
        _write_evaluation(tmp_path, index)
        record = schedule.observe(index, force=True)

    assert record is not None
    assert np.allclose(policy.log_std.detach().exp().numpy(), [0.011, 0.055], rtol=1e-6)
    assert np.allclose(record["action_std_by_action"], [0.011, 0.055], rtol=1e-6)

    _write_evaluation(tmp_path, 4, valid=True, score=42.0)
    schedule.observe(4, force=True)
    assert np.allclose(policy.log_std.detach().exp().numpy(), [0.01, 0.05], rtol=1e-6)


def test_per_action_bounds_require_one_value_per_policy_action(tmp_path):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.zeros(2)))
    with pytest.raises(ValueError, match="one value per action dimension"):
        OfficialExplorationSchedule(
            SimpleNamespace(policy=policy),
            tmp_path,
            std_min=(0.01,),
            std_max=(0.8, 0.8),
        )


@pytest.mark.parametrize("mode", ["bc_diagnostic", "training_probe"])
def test_nonranking_safety_checks_do_not_adapt_exploration(tmp_path, mode):
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.full((2,), math.log(0.2))))
    schedule = OfficialExplorationSchedule(SimpleNamespace(policy=policy), tmp_path)
    for index in range(1, 5):
        _write_evaluation(tmp_path, index, validation_mode=mode)
        assert schedule.observe(index, force=True) is None
    assert schedule.stale_evaluations == 0
    assert math.isclose(_std(policy), 0.2, rel_tol=1e-6)
    assert not (tmp_path / "exploration_adaptation.jsonl").exists()
