from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
import json

import pytest

from src.layer2.rewards import RewardConfig, compute_reward_components
from src.layer4.hub import official_run_manager as run_manager
from src.layer4.hub.campaign_deadline import (
    BC_TIME_COST_CALIBRATION_LAP_S,
    INVALIDATED_TIME_COST_VARIANTS,
    TIME_COST_VARIANTS,
    assert_campaign_training_config,
    campaign_effective_config_sha256,
    declare_campaign_experiment,
    initialize_campaign,
)
from src.layer4.hub.campaign_experiments import build_campaign_decision_plan
from src.layer4.official_settings import OfficialTrainSettings


START = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
BOOT = "isolated-test-boot"
PARENT_SHA = "b" * 64
CONTROL_REPORT_SHA = "c" * 64
LAP_STEPS_S = (
    0.017, 0.033, 0.061, 0.119, 0.47,
    1.23, 3.71, 17.3, 33.1, 54.215550384521,
)


def _active_campaign(path):
    gates = path.parent / "gates.json"
    gates.write_text(json.dumps({
        "updated_policy_probe": {
            "status": "PASS", "useful": True, "checkpoint_sha256": "a" * 64,
        },
        "single_environment_queue_interruption": {"status": "PASS", "passed": True},
    }), encoding="utf-8")
    return initialize_campaign(path, gates, utc_now=START, monotonic_ns=0, boot_id=BOOT)


def _normalized_campaign_request(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "ROOT", tmp_path)
    checkpoint = tmp_path / "logs" / "rl" / "parent.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint placeholder without a lineage sidecar")
    raw = OfficialTrainSettings().model_dump(mode="python")
    raw.update({
        "resume": str(checkpoint),
        "simulator_backend": "windows_external",
        "timeout_s": 180.0,
        "total_timesteps": 0,
        "stop_on_plateau": False,
        "max_duration_hours": 0.0,
        "behavior_cloning_enabled": False,
        "behavior_cloning_only": False,
        "learning_rate": 3e-5,
        "target_kl": 0.01,
        "exploration_profile": "bc_campaign",
        "exploration_initial_std_throttle": 0.01,
        "exploration_initial_std_steering": 0.05,
        "exploration_min_std_throttle": 0.005,
        "exploration_min_std_steering": 0.02,
        "exploration_max_std_throttle": 0.01,
        "exploration_max_std_steering": 0.05,
        "n_envs": 1,
        "n_steps": 2048,
        "n_epochs": 4,
        "device": "cuda",
        "gamma": 0.9995,
        "gae_lambda": 0.95,
        "checkpoint_every": 10000,
        "automatic_evaluation_enabled": True,
        "automatic_evaluation_trigger": "steps",
        "automatic_evaluation_interval": 50000,
        "automatic_evaluation_mode": "training_probe",
        "time_cost_per_simulated_second": 0.0,
    })
    # Test the actual API/run-manager normalization path without relying on an
    # on-disk campaign model or importing the running Mission Control app.
    monkeypatch.setattr(run_manager, "verify_bc_checkpoint", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(run_manager, "verify_ppo_lineage_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_manager, "verify_ppo_warmstart", lambda *_args, **_kwargs: None)
    return run_manager.normalize_official_config(raw, explicit_fields=set(raw))


@pytest.mark.parametrize(("experiment_id", "target_points"), [
    ("time_cost_3_calibrated_retry", 3.0),
    ("time_cost_5_calibrated_retry", 5.0),
])
def test_declared_normalized_time_cost_is_lap_calibrated_reward(tmp_path, monkeypatch, experiment_id, target_points):
    state_path = tmp_path / "active_campaign.json"
    state = _active_campaign(state_path)
    base_request = _normalized_campaign_request(tmp_path, monkeypatch)
    assert base_request["resume"] == "/runs/rl/parent.zip"

    plan = build_campaign_decision_plan(
        state,
        utc_now=START + timedelta(hours=3),
        monotonic_ns=3 * 3600 * 1_000_000_000,
        boot_id=BOOT,
        evidence={
            "baseline_control_parent_sha256": PARENT_SHA,
            "baseline_control_probe_passed": True,
            "baseline_control_probe_report_sha256": CONTROL_REPORT_SHA,
            "baseline_control_median_scored_lap_s": BC_TIME_COST_CALIBRATION_LAP_S,
            "base_train_request": base_request,
        },
    )
    candidates = {row["experiment_id"]: row for row in plan["branch_candidates"]}
    candidate = candidates[experiment_id]
    submitted = candidate["request_settings"]
    expected_coefficient = target_points / BC_TIME_COST_CALIBRATION_LAP_S
    assert TIME_COST_VARIANTS[experiment_id] == pytest.approx(expected_coefficient, abs=1e-15)
    assert submitted["time_cost_per_simulated_second"] == pytest.approx(expected_coefficient, abs=1e-15)

    # The branch is declared and admitted using the same normalized config hash
    # the manager will persist. This catches unit/config drift across planner,
    # ledger, normalization, and training admission.
    declaration = declare_campaign_experiment(
        state_path,
        experiment_id=experiment_id,
        phase="pace_experiments",
        config=submitted,
        parent_checkpoint_sha256=PARENT_SHA,
        evidence=candidate["declaration_evidence"],
        utc_now=START + timedelta(hours=3),
        monotonic_ns=3 * 3600 * 1_000_000_000,
    )
    normalized = run_manager.normalize_official_config(
        declaration["config"], explicit_fields=set(declaration["config"]),
    )
    assert normalized["campaign_experiment_config_sha256"] == campaign_effective_config_sha256(normalized)
    assert normalized["campaign_experiment_config_sha256"] == declaration["config_sha256"]
    assert normalized["time_cost_per_simulated_second"] == pytest.approx(expected_coefficient, abs=1e-15)
    assert_campaign_training_config(normalized, state_path=state_path)

    # The reward implementation charges coefficient × simulated seconds per
    # step. Summing an explicit irregular ten-step lap must yield the declared
    # 3- or 5-point penalty, never 3 or 5 points per second.
    assert sum(LAP_STEPS_S) == pytest.approx(BC_TIME_COST_CALIBRATION_LAP_S, abs=1e-12)
    cfg = RewardConfig(
        time_penalty_per_second=normalized["time_cost_per_simulated_second"],
        backward_speed_penalty_scale=0.0,
        route_progress_scale=0.0,
        collision_penalty_magnitude=0.0,
        episode_failure_penalty_magnitude=0.0,
    )
    per_step = [
        compute_reward_components(
            v_long=0.0, step_duration_s=dt, collision_event=False,
            slip_angle=0.0, prev_steering=0.0, steering=0.0, cfg=cfg,
        )["time_cost"]
        for dt in LAP_STEPS_S
    ]
    assert all(math.isclose(value, -expected_coefficient * dt, rel_tol=0, abs_tol=1e-14)
               for value, dt in zip(per_step, LAP_STEPS_S))
    assert sum(per_step) == pytest.approx(-target_points, abs=1e-12)


@pytest.mark.parametrize("experiment_id", list(INVALIDATED_TIME_COST_VARIANTS))
def test_literal_time_cost_experiment_ids_are_rejected(tmp_path, monkeypatch, experiment_id):
    state_path = tmp_path / "active_campaign.json"
    _active_campaign(state_path)
    base_request = _normalized_campaign_request(tmp_path, monkeypatch)
    candidate = dict(base_request)
    candidate["time_cost_per_simulated_second"] = 3.0 if "_3" in experiment_id else 5.0
    with pytest.raises(ValueError, match="approved|invalidated|calibrated"):
        declare_campaign_experiment(
            state_path,
            experiment_id=experiment_id,
            phase="pace_experiments",
            config=candidate,
            parent_checkpoint_sha256=PARENT_SHA,
            evidence={
                "baseline_control_parent_sha256": PARENT_SHA,
                "baseline_control_probe_passed": True,
                "baseline_control_probe_report_sha256": CONTROL_REPORT_SHA,
                "baseline_control_median_scored_lap_s": BC_TIME_COST_CALIBRATION_LAP_S,
            },
            utc_now=START + timedelta(hours=3),
            monotonic_ns=3 * 3600 * 1_000_000_000,
        )
