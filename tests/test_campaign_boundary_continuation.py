from __future__ import annotations

from pathlib import Path

import pytest

import src.layer4.hub.campaign_phase_runner as phase_runner


@pytest.mark.parametrize("experiment_id", ["continue_best", "continue_best_boundary_recovery"])
def test_five_hour_tick_does_not_stop_a_continuation_inheriting_calibrated_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment_id: str,
):
    coefficient = phase_runner.TIME_COST_VARIANTS["time_cost_3_calibrated_retry"]

    class RecordingAPI:
        def __init__(self):
            self.posts = []

        def get(self, route):
            if route == "/train/runs":
                return {"runs": [{
                    "run_id": "continuation-run",
                    "kind": "training",
                    "state": "running",
                    "display_name": f"Campaign {experiment_id}",
                    "config": {
                        "campaign_experiment_id": experiment_id,
                        "time_cost_per_simulated_second": coefficient,
                    },
                }]}
            if route == "/api/evaluations/history":
                return []
            if route == "/api/replay/status":
                return {"state": "stopped"}
            if route == "/health":
                return {"status": "ok"}
            if route == "/train/runs/continuation-run/telemetry":
                return {}
            raise AssertionError(f"unexpected API read: {route}")

        def post(self, route, body=None):
            self.posts.append((route, body))
            return {"run_id": "unexpected-stop"}

    api = RecordingAPI()
    runner = phase_runner.CampaignPhaseRunner(
        state_path=tmp_path / "active_campaign.json", repo_root=tmp_path, api=api,
    )
    campaign_state = {
        "campaign_id": phase_runner.CAMPAIGN_ID,
        "status": "active",
        "events": [],
    }
    monkeypatch.setattr(phase_runner, "load_campaign_state", lambda _path: campaign_state)
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: {})
    monkeypatch.setattr(phase_runner, "build_campaign_decision_plan", lambda *_args, **_kwargs: {
        "phase": "best_validated_trend",
        "elapsed_hours": 5.2,
        "branch_candidates": [],
        "faults": [],
    })

    result = runner.tick()

    assert result["active_training_ids"] == ["continuation-run"]
    assert result["action"] == "observe_only"
    assert api.posts == []


def test_five_hour_tick_still_stops_the_declared_calibrated_pace_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    experiment_id = "time_cost_3_calibrated_retry"
    coefficient = phase_runner.TIME_COST_VARIANTS[experiment_id]

    class RecordingAPI:
        def __init__(self):
            self.posts = []

        def get(self, route):
            if route == "/train/runs":
                return {"runs": [{
                    "run_id": "pace-run",
                    "kind": "training",
                    "state": "running",
                    "display_name": "Campaign time cost pace",
                    "config": {
                        "campaign_experiment_id": experiment_id,
                        "time_cost_per_simulated_second": coefficient,
                    },
                }]}
            if route == "/api/evaluations/history":
                return []
            if route == "/api/replay/status":
                return {"state": "stopped"}
            if route == "/health":
                return {"status": "ok"}
            if route == "/train/runs/pace-run/telemetry":
                return {}
            raise AssertionError(f"unexpected API read: {route}")

        def post(self, route, body=None):
            self.posts.append((route, body))
            return {}

    api = RecordingAPI()
    runner = phase_runner.CampaignPhaseRunner(
        state_path=tmp_path / "active_campaign.json", repo_root=tmp_path, api=api,
    )
    campaign_state = {
        "campaign_id": phase_runner.CAMPAIGN_ID,
        "status": "active",
        "events": [],
    }
    monkeypatch.setattr(phase_runner, "load_campaign_state", lambda _path: campaign_state)
    monkeypatch.setattr(runner, "collect_evidence", lambda _snapshot: {})
    monkeypatch.setattr(phase_runner, "build_campaign_decision_plan", lambda *_args, **_kwargs: {
        "phase": "pace_experiments",
        "elapsed_hours": 5.2,
        "branch_candidates": [],
        "faults": [],
    })

    result = runner.tick()

    assert result["action"] == "stop_pace_branch_at_phase_boundary"
    assert api.posts == [("/train/runs/pace-run/stop", None)]
