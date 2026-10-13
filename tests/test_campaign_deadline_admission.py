from pathlib import Path
from unittest.mock import Mock

import pytest

from src.layer4.hub.official_evaluation_service import OfficialEvaluationService
from src.layer4.hub.official_run_manager import OfficialRunManager


def test_run_manager_checks_campaign_admission_for_both_run_kinds(tmp_path, monkeypatch):
    admission = Mock()
    monkeypatch.setattr(
        "src.layer4.hub.official_run_manager.assert_campaign_allows", admission
    )
    # This test isolates the manager's final campaign admission boundary. Keep
    # checkpoint normalization/authentication out of this fixture so a fake
    # resume path cannot fail before the admission assertion under test.
    monkeypatch.setattr(
        "src.layer4.hub.official_run_manager.normalize_official_config",
        lambda config: {"n_envs": 1, "resume": "/runs/official_run_fixture/checkpoint.zip", **config},
    )
    manager = OfficialRunManager(
        backend=object(), registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "runs", start_monitor=False,
    )
    try:
        for kind in ("training", "evaluation"):
            with pytest.raises(RuntimeError, match="window closed"):
                admission.side_effect = RuntimeError("window closed")
                manager.create_run(
                    {"n_envs": 1, "resume": "/runs/official_run_fixture/checkpoint.zip"}, kind=kind
                )
        assert [call.args for call in admission.call_args_list] == [
            ("training",), ("evaluation",)
        ]
    finally:
        manager.close()


def test_manager_checks_immutable_campaign_plan_after_normalization(tmp_path, monkeypatch):
    admission = Mock()
    plan_check = Mock(side_effect=RuntimeError("fixed campaign plan required"))
    monkeypatch.setattr("src.layer4.hub.official_run_manager.assert_campaign_allows", admission)
    monkeypatch.setattr(
        "src.layer4.hub.official_run_manager.assert_campaign_training_config", plan_check
    )
    monkeypatch.setattr(
        "src.layer4.hub.official_run_manager.normalize_official_config",
        lambda value: {"n_envs": 1, "resume": "checkpoint.zip", **value},
    )
    manager = OfficialRunManager(
        backend=object(), registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "runs", start_monitor=False,
    )
    try:
        with pytest.raises(RuntimeError, match="fixed campaign plan"):
            manager.create_run({"total_timesteps": 8192}, kind="training")
        plan_check.assert_called_once()
    finally:
        manager.close()


def test_evaluation_queue_checks_deadline_before_checkpoint_work(monkeypatch):
    admission = Mock(side_effect=RuntimeError("hard deadline reached"))
    monkeypatch.setattr(
        "src.layer4.hub.official_evaluation_service.assert_campaign_allows", admission
    )
    service = OfficialEvaluationService.__new__(OfficialEvaluationService)

    with pytest.raises(RuntimeError, match="hard deadline"):
        service.submit(Path("missing.zip"), source="manual")
    admission.assert_called_once_with("evaluation")
