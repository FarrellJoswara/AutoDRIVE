"""Manager-boundary checks for the one-shot campaign maintenance resume."""

from __future__ import annotations

import pytest

from src.layer4.hub import official_run_manager as manager_module
from src.layer4.official_settings import OfficialTrainSettings


class _Backend:
    is_owner = True
    simulator_host = "127.0.0.1"

    def list_official_run_containers(self, _run_id=None):
        return []

    def list_official_run_networks(self, _run_id=None):
        return []

    def find_host_path_for_container_path(self, _path):
        raise RuntimeError("stop test after the one-shot claim")


def test_manager_claims_exact_normalized_resume_once_at_creation_boundary(tmp_path, monkeypatch):
    calls = []
    hub_identity = {
        "pid": 2050,
        "process_start_identity": "pid:2050:start_ticks:900",
        "boot_id": "test-boot",
        "executable": "/usr/bin/python",
        "argv": ["python", "-m", "src.layer4.hub.app"],
        "source_sha256": "a" * 64,
    }
    monkeypatch.setattr(manager_module, "assert_campaign_allows", lambda kind, **kwargs: calls.append(("admit", kind, kwargs)))
    monkeypatch.setattr(manager_module, "assert_campaign_training_config", lambda _config: None)
    monkeypatch.setattr(manager_module, "load_campaign_state", lambda _path: {"hub_maintenance": {"maintenance_id": "m"}})
    monkeypatch.setattr(manager_module, "campaign_hub_maintenance_status", lambda _state: {"maintenance_id": "m"})

    def claim(state_path, *, kind, normalized_config, new_hub_identity, resumed_run_id):
        calls.append(("claim", kind, dict(normalized_config), dict(new_hub_identity), resumed_run_id))
        return {"claimed": True}

    monkeypatch.setattr(manager_module, "claim_campaign_hub_maintenance_resume", claim)
    manager = manager_module.OfficialRunManager(
        backend=_Backend(), registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "runs", start_monitor=False,
    )
    monkeypatch.setattr(manager, "_external_simulator_port_base", lambda *_args: 50_000)
    config = OfficialTrainSettings().model_dump(mode="json")
    context = {"normalized_config": {"ignored": "caller copy"}, "new_hub_identity": hub_identity}

    with pytest.raises(RuntimeError, match="stop test after the one-shot claim"):
        manager.create_run(config, "maintenance-resume", maintenance_resume=context)

    claim_calls = [row for row in calls if row[0] == "claim"]
    assert len(claim_calls) == 1
    _, kind, claimed_config, claimed_identity, resumed_run_id = claim_calls[0]
    assert kind == "training"
    assert claimed_config == manager_module.normalize_official_config(config)
    assert claimed_identity == hub_identity
    assert len(resumed_run_id) == 12 and all(character in "0123456789abcdef" for character in resumed_run_id)
    manager.close()


def test_manager_denies_non_owner_before_maintenance_claim(tmp_path, monkeypatch):
    claims = []
    backend = _Backend()
    backend.is_owner = False
    monkeypatch.setattr(manager_module, "claim_campaign_hub_maintenance_resume", lambda *a, **k: claims.append((a, k)))
    manager = manager_module.OfficialRunManager(
        backend=backend, registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "runs", start_monitor=False,
    )

    with pytest.raises(RuntimeError, match="another Mission Control hub owns"):
        manager.create_run(OfficialTrainSettings().model_dump(mode="json"), "duplicate hub")

    assert claims == []
    manager.close()


def test_exact_maintenance_resume_skips_only_frozen_plan_equality(tmp_path, monkeypatch):
    hub_identity = {
        "pid": 2051,
        "process_start_identity": "pid:2051:start_ticks:901",
        "boot_id": "test-boot",
        "executable": "/usr/bin/python",
        "argv": ["python", "-m", "src.layer4.hub.app"],
        "source_sha256": "b" * 64,
    }
    config = OfficialTrainSettings().model_dump(mode="json")
    calibrated_time_cost = 3.0 / 110.255550384521
    config["time_cost_per_simulated_second"] = calibrated_time_cost
    normalized = manager_module.normalize_official_config(config)

    monkeypatch.setattr(manager_module, "assert_campaign_allows", lambda *_a, **_k: None)
    monkeypatch.setattr(manager_module, "load_campaign_state", lambda _path: {"status": "active"})
    monkeypatch.setattr(
        manager_module, "campaign_hub_maintenance_status",
        lambda state: {"maintenance_id": "maintenance-1"} if state else None,
    )
    decisions = []

    def decide(state, kind, *, maintenance_resume=None, **_kwargs):
        decisions.append((kind, maintenance_resume))
        context = maintenance_resume or {}
        exact = (
            kind == "training"
            and context.get("normalized_config") == normalized
            and context.get("new_hub_identity") == hub_identity
        )
        return exact, "authorized" if exact else "not the exact maintenance resume"

    monkeypatch.setattr(manager_module, "campaign_admission_decision", decide)
    plan_checks = []
    monkeypatch.setattr(
        manager_module,
        "assert_campaign_training_config",
        lambda candidate: plan_checks.append(dict(candidate)) or (_ for _ in ()).throw(
            AssertionError("baseline-plan validation should be skipped only for the authorized maintenance resume")
        ),
    )

    def claim(_path, *, kind, normalized_config, new_hub_identity, resumed_run_id):
        assert kind == "training"
        assert normalized_config == normalized
        assert new_hub_identity == hub_identity
        assert len(resumed_run_id) == 12
        return {"claimed": True}

    monkeypatch.setattr(manager_module, "claim_campaign_hub_maintenance_resume", claim)
    manager = manager_module.OfficialRunManager(
        backend=_Backend(), registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "runs", start_monitor=False,
    )
    monkeypatch.setattr(manager, "_external_simulator_port_base", lambda *_args: 50_000)

    with pytest.raises(RuntimeError, match="stop test after the one-shot claim"):
        manager.create_run(
            config,
            "calibrated maintenance resume",
            maintenance_resume={"new_hub_identity": hub_identity},
        )

    assert decisions and decisions[-1][1]["normalized_config"] == normalized
    assert normalized["time_cost_per_simulated_second"] == calibrated_time_cost
    assert plan_checks == []
    manager.close()


def test_changed_or_nonmaintenance_config_keeps_campaign_plan_check(tmp_path, monkeypatch):
    hub_identity = {"pid": 2052, "process_start_identity": "ticks:902"}
    config = OfficialTrainSettings().model_dump(mode="json")
    config["time_cost_per_simulated_second"] = 3.0 / 110.255550384521
    monkeypatch.setattr(manager_module, "assert_campaign_allows", lambda *_a, **_k: None)
    monkeypatch.setattr(manager_module, "load_campaign_state", lambda _path: None)
    monkeypatch.setattr(manager_module, "campaign_hub_maintenance_status", lambda _state: None)
    monkeypatch.setattr(manager_module, "campaign_admission_decision", lambda *_a, **_k: (False, "no active intent"))
    checked = []

    def reject_plan(candidate):
        checked.append(dict(candidate))
        raise RuntimeError("frozen campaign plan mismatch")

    monkeypatch.setattr(manager_module, "assert_campaign_training_config", reject_plan)
    manager = manager_module.OfficialRunManager(
        backend=_Backend(), registry_path=tmp_path / "registry.json",
        runs_root=tmp_path / "runs", start_monitor=False,
    )
    with pytest.raises(RuntimeError, match="frozen campaign plan mismatch"):
        manager.create_run(
            config,
            "unauthorized calibrated config",
            maintenance_resume={"new_hub_identity": hub_identity},
        )
    assert checked and checked[0]["time_cost_per_simulated_second"] == 3.0 / 110.255550384521
    manager.close()
