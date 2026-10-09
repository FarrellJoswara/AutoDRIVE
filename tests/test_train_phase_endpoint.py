import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.layer4.hub import app as hub_app


def test_behavior_cloning_ready_phase_is_accepted_and_published(monkeypatch):
    payload = {
        "kind": "phase",
        "phase": "behavior_cloning_ready",
        "run_id": "bc-test",
        "step": 0,
        "checkpoint": "behavior_cloned_policy.zip",
        "transitions": 4000,
    }
    ingested = []
    published = []
    monkeypatch.setattr(hub_app.official_runs, "ingest_telemetry", ingested.append)
    monkeypatch.setattr(hub_app.bus, "publish_train_phase", published.append)

    async def json_payload():
        return payload

    response = asyncio.run(hub_app.post_train_phase(SimpleNamespace(json=json_payload)))

    assert response == {"ok": "1"}
    assert ingested == [payload]
    assert published == [payload]


def test_unknown_training_phase_is_rejected():
    async def json_payload():
        return {"kind": "phase", "phase": "not-a-real-phase"}

    with pytest.raises(HTTPException, match="invalid training phase"):
        asyncio.run(hub_app.post_train_phase(SimpleNamespace(json=json_payload)))
