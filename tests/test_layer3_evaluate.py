from __future__ import annotations

from src.layer3 import evaluate


def test_evaluate_cli_forwards_throttle_mode(monkeypatch) -> None:
    received = {}

    def fake_evaluate_policy(model, **kwargs):
        received.update(kwargs)
        return {}

    monkeypatch.setattr(evaluate, "evaluate_policy", fake_evaluate_policy)

    result = evaluate.main([
        "--model", "checkpoint.zip",
        "--map-id", "porto",
        "--throttle-mode", "forward_only",
    ])

    assert result == 0
    assert received["throttle_mode"] == "forward_only"
