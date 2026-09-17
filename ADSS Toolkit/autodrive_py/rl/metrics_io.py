"""Metrics + leaderboard helpers (contracts.md §6–7)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import CONTRACTS_VERSION

LEADERBOARD_HEADER = [
    "run_id",
    "policy",
    "backend",
    "adjusted_time",
    "mean_lap_time",
    "total_collisions",
    "n_episodes",
    "timestamp",
    "contracts_version",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_metrics(
    *,
    run_id: str,
    policy: str,
    backend: str,
    mean_lap_time: float | None,
    total_collisions: int,
    n_episodes: int,
    tracks_eval: list[str],
    **extra: Any,
) -> dict:
    adjusted = None
    if mean_lap_time is not None:
        adjusted = float(mean_lap_time) + 10.0 * int(total_collisions)
    out = {
        "contracts_version": CONTRACTS_VERSION,
        "run_id": run_id,
        "policy": policy,
        "backend": backend,
        "mean_lap_time": mean_lap_time,
        "total_collisions": int(total_collisions),
        "adjusted_time": adjusted,
        "n_episodes": int(n_episodes),
        "tracks_eval": list(tracks_eval),
        "timestamp": utc_now(),
    }
    out.update(extra)
    return out


def write_run_artifacts(
    models_root: Path,
    run_id: str,
    config: dict,
    metrics: dict,
    train_tracks: list[str] | None = None,
) -> Path:
    run_dir = models_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tracks = train_tracks or metrics.get("tracks_eval") or []
    (run_dir / "train_tracks.txt").write_text(
        "\n".join(str(t) for t in tracks) + ("\n" if tracks else ""),
        encoding="utf-8",
    )
    append_leaderboard(models_root / "leaderboard.csv", metrics)
    return run_dir


def append_leaderboard(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(
            {
                "run_id": metrics.get("run_id"),
                "policy": metrics.get("policy"),
                "backend": metrics.get("backend"),
                "adjusted_time": metrics.get("adjusted_time"),
                "mean_lap_time": metrics.get("mean_lap_time"),
                "total_collisions": metrics.get("total_collisions"),
                "n_episodes": metrics.get("n_episodes"),
                "timestamp": metrics.get("timestamp"),
                "contracts_version": metrics.get("contracts_version", CONTRACTS_VERSION),
            }
        )


def read_leaderboard(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
