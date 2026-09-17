"""Official held-out eval API for train / compare / operator siblings.

Public API
----------
load_protocol(path: Path | None = None) -> dict
protocol_map_ids(protocol=None) -> list[str]
sealed_map_ids(protocol=None) -> list[str]
is_holdout_map(map_id: str, protocol=None) -> bool
map_file_hash(map_yaml: Path) -> str

eval_ftg_protocol(*, protocol=None, maps_root=None, n_lidar=180, episodes_override=None) -> dict
eval_ppo_protocol(model_path, *, protocol=None, maps_root=None, n_lidar=180, run_id=None, episodes_override=None) -> dict
    Both return metrics with kind=\"official\", adjusted_time = lap + 10·collisions
    (or protocol timeout_s + 10·cols when no scored lap).

race_score_key(metrics: dict) -> tuple
    Sort key: finishers ahead of DNFs, then lower adjusted_time.
row_race_score_key(row: dict) -> tuple
    Same ranking for leaderboard CSV rows.
protocol_timeout_s() -> float
    Official episode budget from eval_protocol.yaml (400 under official_v2).

CLI: python -m rl.eval_cli --official --policy ftg|ppo [--model PATH] [--episodes N]
"""

from __future__ import annotations

from .eval_protocol import (
    PROTOCOL_PATH,
    eval_ftg_protocol,
    eval_ppo_protocol,
    is_holdout_map,
    load_protocol,
    map_file_hash,
    protocol_map_ids,
    protocol_timeout_s,
    race_score_key,
    row_race_score_key,
    sealed_map_ids,
)

__all__ = [
    "PROTOCOL_PATH",
    "eval_ftg_protocol",
    "eval_ppo_protocol",
    "is_holdout_map",
    "load_protocol",
    "map_file_hash",
    "protocol_map_ids",
    "protocol_timeout_s",
    "race_score_key",
    "row_race_score_key",
    "sealed_map_ids",
]
