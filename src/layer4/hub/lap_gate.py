"""Map API persistence for Layer 2's centerline-based lap gate."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.layer2.lap_tracker import map_lap_gate_config
from src.layer4.hub.meshgen.occupancy import load_occupancy
from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir


def get_map_lap_gate(map_id: str, maps_root: Path) -> Dict[str, Any]:
    """Return resolved gate geometry for a map, defaulting to its spawn pose."""
    map_dir = _map_dir(map_id, maps_root)
    config = map_lap_gate_config(
        map_id, repository_root=maps_root.resolve().parent.parent
    )
    if not config.get("supported"):
        return config

    occ = _occupancy_dir(map_dir)
    grid = load_occupancy(occ, _find_yaml(occ))
    pixels = []
    for x, z in config["gate_line"]:
        col = (float(x) - grid.origin_x) / grid.resolution
        row = grid.occupied.shape[0] - (float(z) - grid.origin_z) / grid.resolution
        pixels.append([float(col), float(row)])
    return {
        **config,
        "preview": {
            "width": int(grid.occupied.shape[1]),
            "height": int(grid.occupied.shape[0]),
            "gate_line": pixels,
        },
    }


def save_map_lap_gate(
    map_id: str, maps_root: Path, progress_m: Optional[float]
) -> Dict[str, Any]:
    """Save a route-distance gate override, or clear it to return to spawn."""
    map_dir = _map_dir(map_id, maps_root)
    config = get_map_lap_gate(map_id, maps_root)
    if not config.get("supported"):
        raise ValueError(f"map '{map_id}' has no usable closed centerline lap gate")

    meta_path = _occupancy_dir(map_dir) / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}

    if progress_m is None:
        meta.pop("lap_gate", None)
    else:
        if isinstance(progress_m, bool):
            raise ValueError("progress_m must be a finite number")
        try:
            station = float(progress_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("progress_m must be a finite number") from exc
        if not math.isfinite(station):
            raise ValueError("progress_m must be a finite number")
        if not 0.0 <= station < float(config["route_length_m"]):
            raise ValueError(
                f"progress_m must be between 0 and {config['route_length_m']:.3f} metres"
            )
        meta["lap_gate"] = {
            "progress_m": station,
            "centerline_built_at": config.get("centerline_built_at"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return get_map_lap_gate(map_id, maps_root)


def _map_dir(map_id: str, maps_root: Path) -> Path:
    root = Path(maps_root).resolve()
    map_dir = (root / map_id).resolve()
    if not map_dir.is_relative_to(root):
        raise ValueError(f"invalid map id: {map_id!r}")
    if not map_dir.is_dir():
        raise FileNotFoundError(f"unknown map id: {map_id}")
    return map_dir
