"""TrackLoader AiCarTrack align metadata (ros_identity).

TrackLoader places occupancy meshes in ROS/map metres (identity XZ: scale=1,
tx=tz=0; only Y is lifted to duct floor). Unity world XZ ≈ ROS yaml metres.
Watch plots published fleet poses directly — no client-side Unity→map remap.

Deprecated: older players matched mesh XZ AABB to builtin Porto/Berlin duct.
After installing a TrackLoader build with ros_identity align, builtin_aabb is
obsolete — do not reintroduce canvas-only hacks for the AABB transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

# Identity: Unity world XZ == mesh/ROS metres (current TrackLoader).
_IDENTITY = {"scale": 1.0, "tx": 0.0, "tz": 0.0}


def ensure_align(map_id: str, maps_root: Path, *, persist: bool = True) -> Dict[str, float]:
    """Return {scale, tx, tz}; rewrite meta.align to ros_identity when needed."""
    occ = maps_root / map_id / "occupancy"
    meta_path = occ / "meta.json"
    meta = _read_meta(meta_path)
    existing = meta.get("align")
    align = compute_align(map_id, maps_root / map_id)

    # Keep existing only if already identity; upgrade stale builtin_aabb.
    if isinstance(existing, dict) and _valid_align(existing) and _is_identity(existing):
        return {
            "scale": float(existing["scale"]),
            "tx": float(existing["tx"]),
            "tz": float(existing["tz"]),
        }

    if persist and occ.is_dir():
        meta["align"] = {
            **align,
            "source": "ros_identity",
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return align


def compute_align(map_id: str, map_dir: Path) -> Dict[str, float]:
    """Match TrackLoader.AlignTrackTransform — identity XZ (no yaw)."""
    del map_id, map_dir
    return dict(_IDENTITY)


def unity_to_map(align: Dict[str, float], x: float, z: float) -> Tuple[float, float]:
    """Unity world XZ → mesh/ROS metres (no-op when align is identity)."""
    s = float(align.get("scale", 1.0)) or 1.0
    tx = float(align.get("tx", 0.0))
    tz = float(align.get("tz", 0.0))
    return (x - tx) / s, (z - tz) / s


def _is_identity(a: Dict[str, Any]) -> bool:
    try:
        return (
            abs(float(a["scale"]) - 1.0) < 1e-4
            and abs(float(a["tx"])) < 1e-4
            and abs(float(a["tz"])) < 1e-4
        )
    except (KeyError, TypeError, ValueError):
        return False


def _valid_align(a: Dict[str, Any]) -> bool:
    try:
        s = float(a["scale"])
        float(a["tx"])
        float(a["tz"])
        return s > 1e-6
    except (KeyError, TypeError, ValueError):
        return False


def _read_meta(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}
