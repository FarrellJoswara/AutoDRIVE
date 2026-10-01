"""Compute and persist vehicle spawn pose for a map (Phase 3/4).

Writes occupancy/meta.json:
  "spawn": { "x": ..., "y": ..., "z": ..., "yaw": ..., "source": "centerline"|"freespace" }

Coordinates are ROS/mesh metres (x, z ground plane; y up). TrackLoader transforms
them through the AiCarTrack align matrix into Unity world space.
Yaw is radians, Unity Y-up: 0 = +Z forward, positive = CCW when viewed from above.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.layer4.hub.centerline import find_centerline, generate_centerline
from src.layer4.hub.meshgen.occupancy import load_occupancy, pixel_to_world
from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir


def ensure_spawn(
    map_id: str,
    maps_root: Path,
    *,
    generate_if_missing: bool = True,
) -> Dict[str, Any]:
    """Ensure meta.json has a spawn; generate centerline first if needed."""
    map_dir = maps_root / map_id
    if not map_dir.is_dir():
        raise FileNotFoundError(f"unknown map id: {map_id}")
    occ = _occupancy_dir(map_dir)
    meta_path = occ / "meta.json"
    meta = _read_meta(meta_path)
    existing = meta.get("spawn")
    if isinstance(existing, dict) and _valid_spawn(existing):
        return dict(existing)

    cl = find_centerline(map_dir)
    if cl is None and generate_if_missing:
        generate_centerline(map_id, maps_root)
        cl = find_centerline(map_dir)
        meta = _read_meta(meta_path)

    spawn: Dict[str, Any]
    if cl is not None:
        spawn = _spawn_from_centerline(cl)
    else:
        spawn = _spawn_from_freespace(occ)

    spawn["built_at"] = datetime.now(timezone.utc).isoformat()
    meta["spawn"] = spawn
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return spawn


def _valid_spawn(s: Dict[str, Any]) -> bool:
    try:
        float(s["x"])
        float(s["z"])
        float(s.get("yaw", 0.0))
        return True
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


def _spawn_from_centerline(path: Path) -> Dict[str, Any]:
    pts: list[Tuple[float, float, float]] = []  # x, z, width
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            z = float(parts[1])
            w = float(parts[2]) if len(parts) >= 3 else 0.0
        except ValueError:
            continue
        pts.append((x, z, w))
    if not pts:
        raise RuntimeError(f"empty centerline: {path}")

    # Prefer the widest corridor, but ignore skeleton tips that leak into
    # huge open free-space outside the circuit (width >> typical lane).
    widths = np.array([p[2] for p in pts], dtype=np.float64)
    positive = widths[widths > 1e-3]
    if positive.size:
        med = float(np.median(positive))
        cap = max(med * 2.5, med + 0.75)
        eligible = np.where((widths > 1e-3) & (widths <= cap))[0]
        if eligible.size == 0:
            eligible = np.where(widths > 1e-3)[0]
        i0 = int(eligible[int(np.argmax(widths[eligible]))])
    else:
        i0 = min(2, len(pts) - 1)

    i1 = i0 + 1 if i0 + 1 < len(pts) else max(0, i0 - 1)
    x0, z0, _w = pts[i0]
    x1, z1, _ = pts[i1]
    dx, dz = x1 - x0, z1 - z0
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        # Fall back to a farther neighbour for heading.
        for j in (i0 + 2, i0 - 2, 0, len(pts) - 1):
            if 0 <= j < len(pts) and j != i0:
                x1, z1, _ = pts[j]
                dx, dz = x1 - x0, z1 - z0
                if abs(dx) > 1e-6 or abs(dz) > 1e-6:
                    break
    yaw = math.atan2(dx, dz)  # Unity: 0 = +Z, CCW about +Y
    return {
        "x": round(x0, 6),
        "y": 0.05,
        "z": round(z0, 6),
        "yaw": round(yaw, 6),
        "source": "centerline",
        "file": path.name,
        "width_m": round(float(_w), 6),
        "index": i0,
    }


def _spawn_from_freespace(occ: Path) -> Dict[str, Any]:
    grid = load_occupancy(occ, _find_yaml(occ))
    free = (~grid.occupied).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
    x, z = pixel_to_world(grid, float(c), float(r))
    return {
        "x": round(float(x), 6),
        "y": 0.05,
        "z": round(float(z), 6),
        "yaw": 0.0,
        "source": "freespace",
    }
