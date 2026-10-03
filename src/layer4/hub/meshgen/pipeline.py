"""End-to-end occupancy → OBJ mesh generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .contours import extract_wall_polylines
from .extrude import build_visual_and_collider
from .obj_io import write_obj
from .occupancy import load_occupancy


@dataclass
class MeshGenOptions:
    wall_height_m: float = 1.0
    wall_thickness_m: float = 0.12
    collider_stride: int = 1
    min_area_px: float = 40.0
    simplify_eps_px: float = 1.5
    morph_close_k: int = 3
    max_contours: int = 24


@dataclass
class MeshResult:
    map_id: str
    track_obj: Path
    track_col_obj: Path
    n_contours: int
    n_verts_visual: int
    n_faces_visual: int
    n_verts_collider: int
    n_faces_collider: int

    def to_api(self) -> Dict[str, Any]:
        d = asdict(self)
        d["track_obj"] = str(self.track_obj)
        d["track_col_obj"] = str(self.track_col_obj)
        return d


def _occupancy_dir(map_dir: Path) -> Path:
    nested = map_dir / "occupancy"
    return nested if nested.is_dir() else map_dir


def _find_yaml(occ: Path) -> Path:
    meta_path = occ / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except (OSError, json.JSONDecodeError):
            pass
    preferred = meta.get("map_yaml")
    if isinstance(preferred, str) and (occ / preferred).is_file():
        return occ / preferred
    for name in ("map.yaml", "Map.yaml"):
        if (occ / name).is_file():
            return occ / name
    yamls = sorted(occ.glob("*.yaml")) + sorted(occ.glob("*.yml"))
    for p in yamls:
        if p.name.lower() != "meta.yaml":
            return p
    raise FileNotFoundError(f"no ROS yaml in {occ}")


def _update_meta(occ: Path, mesh_info: Dict[str, Any]) -> None:
    meta_path = occ / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except (OSError, json.JSONDecodeError):
            pass
    meta["mesh"] = mesh_info
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def generate_mesh(
    map_id: str,
    maps_root: Path,
    opts: Optional[MeshGenOptions] = None,
) -> MeshResult:
    """Generate track.obj + track_col.obj under maps/<id>/mesh/."""
    opts = opts or MeshGenOptions()
    map_dir = maps_root / map_id
    if not map_dir.is_dir():
        raise FileNotFoundError(f"unknown map id: {map_id}")
    occ = _occupancy_dir(map_dir)
    yaml_path = _find_yaml(occ)
    grid = load_occupancy(occ, yaml_path)

    polylines = extract_wall_polylines(
        grid,
        min_area_px=opts.min_area_px,
        simplify_eps_px=opts.simplify_eps_px,
        morph_close_k=opts.morph_close_k,
    )
    if opts.max_contours > 0:
        polylines = polylines[: opts.max_contours]
    if not polylines:
        raise RuntimeError(f"no wall contours found for map {map_id}")

    visual, collider = build_visual_and_collider(
        polylines,
        height_m=opts.wall_height_m,
        thickness_m=opts.wall_thickness_m,
        collider_stride=opts.collider_stride,
    )
    if len(visual.vertices) == 0:
        raise RuntimeError(f"empty mesh for map {map_id}")

    mesh_dir = map_dir / "mesh"
    track_obj = mesh_dir / "track.obj"
    track_col = mesh_dir / "track_col.obj"
    write_obj(track_obj, visual, comment=f"map={map_id} visual")
    write_obj(track_col, collider, comment=f"map={map_id} collider")

    built_at = datetime.now(timezone.utc).isoformat()
    _update_meta(
        occ,
        {
            "status": "ready",
            "built_at": built_at,
            "wall_height_m": opts.wall_height_m,
            "wall_thickness_m": opts.wall_thickness_m,
            "collider_stride": opts.collider_stride,
            "n_contours": len(polylines),
            "n_verts_visual": int(len(visual.vertices)),
            "n_faces_visual": int(len(visual.faces)),
            "n_verts_collider": int(len(collider.vertices)),
            "n_faces_collider": int(len(collider.faces)),
            "track_obj": "mesh/track.obj",
            "track_col_obj": "mesh/track_col.obj",
        },
    )

    try:
        from src.layer4.hub.map_align import ensure_align

        ensure_align(map_id, maps_root, persist=True)
    except Exception:
        pass

    return MeshResult(
        map_id=map_id,
        track_obj=track_obj,
        track_col_obj=track_col,
        n_contours=len(polylines),
        n_verts_visual=int(len(visual.vertices)),
        n_faces_visual=int(len(visual.faces)),
        n_verts_collider=int(len(collider.vertices)),
        n_faces_collider=int(len(collider.faces)),
    )
