"""Orthographic top-down mesh preview (no Unity) — Phase 4.

Renders track.obj into the same image-frame orientation as the Fleet canvas /
occupancy PNG (ROS: row 0 = top = high world Z).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.layer4.hub.meshgen.occupancy import load_occupancy, parse_ros_yaml
from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir

PREVIEW_FILENAME = "preview.png"


@dataclass
class MeshPreviewResult:
    map_id: str
    path: Path
    url: str
    n_verts: int
    n_faces: int
    size_px: int

    def to_api(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def mesh_preview_path(map_dir: Path) -> Path:
    return map_dir / "mesh" / PREVIEW_FILENAME


def find_mesh_preview(map_dir: Path) -> Optional[Path]:
    p = mesh_preview_path(map_dir)
    return p if p.is_file() else None


def _parse_obj(obj_path: Path) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            idx: List[int] = []
            for tok in line.split()[1:]:
                idx.append(int(tok.split("/")[0]) - 1)
            if len(idx) >= 3:
                faces.append((idx[0], idx[1], idx[2]))
    return np.asarray(verts, dtype=np.float64), faces


def _world_to_image_px(
    x: float,
    z: float,
    *,
    origin_x: float,
    origin_z: float,
    resolution: float,
    height_px: int,
) -> Tuple[float, float]:
    """World (x, z) → image (col, row) with row 0 = top (matches canvas / PNG)."""
    col = (x - origin_x) / resolution
    row = height_px - (z - origin_z) / resolution
    return col, row


def render_mesh_ortho(
    obj_path: Path,
    *,
    size: int = 512,
    bg: Tuple[int, int, int] = (24, 26, 30),
    edge_color: Tuple[int, int, int] = (70, 170, 255),
    occupancy_underlay: Optional[Path] = None,
    origin_x: Optional[float] = None,
    origin_z: Optional[float] = None,
    resolution: Optional[float] = None,
    height_px: Optional[int] = None,
    width_px: Optional[int] = None,
) -> Tuple[Image.Image, int, int]:
    """Top-down mesh wireframe in occupancy image orientation (not free AABB flip)."""
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    if not obj_path.is_file():
        draw.text((12, 12), "no mesh", fill=(220, 80, 80))
        return img, 0, 0

    verts, faces = _parse_obj(obj_path)
    if len(verts) == 0:
        draw.text((12, 12), "empty mesh", fill=(220, 80, 80))
        return img, 0, 0

    use_ros = (
        origin_x is not None
        and origin_z is not None
        and resolution is not None
        and height_px is not None
        and width_px is not None
        and resolution > 0
        and height_px > 0
        and width_px > 0
    )

    if use_ros:
        assert origin_x is not None and origin_z is not None
        assert resolution is not None and height_px is not None and width_px is not None
        # Square letterbox of the occupancy frame (same aspect handling as canvas fit).
        scale = (size - 20) / float(max(width_px, height_px))
        ox = (size - width_px * scale) / 2.0
        oy = (size - height_px * scale) / 2.0

        if occupancy_underlay is not None and occupancy_underlay.is_file():
            try:
                under = Image.open(occupancy_underlay).convert("L").resize(
                    (max(1, int(width_px * scale)), max(1, int(height_px * scale))),
                    Image.Resampling.NEAREST,
                )
                under_rgb = Image.merge("RGB", (under, under, under))
                # Darken so blue edges stay readable
                under_rgb = under_rgb.point(lambda p: int(p * 0.35))
                img.paste(under_rgb, (int(ox), int(oy)))
                draw = ImageDraw.Draw(img)
            except OSError:
                pass

        def to_px(x: float, z: float) -> Tuple[int, int]:
            col, row = _world_to_image_px(
                x,
                z,
                origin_x=origin_x,
                origin_z=origin_z,
                resolution=resolution,
                height_px=height_px,
            )
            return int(ox + col * scale), int(oy + row * scale)

    else:
        # Fallback AABB (legacy) — still Z-up on screen to match canvas world view.
        xs = verts[:, 0]
        zs = verts[:, 2]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_z, max_z = float(zs.min()), float(zs.max())
        pad = 0.08
        span = max(max_x - min_x, max_z - min_z, 1e-3) * (1 + pad)
        cx = (min_x + max_x) / 2
        cz = (min_z + max_z) / 2

        def to_px(x: float, z: float) -> Tuple[int, int]:
            u = int((x - cx) / span * (size - 20) + size / 2)
            v = int(size / 2 - (z - cz) / span * (size - 20))
            return u, v

    edges: set[Tuple[int, int]] = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add((i, j) if i < j else (j, i))

    edge_list = list(edges)
    if len(edge_list) > 80000:
        step = max(1, len(edge_list) // 80000)
        edge_list = edge_list[::step]

    for i, j in edge_list:
        p0 = to_px(float(verts[i, 0]), float(verts[i, 2]))
        p1 = to_px(float(verts[j, 0]), float(verts[j, 2]))
        draw.line([p0, p1], fill=edge_color, width=1)

    draw.text(
        (8, 8),
        f"{len(verts)}v {len(faces)}f",
        fill=(190, 190, 190),
    )
    return img, len(verts), len(faces)


def generate_mesh_preview(
    map_id: str,
    maps_root: Path,
    *,
    size: int = 512,
) -> MeshPreviewResult:
    """Render mesh/track.obj → mesh/preview.png (requires generate-mesh first)."""
    map_dir = maps_root / map_id
    if not map_dir.is_dir():
        raise FileNotFoundError(f"unknown map id: {map_id}")
    obj = map_dir / "mesh" / "track.obj"
    if not obj.is_file():
        col = map_dir / "mesh" / "track_col.obj"
        if col.is_file():
            obj = col
        else:
            raise FileNotFoundError(
                f"no track.obj for '{map_id}' — run generate-mesh first"
            )

    occ = _occupancy_dir(map_dir)
    underlay: Optional[Path] = None
    origin_x = origin_z = resolution = None
    height_px = width_px = None
    try:
        yaml_path = _find_yaml(occ)
        grid = load_occupancy(occ, yaml_path)
        origin_x, origin_z = grid.origin_x, grid.origin_z
        resolution = grid.resolution
        height_px, width_px = int(grid.occupied.shape[0]), int(grid.occupied.shape[1])
        meta = parse_ros_yaml(yaml_path.read_text(encoding="utf-8"))
        img_name = str(meta["image"])
        candidate = occ / img_name
        if candidate.is_file():
            underlay = candidate
    except (OSError, ValueError, FileNotFoundError, KeyError):
        pass

    img, nv, nf = render_mesh_ortho(
        obj,
        size=size,
        occupancy_underlay=underlay,
        origin_x=origin_x,
        origin_z=origin_z,
        resolution=resolution,
        height_px=height_px,
        width_px=width_px,
    )
    out = mesh_preview_path(map_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")

    _update_meta_preview(map_dir, {
        "status": "ready",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "file": f"mesh/{PREVIEW_FILENAME}",
        "size_px": size,
        "n_verts": nv,
        "n_faces": nf,
        "frame": "occupancy_image",
    })

    return MeshPreviewResult(
        map_id=map_id,
        path=out,
        url=f"/maps/{map_id}/mesh/{PREVIEW_FILENAME}",
        n_verts=nv,
        n_faces=nf,
        size_px=size,
    )


def _update_meta_preview(map_dir: Path, info: Dict[str, Any]) -> None:
    occ = _occupancy_dir(map_dir)
    meta_path = occ / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta["mesh_preview"] = info
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def ensure_occupancy_thumbnail(
    map_id: str,
    maps_root: Path,
    *,
    max_edge: int = 256,
) -> Optional[Path]:
    """Ensure occupancy/preview.png exists (downscale of map image)."""
    from PIL import Image as PILImage

    map_dir = maps_root / map_id
    occ = _occupancy_dir(map_dir)
    out = occ / "preview.png"
    if out.is_file():
        return out
    yaml_path = _find_yaml(occ)
    parsed = parse_ros_yaml(yaml_path.read_text(encoding="utf-8"))
    image_path = occ / str(parsed["image"])
    if not image_path.is_file():
        return None
    img = PILImage.open(image_path)
    w, h = img.size
    scale = min(1.0, float(max_edge) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            PILImage.Resampling.BILINEAR,
        )
    img.save(out, format="PNG")
    return out
