#!/usr/bin/env python3
"""Write side-by-side occupancy + mesh top-down previews to logs/layer4/map_previews/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.layer4.hub.meshgen.occupancy import load_occupancy
from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir


def _occ_preview(occupied: np.ndarray, size: int = 520) -> Image.Image:
    h, w = occupied.shape
    gray = np.where(occupied, 40, 220).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").convert("RGB")
    if w >= h:
        img = img.resize((size, max(1, int(size * h / w))), Image.NEAREST)
    else:
        img = img.resize((max(1, int(size * w / h)), size), Image.NEAREST)
    return img


def _parse_obj(obj_path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                idx.append(int(tok.split("/")[0]) - 1)
            if len(idx) >= 3:
                faces.append((idx[0], idx[1], idx[2]))
    return np.asarray(verts, dtype=np.float64), faces


def _mesh_preview(obj_path: Path, size: int = 520) -> Image.Image:
    """Top-down XZ: draw every triangle edge so walls look continuous."""
    img = Image.new("RGB", (size, size), (24, 26, 30))
    draw = ImageDraw.Draw(img)
    if not obj_path.is_file():
        draw.text((12, 12), "no mesh", fill=(220, 80, 80))
        return img

    verts, faces = _parse_obj(obj_path)
    if len(verts) == 0:
        draw.text((12, 12), "empty mesh", fill=(220, 80, 80))
        return img

    xs = verts[:, 0]
    zs = verts[:, 2]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_z, max_z = float(zs.min()), float(zs.max())
    pad = 0.08
    span = max(max_x - min_x, max_z - min_z, 1e-3) * (1 + pad)
    cx = (min_x + max_x) / 2
    cz = (min_z + max_z) / 2

    def to_px(x: float, z: float) -> tuple[int, int]:
        u = int((x - cx) / span * (size - 20) + size / 2)
        v = int(size / 2 - (z - cz) / span * (size - 20))
        return u, v

    # Unique undirected edges from faces
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edges.add((i, j) if i < j else (j, i))

    for i, j in edges:
        p0 = to_px(float(verts[i, 0]), float(verts[i, 2]))
        p1 = to_px(float(verts[j, 0]), float(verts[j, 2]))
        draw.line([p0, p1], fill=(70, 170, 255), width=1)

    draw.text(
        (8, 8),
        f"{len(verts)} verts  {len(faces)} faces  {len(edges)} edges",
        fill=(190, 190, 190),
    )
    return img


def main() -> int:
    out = ROOT / "logs" / "layer4" / "map_previews"
    out.mkdir(parents=True, exist_ok=True)
    maps_root = ROOT / "simulator" / "maps"
    for map_id in ("porto", "berlin", "icra2026_classic", "icra2026_master"):
        map_dir = maps_root / map_id
        if not map_dir.is_dir():
            continue
        occ = _occupancy_dir(map_dir)
        grid = load_occupancy(occ, _find_yaml(occ))
        left = _occ_preview(grid.occupied)
        obj = map_dir / "mesh" / "track.obj"
        right = _mesh_preview(obj)
        h = max(left.height, right.height)
        canvas = Image.new("RGB", (left.width + right.width + 16, h + 8), (18, 18, 20))
        canvas.paste(left, (4, 4))
        canvas.paste(right, (left.width + 12, 4))
        path = out / f"{map_id}_preview.png"
        canvas.save(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
