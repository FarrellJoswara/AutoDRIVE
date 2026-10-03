"""CLI: python -m src.layer4.hub.meshgen --map-id porto"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.layer4.hub.maps_catalog import resolve_maps_root
from src.layer4.hub.meshgen.pipeline import MeshGenOptions, generate_mesh
from src.layer4.settings import ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate track OBJ meshes from occupancy maps")
    p.add_argument("--map-id", required=True, help="Folder name under simulator/maps/")
    p.add_argument(
        "--maps-root",
        type=Path,
        default=None,
        help="Override maps root (default: AICAR_MAPS_DIR / simulator/maps)",
    )
    p.add_argument("--wall-height", type=float, default=1.0)
    p.add_argument("--wall-thickness", type=float, default=0.12)
    p.add_argument("--collider-stride", type=int, default=1)
    p.add_argument("--min-area-px", type=float, default=40.0)
    p.add_argument("--max-contours", type=int, default=24)
    args = p.parse_args(argv)

    root = args.maps_root or resolve_maps_root(ROOT)
    if root is None:
        print("maps root not found", file=sys.stderr)
        return 1
    opts = MeshGenOptions(
        wall_height_m=args.wall_height,
        wall_thickness_m=args.wall_thickness,
        collider_stride=args.collider_stride,
        min_area_px=args.min_area_px,
        max_contours=args.max_contours,
    )
    try:
        result = generate_mesh(args.map_id, root, opts)
    except Exception as exc:
        print(f"meshgen failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"OK {result.map_id}: contours={result.n_contours} "
        f"visual={result.n_verts_visual}v/{result.n_faces_visual}f "
        f"collider={result.n_verts_collider}v/{result.n_faces_collider}f"
    )
    print(f"  {result.track_obj}")
    print(f"  {result.track_col_obj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
