"""Extrude polylines into wall ribbon meshes (visual + simplified collider)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .contours import decimate_polyline


@dataclass
class TriangleMesh:
    vertices: np.ndarray  # (V, 3) float64 — X,Y,Z with Y up
    faces: np.ndarray  # (F, 3) int64
    uvs: np.ndarray | None = None  # (V, 2) float64 — optional duct tiling UVs


def _perp2d(dx: float, dz: float) -> Tuple[float, float]:
    n = np.hypot(dx, dz)
    if n < 1e-9:
        return 0.0, 0.0
    return -dz / n, dx / n


def extrude_wall_ribbons(
    polylines: Sequence[np.ndarray],
    *,
    height_m: float = 0.35,
    thickness_m: float = 0.08,
    y_base: float = 0.0,
) -> TriangleMesh:
    """Build vertical wall quads along each polyline (thin prism per segment)."""
    verts: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    half = thickness_m * 0.5
    y0 = y_base
    y1 = y_base + height_m

    def add_vert(x: float, y: float, z: float, u: float, v: float) -> int:
        verts.append((x, y, z))
        uvs.append((u, v))
        return len(verts) - 1

    for poly in polylines:
        if len(poly) < 2:
            continue
        # Arc-length parameter along polyline for U (metres → UV repeats).
        cum = [0.0]
        for i in range(len(poly) - 1):
            d = float(np.hypot(poly[i + 1, 0] - poly[i, 0], poly[i + 1, 1] - poly[i, 1]))
            cum.append(cum[-1] + d)
        for i in range(len(poly) - 1):
            x0, z0 = float(poly[i, 0]), float(poly[i, 1])
            x1, z1 = float(poly[i + 1, 0]), float(poly[i + 1, 1])
            nx, nz = _perp2d(x1 - x0, z1 - z0)
            if nx == 0.0 and nz == 0.0:
                continue
            ax0, az0 = x0 + nx * half, z0 + nz * half
            bx0, bz0 = x0 - nx * half, z0 - nz * half
            ax1, az1 = x1 + nx * half, z1 + nz * half
            bx1, bz1 = x1 - nx * half, z1 - nz * half
            u0, u1 = cum[i], cum[i + 1]
            v0, v1 = 0.0, float(height_m)

            i_a0 = add_vert(ax0, y0, az0, u0, v0)
            i_b0 = add_vert(bx0, y0, bz0, u0, v0)
            i_a1 = add_vert(ax1, y0, az1, u1, v0)
            i_b1 = add_vert(bx1, y0, bz1, u1, v0)
            i_a0t = add_vert(ax0, y1, az0, u0, v1)
            i_b0t = add_vert(bx0, y1, bz0, u0, v1)
            i_a1t = add_vert(ax1, y1, az1, u1, v1)
            i_b1t = add_vert(bx1, y1, bz1, u1, v1)

            faces.append((i_a0, i_a1, i_a1t))
            faces.append((i_a0, i_a1t, i_a0t))
            faces.append((i_b0, i_b0t, i_b1t))
            faces.append((i_b0, i_b1t, i_b1))
            faces.append((i_a0t, i_a1t, i_b1t))
            faces.append((i_a0t, i_b1t, i_b0t))

    if not verts:
        return TriangleMesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            uvs=np.zeros((0, 2), dtype=np.float64),
        )
    return TriangleMesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs=np.asarray(uvs, dtype=np.float64),
    )


def build_visual_and_collider(
    polylines: Sequence[np.ndarray],
    *,
    height_m: float = 1.0,
    thickness_m: float = 0.12,
    collider_stride: int = 1,
) -> Tuple[TriangleMesh, TriangleMesh]:
    visual = extrude_wall_ribbons(
        polylines, height_m=height_m, thickness_m=thickness_m
    )
    simplified = [decimate_polyline(p, collider_stride) for p in polylines]
    collider = extrude_wall_ribbons(
        simplified, height_m=height_m, thickness_m=thickness_m
    )
    return visual, collider
