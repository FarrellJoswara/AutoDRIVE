"""Wavefront OBJ read/write (triangles only)."""

from __future__ import annotations

from pathlib import Path

from .extrude import TriangleMesh


def write_obj(path: Path, mesh: TriangleMesh, *, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# AiCar meshgen", f"# verts={len(mesh.vertices)} faces={len(mesh.faces)}"]
    if comment:
        lines.append(f"# {comment}")
    has_uv = mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices)
    for x, y, z in mesh.vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    if has_uv:
        for u, v in mesh.uvs:
            lines.append(f"vt {u:.6f} {v:.6f}")
    for a, b, c in mesh.faces:
        if has_uv:
            # 1-based indices; vt shares vertex index (per-vertex UVs).
            lines.append(
                f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}"
            )
        else:
            lines.append(f"f {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_obj_stats(path: Path) -> tuple[int, int]:
    """Return (n_verts, n_faces)."""
    nv = nf = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            nv += 1
        elif line.startswith("f "):
            nf += 1
    return nv, nf
