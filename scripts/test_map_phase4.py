#!/usr/bin/env python3
"""Unit tests for centerline + mesh preview (Phase 4)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layer4.hub.centerline import (  # noqa: E402
    find_centerline,
    generate_centerline,
)
from src.layer4.hub.map_status import map_status_payload  # noqa: E402
from src.layer4.hub.map_activate import write_active_map  # noqa: E402
from src.layer4.hub.mesh_preview import generate_mesh_preview  # noqa: E402
from src.layer4.hub.meshgen.obj_io import write_obj  # noqa: E402
from src.layer4.hub.meshgen.extrude import TriangleMesh  # noqa: E402


def _write_ring_map(root: Path, map_id: str = "ring") -> Path:
    """Small free-space ring so skeletonization has a path."""
    occ = root / map_id / "occupancy"
    occ.mkdir(parents=True)
    h = w = 64
    gray = np.full((h, w), 255, dtype=np.uint8)  # free
    # Outer and inner walls (dark = occupied in ROS convention)
    gray[4:8, :] = 0
    gray[-8:-4, :] = 0
    gray[:, 4:8] = 0
    gray[:, -8:-4] = 0
    gray[28:36, 28:36] = 0  # center island
    # Write PGM
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    (occ / "map.pgm").write_bytes(header + gray.tobytes())
    (occ / "map.yaml").write_text(
        "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n",
        encoding="utf-8",
    )
    (occ / "meta.json").write_text(json.dumps({"label": "Ring"}), encoding="utf-8")
    return root / map_id


def _write_tiny_mesh(map_dir: Path) -> None:
    mesh_dir = map_dir / "mesh"
    mesh_dir.mkdir(parents=True)
    # Simple triangle in XZ
    verts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.35, 2.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    write_obj(mesh_dir / "track.obj", TriangleMesh(vertices=verts, faces=faces))


class CenterlinePreviewTest(unittest.TestCase):
    def test_generate_centerline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_ring_map(root, "ring")
            result = generate_centerline("ring", root, spacing_m=0.2)
            self.assertGreater(result.n_points, 4)
            path = find_centerline(root / "ring")
            self.assertIsNotNone(path)
            assert path is not None
            text = path.read_text(encoding="utf-8")
            self.assertIn("x_m", text)
            self.assertGreater(len(text.splitlines()), 5)

    def test_mesh_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            map_dir = _write_ring_map(root, "ring")
            _write_tiny_mesh(map_dir)
            result = generate_mesh_preview("ring", root, size=128)
            self.assertTrue(result.path.is_file())
            self.assertEqual(result.n_verts, 3)
            self.assertTrue(result.url.endswith("/mesh/preview.png"))

    def test_selection_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_ring_map(root, "ring")
            write_active_map(root, "ring")
            st = map_status_payload(root, selected_id="porto")
            self.assertTrue(st["selection_mismatch"])
            self.assertTrue(st["warnings"])
            ok = map_status_payload(root, selected_id="ring")
            self.assertFalse(ok["selection_mismatch"])


if __name__ == "__main__":
    unittest.main()
