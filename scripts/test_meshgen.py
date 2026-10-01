#!/usr/bin/env python3
"""Tests for occupancy → OBJ meshgen (needs numpy, Pillow, opencv)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_rect_map(root: Path, map_id: str = "toy") -> Path:
    """Hollow rectangle wall in a small occupancy grid."""
    occ = root / map_id / "occupancy"
    occ.mkdir(parents=True)
    h, w = 80, 100
    img = np.full((h, w), 254, dtype=np.uint8)  # free
    img[10:15, 10:90] = 0
    img[65:70, 10:90] = 0
    img[10:70, 10:15] = 0
    img[10:70, 85:90] = 0
    Image.fromarray(img, mode="L").save(occ / "map.png")
    (occ / "map.yaml").write_text(
        "image: map.png\nresolution: 0.05\norigin: [-2.0, -2.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
        encoding="utf-8",
    )
    (occ / "meta.json").write_text(
        json.dumps({"label": "Toy"}), encoding="utf-8"
    )
    return root


class MeshgenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import cv2  # noqa: F401
            from PIL import Image as _I  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"meshgen deps missing: {exc}") from exc

    def test_toy_rectangle_generates_obj(self) -> None:
        from src.layer4.hub.meshgen import generate_mesh

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_rect_map(Path(tmp))
            result = generate_mesh("toy", root)
            self.assertTrue(result.track_obj.is_file())
            self.assertTrue(result.track_col_obj.is_file())
            self.assertGreater(result.n_contours, 0)
            self.assertGreater(result.n_verts_visual, 0)
            self.assertGreaterEqual(result.n_verts_visual, result.n_verts_collider)
            meta = json.loads(
                (root / "toy" / "occupancy" / "meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["mesh"]["status"], "ready")
            # Bounds roughly in metres near origin ± a few metres
            text = result.track_obj.read_text(encoding="utf-8")
            xs, zs = [], []
            for line in text.splitlines():
                if line.startswith("v "):
                    _, x, _y, z = line.split()[:4]
                    xs.append(float(x))
                    zs.append(float(z))
            self.assertTrue(xs and zs)
            self.assertLess(max(xs) - min(xs), 10.0)
            self.assertLess(max(zs) - min(zs), 10.0)

    def test_porto_smoke(self) -> None:
        from src.layer4.hub.meshgen import generate_mesh

        maps_root = ROOT / "simulator" / "maps"
        if not (maps_root / "porto" / "occupancy").is_dir():
            self.skipTest("porto map missing")
        result = generate_mesh("porto", maps_root)
        self.assertGreater(result.n_verts_visual, 100)
        self.assertTrue(result.track_obj.is_file())


if __name__ == "__main__":
    unittest.main()
