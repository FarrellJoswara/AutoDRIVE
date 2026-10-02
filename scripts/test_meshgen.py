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
    img = np.full((h, w), 205, dtype=np.uint8)  # ROS unknown background
    img[15:65, 15:85] = 254  # known free region
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

    def test_ros_unknown_cells_are_not_free(self) -> None:
        from src.layer4.hub.meshgen.occupancy import load_occupancy
        from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_rect_map(Path(tmp))
            occ = _occupancy_dir(root / "toy")
            grid = load_occupancy(occ, _find_yaml(occ))
            self.assertFalse(grid.free[0, 0])  # grayscale 205 = ROS unknown
            self.assertTrue(grid.free[30, 30])  # grayscale 254 = known free
            self.assertTrue(grid.occupied[10, 30])  # dark wall

    def test_stale_spawn_in_unknown_region_is_rebuilt(self) -> None:
        from src.layer4.hub.meshgen.occupancy import load_occupancy
        from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir
        from src.layer4.hub.spawn import ensure_spawn

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_rect_map(Path(tmp))
            map_dir = root / "toy"
            occ = _occupancy_dir(map_dir)
            meta_path = occ / "meta.json"
            meta_path.write_text(
                json.dumps({
                    "spawn": {"x": -2.0, "y": 0.05, "z": 2.0, "yaw": 0.0},
                    "centerline": {"status": "ready", "file": "centerline.csv"},
                }),
                encoding="utf-8",
            )
            (occ / "centerline.csv").write_text(
                "# x_m,y_m,w_tr_right_m,w_tr_left_m\n-2,2,1,1\n-1.5,2,1,1\n",
                encoding="utf-8",
            )

            spawn = ensure_spawn("toy", root)
            grid = load_occupancy(occ, _find_yaml(occ))
            col = round((spawn["x"] - grid.origin_x) / grid.resolution)
            row = round(grid.free.shape[0] - (spawn["z"] - grid.origin_z) / grid.resolution)

            self.assertTrue(grid.free[row, col])
            self.assertEqual(
                json.loads(meta_path.read_text(encoding="utf-8"))["centerline"][
                    "free_space_semantics"
                ],
                "known_free_v1",
            )

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
