#!/usr/bin/env python3
"""Unit tests for map zip upload (Phase 4)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layer4.hub.map_upload import install_map_zip  # noqa: E402
from src.layer4.hub.maps_catalog import list_maps_api, scan_maps  # noqa: E402


def _make_zip_flat() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "map.yaml",
            "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n",
        )
        zf.writestr("map.pgm", b"P5\n2 2\n255\n" + bytes([0, 255, 255, 0]))
    return buf.getvalue()


def _make_zip_nested(map_id: str = "demo_track") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            f"{map_id}/occupancy/map.yaml",
            "image: map.pgm\nresolution: 0.05\norigin: [1.0, 2.0, 0.0]\nnegate: 0\n",
        )
        zf.writestr(
            f"{map_id}/occupancy/map.pgm",
            b"P5\n2 2\n255\n" + bytes([0, 255, 255, 0]),
        )
        zf.writestr(
            f"{map_id}/occupancy/meta.json",
            json.dumps({"label": "Demo Track"}),
        )
    return buf.getvalue()


class MapUploadTest(unittest.TestCase):
    def test_flat_zip_requires_map_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                install_map_zip(root, _make_zip_flat())

    def test_flat_zip_installs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = install_map_zip(root, _make_zip_flat(), map_id="uploaded")
            self.assertTrue(result["ok"])
            self.assertEqual(result["id"], "uploaded")
            occ = root / "uploaded" / "occupancy"
            self.assertTrue((occ / "map.yaml").is_file())
            self.assertTrue((occ / "map.pgm").is_file())
            self.assertTrue((occ / "meta.json").is_file())
            found = scan_maps(root)
            self.assertEqual([m.id for m in found], ["uploaded"])
            self.assertIsNotNone(found[0].thumbnail_url)

    def test_nested_zip_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = install_map_zip(root, _make_zip_nested("demo_track"))
            self.assertEqual(result["id"], "demo_track")
            api = list_maps_api(root)
            ids = [m["id"] for m in api]
            self.assertIn("demo_track", ids)
            entry = next(m for m in api if m["id"] == "demo_track")
            self.assertEqual(entry["label"], "Demo Track")

    def test_overwrite_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install_map_zip(root, _make_zip_nested("demo_track"))
            with self.assertRaises(ValueError):
                install_map_zip(root, _make_zip_nested("demo_track"))
            again = install_map_zip(
                root, _make_zip_nested("demo_track"), overwrite=True
            )
            self.assertTrue(again["overwrite"])


if __name__ == "__main__":
    unittest.main()
