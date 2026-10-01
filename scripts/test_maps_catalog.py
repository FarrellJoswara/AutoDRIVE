#!/usr/bin/env python3
"""Unit tests for hub maps catalog discovery (no Unity / hub server)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_maps_catalog():
    """Load maps_catalog without importing src.layer4 (avoids pydantic at test time)."""
    path = ROOT / "src" / "layer4" / "hub" / "maps_catalog.py"
    spec = importlib.util.spec_from_file_location("aicar_maps_catalog", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_mc = _load_maps_catalog()
scan_maps = _mc.scan_maps
list_maps_api = _mc.list_maps_api


def _write_map(root: Path, map_id: str, *, good: bool = True) -> None:
    occ = root / map_id / "occupancy"
    occ.mkdir(parents=True)
    (occ / "meta.json").write_text(
        json.dumps({"label": map_id.upper(), "source": "test"}),
        encoding="utf-8",
    )
    if not good:
        return
    (occ / "map.yaml").write_text(
        "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n",
        encoding="utf-8",
    )
    (occ / "map.pgm").write_bytes(b"P5\n1 1\n255\n\x00")


class MapsCatalogTest(unittest.TestCase):
    def test_scan_skips_invalid_and_finds_good(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_map(root, "alpha", good=True)
            _write_map(root, "broken", good=False)
            (root / "not_a_map.txt").write_text("x", encoding="utf-8")
            found = scan_maps(root)
            ids = [m.id for m in found]
            self.assertEqual(ids, ["alpha"])
            self.assertEqual(found[0].label, "ALPHA")
            self.assertTrue(found[0].yaml_url.endswith("/alpha/occupancy/map.yaml"))
            self.assertTrue(found[0].overlay_only)
            self.assertFalse(found[0].active)
            self.assertEqual(found[0].mesh_status, "none")

    def test_list_maps_api_includes_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_map(root, "porto", good=True)
            api = list_maps_api(root)
            ids = [m["id"] for m in api]
            self.assertIn("porto", ids)
            self.assertEqual(ids[-1], "none")
            self.assertIsNone(api[-1]["yaml_url"])

    def test_repo_simulator_maps_present(self) -> None:
        maps_root = ROOT / "simulator" / "maps"
        self.assertTrue(maps_root.is_dir(), "simulator/maps missing")
        found = scan_maps(maps_root)
        ids = {m.id for m in found}
        for expected in ("porto", "berlin", "icra2026_classic", "icra2026_master"):
            self.assertIn(expected, ids, f"missing map {expected}")


if __name__ == "__main__":
    unittest.main()
