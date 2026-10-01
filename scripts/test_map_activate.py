#!/usr/bin/env python3
"""Unit tests for map activate persistence (no Docker / Unity)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # maps_catalog must be importable as sibling dependency name used inside map_activate
    spec.loader.exec_module(mod)
    return mod


def _load_map_activate():
    # map_activate imports src.layer4.hub.maps_catalog — load via package path if possible
    sys.path.insert(0, str(ROOT))
    from src.layer4.hub import map_activate  # type: ignore

    return map_activate


def _write_map(root: Path, map_id: str, *, mesh: bool = False) -> None:
    occ = root / map_id / "occupancy"
    occ.mkdir(parents=True)
    (occ / "map.yaml").write_text(
        "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\n",
        encoding="utf-8",
    )
    (occ / "map.pgm").write_bytes(b"P5\n1 1\n255\n\x00")
    (occ / "meta.json").write_text(
        json.dumps({"label": map_id, "mesh": {"status": "ready" if mesh else "none"}}),
        encoding="utf-8",
    )
    if mesh:
        mesh_dir = root / map_id / "mesh"
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "track.obj").write_text("# test\nv 0 0 0\nv 1 0 0\nv 0 0 1\nf 1 2 3\n", encoding="utf-8")


class MapActivateTest(unittest.TestCase):
    def test_write_and_read_active(self) -> None:
        ma = _load_map_activate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_map(root, "porto", mesh=True)
            written = ma.write_active_map(root, "porto")
            self.assertEqual(written["id"], "porto")
            self.assertTrue(Path(written["path"]).is_file())
            read = ma.read_active_map(root)
            self.assertEqual(read["id"], "porto")
            cleared = ma.write_active_map(root, "none")
            self.assertIsNone(cleared["id"])

    def test_activate_requires_mesh(self) -> None:
        ma = _load_map_activate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_map(root, "porto", mesh=False)
            with self.assertRaises(ValueError):
                ma.activate_map(root, "porto", restart=False)

    def test_activate_ok_no_restart(self) -> None:
        ma = _load_map_activate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_map(root, "porto", mesh=True)
            result = ma.activate_map(root, "porto", restart=False)
            self.assertTrue(result["ok"])
            self.assertEqual(result["active"]["id"], "porto")
            self.assertEqual(result["restarted"], [])

    def test_train_blocks_without_force(self) -> None:
        ma = _load_map_activate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_map(root, "porto", mesh=True)
            with self.assertRaises(RuntimeError):
                ma.activate_map(root, "porto", restart=False, train_running=True, force=False)
            result = ma.activate_map(root, "porto", restart=False, train_running=True, force=True)
            self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
