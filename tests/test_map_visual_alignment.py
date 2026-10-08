"""Regression checks for the Porto map → official Watch coordinate alignment."""

from __future__ import annotations

import csv
import math
import json
from pathlib import Path

from src.layer4.hub.maps_catalog import scan_maps


ROOT = Path(__file__).resolve().parents[1]


def _centerline(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(line for line in stream if not line.lstrip().startswith("#"))
        return [(float(row[0]), float(row[1])) for row in rows if row]


def test_porto_catalog_exposes_official_display_alignment() -> None:
    maps = scan_maps(ROOT / "simulator" / "maps")
    porto = next(entry for entry in maps if entry.id == "porto")
    alignment = porto.to_api()["official_world_alignment"]
    assert alignment is not None
    assert math.isclose(alignment["rotation_rad"], math.radians(-89.40842075), abs_tol=1e-8)


def test_porto_display_alignment_registers_official_route_to_occupancy() -> None:
    map_points = _centerline(ROOT / "simulator" / "maps" / "porto" / "occupancy" / "centerline.csv")
    official_points = _centerline(ROOT / "competition" / "iros2026" / "official_centerline.csv")
    metadata = json.loads(
        (ROOT / "simulator" / "maps" / "porto" / "occupancy" / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    alignment = metadata["official_world_alignment"]
    c = math.cos(alignment["rotation_rad"])
    s = math.sin(alignment["rotation_rad"])

    assert len(map_points) == len(official_points)
    errors = []
    for (x, z), (expected_x, expected_z) in zip(map_points, official_points):
        actual_x = c * x - s * z + alignment["translation_x_m"]
        actual_z = s * x + c * z + alignment["translation_z_m"]
        errors.append(math.hypot(actual_x - expected_x, actual_z - expected_z))

    assert max(errors) < 1e-5
