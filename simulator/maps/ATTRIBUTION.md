# Map assets — attribution

Occupancy grids for Mission Control Fleet live under
[`simulator/maps/`](../simulator/maps/).

## Vendored tracks

| Id | Path | Upstream | Notes |
| :--- | :--- | :--- | :--- |
| `porto` | `simulator/maps/porto/occupancy/` | [AutoDRIVE-RoboRacer-Racetracks](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Racetracks) `Legacy Tracks/Porto Track/` | ROS grid, 0.05 m/px · **BSD-2-Clause** |
| `berlin` | `simulator/maps/berlin/occupancy/` | same repo `Legacy Tracks/Berlin Track/` | Occupancy PNG + yaml · **BSD-2-Clause** |
| `icra2026_classic` | `simulator/maps/icra2026_classic/occupancy/` | [ICRA 2026 Race Resources](https://icra2026-race.roboracer.ai/race_resources.html) | Classic Cup `.pgm` + `.yaml` |
| `icra2026_master` | `simulator/maps/icra2026_master/occupancy/` | same | Master Cup `.pgm` + `.yaml` |

AutoDRIVE racetrack assets: Copyright (c) AutoDRIVE / Tinker-Twins — **BSD-2-Clause**.
Retain copyright notice when redistributing.

Do **not** vendor `.fbx` / `.skp` meshes here (Unity import assets).

Generated wall OBJs (Phase 2) live under `simulator/maps/<id>/mesh/` (`track.obj`, `track_col.obj`):

```bash
python -m src.layer4.hub.meshgen --map-id icra2026_classic
# or POST /api/maps/icra2026_classic/generate-mesh
```

Ops (upload, centerline, mesh preview, Activate): see [`README.md`](README.md).


## Adding a map

1. Create `simulator/maps/<id>/occupancy/` **or** upload a zip via Mission Control
2. Add ROS `.yaml` + image (`.pgm` / `.png`); yaml `image:` must match the filename
3. Optional `meta.json` with `label`, `source`
4. Refresh Mission Control — `GET /api/maps` auto-detects the folder
