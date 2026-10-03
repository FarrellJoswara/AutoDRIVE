# Maps (Mission Control Fleet + TrackLoader)

Occupancy packages live under `simulator/maps/<id>/occupancy/`.
Generated wall meshes: `simulator/maps/<id>/mesh/` (`track.obj`, `track_col.obj`).
Active physics map: `simulator/maps/.active_map.json` (written by **Activate**).

Pointer from older paths: [`assets/maps/README.md`](../../assets/maps/README.md).
Attribution: [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Layout

```
simulator/maps/<id>/
  occupancy/
    *.yaml + image (.pgm/.png)   # required (ROS map_server)
    meta.json                    # optional label/source/mesh status
    preview.png                  # catalog thumbnail (auto on upload)
    centerline.csv               # optional / generate via API
  mesh/
    track.obj / track_col.obj    # Generate mesh
    preview.png                  # ortho wireframe (Generate mesh preview)
```

## Upload → mesh → Activate

1. **Upload** a zip (Fleet **Upload zip**, or `POST /api/maps/upload`):
   - Nested: `<id>/occupancy/<yaml+image>`
   - Or flat yaml+image with `?map_id=<id>`
2. Catalog picks it up via `GET /api/maps`.
3. **Generate mesh** → `POST /api/maps/{id}/generate-mesh`
4. Optional: **Centerline**, **Mesh preview**
5. **Activate** → writes `.active_map.json` + restarts compose `sim` containers.
   Entrypoint reads the file into `AICAR_MAP_ID` (env only, not Unity argv). Requires TrackLoader player
   (`simulator/.aicar_trackloader`). Local headed sims: relaunch after Activate.

Selecting a track in Fleet only changes the **canvas overlay**. Physics change
requires **Activate** (+ sim restart / relaunch).

## API (hub)

| Method | Path | Role |
| :--- | :--- | :--- |
| GET | `/api/maps` | Catalog (+ thumbnails / centerline / mesh preview URLs) |
| POST | `/api/maps/upload` | Zip body; query `map_id`, `overwrite`, `label` |
| POST | `/api/maps/{id}/generate-mesh` | Occupancy → OBJ |
| POST | `/api/maps/{id}/generate-centerline` | Free-space skeleton → CSV |
| GET | `/api/maps/{id}/centerline.csv` | Download CSV |
| POST | `/api/maps/{id}/generate-thumbnail` | Ensure occupancy preview.png |
| POST | `/api/maps/{id}/generate-mesh-preview` | Ortho PNG of track.obj |
| GET | `/api/maps/{id}/mesh-preview.png` | Serve mesh preview |
| GET | `/api/maps/active?selected=` | Active id + selection mismatch warnings |
| POST | `/api/maps/{id}/activate` | Persist active + restart sims |

## Adding a map by hand

1. Create `simulator/maps/<id>/occupancy/`
2. Add ROS `.yaml` + image; yaml `image:` must match the filename
3. Optional `meta.json` with `label`, `source`
4. Refresh Mission Control
