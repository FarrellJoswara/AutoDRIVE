"""Centerline CSV generate / export for occupancy maps (Phase 4).

Format (f1tenth-compatible header):
  # x_m,y_m,w_tr_right_m,w_tr_left_m
  x,y,w_right,w_left

Coordinates are world metres (x, z) matching Fleet / meshgen (ROS origin).
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.layer4.hub.meshgen.occupancy import load_occupancy, pixel_to_world
from src.layer4.hub.meshgen.pipeline import _find_yaml, _occupancy_dir


CENTERLINE_FILENAME = "centerline.csv"
CENTERLINE_ALGORITHM = "connected_skeleton_longest_valid_cycle_v2"


@dataclass
class CenterlineResult:
    map_id: str
    path: Path
    n_points: int
    closed: bool

    def to_api(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def centerline_path(map_dir: Path) -> Path:
    return _occupancy_dir(map_dir) / CENTERLINE_FILENAME


def find_centerline(map_dir: Path) -> Optional[Path]:
    occ = _occupancy_dir(map_dir)
    preferred = occ / CENTERLINE_FILENAME
    if preferred.is_file():
        return preferred
    matches = sorted(occ.glob("*centerline*.csv"))
    return matches[0] if matches else None


def read_centerline_csv(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resample_polyline(pts: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(pts) < 2:
        return pts
    diffs = np.diff(pts, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(seg))
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(round(total / max(spacing_m, 1e-3))) + 1)
    target = np.linspace(0.0, total, n)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.zeros((n, 2), dtype=np.float64)
    for i, t in enumerate(target):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = max(0, min(j, len(pts) - 2))
        span = cum[j + 1] - cum[j]
        alpha = 0.0 if span < 1e-9 else (t - cum[j]) / span
        out[i] = pts[j] * (1 - alpha) + pts[j + 1] * alpha
    return out


def _track_widths(
    free: np.ndarray,
    cols: np.ndarray,
    rows: np.ndarray,
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate left/right track widths via distance transform (metres)."""
    # distance to nearest occupied (wall), in pixels
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    w = np.zeros(len(cols), dtype=np.float64)
    for i, (c, r) in enumerate(zip(cols, rows)):
        rr, cc = int(round(r)), int(round(c))
        if 0 <= rr < dist.shape[0] and 0 <= cc < dist.shape[1]:
            w[i] = float(dist[rr, cc]) * resolution
        else:
            w[i] = 0.0
    # Symmetric half-widths when we only have clearance-to-wall
    return w, w


def generate_centerline(
    map_id: str,
    maps_root: Path,
    *,
    spacing_m: float = 0.25,
) -> CenterlineResult:
    """Extract the longest validated closed route from known-free map cells."""
    map_dir = maps_root / map_id
    if not map_dir.is_dir():
        raise FileNotFoundError(f"unknown map id: {map_id}")
    occ = _occupancy_dir(map_dir)
    yaml_path = _find_yaml(occ)
    grid = load_occupancy(occ, yaml_path)
    # Unknown ROS cells (commonly grayscale 205) are not track surface.
    free = grid.free
    # Thin only confirmed free cells. Closing gaps here can create a synthetic
    # bridge straight through a real wall, even if the later route is smooth.
    free_u8 = free.astype(np.uint8)
    skel = _thin_binary(free_u8)
    resampled = _extract_valid_closed_route(grid, skel, spacing_m)
    closed = True
    # Map resampled back to pixel for width — approximate via nearest skeleton pixel
    cols = (resampled[:, 0] - grid.origin_x) / grid.resolution
    h = grid.occupied.shape[0]
    rows = h - (resampled[:, 1] - grid.origin_z) / grid.resolution
    w_r, w_l = _track_widths(free, cols, rows, grid.resolution)

    out_path = occ / CENTERLINE_FILENAME
    buf = io.StringIO()
    buf.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
    writer = csv.writer(buf, lineterminator="\n")
    for (x, y), wr, wl in zip(resampled, w_r, w_l):
        writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{wr:.6f}", f"{wl:.6f}"])
    out_path.write_text(buf.getvalue(), encoding="utf-8")

    _update_meta_centerline(occ, {
        "status": "ready",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_points": len(resampled),
        "spacing_m": spacing_m,
        "closed": closed,
        "file": CENTERLINE_FILENAME,
        "free_space_semantics": "known_free_v1",
        "algorithm": CENTERLINE_ALGORITHM,
    })

    # Keep vehicle spawn in sync with the new centerline (TrackLoader reads meta.spawn).
    try:
        from src.layer4.hub.spawn import ensure_spawn

        # Force rewrite: clear cached spawn so ensure recomputes from this CSV.
        meta_path = occ / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict) and "spawn" in meta:
                    del meta["spawn"]
                    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        ensure_spawn(map_id, maps_root, generate_if_missing=False)
    except Exception:
        pass

    return CenterlineResult(
        map_id=map_id,
        path=out_path,
        n_points=len(resampled),
        closed=closed,
    )


def _update_meta_centerline(occ: Path, info: Dict[str, Any]) -> None:
    meta_path = occ / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta["centerline"] = info
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _thin_binary(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning; keeps a one-pixel, 8-connected medial skeleton."""
    image = (mask > 0).astype(np.uint8)
    h, w = image.shape
    while True:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1)
            p2 = padded[0:h, 1 : w + 1]
            p3 = padded[0:h, 2 : w + 2]
            p4 = padded[1 : h + 1, 2 : w + 2]
            p5 = padded[2 : h + 2, 2 : w + 2]
            p6 = padded[2 : h + 2, 1 : w + 1]
            p7 = padded[2 : h + 2, 0:w]
            p8 = padded[1 : h + 1, 0:w]
            p9 = padded[0:h, 0:w]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum(
                ((neighbors[i] == 0) & (neighbors[(i + 1) % 8] > 0)).astype(np.uint8)
                for i in range(8)
            )
            remove = (image > 0) & (count >= 2) & (count <= 6) & (transitions == 1)
            if phase == 0:
                remove &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                remove &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if np.any(remove):
                image[remove] = 0
                changed = True
        if not changed:
            return image


def _extract_valid_closed_route(
    grid: Any,
    skeleton: np.ndarray,
    spacing_m: float,
    *,
    minimum_clearance_m: float = 0.12,
) -> np.ndarray:
    """Trace connected skeleton cycles and choose the longest wall-safe one."""
    import networkx as nx

    rows, cols = np.where(skeleton > 0)
    if len(rows) < 8:
        raise RuntimeError("could not extract a usable centerline skeleton")
    pixels = set(zip(rows.tolist(), cols.tolist()))
    graph = nx.Graph()
    graph.add_nodes_from(sorted(pixels))
    scale = grid.resolution
    # Add each undirected neighbour once. Suppress diagonal corner edges when
    # an orthogonal route already connects the same pixel pair.
    for row, col in sorted(pixels):
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = (row + dr, col + dc)
            if neighbor not in pixels:
                continue
            if dr and dc and (
                (row, col + dc) in pixels or (row + dr, col) in pixels
            ):
                continue
            graph.add_edge(
                (row, col), neighbor,
                weight=scale * (np.sqrt(2.0) if dr and dc else 1.0),
            )

    largest_component = max(nx.connected_components(graph), key=len)
    connected = graph.subgraph(largest_component).copy()
    _prune_short_skeleton_spurs(connected, max_length_m=1.0)
    cycles = nx.cycle_basis(connected)
    if not cycles:
        raise RuntimeError("centerline skeleton has no closed drivable loop")

    h = grid.free.shape[0]
    clearance = cv2.distanceTransform(grid.free.astype(np.uint8), cv2.DIST_L2, 5)
    candidates = []
    for cycle in cycles:
        if len(cycle) < 3:
            continue
        length = sum(
            connected[a][b]["weight"]
            for a, b in zip(cycle, cycle[1:] + cycle[:1])
        )
        if length < 3.0:
            continue
        world = np.asarray(
            [pixel_to_world(grid, float(col), float(row)) for row, col in cycle],
            dtype=np.float64,
        )
        world = np.vstack((world, world[0]))
        sampled = _resample_polyline(world, spacing_m)
        if _route_is_drivable(
            grid, sampled, clearance, minimum_clearance_m=minimum_clearance_m
        ):
            candidates.append((float(length), sampled))

    if not candidates:
        raise RuntimeError(
            "no closed centerline cycle stays in known-free space with "
            f"at least {minimum_clearance_m:.2f} m clearance"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _prune_short_skeleton_spurs(graph: Any, *, max_length_m: float) -> None:
    """Remove short dead ends attached to a junction without cutting loops."""
    while True:
        remove = set()
        for endpoint in [node for node in graph if graph.degree(node) == 1]:
            chain = [endpoint]
            previous = None
            current = endpoint
            length = 0.0
            while graph.degree(current) <= 2:
                next_nodes = [node for node in graph.neighbors(current) if node != previous]
                if not next_nodes:
                    break
                following = next_nodes[0]
                length += graph[current][following]["weight"]
                previous, current = current, following
                if graph.degree(current) != 2:
                    break
                chain.append(current)
            if graph.degree(current) >= 3 and length < max_length_m:
                remove.update(chain)
        if not remove:
            return
        graph.remove_nodes_from(remove)


def _route_is_drivable(
    grid: Any,
    route: np.ndarray,
    clearance_px: np.ndarray,
    *,
    minimum_clearance_m: float,
) -> bool:
    """Check the full interpolated route, not just its sampled waypoints."""
    h, w = grid.free.shape
    minimum_px = minimum_clearance_m / grid.resolution
    for start, end in zip(route[:-1], route[1:]):
        distance = float(np.linalg.norm(end - start))
        samples = max(1, int(np.ceil(distance / (grid.resolution * 0.5))))
        for t in np.linspace(0.0, 1.0, samples + 1):
            x, z = start * (1.0 - t) + end * t
            col = int(round((x - grid.origin_x) / grid.resolution))
            row = int(round(h - (z - grid.origin_z) / grid.resolution))
            if not (0 <= row < h and 0 <= col < w and grid.free[row, col]):
                return False
            if clearance_px[row, col] < minimum_px:
                return False
    return True
