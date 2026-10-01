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
    """Skeletonize free space → polyline CSV under occupancy/centerline.csv."""
    map_dir = maps_root / map_id
    if not map_dir.is_dir():
        raise FileNotFoundError(f"unknown map id: {map_id}")
    occ = _occupancy_dir(map_dir)
    yaml_path = _find_yaml(occ)
    grid = load_occupancy(occ, yaml_path)
    free = ~grid.occupied
    free_u8 = (free.astype(np.uint8)) * 255
    # Close small gaps then skeletonize
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    free_u8 = cv2.morphologyEx(free_u8, cv2.MORPH_CLOSE, k)
    skel = _morph_skeleton(free_u8)
    ys, xs = np.where(skel > 0)
    if len(xs) < 8:
        raise RuntimeError(f"could not extract centerline skeleton for '{map_id}'")

    # Subsample dense skeleton before ordering (keeps NN walk tractable)
    if len(xs) > 4000:
        step = max(1, len(xs) // 3000)
        xs, ys = xs[::step], ys[::step]
    pts_px = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])  # col, row
    ordered = _order_skeleton(pts_px)
    world = np.zeros((len(ordered), 2), dtype=np.float64)
    for i, (col, row) in enumerate(ordered):
        world[i, 0], world[i, 1] = pixel_to_world(grid, float(col), float(row))

    closed = bool(
        len(world) > 3
        and np.linalg.norm(world[0] - world[-1]) < max(0.5, 4 * grid.resolution)
    )
    if closed and not np.allclose(world[0], world[-1]):
        world = np.vstack([world, world[0]])

    resampled = _resample_polyline(world, spacing_m)
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


def _morph_skeleton(img: np.ndarray) -> np.ndarray:
    """Morphological skeleton (opencv-python-headless; no ximgproc required)."""
    img = (img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _order_skeleton(pts: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbour ordering; start at an endpoint-ish extreme."""
    n = len(pts)
    if n <= 2:
        return pts
    # Start at point with largest distance from centroid (outer loop-ish)
    c = pts.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(pts - c, axis=1)))
    used = np.zeros(n, dtype=bool)
    order = [start]
    used[start] = True
    cur = pts[start]
    for _ in range(n - 1):
        dists = np.linalg.norm(pts - cur, axis=1)
        dists[used] = np.inf
        nxt = int(np.argmin(dists))
        if not np.isfinite(dists[nxt]):
            break
        order.append(nxt)
        used[nxt] = True
        cur = pts[nxt]
    return pts[order]
