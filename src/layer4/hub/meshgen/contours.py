"""Extract wall polylines from an occupancy mask."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .occupancy import OccupancyGrid, pixel_to_world


def extract_wall_polylines(
    grid: OccupancyGrid,
    *,
    min_area_px: float = 40.0,
    simplify_eps_px: float = 1.5,
    morph_close_k: int = 3,
) -> List[np.ndarray]:
    """
    Return list of (N, 2) polylines in world metres (x, z), closed loops preferred.

    Uses outer contours of the occupied mask (walls / barriers).
    """
    mask = grid.occupied.astype(np.uint8) * 255
    if morph_close_k > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_k, morph_close_k)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines: List[np.ndarray] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area_px:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_eps_px, closed=True)
        if approx is None or len(approx) < 3:
            continue
        pts_px = approx.reshape(-1, 2).astype(np.float64)  # (col, row)
        world = np.zeros((len(pts_px), 2), dtype=np.float64)
        for i, (col, row) in enumerate(pts_px):
            world[i, 0], world[i, 1] = pixel_to_world(grid, float(col), float(row))
        # Close loop if needed
        if not np.allclose(world[0], world[-1]):
            world = np.vstack([world, world[0]])
        polylines.append(world)

    # Longest first (outer track bounds tend to be larger)
    polylines.sort(key=lambda p: _polyline_length(p), reverse=True)
    return polylines


def _polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def decimate_polyline(pts: np.ndarray, step: int) -> np.ndarray:
    """Keep every `step`-th vertex (always keep first/last)."""
    if step <= 1 or len(pts) <= 4:
        return pts
    idx = list(range(0, len(pts) - 1, step))
    if idx[-1] != len(pts) - 1:
        idx.append(len(pts) - 1)
    return pts[idx]
