"""Stateful, forward-only projection of a car onto a map centerline."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class RouteProgressTracker:
    """Project poses onto an ordered route and expose its across-track frontier.

    ``frontier_s`` is monotonic for the lifetime of an episode. Projection after
    initialization is constrained to a short window ahead of the frontier, so a
    nearby later section cannot award a large progress jump after a shortcut.
    """

    def __init__(
        self,
        points: np.ndarray,
        widths_right: Optional[np.ndarray] = None,
        widths_left: Optional[np.ndarray] = None,
        *,
        max_forward_m: float = 2.0,
        backward_tolerance_m: float = 0.5,
        max_lateral_m: float = 3.0,
        min_push_m: float = 0.04,
    ) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("centerline must contain at least three 2D points")
        if not np.all(np.isfinite(pts)):
            raise ValueError("centerline contains non-finite coordinates")

        seg = np.diff(pts, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        keep = lengths > 1e-6
        self.points = pts
        self._seg = seg
        self._lengths = lengths
        self._cum = np.concatenate(([0.0], np.cumsum(lengths)))
        self.length_m = float(self._cum[-1])
        self.closed = bool(np.linalg.norm(pts[0] - pts[-1]) <= 0.5)
        if self.closed:
            close_seg = pts[0] - pts[-1]
            close_len = float(np.linalg.norm(close_seg))
            if close_len > 1e-6:
                self.points = np.vstack((pts, pts[0]))
                self._seg = np.vstack((seg, close_seg))
                self._lengths = np.append(lengths, close_len)
                self._cum = np.concatenate(([0.0], np.cumsum(self._lengths)))
                self.length_m = float(self._cum[-1])

        if self.length_m <= 1e-3 or not np.any(keep):
            raise ValueError("centerline has no usable route length")

        n = len(self.points)
        self.widths_right = self._prepare_widths(widths_right, n)
        self.widths_left = self._prepare_widths(widths_left, n)
        self.max_forward_m = max(0.1, float(max_forward_m))
        self.backward_tolerance_m = max(0.0, float(backward_tolerance_m))
        self.max_lateral_m = max(0.1, float(max_lateral_m))
        self.min_push_m = max(0.0, float(min_push_m))
        self.frontier_s: Optional[float] = None
        self.last_push_time: Optional[float] = None
        self.last_time: Optional[float] = None
        self.frontier_speed_mps = 0.0

    @classmethod
    def from_csv(cls, path: Path, **kwargs: float) -> "RouteProgressTracker":
        rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            for row in csv.reader(line for line in f if not line.lstrip().startswith("#")):
                if not row:
                    continue
                try:
                    rows.append([float(value) for value in row[:4]])
                except (ValueError, TypeError):
                    continue
        if len(rows) < 3:
            raise ValueError(f"centerline CSV has fewer than three points: {path}")
        data = np.asarray(rows, dtype=np.float64)
        right = data[:, 2] if data.shape[1] > 2 else None
        left = data[:, 3] if data.shape[1] > 3 else None
        return cls(data[:, :2], right, left, **kwargs)

    @staticmethod
    def _prepare_widths(widths: Optional[np.ndarray], n: int) -> np.ndarray:
        if widths is None:
            return np.full(n, 0.5, dtype=np.float64)
        values = np.asarray(widths, dtype=np.float64).reshape(-1)
        if len(values) != n:
            # A closed route may append its first point after loading.
            if len(values) == n - 1:
                values = np.append(values, values[0])
            else:
                return np.full(n, 0.5, dtype=np.float64)
        valid = np.isfinite(values) & (values > 0.05)
        if not np.any(valid):
            return np.full(n, 0.5, dtype=np.float64)
        idx = np.arange(n)
        values = np.interp(idx, idx[valid], values[valid])
        return np.clip(values, 0.15, 5.0)

    def reset(self, x: float, z: float, now: Optional[float] = None) -> dict:
        """Initialize the frontier at the closest point to the spawn pose."""
        t = time.monotonic() if now is None else float(now)
        candidate = self._project(float(x), float(z), None)
        if candidate is None:
            raise ValueError("spawn pose is too far from the selected map centerline")
        self.frontier_s = candidate[0]
        self.last_push_time = t
        self.last_time = t
        self.frontier_speed_mps = 0.0
        return self.sample(t)

    def update(self, x: float, z: float, now: Optional[float] = None) -> dict:
        """Update progress; the returned frontier can never retreat."""
        t = time.monotonic() if now is None else float(now)
        if self.frontier_s is None:
            return self.reset(x, z, t)

        old = self.frontier_s
        candidate = self._project(float(x), float(z), old)
        dt = max(1e-6, t - (self.last_time if self.last_time is not None else t))
        pushed = 0.0
        if candidate is not None and candidate[0] - old >= self.min_push_m:
            self.frontier_s = candidate[0]
            pushed = self.frontier_s - old
            self.last_push_time = t
            self.frontier_speed_mps = pushed / dt
        else:
            self.frontier_speed_mps = 0.0
        self.last_time = t
        result = self.sample(t)
        result["advanced_m"] = pushed
        return result

    def sample(self, now: Optional[float] = None) -> dict:
        t = time.monotonic() if now is None else float(now)
        if self.frontier_s is None:
            return {"progress_m": 0.0, "line": None, "time_since_push_s": None,
                    "speed_mps": 0.0, "advanced_m": 0.0}
        line = self._line_at(self.frontier_s)
        return {
            "progress_m": float(self.frontier_s),
            "line": [[float(line[0][0]), float(line[0][1])],
                     [float(line[1][0]), float(line[1][1])]],
            "time_since_push_s": max(0.0, t - self.last_push_time)
            if self.last_push_time is not None else None,
            "speed_mps": float(self.frontier_speed_mps),
            "advanced_m": 0.0,
        }

    def project_pose(self, x: float, z: float) -> Optional[float]:
        """Return a pose's nearest centerline distance without changing progress."""
        candidate = self._project(float(x), float(z), None)
        return None if candidate is None else float(candidate[0])

    def gate_line_at(self, route_s: float):
        """Return the full-width centerline cross-section at route distance ``s``."""
        if not self.closed:
            return None
        line = self._line_at(float(route_s) % self.length_m)
        return [
            [float(line[0][0]), float(line[0][1])],
            [float(line[1][0]), float(line[1][1])],
        ]

    def _project(self, x: float, z: float, frontier: Optional[float]):
        pos = np.asarray([x, z], dtype=np.float64)
        laps = (0,)
        if self.closed and frontier is not None:
            base = int(math.floor(frontier / self.length_m))
            laps = (base - 1, base, base + 1)
        valid_seg = self._lengths > 1e-6
        vectors = self._seg[valid_seg]
        starts = self.points[:-1][valid_seg]
        lengths = self._lengths[valid_seg]
        offsets = self._cum[:-1][valid_seg]
        segment_ids = np.flatnonzero(valid_seg)
        t = np.clip(np.sum((pos - starts) * vectors, axis=1) / (lengths * lengths), 0.0, 1.0)
        projected = starts + t[:, None] * vectors
        offset = pos - projected
        lateral = np.linalg.norm(offset, axis=1)
        signed_lateral = (vectors[:, 0] * offset[:, 1] - vectors[:, 1] * offset[:, 0]) / lengths
        right_width = (
            (1.0 - t) * self.widths_right[segment_ids]
            + t * self.widths_right[segment_ids + 1]
        )
        left_width = (
            (1.0 - t) * self.widths_left[segment_ids]
            + t * self.widths_left[segment_ids + 1]
        )
        base_s = offsets + t * lengths

        best = None
        for lap in laps:
            route_s = base_s + (lap * self.length_m if self.closed else 0.0)
            acceptable = lateral <= self.max_lateral_m
            # Widths are map-derived half-corridor estimates, so keep a small
            # tolerance for rasterization while rejecting off-corridor poses.
            acceptable &= signed_lateral <= left_width + 0.15
            acceptable &= signed_lateral >= -(right_width + 0.15)
            if frontier is not None:
                acceptable &= route_s >= frontier - self.backward_tolerance_m
                acceptable &= route_s <= frontier + self.max_forward_m
            indices = np.flatnonzero(acceptable)
            if len(indices):
                idx = indices[int(np.argmin(lateral[indices]))]
                candidate = (float(lateral[idx]), float(route_s[idx]))
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            return None
        lateral_distance, route_s = best
        return route_s, lateral_distance

    def _line_at(self, route_s: float) -> Tuple[np.ndarray, np.ndarray]:
        local_s = route_s % self.length_m if self.closed else min(route_s, self.length_m)
        i = int(np.searchsorted(self._cum, local_s, side="right") - 1)
        i = min(max(i, 0), len(self._lengths) - 1)
        length = self._lengths[i]
        alpha = 0.0 if length <= 1e-8 else (local_s - self._cum[i]) / length
        a = self.points[i]
        b = self.points[i + 1]
        center = a + alpha * (b - a)
        tangent = (b - a) / max(length, 1e-8)
        left_normal = np.asarray([-tangent[1], tangent[0]])
        wr = self.widths_right[min(i, len(self.widths_right) - 1)]
        wl = self.widths_left[min(i, len(self.widths_left) - 1)]
        right_pt = center - left_normal * wr
        left_pt = center + left_normal * wl
        return right_pt, left_pt


def map_centerline_path(map_id: str, repository_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve a generated map centerline; builtin Unity map has none."""
    if not map_id or map_id == "none":
        return None
    root = repository_root or Path(__file__).resolve().parents[2]
    maps_root = (root / "simulator" / "maps").resolve()
    path = (maps_root / map_id / "occupancy" / "centerline.csv").resolve()
    if not path.is_relative_to(maps_root):
        raise ValueError(f"invalid map id: {map_id!r}")
    if not path.is_file():
        return None
    return path
