"""LiDAR observation helpers — contracts.md §2."""

from __future__ import annotations

import numpy as np

from .contracts import LIDAR_MAX_M, N_LIDAR_DEFAULT


def downsample_lidar(raw_ranges: np.ndarray, n_lidar: int = N_LIDAR_DEFAULT) -> np.ndarray:
    """Downsample / interpolate raw ranges to n_lidar beams, then clip+normalize to [0, 1]."""
    raw = np.asarray(raw_ranges, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        r = np.full(n_lidar, LIDAR_MAX_M, dtype=np.float32)
    elif raw.size == n_lidar:
        r = raw.copy()
    else:
        src = np.linspace(0.0, 1.0, num=raw.size, dtype=np.float32)
        dst = np.linspace(0.0, 1.0, num=n_lidar, dtype=np.float32)
        r = np.interp(dst, src, raw).astype(np.float32)

    r = np.where(np.isfinite(r), r, LIDAR_MAX_M)
    r = np.clip(r, 0.0, LIDAR_MAX_M)
    return (r / LIDAR_MAX_M).astype(np.float32)


def build_observation(
    raw_ranges: np.ndarray,
    prev_throttle: float,
    prev_steering: float,
    n_lidar: int = N_LIDAR_DEFAULT,
) -> np.ndarray:
    lidar = downsample_lidar(raw_ranges, n_lidar=n_lidar)
    return np.concatenate(
        [lidar, np.array([prev_throttle, prev_steering], dtype=np.float32)]
    ).astype(np.float32)
