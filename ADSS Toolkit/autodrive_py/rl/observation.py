"""Observation helpers — contracts.md §2 (v2.0.0).

Layout (1-D float32):
  [ lidar[0:N], prev_throttle, prev_steering, speed_norm, yaw_rate_n, ax_n, ay_n ]

Camera is intentionally excluded; adding vision requires a contracts major bump
(or a separate vision head), not silent zero-padding.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    ACCEL_MAX_MPS2,
    LIDAR_MAX_M,
    N_LIDAR_DEFAULT,
    OBS_PROPRIO_DIM,
    SPEED_MAX_MPS,
    YAW_RATE_MAX_RAD_S,
    obs_dim,
)


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


def normalize_speed(speed_mps: float, speed_max: float = SPEED_MAX_MPS) -> float:
    """Forward speed (m/s) → [0, 1] via clip(v / speed_max)."""
    sm = max(float(speed_max), 1e-6)
    return float(np.clip(float(speed_mps) / sm, 0.0, 1.0))


def normalize_imu(
    yaw_rate: float,
    ax: float,
    ay: float,
    *,
    yaw_rate_max: float = YAW_RATE_MAX_RAD_S,
    accel_max: float = ACCEL_MAX_MPS2,
) -> np.ndarray:
    """Compact IMU → 3 floats in [-1, 1]: yaw_rate, ax, ay."""
    yr_m = max(float(yaw_rate_max), 1e-6)
    a_m = max(float(accel_max), 1e-6)
    return np.array(
        [
            np.clip(float(yaw_rate) / yr_m, -1.0, 1.0),
            np.clip(float(ax) / a_m, -1.0, 1.0),
            np.clip(float(ay) / a_m, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def observation_bounds(n_lidar: int = N_LIDAR_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Gymnasium Box low/high for the v2 observation vector."""
    n = int(n_lidar)
    low = np.concatenate(
        [
            np.zeros(n, dtype=np.float32),
            np.array([-1.0, -1.0], dtype=np.float32),  # prev throttle / steering
            np.array([0.0], dtype=np.float32),  # speed
            np.full(3, -1.0, dtype=np.float32),  # imu
        ]
    )
    high = np.concatenate(
        [
            np.ones(n, dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.full(3, 1.0, dtype=np.float32),
        ]
    )
    assert low.shape[0] == obs_dim(n) == n + OBS_PROPRIO_DIM
    return low, high


def build_observation(
    raw_ranges: np.ndarray,
    prev_throttle: float,
    prev_steering: float,
    speed_mps: float,
    yaw_rate: float,
    ax: float,
    ay: float,
    n_lidar: int = N_LIDAR_DEFAULT,
    speed_max: float = SPEED_MAX_MPS,
) -> np.ndarray:
    """Build the fixed v2 observation vector (no camera / no vision pad)."""
    lidar = downsample_lidar(raw_ranges, n_lidar=n_lidar)
    proprio = np.array(
        [
            float(prev_throttle),
            float(prev_steering),
            normalize_speed(speed_mps, speed_max=speed_max),
        ],
        dtype=np.float32,
    )
    imu = normalize_imu(yaw_rate, ax, ay)
    return np.concatenate([lidar, proprio, imu]).astype(np.float32)


def build_observation_from_f1tenth(
    vehicle: Any,
    prev_throttle: float,
    prev_steering: float,
    *,
    speed_mps: float | None = None,
    n_lidar: int = N_LIDAR_DEFAULT,
    speed_max: float = SPEED_MAX_MPS,
    prev_encoder_angles: np.ndarray | None = None,
    dt: float = 0.05,
    wheel_radius_m: float = 0.05,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """
    Same obs layout from AutoDRIVE Bridge ``F1TENTH`` fields.

    Speed: use ``speed_mps`` if given; else estimate from encoder angle delta
    (mean wheel ω * radius). IMU: ``angular_velocity[2]`` (yaw rate),
    ``linear_acceleration[0:2]`` (ax, ay). Camera ignored.

    Returns ``(obs, speed_mps_used, encoder_angles_copy)`` for dt tracking.
    """
    ranges = getattr(vehicle, "lidar_range_array", None)
    if ranges is None:
        ranges = np.full(n_lidar, LIDAR_MAX_M, dtype=np.float32)

    enc = getattr(vehicle, "encoder_angles", None)
    enc_arr = np.asarray(enc, dtype=np.float64).reshape(-1) if enc is not None else None

    if speed_mps is None:
        speed_mps = 0.0
        if enc_arr is not None and prev_encoder_angles is not None and dt > 1e-6:
            prev = np.asarray(prev_encoder_angles, dtype=np.float64).reshape(-1)
            n = min(enc_arr.size, prev.size)
            if n > 0:
                d_ang = enc_arr[:n] - prev[:n]
                # unwrap roughly ±π
                d_ang = (d_ang + np.pi) % (2.0 * np.pi) - np.pi
                omega = float(np.mean(d_ang) / dt)
                speed_mps = abs(omega * float(wheel_radius_m))

    ang = getattr(vehicle, "angular_velocity", None)
    lin = getattr(vehicle, "linear_acceleration", None)
    yaw_rate = float(ang[2]) if ang is not None and len(ang) >= 3 else 0.0
    ax = float(lin[0]) if lin is not None and len(lin) >= 1 else 0.0
    ay = float(lin[1]) if lin is not None and len(lin) >= 2 else 0.0

    obs = build_observation(
        np.asarray(ranges, dtype=np.float32),
        prev_throttle,
        prev_steering,
        speed_mps,
        yaw_rate,
        ax,
        ay,
        n_lidar=n_lidar,
        speed_max=speed_max,
    )
    enc_out = enc_arr.copy() if enc_arr is not None else None
    return obs, float(speed_mps), enc_out
