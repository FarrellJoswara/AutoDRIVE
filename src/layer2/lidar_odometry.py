"""Short-horizon ego-motion estimation from successive planar LiDAR scans.

This uses only the permitted LiDAR sensor stream. Simulator pose is not an
input; it may be used by offline diagnostics to measure estimator error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


_MAX_BODY_SPEED_MPS = 22.88
_SPEED_INNOVATION_TOLERANCE_MPS = 1.5


def acceleration_consistent_lidar_speed(
    measured_speed_mps: float,
    previous_speed_mps: float,
    forward_acceleration_mps2: float,
    elapsed_s: float,
) -> float:
    """Reject scan-matching speed jumps inconsistent with the IMU.

    A LiDAR scan can match a different nearby wall configuration after a
    checkpoint reset or in a repetitive corridor. Such a match can be marked
    geometrically valid while implying an impossible one-frame velocity jump.
    The IMU prediction provides a sensor-only plausibility check. A near-zero
    LiDAR estimate is retained as a stop/reset observation even if it conflicts
    with the prediction, so a checkpoint teleport cannot leave stale speed.
    """
    measured = float(measured_speed_mps)
    previous = float(previous_speed_mps)
    acceleration = float(forward_acceleration_mps2)
    dt = float(elapsed_s)
    if not np.isfinite([measured, previous, acceleration, dt]).all() or dt <= 0.0:
        return 0.0

    predicted = float(np.clip(
        previous + acceleration * dt,
        -_MAX_BODY_SPEED_MPS,
        _MAX_BODY_SPEED_MPS,
    ))
    tolerance = max(
        _SPEED_INNOVATION_TOLERANCE_MPS,
        abs(acceleration) * dt * 1.5 + 0.25,
    )
    if abs(measured - predicted) <= tolerance:
        return float(np.clip(measured, -_MAX_BODY_SPEED_MPS, _MAX_BODY_SPEED_MPS))

    # A near-zero scan displacement is evidence that the body stopped or was
    # teleported to a checkpoint. Do not preserve a stale pre-reset velocity.
    if abs(measured) <= 0.75:
        return 0.0
    return predicted


@dataclass(frozen=True)
class ScanMotion:
    """Estimated relative motion from the previous scan to the current scan."""

    forward_m: float = 0.0
    lateral_m: float = 0.0
    yaw_rad: float = 0.0
    rmse_m: float = math.inf
    inlier_fraction: float = 0.0
    valid: bool = False


def _scan_points(
    ranges_m: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    beam_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
    if ranges.size not in (1080, 1081):
        raise ValueError(f"LiDAR odometry requires 1080 or 1081 beams, got {ranges.size}")
    angles = np.linspace(-3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0, ranges.size)
    keep = (
        np.isfinite(ranges)
        & (ranges > max(0.10, min_range_m))
        & (ranges < max_range_m * 0.98)
    )
    keep &= (np.arange(ranges.size) % beam_stride) == 0
    distance = ranges[keep]
    angle = angles[keep]
    points = np.column_stack((distance * np.cos(angle), distance * np.sin(angle)))
    return points, np.flatnonzero(keep)


def estimate_scan_motion(
    previous_ranges_m: np.ndarray,
    current_ranges_m: np.ndarray,
    *,
    min_range_m: float = 0.06,
    max_range_m: float = 10.0,
    beam_stride: int = 1,
    max_correspondence_m: float = 0.75,
    min_inliers: int = 40,
    iterations: int = 12,
    yaw_delta_rad: float | None = None,
) -> ScanMotion:
    """Align current scan points into the previous scan's vehicle frame.

    A trimmed point-to-point ICP fit rejects moving/disappearing returns and
    reports invalid motion when the scans do not have enough geometric overlap.
    Forward is +X and lateral-left is +Y, matching the Layer 2 body axes.
    """
    if beam_stride < 1 or min_inliers < 3 or iterations < 1:
        raise ValueError("beam_stride, min_inliers, and iterations must be positive")
    previous, _ = _scan_points(
        previous_ranges_m,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        beam_stride=beam_stride,
    )
    current, _ = _scan_points(
        current_ranges_m,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        beam_stride=beam_stride,
    )
    if min(len(previous), len(current)) < min_inliers:
        return ScanMotion()

    tree = cKDTree(previous)
    # The point order still follows LiDAR beam angle. A local tangent across
    # neighboring returns provides a surface normal for point-to-line ICP,
    # which is less biased by sparse angular sampling than point-to-point fit.
    tangent = np.gradient(previous, axis=0)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    normal_length = np.linalg.norm(normal, axis=1)
    normal_ok = normal_length > 1e-8
    normal[normal_ok] /= normal_length[normal_ok, None]
    if yaw_delta_rad is not None and not np.isfinite(yaw_delta_rad):
        raise ValueError("yaw_delta_rad must be finite when provided")
    fixed_yaw = yaw_delta_rad is not None
    initial_yaw = float(yaw_delta_rad or 0.0)
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    translation = np.zeros(2, dtype=np.float64)
    rmse = math.inf
    inlier_fraction = 0.0
    for _ in range(iterations):
        transformed = current @ rotation.T + translation
        distances, indices = tree.query(transformed, k=1, workers=1)
        inliers = (distances <= max_correspondence_m) & normal_ok[indices]
        if int(np.count_nonzero(inliers)) < min_inliers:
            return ScanMotion(inlier_fraction=float(np.mean(inliers)))

        # Trim the worst quarter of remaining matches so a few dynamic or
        # occluded returns cannot pull the rigid transform toward an outlier.
        cutoff = min(
            max_correspondence_m,
            max(0.20, float(np.quantile(distances[inliers], 0.75))),
        )
        inliers &= distances <= cutoff
        source = transformed[inliers]
        target = previous[indices[inliers]]
        normals = normal[indices[inliers]]
        signed_residual = np.sum(normals * (source - target), axis=1)
        jacobian = np.column_stack((normals[:, 0], normals[:, 1]))
        if not fixed_yaw:
            jacobian = np.column_stack((
                jacobian,
                normals[:, 0] * -source[:, 1] + normals[:, 1] * source[:, 0],
            ))
        try:
            delta, *_ = np.linalg.lstsq(jacobian, -signed_residual, rcond=None)
        except np.linalg.LinAlgError:
            return ScanMotion(inlier_fraction=float(len(source) / len(current)))
        delta_translation = np.asarray(delta[:2], dtype=np.float64)
        if fixed_yaw:
            translation += delta_translation
        else:
            delta_yaw = float(np.clip(delta[2], -0.10, 0.10))
            c, s = math.cos(delta_yaw), math.sin(delta_yaw)
            delta_rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
            rotation = delta_rotation @ rotation
            translation = delta_rotation @ translation + delta_translation
        rmse = float(np.sqrt(np.mean(np.square(signed_residual))))
        inlier_fraction = float(len(source) / len(current))

    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    if (
        not np.isfinite([*translation, yaw, rmse, inlier_fraction]).all()
        or np.linalg.norm(translation) > 0.75
        or abs(yaw) > 0.30
        or rmse > 0.20
        or inlier_fraction < 0.15
    ):
        return ScanMotion(
            forward_m=float(translation[0]),
            lateral_m=float(translation[1]),
            yaw_rad=yaw,
            rmse_m=rmse,
            inlier_fraction=inlier_fraction,
        )
    return ScanMotion(
        forward_m=float(translation[0]),
        lateral_m=float(translation[1]),
        yaw_rad=yaw,
        rmse_m=rmse,
        inlier_fraction=inlier_fraction,
        valid=True,
    )


class LidarOdometry:
    """Stateful scan-to-scan ego-motion estimator for a vehicle sensor stream."""

    def __init__(self, *, min_range_m: float = 0.06, max_range_m: float = 10.0) -> None:
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self._previous_ranges: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the previous scan at a vehicle teleport or episode reset."""
        self._previous_ranges = None

    def update(
        self,
        ranges_m: np.ndarray,
        elapsed_s: float,
        *,
        yaw_delta_rad: float | None = None,
    ) -> ScanMotion:
        """Return scan-derived velocity for one fresh observation."""
        current = np.asarray(ranges_m, dtype=np.float32).reshape(-1).copy()
        if current.size not in (1080, 1081):
            raise ValueError(f"LiDAR odometry requires 1080 or 1081 beams, got {current.size}")
        previous = self._previous_ranges
        self._previous_ranges = current
        if previous is None or not np.isfinite(elapsed_s) or elapsed_s <= 0.0:
            return ScanMotion()
        motion = estimate_scan_motion(
            previous,
            current,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            yaw_delta_rad=yaw_delta_rad,
        )
        if not motion.valid:
            return motion
        return ScanMotion(
            forward_m=motion.forward_m / float(elapsed_s),
            lateral_m=motion.lateral_m / float(elapsed_s),
            yaw_rad=motion.yaw_rad / float(elapsed_s),
            rmse_m=motion.rmse_m,
            inlier_fraction=motion.inlier_fraction,
            valid=True,
        )
