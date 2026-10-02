"""Complete AutoDRIVE RoboRacer Telemetry Data Model, Dynamics, and CSV Logger.

Captures 100% of raw simulator telemetry signals (3D kinematics, accelerations,
wheel encoders, 1080-beam LiDAR, lap timers, collisions) and calculates derived
vehicle dynamics (true speed, planar yaw, body-frame velocities, slip angle, lateral G).
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


import base64
import gzip


def _parse_vec3(val: Any) -> Tuple[float, float, float]:
    """Parse a 3D vector from string, list, tuple, or dict format."""
    if isinstance(val, str):
        parts = val.strip().split()
        if len(parts) >= 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3:
        return (float(val[0]), float(val[1]), float(val[2]))
    if isinstance(val, dict):
        return (float(val.get("x", 0.0)), float(val.get("y", 0.0)), float(val.get("z", 0.0)))
    return (0.0, 0.0, 0.0)


def _bridge_position_to_unity(raw: Any) -> Tuple[float, float, float]:
    """AutoDRIVE Bridge V1 Position -> Unity world metres.

    Bridge space-separated triple is NOT Unity xyz. Measured vs TrackLoader spawn:
    bridge (bz, bx, by) maps to Unity (-bx, by, bz) i.e. order (z, -x, y).
    List/tuple/dict inputs are already-Unity. Velocity is already Unity.
    """
    if isinstance(raw, str):
        bz, bx, by = _parse_vec3(raw)
        return (-bx, by, bz)
    return _parse_vec3(raw)


def _parse_quat(val: Any) -> Tuple[float, float, float, float]:
    """Parse a quaternion (x, y, z, w) from string, list, tuple, or dict format."""
    if isinstance(val, str):
        parts = val.strip().split()
        if len(parts) >= 4:
            return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        # AutoDRIVE HUD/recorder sometimes send euler "roll pitch yaw"
        if len(parts) == 3:
            return _quat_from_euler(float(parts[0]), float(parts[1]), float(parts[2]))
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) >= 4:
            return (float(val[0]), float(val[1]), float(val[2]), float(val[3]))
        if len(val) == 3:
            # RoboRacer guide IMU: Orientation [x,y,z] rad = roll, pitch, yaw
            return _quat_from_euler(float(val[0]), float(val[1]), float(val[2]))
    if isinstance(val, dict):
        if "w" in val or "W" in val:
            return (
                float(val.get("x", 0.0)),
                float(val.get("y", 0.0)),
                float(val.get("z", 0.0)),
                float(val.get("w", val.get("W", 1.0))),
            )
        if "yaw" in val or "Yaw" in val:
            return _quat_from_euler(
                float(val.get("roll", val.get("x", 0.0))),
                float(val.get("pitch", val.get("y", 0.0))),
                float(val.get("yaw", val.get("Yaw", 0.0))),
            )
    return (0.0, 0.0, 0.0, 1.0)


def _parse_euler_rpy(val: Any) -> Optional[Tuple[float, float, float]]:
    """Parse Bridge euler triple as (roll, pitch, yaw) radians, or None."""
    if isinstance(val, str):
        parts = val.strip().split()
        if len(parts) >= 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        return None
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3:
        return (float(val[0]), float(val[1]), float(val[2]))
    if isinstance(val, dict):
        return (
            float(val.get("roll", val.get("x", 0.0))),
            float(val.get("pitch", val.get("y", 0.0))),
            float(val.get("yaw", val.get("Yaw", val.get("z", 0.0)))),
        )
    return None


def _parse_pair(val: Any) -> Optional[Tuple[float, float]]:
    """Parse a left/right pair from string, list, or tuple."""
    if isinstance(val, str):
        parts = val.strip().split()
        if len(parts) >= 2:
            return (float(parts[0]), float(parts[1]))
        return None
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
        return (float(val[0]), float(val[1]))
    return None


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """RPY (radians) → quaternion (x, y, z, w) for snapshot storage."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def _quat_is_identity(quat: Tuple[float, float, float, float], eps: float = 1e-3) -> bool:
    qx, qy, qz, qw = quat
    return abs(qx) + abs(qy) + abs(qz) < eps and abs(abs(qw) - 1.0) < eps


def _yaw_from_quat(quat: Tuple[float, float, float, float]) -> float:
    """Planar yaw about Unity +Y from quaternion (x, y, z, w).

    Matches Unity Transform.rotation (Y-up). Residual risk: if a given
    AutoDRIVE build remaps IMU axes differently from Watch heading, yaw
    may need an axis remap — do not assume Z-up ROS here without evidence.
    """
    qx, qy, qz, qw = quat
    siny_cosp = 2.0 * (qw * qy + qz * qx)
    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(siny_cosp, cosy_cosp)


def _bridge_quat_to_unity(
    quat: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert Bridge orientation axes to Unity XYZ axes.

    Bridge position ``(bz, bx, by)`` maps to Unity ``(-bx, by, bz)``.
    Orientation is an axial vector, so the handedness change contributes a
    determinant sign: qUnity.xyz = (qBridge.y, -qBridge.z, -qBridge.x).
    Live Socket samples confirm a Bridge Z rotation is Unity Y yaw.
    """
    qx, qy, qz, qw = quat
    return (qy, -qz, -qx, qw)


def _bridge_orientation(
    data: Dict[str, Any],
) -> Tuple[Tuple[float, float, float, float], Optional[float]]:
    """Resolve Bridge orientation → (quat xyzw, optional yaw from euler).

    Socket.cs emits:
      - ``V1 Orientation Quaternion`` → ``qx qy qz qw``
      - ``V1 Orientation Euler Angles`` → ``roll pitch yaw`` (rad)
    It does **not** emit bare ``V1 Orientation``. Convert canonical Bridge
    orientation axes to Unity axes before storing or extracting yaw.
    """
    q_raw = data.get("V1 Orientation Quaternion")
    if q_raw is not None and q_raw != "":
        return _bridge_quat_to_unity(_parse_quat(q_raw)), None

    e_raw = data.get("V1 Orientation Euler Angles")
    if e_raw is not None and e_raw != "":
        rpy = _parse_euler_rpy(e_raw)
        if rpy is not None:
            roll, pitch, yaw = rpy
            unity_quat = _bridge_quat_to_unity(_quat_from_euler(roll, pitch, yaw))
            return unity_quat, _yaw_from_quat(unity_quat)

    legacy = data.get("V1 Orientation", data.get("orientation"))
    if legacy is None or legacy == "":
        return (0.0, 0.0, 0.0, 1.0), None

    # Legacy 3-vector is treated as Bridge Euler; convert its axes like the
    # canonical Socket keys. Four-component legacy values are quaternions.
    rpy = _parse_euler_rpy(legacy)
    if rpy is not None:
        # Distinguish quat-looking 4-vectors from euler triples.
        if isinstance(legacy, str):
            n = len(legacy.strip().split())
        elif isinstance(legacy, (list, tuple, np.ndarray)):
            n = len(legacy)
        else:
            n = 4 if isinstance(legacy, dict) and ("w" in legacy or "W" in legacy) else 3
        if n == 3:
            roll, pitch, yaw = rpy
            return _bridge_quat_to_unity(_quat_from_euler(roll, pitch, yaw)), None

    return _parse_quat(legacy), None


@dataclass
class TelemetrySnapshot:
    """100% Raw AutoDRIVE Telemetry + Derived Vehicle Dynamics."""

    # Timestamp & Step ID
    timestamp: float = 0.0
    step_id: int = 0

    # Raw 3D Kinematics (Unity world: X=Right, Y=Up, Z=Forward).
    # Angular velocity from Bridge is AutoDRIVE body frame [wx, wy, wz].
    # Linear vel/accel are Unity world
    # XYZ when projecting to body frame below — if a build ever sends body-frame
    # lin_vel/accel, that projection would double-rotate (do not change without evidence).
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Wheel Encoders (Left & Right wheel angles; optional tick counts)
    encoder_left: float = 0.0
    encoder_right: float = 0.0
    encoder_ticks_left: float = 0.0
    encoder_ticks_right: float = 0.0

    # 1080-Beam 2D LiDAR Range Array (in meters)
    lidar_ranges: np.ndarray = field(default_factory=lambda: np.zeros(1080, dtype=np.float32))
    lidar_scan_rate: float = 40.0
    lidar_range_min: float = 0.06
    lidar_range_max: float = 10.0

    # Actuator States & Feedback
    throttle: float = 0.0
    steering: float = 0.0

    # Lap Timing & Race Statistics
    lap_count: int = 0
    lap_time: float = 0.0
    last_lap_time: float = 0.0
    best_lap_time: float = 0.0

    # Safety & Collision Signals
    collision: bool = False
    collision_count: int = 0

    # Derived Vehicle Dynamics (Planar 2D Physics)
    true_speed: float = 0.0  # Scalar magnitude of linear velocity (m/s)
    heading_yaw: float = 0.0  # Planar yaw angle psi in radians [-pi, pi]
    v_long: float = 0.0  # Longitudinal velocity in vehicle body frame (m/s)
    v_lat: float = 0.0  # Lateral velocity in vehicle body frame (m/s)
    slip_angle: float = 0.0  # Sideslip angle beta in radians [-pi, pi]
    lateral_g: float = 0.0  # Lateral G-force (g's)

    # Frenet Coordinates (Track-Relative, updated if waypoints exist)
    frenet_s: float = 0.0  # Distance along centerline (m)
    frenet_d: float = 0.0  # Lateral deviation from centerline (m)

    @classmethod
    def from_raw_dict(cls, data: Dict[str, Any], step_id: int = 0) -> TelemetrySnapshot:
        """Parse raw AutoDRIVE 'Bridge' dictionary and compute vehicle dynamics."""
        # 1. Raw signals extraction with tolerant key lookups
        pos = _bridge_position_to_unity(
            data.get("V1 Position", data.get("position", [0, 0, 0]))
        )
        quat, euler_yaw = _bridge_orientation(data)
        # Velocity/accel arrive as Unity XYZ (verified vs pose deltas); only Position is swizzled.
        lin_vel = _parse_vec3(data.get("V1 Linear Velocity", data.get("linear_velocity", [0, 0, 0])))
        # Body RHS: [wx, wy, wz]; planar yaw rate is wz (index 2), not wy.
        ang_vel = _parse_vec3(data.get("V1 Angular Velocity", data.get("angular_velocity", [0, 0, 0])))
        lin_acc = _parse_vec3(data.get("V1 Linear Acceleration", data.get("linear_acceleration", [0, 0, 0])))

        # Encoders: angles (rad) and optional tick counts — space-separated "left right"
        enc_pair = _parse_pair(data.get("V1 Encoder Angles", data.get("encoder_angles")))
        if enc_pair is not None:
            enc_l, enc_r = enc_pair
        else:
            enc_l = float(data.get("V1 Left Encoder", data.get("encoder_left", 0.0)))
            enc_r = float(data.get("V1 Right Encoder", data.get("encoder_right", 0.0)))

        ticks_pair = _parse_pair(data.get("V1 Encoder Ticks", data.get("encoder_ticks")))
        if ticks_pair is not None:
            ticks_l, ticks_r = ticks_pair
        else:
            ticks_l = float(data.get("V1 Left Encoder Ticks", data.get("encoder_ticks_left", 0.0)))
            ticks_r = float(data.get("V1 Right Encoder Ticks", data.get("encoder_ticks_right", 0.0)))

        # LiDAR: may be base64-encoded gzip string or float array
        raw_lidar = data.get("V1 LIDAR Range Array", data.get("V1 Lidar Scan", data.get("lidar_scan", [])))
        if isinstance(raw_lidar, str) and len(raw_lidar) > 0:
            try:
                decomp = gzip.decompress(base64.b64decode(raw_lidar)).decode("utf-8")
                lidar_arr = np.fromstring(decomp, dtype=np.float32, sep="\n")
            except Exception:
                lidar_arr = np.zeros(1080, dtype=np.float32)
        elif isinstance(raw_lidar, (list, tuple, np.ndarray)) and len(raw_lidar) > 0:
            lidar_arr = np.asarray(raw_lidar, dtype=np.float32)
        else:
            lidar_arr = np.zeros(1080, dtype=np.float32)

        # Bridge serializes the planar scan clockwise (end angle → start
        # angle), while this stack's canonical convention is CCW from
        # angle_min. Live 74-pose comparison against Porto occupancy walls
        # confirms reversing the Bridge scan aligns the ranges with pose+yaw.
        lidar_arr = lidar_arr[::-1].copy()

        th = float(data.get("V1 Throttle", data.get("throttle", 0.0)))
        st = float(data.get("V1 Steering", data.get("steering", 0.0)))

        lap_c = int(float(data.get("V1 Lap Count", data.get("lap_count", 0))))
        lap_t = float(data.get("V1 Lap Time", data.get("lap_time", 0.0)))
        last_t = float(data.get("V1 Last Lap Time", data.get("last_lap_time", 0.0)))
        best_t = float(data.get("V1 Best Lap Time", data.get("best_lap_time", 0.0)))

        # Socket.cs primary key is "V1 Collisions" (flag/count). Legacy: V1 Collision.
        raw_col = data.get("V1 Collisions", data.get("V1 Collision", data.get("collision", False)))
        is_col = bool(raw_col) if not isinstance(raw_col, (int, float, str)) else (float(raw_col) > 0)
        if "V1 Collision Count" in data:
            col_count = int(float(data["V1 Collision Count"]))
        elif "V1 Collisions" in data:
            try:
                col_count = int(float(data["V1 Collisions"]))
            except (TypeError, ValueError):
                col_count = int(is_col)
        else:
            col_count = int(float(data.get("collision_count", int(is_col))))

        # 2. Derive 2D Dynamics
        # True 3D scalar speed
        speed = math.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2 + lin_vel[2] ** 2)

        # Heading from real IMU orientation keys. Velocity yaw is last resort only
        # when quat/euler are still identity/missing AND speed is high enough —
        # not because "Unity sends identity" (that was a wrong-key bug).
        if euler_yaw is not None:
            yaw = float(euler_yaw)
            orientation_ok = True
        elif not _quat_is_identity(quat):
            yaw = _yaw_from_quat(quat)
            orientation_ok = True
        else:
            yaw = 0.0
            orientation_ok = False
            explicit = data.get("V1 Yaw", data.get("yaw", None))
            if explicit is not None and explicit != "":
                try:
                    yaw = float(explicit)
                    orientation_ok = True
                except (TypeError, ValueError):
                    pass
        if not orientation_ok and speed > 0.05:
            yaw = math.atan2(lin_vel[0], lin_vel[2])

        # Body-frame velocities (Unity horizontal plane is X-Z):
        vx_w, _, vz_w = lin_vel
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Forward (longitudinal) and sideways (lateral) velocity
        v_long = vz_w * cos_yaw + vx_w * sin_yaw
        v_lat = -vz_w * sin_yaw + vx_w * cos_yaw

        # Sideslip angle beta: angle between heading vector and velocity vector
        slip = math.atan2(v_lat, max(abs(v_long), 0.05)) if speed > 0.1 else 0.0

        # Lateral acceleration & G-force (assumes lin_acc is world XYZ)
        ax_w, _, az_w = lin_acc
        a_lat = -az_w * sin_yaw + ax_w * cos_yaw
        lat_g = a_lat / 9.80665

        return cls(
            timestamp=time.time(),
            step_id=step_id,
            position=pos,
            orientation_quat=quat,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
            linear_acceleration=lin_acc,
            encoder_left=enc_l,
            encoder_right=enc_r,
            encoder_ticks_left=ticks_l,
            encoder_ticks_right=ticks_r,
            lidar_ranges=lidar_arr,
            lidar_scan_rate=float(
                data.get("V1 LIDAR Scan Rate", data.get("V1 Lidar Scan Rate", 40.0))
            ),
            lidar_range_min=float(data.get("V1 Lidar Range Min", 0.06)),
            lidar_range_max=float(data.get("V1 Lidar Range Max", 10.0)),
            throttle=th,
            steering=st,
            lap_count=lap_c,
            lap_time=lap_t,
            last_lap_time=last_t,
            best_lap_time=best_t,
            collision=is_col,
            collision_count=col_count,
            true_speed=speed,
            heading_yaw=yaw,
            v_long=v_long,
            v_lat=v_lat,
            slip_angle=slip,
            lateral_g=lat_g,
        )


class TrajectoryLogger:
    """In-memory telemetry buffer and CSV exporter for telemetry analysis."""

    CSV_HEADERS = [
        "step_id",
        "timestamp",
        "pos_x",
        "pos_y",
        "pos_z",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
        "true_speed_mps",
        "heading_yaw_rad",
        "v_long_mps",
        "v_lat_mps",
        "slip_angle_rad",
        "lateral_g",
        "throttle",
        "steering",
        "encoder_left",
        "encoder_right",
        "lap_count",
        "lap_time_s",
        "collision",
        "frenet_s",
        "frenet_d",
    ]

    def __init__(self, max_buffer_size: int = 100_000) -> None:
        self.max_buffer_size = max_buffer_size
        self._records: List[Dict[str, Any]] = []

    def log(self, snap: TelemetrySnapshot) -> None:
        """Record a telemetry snapshot into memory."""
        if len(self._records) >= self.max_buffer_size:
            self._records.pop(0)

        record = {
            "step_id": snap.step_id,
            "timestamp": snap.timestamp,
            "pos_x": snap.position[0],
            "pos_y": snap.position[1],
            "pos_z": snap.position[2],
            "quat_x": snap.orientation_quat[0],
            "quat_y": snap.orientation_quat[1],
            "quat_z": snap.orientation_quat[2],
            "quat_w": snap.orientation_quat[3],
            "true_speed_mps": round(snap.true_speed, 4),
            "heading_yaw_rad": round(snap.heading_yaw, 4),
            "v_long_mps": round(snap.v_long, 4),
            "v_lat_mps": round(snap.v_lat, 4),
            "slip_angle_rad": round(snap.slip_angle, 4),
            "lateral_g": round(snap.lateral_g, 4),
            "throttle": round(snap.throttle, 3),
            "steering": round(snap.steering, 3),
            "encoder_left": round(snap.encoder_left, 3),
            "encoder_right": round(snap.encoder_right, 3),
            "lap_count": snap.lap_count,
            "lap_time_s": round(snap.lap_time, 3),
            "collision": int(snap.collision),
            "frenet_s": round(snap.frenet_s, 3),
            "frenet_d": round(snap.frenet_d, 3),
        }
        self._records.append(record)

    def save_to_csv(self, file_path: Union[str, Path]) -> str:
        """Export all buffered records to a CSV file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            writer.writerows(self._records)

        return str(path.resolve())

    def clear(self) -> None:
        """Clear recorded history."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
