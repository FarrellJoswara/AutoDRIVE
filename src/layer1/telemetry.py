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


def _parse_quat(val: Any) -> Tuple[float, float, float, float]:
    """Parse a quaternion (x, y, z, w) from string, list, tuple, or dict format."""
    if isinstance(val, str):
        parts = val.strip().split()
        if len(parts) >= 4:
            return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 4:
        return (float(val[0]), float(val[1]), float(val[2]), float(val[3]))
    if isinstance(val, dict):
        return (
            float(val.get("x", 0.0)),
            float(val.get("y", 0.0)),
            float(val.get("z", 0.0)),
            float(val.get("w", 1.0)),
        )
    return (0.0, 0.0, 0.0, 1.0)


@dataclass
class TelemetrySnapshot:
    """100% Raw AutoDRIVE Telemetry + Derived Vehicle Dynamics."""

    # Timestamp & Step ID
    timestamp: float = 0.0
    step_id: int = 0

    # Raw 3D Kinematics (Unity Coordinate Frame: X=Right, Y=Up, Z=Forward)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Wheel Encoders (Left & Right wheel angles/speed)
    encoder_left: float = 0.0
    encoder_right: float = 0.0

    # 1080-Beam 2D LiDAR Range Array (in meters)
    lidar_ranges: np.ndarray = field(default_factory=lambda: np.zeros(1080, dtype=np.float32))
    lidar_scan_rate: float = 40.0
    lidar_range_min: float = 0.05
    lidar_range_max: float = 30.0

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
        pos = _parse_vec3(data.get("V1 Position", data.get("position", [0, 0, 0])))
        quat = _parse_quat(data.get("V1 Orientation", data.get("orientation", [0, 0, 0, 1])))
        lin_vel = _parse_vec3(data.get("V1 Linear Velocity", data.get("linear_velocity", [0, 0, 0])))
        ang_vel = _parse_vec3(data.get("V1 Angular Velocity", data.get("angular_velocity", [0, 0, 0])))
        lin_acc = _parse_vec3(data.get("V1 Linear Acceleration", data.get("linear_acceleration", [0, 0, 0])))

        # Encoders: may be space-separated string "left right" or separate keys
        enc_raw = data.get("V1 Encoder Angles", data.get("encoder_angles", None))
        if isinstance(enc_raw, str):
            enc_parts = enc_raw.strip().split()
            enc_l = float(enc_parts[0]) if len(enc_parts) > 0 else 0.0
            enc_r = float(enc_parts[1]) if len(enc_parts) > 1 else 0.0
        elif isinstance(enc_raw, (list, tuple, np.ndarray)) and len(enc_raw) >= 2:
            enc_l, enc_r = float(enc_raw[0]), float(enc_raw[1])
        else:
            enc_l = float(data.get("V1 Left Encoder", data.get("encoder_left", 0.0)))
            enc_r = float(data.get("V1 Right Encoder", data.get("encoder_right", 0.0)))

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

        th = float(data.get("V1 Throttle", data.get("throttle", 0.0)))
        st = float(data.get("V1 Steering", data.get("steering", 0.0)))

        lap_c = int(float(data.get("V1 Lap Count", data.get("lap_count", 0))))
        lap_t = float(data.get("V1 Lap Time", data.get("lap_time", 0.0)))
        last_t = float(data.get("V1 Last Lap Time", data.get("last_lap_time", 0.0)))
        best_t = float(data.get("V1 Best Lap Time", data.get("best_lap_time", 0.0)))

        raw_col = data.get("V1 Collision", data.get("V1 Collisions", data.get("collision", False)))
        is_col = bool(raw_col) if not isinstance(raw_col, (int, float, str)) else (float(raw_col) > 0)
        col_count = int(float(data.get("V1 Collision Count", data.get("V1 Collisions", data.get("collision_count", int(is_col))))))

        # 2. Derive 2D Dynamics
        # True 3D scalar speed
        speed = math.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2 + lin_vel[2] ** 2)

        # Planar heading yaw psi from quaternion (x, y, z, w)
        # Unity standard: Y-up, yaw is rotation around Y-axis
        qx, qy, qz, qw = quat
        siny_cosp = 2.0 * (qw * qy + qz * qx)
        cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Body-frame velocities (Unity horizontal plane is X-Z):
        # Let forward velocity be along vehicle orientation:
        # vx_world, vz_world rotated by -yaw:
        vx_w, _, vz_w = lin_vel
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Forward (longitudinal) and sideways (lateral) velocity
        v_long = vz_w * cos_yaw + vx_w * sin_yaw
        v_lat = -vz_w * sin_yaw + vx_w * cos_yaw

        # Sideslip angle beta: angle between heading vector and velocity vector
        slip = math.atan2(v_lat, max(abs(v_long), 0.05)) if speed > 0.1 else 0.0

        # Lateral acceleration & G-force
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
            lidar_ranges=lidar_arr,
            lidar_scan_rate=float(data.get("V1 Lidar Scan Rate", 40.0)),
            lidar_range_min=float(data.get("V1 Lidar Range Min", 0.05)),
            lidar_range_max=float(data.get("V1 Lidar Range Max", 30.0)),
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
