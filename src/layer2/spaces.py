"""
Layer 2 observation / action space helpers.

Gymnasium requires every env to declare:
  - action_space  → what legal actions look like
  - observation_space → what legal observations look like

A "Box" is Gymnasium's name for a block of continuous numbers with a shape
and optional low/high bounds (e.g. two floats in [-1, 1]).

This file also converts Layer 1 TelemetrySnapshot → the obs dict the learner sees.
It does not talk to Unity; AutoDriveEnv calls snapshot_to_obs after each step.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from src.layer1.telemetry import TelemetrySnapshot

# Fixed LiDAR beam count for Layer 2 v1 (PLAN: no downsampling).
LIDAR_BEAMS: int = 1080

# Length of the kinematic "state" vector packed into obs["state"].
# Order is documented in snapshot_to_obs below.
STATE_DIM: int = 8


def make_action_space() -> spaces.Box:
    """
    Legal actions: shape (2,) float32.

    index 0 = throttle in [-1, 1]  (positive accel, negative brake)
    index 1 = steering in [-1, 1]  (left/right convention from the sim)
    """
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def make_observation_space(lidar_beams: int = LIDAR_BEAMS) -> spaces.Dict:
    """
    Legal observations: a Dict space with two keys.

    "lidar":
        1080 range readings scaled to [0, 1] (near → far).
    "state":
        8 kinematic / control floats. Layer 2 v1 does NOT rescale these
        into [0, 1] ("state scaling" is deferred — see env README).
        Bounds are ±inf so Gym only checks shape/dtype.
    """
    return spaces.Dict(
        {
            "lidar": spaces.Box(low=0.0, high=1.0, shape=(lidar_beams,), dtype=np.float32),
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(STATE_DIM,),
                dtype=np.float32,
            ),
        }
    )


def _body_frame_accel(snap: TelemetrySnapshot) -> Tuple[float, float]:
    """
    Convert world-frame linear acceleration into body forward / sideways.

    Unity: X = right, Y = up, Z = forward on the ground plane.
    Same rotation idea Layer 1 uses for v_long / v_lat.
    """
    ax_w, _, az_w = snap.linear_acceleration
    yaw = snap.heading_yaw
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    # Forward (longitudinal) accel along the car nose.
    a_long = az_w * cos_yaw + ax_w * sin_yaw
    # Sideways (lateral) accel.
    a_lat = -az_w * sin_yaw + ax_w * cos_yaw
    return float(a_long), float(a_lat)


def _normalize_lidar(snap: TelemetrySnapshot, lidar_beams: int = LIDAR_BEAMS) -> np.ndarray:
    """
    Force lidar to length lidar_beams and map meters → [0, 1].

    0 ≈ minimum useful range, 1 ≈ maximum sensor range (from the snapshot).
    """
    raw = np.asarray(snap.lidar_ranges, dtype=np.float32).reshape(-1)
    # Pad with zeros if Unity sent fewer beams; truncate if more.
    if raw.size < lidar_beams:
        padded = np.zeros(lidar_beams, dtype=np.float32)
        padded[: raw.size] = raw
        raw = padded
    elif raw.size > lidar_beams:
        raw = raw[:lidar_beams].copy()

    rmax = float(snap.lidar_range_max) if snap.lidar_range_max > 0 else 30.0
    rmin = float(snap.lidar_range_min)
    clipped = np.clip(raw, rmin, rmax)
    # Linear map [rmin, rmax] → [0, 1].
    norm = (clipped - rmin) / max(rmax - rmin, 1e-6)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def snapshot_to_obs(
    snap: TelemetrySnapshot,
    prev_throttle: float,
    prev_steering: float,
    lidar_beams: int = LIDAR_BEAMS,
) -> Dict[str, np.ndarray]:
    """
    Build the Gym observation dict from one Layer 1 TelemetrySnapshot.

    state vector layout (index → meaning):
      0  v_long          forward speed (m/s), body frame
      1  v_lat           sideways speed (m/s), body frame
      2  yaw_rate        rad/s about Unity Y (up)
      3  a_long          forward accel (m/s^2), body frame
      4  a_lat           sideways accel (m/s^2), body frame
      5  slip_angle      sideslip β (radians)
      6  prev_throttle   last throttle command we sent [-1, 1]
      7  prev_steering   last steering command we sent [-1, 1]
    """
    a_long, a_lat = _body_frame_accel(snap)
    # Unity Y-up: yaw rate ≈ angular_velocity.y
    yaw_rate = float(snap.angular_velocity[1])

    state = np.array(
        [
            float(snap.v_long),
            float(snap.v_lat),
            yaw_rate,
            a_long,
            a_lat,
            float(snap.slip_angle),
            float(prev_throttle),
            float(prev_steering),
        ],
        dtype=np.float32,
    )

    return {
        "lidar": _normalize_lidar(snap, lidar_beams=lidar_beams),
        "state": state,
    }
