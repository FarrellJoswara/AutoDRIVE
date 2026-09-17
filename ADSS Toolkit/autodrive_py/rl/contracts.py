"""Frozen interface constants — contracts.md v2.0.0.

Obs dim = N_LIDAR + 6 (prev action ×2, speed, IMU ×3). Action MultiDiscrete[4,11].
Race score λ=10: ``adjusted_time = lap_time + 10·collisions``.
Do not load v1 LiDAR-only zips into a v2 env (refuse-load in ``compat``).
"""

from __future__ import annotations

import numpy as np

CONTRACTS_VERSION = "2.0.0"

N_LIDAR_DEFAULT = 180
LIDAR_MAX_M = 10.0

# Proprioception / IMU (legal race-time; camera deferred)
N_PREV_ACTION = 2
N_SPEED = 1
N_IMU = 3  # yaw_rate, ax, ay
OBS_PROPRIO_DIM = N_PREV_ACTION + N_SPEED + N_IMU  # 6
# default obs_dim = 180 + 6 = 186

SPEED_MAX_MPS = 6.0  # normalize speed → [0, 1]
YAW_RATE_MAX_RAD_S = 10.0  # normalize yaw_rate → [-1, 1]
ACCEL_MAX_MPS2 = 10.0  # normalize ax, ay → [-1, 1]

D_WALL_M = 0.3
COLLISION_PENALTY = 10.0
TIMEOUT_S = 60.0
# A lap only counts once this fraction of the centerline has been covered as
# genuine forward high-water progress. Geometric "near the start line again"
# is not sufficient: projection re-anchors can jump s by most of a lap.
LAP_MIN_PROGRESS_FRAC = 0.99

# How a race score was computed. Independent of CONTRACTS_VERSION, which pins the
# obs/action ABI: a scoring fix does not invalidate a policy's weights, but it does
# invalidate old leaderboard rows.
#   rev 1 — lap credited on geometric proximity to the start line (unsound: a
#           projection re-anchor could claim a full lap after 0 m of travel)
#   rev 2 — lap credited on high-water forward progress (LAP_MIN_PROGRESS_FRAC)
SCORING_REV = 2
# Truncate if no meaningful forward centerline progress for this long (sim time).
STALL_TIMEOUT_S = 8.0
# Minimum forward Δs (m) that counts as "meaningful" progress for stall reset.
PROGRESS_EPS_M = 0.05

W_PROGRESS = 1.0
W_SPEED = 0.01
W_WALL = 0.1
W_TIME = 0.001

# Collision-first / speed-gate curriculum (CLI flags; defaults off for legacy smoke).
COLLISION_FIRST_PENALTY = 50.0  # terminal collision cost when --collision-first
SPEED_GATE_CRASH_RATE = 0.05  # unlock speed reward only under this rolling crash rate
SPEED_GATE_WINDOW = 20  # episodes in rolling window (env-local estimate)

# TTC / frontal collapse truncate (LiDAR-only soft truncate).
TTC_FRONT_BEAMS = 21  # odd; center ±10 of FOV
TTC_MIN_RANGE_M = 0.45  # frontal min-range truncate threshold
TTC_HORIZON_S = 0.6  # if closing fast, also truncate

# Spawn jitter (train generalization)
SPAWN_JITTER_LATERAL_M = 0.25
SPAWN_JITTER_LONG_M = 0.8
SPAWN_JITTER_HEADING_RAD = 0.15

# Light LiDAR domain randomization (train only)
LIDAR_DR_NOISE_STD = 0.02  # on normalized [0,1] ranges
LIDAR_DR_DROPOUT = 0.02
LIDAR_DR_MAX_RANGE_SCALE = 0.95  # occasional shorter max

THROTTLE_BINS = np.array([0.00, 0.33, 0.66, 1.00], dtype=np.float32)
STEERING_BINS = np.linspace(-1.0, 1.0, 11, dtype=np.float32)

ACTION_NVEC = (4, 11)


def obs_dim(n_lidar: int = N_LIDAR_DEFAULT) -> int:
    """Total observation vector length for contracts v2."""
    return int(n_lidar) + OBS_PROPRIO_DIM


def decode_throttle(i: int) -> float:
    return float(THROTTLE_BINS[int(i)])


def decode_steering(j: int) -> float:
    return float(STEERING_BINS[int(j)])


def decode_action(action) -> tuple[float, float]:
    """MultiDiscrete [throttle_idx, steering_idx] → (throttle, steering) in [-1, 1]."""
    return decode_throttle(action[0]), decode_steering(action[1])


def encode_throttle(throttle: float) -> int:
    return int(np.argmin(np.abs(THROTTLE_BINS - float(throttle))))


def encode_steering(steering: float) -> int:
    return int(np.argmin(np.abs(STEERING_BINS - float(steering))))


def encode_action(throttle: float, steering: float) -> np.ndarray:
    return np.array([encode_throttle(throttle), encode_steering(steering)], dtype=np.int64)
