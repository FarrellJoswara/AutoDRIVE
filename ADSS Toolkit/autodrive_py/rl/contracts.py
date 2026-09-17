"""Frozen interface constants — contracts.md v2.0.0."""

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

W_PROGRESS = 1.0
W_SPEED = 0.01
W_WALL = 0.1
W_TIME = 0.001

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
