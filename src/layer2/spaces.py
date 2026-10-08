"""
Layer 2 observation / action space helpers.

Gymnasium requires every env to declare legal actions and observations. This
module converts Layer 1 TelemetrySnapshot values into the policy observation;
it does not communicate with Unity.
"""

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
from gymnasium import spaces

from src.layer1.telemetry import TelemetrySnapshot

# The Bridge scan is configured for -135..+135 degrees at 0.25-degree steps.
# Both endpoints are present in live packets, hence 1081 measurements.
LIDAR_BEAMS: int = 1081
CAMERA_HEIGHT: int = 90
CAMERA_WIDTH: int = 160

# Normalized state channels, kept distinct from policy commands so PPO can
# observe when the simulator's actuator feedback differs from what was sent.
STATE_DIM: int = 9
NegativeThrottleMode = Literal["allow", "zero", "positive_magnitude"]
ThrottleMode = Literal["bidirectional", "forward_only"]
SteeringMode = Literal["normal", "invert"]

_SPEED_SCALE_MPS = 22.88  # simulator RoboRacer maximum speed (82.4 km/h)
_YAW_RATE_SCALE_RAD_S = 10.0  # leaves headroom above observed peak ~5.5 rad/s
_ACCEL_SCALE_M_S2 = 9.80665  # one g
_STEERING_SCALE_RAD = 0.5236  # 30 degrees, Bridge actuator-feedback limit
_STATE_CLIP = 5.0


def make_action_space() -> spaces.Box:
    """Throttle/brake and steering commands, both in [-1, 1]."""
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def map_throttle_action(value: float, mode: NegativeThrottleMode = "allow") -> float:
    """Map a normalized policy throttle into the competition actuator command."""
    if mode not in ("allow", "zero", "positive_magnitude"):
        raise ValueError(f"Unknown negative throttle mode: {mode}")
    throttle = float(np.clip(float(value), -1.0, 1.0))
    if throttle < 0.0:
        if mode == "zero":
            return 0.0
        if mode == "positive_magnitude":
            return -throttle
    return throttle


def map_policy_throttle(value: float, mode: ThrottleMode = "bidirectional") -> float:
    """Map PPO's normalized throttle action to the vehicle actuator range.

    ``forward_only`` maps [-1, 1] onto [0, 1]. This preserves a centered
    initial PPO distribution as half throttle while making reverse impossible.
    """
    if mode not in ("bidirectional", "forward_only"):
        raise ValueError(f"Unknown throttle mode: {mode}")
    normalized = float(np.clip(float(value), -1.0, 1.0))
    if mode == "forward_only":
        return 0.5 * (normalized + 1.0)
    return normalized


def map_steering_action(value: float, mode: SteeringMode = "normal") -> float:
    """Map normalized policy steering to the official actuator convention."""
    if mode not in ("normal", "invert"):
        raise ValueError(f"Unknown steering mode: {mode}")
    steering = float(np.clip(float(value), -1.0, 1.0))
    return -steering if mode == "invert" else steering


def transform_policy_action(
    action: np.ndarray,
    *,
    throttle_mode: ThrottleMode = "bidirectional",
    negative_throttle_mode: NegativeThrottleMode = "allow",
    steering_mode: SteeringMode = "normal",
    steering_action_scale: float = 1.0,
    straight_throttle_gain: float = 1.0,
    straight_throttle_steering_threshold: float = 0.15,
) -> tuple[float, float]:
    """Map one normalized policy action to official throttle/steering commands."""
    command = np.asarray(action, dtype=np.float32).reshape(2)
    throttle = (
        map_policy_throttle(command[0], throttle_mode)
        if throttle_mode == "forward_only"
        else map_throttle_action(command[0], negative_throttle_mode)
    )
    steering = map_steering_action(command[1], steering_mode)
    steering = float(steering * float(steering_action_scale))
    if (
        throttle > 0.0
        and abs(steering) < float(straight_throttle_steering_threshold)
    ):
        throttle = min(1.0, throttle * float(straight_throttle_gain))
    return float(throttle), steering


def make_observation_space(
    lidar_beams: int = LIDAR_BEAMS, *, lidar_history_frames: int = 1,
    include_camera: bool = False,
) -> spaces.Dict:
    """Declare canonical range readings and nine individually scaled state values."""
    if lidar_beams != LIDAR_BEAMS:
        raise ValueError(f"Canonical observation requires {LIDAR_BEAMS} LiDAR beams, got {lidar_beams}")
    if lidar_history_frames < 1:
        raise ValueError("lidar_history_frames must be >= 1")
    lidar_shape = (
        (lidar_beams,) if lidar_history_frames == 1
        else (lidar_history_frames, lidar_beams)
    )
    observation_spaces = {
        "lidar": spaces.Box(low=0.0, high=1.0, shape=lidar_shape, dtype=np.float32),
        "state": spaces.Box(
            low=-_STATE_CLIP,
            high=_STATE_CLIP,
            shape=(STATE_DIM,),
            dtype=np.float32,
        ),
    }
    if include_camera:
        observation_spaces["camera"] = spaces.Box(
            low=0, high=255, shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8
        )
    return spaces.Dict(observation_spaces)


def _normalize_lidar(snap: TelemetrySnapshot, lidar_beams: int = LIDAR_BEAMS) -> np.ndarray:
    """Validate a complete Bridge scan, normalize meters to [0, 1], and align it.

    Positive infinity is AutoDRIVE's no-return value and means max range. Missing,
    malformed, NaN, negative-infinite, or zero-length measurements are rejected;
    inventing zero-range readings would make a telemetry fault look like a wall.
    Both the live 1081-beam scan and the documented 1080-beam variant are accepted.
    """
    if not snap.lidar_valid:
        raise ValueError("AutoDRIVE LiDAR scan is missing or malformed; refusing to build a policy observation")

    raw = np.asarray(snap.lidar_ranges, dtype=np.float32).reshape(-1)
    if raw.size not in (1080, 1081):
        raise ValueError(
            f"AutoDRIVE LiDAR scan must contain 1080 or 1081 beams, got {raw.size}"
        )
    if np.isnan(raw).any() or np.isneginf(raw).any():
        raise ValueError("AutoDRIVE LiDAR scan contains NaN or negative-infinite ranges")
    if (raw <= 0.0).any():
        raise ValueError("AutoDRIVE LiDAR scan contains nonpositive ranges")

    rmin = float(snap.lidar_range_min)
    rmax = float(snap.lidar_range_max)
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin < 0 or rmax <= rmin:
        raise ValueError(f"Invalid AutoDRIVE LiDAR range limits: min={rmin}, max={rmax}")

    # AutoDRIVE encodes a ray with no hit as +inf; it also suppresses hits
    # nearer than the minimum range. Treat those sub-min echoes as no-return,
    # then map them to max range. Do not reverse here: Layer 1 has already
    # canonicalized Bridge's clockwise serialization to left→right.
    no_return = (raw < rmin) | np.isposinf(raw)
    interpreted = np.where(no_return, rmax, raw)
    clipped = np.clip(interpreted, rmin, rmax)
    normalized = (clipped - rmin) / (rmax - rmin)

    if raw.size == 1080:
        # The documented endpoint-exclusive variant omits the final 0.25° ray.
        # Interpolate by angle to the canonical inclusive -135..+135 grid.
        source_angles = np.linspace(0.0, 1.0, num=1080, endpoint=True)
        target_angles = np.linspace(0.0, 1.0, num=lidar_beams, endpoint=True)
        normalized = np.interp(target_angles, source_angles, normalized)
    elif raw.size != lidar_beams:
        raise ValueError(
            f"Observation requires {lidar_beams} canonical LiDAR beams, got {raw.size}"
        )

    if normalized.size != lidar_beams:
        raise ValueError(f"LiDAR normalization produced {normalized.size} beams, expected {lidar_beams}")
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def snapshot_to_obs(
    snap: TelemetrySnapshot,
    prev_throttle: float,
    prev_steering: float,
    lidar_beams: int = LIDAR_BEAMS,
    *,
    forward_speed_mps: float | None = None,
    lateral_speed_mps: float | None = None,
    lidar_history: list[np.ndarray] | None = None,
    include_camera: bool = False,
) -> Dict[str, np.ndarray]:
    """Build the normalized policy observation from one Bridge snapshot.

    State channels, individually normalized:
      0  body forward velocity or supplied sensor-derived estimate / max speed
      1  body lateral velocity (positive right) / max speed, or supplied zero
      2  Unity-convention yaw rate / 10 rad/s
      3  body forward acceleration / 1 g
      4  body lateral acceleration (positive right) / 1 g
      5  measured throttle feedback [-1, 1]
      6  measured steering feedback / 30 degrees
      7  previous throttle command [-1, 1]
      8  previous steering command [-1, 1]
    """
    if lidar_beams != LIDAR_BEAMS:
        raise ValueError(f"Canonical observation requires {LIDAR_BEAMS} LiDAR beams, got {lidar_beams}")

    yaw_rate = -float(snap.angular_velocity[2])
    a_long = float(snap.linear_acceleration[0])
    a_lat = -float(snap.linear_acceleration[1])
    state_raw = np.asarray(
        [
            float(snap.v_long if forward_speed_mps is None else forward_speed_mps) / _SPEED_SCALE_MPS,
            float(snap.v_lat if lateral_speed_mps is None else lateral_speed_mps) / _SPEED_SCALE_MPS,
            yaw_rate / _YAW_RATE_SCALE_RAD_S,
            a_long / _ACCEL_SCALE_M_S2,
            a_lat / _ACCEL_SCALE_M_S2,
            float(np.clip(float(snap.throttle), -1.0, 1.0)),
            float(np.clip(float(snap.steering) / _STEERING_SCALE_RAD, -1.0, 1.0)),
            float(np.clip(float(prev_throttle), -1.0, 1.0)),
            float(np.clip(float(prev_steering), -1.0, 1.0)),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(state_raw).all():
        raise ValueError("AutoDRIVE state telemetry contains NaN or infinite values")

    current_lidar = _normalize_lidar(snap, lidar_beams=lidar_beams)
    if lidar_history is None:
        lidar_observation = current_lidar
    else:
        history = [np.asarray(frame, dtype=np.float32).reshape(-1) for frame in lidar_history]
        if not history or any(frame.shape != current_lidar.shape for frame in history):
            raise ValueError("LiDAR history must contain canonical normalized scans")
        if any(not np.isfinite(frame).all() or np.any((frame < 0) | (frame > 1)) for frame in history):
            raise ValueError("LiDAR history frames must be finite normalized ranges in [0, 1]")
        # The final channel is always the current scan; earlier channels are
        # ordered oldest→newest so temporal convolutions have a stable meaning.
        lidar_observation = np.stack([*history, current_lidar], axis=0).astype(np.float32)

    observation = {
        "lidar": lidar_observation,
        "state": np.clip(state_raw, -_STATE_CLIP, _STATE_CLIP).astype(np.float32),
    }
    if include_camera:
        raw_image = snap.camera_image_rgb
        if raw_image is None:
            raise ValueError("front camera image is required by the official camera observation profile")
        image = np.asarray(raw_image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("front camera image must be an HxWx3 uint8 RGB array")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("front camera image must have nonzero width and height")
        rows = np.linspace(0, image.shape[0] - 1, CAMERA_HEIGHT).astype(np.intp)
        cols = np.linspace(0, image.shape[1] - 1, CAMERA_WIDTH).astype(np.intp)
        observation["camera"] = np.ascontiguousarray(image[rows[:, None], cols[None, :], :])
    return observation
