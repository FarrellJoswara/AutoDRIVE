"""Layer 1 transport adapter for the official AutoDRIVE ROS 2 Devkit.

The adapter presents the same small ``step/telemetry`` surface as the
Socket.IO ``Racer``. ROS messages are converted back to Bridge V1 fields
consumed by :class:`TelemetrySnapshot`, keeping coordinate and sensor
interpretation in one place instead of creating a second parser.
"""

from __future__ import annotations

import threading
import time
import math
import os
import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional

from .telemetry import TelemetrySnapshot


class OfficialRosTransportError(RuntimeError):
    """A fatal or timed-out official ROS transport operation."""


class OfficialTrainingResetCancelled(OfficialRosTransportError):
    """Training reset exited promptly because the run was asked to stop."""


@dataclass(frozen=True)
class RaceMetrics:
    """Restricted simulator data for evaluation-only instrumentation."""

    lap_count: int = 0
    lap_time: float = 0.0
    last_lap_time: float = 0.0
    best_lap_time: float = 0.0
    collision_count: int = 0
    position: Optional[tuple[float, float, float]] = None
    position_received_at: float = 0.0
    lap_count_update_count: int = 0
    lap_count_change_count: int = 0
    last_lap_time_update_count: int = 0
    last_lap_time_update_count_at_lap_count: int = 0


def _xyz(value: Any) -> str:
    return f"{float(value.x)} {float(value.y)} {float(value.z)}"


def ros_image_to_rgb(message: Any) -> Any:
    """Decode a ROS sensor_msgs/Image into a contiguous uint8 RGB array."""
    import numpy as np

    height, width = int(message.height), int(message.width)
    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "rgb8": (3, "rgb"), "bgr8": (3, "bgr"),
        "rgba8": (4, "rgba"), "bgra8": (4, "bgra"),
        "mono8": (1, "mono"), "8uc1": (1, "mono"),
        "8uc3": (3, "bgr"),
    }
    if height <= 0 or width <= 0 or encoding not in channels_by_encoding:
        raise ValueError(f"unsupported ROS camera image {width}x{height} encoding={encoding!r}")
    channels, layout = channels_by_encoding[encoding]
    row_bytes = width * channels
    step = int(getattr(message, "step", row_bytes) or row_bytes)
    if step < row_bytes:
        raise ValueError(f"ROS camera row step {step} is smaller than {row_bytes} bytes")
    data = memoryview(message.data)
    if len(data) < height * step:
        raise ValueError(
            f"ROS camera payload has {len(data)} bytes; expected at least {height * step}"
        )
    rows = np.frombuffer(data, dtype=np.uint8, count=height * step).reshape(height, step)
    pixels = rows[:, :row_bytes].reshape(height, width, channels)
    if layout == "mono":
        pixels = np.repeat(pixels, 3, axis=2)
    elif layout in ("bgr", "bgra"):
        pixels = pixels[:, :, 2::-1]
    else:
        pixels = pixels[:, :, :3]
    return np.ascontiguousarray(pixels, dtype=np.uint8)


def _next_scan_deadline(
    command_time: float, last_delivered_scan_time: float, scan_rate_hz: float
) -> float:
    """Return the earliest time a distinct policy observation may be delivered."""
    rate = max(1.0, float(scan_rate_hz or 40.0))
    if last_delivered_scan_time <= 0.0:
        return command_time
    return max(command_time, last_delivered_scan_time + 1.0 / rate)


def ros_messages_to_bridge_payload(
    *,
    scan: Any,
    imu: Any,
    throttle: Any,
    steering: Any,
    left_encoder: Any,
    right_encoder: Any,
    forward_speed_mps: float = 0.0,
    camera_image_rgb: Any = None,
) -> Dict[str, Any]:
    """Adapt only permitted sensor/actuator topics to Bridge V1 fields.

    Odometry, IPS, `/tf`, lap counters, collision counters, and reset are
    restricted by the IROS 2026 technical guide. The pose/race values below
    are neutral placeholders; the policy never receives them. Longitudinal
    speed is estimated from permitted rear-wheel encoder angles by Layer 1.
    """
    orientation = imu.orientation
    left = left_encoder.position[0] if left_encoder.position else 0.0
    right = right_encoder.position[0] if right_encoder.position else 0.0
    payload = {
        "V1 Position": "0 0 0",
        "V1 Orientation Quaternion": (
            f"{orientation.x} {orientation.y} {orientation.z} {orientation.w}"
        ),
        "V1 Linear Velocity": f"{float(forward_speed_mps)} 0 0",
        "V1 Angular Velocity": _xyz(imu.angular_velocity),
        "V1 Linear Acceleration": _xyz(imu.linear_acceleration),
        "V1 Encoder Angles": f"{float(left)} {float(right)}",
        "V1 LIDAR Range Array": list(scan.ranges),
        # Keep LaserScan geometry with the ranges. Layer 1 must not guess scan
        # handedness from the payload source: ROS defines beam i at
        # angle_min + i * angle_increment.
        "V1 LIDAR Angle Min": float(scan.angle_min) if hasattr(scan, "angle_min") else None,
        "V1 LIDAR Angle Increment": float(scan.angle_increment) if hasattr(scan, "angle_increment") else None,
        "V1 LIDAR Angle Max": float(scan.angle_max) if hasattr(scan, "angle_max") else None,
        "V1 LIDAR Scan Rate": (
            1.0 / float(scan.scan_time) if float(scan.scan_time) > 0 else 40.0
        ),
        "V1 LIDAR Range Min": float(scan.range_min),
        "V1 LIDAR Range Max": float(scan.range_max),
        "V1 Throttle": float(throttle.data),
        "V1 Steering": float(steering.data),
        "V1 Lap Count": 0,
        "V1 Lap Time": 0.0,
        "V1 Last Lap Time": 0.0,
        "V1 Best Lap Time": 0.0,
        "V1 Collisions": 0,
    }
    if camera_image_rgb is not None:
        payload["V1 Front Camera Image"] = camera_image_rgb
    return payload


def encoder_forward_speed_mps(
    current: tuple[float, float],
    previous: tuple[float, float],
    elapsed_s: float,
    wheel_radius_m: float = 0.059,
) -> float:
    """Estimate forward speed from permitted wheel-angle encoders."""
    if elapsed_s <= 0.0:
        return 0.0
    # The official bridge publishes accumulated wheel angles. Do not wrap the
    # delta: at race speeds a wheel can rotate more than pi between samples.
    left_delta = float(current[0]) - float(previous[0])
    right_delta = float(current[1]) - float(previous[1])
    return 0.5 * (left_delta + right_delta) * float(wheel_radius_m) / float(elapsed_s)


def _ros_message_timestamp(message: Any) -> Optional[float]:
    """Return a positive ROS header time in seconds, when the source provides one."""
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    try:
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


# Retain the original private name for callers/tests created before this
# measurement was shared with the Layer 2 sensor-parity observation profile.
_encoder_forward_speed_mps = encoder_forward_speed_mps


class RacerRos2:
    """ROS 2 implementation of the Layer 1 racer transport contract.

    ``step`` publishes an actuator command and waits for the next fresh
    official LaserScan.  The AutoDRIVE bridge applies that command on its next
    Bridge exchange, matching the existing Layer 1 step semantics.
    """

    def __init__(
        self,
        *,
        vehicle_id: str = "roboracer_1",
        timeout_s: float = 5.0,
        frame_timeout_s: float = 5.0,
        node: Optional[Any] = None,
        include_race_metrics: bool = False,
        allow_training_reset: bool = False,
        require_camera: bool = False,
        safe_neutral_fallback_s: Optional[float] = None,
    ) -> None:
        self.port = 0  # compatibility with Layer 2 diagnostics; ROS uses topics
        self.action_interval_s = None
        self.timeout_s = float(timeout_s)
        self.frame_timeout_s = float(frame_timeout_s)
        self.include_race_metrics = bool(include_race_metrics)
        self.allow_training_reset = bool(allow_training_reset)
        self.require_camera = bool(require_camera)
        if safe_neutral_fallback_s is None:
            configured_fallback = os.getenv("AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S", "0")
            try:
                safe_neutral_fallback_s = float(configured_fallback)
            except (TypeError, ValueError) as exc:
                raise ValueError("AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S must be numeric") from exc
        self.safe_neutral_fallback_s = float(safe_neutral_fallback_s)
        if not math.isfinite(self.safe_neutral_fallback_s) or not 0.0 <= self.safe_neutral_fallback_s < 0.25:
            raise ValueError("safe_neutral_fallback_s must be finite and in [0, 0.25)")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.frame_timeout_s <= 0:
            raise ValueError("frame_timeout_s must be positive")
        self._owned_node = node is None
        self._rclpy = None
        if node is None:
            import rclpy
            from geometry_msgs.msg import Point
            from sensor_msgs.msg import Image, Imu, JointState, LaserScan
            from std_msgs.msg import Bool, Float32, Int32

            self._rclpy = rclpy
            if not rclpy.ok():
                # Application owners (trainer/policy) handle process signals.
                # rclpy's default SIGTERM hook otherwise destroys the context
                # before PPO's cooperative stop callback can save a checkpoint.
                from rclpy.signals import SignalHandlerOptions
                rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
            node = rclpy.create_node("aicar_policy_transport")
            self._msg_types = {
                "Float32": Float32,
                "Int32": Int32,
                "Point": Point,
                "Imu": Imu,
                "JointState": JointState,
                "LaserScan": LaserScan,
                "Image": Image,
                "Bool": Bool,
            }
        else:
            self._msg_types = None
        self.node = node
        self._condition = threading.Condition()
        self._raw: Dict[str, Any] = {}
        self._raw_receipts: Dict[str, float] = {}
        self._spin_error: Optional[BaseException] = None
        self._spin_state = "starting" if node is None else "externally-managed"
        self._topic_errors: Dict[str, str] = {}
        self._cleanup_complete = False
        self._previous_encoder_positions: Optional[tuple[float, float]] = None
        self._previous_encoder_receipt = 0.0
        self._encoder_speed_mps = 0.0
        self._step_counter = 0
        self._telemetry = TelemetrySnapshot(lidar_valid=False)
        self._race_metrics = RaceMetrics()
        self._race_metrics_received: set[str] = set()
        self._race_metric_update_counts: Dict[str, int] = {}
        self._position_update_count = 0
        self._position_received_unix_ns = 0
        self._position_history: list[Dict[str, Any]] = []
        self._last_scan_receipt = 0.0
        self._last_scan_source_time: Optional[float] = None
        self._last_control_source_time: Optional[float] = None
        self._last_control_interval_source = "receipt_clock"
        self._last_scan_interval_source = "receipt_clock"
        self._last_control_scan_receipt = 0.0
        self._last_action_time = 0.0
        self._last_policy_action = {"throttle": 0.0, "steering": 0.0}
        self._command_expired = False
        self.command_expiry_count = 0
        self._command_expiry_diagnostic: Optional[Dict[str, Any]] = None
        self.neutral_fallback_count = 0
        self._neutral_fallback_diagnostic: Optional[Dict[str, Any]] = None
        self._callback_receipts: Dict[str, float] = {}
        self._callback_gaps_s: Dict[str, float] = {}
        self._callback_durations_s: Dict[str, float] = {}
        self._max_callback_durations_s: Dict[str, float] = {}
        self._spin_iteration_started_at = 0.0
        self._spin_iteration_finished_at = 0.0
        self._last_spin_gap_s = 0.0
        self._last_spin_iteration_s = 0.0
        self._max_spin_iteration_s = 0.0
        self._last_control_interval_s = 0.0
        self._last_control_wall_interval_s = 0.0
        self._sim_time = 0.0
        self._last_step_duration_s = 0.0
        self._closed = False
        self._subscriptions = []
        self._thread: Optional[threading.Thread] = None
        self._publishers: Dict[str, Any] = {}
        if self._msg_types is not None:
            self._create_ros_entities(vehicle_id)
            self._thread = threading.Thread(target=self._spin, name="aicar-ros2", daemon=True)
            self._thread.start()

    def _create_ros_entities(self, vehicle_id: str) -> None:
        types = self._msg_types
        prefix = f"/autodrive/{vehicle_id}"
        self._publishers = {
            "throttle": self.node.create_publisher(types["Float32"], f"{prefix}/throttle_command", 10),
            "steering": self.node.create_publisher(types["Float32"], f"{prefix}/steering_command", 10),
        }
        if self.allow_training_reset:
            self._publishers["training_reset"] = self.node.create_publisher(
                types["Bool"], "/autodrive/reset_command", 10
            )
        callbacks = (
            (f"{prefix}/imu", types["Imu"], "imu"),
            (f"{prefix}/throttle", types["Float32"], "throttle"),
            (f"{prefix}/steering", types["Float32"], "steering"),
            (f"{prefix}/left_encoder", types["JointState"], "left_encoder"),
            (f"{prefix}/right_encoder", types["JointState"], "right_encoder"),
            (f"{prefix}/lidar", types["LaserScan"], "scan"),
            (f"{prefix}/front_camera", types["Image"], "camera"),
        )
        metric_callbacks = (
            (f"{prefix}/lap_count", types["Int32"], "lap_count"),
            (f"{prefix}/lap_time", types["Float32"], "lap_time"),
            (f"{prefix}/last_lap_time", types["Float32"], "last_lap_time"),
            (f"{prefix}/best_lap_time", types["Float32"], "best_lap_time"),
            (f"{prefix}/collision_count", types["Int32"], "collision_count"),
            # Restricted IPS is subscribed only by the local evaluation
            # transport. It is never present in the deployed policy path and
            # is used solely to reject runs where wheel encoders spin in place.
            (f"{prefix}/ips", types["Point"], "position"),
        ) if self.include_race_metrics else ()
        for topic, message_type, name in (*callbacks, *metric_callbacks):
            self._subscriptions.append(
                self.node.create_subscription(
                    message_type,
                    topic,
                    lambda message, field=name: self._on_message(field, message),
                    10,
                )
            )

    def _spin(self) -> None:
        self._spin_state = "running"
        try:
            while not self._closed and self._rclpy is not None and self._rclpy.ok():
                iteration_started = time.monotonic()
                self._rclpy.spin_once(self.node, timeout_sec=0.05)
                iteration_finished = time.monotonic()
                with self._condition:
                    self._spin_iteration_started_at = iteration_started
                    self._last_spin_gap_s = (
                        max(0.0, iteration_started - self._spin_iteration_finished_at)
                        if self._spin_iteration_finished_at else 0.0
                    )
                    self._last_spin_iteration_s = max(0.0, iteration_finished - iteration_started)
                    self._max_spin_iteration_s = max(
                        self._max_spin_iteration_s, self._last_spin_iteration_s
                    )
                    self._spin_iteration_finished_at = iteration_finished
                self._expire_stale_command()
            if not self._closed:
                self._fail_transport("ROS context stopped while the racer was active")
        except Exception as exc:
            # rclpy raises ExternalShutdownException when its context is shut
            # down while spin_once is waiting. During an intentional close this
            # is expected; otherwise make it visible to every blocked caller.
            if not self._closed:
                if type(exc).__name__ in {"ExternalShutdownException", "ShutdownException"}:
                    self._fail_transport("ROS context shut down while waiting for simulator messages")
                else:
                    self._fail_transport(
                        f"ROS callback executor failed ({type(exc).__name__}: {exc})", exc
                    )
        finally:
            if self._spin_state != "failed":
                self._spin_state = "closed" if self._closed else "stopped"

    def _expire_stale_command(self) -> None:
        """Apply an opted-in neutral fallback, then retain the hard lease guard.

        Run on the ROS receive thread, independently of policy/preview work.
        The fallback sends only zero throttle and steering. It never replays a
        cached action; the caller must deliver a new action after a fresh frame.
        This cannot protect against suspension of the entire host or ROS process.
        """
        with self._condition:
            if not self._publishers or not self._last_action_time or self._command_expired:
                return
            now = time.monotonic()
            action_age = max(0.0, now - self._last_action_time)
            policy_action = getattr(self, "_last_policy_action", {"throttle": 0.0, "steering": 0.0})
            fallback_s = float(getattr(self, "safe_neutral_fallback_s", 0.0))
            action_is_non_neutral = any(
                abs(float(value)) > 1e-9 for value in policy_action.values()
            )
            if (fallback_s > 0.0 and action_is_non_neutral and action_age <= 0.25
                    and action_age >= fallback_s):
                self._publish_neutral_command()
                self._neutral_fallback_diagnostic = {
                    "step_counter": int(getattr(self, "_step_counter", 0)),
                    "neutralized_at_monotonic_s": now,
                    "action_age_s": action_age,
                    "threshold_s": fallback_s,
                    "reason": "policy_action_waiting_for_fresh_sensor_frame",
                    "previous_policy_action": dict(policy_action),
                    "active_action": {"throttle": 0.0, "steering": 0.0},
                    "feedback": self._actuator_feedback_diagnostic(),
                    "last_scan_age_s": (
                        None if not getattr(self, "_last_scan_receipt", 0.0)
                        else max(0.0, now - self._last_scan_receipt)
                    ),
                    "topic_age_s": {
                        name: max(0.0, now - receipt)
                        for name, receipt in getattr(self, "_raw_receipts", {}).items()
                    },
                }
                self.neutral_fallback_count += 1
                self._last_policy_action = {"throttle": 0.0, "steering": 0.0}
                self._command_expired = True
                return
            if action_age <= 0.25:
                return
            self._command_expiry_diagnostic = {
                "step_counter": int(getattr(self, "_step_counter", 0)),
                "expired_at_monotonic_s": now,
                "last_action_at_monotonic_s": self._last_action_time,
                "action_age_s": action_age,
                "reason": "hard_command_lease_exceeded_before_neutral_fallback",
                "active_action": dict(policy_action),
                "feedback": self._actuator_feedback_diagnostic(),
                "last_scan_age_s": (
                    None if not getattr(self, "_last_scan_receipt", 0.0)
                    else max(0.0, now - self._last_scan_receipt)
                ),
                "topic_age_s": {
                    name: max(0.0, now - receipt)
                    for name, receipt in getattr(self, "_raw_receipts", {}).items()
                },
                "last_callback_gap_s": dict(getattr(self, "_callback_gaps_s", {})),
                "last_callback_duration_s": dict(getattr(self, "_callback_durations_s", {})),
                "max_callback_duration_s": dict(getattr(self, "_max_callback_durations_s", {})),
                "last_spin_gap_s": float(getattr(self, "_last_spin_gap_s", 0.0)),
                "last_spin_iteration_s": float(getattr(self, "_last_spin_iteration_s", 0.0)),
                "max_spin_iteration_s": float(getattr(self, "_max_spin_iteration_s", 0.0)),
                "spin_state": getattr(self, "_spin_state", "unknown"),
            }
            if fallback_s > 0.0:
                self._publish_neutral_command()
            else:
                self._publish_throttle_zero()
            self._last_policy_action = {"throttle": 0.0, "steering": 0.0}
            self._command_expired = True
            self.command_expiry_count += 1

    def _publish_neutral_command(self) -> None:
        """Publish a fresh all-zero command without reusing any policy values."""
        for actuator in ("throttle", "steering"):
            publisher = self._publishers.get(actuator)
            if publisher is None:
                continue
            message = self._msg_types["Float32"]()
            message.data = 0.0
            publisher.publish(message)

    def _publish_throttle_zero(self) -> None:
        publisher = self._publishers.get("throttle")
        if publisher is not None:
            message = self._msg_types["Float32"]()
            message.data = 0.0
            publisher.publish(message)

    def _actuator_feedback_diagnostic(self) -> Dict[str, Any]:
        raw = getattr(self, "_raw", {})
        return {
            name: {
                "value": float(raw[name].data),
                "age_s": max(0.0, time.monotonic() - self._raw_receipts[name]),
            }
            for name in ("throttle", "steering")
            if name in raw and hasattr(raw[name], "data") and name in self._raw_receipts
        }

    def _fail_transport(self, message: str, cause: Optional[BaseException] = None) -> None:
        with self._condition:
            self._spin_error = cause or OfficialRosTransportError(message)
            self._spin_state = "failed"
            self._condition.notify_all()

    def _transport_diagnostic(self, now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        required = ["scan", "imu", "throttle", "steering", "left_encoder", "right_encoder"]
        if self.require_camera:
            required.append("camera")
        camera_max_age = min(self.frame_timeout_s, 0.5)
        ages = {
            name: (None if name not in self._raw_receipts else max(0.0, now - self._raw_receipts[name]))
            for name in required
        }
        absent = [name for name, age in ages.items() if age is None]
        stale = [
            f"{name}={age:.2f}s"
            for name, age in ages.items()
            if age is not None and age > (camera_max_age if name == "camera" else self.frame_timeout_s)
        ]
        details = []
        if absent:
            details.append("never received: " + ", ".join(absent))
        if stale:
            details.append("stale: " + ", ".join(stale))
        if self._topic_errors:
            details.append("invalid messages: " + ", ".join(
                f"{topic} ({reason})" for topic, reason in sorted(self._topic_errors.items())
            ))
        if not details:
            details.append("all required topic streams have recent messages")
        return f"spin={self._spin_state}; " + "; ".join(details)

    def transport_status(self) -> Dict[str, Any]:
        """Return monotonic receive-age diagnostics for each required topic."""
        with self._condition:
            now = time.monotonic()
            required = ["scan", "imu", "throttle", "steering", "left_encoder", "right_encoder"]
            if self.require_camera:
                required.append("camera")
            camera_max_age = min(self.frame_timeout_s, 0.5)
            topic_age_s = {
                name: (None if name not in self._raw_receipts else max(0.0, now - self._raw_receipts[name]))
                for name in required
            }
            fresh = all(
                age is not None
                and age <= (camera_max_age if name == "camera" else self.frame_timeout_s)
                for name, age in topic_age_s.items()
            )
            return {
                "spin_state": self._spin_state,
                "spin_error": None if self._spin_error is None else str(self._spin_error),
                "topic_errors": dict(self._topic_errors),
                "step_counter": self._step_counter,
                "last_scan_age_s": (
                    None if not self._last_scan_receipt else max(0.0, now - self._last_scan_receipt)
                ),
                "topic_age_s": topic_age_s,
                "topic_fresh": fresh,
                "required_topics": required,
                "command_expiry_diagnostic": getattr(
                    self, "_command_expiry_diagnostic", None
                ),
                "safe_neutral_fallback_s": float(getattr(self, "safe_neutral_fallback_s", 0.0)),
                "neutral_fallback_count": int(getattr(self, "neutral_fallback_count", 0)),
                "neutral_fallback_diagnostic": getattr(
                    self, "_neutral_fallback_diagnostic", None
                ),
            }

    def request_pause_neutral(self) -> Dict[str, Any]:
        """Publish a bounded neutral actuator command at a learner pause point.

        This confirms only that the ROS publisher accepted the calls; the
        transport does not expose a physical-actuator acknowledgement.
        """
        with self._condition:
            self._check_wait_state()
            throttle_publisher = self._publishers.get("throttle")
            steering_publisher = self._publishers.get("steering")
            if throttle_publisher is None or steering_publisher is None or self._msg_types is None:
                raise OfficialRosTransportError("official ROS actuator publishers are unavailable")
            throttle = self._msg_types["Float32"]()
            throttle.data = 0.0
            steering = self._msg_types["Float32"]()
            steering.data = 0.0
            throttle_publisher.publish(throttle)
            steering_publisher.publish(steering)
            self._last_action_time = time.monotonic()
            self._command_expired = True
            self.command_expiry_count += 1
            status = self.transport_status()
            return {
                "published": True,
                "command": {"throttle": 0.0, "steering": 0.0},
                "transport_acknowledged": False,
                **status,
            }

    def _check_wait_state(self) -> None:
        if self._closed:
            raise OfficialRosTransportError("official ROS racer was closed while waiting for a sensor frame")
        if self._spin_error is not None:
            raise OfficialRosTransportError(
                f"official ROS receive thread stopped: {self._spin_error}; {self._transport_diagnostic()}"
            ) from self._spin_error

    def _on_message(self, field: str, message: Any) -> None:
        now = time.monotonic()
        with self._condition:
            callback_receipts = getattr(self, "_callback_receipts", None)
            if callback_receipts is None:
                callback_receipts = self._callback_receipts = {}
            previous = callback_receipts.get(field)
            if previous is not None:
                callback_gaps = getattr(self, "_callback_gaps_s", None)
                if callback_gaps is None:
                    callback_gaps = self._callback_gaps_s = {}
                callback_gaps[field] = max(0.0, now - previous)
            callback_receipts[field] = now
        try:
            self._on_message_impl(field, message, received_at=now)
        finally:
            finished = time.monotonic()
            duration = max(0.0, finished - now)
            with self._condition:
                callback_durations = getattr(self, "_callback_durations_s", None)
                if callback_durations is None:
                    callback_durations = self._callback_durations_s = {}
                max_callback_durations = getattr(self, "_max_callback_durations_s", None)
                if max_callback_durations is None:
                    max_callback_durations = self._max_callback_durations_s = {}
                callback_durations[field] = duration
                max_callback_durations[field] = max(
                    duration, max_callback_durations.get(field, 0.0)
                )

    def _on_message_impl(
        self, field: str, message: Any, *, received_at: Optional[float] = None
    ) -> None:
        now = time.monotonic() if received_at is None else received_at
        source_time = _ros_message_timestamp(message) if field == "scan" else None
        with self._condition:
            if field == "camera":
                try:
                    image = ros_image_to_rgb(message)
                    self._raw["camera"] = image
                    self._raw["camera_image_rgb"] = image
                    self._raw_receipts["camera"] = now
                    self._topic_errors.pop("camera", None)
                except (TypeError, ValueError) as exc:
                    self._topic_errors["camera"] = str(exc)
                self._condition.notify_all()
                return
            self._raw[field] = message
            self._raw_receipts[field] = now
            if field == "position":
                self._race_metrics = replace(
                    self._race_metrics,
                    position=(float(message.x), float(message.y), float(message.z)),
                    position_received_at=now,
                )
                self._race_metrics_received.add(field)
                self._position_update_count += 1
                self._position_received_unix_ns = time.time_ns()
                self._position_history.append({
                    "callback_count": self._position_update_count,
                    "received_monotonic_s": now,
                    "received_unix_ns": self._position_received_unix_ns,
                    "position_xy": [float(message.x), float(message.y)],
                })
                self._position_history = self._position_history[-256:]
                self._condition.notify_all()
                return
            if field in {
                "lap_count", "lap_time", "last_lap_time", "best_lap_time", "collision_count"
            }:
                value = int(message.data) if field in {"lap_count", "collision_count"} else float(message.data)
                self._race_metrics_received.add(field)
                self._race_metric_update_counts[field] = (
                    self._race_metric_update_counts.get(field, 0) + 1
                )
                update = {field: value}
                if field == "lap_count":
                    update["lap_count_update_count"] = self._race_metric_update_counts[field]
                    update["lap_count_change_count"] = (
                        self._race_metrics.lap_count_change_count
                        + int(int(value) != int(self._race_metrics.lap_count))
                    )
                    update["last_lap_time_update_count_at_lap_count"] = (
                        self._race_metric_update_counts.get("last_lap_time", 0)
                    )
                elif field == "last_lap_time":
                    update["last_lap_time_update_count"] = self._race_metric_update_counts[field]
                self._race_metrics = replace(self._race_metrics, **update)
                self._condition.notify_all()
                return
            if field == "scan":
                required = (
                    "imu", "throttle", "steering", "left_encoder", "right_encoder",
                )
                if self.require_camera:
                    required = (*required, "camera")
                camera_max_age = min(self.frame_timeout_s, 0.5)
                scan_ranges = getattr(self._raw.get("scan"), "ranges", None)
                if not scan_ranges:
                    self._topic_errors["scan"] = "LaserScan.ranges is empty"
                    self._condition.notify_all()
                    return
                fresh_topics = all(
                    key in self._raw and now - self._raw_receipts[key] <= self.frame_timeout_s
                    and (key != "camera" or now - self._raw_receipts[key] <= camera_max_age)
                    for key in required
                )
                if fresh_topics:
                    left_values = self._raw["left_encoder"].position
                    right_values = self._raw["right_encoder"].position
                    if left_values and right_values:
                        self._topic_errors.pop("left_encoder", None)
                        self._topic_errors.pop("right_encoder", None)
                        positions = (float(left_values[0]), float(right_values[0]))
                        encoder_receipt = min(
                            self._raw_receipts["left_encoder"],
                            self._raw_receipts["right_encoder"],
                        )
                        if self._previous_encoder_positions is None:
                            self._previous_encoder_positions = positions
                            self._previous_encoder_receipt = encoder_receipt
                        elif encoder_receipt > self._previous_encoder_receipt:
                            elapsed = encoder_receipt - self._previous_encoder_receipt
                            self._encoder_speed_mps = encoder_forward_speed_mps(
                                positions,
                                self._previous_encoder_positions,
                                elapsed,
                            )
                            self._previous_encoder_positions = positions
                            self._previous_encoder_receipt = encoder_receipt
                        elif now - self._previous_encoder_receipt > 0.1:
                            self._encoder_speed_mps = 0.0
                    else:
                        # Encoders are an optional speed estimate, not a
                        # prerequisite for a valid scan/IMU observation.
                        # Preserve the historical fallback: report zero
                        # encoder-derived speed and keep advancing on LiDAR.
                        self._encoder_speed_mps = 0.0
                        if not left_values:
                            self._topic_errors["left_encoder"] = "JointState.position is empty; speed fallback is zero"
                        if not right_values:
                            self._topic_errors["right_encoder"] = "JointState.position is empty; speed fallback is zero"
                    payload = ros_messages_to_bridge_payload(
                        scan=self._raw["scan"],
                        imu=self._raw["imu"], throttle=self._raw["throttle"],
                        steering=self._raw["steering"], left_encoder=self._raw["left_encoder"],
                        right_encoder=self._raw["right_encoder"],
                        forward_speed_mps=self._encoder_speed_mps,
                        camera_image_rgb=self._raw.get("camera_image_rgb"),
                    )
                    next_step = self._step_counter + 1
                    try:
                        snapshot = TelemetrySnapshot.from_raw_dict(payload, step_id=next_step)
                    except Exception as exc:
                        self._topic_errors["scan/telemetry"] = (
                            f"could not build snapshot ({type(exc).__name__}: {exc})"
                        )
                    else:
                        self._topic_errors.pop("scan/telemetry", None)
                        self._step_counter = next_step
                        self._telemetry = snapshot
                        if self._last_scan_receipt:
                            source_delta = (
                                source_time - self._last_scan_source_time
                                if source_time is not None and self._last_scan_source_time is not None
                                else 0.0
                            )
                            if source_delta >= 0 and source_time is not None and self._last_scan_source_time is not None:
                                self._last_step_duration_s = source_delta
                                self._last_scan_interval_source = "ros_sensor_stamp"
                            else:
                                self._last_step_duration_s = max(0.0, now - self._last_scan_receipt)
                                self._last_scan_interval_source = "receipt_clock"
                            self._sim_time += self._last_step_duration_s
                        self._last_scan_receipt = now
                        self._last_scan_source_time = source_time
                self._condition.notify_all()

    def _wait_for_scan_after(
        self, previous_step: int, *, timeout_s: Optional[float] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TelemetrySnapshot:
        wait_s = self.frame_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + wait_s
        with self._condition:
            while self._step_counter <= previous_step and not self._closed:
                self._raise_if_training_reset_cancelled(cancel_check)
                self._check_wait_state()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OfficialRosTransportError(
                        f"Timed out after {wait_s:.1f}s waiting for a fresh official ROS "
                        f"sensor frame; {self._transport_diagnostic()}"
                    )
                self._condition.wait(min(remaining, 0.1))
            self._check_wait_state()
            self._raise_if_training_reset_cancelled(cancel_check)
            return self._telemetry

    @staticmethod
    def _raise_if_training_reset_cancelled(
        cancel_check: Optional[Callable[[], bool]],
    ) -> None:
        if cancel_check is not None and cancel_check():
            raise OfficialTrainingResetCancelled(
                "official training simulator reset was cancelled by the run stop request"
            )

    def _wait_for_control_frame(
        self, previous_step: int, *, command_time: float, earliest_time: float
    ) -> TelemetrySnapshot:
        """Wait for a scan received after this command at the advertised scan rate.

        Some Devkit/simulator combinations publish duplicate LaserScan messages
        much faster than ``scan_time``.  Those are useful as transport updates,
        but do not represent new sensor measurements.  Holding each action for
        at least one advertised scan period avoids training/inference at the DDS
        callback rate with repeated observations.
        """
        deadline = time.monotonic() + self.frame_timeout_s
        with self._condition:
            while not self._closed:
                self._check_wait_state()
                fresh = (
                    self._step_counter > previous_step
                    and self._last_scan_receipt >= command_time
                    and self._last_scan_receipt >= earliest_time
                )
                if fresh:
                    return self._telemetry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OfficialRosTransportError(
                        f"Timed out after {self.frame_timeout_s:.1f}s waiting for a post-command official "
                        f"LiDAR frame at its declared cadence; {self._transport_diagnostic()}"
                    )
                self._condition.wait(min(remaining, 0.1))
            self._check_wait_state()

    def wait_until_ready(
        self, *, cancel_check: Optional[Callable[[], bool]] = None
    ) -> TelemetrySnapshot:
        # Initial discovery can take longer than an ordinary policy step.
        return self._wait_for_scan_after(0, timeout_s=self.timeout_s, cancel_check=cancel_check)

    def wait_for_race_metrics(
        self, *, cancel_check: Optional[Callable[[], bool]] = None
    ) -> RaceMetrics:
        if not self.include_race_metrics:
            raise RuntimeError("Race metrics are disabled for the policy transport")
        required = {
            "lap_count", "lap_time", "last_lap_time", "best_lap_time",
            "collision_count", "position",
        }
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            while not required.issubset(self._race_metrics_received):
                self._raise_if_training_reset_cancelled(cancel_check)
                self._check_wait_state()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OfficialRosTransportError(
                        "Timed out waiting for official evaluator topics "
                        "(race counters and IPS motion check); " + self._transport_diagnostic()
                    )
                self._condition.wait(min(remaining, 0.1))
            self._check_wait_state()
            self._raise_if_training_reset_cancelled(cancel_check)
            return self._race_metrics

    def wait_for_lap_time_update(
        self,
        after_update_count: int,
        *,
        timeout_s: Optional[float] = None,
    ) -> Optional[RaceMetrics]:
        """Wait for a last-lap-time callback newer than a lap-count event."""
        timeout = self.frame_timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._race_metrics.last_lap_time_update_count <= int(after_update_count):
                self._check_wait_state()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.05))
            return self._race_metrics

    def _wait_for_fresh_reset_metrics(
        self,
        previous_update_counts: Dict[str, int],
        *,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> RaceMetrics:
        """Wait for a stable lap counter and fresh race fields after reset.

        A training reset can move the official lap counter back to zero. The
        first cached metric snapshot may still contain the pre-reset count, so
        the environment must not anchor its episode baseline to that snapshot.
        """
        required = {
            "lap_count", "lap_time", "last_lap_time", "best_lap_time",
            "collision_count",
        }
        deadline = time.monotonic() + self.frame_timeout_s
        stable_lap_updates = 0
        last_lap_update_count = int(previous_update_counts.get("lap_count", 0))
        last_lap_value: Optional[int] = None
        with self._condition:
            while True:
                self._raise_if_training_reset_cancelled(cancel_check)
                self._check_wait_state()
                counts = getattr(self, "_race_metric_update_counts", {})
                all_fresh = all(
                    int(counts.get(name, 0)) > int(previous_update_counts.get(name, 0))
                    for name in required
                )
                lap_update_count = int(counts.get("lap_count", 0))
                if lap_update_count > last_lap_update_count:
                    lap_value = int(self._race_metrics.lap_count)
                    stable_lap_updates = (
                        stable_lap_updates + 1 if last_lap_value == lap_value else 1
                    )
                    last_lap_value = lap_value
                    last_lap_update_count = lap_update_count
                if all_fresh and stable_lap_updates >= 2:
                    return self._race_metrics
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OfficialRosTransportError(
                        "Timed out waiting for fresh, stable official race metrics after training reset; "
                        f"fresh_fields={sorted(name for name in required if int(counts.get(name, 0)) > int(previous_update_counts.get(name, 0)))}, "
                        f"stable_lap_updates={stable_lap_updates}, lap_count={self._race_metrics.lap_count}; "
                        + self._transport_diagnostic()
                    )
                self._condition.wait(min(remaining, 0.1))

    def reset_simulation_for_training(
        self,
        *,
        expected_position: Optional[tuple[float, float]] = None,
        position_tolerance_m: float = 0.75,
        reset_attempts: int = 3,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TelemetrySnapshot:
        """Reset the official simulator through its documented training API.

        This is intentionally unavailable to the deployed policy transport.
        The competition guide permits the restricted reset topic for training;
        policy mode never creates its publisher.
        """
        if not self.allow_training_reset:
            raise PermissionError("official simulator reset is training-only")
        publisher = self._publishers.get("training_reset")
        if publisher is None or self._msg_types is None:
            raise RuntimeError("training reset publisher is unavailable")

        # Initial startup has its own budget; do not use the short frame
        # timeout while Unity and the Devkit are still connecting.
        if reset_attempts < 1:
            raise ValueError("reset_attempts must be positive")
        if position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be positive")
        snap = (
            self.wait_until_ready()
            if cancel_check is None
            else self.wait_until_ready(cancel_check=cancel_check)
        )
        reset_metric_update_counts: Dict[str, int] = {}
        diagnostic_path = os.environ.get("AICAR_RESET_DIAGNOSTICS_PATH")
        environment_index = os.environ.get("AICAR_ENV_INDEX", "0")
        target = (
            None if expected_position is None
            else (float(expected_position[0]), float(expected_position[1]))
        )
        attempts_diagnostics: list[Dict[str, Any]] = []
        final_validated_snapshot: Optional[Dict[str, Any]] = None

        def wait_for_stable_pose(after_count: int, timeout_s: float) -> tuple[bool, list[dict[str, Any]]]:
            """Require three distinct recent IPS callbacks clustered inside the existing guard."""
            deadline = time.monotonic() + timeout_s
            samples: list[dict[str, Any]] = []
            last_count = after_count
            with self._condition:
                while True:
                    self._raise_if_training_reset_cancelled(cancel_check)
                    self._check_wait_state()
                    history = getattr(self, "_position_history", [])
                    unseen = [sample for sample in history if int(sample["callback_count"]) > last_count]
                    for received in unseen:
                        position_xy = received["position_xy"]
                        distance = math.dist(position_xy, target) if target is not None else 0.0
                        samples.append({**received, "distance_m": distance})
                        samples = samples[-3:]
                        last_count = int(received["callback_count"])
                    if target is None or (
                        len(samples) == 3
                        and all(sample["distance_m"] <= position_tolerance_m for sample in samples)
                        and max(
                            math.dist(a["position_xy"], b["position_xy"])
                            for index, a in enumerate(samples)
                            for b in samples[index + 1:]
                        ) <= 0.10
                    ):
                        return True, samples
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False, samples
                    self._condition.wait(min(remaining, 0.1))

        for attempt in range(reset_attempts):
            self._raise_if_training_reset_cancelled(cancel_check)
            with self._condition:
                previous_step = self._step_counter
                previous_position_update = self._position_update_count
                reset_metric_update_counts = dict(
                    getattr(self, "_race_metric_update_counts", {})
                )
                # A collision or stalled episode can leave the last learner
                # command active when this restricted reset is requested.
                # Neutralize it under the same lock used by step() before
                # teleporting, so stale throttle cannot carry the car away
                # from the validated spawn while reset telemetry settles.
                self._publish_neutral_command()
                self._last_policy_action = {"throttle": 0.0, "steering": 0.0}
                self._last_action_time = 0.0
                self._command_expired = True
                reset_msg = self._msg_types["Bool"]()
                reset_msg.data = True
                reset_requested_at_unix_ns = time.time_ns()
                reset_publish_unix_ns = []
                reset_publish_monotonic = []
            # Devkit stores only the latest Bool and includes it in the next
            # Bridge packet. Re-publish True across several packet intervals
            # so a single delayed/lost ROS delivery cannot consume the reset.
            pulse_deadline = time.monotonic() + min(0.30, max(0.12, self.frame_timeout_s))
            try:
                publisher.publish(reset_msg)
                reset_publish_unix_ns.append(time.time_ns())
                reset_publish_monotonic.append(time.monotonic())
                while time.monotonic() < pulse_deadline:
                    self._raise_if_training_reset_cancelled(cancel_check)
                    with self._condition:
                        self._check_wait_state()
                        remaining = pulse_deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(min(0.05, remaining))
                        if time.monotonic() < pulse_deadline:
                            publisher.publish(reset_msg)
                            reset_publish_unix_ns.append(time.time_ns())
                            reset_publish_monotonic.append(time.monotonic())
            finally:
                reset_msg.data = False
                publisher.publish(reset_msg)
                reset_deasserted_unix_ns = time.time_ns()
            snap = (
                self._wait_for_scan_after(previous_step)
                if cancel_check is None
                else self._wait_for_scan_after(previous_step, cancel_check=cancel_check)
            )

            if target is None or not self.include_race_metrics:
                break
            attempt_diagnostic: Dict[str, Any] = {
                "attempt": attempt + 1,
                "command_delivery": "publisher calls recorded; Devkit exposes no delivery acknowledgment",
                "reset_true_publish_count": len(reset_publish_unix_ns),
                "reset_requested_unix_ns": reset_requested_at_unix_ns,
                "reset_published_unix_ns": reset_publish_unix_ns,
                "reset_published_monotonic_s": reset_publish_monotonic,
                "reset_deasserted_unix_ns": reset_deasserted_unix_ns,
                "position_callback_count_before": previous_position_update,
                "configured_ips_spawn_xy": list(target),
                "guard_limit_m": position_tolerance_m,
                "position_sequence": [],
                "metrics_result": "pending",
                "final_pose_result": "pending",
            }
            reached_spawn, pose_sequence = wait_for_stable_pose(
                previous_position_update, self.frame_timeout_s
            )
            attempt_diagnostic["position_sequence"] = pose_sequence
            if reached_spawn:
                try:
                    self._wait_for_fresh_reset_metrics(
                        reset_metric_update_counts,
                        cancel_check=cancel_check,
                    )
                    attempt_diagnostic["metrics_result"] = "fresh_stable_metrics"
                    with self._condition:
                        post_metrics_position_count = self._position_update_count
                    final_in_range, final_sequence = wait_for_stable_pose(
                        post_metrics_position_count, self.frame_timeout_s
                    )
                    attempt_diagnostic["final_pose_sequence"] = final_sequence
                    if final_in_range:
                        with self._condition:
                            final_position = self._race_metrics.position
                            final_count = int(self._position_update_count)
                            final_received_unix_ns = int(
                                getattr(self, "_position_received_unix_ns", 0)
                            )
                            final_distance = (
                                math.dist(final_position[:2], target)
                                if final_position is not None else math.inf
                            )
                            newest_validated = final_sequence[-1]
                            still_stable = (
                                final_position is not None
                                and final_distance <= position_tolerance_m
                                and math.dist(
                                    final_position[:2], newest_validated["position_xy"]
                                ) <= 0.10
                            )
                            if still_stable:
                                final_validated_snapshot = {
                                    "position_xy": list(final_position[:2]),
                                    "callback_count": final_count,
                                    "received_unix_ns": final_received_unix_ns,
                                    "distance_m": final_distance,
                                }
                        attempt_diagnostic["final_pose_result"] = (
                            "stable_in_range" if still_stable else "drifted_after_stable_sequence"
                        )
                    else:
                        attempt_diagnostic["final_pose_result"] = "timeout_or_unstable"
                    if final_validated_snapshot is not None:
                        attempts_diagnostics.append(attempt_diagnostic)
                        break
                except OfficialRosTransportError as exc:
                    if isinstance(exc, OfficialTrainingResetCancelled):
                        raise
                    attempt_diagnostic["metrics_result"] = f"failed: {type(exc).__name__}: {exc}"
                    attempt_diagnostic["final_pose_result"] = "skipped_metrics_failed"
            else:
                attempt_diagnostic["metrics_result"] = "not_waited_pose_not_stable"
                attempt_diagnostic["final_pose_result"] = "initial_pose_timeout_or_unstable"
            attempts_diagnostics.append(attempt_diagnostic)
            if attempt + 1 >= reset_attempts:
                with self._condition:
                    actual = self._race_metrics.position
                    callback_count = int(self._position_update_count)
                result = {
                    "result": "failed",
                    "environment_index": environment_index,
                    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
                    "simulator_port": os.environ.get("AICAR_SIMULATOR_PORT") or os.environ.get("AUTODRIVE_PORT") or os.environ.get("PORT"),
                    "configured_ips_spawn_xy": list(target),
                    "guard_limit_m": position_tolerance_m,
                    "attempts": attempts_diagnostics,
                    "final_position_xy": None if actual is None else list(actual[:2]),
                    "final_position_callback_count": callback_count,
                }
                if diagnostic_path:
                    try:
                        with open(diagnostic_path, "a", encoding="utf-8") as stream:
                            stream.write(json.dumps(result, sort_keys=True) + "\n")
                    except OSError as exc:
                        print(f"WARNING: Could not append reset diagnostic {diagnostic_path}: {exc}")
                raise OfficialRosTransportError(
                    "Official training reset failed stable spawn/metrics validation "
                    f"after {reset_attempts} attempts; expected={target}, actual={actual}; "
                    + self._transport_diagnostic()
                )
        else:
            attempts_diagnostics.append({"attempt": attempt + 1, "result": "not_pose_validated"})
        if target is None and self.include_race_metrics:
            self._wait_for_fresh_reset_metrics(
                reset_metric_update_counts,
                cancel_check=cancel_check,
            )
        if diagnostic_path and target is not None and self.include_race_metrics:
            assert final_validated_snapshot is not None
            result = {
                "result": "success",
                "environment_index": environment_index,
                "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
                "simulator_port": os.environ.get("AICAR_SIMULATOR_PORT") or os.environ.get("AUTODRIVE_PORT") or os.environ.get("PORT"),
                "configured_ips_spawn_xy": list(target),
                "guard_limit_m": position_tolerance_m,
                "attempts": attempts_diagnostics,
                "final_position_xy": final_validated_snapshot["position_xy"],
                "final_position_callback_count": final_validated_snapshot["callback_count"],
                "final_position_received_unix_ns": final_validated_snapshot["received_unix_ns"],
                "final_position_distance_m": final_validated_snapshot["distance_m"],
            }
            try:
                with open(diagnostic_path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, sort_keys=True) + "\n")
            except OSError as exc:
                print(f"WARNING: Could not append reset diagnostic {diagnostic_path}: {exc}")
        with self._condition:
            self._last_action_time = 0.0
            self._last_control_scan_receipt = 0.0
            self._last_control_interval_s = 0.0
            self._last_step_duration_s = 0.0
            self._last_control_source_time = self._last_scan_source_time
        return snap

    @property
    def race_metrics(self) -> RaceMetrics:
        if not self.include_race_metrics:
            raise RuntimeError("Race metrics are disabled for the policy transport")
        with self._condition:
            return self._race_metrics

    def step(self, throttle: float, steering: float) -> TelemetrySnapshot:
        with self._condition:
            self._check_wait_state()
            previous_step = self._step_counter
            command_time = time.monotonic()
            prior_action_time = self._last_action_time
            prior_control_source_time = self._last_control_source_time
            earliest_time = _next_scan_deadline(
                command_time,
                self._last_control_scan_receipt,
                self._telemetry.lidar_scan_rate,
            )
            self._last_action_time = command_time
            self._last_policy_action = {
                "throttle": float(throttle), "steering": float(steering)
            }
            self._command_expired = False
            # Keep the publish in the same critical section as the closed
            # check. kill() takes this lock before its terminal neutral, so a
            # policy action cannot slip onto the DDS topics after shutdown has
            # released the actuators.
            if self._publishers:
                throttle_msg = self._msg_types["Float32"]()
                throttle_msg.data = float(throttle)
                steering_msg = self._msg_types["Float32"]()
                steering_msg.data = float(steering)
                self._publishers["throttle"].publish(throttle_msg)
                self._publishers["steering"].publish(steering_msg)
        snap = self._wait_for_control_frame(
            previous_step, command_time=command_time, earliest_time=earliest_time
        )
        with self._condition:
            wall_interval = command_time - prior_action_time if prior_action_time else 0.0
            self._last_control_wall_interval_s = max(0.0, wall_interval)
            source_interval = (
                self._last_scan_source_time - prior_control_source_time
                if self._last_scan_source_time is not None and prior_control_source_time is not None
                else 0.0
            )
            if source_interval >= 0 and self._last_scan_source_time is not None and prior_control_source_time is not None:
                self._last_control_interval_s = source_interval
                self._last_control_interval_source = "ros_sensor_stamp"
            else:
                self._last_control_interval_s = max(0.0, wall_interval)
                self._last_control_interval_source = "receipt_clock"
            self._last_control_source_time = self._last_scan_source_time
            self._last_control_scan_receipt = self._last_scan_receipt
        return snap

    def simulation_time(self) -> float:
        return self._sim_time

    @property
    def telemetry(self) -> TelemetrySnapshot:
        return self._telemetry

    @property
    def step_counter(self) -> int:
        return self._step_counter

    @property
    def is_connected(self) -> bool:
        return self._step_counter > 0

    @property
    def last_step_duration_s(self) -> float:
        return self._last_step_duration_s

    @property
    def last_control_interval_s(self) -> float:
        return self._last_control_interval_s

    @property
    def control_interval_source(self) -> str:
        return self._last_control_interval_source

    @property
    def last_control_wall_interval_s(self) -> float:
        return self._last_control_wall_interval_s

    @property
    def scan_interval_source(self) -> str:
        return self._last_scan_interval_source

    def set_simulation_paused(self, paused: bool) -> None:
        # The official ROS 2 Devkit exposes no pause topic.
        if paused:
            raise NotImplementedError("The official competition Devkit has no simulation-pause topic")

    def resume_simulation(self) -> None:
        return None

    def kill(self) -> None:
        if getattr(self, "_cleanup_complete", False):
            return
        condition = getattr(self, "_condition", None)
        if condition is None:
            self._closed = True
            self._cleanup_complete = True
            return
        with condition:
            if getattr(self, "_cleanup_complete", False):
                return
            # Mark closed before publishing neutral, while holding the same
            # lock step() uses through its command publish. A concurrent caller
            # can therefore neither start a new action nor resume with stale
            # throttle after shutdown begins.
            self._closed = True
            self._last_policy_action = {"throttle": 0.0, "steering": 0.0}
            self._last_action_time = 0.0
            self._command_expired = True
            publishers = getattr(self, "_publishers", {})
            msg_types = getattr(self, "_msg_types", None)
            if (publishers.get("throttle") is not None
                    and publishers.get("steering") is not None
                    and isinstance(msg_types, dict) and msg_types.get("Float32") is not None):
                publish_errors = {}
                for actuator in ("throttle", "steering"):
                    try:
                        message = msg_types["Float32"]()
                        message.data = 0.0
                        publishers[actuator].publish(message)
                    except Exception as exc:
                        publish_errors[actuator] = f"{type(exc).__name__}: {exc}"
                self._shutdown_neutral_publish_errors = publish_errors
            self._condition.notify_all()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            # spin_once is bounded to 50 ms. Join it before destroying the node
            # or shutting down the context, avoiding concurrent executor teardown.
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise OfficialRosTransportError(
                    "ROS receive thread did not exit after its bounded spin wait; "
                    "node/context were left intact to avoid a teardown race"
                )
        if (getattr(self, "_owned_node", False)
                and getattr(self, "node", None) is not None):
            try:
                self.node.destroy_node()
            finally:
                rclpy = getattr(self, "_rclpy", None)
                if rclpy is not None and rclpy.ok():
                    self._rclpy.shutdown()
        self._cleanup_complete = True
