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
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from .telemetry import TelemetrySnapshot


class OfficialRosTransportError(RuntimeError):
    """A fatal or timed-out official ROS transport operation."""


@dataclass(frozen=True)
class RaceMetrics:
    """Restricted simulator data for evaluation-only instrumentation."""

    lap_count: int = 0
    lap_time: float = 0.0
    last_lap_time: float = 0.0
    best_lap_time: float = 0.0
    collision_count: int = 0
    position: Optional[tuple[float, float, float]] = None


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
    ) -> None:
        self.port = 0  # compatibility with Layer 2 diagnostics; ROS uses topics
        self.action_interval_s = None
        self.timeout_s = float(timeout_s)
        self.frame_timeout_s = float(frame_timeout_s)
        self.include_race_metrics = bool(include_race_metrics)
        self.allow_training_reset = bool(allow_training_reset)
        self.require_camera = bool(require_camera)
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
        self._position_update_count = 0
        self._last_scan_receipt = 0.0
        self._last_scan_source_time: Optional[float] = None
        self._last_control_source_time: Optional[float] = None
        self._last_control_interval_source = "receipt_clock"
        self._last_scan_interval_source = "receipt_clock"
        self._last_control_scan_receipt = 0.0
        self._last_action_time = 0.0
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
                self._rclpy.spin_once(self.node, timeout_sec=0.05)
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
            return {
                "spin_state": self._spin_state,
                "spin_error": None if self._spin_error is None else str(self._spin_error),
                "topic_errors": dict(self._topic_errors),
                "step_counter": self._step_counter,
                "last_scan_age_s": (
                    None if not self._last_scan_receipt else max(0.0, now - self._last_scan_receipt)
                ),
                "topic_age_s": {
                    name: (None if name not in self._raw_receipts else max(0.0, now - self._raw_receipts[name]))
                    for name in required
                },
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
                )
                self._race_metrics_received.add(field)
                self._position_update_count += 1
                self._condition.notify_all()
                return
            if field in {
                "lap_count", "lap_time", "last_lap_time", "best_lap_time", "collision_count"
            }:
                value = int(message.data) if field in {"lap_count", "collision_count"} else float(message.data)
                self._race_metrics = replace(self._race_metrics, **{field: value})
                self._race_metrics_received.add(field)
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
        self, previous_step: int, *, timeout_s: Optional[float] = None
    ) -> TelemetrySnapshot:
        wait_s = self.frame_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + wait_s
        with self._condition:
            while self._step_counter <= previous_step and not self._closed:
                self._check_wait_state()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OfficialRosTransportError(
                        f"Timed out after {wait_s:.1f}s waiting for a fresh official ROS "
                        f"sensor frame; {self._transport_diagnostic()}"
                    )
                self._condition.wait(min(remaining, 0.1))
            self._check_wait_state()
            return self._telemetry

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

    def wait_until_ready(self) -> TelemetrySnapshot:
        # Initial discovery can take longer than an ordinary policy step.
        return self._wait_for_scan_after(0, timeout_s=self.timeout_s)

    def wait_for_race_metrics(self) -> RaceMetrics:
        if not self.include_race_metrics:
            raise RuntimeError("Race metrics are disabled for the policy transport")
        required = {
            "lap_count", "lap_time", "last_lap_time", "best_lap_time",
            "collision_count", "position",
        }
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            while not required.issubset(self._race_metrics_received):
                self._check_wait_state()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OfficialRosTransportError(
                        "Timed out waiting for official evaluator topics "
                        "(race counters and IPS motion check); " + self._transport_diagnostic()
                    )
                self._condition.wait(min(remaining, 0.1))
            self._check_wait_state()
            return self._race_metrics

    def reset_simulation_for_training(
        self,
        *,
        expected_position: Optional[tuple[float, float]] = None,
        position_tolerance_m: float = 0.75,
        reset_attempts: int = 3,
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
        snap = self.wait_until_ready()
        for attempt in range(reset_attempts):
            with self._condition:
                previous_step = self._step_counter
                previous_position_update = self._position_update_count
            reset_msg = self._msg_types["Bool"]()
            reset_msg.data = True
            publisher.publish(reset_msg)
            # The official simulator samples this command during its update
            # loop. Hold it across several 40 Hz ticks, then deassert it.
            deadline = time.monotonic() + 0.1
            while time.monotonic() < deadline:
                with self._condition:
                    self._check_wait_state()
                    self._condition.wait(min(0.02, deadline - time.monotonic()))
            reset_msg.data = False
            publisher.publish(reset_msg)
            snap = self._wait_for_scan_after(previous_step)

            if expected_position is None or not self.include_race_metrics:
                break
            target = (float(expected_position[0]), float(expected_position[1]))
            position_deadline = time.monotonic() + self.frame_timeout_s
            reached_spawn = False
            with self._condition:
                while True:
                    self._check_wait_state()
                    position = self._race_metrics.position
                    is_fresh = self._position_update_count > previous_position_update
                    if is_fresh and position is not None:
                        distance = math.dist(position[:2], target)
                        if distance <= position_tolerance_m:
                            reached_spawn = True
                            break
                    remaining = position_deadline - time.monotonic()
                    if remaining <= 0:
                        if attempt + 1 >= reset_attempts:
                            actual = self._race_metrics.position
                            raise OfficialRosTransportError(
                                "Official training reset did not reach the configured IPS spawn "
                                f"after {reset_attempts} attempts; expected={target}, "
                                f"actual={actual}, fresh_position={is_fresh}; "
                                + self._transport_diagnostic()
                            )
                        break
                    self._condition.wait(min(remaining, 0.1))
            if reached_spawn:
                break
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
        if self._cleanup_complete:
            return
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            # spin_once is bounded to 50 ms. Join it before destroying the node
            # or shutting down the context, avoiding concurrent executor teardown.
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise OfficialRosTransportError(
                    "ROS receive thread did not exit after its bounded spin wait; "
                    "node/context were left intact to avoid a teardown race"
                )
        if self._owned_node:
            try:
                self.node.destroy_node()
            finally:
                if self._rclpy is not None and self._rclpy.ok():
                    self._rclpy.shutdown()
        self._cleanup_complete = True
