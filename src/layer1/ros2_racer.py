"""Layer 1 transport adapter for the official AutoDRIVE ROS 2 Devkit.

The adapter presents the same small ``step/telemetry`` surface as the
Socket.IO ``Racer``. ROS messages are converted back to Bridge V1 fields
consumed by :class:`TelemetrySnapshot`, keeping coordinate and sensor
interpretation in one place instead of creating a second parser.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from .telemetry import TelemetrySnapshot


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
    return {
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
        node: Optional[Any] = None,
        include_race_metrics: bool = False,
    ) -> None:
        self.port = 0  # compatibility with Layer 2 diagnostics; ROS uses topics
        self.action_interval_s = None
        self.timeout_s = float(timeout_s)
        self.include_race_metrics = bool(include_race_metrics)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._owned_node = node is None
        self._rclpy = None
        if node is None:
            import rclpy
            from geometry_msgs.msg import Point
            from sensor_msgs.msg import Imu, JointState, LaserScan
            from std_msgs.msg import Float32, Int32

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            node = rclpy.create_node("aicar_policy_transport")
            self._msg_types = {
                "Float32": Float32,
                "Int32": Int32,
                "Point": Point,
                "Imu": Imu,
                "JointState": JointState,
                "LaserScan": LaserScan,
            }
        else:
            self._msg_types = None
        self.node = node
        self._condition = threading.Condition()
        self._raw: Dict[str, Any] = {}
        self._raw_receipts: Dict[str, float] = {}
        self._previous_encoder_positions: Optional[tuple[float, float]] = None
        self._previous_encoder_receipt = 0.0
        self._encoder_speed_mps = 0.0
        self._step_counter = 0
        self._telemetry = TelemetrySnapshot(lidar_valid=False)
        self._race_metrics = RaceMetrics()
        self._race_metrics_received: set[str] = set()
        self._last_scan_receipt = 0.0
        self._last_control_scan_receipt = 0.0
        self._last_action_time = 0.0
        self._last_control_interval_s = 0.0
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
        callbacks = (
            (f"{prefix}/imu", types["Imu"], "imu"),
            (f"{prefix}/throttle", types["Float32"], "throttle"),
            (f"{prefix}/steering", types["Float32"], "steering"),
            (f"{prefix}/left_encoder", types["JointState"], "left_encoder"),
            (f"{prefix}/right_encoder", types["JointState"], "right_encoder"),
            (f"{prefix}/lidar", types["LaserScan"], "scan"),
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
        while not self._closed and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    def _on_message(self, field: str, message: Any) -> None:
        now = time.monotonic()
        with self._condition:
            self._raw[field] = message
            self._raw_receipts[field] = now
            if field == "position":
                self._race_metrics = replace(
                    self._race_metrics,
                    position=(float(message.x), float(message.y), float(message.z)),
                )
                self._race_metrics_received.add(field)
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
                if all(key in self._raw for key in required):
                    left_values = self._raw["left_encoder"].position
                    right_values = self._raw["right_encoder"].position
                    if left_values and right_values:
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
                    payload = ros_messages_to_bridge_payload(
                        scan=self._raw["scan"],
                        imu=self._raw["imu"], throttle=self._raw["throttle"],
                        steering=self._raw["steering"], left_encoder=self._raw["left_encoder"],
                        right_encoder=self._raw["right_encoder"],
                        forward_speed_mps=self._encoder_speed_mps,
                    )
                    self._step_counter += 1
                    self._telemetry = TelemetrySnapshot.from_raw_dict(
                        payload, step_id=self._step_counter
                    )
                    if self._last_scan_receipt:
                        self._last_step_duration_s = max(0.0, now - self._last_scan_receipt)
                        self._sim_time += self._last_step_duration_s
                    self._last_scan_receipt = now
                    self._condition.notify_all()

    def _wait_for_scan_after(self, previous_step: int) -> TelemetrySnapshot:
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            while self._step_counter <= previous_step and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No fresh official ROS 2 LiDAR/telemetry frame within {self.timeout_s:.1f}s"
                    )
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("ROS 2 racer is closed")
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
        deadline = time.monotonic() + self.timeout_s
        with self._condition:
            while not self._closed:
                fresh = (
                    self._step_counter > previous_step
                    and self._last_scan_receipt >= command_time
                    and self._last_scan_receipt >= earliest_time
                )
                if fresh:
                    return self._telemetry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"No official LiDAR frame at the declared scan cadence within {self.timeout_s:.1f}s"
                    )
                self._condition.wait(remaining)
            raise RuntimeError("ROS 2 racer is closed")

    def wait_until_ready(self) -> TelemetrySnapshot:
        return self._wait_for_scan_after(0)

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
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Official evaluator topics were not ready (race counters and IPS motion check)"
                    )
                self._condition.wait(remaining)
            return self._race_metrics

    @property
    def race_metrics(self) -> RaceMetrics:
        if not self.include_race_metrics:
            raise RuntimeError("Race metrics are disabled for the policy transport")
        with self._condition:
            return self._race_metrics

    def step(self, throttle: float, steering: float) -> TelemetrySnapshot:
        with self._condition:
            previous_step = self._step_counter
            command_time = time.monotonic()
            prior_action_time = self._last_action_time
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
            self._last_control_interval_s = (
                command_time - prior_action_time if prior_action_time else 0.0
            )
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

    def set_simulation_paused(self, paused: bool) -> None:
        # The official ROS 2 Devkit exposes no pause topic.
        if paused:
            raise NotImplementedError("The official competition Devkit has no simulation-pause topic")

    def resume_simulation(self) -> None:
        return None

    def kill(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._owned_node:
            try:
                self.node.destroy_node()
            finally:
                if self._rclpy is not None and self._rclpy.ok():
                    self._rclpy.shutdown()
