"""Layer 1 transport adapter for the official AutoDRIVE ROS 2 Devkit.

The adapter presents the same small ``step/reset/telemetry`` surface as the
Socket.IO ``Racer``.  ROS messages are converted back to the Bridge V1 fields
consumed by :class:`TelemetrySnapshot`, keeping coordinate and sensor
interpretation in one place instead of creating a second parser.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from .telemetry import TelemetrySnapshot


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
    odometry: Any,
    imu: Any,
    throttle: Any,
    steering: Any,
    left_encoder: Any,
    right_encoder: Any,
    lap_count: Any,
    lap_time: Any,
    last_lap_time: Any,
    best_lap_time: Any,
    collision_count: Any,
) -> Dict[str, Any]:
    """Map official Devkit messages to the original Bridge V1 signal names.

    The official bridge publishes raw Bridge vectors in Odometry/IMU and
    publishes its range array directly as LaserScan.ranges.  In particular,
    this function deliberately does not rotate, reverse, normalize, or
    otherwise reinterpret those values; ``TelemetrySnapshot`` and Layer 2
    remain the sole owners of those established transformations.
    """
    pose = odometry.pose.pose
    twist = odometry.twist.twist
    orientation = imu.orientation
    left = left_encoder.position[0] if left_encoder.position else 0.0
    right = right_encoder.position[0] if right_encoder.position else 0.0
    return {
        "V1 Position": _xyz(pose.position),
        "V1 Orientation Quaternion": (
            f"{orientation.x} {orientation.y} {orientation.z} {orientation.w}"
        ),
        "V1 Linear Velocity": _xyz(twist.linear),
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
        "V1 Lap Count": int(lap_count.data),
        "V1 Lap Time": float(lap_time.data),
        "V1 Last Lap Time": float(last_lap_time.data),
        "V1 Best Lap Time": float(best_lap_time.data),
        "V1 Collisions": int(collision_count.data),
    }


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
    ) -> None:
        self.port = 0  # compatibility with Layer 2 diagnostics; ROS uses topics
        self.action_interval_s = None
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._owned_node = node is None
        self._rclpy = None
        if node is None:
            import rclpy
            from sensor_msgs.msg import Imu, JointState, LaserScan
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Bool, Float32, Int32

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            node = rclpy.create_node("aicar_policy_transport")
            self._msg_types = {
                "Bool": Bool,
                "Float32": Float32,
                "Int32": Int32,
                "Imu": Imu,
                "JointState": JointState,
                "LaserScan": LaserScan,
                "Odometry": Odometry,
            }
        else:
            self._msg_types = None
        self.node = node
        self._condition = threading.Condition()
        self._raw: Dict[str, Any] = {}
        self._step_counter = 0
        self._telemetry = TelemetrySnapshot(lidar_valid=False)
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
            "reset": self.node.create_publisher(types["Bool"], "/autodrive/reset_command", 10),
        }
        callbacks = (
            (f"{prefix}/odom", types["Odometry"], "odometry"),
            (f"{prefix}/imu", types["Imu"], "imu"),
            (f"{prefix}/throttle", types["Float32"], "throttle"),
            (f"{prefix}/steering", types["Float32"], "steering"),
            (f"{prefix}/left_encoder", types["JointState"], "left_encoder"),
            (f"{prefix}/right_encoder", types["JointState"], "right_encoder"),
            (f"{prefix}/lap_count", types["Int32"], "lap_count"),
            (f"{prefix}/lap_time", types["Float32"], "lap_time"),
            (f"{prefix}/last_lap_time", types["Float32"], "last_lap_time"),
            (f"{prefix}/best_lap_time", types["Float32"], "best_lap_time"),
            (f"{prefix}/collision_count", types["Int32"], "collision_count"),
            (f"{prefix}/lidar", types["LaserScan"], "scan"),
        )
        for topic, message_type, name in callbacks:
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
            if field == "scan":
                required = (
                    "odometry", "imu", "throttle", "steering", "left_encoder",
                    "right_encoder", "lap_count", "lap_time", "last_lap_time",
                    "best_lap_time", "collision_count",
                )
                if all(key in self._raw for key in required):
                    payload = ros_messages_to_bridge_payload(
                        scan=self._raw["scan"], odometry=self._raw["odometry"],
                        imu=self._raw["imu"], throttle=self._raw["throttle"],
                        steering=self._raw["steering"], left_encoder=self._raw["left_encoder"],
                        right_encoder=self._raw["right_encoder"], lap_count=self._raw["lap_count"],
                        lap_time=self._raw["lap_time"], last_lap_time=self._raw["last_lap_time"],
                        best_lap_time=self._raw["best_lap_time"],
                        collision_count=self._raw["collision_count"],
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

    def reset(self) -> TelemetrySnapshot:
        if not self._publishers:
            return self.wait_until_ready()
        with self._condition:
            self._last_action_time = 0.0
            self._last_control_interval_s = 0.0
        with self._condition:
            previous_step = self._step_counter
        self._publish_command(0.0, 0.0)
        reset_msg = self._msg_types["Bool"]()
        reset_msg.data = True
        self._publishers["reset"].publish(reset_msg)
        self._wait_for_scan_after(previous_step)
        reset_msg.data = False
        self._publishers["reset"].publish(reset_msg)
        with self._condition:
            previous_step = self._step_counter
        snap = self._wait_for_scan_after(previous_step)
        with self._condition:
            self._last_control_scan_receipt = self._last_scan_receipt
        return snap

    def _publish_command(self, throttle: float, steering: float) -> None:
        if not self._publishers:
            return
        t = self._msg_types["Float32"]()
        t.data = float(throttle)
        s = self._msg_types["Float32"]()
        s.data = float(steering)
        self._publishers["throttle"].publish(t)
        self._publishers["steering"].publish(s)

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
