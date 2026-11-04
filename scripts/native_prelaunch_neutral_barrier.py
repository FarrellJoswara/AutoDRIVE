"""Hold both official actuators at zero before a native practice GUI starts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class NeutralBarrier(Node):
    def __init__(self) -> None:
        super().__init__("aicar_native_prelaunch_neutral_barrier")
        self.throttle = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10
        )
        self.steering = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10
        )

    def publish_zero(self) -> None:
        self.throttle.publish(Float32(data=0.0))
        self.steering.publish(Float32(data=0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--subscriber-timeout-s", type=float, default=20.0)
    args = parser.parse_args()
    if args.duration_s < 1.0 or args.subscriber_timeout_s <= 0.0:
        raise ValueError("barrier duration and subscriber timeout must be positive")

    rclpy.init()
    node = NeutralBarrier()
    started = time.monotonic()
    deadline = started + args.subscriber_timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.throttle.get_subscription_count() > 0 and node.steering.get_subscription_count() > 0:
            break
        time.sleep(0.03)
    subscribers = {
        "throttle": node.throttle.get_subscription_count(),
        "steering": node.steering.get_subscription_count(),
    }
    verified = subscribers["throttle"] > 0 and subscribers["steering"] > 0
    sent = 0
    if verified:
        hold_until = time.monotonic() + args.duration_s
        while time.monotonic() < hold_until:
            node.publish_zero()
            sent += 1
            rclpy.spin_once(node, timeout_sec=0.025)
            time.sleep(0.025)
        node.publish_zero()
        sent += 1
    report = {
        "verified": verified and sent >= 20,
        "zero_commands_sent": sent,
        "duration_s": time.monotonic() - started,
        "hold_duration_s": args.duration_s if sent else 0.0,
        "ros_domain_id": int(os.environ.get("ROS_DOMAIN_ID", "0")),
        "throttle_subscribers": subscribers["throttle"],
        "steering_subscribers": subscribers["steering"],
        "throttle_value": 0.0,
        "steering_value": 0.0,
        "simulator_not_yet_launched": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    node.publish_zero()
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps(report), flush=True)
    return 0 if report["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
