"""Start and verify four native official Devkit/Layer 1 bridge pairs, then stop them."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.layer4.hub.native_official_backend import NativeOfficialBackend


def main() -> int:
    backend = NativeOfficialBackend()
    records_by_env: list[list[dict]] = []
    with tempfile.TemporaryDirectory(prefix="aicar-native-bridge-smoke-") as temp:
        run_dir = Path(temp)
        try:
            for index in range(4):
                records: list[dict] = []
                records_by_env.append(records)
                backend._start_bridge_pair(
                    records, f"bridge-smoke-{os.getpid()}", index, os.environ.copy(),
                    run_dir, relay_port=52000 + index,
                )

            for index in range(4):
                env = os.environ.copy()
                env.update({"ROS_DOMAIN_ID": str(index), "PYTHONNOUSERSITE": "1"})
                result = subprocess.run(
                    ["bash", "-lc", "source /opt/ros/jazzy/setup.bash; "
                     f"source '{backend.ros_workspace}/install/setup.bash'; ros2 node list"],
                    env=env, check=True, capture_output=True, text=True, timeout=20,
                )
                nodes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if "/autodrive_bridge" not in nodes:
                    raise RuntimeError(f"ROS domain {index} did not contain the official bridge node: {nodes}")
                print(f"ROS domain {index}: official Devkit active; Layer 1 relay listening on 5200{index}")
        finally:
            for records in reversed(records_by_env):
                for record in records:
                    process = record["process"]
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 4
                for record in records:
                    process = record["process"]
                    try:
                        process.wait(timeout=max(0.1, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            process.kill()
                        process.wait(timeout=2)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    record["log"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
