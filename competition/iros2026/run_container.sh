#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash

if [[ ! -f "${AICAR_MODEL_PATH:-/models/policy.zip}" ]]; then
  echo "ERROR: set AICAR_MODEL_PATH to a readable PPO checkpoint" >&2
  exit 2
fi

# The competition Devkit API is a ROS node in the official workspace. Start it
# unchanged, then run our policy/evaluator node in the same Devkit container.
ros2 launch autodrive_roboracer bringup_headless.launch.py &
bridge_pid=$!
cleanup() {
  kill "$bridge_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "${AICAR_MODE:-policy}" == "evaluate" ]]; then
  echo "Running local evaluation monitor; restricted score topics are read for measurement only."
  python3 -m src.layer3.official_evaluate \
    --model "${AICAR_MODEL_PATH:-/models/policy.zip}" \
    --attempts "${AICAR_EVALUATION_ATTEMPTS:-1}" \
    --device cpu \
    --timeout-s "${AICAR_SENSOR_TIMEOUT_S:-180}" \
    --wall-timeout-s "${AICAR_RACE_WALL_TIMEOUT_S:-300}" \
    --max-steps "${AICAR_RACE_STEP_GUARD:-150000}" \
    --out "${AICAR_EVALUATION_OUTPUT:-/tmp/aicar-iros-evaluation.json}"
else
  echo "Running competition policy; only allowed sensor and actuator topics are enabled."
  python3 -m src.layer3.official_policy \
    --model "${AICAR_MODEL_PATH:-/models/policy.zip}" \
    --device cpu \
    --timeout-s "${AICAR_SENSOR_TIMEOUT_S:-5}"
fi

# Keep the official bridge alive for inspection and rosbag recording after the
# race, as required by the organizer's workflow.
wait "$bridge_pid"
