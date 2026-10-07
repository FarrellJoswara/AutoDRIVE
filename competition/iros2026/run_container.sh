#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash

controller="${AICAR_CONTROLLER:-ppo}"
mode="${AICAR_MODE:-policy}"
if [[ "$mode" != "train" && "$controller" == "ppo" && ! -f "${AICAR_MODEL_PATH:-/models/policy.zip}" ]]; then
  echo "ERROR: set AICAR_MODEL_PATH to a readable PPO checkpoint" >&2
  exit 2
fi

# The competition Devkit API is a ROS node in the official workspace. Start it
# unchanged, then run our policy/evaluator node in the same Devkit container.
ros2 launch autodrive_roboracer bringup_headless.launch.py &
bridge_pid=$!
python3 -u -m src.layer1.official_bridge_relay \
  --devkit-url "${AICAR_DEVKIT_URL:-http://127.0.0.1:4567}" \
  --host "${AICAR_RELAY_HOST:-0.0.0.0}" \
  --port "${AICAR_RELAY_PORT:-4568}" \
  --response-timeout-s "${AICAR_BRIDGE_RESPONSE_TIMEOUT_S:-2}" &
relay_pid=$!
cleanup() {
  kill "$relay_pid" 2>/dev/null || true
  wait "$relay_pid" 2>/dev/null || true
  kill "$bridge_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$mode" == "train" ]]; then
  echo "Training PPO against the official simulator; restricted metrics/reset are training-only."
  train_args=(
    --out "${AICAR_TRAIN_OUT:-/runs/official}"
    --total-timesteps "${AICAR_TRAIN_TIMESTEPS:-1000000}"
    --seed "${AICAR_TRAIN_SEED:-0}"
    --checkpoint-every "${AICAR_TRAIN_CHECKPOINT_EVERY:-10000}"
    --n-steps "${AICAR_PPO_N_STEPS:-2048}"
    --learning-rate "${AICAR_PPO_LEARNING_RATE:-0.0001}"
    --n-epochs "${AICAR_PPO_N_EPOCHS:-4}"
    --gamma "${AICAR_PPO_GAMMA:-0.9995}"
    --gae-lambda "${AICAR_PPO_GAE_LAMBDA:-0.95}"
    --timeout-s "${AICAR_SENSOR_TIMEOUT_S:-180}"
    --training-timeout-s "${AICAR_TRAIN_WATCHDOG_S:-600}"
    --observation-profile "${AICAR_OBSERVATION_PROFILE:-official_sensors}"
    --steering-action-scale "${AICAR_STEERING_ACTION_SCALE:-1.0}"
    --straight-throttle-gain "${AICAR_STRAIGHT_THROTTLE_GAIN:-1.0}"
    --straight-throttle-steering-threshold "${AICAR_STRAIGHT_THROTTLE_STEERING_THRESHOLD:-0.15}"
    --negative-throttle-mode "${AICAR_NEGATIVE_THROTTLE_MODE:-allow}"
    --steering-mode "${AICAR_STEERING_MODE:-normal}"
    --throttle-mode "${AICAR_THROTTLE_MODE:-bidirectional}"
  )
  if [[ -n "${AICAR_TRAIN_RESUME:-}" ]]; then
    train_args+=(--resume "$AICAR_TRAIN_RESUME")
  fi
  python3 -u -m src.layer3.official_train "${train_args[@]}"
elif [[ "$mode" == "evaluate" ]]; then
  echo "Running local evaluation monitor; restricted score topics are read for measurement only."
  python3 -m src.layer3.official_evaluate \
    --model "${AICAR_MODEL_PATH:-/models/policy.zip}" \
    --controller "$controller" \
    --attempts "${AICAR_EVALUATION_ATTEMPTS:-1}" \
    --device cpu \
    --timeout-s "${AICAR_SENSOR_TIMEOUT_S:-180}" \
    --wall-timeout-s "${AICAR_RACE_WALL_TIMEOUT_S:-300}" \
    --max-steps "${AICAR_RACE_STEP_GUARD:-150000}" \
    --trace-steps "${AICAR_EVALUATION_TRACE_STEPS:-0}" \
    --negative-throttle-mode "${AICAR_NEGATIVE_THROTTLE_MODE:-allow}" \
    --steering-mode "${AICAR_STEERING_MODE:-normal}" \
    --throttle-mode "${AICAR_THROTTLE_MODE:-bidirectional}" \
    --steering-action-scale "${AICAR_STEERING_ACTION_SCALE:-1.0}" \
    --straight-throttle-gain "${AICAR_STRAIGHT_THROTTLE_GAIN:-1.0}" \
    --straight-throttle-steering-threshold "${AICAR_STRAIGHT_THROTTLE_STEERING_THRESHOLD:-0.15}" \
    --out "${AICAR_EVALUATION_OUTPUT:-/tmp/aicar-iros-evaluation.json}"
else
  echo "Running competition policy; only allowed sensor and actuator topics are enabled."
  python3 -m src.layer3.official_policy \
    --model "${AICAR_MODEL_PATH:-/models/policy.zip}" \
    --controller "$controller" \
    --device cpu \
    --timeout-s "${AICAR_SENSOR_TIMEOUT_S:-5}" \
    --negative-throttle-mode "${AICAR_NEGATIVE_THROTTLE_MODE:-allow}" \
    --steering-mode "${AICAR_STEERING_MODE:-normal}" \
    --throttle-mode "${AICAR_THROTTLE_MODE:-bidirectional}" \
    --steering-action-scale "${AICAR_STEERING_ACTION_SCALE:-1.0}" \
    --straight-throttle-gain "${AICAR_STRAIGHT_THROTTLE_GAIN:-1.0}" \
    --straight-throttle-steering-threshold "${AICAR_STRAIGHT_THROTTLE_STEERING_THRESHOLD:-0.15}"
fi

# Keep the official bridge alive for inspection and rosbag recording after the
# race, as required by the organizer's workflow.
wait "$bridge_pid"
