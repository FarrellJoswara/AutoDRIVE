#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash

controller="${AICAR_CONTROLLER:-ppo}"
mode="${AICAR_MODE:-policy}"
if [[ "$mode" != "train" && "$mode" != "bridge" && "$controller" == "ppo" && ! -f "${AICAR_MODEL_PATH:-/models/policy.zip}" ]]; then
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
# Docker signals this shell, not its Python child. Keep sensors alive while
# the trainer exits cooperatively and writes its final checkpoint.
child_pid=""
request_stop() {
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  else
    exit 143
  fi
}
run_child() {
  "$@" &
  child_pid=$!
  local code=0
  while true; do
    if wait "$child_pid"; then code=0; else code=$?; fi
    # A trapped signal interrupts wait before the child has exited.
    kill -0 "$child_pid" 2>/dev/null || break
  done
  child_pid=""
  return "$code"
}
trap cleanup EXIT
trap request_stop INT TERM

if [[ "$mode" == "bridge" ]]; then
  wait "$bridge_pid"
  exit $?
elif [[ "$mode" == "train" ]]; then
  # Keep the official development trainer visible in Mission Control by
  # default. Override AICAR_HUB_URL for non-Docker-Desktop deployments, or
  # provide HUB_URL explicitly when launching the container.
  if [[ -z "${HUB_URL:-}" ]]; then
    export HUB_URL="${AICAR_HUB_URL:-http://host.docker.internal:8090}"
  fi
  echo "Training PPO against the official simulator; restricted metrics/reset are training-only."
  echo "Watch telemetry destination: ${HUB_URL}"
  train_args=(
    --n-envs "${AICAR_TRAIN_N_ENVS:-1}"
    --out "${AICAR_TRAIN_OUT:-/runs/official}"
    --total-timesteps "${AICAR_TRAIN_TIMESTEPS:-1000000}"
    --stop-on-plateau "${AICAR_TRAIN_STOP_ON_PLATEAU:-1}"
    --plateau-min-timesteps "${AICAR_TRAIN_PLATEAU_MIN_TIMESTEPS:-100000}"
    --plateau-window-episodes "${AICAR_TRAIN_PLATEAU_WINDOW_EPISODES:-5}"
    --plateau-patience "${AICAR_TRAIN_PLATEAU_PATIENCE:-5}"
    --plateau-min-improvement-pct "${AICAR_TRAIN_PLATEAU_MIN_IMPROVEMENT_PCT:-1.0}"
    --max-duration-hours "${AICAR_TRAIN_MAX_DURATION_HOURS:-0}"
    --seed "${AICAR_TRAIN_SEED:-0}"
    --device "${AICAR_PPO_DEVICE:-cpu}"
    --checkpoint-every "${AICAR_TRAIN_CHECKPOINT_EVERY:-10000}"
    --n-steps "${AICAR_PPO_N_STEPS:-2048}"
    --learning-rate "${AICAR_PPO_LEARNING_RATE:-0.0001}"
    --n-epochs "${AICAR_PPO_N_EPOCHS:-4}"
    --gamma "${AICAR_PPO_GAMMA:-0.9995}"
    --gae-lambda "${AICAR_PPO_GAE_LAMBDA:-0.95}"
    --timeout-s "${AICAR_SENSOR_TIMEOUT_S:-180}"
    --training-timeout-s "${AICAR_TRAIN_WATCHDOG_S:-600}"
    --race-laps "${AICAR_TRAIN_RACE_LAPS:-10}"
    --warmup-laps "${AICAR_TRAIN_WARMUP_LAPS:-1}"
    --time-cost-per-simulated-second "${AICAR_TRAIN_TIME_COST:-1}"
    --lap-completion-reward "${AICAR_TRAIN_LAP_REWARD:-100}"
    --collision-penalty-base "${AICAR_TRAIN_COLLISION_PENALTY_BASE:-10}"
    --failed-episode-penalty "${AICAR_TRAIN_FAILURE_PENALTY:-1000}"
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
  run_child python3 -u -m src.layer3.official_train "${train_args[@]}"
  exit $?
elif [[ "$mode" == "evaluate" ]]; then
  echo "Running local evaluation monitor; restricted score topics are read for measurement only."
  run_child python3 -m src.layer3.official_evaluate \
    --model "${AICAR_MODEL_PATH:-/models/policy.zip}" \
    --controller "$controller" \
    --attempts "${AICAR_EVALUATION_ATTEMPTS:-1}" \
    --observation-profile "${AICAR_OBSERVATION_PROFILE:-official_sensors}" \
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
  run_child python3 -m src.layer3.official_policy \
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
if [[ "${AICAR_KEEP_BRIDGE:-1}" == "1" ]]; then wait "$bridge_pid"; fi
