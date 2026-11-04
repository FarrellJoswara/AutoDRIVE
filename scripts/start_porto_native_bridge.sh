#!/usr/bin/env bash
set -euo pipefail

ROOT="${AICAR_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${AICAR_PORTO_RUN_DIR:?Set AICAR_PORTO_RUN_DIR to the isolated Porto run directory}"
ROS_DOMAIN="${AICAR_PORTO_ROS_DOMAIN_ID:-86}"
DEVKIT_PORT="${AICAR_PORTO_DEVKIT_PORT:-4567}"
RELAY_PORT="${AICAR_PORTO_RELAY_PORT:-4586}"

mkdir -p "$RUN_DIR"
cd "$ROOT"
export PYTHONNOUSERSITE=1
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/autodrive_ws/install/setup.bash"
set -u
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID="$ROS_DOMAIN"
export AICAR_ROS_DOMAIN_ID_BASE="$ROS_DOMAIN"
export AICAR_DEVKIT_PORT="$DEVKIT_PORT"

setsid ros2 launch autodrive_roboracer bringup_headless.launch.py \
  >"$RUN_DIR/devkit_env0.log" 2>&1 </dev/null &
DEVKIT_PID=$!
cleanup() {
  kill -- "-$DEVKIT_PID" 2>/dev/null || kill "$DEVKIT_PID" 2>/dev/null || true
  wait "$DEVKIT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 60))
until (echo >"/dev/tcp/127.0.0.1/$DEVKIT_PORT") >/dev/null 2>&1; do
  if ! kill -0 "$DEVKIT_PID" 2>/dev/null; then
    cat "$RUN_DIR/devkit_env0.log" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    cat "$RUN_DIR/devkit_env0.log" >&2
    echo "Devkit did not listen on port $DEVKIT_PORT within 60 seconds" >&2
    exit 1
  fi
  sleep 0.25
done

"$HOME/.venvs/aicar/bin/python" -u -m src.layer1.official_bridge_relay \
  --host 0.0.0.0 --port "$RELAY_PORT" \
  --devkit-url "http://127.0.0.1:$DEVKIT_PORT" \
  >"$RUN_DIR/relay_env0.log" 2>&1
