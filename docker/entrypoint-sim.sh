#!/bin/bash
set -euo pipefail

# Virtual display for Unity headless / GLX fallback
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

SIM_PATH="${AICAR_SIMULATOR_PATH:-/app/simulator/AutoDRIVE Simulator.x86_64}"
BRAIN_HOST="${BRAIN_HOST:-brain}"
BASE_PORT="${BASE_PORT:-4567}"

if [ ! -f "$SIM_PATH" ]; then
  echo "[sim] ERROR: simulator binary not found at: $SIM_PATH"
  echo "[sim] Mount ./simulator and set AICAR_SIMULATOR_PATH."
  exit 1
fi

chmod +x "$SIM_PATH" || true

# Unique PORT per scaled replica unless PORT is set explicitly.
# Compose hostnames look like: <project>-sim-1, <project>-sim-2, …
if [ -n "${PORT:-}" ]; then
  SIM_PORT="$PORT"
else
  IDX="$(echo "${HOSTNAME}" | sed -n 's/.*[^0-9]\([0-9][0-9]*\)$/\1/p' || true)"
  IDX="${IDX:-1}"
  SIM_PORT=$((BASE_PORT + IDX - 1))
fi

echo "[sim] DISPLAY=$DISPLAY"
echo "[sim] binary=$SIM_PATH"
echo "[sim] connecting to brain=$BRAIN_HOST port=$SIM_PORT"
echo "[sim] flags: -batchmode -nographics -ip $BRAIN_HOST -port $SIM_PORT"

cd "$(dirname "$SIM_PATH")"
exec "$SIM_PATH" -batchmode -nographics -ip "$BRAIN_HOST" -port "$SIM_PORT"
