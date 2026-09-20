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
# Docker Desktop uses the container ID as $HOSTNAME (hex), so trailing-digit
# parsing is unsafe (e.g. …9656 → port 14222). Prefer stable rank among the
# Compose service DNS answers for SIM_SERVICE_DNS (default: sim).
if [ -n "${PORT:-}" ]; then
  SIM_PORT="$PORT"
else
  SERVICE_DNS="${SIM_SERVICE_DNS:-sim}"
  MY_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
  IDX=""
  if [ -n "${MY_IP:-}" ]; then
    IDX="$(getent hosts "$SERVICE_DNS" 2>/dev/null | awk '{print $1}' | sort -u | awk -v ip="$MY_IP" '$0 == ip { print NR; exit }')"
  fi
  # Fallback only for explicit compose-style names: …-sim-N
  if [ -z "${IDX:-}" ]; then
    IDX="$(echo "${HOSTNAME}" | sed -n 's/.*-sim-\([0-9][0-9]*\)$/\1/p' || true)"
  fi
  IDX="${IDX:-1}"
  SIM_PORT=$((BASE_PORT + IDX - 1))
fi

echo "[sim] DISPLAY=$DISPLAY"
echo "[sim] binary=$SIM_PATH"
echo "[sim] connecting to brain=$BRAIN_HOST port=$SIM_PORT"
echo "[sim] flags: -batchmode -nographics -ip $BRAIN_HOST -port $SIM_PORT"

cd "$(dirname "$SIM_PATH")"
exec "$SIM_PATH" -batchmode -nographics -ip "$BRAIN_HOST" -port "$SIM_PORT"
