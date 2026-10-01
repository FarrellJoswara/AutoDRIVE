#!/bin/bash
set -euo pipefail

# Virtual display for Unity headless / GLX fallback
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

SIM_PATH="${AICAR_SIMULATOR_PATH:-/app/simulator/AutoDRIVE Simulator.x86_64}"
BRAIN_HOST="${BRAIN_HOST:-brain}"
BASE_PORT="${BASE_PORT:-4567}"
MAPS_DIR="${AICAR_MAPS_DIR:-/app/simulator/maps}"
ACTIVE_FILE="${AICAR_ACTIVE_MAP_FILE:-${MAPS_DIR}/.active_map.json}"

if [ ! -f "$SIM_PATH" ]; then
  echo "[sim] ERROR: simulator binary not found at: $SIM_PATH"
  echo "[sim] Mount ./simulator and set AICAR_SIMULATOR_PATH."
  exit 1
fi

chmod +x "$SIM_PATH" || true

# Resolve map id: explicit env wins; else maps/.active_map.json written by hub Activate.
if [ -z "${AICAR_MAP_ID:-}" ] && [ -f "$ACTIVE_FILE" ]; then
  AICAR_MAP_ID="$(sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$ACTIVE_FILE" | head -n 1 || true)"
fi
if [ "${AICAR_MAP_ID:-}" = "none" ]; then
  AICAR_MAP_ID=""
fi
export AICAR_MAP_ID="${AICAR_MAP_ID:-}"

if [ -n "${PORT:-}" ]; then
  SIM_PORT="$PORT"
else
  SERVICE_DNS="${SIM_SERVICE_DNS:-sim}"
  MY_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
  IDX=""

  if [ -n "${MY_IP:-}" ] && [ -f /etc/hosts ]; then
    HOST_LINE="$(awk -v ip="$MY_IP" '$1 == ip { print; exit }' /etc/hosts || true)"
    IDX="$(echo "$HOST_LINE" | sed -n 's/.*-sim-\([0-9][0-9]*\).*/\1/p' | head -n 1 || true)"
  fi

  if [ -z "${IDX:-}" ] && [ -n "${MY_IP:-}" ]; then
    EXPECTED="${AICAR_EXPECTED_SIMS:-0}"
    for _try in $(seq 1 40); do
      IPS="$(getent hosts "$SERVICE_DNS" 2>/dev/null | awk '{print $1}' | sort -u || true)"
      COUNT="$(printf '%s\n' "$IPS" | awk 'NF{c++} END{print c+0}')"
      IDX="$(printf '%s\n' "$IPS" | awk -v ip="$MY_IP" '$0 == ip { print NR; exit }')"
      if [ -n "${IDX:-}" ]; then
        if [ "$EXPECTED" -gt 0 ] && [ "$COUNT" -lt "$EXPECTED" ]; then
          sleep 0.25
          continue
        fi
        sleep 0.25
        IPS2="$(getent hosts "$SERVICE_DNS" 2>/dev/null | awk '{print $1}' | sort -u || true)"
        IDX2="$(printf '%s\n' "$IPS2" | awk -v ip="$MY_IP" '$0 == ip { print NR; exit }')"
        if [ -n "${IDX2:-}" ] && [ "$IDX" = "$IDX2" ]; then
          break
        fi
        IDX="${IDX2:-$IDX}"
      fi
      sleep 0.25
    done
  fi

  if [ -z "${IDX:-}" ]; then
    IDX="$(echo "${HOSTNAME}" | sed -n 's/.*-sim-\([0-9][0-9]*\)$/\1/p' || true)"
  fi
  IDX="${IDX:-1}"
  SIM_PORT=$((BASE_PORT + IDX - 1))
fi

# NOTE: do not pass -map-id on Unity argv. Stock CLIManager parses -ip/-port and
# may ignore or mishandle unknown flags; TrackLoader reads AICAR_MAP_ID / active JSON.
BRAIN_IP="$(getent hosts "$BRAIN_HOST" 2>/dev/null | awk '{print $1; exit}')"
BRAIN_IP="${BRAIN_IP:-$BRAIN_HOST}"

echo "[sim] DISPLAY=$DISPLAY"
echo "[sim] binary=$SIM_PATH"
echo "[sim] connecting to brain=$BRAIN_HOST ($BRAIN_IP) port=$SIM_PORT"
echo "[sim] AICAR_MAP_ID=${AICAR_MAP_ID:-"(builtin)"} (via env/file, not argv)"
echo "[sim] flags: -batchmode -nographics -ip $BRAIN_IP -port $SIM_PORT"

cd "$(dirname "$SIM_PATH")"
exec "$SIM_PATH" -batchmode -nographics -ip "$BRAIN_IP" -port "$SIM_PORT"
