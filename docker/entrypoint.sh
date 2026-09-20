#!/bin/bash
set -e

# Start Xvfb virtual framebuffer on display :99 for headless Unity rendering
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# Ensure simulator executable has execute permissions if present
if [ -f "/app/simulator/AutoDRIVE Simulator.x86_64" ]; then
    chmod +x "/app/simulator/AutoDRIVE Simulator.x86_64"
fi

echo "[AutoDRIVE Docker] Virtual display :99 initialized."
echo "[AutoDRIVE Docker] Starting command: $@"

exec "$@"
