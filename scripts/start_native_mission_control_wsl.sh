#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/autodrive_ws/install/setup.bash"
set -u
export PATH="$HOME/.venvs/aicar/bin:$PATH"
export PYTHONNOUSERSITE=1
export AICAR_RUN_BACKEND=native
export AICAR_HUB_PUBLIC_URL="${AICAR_HUB_PUBLIC_URL:-http://127.0.0.1:8091}"
export HUB_HOST="${HUB_HOST:-0.0.0.0}"
export HUB_PORT="${HUB_PORT:-8091}"
cd "$repo"
exec python -m src.layer4.hub.app
