#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source "$repo/scripts/native-linux-env.sh"

exec "$AICAR_NATIVE_PYTHON" -m uvicorn src.layer4.hub.app:app \
  --host "${HUB_HOST:-0.0.0.0}" \
  --port "${HUB_PORT:-8090}"
