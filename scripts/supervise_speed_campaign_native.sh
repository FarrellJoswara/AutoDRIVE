#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source "$repo/scripts/native-linux-env.sh"

exec "$AICAR_NATIVE_PYTHON" -u "$repo/scripts/supervise_speed_campaign.py" \
  --python "$AICAR_NATIVE_PYTHON" "$@"
