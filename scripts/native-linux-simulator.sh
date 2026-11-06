#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${AICAR_NATIVE_STATE:-$ROOT/.native-linux}"
RELEASE="${AICAR_NATIVE_RELEASE:-practice}"
case "$RELEASE" in
  practice) DEFAULT_RUNTIME="$STATE/runtime" ;;
  competition) DEFAULT_RUNTIME="$STATE/competition-runtime" ;;
  *) echo "Unsupported native Linux official release: $RELEASE" >&2; exit 2 ;;
esac
export AICAR_NATIVE_RELEASE="$RELEASE"
export AICAR_NATIVE_RUNTIME="${AICAR_NATIVE_RUNTIME:-$DEFAULT_RUNTIME}"
if [[ ! -x "$STATE/host-root/usr/bin/xdotool" || ! -f "$STATE/host-root/usr/lib/xorg/modules/drivers/dummy_drv.so" ]]; then
  echo "Native simulator dependencies are not bootstrapped; run: bash scripts/native-linux-bootstrap.sh" >&2
  exit 2
fi
python3 "$ROOT/scripts/native-linux-runner.py"
