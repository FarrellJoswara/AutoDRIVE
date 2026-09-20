#!/bin/bash
set -euo pipefail

echo "[brain] starting: $*"
exec "$@"
