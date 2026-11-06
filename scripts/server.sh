#!/usr/bin/env bash
# The hub itself needs no GPU; individual learners and simulators request GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
compose=(docker compose -f docker-compose.server.yml)
case "${1:-help}" in
  doctor)
    test "$(uname -s)" = Linux || { echo 'Linux is required'; exit 1; }
    test "$(uname -m)" = x86_64 || { echo 'Official simulator requires x86_64'; exit 1; }

    if ! command -v docker >/dev/null 2>&1; then
      echo 'Docker CLI is missing. Install/enable the administrator-approved Docker Engine and Compose v2, then rerun doctor.' >&2
      exit 1
    fi
    compose_version=$(docker compose version --short 2>&1) || {
      echo "Docker Compose v2 is unavailable ($compose_version). Install/enable the Compose v2 plugin, then rerun doctor." >&2
      exit 1
    }
    compose_major=${compose_version%%.*}
    if [[ ! "$compose_major" =~ ^[0-9]+$ ]] || (( compose_major < 2 )); then
      echo "Docker Compose v2 is required; found '$compose_version'." >&2
      exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
      echo 'Docker daemon is unavailable or this account cannot access it. Ask the host administrator to start/enable the approved daemon and grant this account access; do not install or enable a privileged service from this script.' >&2
      exit 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo 'nvidia-smi is missing. An administrator must provide a working NVIDIA driver before GPU simulator/training checks.' >&2
      exit 1
    fi
    if ! nvidia-smi >/dev/null 2>&1; then
      echo 'nvidia-smi cannot communicate with the host NVIDIA driver. Ask the host administrator to repair GPU driver access.' >&2
      exit 1
    fi
    "${compose[@]}" config --quiet
    nvidia-smi
    docker run --rm --gpus all --entrypoint /bin/bash aicar-server-runtime:local -lc '
source /opt/ros/jazzy/setup.bash
python - <<"PY"
import cv2, rclpy, torch, stable_baselines3
assert torch.cuda.is_available(), "CUDA is not available inside Docker"
x = torch.ones(8, device="cuda"); print(x.sum().item())
print({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "sb3": stable_baselines3.__version__, "opencv": cv2.__version__})
PY'
    ;;
  build)
    "${compose[@]}" build brain
    docker pull autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice
    docker pull autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete
    ;;
  start)
    mkdir -p logs/layer4 simulator/maps
    "${compose[@]}" up -d --no-build --wait brain
    echo 'Mission Control: use ssh -N -L 8090:127.0.0.1:8090 USER@SERVER'
    ;;
  logs) "${compose[@]}" logs -f --tail 100 brain ;;
  status) "${compose[@]}" ps ;;
  stop)
    # Request cooperative checkpoint/stop before stopping the hub. Fail closed
    # if the API rejects the stop, rather than orphaning active worker containers.
    "${compose[@]}" exec -T brain python -c '
import json, urllib.request
base = "http://127.0.0.1:8090"
runs = json.load(urllib.request.urlopen(base + "/train/runs", timeout=15))["runs"]
for row in runs:
    if row.get("state") in {"starting", "running", "stopping"}:
        req = urllib.request.Request(base + "/train/runs/" + row["run_id"] + "/stop", data=b"{}", headers={"Content-Type": "application/json"})
        result = json.load(urllib.request.urlopen(req, timeout=180))
        assert result.get("state") == "stopped" and not result.get("cleanup_error"), result
        print(result)'
    "${compose[@]}" stop brain
    ;;
  *) echo 'Usage: bash scripts/server.sh {build|doctor|start|status|logs|stop}' ;;
esac
