#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python="${HOME}/.venvs/aicar/bin/python"
if [[ ! -x "$python" ]]; then
  python3 -m venv --system-site-packages "${HOME}/.venvs/aicar"
  python="${HOME}/.venvs/aicar/bin/python"
fi

export PYTHONNOUSERSITE=1
"$python" -m pip install --upgrade pip
"$python" -m pip install 'torch==2.7.1' --extra-index-url https://download.pytorch.org/whl/cu126
"$python" -m pip install -r "$repo/requirements.txt" \
  --constraint "$repo/scripts/requirements-wsl-native-constraints.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu126

set +u
source /opt/ros/jazzy/setup.bash
source "$HOME/autodrive_ws/install/setup.bash"
set -u
"$python" - <<'PY'
import cv2
import rclpy
import stable_baselines3
import torch
print({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "opencv": cv2.__version__,
    "stable_baselines3": stable_baselines3.__version__,
    "rclpy": rclpy.__file__,
})
PY
