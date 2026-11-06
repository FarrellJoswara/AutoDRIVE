#!/usr/bin/env bash
# Source this file to activate the user-space RoboStack Jazzy + learner stack.
# RoboStack owns Python, ROS, NumPy, OpenCV, and cv_bridge. The existing learner
# venv is overlaid except for NumPy/OpenCV, whose ABI must match cv_bridge.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source scripts/native-linux-env.sh" >&2
  exit 2
fi

_aicar_native_fail() {
  echo "native-linux-env: $*" >&2
  return 1
}

AICAR_NATIVE_ROOT="${AICAR_NATIVE_ROOT:-/tmp/aicar-ros-O15Zu3}"
AICAR_LEARNER_VENV="${AICAR_LEARNER_VENV:-$HOME/.venvs/aicar}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
AICAR_ROS_PREFIX="$AICAR_NATIVE_ROOT/jazzy"
AICAR_ROS_WORKSPACE="$AICAR_NATIVE_ROOT/autodrive_ws"
AICAR_VENV_SITE="$AICAR_LEARNER_VENV/lib/python3.12/site-packages"
AICAR_VENV_OVERLAY="$AICAR_NATIVE_ROOT/venv-overlay"

[[ -x "$MICROMAMBA_BIN" ]] || { _aicar_native_fail "micromamba is missing: $MICROMAMBA_BIN"; return 1; }
[[ -x "$AICAR_ROS_PREFIX/bin/python" ]] || { _aicar_native_fail "RoboStack Python is missing: $AICAR_ROS_PREFIX/bin/python"; return 1; }
[[ -f "$AICAR_ROS_PREFIX/setup.bash" ]] || { _aicar_native_fail "ROS setup is missing: $AICAR_ROS_PREFIX/setup.bash"; return 1; }
[[ -f "$AICAR_ROS_WORKSPACE/install/setup.bash" ]] || { _aicar_native_fail "Devkit setup is missing: $AICAR_ROS_WORKSPACE/install/setup.bash"; return 1; }
[[ -x "$AICAR_LEARNER_VENV/bin/python" && -d "$AICAR_VENV_SITE" ]] || { _aicar_native_fail "learner venv is missing: $AICAR_LEARNER_VENV"; return 1; }

_aicar_ros_abi="$("$AICAR_ROS_PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')" || return 1
_aicar_learner_abi="$("$AICAR_LEARNER_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')" || return 1
[[ "$_aicar_ros_abi" == "$_aicar_learner_abi" ]] || {
  _aicar_native_fail "Python ABI mismatch: RoboStack=$_aicar_ros_abi learner=$_aicar_learner_abi"
  return 1
}

# Keep RoboStack's NumPy/OpenCV first-class: cv_bridge is compiled against them.
mkdir -p "$AICAR_VENV_OVERLAY" || return 1
"$AICAR_ROS_PREFIX/bin/python" - "$AICAR_VENV_SITE" "$AICAR_VENV_OVERLAY" <<'PY' || return 1
import os
import sys
from pathlib import Path

source, overlay = map(Path, sys.argv[1:])
def excluded(name):
    return (
        name in {"cv2", "numpy", "numpy.libs"}
        or name.startswith(("numpy-", "opencv_python", "cv2-"))
    )

for entry in source.iterdir():
    if excluded(entry.name):
        continue
    link = overlay / entry.name
    if os.path.lexists(link):
        if not link.is_symlink() or link.resolve() != entry.resolve():
            raise SystemExit(f"Refusing conflicting overlay entry: {link}")
        continue
    link.symlink_to(entry)
PY

export MAMBA_ROOT_PREFIX="$AICAR_NATIVE_ROOT/mamba"
_aicar_had_nounset=0
case "$-" in *u*) _aicar_had_nounset=1 ;; esac
set +u
eval "$("$MICROMAMBA_BIN" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")" || {
  if (( _aicar_had_nounset )); then set -u; fi
  return 1
}
micromamba activate "$AICAR_ROS_PREFIX" || {
  if (( _aicar_had_nounset )); then set -u; fi
  return 1
}
source "$AICAR_ROS_PREFIX/setup.bash"
_aicar_source_status=$?
if (( _aicar_source_status == 0 )); then
  source "$AICAR_ROS_WORKSPACE/install/setup.bash"
  _aicar_source_status=$?
fi
if (( _aicar_had_nounset )); then set -u; fi
(( _aicar_source_status == 0 )) || { _aicar_native_fail "failed to source ROS or Devkit setup"; return 1; }

export PYTHONNOUSERSITE=1
case ":${PYTHONPATH:-}:" in
  *":$AICAR_VENV_OVERLAY:"*) ;;
  *) export PYTHONPATH="$AICAR_VENV_OVERLAY${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export AICAR_NATIVE_PYTHON="$AICAR_ROS_PREFIX/bin/python"
export AICAR_NATIVE_ROS_SETUP="$AICAR_ROS_PREFIX/setup.bash"
export AICAR_NATIVE_ROS_WORKSPACE="$AICAR_ROS_WORKSPACE"
export AICAR_RUN_BACKEND=native_linux
export AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S="${AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S:-0.20}"

"$AICAR_NATIVE_PYTHON" - <<'PY' || { _aicar_native_fail "unified Torch/ROS/Devkit preflight failed"; return 1; }
import sys
import sysconfig
import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
import autodrive_roboracer.autodrive_bridge

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the learner Torch build")
bridge = CvBridge()
image = np.zeros((2, 3, 3), dtype=np.uint8)
roundtrip = bridge.imgmsg_to_cv2(
    bridge.cv2_to_imgmsg(image, encoding="bgr8"), desired_encoding="bgr8"
)
if roundtrip.shape != image.shape:
    raise SystemExit(f"cv_bridge image shape mismatch: {roundtrip.shape}")
print(f"native-linux-env: Python {sys.version.split()[0]} {sysconfig.get_config_var('SOABI')}")
print(f"native-linux-env: Torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}")
print(f"native-linux-env: NumPy {np.__version__}, RoboStack OpenCV {cv2.__version__}, rclpy {rclpy.__file__}")
print(f"native-linux-env: Devkit bridge {autodrive_roboracer.autodrive_bridge.__file__}")
PY

unset _aicar_ros_abi _aicar_learner_abi _aicar_had_nounset _aicar_source_status
