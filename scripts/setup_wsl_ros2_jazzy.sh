#!/usr/bin/env bash
set -euo pipefail

if [[ "$(. /etc/os-release && echo "$VERSION_ID")" != "24.04" ]]; then
  echo "This setup requires Ubuntu 24.04 (ROS 2 Jazzy)." >&2
  exit 2
fi

apt-get update
apt-get install -y --no-install-recommends \
  curl gnupg lsb-release software-properties-common git python3-pip python3-venv
add-apt-repository -y universe

ros_apt_version="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')"
ubuntu_codename="$(. /etc/os-release && echo "$UBUNTU_CODENAME")"
ros_apt_package="ros2-apt-source_${ros_apt_version}.${ubuntu_codename}_all.deb"
curl -fsSL \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ros_apt_version}/${ros_apt_package}" \
  -o "/tmp/${ros_apt_package}"
dpkg -i "/tmp/${ros_apt_package}"
apt-get update
apt-get install -y --no-install-recommends \
  ros-jazzy-ros-base ros-jazzy-cv-bridge ros-jazzy-tf-transformations \
  python3-colcon-common-extensions python3-rosdep python3-gevent \
  python3-gevent-websocket python3-pil python3-opencv
python3 -m pip install --break-system-packages \
  attrdict "python-socketio[client]" python-engineio gevent-websocket

# ROS setup scripts read optional variables that are unset under nounset.
set +u
source /opt/ros/jazzy/setup.bash
set -u
rosdep init 2>/dev/null || true
runuser -u "${SUDO_USER:-farrell}" -- rosdep update

workspace="/home/${SUDO_USER:-farrell}/autodrive_ws"
install -d -o "${SUDO_USER:-farrell}" -g "${SUDO_USER:-farrell}" "${workspace}/src"
if [[ ! -d "${workspace}/src/AutoDRIVE-RoboRacer-Sim-Racing/.git" ]]; then
  runuser -u "${SUDO_USER:-farrell}" -- git clone \
    https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing.git \
    "${workspace}/src/AutoDRIVE-RoboRacer-Sim-Racing"
fi

# The official bridge defaults every instance to TCP 4567. Native parallel
# training runs one official bridge per ROS domain, so give each process its
# own Socket.IO port without changing its topics or payload handling.
bridge="${workspace}/src/AutoDRIVE-RoboRacer-Sim-Racing/autodrive_devkit/autodrive_roboracer/autodrive_bridge.py"
python3 - "$bridge" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
anchor = "import base64 # Base64 binary-to-text encoding/decoding scheme"
if "AICAR_DEVKIT_PORT" not in text:
    if anchor not in text:
        raise SystemExit(f"Cannot find official bridge import anchor in {path}")
    text = text.replace(anchor, anchor + "\nimport os # Native parallel-run port override", 1)
old = "pywsgi.WSGIServer(('', 4567), app, handler_class=WebSocketHandler).serve_forever()"
new = "pywsgi.WSGIServer(('', int(os.environ.get('AICAR_DEVKIT_PORT', '4567'))), app, handler_class=WebSocketHandler).serve_forever()"
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit(f"Cannot find official bridge server binding in {path}")
path.write_text(text, encoding="utf-8")
PY

cd "$workspace"
chown -R "${SUDO_USER:-farrell}:${SUDO_USER:-farrell}" "$workspace"
runuser -u "${SUDO_USER:-farrell}" -- bash -lc 'set +u; source /opt/ros/jazzy/setup.bash; set -u; cd "$HOME/autodrive_ws"; rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy; colcon build --symlink-install'

echo "ROS 2 Jazzy and the official AutoDRIVE RoboRacer Devkit are ready at ${workspace}."
