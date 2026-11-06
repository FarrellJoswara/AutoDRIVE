#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${AICAR_NATIVE_STATE:-$ROOT/.native-linux}"
RELEASE="${AICAR_NATIVE_RELEASE:-practice}"
while (($#)); do
  case "$1" in
    --release) [[ $# -ge 2 ]] || { echo "--release requires practice or competition" >&2; exit 2; }; RELEASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$RELEASE" in
  practice) DEFAULT_RUNTIME="$STATE/runtime" ;;
  competition) DEFAULT_RUNTIME="$STATE/competition-runtime" ;;
  *) echo "Unsupported native Linux official release: $RELEASE" >&2; exit 2 ;;
esac
RUNTIME="${AICAR_NATIVE_RUNTIME:-$DEFAULT_RUNTIME}"
HOST_ROOT="$STATE/host-root"
mkdir -p "$STATE/debs" "$HOST_ROOT"

python3 "$ROOT/scripts/native-linux-export.py" --release "$RELEASE" \
  --blob-cache "$STATE/runtime/blobs" "$RUNTIME"

# Extract the Ubuntu dummy Xorg driver into user space. This does not run dpkg
# or install packages on the host; apt-get download only fetches the .deb.
command -v apt-get >/dev/null || { echo "apt-get is required to fetch user-space graphics utilities" >&2; exit 2; }
(cd "$STATE/debs" && apt-get download xserver-xorg-video-dummy xdotool libxdo3)
while IFS= read -r deb; do dpkg-deb -x "$deb" "$HOST_ROOT"; done < <(find "$STATE/debs" -maxdepth 1 -type f -name '*.deb' -print)
python3 - "$STATE/debs" "$STATE/HOST-DEPS.json" <<'PY'
import hashlib,json,pathlib,subprocess,sys
items=[]
for deb in sorted(pathlib.Path(sys.argv[1]).glob('*.deb')):
 fields=subprocess.check_output(['dpkg-deb','-f',str(deb),'Package','Version','Architecture'],text=True).splitlines()
 items.append({'package':fields[0],'version':fields[1],'architecture':fields[2],'deb':deb.name,'deb_sha256':hashlib.sha256(deb.read_bytes()).hexdigest()})
pathlib.Path(sys.argv[2]).write_text(json.dumps({'packages':items},indent=2)+'\n')
PY
[[ -f "$HOST_ROOT/usr/lib/xorg/modules/drivers/dummy_drv.so" ]] || { echo "dummy Xorg driver extraction failed" >&2; exit 2; }
[[ -x "$HOST_ROOT/usr/bin/xdotool" && -f "$HOST_ROOT/usr/lib/x86_64-linux-gnu/libxdo.so.3" ]] || { echo "xdotool user-space extraction failed" >&2; exit 2; }

for tool in /usr/lib/xorg/Xorg /usr/bin/xdpyinfo /usr/bin/vulkaninfo; do
  [[ -x "$tool" ]] || { echo "Required host graphics tool is missing: $tool" >&2; exit 2; }
done
cat > "$STATE/dummy-xorg.conf" <<'EOF'
Section "ServerLayout"
    Identifier "NativeDummy"
    Screen 0 "Screen0"
EndSection
Section "Device"
    Identifier "Dummy0"
    Driver "dummy"
    VideoRam 256000
EndSection
Section "Monitor"
    Identifier "Monitor0"
    HorizSync 10.0-200.0
    VertRefresh 10.0-200.0
    Modeline "1280x720" 74.50 1280 1344 1472 1664 720 723 728 748
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Dummy0"
    Monitor "Monitor0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1280x720"
    EndSubSection
EndSection
EOF

echo "Native runtime prepared: $RUNTIME"
echo "Official simulator: $RUNTIME/rootfs/home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
echo "User-space Xorg dummy driver: $HOST_ROOT/usr/lib/xorg/modules/drivers/dummy_drv.so"
echo "No host packages or services were installed or changed."
