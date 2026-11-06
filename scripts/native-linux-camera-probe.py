#!/usr/bin/env python3
"""Bounded camera/render probe for the official AutoDRIVE player; neutral only."""
import argparse
import base64
import io
import hashlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import socketio
from PIL import Image
from gevent import pywsgi, sleep as gevent_sleep
from geventwebsocket.handler import WebSocketHandler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=pathlib.Path, default=pathlib.Path(".native-linux/runtime"))
    ap.add_argument("--port", type=int, default=4567)
    ap.add_argument("--display", type=int, default=173)
    ap.add_argument("--seconds", type=float, default=35)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--report", type=pathlib.Path, help="write the JSON evidence report to this new path")
    ap.add_argument("--frame-output", type=pathlib.Path, help="write the first decoded camera image to this new path")
    ap.add_argument("--listen-only", action="store_true", help="capture camera from the external native-linux-simulator.sh wrapper")
    args = ap.parse_args()
    root = args.runtime.resolve() / "rootfs"
    sim = root / "home/autodrive_simulator/AutoDRIVE Simulator.x86_64"
    if not sim.is_file(): raise SystemExit(f"Official simulator absent: {sim}")
    repo = pathlib.Path(__file__).resolve().parents[1]
    hostroot = args.runtime.resolve().parent / "host-root"
    dummy = hostroot / "usr/lib/xorg/modules/drivers/dummy_drv.so"
    xorg = pathlib.Path("/usr/lib/xorg/Xorg")
    config = args.runtime.resolve().parent / "dummy-xorg.conf"
    default_report = args.runtime.resolve() / "camera-probe.json"
    report_path = args.report.expanduser().resolve() if args.report else default_report
    if report_path.exists():
        try:
            existing_report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            existing_report = {}
        if args.report or existing_report.get("probe_passed") is True:
            raise SystemExit(f"Refusing to overwrite existing probe report: {report_path}")
    default_frame = (report_path.with_name(report_path.stem + "-frame.jpg") if args.report
                     else args.runtime.resolve() / "camera-frame.jpg")
    frame_path = args.frame_output.expanduser().resolve() if args.frame_output else default_frame
    if frame_path.exists():
        if args.frame_output or args.report:
            raise SystemExit(f"Refusing to overwrite existing camera frame: {frame_path}")
        frame_path = frame_path.with_name(f"{frame_path.stem}-{os.getpid()}{frame_path.suffix}")
    sim_log = args.runtime.resolve() / "camera-simulator.log"
    xorg_log = args.runtime.resolve() / "camera-xorg.log"
    display = f":{args.display}"
    if not args.listen_only and (not dummy.is_file() or not xorg.is_file() or not config.is_file()):
        raise SystemExit("Missing user-space dummy Xorg driver/config; run native-linux-bootstrap.sh")
    if not args.listen_only and pathlib.Path(f"/tmp/.X11-unix/X{args.display}").exists():
        raise SystemExit(f"Display {display} is already occupied; select another --display")
    with socket.socket() as check:
        if check.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"TCP port {args.port} is already occupied")

    records, errors, socket_connections, socket_disconnects = [], [], [], []
    sio = socketio.Server(async_mode="gevent", cors_allowed_origins="*")
    original = sio._handle_event
    def auto_connect(eio_sid, namespace, event_id, data):
        ns = namespace or "/"
        sid = sio.manager.sid_from_eio_sid(eio_sid, ns)
        if sid is None or not sio.manager.is_connected(sid, ns):
            sio._handle_connect(eio_sid, ns, None)
        return original(eio_sid, namespace, event_id, data)
    sio._handle_event = auto_connect

    @sio.on("connect")
    def socket_connected(sid, environ, auth=None):
        socket_connections.append({"sid": sid, "query_string": environ.get("QUERY_STRING"),
                                   "request_uri": environ.get("RAW_URI"),
                                   "user_agent": environ.get("HTTP_USER_AGENT")})

    @sio.on("disconnect")
    def socket_disconnected(sid, reason=None):
        socket_disconnects.append({"sid": sid, "reason": str(reason) if reason is not None else None})

    @sio.on("Bridge")
    def bridge(sid, payload):
        try:
            field = payload.get("V1 Front Camera Image") if isinstance(payload, dict) else None
            entry = {"at_monotonic": time.monotonic(), "payload_keys": sorted(payload) if isinstance(payload, dict) else [],
                     "camera_present": field is not None, "camera_type": type(field).__name__ if field is not None else None}
            if field is not None:
                raw = json.dumps(field, separators=(",", ":"), ensure_ascii=True).encode()
                entry["camera_raw_json_bytes"] = len(raw)
                entry["camera_raw_sha256"] = hashlib.sha256(raw).hexdigest()
                try:
                    if isinstance(field, str):
                        encoded = field.split(",", 1)[1] if field.startswith("data:") and "," in field else field
                        image_bytes = base64.b64decode(encoded, validate=False)
                        entry["camera_encoded_bytes"] = len(image_bytes)
                        entry["camera_encoded_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                        if not any(x.get("camera_present") for x in records):
                            frame_path.parent.mkdir(parents=True, exist_ok=True)
                            frame_path.write_bytes(image_bytes)
                        with Image.open(io.BytesIO(image_bytes)) as im:
                            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                    else:
                        arr = np.asarray(field, dtype=np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        entry.update({"shape": list(arr.shape), "dtype": str(arr.dtype),
                                      "min": int(arr.min()), "max": int(arr.max()),
                                      "mean": float(arr.mean()),
                                      "nonblack_pixel_fraction": float(np.any(arr != 0, axis=-1).mean()),
                                      "rgb_sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
                except (TypeError, ValueError, OverflowError) as exc:
                    entry["decode_error"] = str(exc)
            records.append(entry)
            # Official Bridge response; neutral throttle/steering, never reset.
            sio.emit("Bridge", data={"V1 Throttle":"0.0", "V1 Steering":"0.0", "V1 Reset":"False"}, to=sid)
        except Exception as exc:
            errors.append(repr(exc))
    app = socketio.WSGIApp(sio)
    server = pywsgi.WSGIServer(("127.0.0.1", args.port), app, handler_class=WebSocketHandler, log=None, error_log=None)
    xproc = simproc = None
    gl = None
    result = {"image_identity": str(args.runtime.resolve()/"IMAGE-IDENTITY.json"), "display": display,
              "bridge_host": "127.0.0.1", "bridge_port": args.port,
              "neutral_reply": {"V1 Throttle":"0.0", "V1 Steering":"0.0", "V1 Reset":"False"},
              "camera_frames": records, "listener_errors": errors,
              "socket_connections": socket_connections, "socket_disconnects": socket_disconnects}
    try:
        if args.listen_only:
            server.start()
            print(f"Neutral Socket.IO camera listener ready on 127.0.0.1:{args.port}", flush=True)
            end = time.monotonic() + args.seconds
            while time.monotonic() < end and sum(bool(x.get("shape")) for x in records) < args.frames:
                gevent_sleep(.1)
        else:
            xproc = subprocess.Popen([str(xorg), display, "-config", str(config), "-modulepath",
            f"{hostroot}/usr/lib/xorg/modules,/usr/lib/xorg/modules", "-noreset", "-nolisten", "tcp",
            "-logfile", str(xorg_log)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            xready_deadline = time.monotonic() + 12
            while time.monotonic() < xready_deadline:
                if xproc.poll() is not None: raise RuntimeError(f"Xorg exited {xproc.returncode}; see {xorg_log}")
                probe = subprocess.run(["xdpyinfo", "-display", display], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if probe.returncode == 0: break
                time.sleep(.15)
            else: raise TimeoutError("Xorg did not answer an X11 client within 12 seconds")
            gl = subprocess.run(["glxinfo", "-display", display, "-B"], capture_output=True, text=True, timeout=8)
            result["glxinfo"] = gl.stdout
            if gl.returncode != 0 or "direct rendering: Yes" not in gl.stdout:
                raise RuntimeError(f"GLX direct-rendering check failed: {gl.stderr[:500]}")
            server.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.2): break
                except OSError: time.sleep(.05)
            else: raise TimeoutError("Socket.IO listener did not bind")
            libs = f"/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/nvidia/xorg:{root}/usr/lib/x86_64-linux-gnu:{root}/usr/lib"
            nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
            env = os.environ.copy(); env.update({"DISPLAY":display,
                "LD_LIBRARY_PATH":libs, "VK_ICD_FILENAMES":nvidia_icd, "VK_DRIVER_FILES":nvidia_icd,
                "LIBGL_DRIVERS_PATH":str(root/"usr/lib/x86_64-linux-gnu/dri")})
            # Run the normal GUI loop so the official front-camera sensor renders.
            simproc = subprocess.Popen([str(sim), "-force-vulkan", "-ip", "127.0.0.1", "-port", str(args.port), "-logFile", str(sim_log)],
                cwd=sim.parent, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
            result["simulator_pid"] = simproc.pid
            end = time.monotonic() + args.seconds
            while time.monotonic() < end and sum(bool(x.get("shape")) for x in records) < args.frames:
                if simproc.poll() is not None: raise RuntimeError(f"official simulator exited {simproc.returncode}; see {sim_log}")
                gevent_sleep(.1)
        result["frame_count"] = sum(bool(x.get("shape")) for x in records)
        result["bridge_event_count"] = len(records)
        result["simulator_returncode_before_cleanup"] = simproc.poll() if simproc is not None else None
        result["xorg_renderer"] = next((line.strip() for line in gl.stdout.splitlines() if "Device:" in line), None) if gl else None
        result["camera_hashes_distinct"] = len({x.get("rgb_sha256", x.get("camera_raw_sha256")) for x in records if x.get("shape")})
        result["camera_frames_nonblack"] = sum(1 for x in records if x.get("shape") and x.get("nonblack_pixel_fraction", 0) > 0)
        result["frame_output"] = str(frame_path) if frame_path.exists() else None
        result["probe_passed"] = (result["frame_count"] >= args.frames and result["camera_hashes_distinct"] >= 2
                                   and result["camera_frames_nonblack"] >= 1 and not errors)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["frame_count"] = sum(bool(x.get("shape")) for x in records)
        result["bridge_event_count"] = len(records)
        result["camera_hashes_distinct"] = len({x.get("rgb_sha256", x.get("camera_raw_sha256")) for x in records if x.get("shape")})
        result["camera_frames_nonblack"] = sum(1 for x in records if x.get("shape") and x.get("nonblack_pixel_fraction", 0) > 0)
        result["frame_output"] = str(frame_path) if frame_path.exists() else None
        result["probe_passed"] = False
    finally:
        if simproc is not None and simproc.poll() is None:
            simproc.send_signal(signal.SIGTERM)
            try: simproc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                simproc.kill(); simproc.wait(timeout=2)
        server.stop(timeout=1)
        if xproc is not None and xproc.poll() is None:
            xproc.send_signal(signal.SIGTERM)
            try: xproc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xproc.kill(); xproc.wait(timeout=2)
        report_path.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("camera_frames", "glxinfo")}, indent=2))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Full report: {report_path}\nFrame image: {frame_path if frame_path.exists() else 'not captured'}\nSimulator log: {sim_log}\nXorg log: {xorg_log}")
    return 0 if result.get("probe_passed") else 1

if __name__ == "__main__": raise SystemExit(main())
