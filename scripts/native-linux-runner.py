#!/usr/bin/env python3
"""Supervise the official Vulkan player on a user-space Xorg dummy display."""
import hashlib
import ctypes
import json
import os
import pathlib
import select
import signal
import socket
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STATE = pathlib.Path(os.environ.get("AICAR_NATIVE_STATE", REPO_ROOT / ".native-linux")).resolve()
RELEASES = {
    "practice": ("2026-iros-practice", "sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76"),
    "competition": ("2026-iros-compete", "sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d"),
}
RELEASE = os.environ.get("AICAR_NATIVE_RELEASE", "practice")
if RELEASE not in RELEASES:
    raise RuntimeError(f"Unsupported native Linux official release: {RELEASE!r}")
DEFAULT_RUNTIME = STATE / ("runtime" if RELEASE == "practice" else "competition-runtime")
RUNTIME = pathlib.Path(os.environ.get("AICAR_NATIVE_RUNTIME", DEFAULT_RUNTIME)).resolve()
ROOTFS = RUNTIME / "rootfs"
SIM = pathlib.Path(os.environ.get("AICAR_SIMULATOR_PATH", ROOTFS / "home/autodrive_simulator/AutoDRIVE Simulator.x86_64")).resolve()
HOST_ROOT = STATE / "host-root"
HOST = os.environ.get("BRAIN_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4567"))
WAIT_S = float(os.environ.get("SIM_BRIDGE_WAIT_S", "30"))
NVIDIA_ICD = pathlib.Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
stop_requested = False


def stop_signal(_signum, _frame):
    global stop_requested
    stop_requested = True


def require_identity(release=RELEASE, runtime=RUNTIME, simulator=SIM):
    rootfs = pathlib.Path(runtime) / "rootfs"
    simulator = pathlib.Path(simulator)
    identity = json.loads((pathlib.Path(runtime) / "IMAGE-IDENTITY.json").read_text())
    expected_tag, expected_digest = RELEASES[release]
    if (identity.get("repository") != "autodriveecosystem/autodrive_roboracer_sim"
            or identity.get("tag") != expected_tag or identity.get("digest") != expected_digest
            or identity.get("manifest_verified") is not True
            or identity.get("architecture") != "linux/amd64"
            or not identity.get("rootfs_tree_digest")):
        raise RuntimeError(f"Runtime identity is not the pinned official linux/amd64 {release} image")
    try:
        complete = json.loads((rootfs / ".OCI-COMPLETE.json").read_text())
    except (OSError, ValueError):
        raise RuntimeError("Runtime rootfs has no completed OCI extraction marker; rerun native-linux-bootstrap.sh")
    if (complete.get("tag", expected_tag) != expected_tag or complete.get("digest") != expected_digest
            or not isinstance(complete.get("verified_layers"), int) or complete["verified_layers"] <= 0):
        raise RuntimeError("Runtime OCI completion marker does not match the selected pinned release")
    if not simulator.is_file() or not os.access(simulator, os.X_OK):
        raise RuntimeError(f"Official simulator is absent or not executable: {simulator}")
    actual = hashlib.sha256(simulator.read_bytes()).hexdigest()
    if identity.get("simulator_sha256") != actual:
        raise RuntimeError("Simulator executable hash differs from the verified OCI identity")
    data = simulator.parent / "Data"
    for filename, key in (("level0", "level0_sha256"), ("globalgamemanagers", "globalgamemanagers_sha256")):
        path = data / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != identity.get(key):
            raise RuntimeError(f"Official scene/build asset hash check failed: {path}")


def wait_for_tcp():
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline and not stop_requested:
        try:
            with socket.create_connection((HOST, PORT), timeout=.4):
                print(f"Bridge TCP listener ready at {HOST}:{PORT}", flush=True)
                return
        except OSError:
            time.sleep(.2)
    raise TimeoutError(f"Bridge listener {HOST}:{PORT} did not become ready within {WAIT_S:g}s")


def start_xorg(log_path):
    xorg = pathlib.Path("/usr/lib/xorg/Xorg")
    dummy = HOST_ROOT / "usr/lib/xorg/modules/drivers/dummy_drv.so"
    config = STATE / "dummy-xorg.conf"
    if not all(x.is_file() for x in (xorg, dummy, config)):
        raise RuntimeError("Xorg/dummy driver/config missing; run scripts/native-linux-bootstrap.sh")
    read_fd, write_fd = os.pipe()
    log = log_path.open("wb")
    proc = subprocess.Popen([str(xorg), "-displayfd", str(write_fd), "-config", str(config),
        "-modulepath", f"{HOST_ROOT}/usr/lib/xorg/modules,/usr/lib/xorg/modules",
        "-noreset", "-nolisten", "tcp", "-logfile", str(log_path)],
        pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=log, start_new_session=True)
    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], 15)
        if not ready:
            raise TimeoutError(f"Xorg did not allocate a display; see {log_path}")
        display_number = os.read(read_fd, 32).decode("ascii").strip()
        if not display_number.isdigit():
            raise RuntimeError(f"Xorg returned invalid display number {display_number!r}")
    except Exception:
        proc.terminate()
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        raise
    finally:
        os.close(read_fd)
        log.close()
    return proc, f":{display_number}"


def preflight_display(display, log_dir):
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and not stop_requested:
        check = subprocess.run(["/usr/bin/xdpyinfo", "-display", display], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0:
            break
        time.sleep(.15)
    else:
        raise TimeoutError(f"No X11 client could connect to {display}")
    env = os.environ.copy(); env["DISPLAY"] = display
    env["VK_ICD_FILENAMES"] = str(NVIDIA_ICD); env["VK_DRIVER_FILES"] = str(NVIDIA_ICD)
    vk = subprocess.run(["/usr/bin/vulkaninfo", "--text"], env=env, capture_output=True, text=True, timeout=20)
    (log_dir / "vulkaninfo.txt").write_text(vk.stdout + vk.stderr)
    if vk.returncode or not any(line.strip() == "deviceName        = NVIDIA L40" for line in vk.stdout.splitlines()):
        raise RuntimeError(f"NVIDIA Vulkan device/WSI preflight failed; see {log_dir / 'vulkaninfo.txt'}")
    if not any(line.strip().startswith("present support") and line.strip().endswith("= true")
               for line in vk.stdout.splitlines()):
        raise RuntimeError(f"NVIDIA Vulkan has no present-capable queue for {display}; see {log_dir / 'vulkaninfo.txt'}")
    print("X11 client, NVIDIA L40 Vulkan, and Vulkan presentation checks passed", flush=True)


def focus_x11_window(display, window_id):
    """Set and verify X input focus directly; this display has no window manager."""
    x11 = ctypes.CDLL("libX11.so.6")
    x11.XOpenDisplay.argtypes = (ctypes.c_char_p,)
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XSetInputFocus.argtypes = (ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    x11.XSync.argtypes = (ctypes.c_void_p, ctypes.c_int)
    x11.XGetInputFocus.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int))
    x11.XCloseDisplay.argtypes = (ctypes.c_void_p,)
    connection = x11.XOpenDisplay(display.encode("ascii"))
    if not connection:
        raise RuntimeError(f"Could not open X display {display} to focus the Unity window")
    try:
        x11.XSetInputFocus(connection, int(window_id), 2, 0)  # RevertToParent, CurrentTime
        x11.XSync(connection, 0)
        focused = ctypes.c_ulong()
        revert = ctypes.c_int()
        x11.XGetInputFocus(connection, ctypes.byref(focused), ctypes.byref(revert))
        if focused.value != int(window_id):
            raise RuntimeError(f"X input focus is {focused.value}, expected Unity window {window_id}")
    finally:
        x11.XCloseDisplay(connection)
    return str(focused.value)


def auto_connect(window_log, display):
    xdotool = HOST_ROOT / "usr/bin/xdotool"
    xdo_lib = HOST_ROOT / "usr/lib/x86_64-linux-gnu"
    if not xdotool.is_file():
        raise RuntimeError("User-space xdotool missing; rerun native-linux-bootstrap.sh")
    env = os.environ.copy(); env["DISPLAY"] = display
    env["LD_LIBRARY_PATH"] = str(xdo_lib) + ":/usr/lib/x86_64-linux-gnu"
    deadline = time.monotonic() + 25
    window = None
    visible_windows = []
    while time.monotonic() < deadline and not stop_requested:
        result = subprocess.run([str(xdotool), "search", "--onlyvisible", "--name", "AutoDRIVE Simulator"], env=env, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            window = result.stdout.strip().splitlines()[-1]
            break
        all_windows = subprocess.run([str(xdotool), "search", "--onlyvisible", "--name", ".*"], env=env, capture_output=True, text=True)
        if all_windows.returncode == 0:
            visible_windows = all_windows.stdout.splitlines()
            if visible_windows:
                # Unity's title can be delayed or exposed under its executable
                # class, while the GUI window is already visible.
                window = visible_windows[-1]
                break
        time.sleep(.2)
    if window is None:
        raise TimeoutError(f"Official player window did not appear for its Connect control; visible window ids={visible_windows}")
    # Unity creates its top-level X window before the menu canvas is ready to
    # receive input. The manual-probe screenshots showed the panel only after
    # the initial render had settled; don't send clicks during that startup gap.
    time.sleep(float(os.environ.get("AICAR_NATIVE_UI_READY_WAIT_S", "15")))
    geometry = subprocess.run([str(xdotool), "getwindowgeometry", "--shell", window], env=env,
                               capture_output=True, text=True, check=True).stdout
    dimensions = {line.split("=", 1)[0]: int(line.split("=", 1)[1])
                  for line in geometry.splitlines() if "=" in line and line.split("=", 1)[1].isdigit()}
    scale_x = dimensions.get("WIDTH", 1280) / 1280
    scale_y = dimensions.get("HEIGHT", 720) / 720
    def click(x, y, hold_s=0):
        x = round(x * scale_x); y = round(y * scale_y)
        subprocess.run([str(xdotool), "mousemove", "--sync", str(x), str(y)],
                       env=env, check=True, stdout=subprocess.DEVNULL)
        location = subprocess.run([str(xdotool), "getmouselocation", "--shell"], env=env,
                                  capture_output=True, text=True, check=True).stdout
        pointer = {line.split("=", 1)[0]: line.split("=", 1)[1]
                   for line in location.splitlines() if "=" in line}
        target = pointer.get("WINDOW", window)
        focused = focus_x11_window(display, target)
        if hold_s:
            # Keep the button down across Unity frames, then release separately.
            subprocess.run([str(xdotool), "mousedown", "1"],
                           env=env, check=True, stdout=subprocess.DEVNULL)
            time.sleep(hold_s)
            subprocess.run([str(xdotool), "mouseup", "1"], env=env, check=True, stdout=subprocess.DEVNULL)
        else:
            subprocess.run([str(xdotool), "click", "1"],
                           env=env, check=True, stdout=subprocess.DEVNULL)
        time.sleep(.2)
        return focused, pointer
    # Official CLI arguments prefill these inputs (confirmed in the live panel).
    # Leave their values intact; open the panel and press its plug icon.
    menu_focus, _ = click(75, 42); time.sleep(1.25)
    button_focus, pointer = click(75, 407, hold_s=.2)
    pointer_text = ";".join(f"{key}={value}" for key, value in pointer.items())
    window_log.write_text(f"window_id={window}\nmenu_input_focus_id={menu_focus}\nconnect_input_focus_id={button_focus}\ndisplay={display}\nwindow_geometry={dimensions}\nbridge={HOST}:{PORT}\npointer_after_click={pointer_text}\nconnect_icon_clicked=true\n")
    print(f"Clicked the official simulator Connect icon for {HOST}:{PORT}; verifying its TCP session", flush=True)


def wait_for_player_connection(simproc):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not stop_requested:
        result = subprocess.run(["/usr/bin/ss", "-Htnp", "state", "established"],
                                capture_output=True, text=True, check=False)
        for line in result.stdout.splitlines():
            columns = line.split()
            if (len(columns) >= 4 and columns[3].rsplit(":", 1)[-1] == str(PORT)
                    and f"pid={simproc.pid}," in line):
                print(f"Official simulator TCP connection verified to {HOST}:{PORT}", flush=True)
                return
        if simproc.poll() is not None:
            raise RuntimeError(f"Official simulator exited {simproc.returncode} before connecting")
        time.sleep(.2)
    raise TimeoutError(f"Connect icon was clicked, but simulator pid={simproc.pid} did not establish TCP to {HOST}:{PORT}")


def terminate(proc, label, process_group=True):
    if proc is None or proc.poll() is not None:
        return
    try:
        (os.killpg(proc.pid, signal.SIGTERM) if process_group else proc.send_signal(signal.SIGTERM))
        proc.wait(timeout=6)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            (os.killpg(proc.pid, signal.SIGKILL) if process_group else proc.kill())
        except ProcessLookupError:
            pass
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired: print(f"Warning: {label} did not exit after SIGKILL", file=sys.stderr)


def main():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    logs = RUNTIME / "manager-launch"
    logs.mkdir(parents=True, exist_ok=True)
    require_identity()
    wait_for_tcp()
    xproc = simproc = None
    old_handlers = {sig:signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        xproc, display = start_xorg(logs / "xorg.log")
        preflight_display(display, logs)
        env = os.environ.copy()
        env.update({"DISPLAY":display,
            "VK_ICD_FILENAMES":str(NVIDIA_ICD), "VK_DRIVER_FILES":str(NVIDIA_ICD),
            "LD_LIBRARY_PATH":f"/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/nvidia/xorg:{ROOTFS}/usr/lib/x86_64-linux-gnu:{ROOTFS}/usr/lib"})
        sim_log = logs / "official-simulator.log"
        out = sim_log.open("wb")
        simproc = subprocess.Popen([str(SIM), "-force-vulkan", "-ip", HOST, "-port", str(PORT),
            "-logFile", str(sim_log)], cwd=SIM.parent, env=env, stdout=out, stderr=subprocess.STDOUT)
        out.close()
        print(f"Started pinned official player pid={simproc.pid} display={display} backend=Vulkan/NVIDIA L40", flush=True)
        if os.environ.get("AICAR_NATIVE_AUTOCONNECT", "1") == "1":
            auto_connect(logs / "connect.txt", display)
            wait_for_player_connection(simproc)
        else:
            print("AICAR_NATIVE_AUTOCONNECT=0: click Connect in the simulator window", flush=True)
        while not stop_requested:
            code = simproc.poll()
            if code is not None:
                raise RuntimeError(f"Official simulator exited {code}; see {sim_log}")
            time.sleep(.2)
    finally:
        terminate(simproc, "official simulator", process_group=False)
        terminate(xproc, "Xorg")
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)
        print("Stopped official simulator and its Xorg display", flush=True)

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(f"native-linux runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
