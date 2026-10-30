#!/usr/bin/env python3
"""
AiCar — one command to start everything.

  python main.py

Builds missing/stale runtime images, starts Mission Control,
and opens its bundled web UI in the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HUB_URL = "http://127.0.0.1:8090"
HEALTH_URL = f"{HUB_URL}/health"
HOST_HUB_PID = ROOT / "logs" / "layer4" / "macos_host_hub.pid"
HOST_HUB_LOG = ROOT / "logs" / "layer4" / "macos_host_hub.log"
DEFAULT_SIMS = 2
SERVICE_IMAGES = {
    "brain": "aicar-brain:latest",
}

# GitHub release asset with gitignored Unity binaries (linux + windows).
SIM_RELEASE_TAG = "simulator-binaries"
SIM_ASSET_NAME = "autodrive-simulator.zip"
DEFAULT_SIM_DOWNLOAD = (
    f"https://github.com/FarrellJoswara/AutoDRIVE/releases/download/"
    f"{SIM_RELEASE_TAG}/{SIM_ASSET_NAME}"
)


def _banner(msg: str) -> None:
    print()
    print("=" * 60)
    print(msg)
    print("=" * 60)


def _find_docker() -> str:
    exe = shutil.which("docker")
    if exe:
        return exe
    # Docker Desktop on Windows often not on PATH in older shells
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Docker"
        / "Docker"
        / "resources"
        / "bin"
        / "docker.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs"
        / "DockerDesktop"
        / "resources"
        / "bin"
        / "docker.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    _banner("Docker not found")
    print("Install Docker Desktop, start it, then re-run:  python main.py")
    print("https://www.docker.com/products/docker-desktop/")
    sys.exit(1)


def _docker_ok(docker: str) -> None:
    try:
        subprocess.run(
            [docker, "info"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        _banner("Docker is installed but not running")
        print("Start Docker Desktop, wait until it says Running, then re-run:")
        print("  python main.py")
        sys.exit(1)


def _linux_sim_present() -> bool:
    return (ROOT / "simulator" / "AutoDRIVE Simulator.x86_64").is_file()


def _windows_sim_present() -> bool:
    return (ROOT / "simulator" / "windows" / "AutoDRIVE Simulator.exe").is_file()


def _macos_practice_sim_present() -> bool:
    candidates = [
        Path.home() / "Downloads" / "autodrive_simulator" / "AutoDRIVE Simulator.app",
        Path("/Applications") / "AutoDRIVE Simulator.app",
        ROOT / "simulator" / "macos" / "AutoDRIVE Simulator.app",
    ]
    cached = ROOT / "logs" / "layer4" / "official_simulator_path.txt"
    if cached.is_file():
        candidates.insert(0, Path(cached.read_text(encoding="utf-8").strip()))
    for candidate in candidates:
        binary = candidate / "Contents" / "MacOS" / "AutoDRIVE Simulator"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
        if binary.is_file() and os.access(binary, os.X_OK):
            return True
    return False


def _ensure_simulator(download_url: str, skip_download: bool) -> None:
    if sys.platform == "darwin" and _macos_practice_sim_present():
        print("Simulator: OK (native macOS AutoDRIVE practice .app)")
        return
    if _linux_sim_present():
        print("Simulator: OK (linux binary present for Docker)")
        return
    if skip_download:
        _banner("Simulator missing")
        print("Expected: simulator/AutoDRIVE Simulator.x86_64")
        print("Download the release zip and extract into ./simulator/")
        print(f"  {download_url}")
        sys.exit(1)

    _banner("Downloading AutoDRIVE simulator binaries")
    print(f"From: {download_url}")
    dest_zip = ROOT / "simulator" / SIM_ASSET_NAME
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(download_url, dest_zip)
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc}")
        print("Manual: download the zip from GitHub Releases and extract into ./simulator/")
        print(f"  {download_url}")
        sys.exit(1)

    print("Extracting…")
    import zipfile

    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(ROOT / "simulator")
    try:
        dest_zip.unlink()
    except OSError:
        pass

    if not _linux_sim_present():
        print("Zip extracted but linux binary still missing.")
        print("Check the zip layout — binaries should land under ./simulator/")
        sys.exit(1)
    print("Simulator: ready")


def _compose(docker: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [docker, "compose", *args]
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT), check=check)


def _image_exists(docker: str, image: str) -> bool:
    result = subprocess.run(
        [docker, "image", "inspect", image],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


OFFICIAL_API_IMAGE = os.environ.get("AICAR_OFFICIAL_API_IMAGE", "aicar-iros2026")
OFFICIAL_SIM_IMAGE = os.environ.get("AICAR_OFFICIAL_SIM_IMAGE", "autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete")


def _official_source_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "competition/iros2026/Dockerfile", ROOT / "competition/iros2026/run_container.sh"]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ensure_images(docker: str, *, force_build: bool, no_build: bool) -> None:
    brain_missing = not _image_exists(docker, SERVICE_IMAGES["brain"])
    api_missing = not _image_exists(docker, OFFICIAL_API_IMAGE)
    # macOS practice training launches AutoDRIVE Simulator.app on the host, so the
    # Linux/Windows Docker simulator image is optional on Darwin.
    sim_required = sys.platform != "darwin"
    sim_missing = sim_required and not _image_exists(docker, OFFICIAL_SIM_IMAGE)
    source_hash = _official_source_hash()
    image_hash = subprocess.run(
        [docker, "image", "inspect", "--format", '{{index .Config.Labels "aicar.source_hash"}}', OFFICIAL_API_IMAGE],
        capture_output=True, text=True, check=False,
    ).stdout.strip() if not api_missing else ""
    # Hub runs on the Mac host; API image staleness only matters for trainer code
    # baked into aicar-iros2026. Allow --no-build when the image already exists.
    api_stale = image_hash != source_hash
    if no_build and (api_missing or sim_missing or (sys.platform != "darwin" and (brain_missing or api_stale))):
        raise RuntimeError("Official runtime images are missing or do not match this checkout. Run without --no-build.")
    if sys.platform != "darwin" and (force_build or brain_missing):
        _compose(docker, "build", "brain")
    if force_build or (api_stale and sys.platform != "darwin") or api_missing:
        if api_missing or force_build or api_stale:
            subprocess.run([docker, "build", "-f", "competition/iros2026/Dockerfile", "-t", OFFICIAL_API_IMAGE,
                            "--label", f"aicar.source_hash={source_hash}", "."], cwd=ROOT, check=True)
    if sim_missing:
        subprocess.run([docker, "pull", OFFICIAL_SIM_IMAGE], check=True)
    if sys.platform == "darwin" and api_stale and not force_build:
        print(
            "Note: aicar-iros2026 source hash differs from this checkout; "
            "re-run without --no-build (or with --build) if trainer behavior looks stale."
        )


def _stop_managed_runs(docker: str) -> None:
    try:
        with urllib.request.urlopen(f"{HUB_URL}/train/runs", timeout=10) as response:
            runs = json.load(response)["runs"]
        for run in runs:
            if run.get("state") in {"starting", "running", "stopping"}:
                request = urllib.request.Request(f"{HUB_URL}/train/runs/{run['run_id']}/stop", method="POST")
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                if result.get("cleanup_error") or result.get("state") == "failed":
                    raise RuntimeError(f"Run cleanup did not complete: {result}")
    except (urllib.error.URLError, OSError):
        running = subprocess.run([docker, "ps", "-q", "--filter", "label=aicar.managed=official-run"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        if running:
            raise RuntimeError("Mission Control is unreachable and official runs still exist. Restore the hub and stop them before shutting down.")


def _ensure_web_ui() -> None:
    """Build the bind-mounted UI only when its source changed; otherwise reuse dist."""
    web_dir = ROOT / "src" / "layer4" / "web"
    dist_index = web_dir / "dist" / "index.html"
    npm = shutil.which("npm")
    if not npm:
        print("Web UI: using the bundle already packaged in the brain image (npm not found).")
        return

    source_files = [
        web_dir / "index.html",
        web_dir / "vite.config.ts",
        web_dir / "package.json",
        web_dir / "package-lock.json",
        *[p for p in (web_dir / "src").rglob("*") if p.is_file()],
    ]
    newest_source = max((p.stat().st_mtime for p in source_files if p.exists()), default=0)
    if dist_index.is_file() and dist_index.stat().st_mtime >= newest_source:
        print("Web UI: existing production bundle is current.")
        return

    node_modules = web_dir / "node_modules"
    install_marker = node_modules / ".package-lock.json"
    lock_file = web_dir / "package-lock.json"
    needs_install = (
        not node_modules.is_dir()
        or not install_marker.is_file()
        or (lock_file.is_file() and install_marker.stat().st_mtime < lock_file.stat().st_mtime)
    )
    if needs_install:
        _banner("Installing web UI dependencies (first run or lockfile changed)")
        subprocess.run(
            [npm, "ci", "--no-audit", "--no-fund", "--legacy-peer-deps"],
            cwd=str(web_dir),
            check=True,
            shell=(os.name == "nt"),
        )
    _banner("Building updated Mission Control web UI")
    subprocess.run(
        [npm, "run", "build"],
        cwd=str(web_dir),
        check=True,
        shell=(os.name == "nt"),
    )


def _wait_health(timeout_s: float = 300.0) -> None:
    print(f"Waiting for Mission Control at {HEALTH_URL} …")
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
                if resp.status == 200:
                    print("Mission Control is up.")
                    return
        except Exception as exc:  # noqa: BLE001 — any connect failure is retryable
            last_err = str(exc)
        time.sleep(2)
    _banner("Timed out waiting for Mission Control")
    print(last_err or "no response")
    if sys.platform == "darwin":
        print(f"Check: {HOST_HUB_LOG}")
    else:
        print("Check: docker compose logs brain")
    sys.exit(1)


def _host_python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _stop_host_hub() -> None:
    if not HOST_HUB_PID.is_file():
        return
    try:
        pid = int(HOST_HUB_PID.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        HOST_HUB_PID.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        pass
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    HOST_HUB_PID.unlink(missing_ok=True)


def _start_host_hub() -> None:
    """Run Mission Control on the Mac host so it can launch AutoDRIVE Simulator.app."""
    _stop_host_hub()
    # Free :8090 if an old aicar_brain hub is still bound.
    subprocess.run(
        ["docker", "rm", "-f", "aicar_brain"],
        cwd=str(ROOT),
        capture_output=True,
        check=False,
    )
    HOST_HUB_PID.parent.mkdir(parents=True, exist_ok=True)
    HOST_HUB_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AICAR_RUN_BACKEND"] = "docker"
    env["AICAR_SIMULATOR_BACKEND"] = "macos_external"
    env["PYTHONPATH"] = str(ROOT)
    env["HUB_URL"] = HUB_URL
    env["HUB_PORT"] = "8090"
    log = HOST_HUB_LOG.open("ab", buffering=0)
    process = subprocess.Popen(
        [
            _host_python(), "-m", "uvicorn",
            "src.layer4.hub.app:app",
            "--host", "127.0.0.1",
            "--port", "8090",
        ],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    HOST_HUB_PID.write_text(str(process.pid), encoding="ascii")
    print(f"Host Mission Control pid {process.pid} (log: {HOST_HUB_LOG})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start AiCar Mission Control (Docker brain + sims + UI).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Save the default official environment count for the next UI training run (1-8)",
    )
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "--build",
        action="store_true",
        help="Rebuild Mission Control and the official policy image",
    )
    build_group.add_argument(
        "--no-build",
        action="store_true",
        help="Never build images; exit with instructions if an image is missing",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser",
    )
    parser.add_argument(
        "--no-download-sim",
        action="store_true",
        help="Fail if simulator missing instead of downloading",
    )
    parser.add_argument(
        "--sim-url",
        default=os.environ.get("AICAR_SIM_URL", DEFAULT_SIM_DOWNLOAD),
        help="URL for simulator zip if missing locally",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the stack and exit",
    )
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    docker = _find_docker()
    _docker_ok(docker)

    if args.stop:
        _banner("Stopping AiCar")
        _stop_managed_runs(docker)
        if sys.platform == "darwin":
            _stop_host_hub()
        _compose(docker, "down", check=True)
        print("Stopped.")
        return 0

    if args.sims is not None:
        if not 1 <= args.sims <= 8:
            parser.error("--sims must be between 1 and 8")
        settings_path = ROOT / "logs/layer4/official_train_settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        settings["n_envs"] = args.sims
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if sys.platform == "darwin":
        settings_path = ROOT / "logs/layer4/official_train_settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        settings["simulator_backend"] = "macos_external"
        settings["observation_profile"] = settings.get("observation_profile") or "official_sensors"
        if settings.get("observation_profile") == "official_sensors_camera":
            settings["observation_profile"] = "official_sensors"
        settings["simulator_display_mode"] = settings.get("simulator_display_mode") or "offscreen"
        settings["automatic_evaluation_enabled"] = False
        settings["device"] = "cpu"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    _ensure_web_ui()
    if sys.platform != "darwin":
        _ensure_simulator(args.sim_url, args.no_download_sim)
    else:
        _ensure_simulator(args.sim_url, skip_download=True)
    _ensure_images(docker, force_build=args.build, no_build=args.no_build)

    if sys.platform == "darwin":
        _banner("Starting Mission Control on this Mac (native practice simulator)")
        print("ROS/Devkit trainer containers still use Docker; Unity runs as AutoDRIVE Simulator.app")
        _start_host_hub()
    else:
        _banner("Starting Mission Control (official simulators start with each run)")
        up_args = ["up", "-d", "--no-build", "brain"]
        _compose(docker, *up_args)

    _wait_health()

    _banner("OPEN THIS IN YOUR BROWSER")
    print(f"  {HUB_URL}")
    print()
    print("Pages: Train · Maps · Watch · Replay")
    if sys.platform == "darwin":
        print("Simulator: native macOS AutoDRIVE practice build")
    print("Stop later:  python main.py --stop")
    print("=" * 60)

    if not args.no_browser:
        try:
            webbrowser.open(HUB_URL)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
