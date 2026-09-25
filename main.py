#!/usr/bin/env python3
"""
AiCar — one command to start everything.

  python main.py

Starts Docker (brain + sims), waits for Mission Control, opens the browser.
"""

from __future__ import annotations

import argparse
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
DEFAULT_SIMS = 2

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


def _ensure_simulator(download_url: str, skip_download: bool) -> None:
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
    print("Check: docker compose logs brain")
    sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start AiCar Mission Control (Docker brain + sims + UI).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=DEFAULT_SIMS,
        help=f"Number of sim containers (default {DEFAULT_SIMS})",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip --build (faster if images already exist)",
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
        _compose(docker, "down", check=False)
        print("Stopped.")
        return 0

    _ensure_simulator(args.sim_url, skip_download=args.no_download_sim)

    _banner("Starting AiCar (brain + sims + Mission Control)")
    up_args = ["up", "-d"]
    if not args.no_build:
        up_args.append("--build")
    up_args.extend(["--scale", f"sim={max(1, args.sims)}"])
    _compose(docker, *up_args)

    _wait_health()

    _banner("OPEN THIS IN YOUR BROWSER")
    print(f"  {HUB_URL}")
    print()
    print("Pages: Settings · Train · Live · Fleet")
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
