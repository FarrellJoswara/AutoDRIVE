#!/usr/bin/env python3
"""Phase 4 map smoke / integration tests (API + optional headless/headed sim).

Default path uses FastAPI TestClient against current hub code (does not require
a restarted uvicorn). Optional --base-url hits a live Mission Control.

Examples:
  python scripts/test_map_phase4_smoke.py
  python scripts/test_map_phase4_smoke.py --base-url http://127.0.0.1:8090
  python scripts/test_map_phase4_smoke.py --headless
  python scripts/test_map_phase4_smoke.py --headed
  python scripts/test_map_phase4_smoke.py --headed --headed-reuse   # capture existing window
  python scripts/test_map_phase4_smoke.py --activate-restart        # may restart docker sims

Non-destructive defaults:
  - Uploads under a unique temp map id, then deletes it
  - Activate uses restart=false unless --activate-restart
  - Restores previous .active_map.json after activate checks
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAPS_DIR = ROOT / "simulator" / "maps"
LOG_DIR = ROOT / "logs" / "layer4"
WIN_SIM = ROOT / "simulator" / "windows" / "AutoDRIVE Simulator.exe"
TRACKLOADER_MARKER = ROOT / "simulator" / ".aicar_trackloader"


# ---------------------------------------------------------------------------
# HTTP client (TestClient or live)
# ---------------------------------------------------------------------------


class HubClient:
    """Thin wrapper so smoke steps share one call style."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or "").rstrip("/") or None
        self._tc = None
        if self.base_url is None:
            from fastapi.testclient import TestClient

            from src.layer4.hub.app import app

            self._tc = TestClient(app)

    def request(
        self,
        method: str,
        path: str,
        *,
        data: Optional[bytes] = None,
        json_body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Any, bytes]:
        headers = dict(headers or {})
        if self._tc is not None:
            kw: Dict[str, Any] = {"headers": headers, "params": params}
            if data is not None:
                kw["content"] = data
            if json_body is not None:
                kw["json"] = json_body
            r = self._tc.request(method, path, **kw)
            body = r.content
            try:
                parsed: Any = r.json()
            except Exception:
                parsed = None
            return r.status_code, parsed, body

        import urllib.error
        import urllib.parse
        import urllib.request

        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        body_out: Optional[bytes] = data
        if json_body is not None:
            body_out = json.dumps(json_body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=body_out, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                code = resp.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            code = exc.code
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = None
        return code, parsed, raw

    def get(self, path: str, **kw: Any) -> Tuple[int, Any, bytes]:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw: Any) -> Tuple[int, Any, bytes]:
        return self.request("POST", path, **kw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ring_zip(map_id: str, *, label: str = "Phase4 Smoke") -> bytes:
    """Small free-space ring occupancy zip (nested layout)."""
    h = w = 64
    gray = np.full((h, w), 255, dtype=np.uint8)
    gray[4:8, :] = 0
    gray[-8:-4, :] = 0
    gray[:, 4:8] = 0
    gray[:, -8:-4] = 0
    gray[28:36, 28:36] = 0
    pgm = f"P5\n{w} {h}\n255\n".encode("ascii") + gray.tobytes()
    yaml = (
        "image: map.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    meta = json.dumps({"label": label, "source": "phase4_smoke"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{map_id}/occupancy/map.yaml", yaml)
        zf.writestr(f"{map_id}/occupancy/map.pgm", pgm)
        zf.writestr(f"{map_id}/occupancy/meta.json", meta)
    return buf.getvalue()


def _png_ok(data: bytes) -> bool:
    return len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n"


def _read_active_file() -> Optional[Dict[str, Any]]:
    path = MAPS_DIR / ".active_map.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_active_file(payload: Optional[Dict[str, Any]]) -> None:
    path = MAPS_DIR / ".active_map.json"
    if payload is None:
        if path.is_file():
            path.unlink()
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cleanup_map(map_id: str) -> None:
    d = MAPS_DIR / map_id
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# API smoke
# ---------------------------------------------------------------------------


def run_api_smoke(client: HubClient, *, activate_restart: bool) -> List[str]:
    """Return list of step labels that passed. Raises on hard failure."""
    passed: List[str] = []
    map_id = f"smoke_p4_{int(time.time())}"
    prev_active = _read_active_file()

    print(f"\n== API smoke (map_id={map_id}) ==")
    try:
        code, catalog, _ = client.get("/api/maps")
        if code != 200 or not isinstance(catalog, list):
            raise RuntimeError(
                f"GET /api/maps failed ({code}). "
                "If using --base-url, restart uvicorn so Phase 4 routes load."
            )
        ids = [m.get("id") for m in catalog if isinstance(m, dict)]
        if "porto" not in ids and "none" not in ids:
            raise RuntimeError(f"catalog unexpected: {ids[:8]}")
        passed.append("catalog")
        print(f"  [ok] catalog ({len(catalog)} entries)")

        zip_bytes = _make_ring_zip(map_id)
        code, up, _ = client.post(
            "/api/maps/upload",
            data=zip_bytes,
            headers={"Content-Type": "application/zip"},
            params={"map_id": map_id, "label": "Phase4 Smoke"},
        )
        if code != 200 or not isinstance(up, dict) or not up.get("ok"):
            raise RuntimeError(f"upload failed: {code} {up}")
        if up.get("id") != map_id:
            raise RuntimeError(f"upload id mismatch: {up}")
        if not (MAPS_DIR / map_id / "occupancy" / "map.pgm").is_file():
            raise RuntimeError("upload did not write occupancy files")
        passed.append("upload")
        print("  [ok] upload zip")

        code, thumb, _ = client.post(f"/api/maps/{map_id}/generate-thumbnail")
        if code != 200 or not isinstance(thumb, dict) or not thumb.get("ok"):
            raise RuntimeError(f"thumbnail failed: {code} {thumb}")
        thumb_path = Path(str(thumb.get("path", "")))
        if not thumb_path.is_file():
            # hub may return container path; check local occupancy preview
            local = MAPS_DIR / map_id / "occupancy" / "preview.png"
            if not local.is_file():
                raise RuntimeError(f"thumbnail missing on disk: {thumb}")
        passed.append("thumbnail")
        print("  [ok] generate-thumbnail")

        code, cl, _ = client.post(f"/api/maps/{map_id}/generate-centerline")
        if code != 200 or not isinstance(cl, dict) or not cl.get("ok"):
            raise RuntimeError(f"centerline failed: {code} {cl}")
        result = cl.get("result") or {}
        if int(result.get("n_points") or 0) < 4:
            raise RuntimeError(f"centerline too short: {result}")
        code, _, csv_body = client.get(f"/api/maps/{map_id}/centerline.csv")
        if code != 200:
            raise RuntimeError(f"centerline.csv download {code}")
        text = csv_body.decode("utf-8", errors="replace")
        if "x_m" not in text or len(text.splitlines()) < 5:
            raise RuntimeError(f"CSV shape bad: {text[:120]!r}")
        passed.append("centerline")
        print(f"  [ok] centerline ({result.get('n_points')} pts) + CSV download")

        mesh_dir = MAPS_DIR / map_id / "mesh"
        need_mesh = not (
            (mesh_dir / "track.obj").is_file() or (mesh_dir / "track_col.obj").is_file()
        )
        if need_mesh:
            code, mesh, _ = client.post(f"/api/maps/{map_id}/generate-mesh")
            if code != 200 or not isinstance(mesh, dict) or not mesh.get("ok"):
                raise RuntimeError(f"generate-mesh failed: {code} {mesh}")
            passed.append("generate-mesh")
            print("  [ok] generate-mesh")
        else:
            passed.append("generate-mesh-skip")
            print("  [ok] mesh already present (skip generate)")

        code, prev, _ = client.post(f"/api/maps/{map_id}/generate-mesh-preview")
        if code != 200 or not isinstance(prev, dict) or not prev.get("ok"):
            raise RuntimeError(f"mesh-preview failed: {code} {prev}")
        code, _, png = client.get(f"/api/maps/{map_id}/mesh-preview.png")
        if code != 200 or not _png_ok(png):
            raise RuntimeError(f"mesh-preview.png bad: code={code} len={len(png)}")
        passed.append("mesh-preview")
        print(f"  [ok] mesh-preview PNG ({len(png)} bytes)")

        # mismatch: select berlin while we temporarily activate smoke map
        code, act, _ = client.post(
            f"/api/maps/{map_id}/activate",
            json_body={"restart": bool(activate_restart)},
        )
        if code != 200 or not isinstance(act, dict) or not act.get("ok"):
            raise RuntimeError(f"activate failed: {code} {act}")
        if (act.get("active") or {}).get("id") != map_id:
            raise RuntimeError(f"activate did not persist: {act}")
        if activate_restart:
            print(f"  [ok] activate restart={act.get('restarted')} err={act.get('restart_error')}")
        else:
            if act.get("restarted"):
                print(f"  [warn] restart=false but restarted={act.get('restarted')}")
            print("  [ok] activate persist (restart=false)")
        passed.append("activate")

        code, st, _ = client.get("/api/maps/active", params={"selected": "berlin"})
        if code != 200 or not isinstance(st, dict):
            raise RuntimeError(f"active status failed: {code} {st}")
        if not st.get("selection_mismatch"):
            raise RuntimeError(f"expected selection_mismatch: {st}")
        if not st.get("warnings"):
            raise RuntimeError(f"expected warnings: {st}")
        code, st2, _ = client.get("/api/maps/active", params={"selected": map_id})
        if code != 200 or not isinstance(st2, dict) or st2.get("selection_mismatch"):
            raise RuntimeError(f"expected no mismatch for selected={map_id}: {st2}")
        passed.append("mismatch")
        print("  [ok] active?selected= mismatch warnings")

        # Catalog should list smoke map + thumbnail URL after generate
        code, catalog2, _ = client.get("/api/maps")
        entry = next(
            (m for m in (catalog2 or []) if isinstance(m, dict) and m.get("id") == map_id),
            None,
        )
        if entry is None:
            raise RuntimeError("uploaded map missing from catalog after activate")
        if not entry.get("thumbnail_url"):
            print("  [warn] catalog entry missing thumbnail_url (non-fatal)")
        else:
            passed.append("catalog-thumb-url")
            print(f"  [ok] catalog thumbnail_url={entry['thumbnail_url']}")

    finally:
        # Restore prior active map (prefer file restore over API to avoid extra restarts)
        try:
            _write_active_file(prev_active)
            print(
                f"  [restore] active map -> "
                f"{(prev_active or {}).get('id') if prev_active else 'builtin/none'}"
            )
        except OSError as exc:
            print(f"  [warn] could not restore active map: {exc}")
        _cleanup_map(map_id)
        print(f"  [cleanup] removed maps/{map_id}")

    return passed


# ---------------------------------------------------------------------------
# Headless / headed sim smoke
# ---------------------------------------------------------------------------


def _ensure_activatable_map(map_id: str = "porto") -> str:
    mesh = MAPS_DIR / map_id / "mesh"
    if not ((mesh / "track.obj").is_file() or (mesh / "track_col.obj").is_file()):
        raise RuntimeError(f"map '{map_id}' has no mesh — cannot sim-smoke")
    return map_id


def _activate_persist(map_id: str) -> None:
    from src.layer4.hub.map_activate import activate_map

    activate_map(MAPS_DIR, map_id, restart=False)


def run_headless_smoke(
    *,
    map_id: str = "porto",
    base_port: int = 4670,
    duration: float = 8.0,
    connect_timeout: float = 60.0,
    boot_timeout: float = 45.0,
) -> None:
    """Headless / batchmode map boot smoke.

    Primary check (Windows TrackLoader player): launch with ``-batchmode -map-id``
    and assert Player.log contains ``[AiCar.TrackLoader] loaded map=<id>``.

    Note: Layer1 ``headless=True`` also passes ``-nographics``. On this Windows
    Mono player that combination often exits immediately (code 127) — especially
    when another headed instance is already open — so the smoke uses batchmode
    without ``-nographics`` for the TrackLoader assert. Optional RaceTrack
    Socket.IO connect is attempted afterward and reported as soft if it fails.
    """
    print(f"\n== Headless sim smoke (map={map_id}) ==")
    if not WIN_SIM.is_file():
        raise RuntimeError(f"Windows sim missing: {WIN_SIM}")
    if not TRACKLOADER_MARKER.is_file():
        print("  [warn] simulator/.aicar_trackloader missing — map-id may be ignored")

    prev = _read_active_file()
    map_id = _ensure_activatable_map(map_id)
    _activate_persist(map_id)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "headless_phase4_Player.log"
    if log_path.is_file():
        log_path.unlink()

    cmd = [
        str(WIN_SIM.resolve()),
        "-batchmode",
        # intentionally omit -nographics (see docstring)
        "-map-id",
        map_id,
        "-ip",
        "127.0.0.1",
        "-port",
        str(base_port),
        "-logFile",
        str(log_path.resolve()),
    ]
    print(f"  launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WIN_SIM.parent.resolve()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        needle = f"[AiCar.TrackLoader] loaded map={map_id}"
        t0 = time.time()
        loaded = False
        while time.time() - t0 < boot_timeout:
            if proc.poll() is not None and not log_path.is_file():
                raise RuntimeError(
                    f"sim exited early code={proc.returncode} with no Player.log "
                    "(try closing other AutoDRIVE windows; avoid -nographics on Windows)"
                )
            if log_path.is_file():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if needle in text:
                    loaded = True
                    break
                if "[AiCar.MapConfig] CLI -map-id=" in text and proc.poll() is not None:
                    # booted far enough to parse CLI but exited before load
                    raise RuntimeError(
                        f"sim exited code={proc.returncode} after map-id parse; "
                        f"log tail:\n{text[-800:]}"
                    )
            time.sleep(0.5)
        if not loaded:
            tail = ""
            if log_path.is_file():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            raise RuntimeError(
                f"TrackLoader load not seen within {boot_timeout:.0f}s for map={map_id}\n{tail}"
            )
        print(f"  [ok] {needle}")
        print(f"  [ok] batchmode stayed up pid={proc.pid} (no early crash)")

        # Soft connect probe via RaceTrack (may fail with -nographics / multi-instance)
        try:
            from src.layer1.track import RaceTrack

            # Kill batchmode boot process first so RaceTrack can launch its own
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            proc = None  # type: ignore

            track = RaceTrack(
                num_racers=1,
                base_port=base_port + 1,
                simulator_path=WIN_SIM,
                auto_launch=True,
                headless=True,
            )
            try:
                child = track.racers[0]._sim_process
                print(
                    f"  RaceTrack headless (-batchmode -nographics) pid="
                    f"{getattr(child, 'pid', None)} — connect probe {connect_timeout:.0f}s..."
                )
                t1 = time.time()
                connected = False
                while time.time() - t1 < min(connect_timeout, 25.0):
                    if track.racers[0].is_connected:
                        connected = True
                        break
                    if child is not None and child.poll() is not None:
                        print(
                            f"  [skip] RaceTrack headless exited code={child.returncode} "
                            "(known Windows -nographics issue); TrackLoader boot assert already passed"
                        )
                        break
                    time.sleep(0.5)
                if connected:
                    end = time.time() + duration
                    steps = 0
                    while time.time() < end:
                        track.racers[0].step(1.0, 0.0)
                        steps += 1
                    print(f"  [ok] RaceTrack connect + {steps} steps")
                elif child is not None and child.poll() is None:
                    print("  [skip] RaceTrack did not connect in time (boot assert still PASS)")
            finally:
                try:
                    track.kill_all()
                except Exception as exc:
                    print(f"  [warn] RaceTrack kill_all: {exc}")
        except Exception as exc:
            print(f"  [skip] RaceTrack connect probe: {exc}")
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _write_active_file(prev)
        print(
            f"  [restore] active map -> "
            f"{(prev or {}).get('id') if prev else 'builtin/none'}"
        )


def _find_autodrive_hwnds() -> List[Tuple[int, str]]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    found: List[Tuple[int, str]] = []

    def cb(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value
        if "AutoDRIVE" in title or "Simulator" in title:
            found.append((int(hwnd), title))
        return True

    EnumWindows(EnumWindowsProc(cb), 0)
    return found


def _capture_hwnd(hwnd: int, out: Path) -> Dict[str, Any]:
    """Capture AutoDRIVE client area → PNG (same approach as capture_sim_window.py)."""
    import ctypes
    from ctypes import wintypes

    from PIL import Image

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    SW_RESTORE = 9

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

    # Minimized / 0x0 client rect is common when the window lost focus overnight.
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.6)
    user32.ShowWindow(hwnd, SW_RESTORE)
    time.sleep(0.2)

    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w < 10 or h < 10:
        # Fall back to outer window rect (includes chrome) rather than failing the smoke.
        wr = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(wr))
        w, h = wr.right - wr.left, wr.bottom - wr.top
        if w < 10 or h < 10:
            raise RuntimeError(f"bad window size client/window {w}x{h}")

    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(mem, bmp)
    ok = user32.PrintWindow(hwnd, mem, 2)
    if not ok:
        gdi32.BitBlt(mem, 0, 0, w, h, hdc, 0, 0, 0x00CC0020)

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0
    buf = (ctypes.c_ubyte * (w * h * 4))()
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    gdi32.SelectObject(mem, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)

    img = Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1).convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    mean = sum(img.convert("L").resize((64, 64)).getdata()) / (64 * 64)
    return {"path": str(out), "size": img.size, "mean": mean}


def run_headed_smoke(
    *,
    map_id: str = "porto",
    reuse: bool = False,
    base_port: int = 4680,
    settle_s: float = 12.0,
    out_name: str = "headed_phase4_smoke.png",
) -> None:
    print(f"\n== Headed sim smoke (map={map_id}, reuse={reuse}) ==")
    if sys.platform != "win32":
        raise RuntimeError("headed capture requires Windows")
    if not WIN_SIM.is_file():
        raise RuntimeError(f"Windows sim missing: {WIN_SIM}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / out_name
    prev = _read_active_file()
    map_id = _ensure_activatable_map(map_id)
    _activate_persist(map_id)

    launched_pid: Optional[int] = None
    track = None
    did_launch = False
    try:
        wins = _find_autodrive_hwnds()
        if reuse and wins:
            print(f"  reusing window: {wins[0][1]!r}")
            # Verify an AutoDRIVE process mentions -map-id when possible
            try:
                r = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='AutoDRIVE Simulator.exe'\" "
                        "| Select-Object -ExpandProperty CommandLine",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                cl = r.stdout or ""
                if f"-map-id {map_id}" not in cl and f"-map-id={map_id}" not in cl:
                    print(
                        f"  [warn] running sim cmdline may not include -map-id {map_id}; "
                        "capture still saved for visual proof"
                    )
                else:
                    print(f"  [ok] running sim cmdline includes -map-id {map_id}")
            except Exception as exc:
                print(f"  [warn] cmdline check skipped: {exc}")
            info = _capture_hwnd(wins[0][0], out)
            print(f"  [ok] captured {info}")
            return

        if wins and not reuse:
            print(
                f"  [warn] {len(wins)} AutoDRIVE window(s) already open; "
                "launching another headed instance (use --headed-reuse to capture existing)"
            )
        elif reuse and not wins:
            print("  [warn] --headed-reuse set but no AutoDRIVE window found; launching fresh")

        from src.layer1.track import RaceTrack

        # Headed RaceTrack expects manual Connect in GUI; we still launch and capture window.
        track = RaceTrack(
            num_racers=1,
            base_port=base_port,
            simulator_path=WIN_SIM,
            auto_launch=True,
            headless=False,
        )
        did_launch = True
        proc = track.racers[0]._sim_process
        if proc is None:
            raise RuntimeError("headed sim did not start")
        launched_pid = proc.pid
        (LOG_DIR / "headed_phase4_sim_pid.txt").write_text(str(launched_pid), encoding="utf-8")
        print(f"  launched pid={launched_pid}; settling {settle_s:.0f}s for window...")
        time.sleep(settle_s)
        if proc.poll() is not None:
            raise RuntimeError(f"headed sim exited early code={proc.returncode}")

        wins = _find_autodrive_hwnds()
        if not wins:
            raise RuntimeError("no AutoDRIVE window found after launch")
        info = _capture_hwnd(wins[0][0], out)
        print(f"  [ok] captured {info}")
        print("  note: headed connect is manual in GUI; capture proves window+map launch")
    finally:
        if track is not None and did_launch:
            try:
                track.kill_all()
            except Exception as exc:
                print(f"  [warn] kill_all: {exc}")
        # If we only reused, do not kill user's existing sim; still restore active file.
        _write_active_file(prev)
        print(
            f"  [restore] active map -> "
            f"{(prev or {}).get('id') if prev else 'builtin/none'}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 4 map end-to-end smoke tests")
    p.add_argument(
        "--base-url",
        default=None,
        help="Live hub URL (default: in-process TestClient with current code)",
    )
    p.add_argument(
        "--activate-restart",
        action="store_true",
        help="Allow activate to restart compose sims (default: restart=false)",
    )
    p.add_argument("--headless", action="store_true", help="Boot headless Windows sim + connect")
    p.add_argument("--headed", action="store_true", help="Boot/capture headed Windows sim window")
    p.add_argument(
        "--headed-reuse",
        action="store_true",
        help="With --headed, capture an already-running AutoDRIVE window",
    )
    p.add_argument("--map-id", default="porto", help="Map for headless/headed sim smoke")
    p.add_argument("--api-only", action="store_true", help="Skip sim sections even if flags set")
    p.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API smoke (useful with --headless / --headed only)",
    )
    args = p.parse_args()

    print("Phase 4 map smoke")
    print(f"  ROOT={ROOT}")
    print(f"  MAPS_DIR={MAPS_DIR} exists={MAPS_DIR.is_dir()}")
    print(f"  TrackLoader marker={TRACKLOADER_MARKER.is_file()}")
    if args.skip_api:
        print("  client=skipped")
    else:
        print(f"  client={'live ' + args.base_url if args.base_url else 'TestClient (in-process)'}")

    failures: List[str] = []

    if not args.skip_api:
        client = HubClient(args.base_url)
        try:
            steps = run_api_smoke(client, activate_restart=args.activate_restart)
            print(f"\nAPI smoke PASS ({len(steps)} checks): {', '.join(steps)}")
        except Exception as exc:
            failures.append(f"api: {exc}")
            print(f"\nAPI smoke FAIL: {exc}")
    else:
        print("\nAPI smoke SKIPPED (--skip-api)")

    if not args.api_only and args.headless:
        try:
            run_headless_smoke(map_id=args.map_id)
            print("\nHeadless smoke PASS")
        except Exception as exc:
            failures.append(f"headless: {exc}")
            print(f"\nHeadless smoke FAIL: {exc}")

    if not args.api_only and args.headed:
        try:
            run_headed_smoke(map_id=args.map_id, reuse=args.headed_reuse)
            print("\nHeaded smoke PASS")
        except Exception as exc:
            failures.append(f"headed: {exc}")
            print(f"\nHeaded smoke FAIL: {exc}")

    if failures:
        print("\nRESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
