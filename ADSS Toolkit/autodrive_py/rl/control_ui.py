"""Ugly-simple training control panel (stdlib HTTP — no Gradio, no OpenCV).

Launch (from ADSS Toolkit/autodrive_py):
  python -m rl.control_ui
  .\\rl\\start_ui.ps1

Train stays headless. This process starts/stops train + optional TensorBoard,
and reads live_status.json. Watch opens in a separate console.

W8 multi-run: Live runs list, Focus → CURRENT_RUN.txt, banner prefers max
live timesteps, Start lock warn, Stop confirm on selected run. See HANDOFF.md.
Overnight soak is never kill-swept. Page ``UI_BUILD`` may be a later stamp;
hard-refresh if the Live runs list is missing.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .contracts import CONTRACTS_VERSION, N_LIDAR_DEFAULT, obs_dim
from .live_status import find_latest_status, read_live_status, write_live_status
from .ui_ops import (
    PRESETS,
    classify_banner,
    clear_stale_lock,
    coach_hints,
    continue_train_argv,
    delete_run_model,
    find_live_run_locks,
    generate_maps_op,
    list_run_models,
    lock_owner,
    map_labels,
    model_timeline,
    precheck_model,
    race_candidate,
    read_operator_run_pin,
    resolve_model_choice,
    resolve_stop_budget,
    run_curriculum_flags,
    run_map_ids,
    seal_verify_summary,
    start_guard,
    start_preview,
    safe_run_id,
    write_operator_run_pin,
    clear_operator_run_pin,
)

RL_DIR = Path(__file__).resolve().parent
AUTODRIVE_PY = RL_DIR.parent
RUNS_DIR = RL_DIR / "runs"
LOGS_DIR = RL_DIR / "logs"
MODELS_DIR = RL_DIR / "models"
MAPS_DIR = RL_DIR / "maps"
THUMBS_DIR = RL_DIR / "logs" / "map_thumbs"
VENV_PYTHON = RL_DIR / ".venv" / "Scripts" / "python.exe"

TB_HOST = "127.0.0.1"
TB_PORT = 6006
TB_URL = f"http://{TB_HOST}:{TB_PORT}/"
THUMB_SIZE = 256

_state_lock = threading.Lock()
_train_proc: subprocess.Popen | None = None
_train_run_id: str | None = None
_train_log: Path | None = None
_train_n_envs: int = 8
_tb_proc: subprocess.Popen | None = None  # only UI-spawned TB; Stop kills this
_tb_log_f = None  # keep log handle alive for the TB child on Windows
_last_msg = "idle"
_watch_pids: list[int] = []
_maps_cache: list[dict] | None = None
_stopping: bool = False
_last_exit_code: int | None = None
_watch_opened: bool = False
# Banner staleness: remember when timesteps last changed.
_prev_timesteps: int | None = None
_prev_ts_changed_at: float = 0.0
_mapgen_lock = threading.Lock()
# Cache external-train scans: without psutil this shells PowerShell every poll.
_ext_train_cache_ts: float = 0.0
_ext_train_cache_val: tuple[int, str | None] | None = None
_EXT_TRAIN_TTL_S = 2.0
_CLIENT_GONE = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


def _python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def _clamp_n_envs(n: int) -> int:
    return max(1, min(32, int(n)))


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.35)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def _tb_alive() -> bool:
    with _state_lock:
        proc = _tb_proc
    if proc is not None and proc.poll() is None:
        return True
    return _port_in_use(TB_HOST, TB_PORT)


def _kill_pid_tree(pid: int) -> bool:
    """Force-kill a process and its children. Returns True if a kill was attempted."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return True
        os.kill(pid, 15)
        time.sleep(0.4)
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        return True
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False


def _kill_proc_tree(proc: subprocess.Popen | None) -> int:
    """Kill tracked Popen tree. Returns 1 if killed / signalled, else 0."""
    if proc is None:
        return 0
    if proc.poll() is not None:
        return 0
    pid = proc.pid
    killed = 0
    try:
        import psutil

        try:
            parent = psutil.Process(pid)
            kids = parent.children(recursive=True)
            for c in kids:
                try:
                    c.terminate()
                except psutil.Error:
                    pass
            gone, alive = psutil.wait_procs(kids + [parent], timeout=2)
            for p in alive:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            killed = 1
        except psutil.NoSuchProcess:
            killed = 0
    except ImportError:
        if _kill_pid_tree(pid):
            killed = 1
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
            killed = 1
    return killed


PROTECTED_OVERNIGHT_RUN = "overnight_soak_20260917_082739"


def _sweep_train_ppo() -> int:
    """Kill stray python -m rl.train_ppo outside this UI.

    Never touches the protected overnight soak (or any cmdline containing its run_id).
    """
    killed = 0
    try:
        import psutil

        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd_list = p.info.get("cmdline") or []
                cmd = " ".join(cmd_list)
            except (psutil.Error, TypeError):
                continue
            if "control_ui" in cmd:
                continue
            if PROTECTED_OVERNIGHT_RUN in cmd:
                continue
            hit = ("train_ppo" in cmd) or ("start_train" in cmd and "rl" in cmd.lower())
            if not hit:
                continue
            try:
                kids = p.children(recursive=True)
                for c in kids:
                    try:
                        c.kill()
                    except psutil.Error:
                        pass
                p.kill()
                killed += 1
            except psutil.Error:
                pass
        return killed
    except ImportError:
        pass

    if sys.platform == "win32":
        ps = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "$n=0; $prot='" + PROTECTED_OVERNIGHT_RUN + "'; "
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { "
                    "  $_.CommandLine -like '*train_ppo*' -and "
                    "  $_.CommandLine -notlike '*control_ui*' -and "
                    "  $_.CommandLine -notlike ('*'+$prot+'*') "
                    "} | ForEach-Object { "
                    "  taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null; $n++ "
                    "}; Write-Output $n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        try:
            killed += int((ps.stdout or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
    return killed


def _kill_train_tree() -> int:
    """Stop tracked train_ppo child tree + stray trainers. Does not touch TensorBoard.

    ONLY call from explicit Stop (or proven-dead cleanup). Never from Start/Continue —
    a busy false-negative would murder a live overnight then spawn a dual writer.
    """
    global _train_proc, _train_run_id
    killed = 0
    with _state_lock:
        proc = _train_proc
        _train_proc = None
    killed += _kill_proc_tree(proc)
    killed += _sweep_train_ppo()
    with _state_lock:
        _train_run_id = None
    _invalidate_ext_train_cache()
    return killed


def _kill_ui_tensorboard() -> int:
    """Kill only TensorBoard that this UI spawned (tracked PID). Leave external TB alone."""
    global _tb_proc, _tb_log_f
    with _state_lock:
        proc = _tb_proc
        _tb_proc = None
        log_f = _tb_log_f
        _tb_log_f = None
    n = _kill_proc_tree(proc)
    if log_f is not None:
        try:
            log_f.close()
        except OSError:
            pass
    return n


def _tensorboard_argv() -> list[str]:
    """Launch via venv python -m tensorboard.main (reliable; no tensorboard.__main__)."""
    base = [
        "--logdir",
        str(RUNS_DIR.resolve()),
        "--host",
        TB_HOST,
        "--port",
        str(TB_PORT),
        "--reload_interval",
        "5",
    ]
    # Prefer module form: tensorboard.exe sometimes shadows / mis-launches on Windows.
    return [_python(), "-m", "tensorboard.main", *base]


def _tb_http_ok(timeout: float = 0.8) -> bool:
    """True if TensorBoard HTTP root responds (any 2xx/3xx)."""
    try:
        import urllib.request

        req = urllib.request.Request(TB_URL, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= int(getattr(resp, "status", 200)) < 400
    except Exception:
        return False


def _wait_tb_ready(timeout_s: float = 8.0) -> bool:
    """Wait until port is open (and ideally HTTP serves)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _port_in_use(TB_HOST, TB_PORT):
            if _tb_http_ok():
                return True
            # Port up but HTTP still warming — keep waiting a bit.
            if time.time() + 1.5 >= deadline:
                return True
        time.sleep(0.25)
    return _port_in_use(TB_HOST, TB_PORT)


def _ensure_tensorboard() -> str:
    """Spawn tensorboard in background if we don't already track a live one.

    If port 6006 is already serving, report already running and do not crash.
    Only UI-spawned processes are stored in _tb_proc (Stop kills those).
    """
    global _tb_proc, _tb_log_f, _last_msg

    with _state_lock:
        proc = _tb_proc
    if proc is not None and proc.poll() is None:
        if _port_in_use(TB_HOST, TB_PORT) or _tb_http_ok():
            return f"TensorBoard already running (pid={proc.pid}) -> {TB_URL}"
        # Tracked proc alive but not listening yet — wait a bit.
        if _wait_tb_ready(4.0):
            return f"TensorBoard ready (pid={proc.pid}) -> {TB_URL}"

    if _port_in_use(TB_HOST, TB_PORT) or _tb_http_ok():
        with _state_lock:
            # External / orphan listener — do not claim ownership.
            if _tb_proc is not None and _tb_proc.poll() is not None:
                _tb_proc = None
        return f"TensorBoard already running on port {TB_PORT} -> {TB_URL}"

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _tensorboard_argv()
    creationflags = 0
    if sys.platform == "win32":
        # New process group so Stop can kill the tree; allow a console-less child.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )  # type: ignore[attr-defined]

    log_path = LOGS_DIR / "tensorboard_ui.log"
    try:
        log_f = open(log_path, "a", encoding="utf-8")
        log_f.write(f"\n--- spawn {datetime.now().isoformat()} ---\n")
        log_f.write("cmd: " + " ".join(cmd) + "\n")
        log_f.write(f"cwd: {AUTODRIVE_PY}\n")
        log_f.write(f"logdir: {RUNS_DIR.resolve()}\n")
        log_f.flush()
        new_proc = subprocess.Popen(
            cmd,
            cwd=str(AUTODRIVE_PY),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
    except OSError as e:
        try:
            log_f.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return f"TensorBoard failed to start: {e}"

    # Keep log handle open for the child (closing it on Windows can break piping).
    with _state_lock:
        old_log = _tb_log_f
        _tb_log_f = log_f
        _tb_proc = new_proc
    if old_log is not None and old_log is not log_f:
        try:
            old_log.close()
        except OSError:
            pass

    ready = _wait_tb_ready(8.0)
    if new_proc.poll() is not None:
        with _state_lock:
            if _tb_proc is new_proc:
                _tb_proc = None
        if _port_in_use(TB_HOST, TB_PORT) or _tb_http_ok():
            return f"TensorBoard already running on port {TB_PORT} -> {TB_URL}"
        return f"TensorBoard exited immediately (code={new_proc.returncode}) - check logs/tensorboard_ui.log"

    if not ready:
        return (
            f"TensorBoard spawned (pid={new_proc.pid}) but port {TB_PORT} not ready yet "
            f"- try Open TensorBoard shortly -> {TB_URL}"
        )

    http = " (HTTP ok)" if _tb_http_ok() else ""
    return f"TensorBoard started (pid={new_proc.pid}){http} -> {TB_URL}"


def _resolve_map_image(name: str) -> Path | None:
    """Prefer maps/<id>/<id>.png, else .pgm."""
    folder = MAPS_DIR / name
    if not folder.is_dir():
        return None
    for ext in (".png", ".pgm"):
        cand = folder / f"{name}{ext}"
        if cand.is_file():
            return cand
    return None


def _list_maps(*, force: bool = False) -> list[dict]:
    """Discover maps under rl/maps/<name>/ with thumbnail URLs."""
    global _maps_cache
    if _maps_cache is not None and not force:
        return _maps_cache

    maps: list[dict] = []
    if not MAPS_DIR.is_dir():
        maps = [
            {"id": "map0", "thumb": None, "src": None, "sealed": False, "train_safe": True, "role": "train_ok", "label": "map0"},
            {"id": "map1", "thumb": None, "src": None, "sealed": False, "train_safe": True, "role": "train_ok", "label": "map1"},
            {"id": "map2", "thumb": None, "src": None, "sealed": False, "train_safe": True, "role": "train_ok", "label": "map2"},
        ]
        _maps_cache = maps
        return maps

    try:
        sealed_by_id = {m["id"]: m for m in map_labels(MAPS_DIR)}
    except Exception:
        sealed_by_id = {}

    for d in sorted(MAPS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        src = _resolve_map_image(name)
        thumb_url = None
        if src is not None:
            try:
                thumb_path = _ensure_thumb(name, src)
                if thumb_path is not None and thumb_path.is_file():
                    thumb_url = f"/api/map_thumb/{name}"
            except Exception:
                thumb_url = None
        tag = sealed_by_id.get(name, {})
        maps.append(
            {
                "id": name,
                "thumb": thumb_url,
                "src": str(src) if src else None,
                "sealed": bool(tag.get("sealed")),
                "train_safe": bool(tag.get("train_safe", not tag.get("sealed"))),
                "role": str(tag.get("role") or ("holdout" if tag.get("sealed") else "train_ok")),
                "label": str(tag.get("label") or name),
            }
        )

    if not maps:
        maps = [{"id": "map0", "thumb": None, "src": None, "sealed": False, "train_safe": True, "role": "train_ok", "label": "map0"}]
    _maps_cache = maps
    return maps


def _pgm_to_preview_png(src: Path, out: Path, size: int = THUMB_SIZE) -> bool:
    """Fallback: subsample a binary P5 PGM without Pillow."""
    try:
        with open(src, "rb") as f:
            magic = f.readline().strip()
            if magic != b"P5":
                return False
            # skip comments
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            wh = line.split()
            while len(wh) < 2:
                line = f.readline()
                if line.startswith(b"#"):
                    continue
                wh += line.split()
            w, h = int(wh[0]), int(wh[1])
            maxval_line = f.readline()
            while maxval_line.startswith(b"#"):
                maxval_line = f.readline()
            maxval = int(maxval_line.split()[0])
            if maxval > 255 or w <= 0 or h <= 0:
                return False
            raw = f.read(w * h)
        if len(raw) < w * h:
            return False

        # nearest-neighbor downsample into size x size box (contain)
        scale = min(size / w, size / h)
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        pixels = bytearray(tw * th)
        for y in range(th):
            sy = min(h - 1, int(y / scale))
            row = sy * w
            for x in range(tw):
                sx = min(w - 1, int(x / scale))
                pixels[y * tw + x] = raw[row + sx]

        try:
            from PIL import Image

            Image.frombytes("L", (tw, th), bytes(pixels)).convert("RGB").save(out, format="PNG")
            return True
        except Exception:
            pass

        # Minimal RGB PNG via Pillow-free path is awkward; write PGM then retry Pillow once more
        # If Pillow missing entirely, write a tiny valid 1x1 PNG as last resort marker.
        _write_solid_png(out, tw, th, pixels)
        return out.is_file()
    except Exception:
        return False


def _write_solid_png(out: Path, w: int, h: int, gray: bytes) -> None:
    """Write grayscale bytes as RGB PNG using zlib (no Pillow)."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = []
    for y in range(h):
        row = bytearray([0])  # filter none
        base = y * w
        for x in range(w):
            g = gray[base + x]
            row.extend((g, g, g))
        rows.append(bytes(row))
    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    out.write_bytes(png)


def _ensure_thumb(name: str, src: Path | None = None) -> Path | None:
    """Resize map image to ~256px PNG cache under logs/map_thumbs."""
    if src is None:
        src = _resolve_map_image(name)
    if src is None:
        return None

    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMBS_DIR / f"{name}_{THUMB_SIZE}.png"
    try:
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime and out.stat().st_size > 64:
            return out
    except OSError:
        pass

    # Prefer Pillow (handles png + pgm)
    try:
        from PIL import Image

        img = Image.open(src)
        img.load()
        img = img.convert("L")
        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), resample)
        img = img.convert("RGB")
        img.save(out, format="PNG", optimize=True)
        if out.is_file() and out.stat().st_size > 64:
            return out
    except Exception:
        pass

    # PGM-only fallback (huge occupancy grids)
    if src.suffix.lower() == ".pgm" and _pgm_to_preview_png(src, out, THUMB_SIZE):
        return out if out.is_file() else None

    # Last resort: tiny placeholder so the UI still shows *something*
    try:
        _write_solid_png(out, 64, 64, bytes([40] * (64 * 64)))
        return out if out.is_file() else None
    except Exception:
        return None


def _prewarm_map_thumbs() -> None:
    """Build map list + thumbnails once at UI start (avoids empty first paint)."""
    global _maps_cache
    _maps_cache = None
    maps = _list_maps(force=True)
    for m in maps:
        if m.get("thumb"):
            continue
        src = _resolve_map_image(m["id"])
        thumb = _ensure_thumb(m["id"], src)
        if thumb is not None:
            m["thumb"] = f"/api/map_thumb/{m['id']}"
            m["src"] = str(src) if src else None
    _maps_cache = maps


def _set_msg(msg: str) -> str:
    global _last_msg
    with _state_lock:
        _last_msg = msg
    return msg


def _invalidate_ext_train_cache() -> None:
    global _ext_train_cache_ts, _ext_train_cache_val
    _ext_train_cache_ts = 0.0
    _ext_train_cache_val = None


def _preferred_operator_run_id() -> str | None:
    """Optional operator pin (logs/CURRENT_RUN.txt) — used when several trains run."""
    return read_operator_run_pin(LOGS_DIR)


def _set_operator_run_pin(run_id: str) -> None:
    write_operator_run_pin(LOGS_DIR, run_id)


def _list_live_runs(*, selected_run: str | None = None) -> list[dict]:
    """Compact multi-train inventory for the Control status box (W8)."""
    preferred = _preferred_operator_run_id()
    with _state_lock:
        bound = _train_run_id
        ui_pid = (
            _train_proc.pid
            if _train_proc is not None and _train_proc.poll() is None
            else None
        )
    rows: list[dict] = []
    for hit in find_live_run_locks(MODELS_DIR):
        rid = str(hit.get("run_id") or "")
        if not rid:
            continue
        live = read_live_status(RUNS_DIR / rid / "live_status.json") or {}
        pid = hit.get("pid")
        overnight = rid == PROTECTED_OVERNIGHT_RUN or rid.startswith("overnight_soak_")
        if selected_run is not None:
            selected = rid == selected_run
        else:
            selected = bool(
                (preferred and rid == preferred) or (bound and rid == bound)
            )
        rows.append(
            {
                "run_id": rid,
                "pid": pid,
                "timesteps": live.get("timesteps"),
                "phase": live.get("phase") or "?",
                "steps_per_sec": live.get("steps_per_sec"),
                "overnight": overnight,
                "protected": overnight,
                "selected": selected,
                "focused": selected,
                "ui_owned": ui_pid is not None and pid == ui_pid,
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r.get("overnight") else 1,
            0 if r.get("selected") else 1,
            -(int(r["timesteps"]) if isinstance(r.get("timesteps"), (int, float)) else -1),
        )
    )
    return rows


def _resolve_selected_run(
    *,
    ui_owned_run: str | None,
    ui_alive: bool,
    live_rows: list[dict],
) -> str | None:
    """Banner / Watch / Stop bind to one selected run_id (W8-02).

    Order: UI-owned alive → CURRENT_RUN pin if still live → highest live
    timesteps (overnight >> short A/B smokes) → UI-owned trail → None.

    Never keep a dead Focus/pin (e.g. finished overnight) over a live train —
    that paints early_stopped metrics as Stale while a new Start is learning.
    """
    live_ids = {str(r["run_id"]) for r in live_rows}
    preferred = _preferred_operator_run_id()

    if ui_alive and ui_owned_run and ui_owned_run in live_ids:
        return ui_owned_run
    if ui_alive and ui_owned_run and not live_ids:
        # Spawned but lock not visible yet — trust UI-owned id.
        return ui_owned_run
    if preferred and preferred in live_ids:
        return preferred
    if ui_owned_run and ui_owned_run in live_ids:
        return ui_owned_run
    if live_rows:
        best = max(
            live_rows,
            key=lambda r: (
                int(r["timesteps"])
                if isinstance(r.get("timesteps"), (int, float))
                else -1
            ),
        )
        return str(best["run_id"])
    if ui_owned_run:
        return ui_owned_run
    # Dead pin only when nothing is live (show last overnight trail when Idle).
    return preferred


def _start_lock_warn(live_rows: list[dict] | None = None) -> str | None:
    """Visible cue when Start will refuse because N live locks exist (W8-04)."""
    rows = live_rows if live_rows is not None else _list_live_runs()
    if not rows:
        return None
    ids = ", ".join(str(r["run_id"]) for r in rows[:6])
    extra = f" (+{len(rows) - 6} more)" if len(rows) > 6 else ""
    return (
        f"{len(rows)} train(s) locked ({ids}{extra}) — Start refused until Stop. "
        "Multi-train is supported via separate CLI/UI ownership, not dual-Start from this panel."
    )


def _focus_run(run_id: str) -> str:
    """Pin status / ranking to a live (or known) run_id without Start/Stop/kill."""
    global _train_run_id, _train_log
    rid = str(run_id or "").strip()
    if not safe_run_id(rid):
        return _set_msg(f"Refuse focus: bad run_id {run_id!r}")
    model_dir = MODELS_DIR / rid
    if not model_dir.is_dir() and not (RUNS_DIR / rid).is_dir():
        return _set_msg(f"Refuse focus: no models/runs dir for {rid}")
    _set_operator_run_pin(rid)
    with _state_lock:
        _train_run_id = rid
        cand = LOGS_DIR / f"{rid}.log"
        _train_log = cand if cand.is_file() else _train_log
    _invalidate_ext_train_cache()
    owner = lock_owner(MODELS_DIR, rid)
    alive = bool(owner and owner.get("alive"))
    note = f"Focused -> {rid}"
    if rid == PROTECTED_OVERNIGHT_RUN:
        note += " (protected overnight - Stop sweep never kills this run_id)"
    elif alive:
        note += (
            f" (live pid={owner.get('pid')}; banner/Watch/Continue pin only - "
            "Stop still skips protected overnight)"
        )
    else:
        note += " (not live - status may show last trail; Start still refuses other live locks)"
    return _set_msg(note)


def _clear_focus() -> str:
    """Drop CURRENT_RUN pin so banner falls back to UI-owned / highest live timesteps."""
    global _train_run_id
    had = bool(_preferred_operator_run_id())
    clear_operator_run_pin(LOGS_DIR)
    with _state_lock:
        # Keep UI-owned live run_id if training; otherwise clear adopted pin.
        proc = _train_proc
        ui_alive = bool(proc is not None and proc.poll() is None)
        if not ui_alive:
            _train_run_id = None
    _invalidate_ext_train_cache()
    if had:
        return _set_msg(
            "Focus cleared — banner follows live train / highest timesteps (no CURRENT_RUN pin)."
        )
    return _set_msg("Focus already clear (no CURRENT_RUN pin).")


def _live_timesteps_for_run(run_id: str | None) -> int:
    if not run_id:
        return -1
    live = read_live_status(RUNS_DIR / run_id / "live_status.json") or {}
    try:
        return int(live.get("timesteps") or -1)
    except (TypeError, ValueError):
        return -1


def _score_external_train_hit(
    *,
    pid: int,
    run_guess: str | None,
    cmd: str,
    is_leaf: bool,
    preferred_run: str | None,
) -> tuple:
    """Rank competing train_ppo PIDs so short A/B smokes don't eclipse overnight.

    Higher tuple wins. Prefer: CURRENT_RUN pin → highest live timesteps →
    --resume / --unlimited → leaf worker (skip VecEnv parents) → lower pid.
    """
    pin_hit = 1 if preferred_run and run_guess == preferred_run else 0
    ts = _live_timesteps_for_run(run_guess)
    resume = 1 if "--resume" in cmd else 0
    unlimited = 1 if "--unlimited-timesteps" in cmd else 0
    leaf = 1 if is_leaf else 0
    return (pin_hit, ts, resume, unlimited, leaf, -pid)


def _pick_best_external_hit(
    hits: list[tuple[int, str | None, str, bool]],
) -> tuple[int, str | None] | None:
    """hits: (pid, run_guess, cmd, is_leaf)."""
    if not hits:
        return None
    preferred = _preferred_operator_run_id()
    best = max(
        hits,
        key=lambda h: _score_external_train_hit(
            pid=h[0],
            run_guess=h[1],
            cmd=h[2],
            is_leaf=h[3],
            preferred_run=preferred,
        ),
    )
    return best[0], best[1]


def _find_external_train(*, force: bool = False) -> tuple[int, str | None] | None:
    """Return (pid, run_id_guess) for a live train_ppo not owned by this UI, else None.

    Cached briefly so /api/status (0.75s poll) does not spawn PowerShell every time
    when psutil is missing — that was stalling the UI and aborting responses.

    When several train_ppo processes exist (overnight + short A/B smokes), prefer the
    operator pin / highest live_status timesteps / --resume rather than arbitrary
    process-iteration order (which previously made the UI look like a ~6k reset).
    """
    global _ext_train_cache_ts, _ext_train_cache_val
    now = time.time()
    if (
        not force
        and _ext_train_cache_ts > 0
        and (now - _ext_train_cache_ts) < _EXT_TRAIN_TTL_S
    ):
        return _ext_train_cache_val

    with _state_lock:
        owned = _train_proc.pid if _train_proc is not None and _train_proc.poll() is None else None

    result: tuple[int, str | None] | None = None
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    if psutil is not None:
        hits_raw: list[tuple[int, str | None, str, object]] = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd_list = p.info.get("cmdline") or []
                cmd = " ".join(cmd_list)
            except (psutil.Error, TypeError):
                continue
            if "control_ui" in cmd:
                continue
            if "rl.train_ppo" not in cmd_list and "rl.train_ppo" not in cmd:
                continue
            if "multiprocessing.spawn" in cmd or "spawn_main" in cmd:
                continue
            pid = int(p.info["pid"])
            if owned is not None and pid == owned:
                continue
            run_guess = None
            if "--run_id" in cmd_list:
                try:
                    run_guess = cmd_list[cmd_list.index("--run_id") + 1]
                except (ValueError, IndexError):
                    pass
            hits_raw.append((pid, run_guess, cmd, p))
        if hits_raw:
            hit_pids = {h[0] for h in hits_raw}
            scored: list[tuple[int, str | None, str, bool]] = []
            for pid, run_guess, cmd, p in hits_raw:
                is_leaf = True
                try:
                    child_pids = {c.pid for c in p.children(recursive=False)}
                    if child_pids & hit_pids:
                        is_leaf = False
                except (psutil.Error, TypeError):
                    pass
                scored.append((pid, run_guess, cmd, is_leaf))
            result = _pick_best_external_hit(scored)
        _ext_train_cache_ts = time.time()
        _ext_train_cache_val = result
        return result

    # No psutil: Windows CIM fallback (same source as _sweep_train_ppo).
    if sys.platform != "win32":
        _ext_train_cache_ts = time.time()
        _ext_train_cache_val = None
        return None
    try:
        ps = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { "
                    "  $_.CommandLine -like '*rl.train_ppo*' -and "
                    "  $_.CommandLine -notlike '*control_ui*' -and "
                    "  $_.CommandLine -notlike '*spawn_main*' "
                    "} | ForEach-Object { "
                    "  \"$($_.ProcessId)`t$($_.CommandLine)\" "
                    "}"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        _ext_train_cache_ts = time.time()
        _ext_train_cache_val = None
        return None
    scored: list[tuple[int, str | None, str, bool]] = []
    for line in (ps.stdout or "").splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        pid_s, cl = line.split("\t", 1)
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        if owned is not None and pid == owned:
            continue
        run_guess = None
        parts = cl.split()
        for i, tok in enumerate(parts):
            if tok == "--run_id" and i + 1 < len(parts):
                run_guess = parts[i + 1].strip("\"'")
                break
        scored.append((pid, run_guess, cl, True))
    result = _pick_best_external_hit(scored)
    _ext_train_cache_ts = time.time()
    _ext_train_cache_val = result
    return result


def _train_busy(*, force: bool = True) -> str | None:
    """Plain-English reason why a new Start/Continue must be refused, else None.

    ``force=False`` reuses the external-scan cache — good enough for preview,
    which must never shell out to PowerShell on every keystroke.

    Start/Continue must refuse (never kill) when any of: UI-owned proc, external
    train_ppo PID, or a live models/*/train.lock is present.
    """
    global _train_run_id, _train_log

    with _state_lock:
        proc = _train_proc
        rid = _train_run_id
    if proc is not None and proc.poll() is None:
        return (
            f"Training already running: {rid or '?'} (pid={proc.pid}). "
            "Status box shows RUNNING - use Stop first to restart."
        )

    ext = _find_external_train(force=force)
    if ext is not None:
        pid, run_guess = ext
        # Adopt external run into UI status so Refresh / Start messaging stay consistent.
        with _state_lock:
            if run_guess:
                _train_run_id = run_guess
                cand = LOGS_DIR / f"{run_guess}.log"
                if cand.is_file():
                    _train_log = cand
        live_locks = find_live_run_locks(MODELS_DIR)
        multi = ""
        if len(live_locks) > 1:
            ids = ", ".join(str(h.get("run_id")) for h in live_locks[:8])
            multi = f" ({len(live_locks)} live locks: {ids})"
        return (
            f"Training already running outside this UI: pid={pid}"
            + (f" run_id={run_guess}" if run_guess else "")
            + multi
            + ". Status should show RUNNING - use Stop first to restart "
            "(or Focus a run in Live runs - dual-Start from this panel stays refused)."
        )

    # PID scan can false-negative; a live train.lock is still ownership proof.
    live_locks = find_live_run_locks(MODELS_DIR)
    if live_locks:
        ids = ", ".join(str(h.get("run_id")) for h in live_locks[:8])
        hit = live_locks[0]
        return (
            f"Training lock held: {len(live_locks)} live — {ids} "
            f"(showing {hit.get('run_id')} pid={hit.get('pid')}). "
            "Start refused while any lock is live; Focus selects status/Continue pin only."
        )
    return None


def _log_tail(log_path: Path, n: int = 4) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(could not read log)"
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:]) if lines else "(empty log)"


def _mark_live_status_starting(run_id: str, *, action: str) -> None:
    """Wipe frozen crash metrics so the banner does not look like a new failure."""
    global _prev_timesteps, _prev_ts_changed_at
    status_dir = RUNS_DIR / run_id
    try:
        status_dir.mkdir(parents=True, exist_ok=True)
        write_live_status(
            status_dir / "live_status.json",
            {
                "run_id": run_id,
                "phase": "starting",
                "timestamp": datetime.now().astimezone().isoformat(),
                "unix_time": time.time(),
                "timesteps": None,
                "note": f"{action}: prior live_status cleared - waiting for trainer heartbeat",
            },
        )
    except OSError as exc:
        print(f"[control_ui] could not clear live_status for {run_id}: {exc}")
    with _state_lock:
        _prev_timesteps = None
        _prev_ts_changed_at = time.time()


def _spawn_train(
    argv_tail: list[str],
    run_id: str,
    *,
    n_envs: int,
    action: str,
    log_mode: str = "w",
) -> str:
    """Spawn ``python -m rl.train_ppo`` and report honestly if it dies on arrival."""
    global _train_proc, _train_run_id, _train_log, _train_n_envs, _last_exit_code

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{run_id}.log"

    # Clear stale crash trail before spawn so UI never shows old timesteps as "this" crash.
    _mark_live_status_starting(run_id, action=action)

    cmd = [_python(), "-u", "-m", "rl.train_ppo", *argv_tail]
    try:
        log_f = open(log_path, log_mode, encoding="utf-8")
    except OSError as e:
        return _set_msg(f"{action} failed: cannot open log {log_path}: {e}")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(AUTODRIVE_PY),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
    except OSError as e:
        try:
            log_f.close()
        except OSError:
            pass
        return _set_msg(f"{action} failed: could not spawn train_ppo: {e}")

    with _state_lock:
        _train_proc = proc
        _train_run_id = run_id
        _train_log = log_path
        _train_n_envs = _clamp_n_envs(n_envs)
        _last_exit_code = None
    # Banner/Focus must follow THIS start — a leftover overnight pin shows
    # early_stopped @ 1.3M + "Stale" while the new train is actually learning.
    _set_operator_run_pin(run_id)
    _invalidate_ext_train_cache()

    # Keep child stdout open via inherited handle; close only our duplicate.
    try:
        log_f.close()
    except OSError:
        pass

    # Catch immediate import / obs_dim / lock / holdout refusals so the UI reports failure.
    time.sleep(0.8)
    if proc.poll() is not None:
        with _state_lock:
            _train_proc = None
            _last_exit_code = proc.returncode
        return _set_msg(
            f"{action} FAILED (exited code={proc.returncode}): {run_id} - {_log_tail(log_path)}"
        )

    tb_msg = _ensure_tensorboard()
    return _set_msg(
        f"{action}: {run_id} (n_envs={_clamp_n_envs(n_envs)} pid={proc.pid}). "
        f"Log: {log_path.name}. {tb_msg}"
    )


def _next_run_id(map_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_ppo_gym_{map_id}_hard"


def _start_train(
    map_id: str,
    timesteps: int,
    n_envs: int,
    *,
    allow_holdout: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 0.5,
    race_eval_every: int = 0,
    early_stop_warmup_evals: int | None = None,
    early_stop_min_timesteps: int | None = None,
    select_timeout: float | None = None,
    stop_on_budget: bool = True,
    collision_first: bool = False,
    speed_gate: bool = False,
    fast_probe: bool = False,
) -> str:
    busy = _train_busy()
    if busy:
        return _set_msg(busy)

    budget = resolve_stop_budget(
        stop_on_budget=stop_on_budget,
        timesteps=timesteps,
        early_stop_patience=early_stop_patience,
    )
    if not budget["ok"]:
        return _set_msg(budget["error"] or "Refuse Start: invalid stop budget")

    allowed, guard_msg = start_guard(map_id, allow_holdout=allow_holdout, maps_dir=MAPS_DIR)
    if not allowed:
        return _set_msg(guard_msg)

    # NEVER _kill_train_tree() here. Busy false-negative + kill = dead overnight.
    # Refuse only; operator must Stop explicitly to kill.

    n_envs = _clamp_n_envs(n_envs)
    vec_env = "subproc" if n_envs > 1 else "dummy"
    run_id = _next_run_id(map_id)

    owner = lock_owner(MODELS_DIR, run_id)
    if owner and owner.get("alive"):
        return _set_msg(
            f"Refuse Start: models/{run_id}/train.lock held by live pid={owner.get('pid')}."
        )

    patience = int(budget["patience"])
    min_imp = max(0.0, float(early_stop_min_improve))
    eval_every = max(0, int(race_eval_every))
    argv = [
        "--map",
        str(map_id),
        "--timesteps",
        str(int(budget["effective_timesteps"])),
        "--device",
        "auto",
        "--run_id",
        run_id,
        "--n-envs",
        str(n_envs),
        "--vec-env",
        vec_env,
        "--n-steps",
        "2048",
        "--batch-size",
        "1024",
        "--n-epochs",
        "10",
        "--net-arch",
        "512,512",
        "--checkpoint-every",
        "25000",
    ]
    if budget["unlimited"]:
        argv.append("--unlimited-timesteps")
    if allow_holdout:
        argv.append("--allow-holdout")
    if collision_first:
        argv.append("--collision-first")
    if speed_gate:
        argv.append("--speed-gate")
    if fast_probe:
        # NON-OFFICIAL progress probe — never promotes / never ticks patience.
        argv.extend(["--fast-probe-every", "10000", "--fast-probe-timeout", "30"])
    if patience > 0:
        argv.extend(["--early-stop-patience", str(patience)])
        argv.extend(["--early-stop-min-improve", f"{min_imp:g}"])
        if early_stop_warmup_evals is not None:
            argv.extend(["--early-stop-warmup-evals", str(int(early_stop_warmup_evals))])
        if early_stop_min_timesteps is not None:
            argv.extend(["--early-stop-min-timesteps", str(int(early_stop_min_timesteps))])
    if eval_every > 0:
        argv.extend(["--race-eval-every", str(eval_every)])
    if select_timeout is not None and float(select_timeout) > 0:
        argv.extend(["--select-timeout", f"{float(select_timeout):g}"])

    msg = _spawn_train(argv, run_id, n_envs=n_envs, action="Training started")
    if guard_msg and "FAILED" not in msg:
        msg = _set_msg(f"{guard_msg} {msg}")
    return msg


def _continue_train(
    run_id: str,
    timesteps: int,
    n_envs: int,
    *,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 0.5,
    race_eval_every: int = 0,
    early_stop_warmup_evals: int | None = None,
    early_stop_min_timesteps: int | None = None,
    select_timeout: float | None = None,
    stop_on_budget: bool = True,
) -> str:
    """Resume the last complete checkpoint of an existing run (same run_id)."""
    busy = _train_busy()
    if busy:
        return _set_msg(busy)

    rid = (run_id or "").strip()
    if not rid:
        rows = list_run_models(MODELS_DIR)
        resumable = [r for r in rows if r.get("checkpoint")]
        if not resumable:
            return _set_msg("Continue: no run with a complete checkpoint under models/")
        rid = str(resumable[0]["run_id"])

    owner = lock_owner(MODELS_DIR, rid)
    if owner and owner.get("alive"):
        return _set_msg(
            f"Refuse Continue: {rid} is locked by live pid={owner.get('pid')}. "
            "Same run is healthy — Stop that train first (do not Double-Continue)."
        )
    # A crashed / killed trainer leaves its lock behind; train_ppo would then
    # refuse the resume as a dual writer.
    clear_stale_lock(MODELS_DIR, rid)

    n_envs = _clamp_n_envs(n_envs)
    # Reuse the run's own map list; a resume must not silently fall back to map0.
    prior_maps = run_map_ids(MODELS_DIR, rid)
    argv, why = continue_train_argv(
        run_id=rid,
        models_dir=MODELS_DIR,
        timesteps=int(timesteps),
        n_envs=n_envs,
        map_id=",".join(prior_maps) if prior_maps else None,
        stop_on_budget=stop_on_budget,
        early_stop_patience=early_stop_patience,
        early_stop_min_improve=early_stop_min_improve,
        race_eval_every=race_eval_every,
        early_stop_warmup_evals=early_stop_warmup_evals,
        early_stop_min_timesteps=early_stop_min_timesteps,
        select_timeout=select_timeout,
    )
    if not argv:
        return _set_msg(f"Continue refused: {why}")

    ckpt, ckpt_msg = resolve_model_choice(MODELS_DIR, rid, "checkpoint")
    if ckpt is not None:
        ok, precheck_msg = precheck_model(ckpt)
        if not ok:
            return _set_msg(f"Continue refused: {precheck_msg}")

    # NEVER _kill_train_tree() on Continue. If the same run is healthy we already
    # refused above; killing after a busy false-negative murders the overnight.
    msg = _spawn_train(argv, rid, n_envs=n_envs, action="Continue training", log_mode="a")
    return _set_msg(f"{why} ({ckpt_msg}). {msg}")


def _sweep_watch() -> int:
    """Kill any stray ``python -m rl.watch`` (and UI-tracked watch PIDs).

    Never called from status refresh — only from Stop Watch / Open Watch (replace).
    """
    killed = 0
    with _state_lock:
        tracked = list(_watch_pids)
        _watch_pids.clear()
    for pid in tracked:
        if _kill_pid_tree(pid):
            killed += 1
    try:
        import psutil

        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd_list = p.info.get("cmdline") or []
                cmd = " ".join(cmd_list)
            except (psutil.Error, TypeError):
                continue
            if "control_ui" in cmd:
                continue
            if "rl.watch" not in cmd and not (
                "watch.py" in cmd and "rl" in cmd.lower()
            ):
                continue
            try:
                kids = p.children(recursive=True)
                for c in kids:
                    try:
                        c.kill()
                    except psutil.Error:
                        pass
                p.kill()
                killed += 1
            except psutil.Error:
                pass
        return killed
    except ImportError:
        pass

    # No psutil: Windows CIM fallback (same idea as _sweep_train_ppo).
    if sys.platform == "win32":
        ps = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "$n=0; Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { "
                    "  ($_.CommandLine -like '*rl.watch*' -or $_.CommandLine -like '*watch.py*') -and "
                    "  $_.CommandLine -notlike '*control_ui*' "
                    "} | ForEach-Object { "
                    "  taskkill /PID $_.ProcessId /T /F 2>$null | Out-Null; $n++ "
                    "}; Write-Output $n"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        try:
            killed += int((ps.stdout or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
    return killed


def _stop_watch() -> str:
    n = _sweep_watch()
    if n:
        return _set_msg(f"Stopped Watch ({n} process tree(s) killed)")
    return _set_msg("No Watch process found (already closed?)")


def _launch_watch(map_id: str, n_envs: int, *, model_path: Path | None = None) -> str:
    """Launch ``rl.watch``: ``--follow`` twins by default, or one saved policy zip.

    Prefer ``live_status.json`` n_envs when training has written it; else the
    UI slider. Falls back to this-session Start value only while train is alive
    and status has not landed yet.
    """
    global _watch_pids, _watch_opened
    # Replace any stuck old watch windows before opening a new one.
    _sweep_watch()

    if model_path is not None:
        cmd = [
            _python(),
            "-u",
            "-m",
            "rl.watch",
            "--map",
            str(map_id),
            "--policy",
            "ppo",
            "--model",
            str(model_path),
            "--every",
            "5",
            "--n-envs",
            "1",
            "--no-beams",
            "--deterministic",
            "--compact",
        ]
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
        proc = subprocess.Popen(
            cmd,
            cwd=str(AUTODRIVE_PY),
            creationflags=creationflags,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        )
        with _state_lock:
            _watch_pids.append(proc.pid)
            _watch_opened = True
        return (
            f"watch replay of {Path(model_path).name} on {map_id} pid={proc.pid} "
            "(q/Esc in the map window, or Stop Watch)."
        )

    slider_n = _clamp_n_envs(n_envs)
    chosen = slider_n
    try:
        st = _status_payload()
        live = st.get("live") or {}
        if live.get("n_envs") is not None:
            chosen = _clamp_n_envs(int(live["n_envs"]))
        elif st.get("train_alive"):
            with _state_lock:
                chosen = _clamp_n_envs(int(_train_n_envs))
    except Exception:
        chosen = slider_n
    n_envs = _clamp_n_envs(chosen)
    cmd = [
        _python(),
        "-u",
        "-m",
        "rl.watch",
        "--follow",
        "--map",
        str(map_id),
        "--every",
        "8",
        "--n-envs",
        str(n_envs),
        "--no-beams",
        "--compact",
    ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        cmd,
        cwd=str(AUTODRIVE_PY),
        creationflags=creationflags,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )
    with _state_lock:
        _watch_pids.append(proc.pid)
        _watch_opened = True
    return _set_msg(
        f"watch --follow ({n_envs} colored twins) pid={proc.pid}. "
        "Click map window -> q/Esc or X; or press Stop Watch."
    )


def _stop_all() -> str:
    """Stop training tree + UI-spawned TensorBoard, then verify and say what really died."""
    global _stopping, _last_exit_code

    with _state_lock:
        proc = _train_proc
        rid = _train_run_id
        _stopping = True
    try:
        was_running = bool(proc is not None and proc.poll() is None) or (
            _find_external_train(force=True) is not None
        )
        n_train = _kill_train_tree()
        n_tb = _kill_ui_tensorboard()
        with _state_lock:
            _last_exit_code = None  # operator-requested kill is not a crash

        time.sleep(0.4)
        survivor = _find_external_train(force=True)

        lock_note = ""
        if survivor is not None:
            head = (
                f"Stop INCOMPLETE: train pid={survivor[0]} still alive after "
                f"{n_train} kill attempt(s) - kill it in Task Manager"
            )
        elif not was_running:
            head = "Stop: no trainer was running (nothing to kill)"
        else:
            head = f"Stopped train{f' {rid}' if rid else ''} - {n_train} process tree(s), verified gone"
            # A killed trainer never releases its own lock; without this the run
            # looks permanently owned and Continue refuses to resume it.
            if rid:
                cleared = clear_stale_lock(MODELS_DIR, rid)
                if cleared:
                    lock_note = f" ({cleared})"
        head += lock_note

        tb_note = (
            "TensorBoard stopped (ui-spawned)"
            if n_tb
            else "TensorBoard left alone (not ui-spawned or already dead)"
        )
        return _set_msg(f"{head}. {tb_note}. Watch windows untouched (use Stop Watch).")
    finally:
        with _state_lock:
            _stopping = False


def _generate_maps(count: int, seed: int) -> str:
    """Procedurally generate maps, refusing to clobber existing / sealed tracks."""
    global _maps_cache
    if not _mapgen_lock.acquire(blocking=False):
        return _set_msg("Map generation already in progress...")
    try:
        result = generate_maps_op(MAPS_DIR, count=count, seed=seed)
        if result.get("ok"):
            _maps_cache = None
            try:
                _prewarm_map_thumbs()
            except Exception as e:
                return _set_msg(f"{result['msg']} (thumbnail warning: {e})")
        return _set_msg(str(result.get("msg")))
    finally:
        _mapgen_lock.release()


def _delete_model(run_id: str) -> str:
    with _state_lock:
        active = _train_run_id if (_train_proc is not None and _train_proc.poll() is None) else None
    try:
        return _set_msg(delete_run_model(MODELS_DIR, run_id, active_run_id=active))
    except OSError as e:
        return _set_msg(f"Delete failed for {run_id}: {e}")


def _watch_model(run_id: str, which: str, map_id: str) -> str:
    """Load a saved policy into Watch (contracts prechecked before we spawn)."""
    path, why = resolve_model_choice(MODELS_DIR, run_id, which)
    if path is None:
        return _set_msg(f"Load refused: {why}")
    ok, precheck_msg = precheck_model(path)
    if not ok:
        return _set_msg(precheck_msg)

    maps = run_map_ids(MODELS_DIR, run_id)
    chosen_map = maps[0] if maps else (map_id or "map0")
    launch_msg = _launch_watch(chosen_map, _clamp_n_envs(1), model_path=path)
    return _set_msg(f"Loaded {run_id}/{why} on {chosen_map} - {precheck_msg}. {launch_msg}")


def _status_age_s(live: dict | None, status_path: str | None) -> float | None:
    """Seconds since live_status was written (payload clock first, file mtime fallback)."""
    if live:
        unix_time = live.get("unix_time")
        try:
            if unix_time is not None:
                return max(0.0, time.time() - float(unix_time))
        except (TypeError, ValueError):
            pass
    if status_path:
        try:
            return max(0.0, time.time() - Path(status_path).stat().st_mtime)
        except OSError:
            return None
    return None


def _track_timesteps(timesteps: int | None) -> tuple[int | None, float | None]:
    """Remember the previous timesteps value + how long it has been frozen."""
    global _prev_timesteps, _prev_ts_changed_at
    now = time.time()
    with _state_lock:
        prev = _prev_timesteps
        if timesteps is None:
            return prev, None
        if prev is None or int(timesteps) != int(prev):
            _prev_timesteps = int(timesteps)
            _prev_ts_changed_at = now
            return prev, 0.0
        frozen_for = now - _prev_ts_changed_at if _prev_ts_changed_at else None
    return prev, frozen_for


def _status_payload() -> dict:
    global _last_exit_code, _train_run_id, _train_log

    with _state_lock:
        proc = _train_proc
        ui_run_id = _train_run_id
        log = str(_train_log) if _train_log else None
        msg = _last_msg
        ui_n_envs = _train_n_envs
        tb = _tb_proc
        stopping = _stopping
        exit_code = _last_exit_code
        watch_opened = _watch_opened

    # Trainer exited on its own since the last poll — remember why.
    if proc is not None and proc.poll() is not None and exit_code is None and not stopping:
        exit_code = proc.returncode
        with _state_lock:
            _last_exit_code = exit_code

    ui_alive = bool(proc is not None and proc.poll() is None)
    train_pid = proc.pid if proc and ui_alive else None
    train_alive = ui_alive
    # External train_ppo still counts as RUNNING, but selection uses CURRENT_RUN /
    # max timesteps — do not latch onto a short A/B smoke via process order.
    if not ui_alive:
        ext = _find_external_train()
        if ext is not None:
            train_alive = True
            train_pid, _ext_guess = ext

    inventory = _list_live_runs(selected_run="")
    selected = _resolve_selected_run(
        ui_owned_run=ui_run_id,
        ui_alive=ui_alive,
        live_rows=inventory,
    )
    # If UI thinks it owns a dead pin (finished overnight) but a different
    # train_ppo is alive, rebind to the live lock / external argv run_id.
    live_ids = {str(r["run_id"]) for r in inventory}
    if train_alive and selected and selected not in live_ids and live_ids:
        selected = _resolve_selected_run(
            ui_owned_run=None,
            ui_alive=False,
            live_rows=inventory,
        )
    run_id = selected
    if selected and not ui_alive:
        with _state_lock:
            # Never clobber UI state with a dead overnight pin while locks live.
            if selected in live_ids or not live_ids:
                if _train_run_id != selected:
                    _train_run_id = selected
            cand = LOGS_DIR / f"{selected}.log"
            if cand.is_file():
                _train_log = cand
                log = str(cand)
        # Prefer selected lock's pid when external.
        for row in inventory:
            if row.get("run_id") == selected and row.get("pid") is not None:
                train_pid = row.get("pid")
                break
    elif selected and ui_alive and ui_run_id and ui_run_id not in live_ids and selected in live_ids:
        # Start landed but Focus/pin still named the finished soak — snap to live.
        with _state_lock:
            _train_run_id = selected
        _set_operator_run_pin(selected)

    tb_ui = bool(tb is not None and tb.poll() is None)
    tb_port = _port_in_use(TB_HOST, TB_PORT)

    live = None
    status_path = None
    if run_id:
        candidate = RUNS_DIR / run_id / "live_status.json"
        live = read_live_status(candidate)
        if live:
            status_path = str(candidate)
    # Only fall back to newest status when we don't already own a specific run_id
    # (avoids showing a previous run's timesteps right after Start).
    if live is None and run_id is None:
        found = find_latest_status(RUNS_DIR)
        if found:
            status_path, live = str(found[0]), found[1]
            run_id = live.get("run_id")
    elif live is None and run_id is not None and not train_alive:
        # Stopped / failed start: still useful to show last known metrics for this run.
        found = find_latest_status(RUNS_DIR)
        if found and (found[1].get("run_id") == run_id):
            status_path, live = str(found[0]), found[1]

    if log is None and run_id:
        cand_log = LOGS_DIR / f"{run_id}.log"
        if cand_log.is_file():
            log = str(cand_log)

    live = live or {}
    phase = live.get("phase")
    # While Start/Continue is spawning, do not surface the prior crash's timesteps.
    if phase == "starting":
        timesteps = None
    else:
        timesteps = live.get("timesteps")
    age_s = _status_age_s(live, status_path)
    prev_ts, frozen_for = _track_timesteps(timesteps)
    n_envs_live = live.get("n_envs", ui_n_envs)

    # Saving is a soft hint: train just dumped latest_model (mtime within a few
    # seconds). Real "phase=saving" would need a live_status field from train_ppo.
    saving = False
    if train_alive:
        latest_hint = live.get("latest_model")
        if latest_hint:
            try:
                saving = (time.time() - Path(str(latest_hint)).stat().st_mtime) < 4.0
            except OSError:
                saving = False

    banner = classify_banner(
        train_alive=train_alive,
        timesteps=timesteps,
        prev_timesteps=prev_ts,
        # "Stale" means the trail stopped moving: age of the file, or how long
        # timesteps have been frozen while the process claims to be alive.
        # Validating deliberately freezes timesteps — classify_banner uses phase.
        status_age_s=max(age_s or 0.0, frozen_for or 0.0) if (age_s or frozen_for) else None,
        saving=saving,
        stopping=stopping,
        exit_code=None if train_alive else exit_code,
        phase=str(phase) if phase else None,
        msg=live.get("msg") or live.get("note"),
    )

    live_runs = _list_live_runs(selected_run=selected or "")
    bound = selected or run_id or live.get("run_id")
    start_warn = _start_lock_warn(live_runs)
    coach = coach_hints(
        live or None,
        watch_opened=watch_opened,
        n_envs=int(n_envs_live or 1),
        live_run_count=len(live_runs),
    )
    bound_note = (
        f"targets: {bound or '-'} (banner/status). "
        "Continue = Models row run_id. Stop = non-protected train_ppo "
        f"(never {PROTECTED_OVERNIGHT_RUN}). Watch --follow = map + n_envs."
    )

    return {
        "contracts_version": CONTRACTS_VERSION,
        "obs_dim": obs_dim(N_LIDAR_DEFAULT),
        "train_alive": train_alive,
        "train_pid": train_pid,
        "run_id": bound,
        "selected_run": bound,
        "bound_run_id": bound,
        "bound_note": bound_note,
        "live_runs": live_runs,
        "live_run_count": len(live_runs),
        "start_lock_warn": start_warn,
        "log_path": log,
        "status_path": status_path,
        "timesteps": timesteps,
        "ep_rew_mean": live.get("ep_rew_mean"),
        "n_envs": n_envs_live,
        "episode_count_estimate": live.get("episode_count_estimate"),
        "collisions_estimate": live.get("collisions_estimate"),
        "crash_rate_estimate": live.get("crash_rate_estimate"),
        "stall_rate_estimate": live.get("stall_rate_estimate"),
        "stalls_estimate": live.get("stalls_estimate"),
        "mean_progress_frac_estimate": live.get("mean_progress_frac_estimate"),
        "progress_probe_kind": live.get("progress_probe_kind") or live.get("probe_kind"),
        "steps_per_sec": live.get("steps_per_sec"),
        "vec_env_active": live.get("vec_env_active"),
        "rollouts": live.get("rollouts"),
        "latest_model": live.get("latest_model"),
        "status_age_s": round(age_s, 1) if age_s is not None else None,
        "banner": banner,
        "coach": coach,
        "exit_code": exit_code,
        "live": live or None,
        "msg": msg,
        "maps": _list_maps(),
        "presets": PRESETS,
        "tensorboard_url": TB_URL,
        "tensorboard_alive": tb_ui or tb_port,
        "tensorboard_ui_owned": tb_ui,
        "tensorboard_pid": tb.pid if tb_ui else None,
    }


def _models_payload() -> dict:
    with _state_lock:
        active = _train_run_id if (_train_proc is not None and _train_proc.poll() is None) else None
    runs = []
    for row in list_run_models(MODELS_DIR)[:40]:
        rid = str(row["run_id"])
        owner = lock_owner(MODELS_DIR, rid)
        flags = run_curriculum_flags(MODELS_DIR, rid)
        runs.append(
            {
                **row,
                "active": rid == active,
                "locked_by": owner.get("pid") if owner and owner.get("alive") else None,
                "timeline": model_timeline(MODELS_DIR, rid)[:6],
                "collision_first": flags["collision_first"],
                "speed_gate": flags["speed_gate"],
            }
        )
    return {"runs": runs, "race_candidate": race_candidate(MODELS_DIR), "active_run_id": active}


def _verify_seals() -> str:
    """Pack seal check for the Maps / Holdout operator path (does not touch train)."""
    report = seal_verify_summary(MAPS_DIR)
    return _set_msg(report["msg"])


def _preview_payload(
    map_id: str,
    timesteps: int,
    n_envs: int,
    *,
    stop_on_budget: bool = True,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 0.5,
    race_eval_every: int = 0,
    collision_first: bool = False,
    speed_gate: bool = False,
    fast_probe: bool = False,
) -> dict:
    n_envs = _clamp_n_envs(n_envs)
    # The real run_id is stamped at Start; preview only shows its shape.
    preview = start_preview(
        map_id,
        int(timesteps),
        n_envs,
        f"<now>_ppo_gym_{map_id}_hard",
        MODELS_DIR,
        MAPS_DIR,
        stop_on_budget=stop_on_budget,
        early_stop_patience=early_stop_patience,
        early_stop_min_improve=early_stop_min_improve,
        race_eval_every=race_eval_every,
        collision_first=bool(collision_first),
        speed_gate=bool(speed_gate),
    )
    preview["fast_probe"] = bool(fast_probe)
    preview["busy_reason"] = _train_busy(force=False)
    live_locks = find_live_run_locks(MODELS_DIR)
    preview["live_run_count"] = len(live_locks)
    if live_locks:
        ids = ", ".join(str(h.get("run_id")) for h in live_locks[:6])
        preview["live_lock_warn"] = (
            f"{len(live_locks)} live train.lock(s): {ids}. "
            "Start refused from this panel while any lock is live "
            "(Focus pin only selects status ranking - dual-Start stays refused)."
        )
    else:
        preview["live_lock_warn"] = None
    return preview


# Bump when the control panel HTML/JS changes so hard-refresh / ?v= can prove freshness.
UI_BUILD = "w10-clear-focus-20260917"

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RL control</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<style>
body{font-family:Consolas,monospace;margin:16px;background:#111;color:#ddd;max-width:920px;line-height:1.35}
h1{font-size:18px;color:#8f8;letter-spacing:0.02em} button{margin:4px 4px 4px 0;padding:8px 12px;font:inherit;cursor:pointer;border-radius:3px;border:1px solid #555;background:#2a2a2a;color:#eee}
button:hover{border-color:#7a7;background:#303830}
input,select{font:inherit;padding:4px;margin:2px 8px 2px 0;background:#222;color:#eee;border:1px solid #555;border-radius:2px}
.row{margin:10px 0} .box{border:1px solid #444;padding:12px;margin:14px 0;background:#1a1a1a;border-radius:3px}
.sec{font-size:13px;font-weight:bold;color:#cde;margin:0 0 8px 0;padding-bottom:4px;border-bottom:1px solid #333;letter-spacing:0.01em}
.sec .small{font-weight:normal;letter-spacing:0}
.budget-row{border:1px solid #485;background:#152018;padding:8px 10px;margin:12px 0;border-radius:3px}
.ok{color:#8f8} .bad{color:#f88} .warn{color:#fc8} label{display:inline-block;min-width:220px}
.hint{color:#888;font-size:12px;margin:4px 0 0 0} input[type=range]{width:260px;vertical-align:middle}
.maprow{display:flex;flex-wrap:wrap;align-items:flex-start;gap:16px;margin-top:8px}
.mapsel{flex:0 0 auto;min-width:180px}
.mapsel select{min-width:160px;display:block;margin-top:4px}
#map_preview_wrap{flex:0 0 auto;border:1px solid #444;background:#000;padding:4px;border-radius:2px}
#map_preview{display:block;width:256px;height:256px;object-fit:contain;background:#000}
#map_preview_label{color:#888;font-size:12px;margin-top:4px}
a.tb{color:#8cf;font-size:15px;font-weight:bold}
#train_toggle.running{background:#522;color:#fcc;border:1px solid #a44}
#train_toggle.stopped{background:#253;color:#cfc;border:1px solid #484}
#banner{font-size:16px;font-weight:bold;padding:10px 12px;border:1px solid #444;background:#181818;border-radius:3px;border-left:4px solid #666}
#banner_reason{font-size:13px;color:#ccc;margin-top:4px;padding-left:2px}
.state-Learning{color:#8f8} .state-Idle{color:#aaa} .state-Saving{color:#8cf}
.state-Validating{color:#8cf} .state-EarlyStop{color:#8cf}
.state-Stopping{color:#fc8} .state-Stale{color:#fc8} .state-Crashed{color:#f88}
#banner.state-Learning{background:#142018;border-left-color:#4a8;border-color:#355}
#banner.state-Validating{background:#121820;border-left-color:#48c;border-color:#346}
#banner.state-EarlyStop{background:#121820;border-left-color:#48c;border-color:#346}
#banner.state-Saving{background:#121820;border-left-color:#48c;border-color:#346}
#banner.state-Stopping,#banner.state-Stale{background:#1c1810;border-left-color:#c84;border-color:#543}
#banner.state-Crashed{background:#201414;border-left-color:#c44;border-color:#533}
#banner.state-Idle{background:#181818;border-left-color:#666}
ul.coach{margin:6px 0 0 18px;padding:0;color:#cc9;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
td,th{border:1px solid #333;padding:3px 5px;text-align:left;vertical-align:top}
th{color:#8cf} td button{padding:2px 6px;margin:1px}
.seal{color:#fc8} .small{font-size:12px;color:#999}
#preview{white-space:pre-wrap;color:#bdb;font-size:13px}
.term{border-bottom:1px dotted #666;cursor:help}
details.glossary{margin:10px 0;border:1px solid #345;background:#14181c;border-radius:3px;padding:6px 10px}
details.glossary summary{cursor:pointer;color:#8cf;font-weight:bold;list-style:none}
details.glossary summary::-webkit-details-marker{display:none}
details.glossary summary::before{content:"? ";color:#8cf}
.glossary dl{margin:8px 0 4px 0;display:grid;grid-template-columns:minmax(120px,180px) 1fr;gap:4px 12px;font-size:12px}
.glossary dt{color:#9cf;margin:0} .glossary dd{margin:0;color:#bbb}
#live_runs{margin:8px 0 4px 0;padding:0;list-style:none;font-size:12px;max-height:160px;overflow:auto;border:1px solid #333;background:#141414}
#live_runs li{display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:4px 8px;border-bottom:1px solid #2a2a2a}
#live_runs li:last-child{border-bottom:none}
#live_runs li.selected{background:#1a2430;border-left:3px solid #48c}
#live_runs li .badge{color:#fc8;font-size:11px}
#live_runs li button{padding:2px 8px;margin:0;font-size:11px}
#bound_targets{color:#9cf;font-size:12px;margin:6px 0 2px 0}
</style></head><body>
<h1>AutoDRIVE RL — control</h1>
<p class="hint">ui build: <b id="ui_build">__UI_BUILD__</b> · hard-refresh (Ctrl+F5) if this looks stale</p>
<details class="glossary" id="glossary">
  <summary>Glossary — plain English for RL knobs</summary>
  <dl>
    <dt title="Not wall-clock seconds">timesteps</dt>
    <dd>Practice steps the cars take. A safety budget / cap — not “minutes until smart.”</dd>
    <dt>early-stop</dt>
    <dd>Quit when validation race score stops improving for N checks (patience). Patience 0 = only stop at timesteps / manual Stop.</dd>
    <dt>patience</dt>
    <dd>How many no-improvement validation checks before EarlyStop. Overnight uses 5. With budget OFF you <b>must</b> set patience &gt; 0 (otherwise nothing stops the run except the 50M hard abort).</dd>
    <dt>budget OFF</dt>
    <dd>Ignore timesteps as a stop — early-stop only. Pair with patience 3–5 + eval every ≥50k overnight. Not “train forever with no eval.”</dd>
    <dt>collision-first</dt>
    <dd>Heavier terminal collision penalty. Use short crash-first Starts before unlocking speed / long budgets. Does not morph a live overnight — new disposable run for A/B.</dd>
    <dt>speed-gate</dt>
    <dd>Hold speed reward until rolling crash rate is under the gate. Pair with collision-first on smokes; do not morph overnight mid-run.</dd>
    <dt>race_eval_every</dt>
    <dd>Mid-train validation cadence (timesteps). Sparse (e.g. 50k) = more learning, shorter Validating pauses. This is <b>not</b> official protocol (official stays 400s × 5 seeds). Do not densify overnight “to use the GPU.”</dd>
    <dt>adjusted_time</dt>
    <dd>Official race score: lap time + 10×collisions (seconds). Lower is better. Never use ep_rew for ranking.</dd>
    <dt>train_ok</dt>
    <dd>Map role safe to Start. Checklist: role=train_ok (not validation/holdout), pack fingerprint present, <code>assert_train_safe</code> would pass. Never auto-Start overnight from a checklist.</dd>
    <dt>holdout</dt>
    <dd>Sealed map reserved for fair final tests. Start refuses it unless you tick “allow sealed.”</dd>
    <dt>validation</dt>
    <dd>Pinned mid-train race map / score used for early-stop. Banner <b>Validating</b> = scheduled race eval — not hung (timesteps freeze briefly).</dd>
    <dt>n_envs</dt>
    <dd>Parallel practice sims (CPU workers). Higher = faster data, more CPU. Watch shows this many twin cars.</dd>
    <dt>rollouts</dt>
    <dd>How many PPO data-collection batches finished. Rough “iteration” counter — not lap count.</dd>
    <dt>Validating</dt>
    <dd>Scheduled mid-train race eval (select≈220s spirit, 1 seed / validation map). Freezes timesteps; promotes best_model / ticks patience. Official leaderboard stays 400s × 5 seeds — never shorten those to “go faster.”</dd>
    <dt>EarlyStop</dt>
    <dd>Clean exit because learning plateaued (patience). Exit code 0 with a reason — not a crash.</dd>
    <dt>Watch lag-behind</dt>
    <dd>Separate window replaying latest weights. Not live train poses; KPIs are unofficial. Compact strip = phase / crash% / ts|/s + race KPIs.</dd>
    <dt>fast probe</dt>
    <dd>Optional short NON-OFFICIAL progress sniff. Never promotes / never ticks patience. Leave off overnight.</dd>
  </dl>
</details>
<details class="glossary" id="bridge_card">
  <summary>Bridge-readiness (observe only — P3)</summary>
  <p class="hint" style="margin:8px 0 4px 0">Gym→AutoDRIVE transfer constraint (RESEARCH §5) — <b>document only this 9h</b>; no Jetson/latency/bridge code.</p>
  <ul class="hint" style="margin:4px 0 8px 18px;padding:0">
    <li>Freeze contracts <code>2.0.0</code> ABI (obs dim / beams) — refuse-load on mismatch.</li>
    <li>Shaping may use map GT; <b>race obs may not</b> (no IPS/pose/progress-in-obs shortcuts).</li>
    <li>Keep light LiDAR DR; do not invent a second DR stack for gym wins.</li>
    <li>Bridge <code>:4567</code> = Phase 3 <b>after</b> sealed holdout beat-FTG — not this overnight.</li>
  </ul>
</details>
<div class="box">
  <div id="banner" class="state-Idle" title="Idle / Learning / Validating (= scheduled race eval — not hung) / EarlyStop / Stopping / Crashed. Mid-train Validating ≠ official 400s×5.">Idle</div>
  <div id="banner_reason" title="When Validating: scheduled race eval — not hung. Official protocol stays 400s / 5 seeds — do not densify overnight.">no trainer</div>
  <div class="sec" style="margin-top:8px">Live train status <span class="small">(from live_status.json — auto-refresh 0.75s)</span></div>
  <div id="bound_targets">bound: —</div>
  <div class="small" style="margin:4px 0">Live runs (Focus = CURRENT_RUN pin for status ranking; does not Start/Stop/kill)
    <button type="button" onclick="clearFocus()" title="Remove CURRENT_RUN pin. Banner falls back to the UI-owned live train, else highest live timesteps.">Clear focus</button>
  </div>
  <ul id="live_runs"><li class="small">no live train.lock</li></ul>
  <div>train: <span id="alive">?</span> &nbsp; pid: <span id="pid">-</span>
    &nbsp; contracts: <b id="cv">?</b> &nbsp; obs_dim: <b id="od">?</b></div>
  <div>run_id: <b id="rid">-</b></div>
  <div><span class="term" title="Practice steps taken (not wall-clock). Safety budget when Stop-on-budget is ON.">timesteps</span>: <b id="ts">-</b></div>
  <div>ep_rew_mean: <b id="rew">-</b> <span class="small"> — shaping signal only; rank with adjusted_time</span></div>
  <div><span class="term" title="Parallel practice sims / CPU workers. Watch uses this many colored twins.">n_envs</span> (parallel sims): <b id="ne">-</b> &nbsp; vec: <b id="vec">-</b></div>
  <div>steps/sec: <b id="sps">-</b> &nbsp; crash rate: <b id="crash">-</b>
    &nbsp; stall rate: <b id="stall">-</b>
    &nbsp; status age: <b id="age">-</b></div>
  <div>episodes (est.): <b id="eps">-</b> &nbsp; collisions (est.): <b id="cols">-</b></div>
  <div>progress probe (smoke): <b id="prog">-</b>
    <span class="small"> — mean ep progress; not official / never promotes</span></div>
  <div><span class="term" title="PPO data-collection batches finished (iteration-ish). Not lap count.">rollouts</span>: <b id="rolls">-</b></div>
  <div>latest_model: <span id="latest">-</span></div>
  <div>status file: <span id="stpath">-</span></div>
  <div>log: <span id="log">-</span></div>
  <div>msg: <span id="msg" class="warn">-</span></div>
  <p class="hint">Train stays headless. Watch map shows cars only (per-car speed); curves in TensorBoard.
  Watch follow defaults: <code>--every 8 --compact</code> (snappier on weak machines; denser overlay via <code>--no-compact</code>).</p>
</div>
<div class="box">
  <div class="sec">Coach / health <span class="small">(heuristics — official ranking is <span class="term" title="Lap time + 10×collisions; lower is better.">adjusted_time</span>, never ep_rew)</span></div>
  <ul class="coach" id="coach"><li>—</li></ul>
</div>
<div class="box">
  <div class="sec">Train controls</div>
  <div class="row"><label>Map</label></div>
  <div class="maprow">
    <div class="mapsel">
      <label for="map">Track</label>
      <select id="map" onchange="onMapChange()"></select>
      <p class="hint">Select a track; preview updates on the right.</p>
    </div>
    <div>
      <div id="map_preview_wrap">
        <img id="map_preview" alt="map preview" src="">
      </div>
      <div id="map_preview_label">preview: -</div>
    </div>
  </div>

  <div class="row" style="margin-top:10px">
    <label for="gen_count">Generate new maps</label>
    <input id="gen_count" type="number" value="3" min="1" max="10" style="width:60px" title="how many">
    <input id="gen_seed" type="number" value="123" step="1" style="width:100px" title="seed">
    <button type="button" onclick="genMaps()">Generate</button>
  </div>
  <p class="hint">Procedural tracks via the map pack (registered with role + hashes), named
  <code>gen&lt;seed&gt;_N</code>. Existing maps are reused, never redrawn; sealed [HOLDOUT] and the
  [VALIDATION] pin are untouchable. Same seed = same tracks.</p>

  <div class="row" style="margin-top:14px">
    <label>Presets</label>
    <span id="presets"></span>
  </div>

  <div class="budget-row">
    <div class="row" style="margin:0">
      <label for="stop_on_budget"
             title="ON (default): stop when timesteps hit (and early-stop if patience &gt; 0). OFF: ignore timesteps as a stop — early-stop only (patience must be &gt; 0). Hard safety ceiling 50M steps.">Stop on timesteps budget</label>
      <input id="stop_on_budget" type="checkbox" checked onchange="syncBudget(); schedulePreview()"
             title="ON: stop at timesteps (safety cap). OFF: early-stop only — set patience 3–5; timesteps ignored except a 50M hard abort.">
      <span class="small" id="budget_mode_hint">on = stop at timesteps (+ early-stop if patience &gt; 0)</span>
    </div>
    <p class="hint">Recommended: budget ON + timesteps 1–5M (safety) + patience 3–5; <b>or</b> Overnight preset
    (budget OFF + patience 5 + eval every 50k + warmup). Auto eval floor 50k when budget off.</p>
  </div>

  <div class="row">
    <label for="timesteps" title="Maximum practice budget (safety cap) when 'Stop on timesteps budget' is ON. Not wall-clock seconds. Not 'how long until smart'.">Training length (timesteps)</label>
    <input id="timesteps" type="number" value="100000" min="2048" step="1000" style="width:140px"
           oninput="schedulePreview()"
           title="Safety cap when budget stop is ON. Early-stop can finish sooner. Ignored as a stop when budget is OFF (50M hard abort only).">
  </div>
  <p class="hint" id="timesteps_hint">Max practice budget — safety cap so training won't run forever. NOT "how long until smart"
  and NOT wall-clock seconds. 50k–100k = short run; 1–5M = long safety when learning with early-stop. When budget OFF, this field is ignored.</p>
  <div class="row">
    <label for="early_stop" title="Stop when validation race score stops improving for N checks. Try 3–5 for 'stop when done learning'. 0 = only stop at timesteps / manual Stop (requires budget ON).">Early-stop patience</label>
    <input id="early_stop" type="number" value="0" min="0" max="50" step="1" style="width:80px"
           oninput="schedulePreview()"
           title="N validation race-evals with no *meaningful* improvement (see Min improvement). Suggested 3–5. If stops too soon → raise patience or lower Min improvement. If never stops → lower patience / raise Min improvement / ensure mid-train evals fire. Patience 0 + budget OFF is refused.">
  </div>
  <div class="row">
    <label for="min_improve"
           title="Meaningful = beat previous best validation score by at least this much; otherwise the check counts toward patience. Finishers: seconds of adjusted_time (lap + 10·collisions). DNFs: ~1% of lap progress at default 0.5, or the same adj delta. 0 = any tiny race_score_key bump resets patience (old behavior).">Min improvement (s)</label>
    <input id="min_improve" type="number" value="0.5" min="0" max="60" step="0.1" style="width:80px"
           oninput="schedulePreview()"
           title="How much better validation race score must get to reset patience. Finishers: adjusted_time seconds (default 0.5). DNFs: progress ≥ max(0.5%, min×2%) of lap, or adj proxy. Tiny gains still promote best_model but do not reset patience. Suggested 0.5–1.0.">
  </div>
  <div class="row">
    <label for="race_eval_every"
           title="Run a validation race-eval (and patience check) every N timesteps. 0 = auto (~10% of budget, floor 25k; ~25k when budget off) when patience &gt; 0; end-only when patience is 0. Higher N = fewer mid-train pauses. Banner Validating = scheduled race eval — not hung. Official stays 400s×5 — do not densify overnight.">Eval every N timesteps</label>
    <input id="race_eval_every" type="number" value="0" min="0" step="1000" style="width:120px"
           oninput="schedulePreview()"
           title="0 = auto when early-stop is on (~25k when budget off). Overnight prefer 50000 (sparse). Each eval freezes timesteps (Validating ≠ hung). Official protocol still 400s × 5 seeds — never shorten those.">
  </div>
  <p class="hint">Stop when done learning: after N validation race-score checks with no <b>meaningful</b> improvement
  (finishers: ≥ Min improvement s; DNFs: any progress-first key gain). Overnight: patience <b>5</b>, eval every <b>50k</b>,
  mid-train timeout <b>~220s</b> (laps need ~180–210s), warmup so first validations never abort.
  Banner <b>Validating</b> = scheduled race eval — not hung; <b>EarlyStop</b> = clean exit 0 with reason.
  Official leaderboard stays <b>400s × 5 seeds</b> — denser mid-train eval starves learning; never lower official timeouts.</p>

  <div class="row">
    <label for="n_envs" title="How many parallel practice sims (CPU workers). Higher = faster data collection, more CPU. Watch follow opens this many colored twin cars.">Parallel sims / CPU workers</label>
    <input id="n_envs" type="range" min="1" max="32" value="8" oninput="syncN()"
           title="Applies on Start. Stop+Start to change. Weak machine? try 4–8.">
    <b id="n_envs_val">8</b>
  </div>
  <p class="hint">Applies when you press Start. Stop + Start to change.
  Higher = more CPU and faster data collection. Watch uses this many colored twin cars.</p>

  <div class="row">
    <label for="allow_holdout" title="OFF (default): Start refuses sealed [HOLDOUT] and pinned [VALIDATION] maps so official scores stay meaningful. ON = override (claims on those maps are void).">Allow sealed / pinned map</label>
    <input id="allow_holdout" type="checkbox" onchange="schedulePreview()"
           title="Leave off unless you intentionally want to train on holdout/validation.">
    <span class="small">off = Start refuses [HOLDOUT] and [VALIDATION] maps (keeps official scores meaningful)</span>
  </div>
  <div class="row">
    <label for="collision_first" title="Heavier terminal collision penalty (COLLISION_FIRST_PENALTY). Prefer short crash-first runs before unlocking speed / long budgets. Does not morph a live overnight — Start a new disposable run for A/B.">Collision-first curriculum</label>
    <input id="collision_first" type="checkbox" onchange="schedulePreview()">
    <span class="small">on = pass --collision-first (crash rate should drop before long overnight aggression)</span>
  </div>
  <div class="row">
    <label for="speed_gate" title="Hold speed reward until rolling crash rate is under SPEED_GATE_CRASH_RATE. Pair with collision-first on short smokes; do not morph a live overnight.">Speed-gate curriculum</label>
    <input id="speed_gate" type="checkbox" onchange="schedulePreview()">
    <span class="small">on = pass --speed-gate (unlock speed only under crash budget)</span>
  </div>
  <div class="row">
    <label for="fast_probe" title="Optional NON-OFFICIAL mid-train progress probe (short timeout). Writes live_status probe fields only. Does NOT promote best_model, does NOT tick early-stop patience, is NOT race score / FTGΔ. Full 220s select-timeout evals remain the only patience/promote path.">Fast progress probe (NON-OFFICIAL)</label>
    <input id="fast_probe" type="checkbox" onchange="schedulePreview()">
    <span class="small">off by default — never call this a race score; overnight should leave it off</span>
  </div>
  <div class="row">
    <button type="button" onclick="act('verify_seals')" title="Re-hash pack seals (map2/map4 holdouts + validation pin). Does not touch training.">Verify pack seals</button>
    <span class="small" id="seal_msg"> — check before official append / FTGΔ claims</span>
  </div>

  <div class="box" style="background:#151a15">
    <div><b>Start preview</b> <span class="small">(what pressing Start will do)</span></div>
    <div id="preview">-</div>
  </div>

  <div class="row">
    <button id="train_toggle" class="stopped" type="button" onclick="toggleTrain()"
            title="Start refused while any train.lock is live. Stop sweeps non-protected train_ppo only (overnight soak protected).">Start training</button>
    <button type="button" onclick="act('watch')" title="Opens lag-behind Watch (--follow --every 8 --compact). Uses map + n_envs from this panel; KPIs unofficial.">Open watch --follow</button>
    <button type="button" onclick="act('stop_watch')">Stop Watch</button>
  </div>
  <p class="hint" id="op_targets">ops target: status/Focus pin above · Continue = Models row · Stop = non-protected trains (overnight never)</p>
  <div class="row" style="opacity:0.55">
    <button id="auto_train_btn" type="button" disabled title="EXPERIMENTAL preview only — CLI scaffold exists (python -m rl.auto_train). UI spawn deferred until chaos drills. Never starts beside overnight soak.">Auto-train (EXPERIMENTAL)</button>
    <label for="auto_train_enable" title="Opt-in preview only. Does not enable unattended overnight chaining.">Enable preview</label>
    <input id="auto_train_enable" type="checkbox" onchange="syncAutoTrainPreview()"
           title="Check to read the warning. Button stays non-spawning this wave.">
    <span class="small" id="auto_train_hint">disabled — W4 thin CLI only; no UI spawn / no overnight companion</span>
  </div>
  <p class="hint warn" id="auto_train_warn" style="display:none">WARNING: Auto-train UI will not Start/Continue/kill the protected overnight soak.
  Use <code>python -m rl.auto_train --dry-run</code> (or <code>--execute --smoke</code> when idle). Reward mutation and holdout training are refused forever this campaign.</p>
  <p class="hint"><b>Stop</b> = operator halt (clears lock when verified dead) then <b>Continue</b> from last <b>complete</b> zip.
  Do not Task-Manager-kill mid-save — that can corrupt checkpoints. Soft-stop ≠ hard kill.</p>
  <p class="hint">Watch: click the OpenCV map window first, then q/Esc or the window X
  (that ends Watch for good — status refresh does <b>not</b> relaunch it).
  If it ignores keys (Windows focus quirk), press <b>Stop Watch</b> here to kill all watch processes.</p>
</div>
<div class="box">
  <div class="sec">Models <span class="small">(Continue resumes the last complete checkpoint under the same run_id)</span></div>
  <div id="race_candidate" class="small">recommended: -</div>
  <div class="row">
    <button type="button" onclick="loadModels()">Refresh models</button>
    <span class="small">Load = replay that zip in Watch · Delete = remove models/&lt;run_id&gt;/</span>
  </div>
  <div id="models_tbl">loading…</div>
</div>
<div class="box">
  <div class="sec">TensorBoard <span id="tbst">?</span>
    <span id="tbpid" class="small"></span></div>
  <div class="row">
    <a class="tb" id="tburl" href="http://127.0.0.1:6006/" target="_blank" rel="noopener"
       onclick="return openTensorBoard(event)">Open TensorBoard</a>
    &nbsp; <button type="button" onclick="act('tensorboard')">Start TensorBoard</button>
  </div>
  <p class="hint">Start training auto-launches TB if needed. Stop kills train and any TB this UI started
  (external TB on :6006 is left alone). Open link ensures TB is up, then opens http://127.0.0.1:6006/</p>
</div>
<script>
let mapsCache = [];
let trainAlive = false;
let previewTimer = null;
let lastRunId = null;
const STATUS_MS = 750;
function syncN(){ document.getElementById('n_envs_val').textContent = document.getElementById('n_envs').value; }
function syncAutoTrainPreview(){
  const on = document.getElementById('auto_train_enable') && document.getElementById('auto_train_enable').checked;
  const warn = document.getElementById('auto_train_warn');
  const hint = document.getElementById('auto_train_hint');
  const btn = document.getElementById('auto_train_btn');
  if (warn) warn.style.display = on ? 'block' : 'none';
  if (hint) hint.textContent = on
    ? 'preview only — button still does not spawn; chaos drills required for enablement'
    : 'disabled — W4 thin CLI only; no UI spawn / no overnight companion';
  // Keep disabled even when preview checked — refuse live spawn this wave.
  if (btn) btn.disabled = true;
}
function syncBudget(){
  const on = document.getElementById('stop_on_budget').checked;
  const ts = document.getElementById('timesteps');
  const hint = document.getElementById('budget_mode_hint');
  if (ts) {
    ts.disabled = !on;
    ts.style.opacity = on ? '1' : '0.45';
  }
  if (hint) {
    hint.textContent = on
      ? 'on = stop at timesteps (+ early-stop if patience > 0)'
      : 'off = early-stop only (patience must be > 0; 50M hard abort)';
  }
}
function setTrainToggle(alive){
  trainAlive = !!alive;
  const btn = document.getElementById('train_toggle');
  if (!btn) return;
  btn.textContent = trainAlive ? 'Stop training' : 'Start training';
  btn.className = trainAlive ? 'running' : 'stopped';
}
function setPreview(mapId){
  const m = mapsCache.find(x => x.id === mapId);
  const img = document.getElementById('map_preview');
  const lab = document.getElementById('map_preview_label');
  if (m && m.thumb) {
    img.src = m.thumb + '?t=' + Date.now();
    img.style.display = 'block';
    lab.textContent = 'preview: ' + m.id + ' (' + m.thumb + ')';
  } else {
    img.removeAttribute('src');
    img.style.display = 'none';
    lab.textContent = 'preview: ' + (mapId || '-') + ' (missing image)';
  }
}
function onMapChange(){
  setPreview(document.getElementById('map').value);
}
function buildMaps(maps){
  const sel = document.getElementById('map');
  const prev = sel.value;
  mapsCache = maps || [];
  sel.innerHTML = '';
  mapsCache.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = (m.label || m.id) + (m.thumb ? '' : ' (no preview)');
    sel.appendChild(opt);
  });
  if (prev && mapsCache.some(m => m.id === prev)) sel.value = prev;
  else if (mapsCache.length) sel.value = mapsCache[0].id;
  setPreview(sel.value);
}
function buildPresets(presets){
  const host = document.getElementById('presets');
  if (!host || host.dataset.built === '1' || !presets) return;
  Object.keys(presets).forEach(key => {
    const p = presets[key];
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = p.label || key;
    b.onclick = () => {
      document.getElementById('timesteps').value = p.timesteps;
      document.getElementById('n_envs').value = p.n_envs;
      if (p.stop_on_budget === false || p.stop_on_budget === 0) {
        document.getElementById('stop_on_budget').checked = false;
      } else if (p.stop_on_budget === true || p.stop_on_budget === 1) {
        document.getElementById('stop_on_budget').checked = true;
      }
      if (p.early_stop_patience != null) {
        document.getElementById('early_stop').value = p.early_stop_patience;
      }
      if (p.early_stop_min_improve != null) {
        document.getElementById('min_improve').value = p.early_stop_min_improve;
      }
      if (p.race_eval_every != null) {
        document.getElementById('race_eval_every').value = p.race_eval_every;
      }
      // Overnight hint fields stored on window for Start argv (no extra inputs required).
      window.__overnightHint = {
        warmup_evals: p.early_stop_warmup_evals,
        min_timesteps: p.early_stop_min_timesteps,
        select_timeout: p.select_timeout
      };
      // Overnight soak stays collision_first=false; curriculum A/B is a separate Start.
      const cf = document.getElementById('collision_first');
      if (cf && key === 'overnight') cf.checked = false;
      syncBudget();
      syncN();
      schedulePreview();
    };
    host.appendChild(b);
  });
  host.dataset.built = '1';
}
function esc(s){
  return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function schedulePreview(){
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 250);
}
async function refreshPreview(){
  const el = document.getElementById('preview');
  if (!el) return;
  const q = new URLSearchParams({
    map: document.getElementById('map').value,
    timesteps: document.getElementById('timesteps').value,
    n_envs: document.getElementById('n_envs').value,
    early_stop: document.getElementById('early_stop').value,
    min_improve: document.getElementById('min_improve').value,
    race_eval_every: document.getElementById('race_eval_every').value,
    stop_on_budget: document.getElementById('stop_on_budget').checked ? '1' : '0',
    collision_first: document.getElementById('collision_first').checked ? '1' : '0',
    speed_gate: document.getElementById('speed_gate').checked ? '1' : '0',
    fast_probe: document.getElementById('fast_probe').checked ? '1' : '0'
  });
  try {
    const r = await fetch('/api/preview?' + q.toString());
    if (!r.ok) return;
    const p = await r.json();
    const allow = document.getElementById('allow_holdout').checked;
    const evalNote = (p.race_eval_every && p.race_eval_every > 0)
      ? String(p.race_eval_every)
      : 'auto';
    const lines = [
      'map       : ' + (p.map_label || p.map) + '   role=' + (p.map_role || '?'),
      'budget    : ' + (p.budget_label || ('budget: ' + p.timesteps)),
      'patience  : ' + (p.early_stop_patience != null ? p.early_stop_patience : '-'),
      'min_impr  : ' + (p.early_stop_min_improve != null ? p.early_stop_min_improve : '-') + 's adj',
      'eval_every: ' + evalNote,
      'n_envs    : ' + p.n_envs + '   vec=' + p.vec_env,
      'collision : ' + (p.collision_first ? 'FIRST (on)' : 'normal'),
      'speed_gate: ' + (p.speed_gate ? 'on' : 'off'),
      'fast_probe: ' + (p.fast_probe ? 'NON-OFFICIAL on' : 'off'),
      'run_id    : ' + p.run_id,
      'run path  : ' + p.run_path
    ];
    if (p.budget_error) lines.push('REFUSED   : ' + p.budget_error);
    if (p.holdout_warn && !allow) lines.push('REFUSED   : not train-safe — tick "allow sealed / pinned map" to override');
    else if (p.holdout_warn) lines.push('WARNING   : this map is sealed or pinned; official claims on it are void');
    if (p.busy_reason) lines.push('BLOCKED   : ' + p.busy_reason);
    else if (p.live_lock_warn) lines.push('WARN      : ' + p.live_lock_warn);
    if (p.live_run_count != null && p.live_run_count > 0)
      lines.push('live_locks: ' + p.live_run_count);
    el.textContent = lines.join('\\n');
    el.style.color = (p.busy_reason || p.budget_error || (p.holdout_warn && !allow)) ? '#f88' : '#bdb';
  } catch (e) { /* transient */ }
}
async function loadModels(){
  const host = document.getElementById('models_tbl');
  try {
    const r = await fetch('/api/models');
    if (!r.ok) return;
    const j = await r.json();
    const cand = document.getElementById('race_candidate');
    if (j.race_candidate) {
      const c = j.race_candidate;
      let txt = 'recommended (race_score_key / official): ' + c.run_id +
        '  adj=' + c.adjusted_time + '  collisions=' + c.total_collisions;
      if (c.protocol_id) txt += '  protocol=' + c.protocol_id;
      if (c.dnf) txt += '  (DNF — no finished PPO yet)';
      cand.textContent = txt;
      cand.className = 'ok small';
    } else {
      cand.textContent = 'recommended: none yet — no official (non-null) PPO under current protocol';
      cand.className = 'warn small';
    }
    if (!j.runs || !j.runs.length) { host.textContent = 'no models under rl/models/'; return; }
    window.__runsCache = j.runs;
    const bound = window.__boundRunId || null;
    let html = '<table><tr><th>run_id</th><th>artifacts</th><th>timeline</th><th>contracts</th><th>actions</th></tr>';
    j.runs.forEach(run => {
      const rid = esc(run.run_id);
      const arts = [];
      if (run.has_best) arts.push('best');
      if (run.has_latest) arts.push('latest');
      if (run.checkpoint) arts.push('ckpt(' + esc(run.checkpoint.split(/[\\\\/]/).pop()) + ')');
      const state = run.active ? ' <span class="ok">[training]</span>'
        : (run.locked_by ? ' <span class="warn">[locked pid=' + esc(run.locked_by) + ']</span>' : '');
      const focusMark = (bound && run.run_id === bound) ? ' <span class="small">[focused]</span>' : '';
      const tl = (run.timeline || []).slice(0, 4).map(e => {
        const when = e.modified ? String(e.modified).replace('T', ' ').slice(0, 16) : '?';
        return esc(e.kind) + ' ' + esc(e.size_mb) + 'MB @ ' + esc(when);
      }).join('<br>') || '<span class="small">—</span>';
      html += '<tr' + (bound && run.run_id === bound ? ' style="background:#1a2430"' : '') + '><td>' + rid + state + focusMark + '</td><td>' + (arts.join(', ') || '<span class="bad">empty</span>') +
        '</td><td class="small">' + tl + '</td><td>' + esc(run.contracts_version || '?') + '</td><td>' +
        (run.has_best ? '<button type="button" onclick="loadModel(\\'' + rid + '\\',\\'best\\')">Load best</button>' : '') +
        (run.has_latest ? '<button type="button" onclick="loadModel(\\'' + rid + '\\',\\'latest\\')">Load latest</button>' : '') +
        (run.checkpoint ? '<button type="button" onclick="continueRun(\\'' + rid + '\\')">Continue</button>' : '') +
        '<button type="button" onclick="deleteModel(\\'' + rid + '\\')">Delete</button>' +
        '</td></tr>';
    });
    host.innerHTML = html + '</table>';
  } catch (e) { host.textContent = 'models load failed: ' + e; }
}
function loadModel(runId, which){ act('load_model', {run_id: runId, which: which}); }
function continueRun(runId){
  const budgetOn = document.getElementById('stop_on_budget').checked;
  const stepsNote = budgetOn
    ? ('+' + document.getElementById('timesteps').value + ' steps')
    : 'early-stop only (budget off)';
  const run = (window.__runsCache || []).find(r => r.run_id === runId) || {};
  const cf = run.collision_first ? 'on' : 'off';
  const sg = run.speed_gate ? 'on' : 'off';
  if (!confirm(
    'Continue ' + runId + ' from its last COMPLETE checkpoint (' + stepsNote + ')?\\n\\n' +
    'Soft-stop first if RUNNING (UI Stop clears lock when verified dead).\\n' +
    'Curriculum from config.json: collision_first=' + cf + ' speed_gate=' + sg + '\\n' +
    '(Continue restores those flags — does not read the Start checkboxes.)'
  )) return;
  act('continue', {run_id: runId}).then(loadModels);
}
function deleteModel(runId){
  if (!confirm('Delete models/' + runId + '/ permanently?')) return;
  act('delete_model', {run_id: runId}).then(loadModels);
}
function genMaps(){
  act('gen_maps', {
    count: document.getElementById('gen_count').value,
    seed: document.getElementById('gen_seed').value
  }).then(refreshPreview);
}
function renderLiveRuns(rows, boundId){
  const el = document.getElementById('live_runs');
  if (!el) return;
  const list = rows || [];
  if (!list.length) {
    el.innerHTML = '<li class="small">no live train.lock</li>';
    return;
  }
  el.innerHTML = list.map(r => {
    const rid = r.run_id || '?';
    const sel = r.selected || r.focused || rid === boundId;
    const badge = r.overnight || r.protected ? ' <span class="badge">[overnight]</span>' : '';
    const ts = r.timesteps != null ? r.timesteps : '-';
    const sps = r.steps_per_sec != null ? Number(r.steps_per_sec).toFixed(0) : '-';
    return '<li class="' + (sel ? 'selected' : '') + '">'
      + '<span><b>' + esc(rid) + '</b>' + badge + '</span>'
      + '<span class="small">pid=' + esc(r.pid) + ' · ' + esc(r.phase) + ' · ts=' + esc(ts) + ' · /s=' + esc(sps) + '</span>'
      + (sel ? '<span class="small">focused</span> <button type="button" onclick="clearFocus()">Unfocus</button>'
           : '<button type="button" onclick="focusRun(\\'' + esc(rid).replace(/'/g, '') + '\\')">Focus</button>')
      + '</li>';
  }).join('');
}
function focusRun(runId){
  act('focus_run', {run_id: runId}).then(refresh);
}
function clearFocus(){
  act('clear_focus').then(refresh);
}
async function refresh(){
  try {
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const j = await r.json();
    document.getElementById('cv').textContent = j.contracts_version;
    document.getElementById('od').textContent = j.obs_dim;
    const a = document.getElementById('alive');
    a.textContent = j.train_alive ? 'RUNNING' : 'stopped';
    a.className = j.train_alive ? 'ok' : 'bad';
    setTrainToggle(j.train_alive);
    document.getElementById('pid').textContent = j.train_pid || '-';
    document.getElementById('rid').textContent = j.run_id || '-';
    window.__liveRunCount = j.live_run_count != null ? j.live_run_count : ((j.live_runs || []).length);
    window.__startLockWarn = j.start_lock_warn || null;
    window.__boundRunId = j.bound_run_id || j.run_id || null;
    const boundEl = document.getElementById('bound_targets');
    if (boundEl) {
      const n = window.__liveRunCount;
      boundEl.textContent = (j.bound_note || ('bound: ' + (j.bound_run_id || j.run_id || '—')))
        + (n ? (' · live_locks=' + n) : '');
    }
    const opT = document.getElementById('op_targets');
    if (opT) {
      opT.textContent = 'ops: status/Stop/Watch→' + (j.bound_run_id || j.run_id || '—')
        + ' · Continue=Models row · Stop=non-protected (overnight never)';
    }
    renderLiveRuns(j.live_runs || [], j.bound_run_id || j.run_id);
    document.getElementById('ts').textContent = j.timesteps != null ? j.timesteps : '-';
    document.getElementById('rew').textContent = j.ep_rew_mean != null ? Number(j.ep_rew_mean).toFixed(3) : '-';
    document.getElementById('ne').textContent = j.n_envs != null ? j.n_envs : '-';
    document.getElementById('vec').textContent = j.vec_env_active || '-';
    document.getElementById('sps').textContent = j.steps_per_sec != null ? Number(j.steps_per_sec).toFixed(1) : '-';
    document.getElementById('crash').textContent =
      j.crash_rate_estimate != null ? (100 * Number(j.crash_rate_estimate)).toFixed(0) + '%' : '-';
    const stallEl = document.getElementById('stall');
    if (stallEl) {
      stallEl.textContent =
        j.stall_rate_estimate != null ? (100 * Number(j.stall_rate_estimate)).toFixed(0) + '%' : '-';
    }
    document.getElementById('age').textContent = j.status_age_s != null ? j.status_age_s + 's' : '-';
    const ban = j.banner || {state: '?', reason: ''};
    const banEl = document.getElementById('banner');
    banEl.textContent = ban.state + (j.exit_code != null && !j.train_alive ? ' (exit ' + j.exit_code + ')' : '');
    banEl.className = 'state-' + ban.state;
    document.getElementById('banner_reason').textContent = ban.reason || '';
    const coachEl = document.getElementById('coach');
    const hints = j.coach || [];
    const coachSig = hints.join('|');
    if (coachEl.dataset.sig !== coachSig) {
      coachEl.innerHTML = hints.map(h => '<li>' + esc(h) + '</li>').join('') || '<li>—</li>';
      coachEl.dataset.sig = coachSig;
    }
    buildPresets(j.presets);
    if (j.run_id !== lastRunId) { lastRunId = j.run_id; loadModels(); }
    document.getElementById('eps').textContent = j.episode_count_estimate != null ? j.episode_count_estimate : '-';
    document.getElementById('cols').textContent = j.collisions_estimate != null ? j.collisions_estimate : '-';
    const progEl = document.getElementById('prog');
    if (progEl) {
      if (j.mean_progress_frac_estimate != null) {
        progEl.textContent = (100 * Number(j.mean_progress_frac_estimate)).toFixed(1) + '% (' +
          (j.progress_probe_kind || 'smoke') + ')';
      } else {
        progEl.textContent = '-';
      }
    }
    document.getElementById('rolls').textContent = j.rollouts != null ? j.rollouts : '-';
    document.getElementById('latest').textContent = j.latest_model || '-';
    document.getElementById('stpath').textContent = j.status_path || '-';
    document.getElementById('log').textContent = j.log_path || '-';
    document.getElementById('msg').textContent = j.msg || '';
    document.getElementById('tburl').href = j.tensorboard_url || 'http://127.0.0.1:6006/';
    const tbs = document.getElementById('tbst');
    if (j.tensorboard_alive) {
      tbs.textContent = j.tensorboard_ui_owned ? 'RUNNING (ui)' : 'RUNNING (port)';
      tbs.className = 'ok';
    } else {
      tbs.textContent = 'stopped';
      tbs.className = 'bad';
    }
    document.getElementById('tbpid').textContent = j.tensorboard_pid ? (' pid=' + j.tensorboard_pid) : '';
    if (j.maps && j.maps.length) {
      const sig = j.maps.map(m => m.id + ':' + (m.thumb || '')).join('|');
      const old = mapsCache.map(m => m.id + ':' + (m.thumb || '')).join('|');
      if (sig !== old) buildMaps(j.maps);
    }
  } catch (e) {
    /* poll cancelled / transient — next interval retries */
  }
}
async function openTensorBoard(ev){
  if (ev) ev.preventDefault();
  const msgEl = document.getElementById('msg');
  msgEl.className = 'warn';
  msgEl.textContent = 'Ensuring TensorBoard…';
  try {
    const body = new URLSearchParams({
      op: 'tensorboard',
      map: document.getElementById('map').value,
      timesteps: document.getElementById('timesteps').value,
      n_envs: document.getElementById('n_envs').value
    });
    const r = await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body});
    const j = await r.json();
    const url = (j.tensorboard_url || 'http://127.0.0.1:6006/').replace(/\\/?$/, '/');
    document.getElementById('tburl').href = url;
    msgEl.textContent = j.msg || ('Open ' + url);
    msgEl.className = /fail|error|exited/i.test(msgEl.textContent) ? 'bad' : 'ok';
    await refresh();
    msgEl.textContent = j.msg || ('Open ' + url);
    window.open(url, '_blank', 'noopener');
  } catch (e) {
    msgEl.textContent = 'TensorBoard open failed: ' + e;
    msgEl.className = 'bad';
  }
  return false;
}
function toggleTrain(){
  if (trainAlive) {
    const rid = document.getElementById('rid').textContent || '?';
    const pid = document.getElementById('pid').textContent || '?';
    if (!confirm(
      'Stop training targeting selected run_id=' + rid + ' pid=' + pid + '?\\n\\n' +
      'Stop sweeps non-protected train_ppo only — protected overnight is never killed.'
    )) return;
    act('stop');
    return;
  }
  const n = window.__liveRunCount || 0;
  const warn = window.__startLockWarn;
  if (n >= 1 || warn) {
    if (!confirm(
      (warn || (n + ' live train.lock(s) already present')) + '\\n\\n' +
      'Start from this panel is refused while any lock is live ' +
      '(multi-train = separate CLI/UI ownership, not dual-Start). Continue to see refuse?'
    )) return;
  }
  act('start');
}
async function act(op, extra){
  const msgEl = document.getElementById('msg');
  msgEl.className = 'warn';
  if (op === 'start') msgEl.textContent = 'Starting training… (may take a few seconds)';
  else if (op === 'continue') msgEl.textContent = 'Resuming last complete checkpoint…';
  else if (op === 'stop') msgEl.textContent = 'Stopping…';
  else if (op === 'watch') msgEl.textContent = 'Launching watch…';
  else if (op === 'stop_watch') msgEl.textContent = 'Stopping watch…';
  else if (op === 'focus_run') msgEl.textContent = 'Focusing run (status pin)…';
  else if (op === 'clear_focus') msgEl.textContent = 'Clearing focus pin…';
  else if (op === 'load_model') msgEl.textContent = 'Loading model into watch…';
  else if (op === 'delete_model') msgEl.textContent = 'Deleting…';
  else if (op === 'gen_maps') msgEl.textContent = 'Generating maps… (a few seconds)';
  else if (op === 'verify_seals') msgEl.textContent = 'Verifying pack seals…';
  else if (op === 'auto_train') msgEl.textContent = 'Auto-train preview…';
  else if (op === 'tensorboard') msgEl.textContent = 'Starting TensorBoard…';
  else msgEl.textContent = 'Working…';
  const body = new URLSearchParams(Object.assign({
    op, map: document.getElementById('map').value,
    timesteps: document.getElementById('timesteps').value,
    n_envs: document.getElementById('n_envs').value,
    early_stop: document.getElementById('early_stop').value,
    min_improve: document.getElementById('min_improve').value,
    race_eval_every: document.getElementById('race_eval_every').value,
    stop_on_budget: document.getElementById('stop_on_budget').checked ? '1' : '0',
    allow_holdout: document.getElementById('allow_holdout').checked ? '1' : '0',
    collision_first: document.getElementById('collision_first').checked ? '1' : '0',
    speed_gate: document.getElementById('speed_gate').checked ? '1' : '0',
    fast_probe: document.getElementById('fast_probe').checked ? '1' : '0'
  }, extra || {}));
  const hint = window.__overnightHint || {};
  if (hint.warmup_evals != null) body.set('warmup_evals', String(hint.warmup_evals));
  if (hint.min_timesteps != null) body.set('min_timesteps', String(hint.min_timesteps));
  if (hint.select_timeout != null) body.set('select_timeout', String(hint.select_timeout));
  try {
    const r = await fetch('/api/action', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body});
    const j = await r.json();
    const text = j.msg || JSON.stringify(j);
    msgEl.textContent = text;
    const failed = /fail|error|could not|refuse|incomplete/i.test(text);
    const already = /already running|warning|no trainer/i.test(text);
    msgEl.className = failed ? 'bad' : (already ? 'warn' : 'ok');
    await refresh();
    // refresh must not wipe the action result
    msgEl.textContent = text;
    msgEl.className = failed ? 'bad' : (already ? 'warn' : 'ok');
  } catch (e) {
    msgEl.textContent = 'Action failed: ' + e;
    msgEl.className = 'bad';
  }
}
document.getElementById('map').addEventListener('change', schedulePreview);
document.getElementById('n_envs').addEventListener('change', schedulePreview);
syncN(); syncBudget(); refresh(); refreshPreview(); loadModels(); setInterval(refresh, STATUS_MS);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _client_gone(self, exc: BaseException) -> bool:
        if isinstance(exc, _CLIENT_GONE):
            return True
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (10053, 10054):
            return True
        return False

    def _send(self, code: int, data: bytes, content_type: str, *, extra_headers: dict | None = None):
        """Write a full HTTP response; ignore client disconnects (refresh / cancelled poll)."""
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
            try:
                self.wfile.flush()
            except Exception as e:
                if not self._client_gone(e):
                    raise
        except Exception as e:
            if self._client_gone(e):
                return
            raise

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self._send(code, data, "application/json")

    def _html(self, text: str):
        data = text.encode("utf-8")
        # Index is inline HTML — without these headers browsers often keep a stale panel.
        self._send(
            200,
            data,
            "text/html; charset=utf-8",
            extra_headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    def _bytes(self, code: int, data: bytes, content_type: str):
        self._send(
            code,
            data,
            content_type,
            extra_headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
            },
        )

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            page = HTML.replace("__UI_BUILD__", UI_BUILD)
            self._html(page)
            return
        if path == "/api/status":
            self._json(200, _status_payload())
            return
        if path == "/api/models":
            self._json(200, _models_payload())
            return
        if path == "/api/preview":
            q = parse_qs(parsed.query)
            map_id = (q.get("map") or ["map0"])[0]
            try:
                timesteps = int((q.get("timesteps") or ["100000"])[0])
            except ValueError:
                timesteps = 100_000
            try:
                n_envs = int((q.get("n_envs") or ["8"])[0])
            except ValueError:
                n_envs = 8
            try:
                early_stop_patience = max(0, int((q.get("early_stop") or ["0"])[0]))
            except ValueError:
                early_stop_patience = 0
            try:
                early_stop_min_improve = max(0.0, float((q.get("min_improve") or ["0.5"])[0]))
            except ValueError:
                early_stop_min_improve = 0.5
            try:
                race_eval_every = max(0, int((q.get("race_eval_every") or ["0"])[0]))
            except ValueError:
                race_eval_every = 0
            stop_on_budget = (q.get("stop_on_budget") or ["1"])[0] in ("1", "true", "on", "yes")
            collision_first = (q.get("collision_first") or ["0"])[0] in ("1", "true", "on", "yes")
            speed_gate = (q.get("speed_gate") or ["0"])[0] in ("1", "true", "on", "yes")
            fast_probe = (q.get("fast_probe") or ["0"])[0] in ("1", "true", "on", "yes")
            self._json(
                200,
                _preview_payload(
                    map_id,
                    timesteps,
                    n_envs,
                    stop_on_budget=stop_on_budget,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_improve=early_stop_min_improve,
                    race_eval_every=race_eval_every,
                    collision_first=collision_first,
                    speed_gate=speed_gate,
                    fast_probe=fast_probe,
                ),
            )
            return
        if path.startswith("/api/map_thumb/"):
            name = unquote(path[len("/api/map_thumb/") :]).strip("/\\")
            # only allow simple map folder names
            if not name or "/" in name or "\\" in name or ".." in name:
                self._json(400, {"error": "bad map"})
                return
            src = _resolve_map_image(name)
            if src is None:
                self._json(404, {"error": "no map image"})
                return
            thumb = _ensure_thumb(name, src)
            if thumb is None or not thumb.is_file():
                self._json(500, {"error": "thumb failed"})
                return
            self._bytes(200, thumb.read_bytes(), "image/png")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/action":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n).decode("utf-8")
        except Exception as e:
            if self._client_gone(e):
                return
            raise
        form = parse_qs(raw)
        op = (form.get("op") or ["status"])[0]
        map_id = (form.get("map") or ["map0"])[0]
        run_id = (form.get("run_id") or [""])[0].strip()
        which = (form.get("which") or ["best"])[0]
        allow_holdout = (form.get("allow_holdout") or ["0"])[0] in ("1", "true", "on", "yes")
        stop_on_budget = (form.get("stop_on_budget") or ["1"])[0] in ("1", "true", "on", "yes")
        collision_first = (form.get("collision_first") or ["0"])[0] in ("1", "true", "on", "yes")
        speed_gate = (form.get("speed_gate") or ["0"])[0] in ("1", "true", "on", "yes")
        fast_probe = (form.get("fast_probe") or ["0"])[0] in ("1", "true", "on", "yes")
        try:
            timesteps = int((form.get("timesteps") or ["100000"])[0])
        except ValueError:
            timesteps = 100_000
        try:
            n_envs = _clamp_n_envs(int((form.get("n_envs") or ["8"])[0]))
        except ValueError:
            n_envs = 8
        try:
            early_stop_patience = max(0, int((form.get("early_stop") or ["0"])[0]))
        except ValueError:
            early_stop_patience = 0
        try:
            early_stop_min_improve = max(0.0, float((form.get("min_improve") or ["0.5"])[0]))
        except ValueError:
            early_stop_min_improve = 0.5
        try:
            race_eval_every = max(0, int((form.get("race_eval_every") or ["0"])[0]))
        except ValueError:
            race_eval_every = 0
        warmup_evals = None
        min_ts = None
        sel_timeout = None
        try:
            if form.get("warmup_evals"):
                warmup_evals = max(0, int(form.get("warmup_evals")[0]))
        except ValueError:
            warmup_evals = None
        try:
            if form.get("min_timesteps"):
                min_ts = max(0, int(form.get("min_timesteps")[0]))
        except ValueError:
            min_ts = None
        try:
            if form.get("select_timeout"):
                sel_timeout = max(1.0, float(form.get("select_timeout")[0]))
        except ValueError:
            sel_timeout = None

        global _last_msg
        if op == "start":
            msg = _start_train(
                map_id,
                timesteps,
                n_envs,
                allow_holdout=allow_holdout,
                early_stop_patience=early_stop_patience,
                early_stop_min_improve=early_stop_min_improve,
                race_eval_every=race_eval_every,
                early_stop_warmup_evals=warmup_evals,
                early_stop_min_timesteps=min_ts,
                select_timeout=sel_timeout,
                stop_on_budget=stop_on_budget,
                collision_first=collision_first,
                speed_gate=speed_gate,
                fast_probe=fast_probe,
            )
        elif op == "continue":
            msg = _continue_train(
                run_id,
                timesteps,
                n_envs,
                early_stop_patience=early_stop_patience,
                early_stop_min_improve=early_stop_min_improve,
                race_eval_every=race_eval_every,
                early_stop_warmup_evals=warmup_evals,
                early_stop_min_timesteps=min_ts,
                select_timeout=sel_timeout,
                stop_on_budget=stop_on_budget,
            )
        elif op == "stop":
            msg = _stop_all()
        elif op == "focus_run":
            msg = _focus_run(run_id)
        elif op == "clear_focus":
            msg = _clear_focus()
        elif op == "watch":
            msg = _launch_watch(map_id, n_envs)
        elif op == "stop_watch":
            msg = _stop_watch()
        elif op == "load_model":
            msg = _watch_model(run_id, which, map_id)
        elif op == "delete_model":
            msg = _delete_model(run_id)
        elif op == "gen_maps":
            try:
                count = int((form.get("count") or ["3"])[0])
            except ValueError:
                count = 3
            try:
                seed = int((form.get("seed") or ["123"])[0])
            except ValueError:
                seed = 123
            msg = _generate_maps(count, seed)
        elif op == "verify_seals":
            msg = _verify_seals()
        elif op == "auto_train":
            # Preview/refuse only — never spawn beside overnight; chaos drills gate enablement.
            msg = _set_msg(
                "Refuse Auto-train UI spawn (W4 thin): use `python -m rl.auto_train --dry-run` "
                "(or `--execute --smoke` when no live train.lock). Never Start/Continue/kill "
                f"{PROTECTED_OVERNIGHT_RUN}; no reward mutation; no holdout."
            )
        elif op == "tensorboard":
            msg = _ensure_tensorboard()
            with _state_lock:
                _last_msg = msg
        else:
            msg = f"unknown op {op}"
        out = _status_payload()
        out["msg"] = msg
        self._json(200, out)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does not dump stack traces for client aborts."""

    def handle_error(self, request, client_address):
        err = sys.exc_info()[1]
        if isinstance(err, _CLIENT_GONE):
            return
        if isinstance(err, OSError) and getattr(err, "winerror", None) in (10053, 10054):
            return
        super().handle_error(request, client_address)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Minimal RL train control UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Prewarming map previews...")
    try:
        _prewarm_map_thumbs()
        for m in _list_maps():
            print(f"  map {m['id']}: thumb={m.get('thumb')} src={m.get('src')}")
    except Exception as e:
        print(f"  map prewarm warning: {e}")

    server = QuietThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"RL control UI -> {url}")
    print(f"contracts={CONTRACTS_VERSION} obs_dim={obs_dim()} (no camera)")
    print("Toggle Start/Stop = train + auto TensorBoard. Stop = kill train tree + UI-spawned TB.")
    print("Open TensorBoard -> http://127.0.0.1:6006/  |  map dropdown + preview on the right")
    print("Status polls every 0.75s; live_status writes ~every 512 steps.")
    print("Ctrl+C to quit this UI (does not auto-stop a running train unless you hit Stop).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nUI stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
