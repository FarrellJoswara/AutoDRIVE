"""FastAPI Mission Control hub — API + static Vite build on :8090."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.layer4.hub.telemetry import TelemetryBus, WSClient
from src.layer4.hub.train_job import TrainJob
from src.layer4.settings import ROOT, Settings, load_settings, save_settings


def _resolve_web_dist() -> Path:
    """Prefer bind-mounted `web/dist` so host `npm run build` updates UI without image rebuild."""
    source_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
    env = os.environ.get("MISSION_CONTROL_DIST", "").strip()
    candidates = [source_dist]
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/app/static/mission-control"))
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return source_dist


def _resolve_maps_dir() -> Optional[Path]:
    env = os.environ.get("AICAR_MAPS_DIR", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/app/assets/maps"))
    candidates.append(ROOT / "assets" / "maps")
    for c in candidates:
        if c.is_dir():
            return c
    return None


WEB_DIST = _resolve_web_dist()
MAPS_DIR = _resolve_maps_dir()
HUB_HOST = os.environ.get("HUB_HOST", "0.0.0.0")
HUB_PORT = int(os.environ.get("HUB_PORT", os.environ.get("READY_PORT", "8090")))
HUB_PUBLIC_URL = os.environ.get("HUB_URL", f"http://127.0.0.1:{HUB_PORT}")

bus = TelemetryBus()
job = TrainJob(hub_url=HUB_PUBLIC_URL, on_status=lambda s: bus.publish_status(s))
_settings: Settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    bus.bind_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="AiCar Mission Control", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/train/status")
async def train_status() -> Dict[str, Any]:
    st = job.status()
    st["last_telemetry"] = bus.last_metrics
    st["last_fleet"] = bus.last_fleet
    return st


@app.post("/train/start")
async def train_start(request: Request) -> Dict[str, Any]:
    """Body = full Settings snapshot (preferred); empty body → last saved Settings."""
    import asyncio

    global _settings
    settings = _settings
    try:
        raw = await request.json()
        if isinstance(raw, dict) and raw:
            settings = Settings.model_validate(raw)
    except Exception:
        pass
    _settings = settings
    try:
        await asyncio.to_thread(save_settings, _settings)
    except Exception:
        pass
    try:
        # Popen can stall briefly on Windows; never block the event loop.
        return await asyncio.to_thread(job.start, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train/stop")
async def train_stop() -> Dict[str, Any]:
    import asyncio

    return await asyncio.to_thread(job.stop)


@app.get("/settings")
async def get_settings() -> Dict[str, Any]:
    return _settings.model_dump(mode="json")


@app.put("/settings")
async def put_settings(body: Settings) -> Dict[str, Any]:
    global _settings
    _settings = body
    path = save_settings(_settings)
    return {"ok": True, "path": str(path), "settings": _settings.model_dump(mode="json")}


@app.post("/telemetry")
async def post_telemetry(request: Request) -> Dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be object")
    bus.publish("telemetry", payload)
    return {"ok": "1"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = WSClient(ws)
    bus.attach(client)
    # Resync: push last known status + telemetry
    try:
        await ws.send_json({"type": "status", "payload": job.status()})
        for ev in bus.snapshot_for_client():
            await ws.send_json(ev)
    except Exception:
        bus.detach(client)
        return

    import asyncio

    sender = asyncio.create_task(client.sender())
    try:
        while True:
            # Keep connection alive; ignore client messages (v1 is one-way)
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        client.close()
        bus.detach(client)
        sender.cancel()
        try:
            await sender
        except Exception:
            pass


def _spa_index() -> Optional[Path]:
    index = WEB_DIST / "index.html"
    return index if index.is_file() else None


if (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="assets")

if MAPS_DIR is not None:
    app.mount("/maps", StaticFiles(directory=str(MAPS_DIR)), name="maps")


@app.get("/")
async def root() -> Any:
    index = _spa_index()
    if index is not None:
        return FileResponse(index)
    return PlainTextResponse(
        "Mission Control hub OK — build src/layer4/web (npm run build) for UI\n"
    )


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str) -> Any:
    # Do not steal API paths (already registered above). Serve SPA for deep links.
    if full_path.startswith(
        ("health", "train/", "settings", "telemetry", "ws", "assets/", "maps/", "api/")
    ):
        raise HTTPException(status_code=404, detail="not found")
    index = _spa_index()
    if index is not None:
        # Prefer exact static file if present (favicon, etc.)
        candidate = WEB_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="UI not built")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.layer4.hub.app:app",
        host=HUB_HOST,
        port=HUB_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
