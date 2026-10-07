"""FastAPI Mission Control hub — API + static Vite build on :8090."""

from __future__ import annotations

import os
import json
import math
from os import scandir
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src.layer4.hub.map_activate import activate_map
from src.layer4.hub.maps_catalog import list_maps_api, resolve_maps_root
from src.layer4.hub.replay_job import ReplayJob
from src.layer4.hub.telemetry import TelemetryBus, WSClient
from src.layer4.hub.train_job import TrainJob
from src.layer4.settings import ROOT, Settings, load_settings, save_settings
from src.layer3.train import _PPO_N_STEPS


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


WEB_DIST = _resolve_web_dist()
MAPS_DIR = resolve_maps_root(ROOT)
HUB_HOST = os.environ.get("HUB_HOST", "0.0.0.0")
HUB_PORT = int(os.environ.get("HUB_PORT", os.environ.get("READY_PORT", "8090")))
HUB_PUBLIC_URL = os.environ.get("HUB_URL", f"http://127.0.0.1:{HUB_PORT}")

bus = TelemetryBus()
job = TrainJob(hub_url=HUB_PUBLIC_URL, on_status=lambda s: bus.publish_status(s))
replay_job = ReplayJob(hub_url=HUB_PUBLIC_URL, on_status=lambda s: bus.publish_replay_status(s))
_settings: Settings = load_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    bus.bind_loop(asyncio.get_running_loop())
    # A hard hub restart can leave the standalone replay simulator behind.
    # It is safe to remove at startup because no ReplayJob process survives
    # the previous hub process.
    try:
        from src.layer4.hub.docker_control import remove_replay_sim

        await asyncio.to_thread(remove_replay_sim)
    except Exception:
        pass
    try:
        yield
    finally:
        # Stop the Layer 3 process first; its exit watcher also removes Docker.
        # Then remove unconditionally to cover an already-exited process.
        try:
            await asyncio.to_thread(replay_job.stop)
        except Exception:
            pass
        try:
            from src.layer4.hub.docker_control import remove_replay_sim

            await asyncio.to_thread(remove_replay_sim)
        except Exception:
            pass


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


@app.post("/api/shutdown")
async def api_shutdown(request: Request) -> Dict[str, Any]:
    """Stop Mission Control (and optionally the Docker compose stack).

    Query: stack=1 also stops compose containers (brain last).
    """
    import asyncio
    import signal

    qp = request.query_params
    stop_stack = qp.get("stack", "").strip().lower() in {"1", "true", "yes"}
    stopped: list[str] = []
    train_status: Optional[Dict[str, Any]] = None

    try:
        train_status = await asyncio.to_thread(job.stop)
    except Exception:
        train_status = None

    try:
        await asyncio.to_thread(replay_job.stop)
    except Exception:
        pass
    try:
        from src.layer4.hub.docker_control import remove_replay_sim

        await asyncio.to_thread(remove_replay_sim)
    except Exception:
        pass

    if stop_stack:
        try:
            from src.layer4.hub.docker_control import stop_compose_containers

            stopped = await asyncio.to_thread(
                stop_compose_containers, full_stack=True
            )
        except Exception as exc:
            # Still shut down the hub; report docker failure to the client.
            async def _die_after_error() -> None:
                await asyncio.sleep(0.4)
                os.kill(os.getpid(), signal.SIGTERM)

            asyncio.get_running_loop().create_task(_die_after_error())
            return {
                "ok": True,
                "shutting_down": True,
                "stack": True,
                "stopped_containers": stopped,
                "train": train_status,
                "stack_error": str(exc),
            }

    async def _die() -> None:
        await asyncio.sleep(0.4)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.get_running_loop().create_task(_die())
    return {
        "ok": True,
        "shutting_down": True,
        "stack": stop_stack,
        "stopped_containers": stopped,
        "train": train_status,
    }


@app.get("/api/maps")
async def api_maps() -> Any:
    """Auto-detected occupancy maps under simulator/maps/ (+ synthetic 'none')."""
    return list_maps_api(MAPS_DIR)


@app.post("/api/maps/upload")
async def api_upload_map(request: Request) -> Any:
    """Install an occupancy map package from a zip body (Phase 4).

    Body: application/zip (or application/octet-stream).
    Query: map_id (required for flat zips), overwrite=1, label=...
    """
    import asyncio

    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")

    from src.layer4.hub.map_upload import install_map_zip

    body = await request.body()
    qp = request.query_params
    map_id = qp.get("map_id") or None
    label = qp.get("label") or None
    overwrite = qp.get("overwrite", "").strip().lower() in {"1", "true", "yes"}

    try:
        result = await asyncio.to_thread(
            install_map_zip,
            MAPS_DIR,
            body,
            map_id=map_id,
            overwrite=overwrite,
            label=label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["maps"] = list_maps_api(MAPS_DIR)
    return result


@app.post("/api/maps/{map_id}/generate-mesh")
async def api_generate_mesh(map_id: str) -> Any:
    """Generate track.obj / track_col.obj from occupancy (Phase 2)."""
    import asyncio

    if map_id == "none":
        raise HTTPException(status_code=400, detail="cannot generate mesh for 'none'")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    map_dir = MAPS_DIR / map_id
    if not map_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown map id: {map_id}")

    from src.layer4.hub.meshgen import generate_mesh

    try:
        result = await asyncio.to_thread(generate_mesh, map_id, MAPS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "result": result.to_api(), "maps": list_maps_api(MAPS_DIR)}


@app.post("/api/maps/{map_id}/generate-centerline")
async def api_generate_centerline(map_id: str) -> Any:
    """Extract and validate a closed occupancy route → centerline.csv."""
    import asyncio

    if map_id == "none":
        raise HTTPException(status_code=400, detail="cannot generate centerline for 'none'")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    if not (MAPS_DIR / map_id).is_dir():
        raise HTTPException(status_code=404, detail=f"unknown map id: {map_id}")

    from src.layer4.hub.centerline import generate_centerline

    try:
        result = await asyncio.to_thread(generate_centerline, map_id, MAPS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "result": result.to_api(), "maps": list_maps_api(MAPS_DIR)}


@app.get("/api/maps/{map_id}/centerline.csv")
async def api_centerline_csv(map_id: str) -> Any:
    """Download centerline CSV if present (generate first if missing is caller's choice)."""
    if map_id == "none":
        raise HTTPException(status_code=404, detail="no centerline for 'none'")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    map_dir = MAPS_DIR / map_id
    if not map_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown map id: {map_id}")

    from src.layer4.hub.centerline import find_centerline

    path = find_centerline(map_dir)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"no centerline.csv for '{map_id}' — POST .../generate-centerline first",
        )
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{map_id}_centerline.csv",
    )


@app.get("/api/maps/{map_id}/lap-gate")
async def api_map_lap_gate(map_id: str) -> Any:
    """Return the effective full-width lap gate for a map."""
    import asyncio

    if map_id == "none":
        raise HTTPException(status_code=400, detail="builtin track has no centerline lap gate")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    from src.layer4.hub.lap_gate import get_map_lap_gate

    try:
        return await asyncio.to_thread(get_map_lap_gate, map_id, MAPS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/api/maps/{map_id}/lap-gate")
async def api_save_map_lap_gate(map_id: str, request: Request) -> Any:
    """Set route distance for a map's finish gate; null restores spawn default."""
    import asyncio

    if map_id == "none":
        raise HTTPException(status_code=400, detail="builtin track has no centerline lap gate")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="expected JSON body") from exc
    if not isinstance(payload, dict) or "progress_m" not in payload:
        raise HTTPException(status_code=400, detail="body must include progress_m (number or null)")

    from src.layer4.hub.lap_gate import save_map_lap_gate

    try:
        return await asyncio.to_thread(
            save_map_lap_gate, map_id, MAPS_DIR, payload["progress_m"]
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/maps/active")
async def api_maps_active(request: Request) -> Any:
    """Current physics map id + selection mismatch hints (Phase 4)."""
    from src.layer4.hub.map_status import map_status_payload

    selected = request.query_params.get("selected") or None
    payload = map_status_payload(MAPS_DIR, selected_id=selected, repo_root=ROOT)
    payload["maps"] = list_maps_api(MAPS_DIR)
    return payload


@app.post("/api/maps/{map_id}/generate-thumbnail")
async def api_generate_thumbnail(map_id: str) -> Any:
    """Ensure occupancy/preview.png exists for Fleet catalog thumbnails."""
    import asyncio

    if map_id == "none":
        raise HTTPException(status_code=400, detail="cannot thumbnail 'none'")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    if not (MAPS_DIR / map_id).is_dir():
        raise HTTPException(status_code=404, detail=f"unknown map id: {map_id}")

    from src.layer4.hub.mesh_preview import ensure_occupancy_thumbnail

    try:
        path = await asyncio.to_thread(ensure_occupancy_thumbnail, map_id, MAPS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="no occupancy image to thumbnail")
    return {
        "ok": True,
        "path": str(path),
        "url": f"/maps/{map_id}/occupancy/{path.name}",
        "maps": list_maps_api(MAPS_DIR),
    }


@app.post("/api/maps/{map_id}/generate-mesh-preview")
async def api_generate_mesh_preview(map_id: str) -> Any:
    """Orthographic top-down PNG of track.obj (no Unity)."""
    import asyncio

    if map_id == "none":
        raise HTTPException(status_code=400, detail="cannot preview 'none'")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    if not (MAPS_DIR / map_id).is_dir():
        raise HTTPException(status_code=404, detail=f"unknown map id: {map_id}")

    from src.layer4.hub.mesh_preview import generate_mesh_preview

    try:
        result = await asyncio.to_thread(generate_mesh_preview, map_id, MAPS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "result": result.to_api(), "maps": list_maps_api(MAPS_DIR)}


@app.get("/api/maps/{map_id}/mesh-preview.png")
async def api_mesh_preview_png(map_id: str) -> Any:
    """Serve mesh/preview.png; 404 if not generated yet."""
    if map_id == "none":
        raise HTTPException(status_code=404, detail="no preview for 'none'")
    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")
    map_dir = MAPS_DIR / map_id
    if not map_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown map id: {map_id}")

    from src.layer4.hub.mesh_preview import find_mesh_preview

    path = find_mesh_preview(map_dir)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"no mesh preview for '{map_id}' — POST .../generate-mesh-preview first",
        )
    return FileResponse(path, media_type="image/png", filename=f"{map_id}_mesh_preview.png")


@app.post("/api/maps/{map_id}/activate")
async def api_activate_map(map_id: str, request: Request) -> Any:
    """Persist active map + restart compose sims (Phase 3). Logic is in map_activate."""
    import asyncio

    if MAPS_DIR is None:
        raise HTTPException(status_code=404, detail="maps directory not found")

    force = False
    restart = True
    try:
        raw = await request.json()
        if isinstance(raw, dict):
            force = bool(raw.get("force", False))
            if "restart" in raw:
                restart = bool(raw["restart"])
    except Exception:
        pass
    # Also accept ?force=1 for thin clients
    if request.query_params.get("force", "").strip().lower() in {"1", "true", "yes"}:
        force = True

    train_state = job.status().get("state")
    train_running = train_state in {"starting", "running", "stopping"}

    try:
        result = await asyncio.to_thread(
            activate_map,
            MAPS_DIR,
            map_id,
            restart=restart,
            train_running=train_running,
            force=force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["maps"] = list_maps_api(MAPS_DIR)
    return result


@app.get("/train/status")
async def train_status() -> Dict[str, Any]:
    st = job.status()
    st["last_telemetry"] = bus.last_metrics
    st["last_fleet"] = bus.last_fleet
    st["last_train_phase"] = bus.last_train_phase
    try:
        evaluation_path = _settings.resolve_out() / "evaluation_status.json"
        st["last_evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st["last_evaluation"] = None
    return st


@app.get("/train/evaluations")
async def train_evaluations() -> list[Dict[str, Any]]:
    """Return the complete persisted evaluator history for the active run output."""
    settings = job.last_settings or _settings
    # The training job's log file lives beside its artifacts and avoids
    # resolving a fresh timestamped output directory when `out` is unset.
    output_dir = Path(job.log_path).parent if job.log_path else settings.resolve_out()
    history_path = output_dir / "evaluation_history.jsonl"
    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    results: list[Dict[str, Any]] = []
    for line in lines:
        try:
            result = json.loads(line)
        except ValueError:
            continue
        if isinstance(result, dict):
            results.append(result)
    return results


@app.get("/train/ppo-diagnostics")
async def train_ppo_diagnostics() -> Dict[str, Any]:
    """Read durable PPO scalar history from this run's TensorBoard event files."""
    import asyncio

    output_dir = Path(job.log_path).parent if job.log_path else _settings.resolve_out()
    if not output_dir.is_dir():
        # If the hub restarted, its TrainJob no longer remembers the child
        # process path. Recover the latest run directory written by the hub.
        runs_root = ROOT / "logs" / "rl"
        candidates = [path for path in runs_root.glob("*") if path.is_dir() and (path / "hub_train.log").is_file()]
        if candidates:
            output_dir = max(candidates, key=lambda path: (path / "hub_train.log").stat().st_mtime)

    def read_events() -> list[Dict[str, Any]]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError as exc:
            raise RuntimeError("TensorBoard is not installed in the Mission Control environment") from exc

        scalar_names = {
            "fps": "time/fps",
            "approx_kl": "train/approx_kl",
            "clip_fraction": "train/clip_fraction",
            "entropy_loss": "train/entropy_loss",
            "explained_variance": "train/explained_variance",
            "learning_rate": "train/learning_rate",
            "loss": "train/loss",
            "policy_gradient_loss": "train/policy_gradient_loss",
            "std": "train/std",
            "value_loss": "train/value_loss",
        }
        updates: Dict[int, Dict[str, tuple[float, float]]] = {}
        tensorboard_root = output_dir / "tb"
        event_dirs = {item.parent for item in tensorboard_root.rglob("events.out.tfevents.*")} if tensorboard_root.is_dir() else set()
        for event_dir in sorted(event_dirs):
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            available = set(accumulator.Tags().get("scalars", []))
            for key, tag in scalar_names.items():
                if tag not in available:
                    continue
                for event in accumulator.Scalars(tag):
                    if math.isfinite(event.value):
                        scalar_values = updates.setdefault(int(event.step), {})
                        previous = scalar_values.get(key)
                        if previous is None or event.wall_time >= previous[0]:
                            scalar_values[key] = (float(event.wall_time), float(event.value))
        return [
            {"step": step, "values": {key: value for key, (_wall_time, value) in values.items()}}
            for step, values in sorted(updates.items())
            if values
        ]

    try:
        updates = await asyncio.to_thread(read_events)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"run_id": output_dir.name, "updates": updates}


@app.post("/train/start")
async def train_start(request: Request) -> Dict[str, Any]:
    """Body = full Settings snapshot (preferred); empty body → last saved Settings.

    Locks in settings.map_id before Popen: activate that map (restart sims only
    when it differs from the currently active physics map).
    """
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

    # Lock in map selection for this run (Train dropdown → physics + Watch underlay).
    mid = (settings.map_id or "none").strip() or "none"
    if MAPS_DIR is not None:
        try:
            from src.layer4.hub.map_activate import read_active_map

            current = read_active_map(MAPS_DIR).get("id")
            current_key = current if current else "none"
            if mid != current_key:
                await asyncio.to_thread(
                    activate_map,
                    MAPS_DIR,
                    mid,
                    restart=True,
                    train_running=False,
                    force=False,
                )
            elif mid != "none":
                # Same map already active — still ensure spawn/meta without sim bounce.
                await asyncio.to_thread(
                    activate_map,
                    MAPS_DIR,
                    mid,
                    restart=False,
                    train_running=False,
                    force=False,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"map activate failed: {exc}"
            ) from exc

    try:
        # Popen can stall briefly on Windows; never block the event loop.
        if job.status().get("state") not in {"starting", "running", "stopping"}:
            bus.clear_train_phase()
        return await asyncio.to_thread(job.start, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train/stop")
async def train_stop() -> Dict[str, Any]:
    import asyncio

    # Use persisted settings when the hub restarted and no TrainJob instance
    # owns the old subprocess anymore; this lets Stop clean up orphaned sims.
    return await asyncio.to_thread(job.stop, settings=_settings)


@app.get("/api/replay/models")
async def replay_models() -> list[Dict[str, Any]]:
    import asyncio

    models_root = (ROOT / "logs" / "rl").resolve()
    if not models_root.is_dir():
        return []

    def scan() -> list[Dict[str, Any]]:
        models: list[Dict[str, Any]] = []
        try:
            runs = list(scandir(models_root))
        except OSError:
            return []
        for run in runs:
            if not run.is_dir(follow_symlinks=False):
                continue
            candidates: list[Path] = []
            try:
                with scandir(run.path) as entries:
                    candidates.extend(
                        Path(entry.path) for entry in entries
                        if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".zip")
                    )
                checkpoint_dir = Path(run.path) / "ckpt"
                if checkpoint_dir.is_dir():
                    with scandir(checkpoint_dir) as entries:
                        checkpoints = [
                            (entry.stat(follow_symlinks=False).st_mtime, Path(entry.path))
                            for entry in entries
                            if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".zip")
                        ]
                    candidates.extend(path for _, path in sorted(checkpoints, reverse=True)[:5])
            except OSError:
                continue
            for path in candidates:
                try:
                    stat = path.stat()
                    relative = path.relative_to(models_root).as_posix()
                except OSError:
                    continue
                models.append({
                    "id": relative,
                    "label": f"{run.name}/{path.name}",
                    "modified": stat.st_mtime,
                })
        return sorted(models, key=lambda model: model["modified"], reverse=True)[:200]

    return await asyncio.to_thread(scan)


@app.get("/api/replay/status")
async def replay_status() -> Dict[str, Any]:
    result = replay_job.status()
    result["last_fleet"] = bus.last_replay_fleet
    return result


@app.post("/api/replay/start")
async def replay_start(request: Request) -> Dict[str, Any]:
    import asyncio

    try:
        body = await request.json()
        model_id = str(body.get("model_id", "")).strip()
        map_id = str(body.get("map_id", "none")).strip() or "none"
        seed = int(body.get("seed", 0))
        device = str(body.get("device", "auto"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid replay settings") from exc
    if device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be auto, cpu, or cuda")
    models_root = (ROOT / "logs" / "rl").resolve()
    model_path = (models_root / model_id).resolve()
    try:
        model_path.relative_to(models_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid model selection") from exc
    if not model_path.is_file() or model_path.suffix.lower() != ".zip":
        raise HTTPException(status_code=404, detail="checkpoint not found")
    if map_id != "none":
        map_entry = next((m for m in list_maps_api(MAPS_DIR) if m.get("id") == map_id), None)
        if map_entry is None:
            raise HTTPException(status_code=404, detail=f"unknown map: {map_id}")
        if map_entry.get("mesh_status") != "ready":
            raise HTTPException(status_code=400, detail=f"map is not ready for replay: {map_id}")
    replay_settings = _settings
    in_docker = os.environ.get("AICAR_IN_DOCKER", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    try:
        return await asyncio.to_thread(
            replay_job.start,
            model_path=model_path,
            map_id=map_id,
            seed=seed,
            device=device,
            simulator_env=replay_settings.simulator_env(),
            env_kwargs=replay_settings.to_env_kwargs(
                headless=True,
                auto_launch=not in_docker,
                map_id=map_id,
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/replay/stop")
async def replay_stop() -> Dict[str, Any]:
    import asyncio

    return await asyncio.to_thread(replay_job.stop)


@app.post("/api/replay/reset")
async def replay_reset() -> Dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(replay_job.reset)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/settings")
async def get_settings() -> Dict[str, Any]:
    return {
        **_settings.model_dump(mode="json"),
        # Display-only metadata so the UI can calculate rollout-aligned
        # evaluation snapshots using Layer 3's actual PPO configuration.
        "ppo_n_steps": _PPO_N_STEPS,
    }


@app.put("/settings")
async def put_settings(body: Settings) -> Dict[str, Any]:
    global _settings
    _settings = body
    path = save_settings(_settings)
    return {
        "ok": True,
        "path": str(path),
        "settings": {
            **_settings.model_dump(mode="json"),
            "ppo_n_steps": _PPO_N_STEPS,
        },
    }


@app.post("/telemetry")
async def post_telemetry(request: Request) -> Dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be object")
    bus.publish("replay_telemetry" if payload.get("channel") == "replay" else "telemetry", payload)
    return {"ok": "1"}


@app.post("/api/train/phase")
async def post_train_phase(request: Request) -> Dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "phase":
        raise HTTPException(status_code=400, detail="invalid training phase payload")
    if payload.get("phase") not in {"rollout", "ppo_update", "stopped"}:
        raise HTTPException(status_code=400, detail="invalid training phase")
    bus.publish_train_phase(payload)
    return {"ok": "1"}


@app.post("/api/evaluator/live")
async def post_evaluator_live(request: Request) -> Dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be object")
    pose = payload.get("pose")
    if (not isinstance(pose, (list, tuple)) or len(pose) != 2
            or not isinstance(payload.get("yaw"), (int, float))
            or not isinstance(payload.get("speed"), (int, float))):
        raise HTTPException(status_code=400, detail="invalid evaluator pose")
    bus.publish_evaluator_live(payload)
    return {"ok": "1"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = WSClient(ws)
    bus.attach(client)
    # Resync: push last known status + telemetry
    try:
        await ws.send_json({"type": "status", "payload": job.status()})
        await ws.send_json({"type": "replay_status", "payload": replay_job.status()})
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
