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
from src.layer4.hub.official_run_manager import OfficialRunManager
from src.layer4.hub.official_replay_job import OfficialReplayJob
from src.layer4.hub.official_models import list_official_models
from src.layer4.hub.telemetry import TelemetryBus, WSClient
from src.layer4.official_settings import (
    OfficialTrainSettings,
    load_official_train_settings,
    save_official_train_settings,
)
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


WEB_DIST = _resolve_web_dist()
MAPS_DIR = resolve_maps_root(ROOT)
HUB_HOST = os.environ.get("HUB_HOST", "0.0.0.0")
HUB_PORT = int(os.environ.get("HUB_PORT", os.environ.get("READY_PORT", "8090")))
HUB_PUBLIC_URL = os.environ.get("HUB_URL", f"http://127.0.0.1:{HUB_PORT}")

bus = TelemetryBus()
official_runs = OfficialRunManager()
replay_job = OfficialReplayJob(official_runs, on_status=lambda s: bus.publish_replay_status(s))
_settings: Settings = load_settings()
_official_train_settings: OfficialTrainSettings = load_official_train_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    bus.bind_loop(asyncio.get_running_loop())
    # Official training containers are deliberately reconciled rather than
    # stopped here; they own persistent per-run state and may survive a hub
    # restart. Replay remains a separate short-lived development feature.
    try:
        await asyncio.to_thread(official_runs.reconcile)
    except Exception:
        pass
    # A hard hub restart can leave the standalone replay simulator behind.
    try:
        from src.layer4.hub.docker_control import remove_replay_sim

        await asyncio.to_thread(remove_replay_sim)
    except Exception:
        pass
    try:
        yield
    finally:
        official_runs.close()
        # Then clean up the short-lived replay process/container.
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
    train_status: list[Dict[str, Any]] = []
    try:
        active_runs = [
            row for row in official_runs.list_runs()["runs"]
            if row.get("state") in {"starting", "running", "stopping"}
        ]
        train_status = await asyncio.gather(*[
            asyncio.to_thread(official_runs.stop_run, str(row["run_id"]))
            for row in active_runs
        ])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not stop official runs: {exc}") from exc
    if any(row.get("state") != "stopped" or row.get("cleanup_error") for row in train_status):
        raise HTTPException(status_code=503, detail="Official run cleanup is incomplete; keep the hub running and retry Stop.")

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

    train_running = any(
        run.get("state") in {"starting", "running", "stopping"}
        for run in official_runs.list_runs()["runs"]
    )

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
    runs = official_runs.list_runs()["runs"]
    selected = next(
        (item for item in runs if item.get("state") in {"starting", "running", "stopping"}),
        runs[0] if runs else None,
    )
    if selected is None:
        return {"state": "idle", "run_id": None, "last_fleet": None, "last_telemetry": None}
    telemetry = _run_telemetry(selected["run_id"])
    return {
        **selected,
        "pid": None,
        "argv": [],
        "hub_url": HUB_PUBLIC_URL,
        "last_fleet": telemetry.get("last_fleet"),
        "last_fleet_age_ms": _watch_snapshot_age_ms(telemetry.get("last_fleet")),
        "last_telemetry": telemetry.get("last_metrics"),
        "last_train_phase": telemetry.get("last_train_phase"),
        "last_evaluation": telemetry.get("last_evaluation"),
    }


@app.get("/train/official-settings")
async def get_official_train_settings() -> Dict[str, Any]:
    return {"settings": _official_train_settings.model_dump(mode="json")}


@app.put("/train/official-settings")
async def put_official_train_settings(body: OfficialTrainSettings) -> Dict[str, Any]:
    global _official_train_settings
    _official_train_settings = body
    path = save_official_train_settings(body)
    return {"ok": True, "path": str(path), "settings": body.model_dump(mode="json")}


def _run_telemetry(run_id: str) -> Dict[str, Any]:
    latest = bus.run_snapshot(run_id)
    try:
        persisted = official_runs.telemetry(run_id)
    except KeyError:
        raise
    run_dir = _run_output_dir(run_id)

    def read_snapshot(filename: str) -> Optional[Dict[str, Any]]:
        try:
            value = json.loads((run_dir / filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict) or value.get("run_id") != run_id:
            return None
        return value

    fleet = latest.get("last_fleet") or persisted.get("last_fleet") or read_snapshot("watch_latest.json")
    latest_metrics = latest.get("last_metrics") or persisted.get("last_metrics") or read_snapshot("latest_telemetry.json")
    phase = latest.get("last_train_phase") or persisted.get("last_train_phase") or read_snapshot("latest_training_phase.json")
    evaluator_live = latest.get("last_evaluator_live") or persisted.get("last_evaluator_live")
    try:
        evaluation = json.loads((run_dir / "evaluation_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        evaluation = None
    history = []
    try:
        for row in (run_dir / "evaluation_history.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                parsed = json.loads(row)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                history.append(parsed)
    except OSError:
        pass
    return {
        "run_id": run_id,
        "updated_at": (fleet or latest_metrics or {}).get("ts"),
        "fleet": fleet,
        "last_fleet": fleet,
        "last_metrics": latest_metrics,
        "last_train_phase": phase,
        "last_evaluator_live": evaluator_live,
        "last_evaluation": evaluation,
        "map_id": "none",
    }


def _run_output_dir(run_id: str) -> Path:
    run = official_runs.get_run(run_id)
    raw = run.get("output_dir")
    path = Path(raw).resolve() if raw else (ROOT / "logs" / "rl" / f"official_run_{run_id}").resolve()
    root = (ROOT / "logs" / "rl").resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="invalid run output path")
    return path


@app.get("/train/runs")
async def list_official_runs() -> Dict[str, Any]:
    return official_runs.list_runs()


def _latest_run_id() -> Optional[str]:
    runs = official_runs.list_runs()["runs"]
    selected = next(
        (row for row in runs if row.get("state") in {"starting", "running", "stopping"}),
        runs[0] if runs else None,
    )
    return str(selected["run_id"]) if selected and selected.get("run_id") else None


@app.post("/train/runs")
async def create_official_run(request: Request) -> Dict[str, Any]:
    global _official_train_settings
    try:
        raw = await request.json()
        if not isinstance(raw, dict):
            raise ValueError("request body must be an object")
        config = OfficialTrainSettings.model_validate(raw.get("config", raw))
        save_official_train_settings(config)
        _official_train_settings = config
        display_name = raw.get("display_name")
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError("display_name must be a string")
        return await __import__("asyncio").to_thread(
            official_runs.create_run, config.model_dump(mode="json"), display_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/train/runs/{run_id}")
async def get_official_run(run_id: str) -> Dict[str, Any]:
    try:
        return official_runs.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="official run not found") from exc


@app.post("/train/runs/{run_id}/stop")
async def stop_official_run(run_id: str) -> Dict[str, Any]:
    import asyncio

    try:
        return await asyncio.to_thread(official_runs.stop_run, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="official run not found") from exc


@app.get("/train/runs/{run_id}/telemetry")
async def official_run_telemetry(run_id: str) -> Dict[str, Any]:
    try:
        return _run_telemetry(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="official run not found") from exc


@app.get("/train/runs/{run_id}/evaluations")
async def official_run_evaluations(run_id: str) -> list[Dict[str, Any]]:
    try:
        output_dir = _run_output_dir(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="official run not found") from exc
    try:
        rows = (output_dir / "evaluation_history.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for row in rows:
        try:
            item = json.loads(row)
        except ValueError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


@app.get("/train/runs/{run_id}/ppo-diagnostics")
async def official_run_ppo_diagnostics(run_id: str) -> Dict[str, Any]:
    import asyncio

    try:
        output_dir = _run_output_dir(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="official run not found") from exc

    def read_updates() -> list[Dict[str, Any]]:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        tags = {
            "fps": "time/fps", "approx_kl": "train/approx_kl",
            "clip_fraction": "train/clip_fraction", "entropy_loss": "train/entropy_loss",
            "explained_variance": "train/explained_variance", "learning_rate": "train/learning_rate",
            "loss": "train/loss", "policy_gradient_loss": "train/policy_gradient_loss",
            "std": "train/std", "value_loss": "train/value_loss",
        }
        by_step: Dict[int, Dict[str, float]] = {}
        for event_file in output_dir.rglob("events.out.tfevents.*"):
            accumulator = EventAccumulator(str(event_file.parent), size_guidance={"scalars": 0})
            accumulator.Reload()
            available = set(accumulator.Tags().get("scalars", []))
            for key, tag in tags.items():
                if tag in available:
                    for scalar in accumulator.Scalars(tag):
                        if math.isfinite(scalar.value):
                            by_step.setdefault(int(scalar.step), {})[key] = float(scalar.value)
        return [{"step": step, "values": values} for step, values in sorted(by_step.items())]

    try:
        updates = await asyncio.to_thread(read_updates)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="TensorBoard is unavailable") from exc
    return {"run_id": run_id, "updates": updates}


def _watch_snapshot_age_ms(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    """Age of the source frame, not how recently the hub served it."""
    if not isinstance(payload, dict):
        return None
    timestamp = payload.get("ts")
    if not isinstance(timestamp, str):
        return None
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds() * 1000))
    except (TypeError, ValueError, OverflowError):
        return None


def _read_watch_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    """Read one persisted fleet frame, tolerating a concurrent file refresh."""
    for _ in range(3):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("kind") == "fleet":
            return payload
        return None
    return None


def _latest_watch_snapshot() -> Optional[Dict[str, Any]]:
    """Recover an official run's Watch frame after a hub restart."""
    for run in official_runs.list_runs()["runs"]:
        if run.get("state") not in {"starting", "running", "stopping"}:
            continue
        try:
            frame = official_runs.telemetry(str(run["run_id"])).get("fleet")
        except (KeyError, OSError, ValueError):
            frame = None
        if isinstance(frame, dict) and frame.get("runtime") == "official":
            return frame
    return None


@app.get("/train/evaluations")
async def train_evaluations() -> list[Dict[str, Any]]:
    """Return the complete persisted evaluator history for the active run output."""
    run_id = _latest_run_id()
    if not run_id:
        return []
    output_dir = _run_output_dir(run_id)
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
    run_id = _latest_run_id()
    if not run_id:
        return {"run_id": None, "updates": []}
    return await official_run_ppo_diagnostics(run_id)

    # Kept unreachable for one compatibility release while old deployments
    # switch to the per-run endpoint above.
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
    raise HTTPException(status_code=410, detail="Custom-simulator training is disabled. Use POST /train/runs for official AutoDRIVE training.")


@app.post("/train/stop")
async def train_stop() -> Dict[str, Any]:
    raise HTTPException(status_code=410, detail="Use POST /train/runs/{run_id}/stop to stop one official run.")


@app.get("/api/replay/models")
async def replay_models() -> list[Dict[str, Any]]:
    import asyncio

    return await asyncio.to_thread(list_official_models, ROOT / "logs" / "rl")


@app.get("/api/replay/status")
async def replay_status() -> Dict[str, Any]:
    result = replay_job.status()
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
    if map_id != "none" or device not in {"auto", "cpu"}:
        raise HTTPException(status_code=400, detail="Official replay uses its image track and CPU policy runtime")
    try:
        return await asyncio.to_thread(replay_job.start, model_path=model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    return _settings.model_dump(mode="json")


@app.put("/settings")
async def put_settings(body: Settings) -> Dict[str, Any]:
    global _settings
    _settings = body
    path = save_settings(_settings)
    return {
        "ok": True,
        "path": str(path),
        "settings": _settings.model_dump(mode="json"),
    }


@app.post("/telemetry")
async def post_telemetry(request: Request) -> Dict[str, str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be object")
    if payload.get("run_id"):
        official_runs.ingest_telemetry(payload)
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
    official_runs.ingest_telemetry(payload)
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
    official_runs.ingest_evaluator_live(payload)
    bus.publish_evaluator_live(payload)
    return {"ok": "1"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = WSClient(ws)
    bus.attach(client)
    # Resync: push last known status + telemetry
    try:
        await ws.send_json({"type": "status", "payload": await train_status()})
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
