"""Persist active map id and restart compose sims (Phase 3).

Mission Control only calls POST /api/maps/{id}/activate — all logic lives here.
Sims read maps/.active_map.json (or AICAR_MAP_ID) at boot via entrypoint + TrackLoader.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.layer4.hub.maps_catalog import mesh_is_ready

ACTIVE_MAP_FILENAME = ".active_map.json"


def active_map_path(maps_root: Path) -> Path:
    return maps_root / ACTIVE_MAP_FILENAME


def trackloader_available(repo_root: Optional[Path] = None) -> bool:
    """True when a TrackLoader-enabled player was installed (marker or env)."""
    env = os.environ.get("AICAR_TRACKLOADER", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    candidates: List[Path] = []
    if repo_root is not None:
        candidates.append(repo_root / "simulator" / ".aicar_trackloader")
    candidates.append(Path("/app/simulator/.aicar_trackloader"))
    return any(p.is_file() for p in candidates)


def read_active_map(maps_root: Optional[Path]) -> Dict[str, Any]:
    """Return {id, activated_at, path} — id None means builtin default track."""
    if maps_root is None:
        return {"id": None, "activated_at": None, "path": None}
    path = active_map_path(maps_root)
    if not path.is_file():
        return {"id": None, "activated_at": None, "path": str(path)}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"id": None, "activated_at": None, "path": str(path)}
    if not isinstance(raw, dict):
        return {"id": None, "activated_at": None, "path": str(path)}
    mid = raw.get("id")
    if mid is not None and not isinstance(mid, str):
        mid = None
    if isinstance(mid, str):
        mid = mid.strip() or None
    if mid == "none":
        mid = None
    activated_at = raw.get("activated_at")
    if activated_at is not None and not isinstance(activated_at, str):
        activated_at = None
    return {"id": mid, "activated_at": activated_at, "path": str(path)}


def write_active_map(maps_root: Path, map_id: Optional[str]) -> Dict[str, Any]:
    """Write maps/.active_map.json. map_id None/none clears custom track."""
    path = active_map_path(maps_root)
    maps_root.mkdir(parents=True, exist_ok=True)
    mid: Optional[str] = None
    if map_id is not None:
        mid = str(map_id).strip() or None
    if mid == "none":
        mid = None
    payload = {
        # Always a JSON string so naive parsers (Unity MapConfig / shell sed)
        # never mistake the next key ("activated_at") for the map id when id is null.
        "id": mid if mid is not None else "none",
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"id": mid, "activated_at": payload["activated_at"], "path": str(path)}


def resolve_map_id_for_sim(maps_root: Optional[Path] = None) -> Optional[str]:
    """Env AICAR_MAP_ID wins; else maps/.active_map.json. Used by local Racer launch."""
    env = os.environ.get("AICAR_MAP_ID", "").strip()
    if env and env.lower() != "none":
        return env
    if maps_root is None:
        return None
    return read_active_map(maps_root).get("id")


def validate_activatable(maps_root: Path, map_id: str) -> None:
    """Raise ValueError if map cannot be activated."""
    if map_id == "none" or not map_id.strip():
        return  # clear / builtin always ok
    map_dir = maps_root / map_id
    if not map_dir.is_dir():
        raise ValueError(f"unknown map id: {map_id}")
    if not mesh_is_ready(map_dir):
        raise ValueError(
            f"mesh not ready for '{map_id}'; run generate-mesh first"
        )
    mesh_dir = map_dir / "mesh"
    if not (mesh_dir / "track.obj").is_file() and not (mesh_dir / "track_col.obj").is_file():
        raise ValueError(f"no track.obj / track_col.obj under {mesh_dir}")


def activate_map(
    maps_root: Path,
    map_id: str,
    *,
    restart: bool = True,
    train_running: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Validate, persist active map, optionally restart compose sims.

    Returns API payload. Raises ValueError for bad map; RuntimeError if train blocks.
    """
    if train_running and not force:
        raise RuntimeError(
            "train job is running — stop train first, or pass force=true to restart sims anyway"
        )

    mid = map_id.strip() if map_id else "none"
    validate_activatable(maps_root, mid)
    written = write_active_map(maps_root, None if mid == "none" else mid)

    spawn: Optional[Dict[str, Any]] = None
    spawn_error: Optional[str] = None
    if mid and mid != "none":
        try:
            from src.layer4.hub.spawn import ensure_spawn

            spawn = ensure_spawn(mid, maps_root, generate_if_missing=True)
        except Exception as exc:
            spawn_error = str(exc)

    restarted: List[str] = []
    restart_error: Optional[str] = None
    if restart:
        try:
            from src.layer4.hub.docker_control import restart_compose_sims

            restarted = restart_compose_sims()
        except Exception as exc:
            restart_error = str(exc)

    return {
        "ok": True,
        "active": written,
        "spawn": spawn,
        "spawn_error": spawn_error,
        "restarted": restarted,
        "restart_error": restart_error,
        "trackloader": trackloader_available(),
        "note": (
            None
            if trackloader_available()
            else (
                "Active map written and sims restarted, but TrackLoader-enabled player "
                "is not marked installed (touch simulator/.aicar_trackloader after rebuild). "
                "Stock AutoDRIVE binary ignores AICAR_MAP_ID / -map-id."
            )
        ),
    }
