"""Scan simulator/maps for occupancy grids (Fleet catalog)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MapInfo:
    """One discoverable track under simulator/maps/<id>/."""

    id: str
    label: str
    yaml_url: str
    image_url: str
    mesh_status: str
    overlay_only: bool
    active: bool = False
    source: Optional[str] = None
    thumbnail_url: Optional[str] = None
    centerline_url: Optional[str] = None
    centerline_status: str = "none"
    mesh_preview_url: Optional[str] = None

    def to_api(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def resolve_maps_root(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Prefer AICAR_MAPS_DIR, then Docker / host simulator/maps (legacy assets/maps last)."""
    env = os.environ.get("AICAR_MAPS_DIR", "").strip()
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/app/simulator/maps"))
    if repo_root is not None:
        candidates.append(repo_root / "simulator" / "maps")
        candidates.append(repo_root / "assets" / "maps")
    candidates.append(Path("/app/assets/maps"))
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def _read_meta(occupancy_dir: Path) -> Dict[str, Any]:
    meta_path = occupancy_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _parse_ros_yaml(text: str) -> Optional[Dict[str, Any]]:
    """Minimal ROS map_server yaml subset (key: value)."""
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = re.sub(r"#.*$", "", raw).strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            parts = [p.strip() for p in val[1:-1].split(",")]
            try:
                out[key] = [float(p) for p in parts]
            except ValueError:
                out[key] = parts
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    image = out.get("image")
    resolution = out.get("resolution")
    origin = out.get("origin")
    if not isinstance(image, str) or not image:
        return None
    if not isinstance(resolution, (int, float)):
        return None
    if not isinstance(origin, list) or len(origin) < 2:
        return None
    return out


def _find_yaml(occupancy_dir: Path, meta: Dict[str, Any]) -> Optional[Path]:
    preferred = meta.get("map_yaml")
    if isinstance(preferred, str) and preferred.strip():
        p = occupancy_dir / preferred.strip()
        if p.is_file():
            return p
    for name in ("map.yaml", "Map.yaml"):
        p = occupancy_dir / name
        if p.is_file():
            return p
    yamls = sorted(occupancy_dir.glob("*.yaml")) + sorted(occupancy_dir.glob("*.yml"))
    # Prefer non-meta style names; skip nothing special
    for p in yamls:
        if p.name.lower() == "meta.yaml":
            continue
        return p
    return None


def _label_from_id(map_id: str) -> str:
    return map_id.replace("_", " ").replace("-", " ").strip().title() or map_id


def _mesh_status(map_dir: Path, meta: Dict[str, Any]) -> str:
    mesh_meta = meta.get("mesh")
    if isinstance(mesh_meta, dict):
        status = mesh_meta.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    mesh_dir = map_dir / "mesh"
    if (mesh_dir / "track.obj").is_file() or (mesh_dir / "track_col.obj").is_file():
        return "ready"
    return "none"


def mesh_is_ready(map_dir: Path) -> bool:
    """True when generate-mesh (or equivalent) produced track OBJs."""
    occ = _occupancy_dir(map_dir)
    return _mesh_status(map_dir, _read_meta(occ)) == "ready"


def _occupancy_dir(map_dir: Path) -> Path:
    """Support maps/<id>/occupancy/ (preferred) or legacy maps/<id>/*.yaml flat."""
    nested = map_dir / "occupancy"
    if nested.is_dir():
        return nested
    return map_dir


_BROWSER_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _is_browser_image(path: Path) -> bool:
    return path.suffix.lower() in _BROWSER_IMAGE_SUFFIXES


def _write_occupancy_preview_png(
    occ: Path, image_path: Path, *, max_edge: int = 256
) -> Optional[Path]:
    """Downscale occupancy image → occupancy/preview.png (browser-safe)."""
    out = occ / "preview.png"
    if out.is_file():
        return out
    if not image_path.is_file():
        return None
    try:
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        w, h = img.size
        scale = min(1.0, float(max_edge) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                PILImage.Resampling.BILINEAR,
            )
        img.save(out, format="PNG")
        return out
    except Exception:
        return None


def _find_thumbnail(
    occ: Path, image_path: Path, maps_root: Path, map_id: str
) -> Optional[str]:
    """Prefer occupancy/preview.png; else a browser-safe ROS image.

    Never return .pgm/.pbm URLs — browsers cannot render them in <img>, which
    made Maps/Train thumbnails look broken. If only a PGM exists, generate
    occupancy/preview.png.

    Do not prefer arbitrary *_preview.* files — e.g. Porto_preview.png is a wide
    decorative render that does not match Porto.pgm orientation/aspect.
    """
    _ = map_id
    generated = occ / "preview.png"
    if generated.is_file():
        chosen: Optional[Path] = generated
    elif _is_browser_image(image_path) and image_path.is_file():
        chosen = image_path
    else:
        chosen = _write_occupancy_preview_png(occ, image_path)

    if chosen is None or not chosen.is_file() or not _is_browser_image(chosen):
        return None
    try:
        return f"/maps/{chosen.relative_to(maps_root).as_posix()}"
    except ValueError:
        return None


def _find_mesh_preview_url(map_dir: Path, maps_root: Path) -> Optional[str]:
    p = map_dir / "mesh" / "preview.png"
    if not p.is_file():
        return None
    try:
        return f"/maps/{p.relative_to(maps_root).as_posix()}"
    except ValueError:
        return None


def _find_centerline_url(
    occ: Path, maps_root: Path, meta: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    status = "none"
    cl_meta = meta.get("centerline")
    if isinstance(cl_meta, dict):
        st = cl_meta.get("status")
        if isinstance(st, str) and st.strip():
            status = st.strip()
    path: Optional[Path] = None
    preferred = occ / "centerline.csv"
    if preferred.is_file():
        path = preferred
    else:
        matches = sorted(occ.glob("*centerline*.csv"))
        if matches:
            path = matches[0]
    if path is None:
        return None, status if status != "ready" else "none"
    if status == "none":
        status = "ready"
    try:
        return f"/maps/{path.relative_to(maps_root).as_posix()}", status
    except ValueError:
        return None, status


def _read_active_id(maps_root: Path) -> Optional[str]:
    """Best-effort read of maps/.active_map.json without importing map_activate."""
    path = maps_root / ".active_map.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    mid = raw.get("id")
    if not isinstance(mid, str):
        return None
    mid = mid.strip()
    if not mid or mid == "none":
        return None
    return mid


def _trackloader_available() -> bool:
    env = os.environ.get("AICAR_TRACKLOADER", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    return Path("/app/simulator/.aicar_trackloader").is_file() or (
        Path(__file__).resolve().parents[3] / "simulator" / ".aicar_trackloader"
    ).is_file()


def scan_maps(maps_root: Path) -> List[MapInfo]:
    """Discover valid map folders under maps_root."""
    if not maps_root.is_dir():
        return []
    active_id = _read_active_id(maps_root)
    loader_ok = _trackloader_available()
    found: List[MapInfo] = []
    for child in sorted(maps_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        map_id = child.name
        occ = _occupancy_dir(child)
        meta = _read_meta(occ)
        yaml_path = _find_yaml(occ, meta)
        if yaml_path is None:
            continue
        try:
            parsed = _parse_ros_yaml(yaml_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not parsed:
            continue
        image_name = str(parsed["image"])
        image_path = occ / image_name
        if not image_path.is_file():
            continue

        # URLs relative to StaticFiles mount of maps_root
        try:
            yaml_rel = yaml_path.relative_to(maps_root).as_posix()
            image_rel = image_path.relative_to(maps_root).as_posix()
        except ValueError:
            continue

        label = meta.get("label")
        if not isinstance(label, str) or not label.strip():
            label = _label_from_id(map_id)
        else:
            label = label.strip()

        source = meta.get("source")
        if source is not None and not isinstance(source, str):
            source = None

        mesh = _mesh_status(child, meta)
        is_active = active_id == map_id
        # Physics honest: Fleet overlay only unless TrackLoader player + mesh + active
        overlay_only = not (loader_ok and mesh == "ready" and is_active)
        thumb = _find_thumbnail(occ, image_path, maps_root, map_id)
        cl_url, cl_status = _find_centerline_url(occ, maps_root, meta)
        mesh_prev = _find_mesh_preview_url(child, maps_root)

        found.append(
            MapInfo(
                id=map_id,
                label=label,
                yaml_url=f"/maps/{yaml_rel}",
                image_url=f"/maps/{image_rel}",
                mesh_status=mesh,
                overlay_only=overlay_only,
                active=is_active,
                source=source,
                thumbnail_url=thumb,
                centerline_url=cl_url,
                centerline_status=cl_status,
                mesh_preview_url=mesh_prev,
            )
        )
    return found


def list_maps_api(maps_root: Optional[Path]) -> List[Dict[str, Any]]:
    """API payload including synthetic 'none' grid option."""
    items = scan_maps(maps_root) if maps_root is not None else []
    active_id = _read_active_id(maps_root) if maps_root is not None else None
    out = [m.to_api() for m in items]
    out.append(
        {
            "id": "none",
            "label": "Grid only (builtin track)",
            "yaml_url": None,
            "image_url": None,
            "mesh_status": "none",
            "overlay_only": True,
            "active": active_id is None,
            "source": None,
            "thumbnail_url": None,
            "centerline_url": None,
            "centerline_status": "none",
            "mesh_preview_url": None,
        }
    )
    return out
