"""Install occupancy map packages from zip uploads (Phase 4).

Accepted layouts (any one):
  - <id>/occupancy/<yaml+image[+meta/centerline/preview]>
  - <id>/<yaml+image[+...]>  (occupancy files at map root → written under occupancy/)
  - flat yaml+image at zip root (requires map_id form/query field)
"""

from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.layer4.hub.maps_catalog import _find_yaml, _parse_ros_yaml

MAP_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
MAX_ZIP_BYTES = 80 * 1024 * 1024  # 80 MiB
MAX_MEMBERS = 200
MAX_MEMBER_BYTES = 60 * 1024 * 1024

_ALLOWED_SUFFIXES = {
    ".yaml",
    ".yml",
    ".pgm",
    ".png",
    ".jpg",
    ".jpeg",
    ".json",
    ".csv",
    ".txt",
    ".md",
}


def validate_map_id(map_id: str) -> str:
    mid = (map_id or "").strip()
    if not MAP_ID_RE.match(mid):
        raise ValueError(
            "map_id must be 1–64 chars: letters, digits, underscore, hyphen "
            "(must start with alphanumeric)"
        )
    if mid.lower() in {"none", "maps", "active"}:
        raise ValueError(f"reserved map id: {mid}")
    return mid


def _safe_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_MEMBERS:
        raise ValueError(f"zip has too many files (>{MAX_MEMBERS})")
    out: List[zipfile.ZipInfo] = []
    for info in infos:
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"unsafe zip path: {info.filename}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"zip member too large: {name}")
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in _ALLOWED_SUFFIXES:
            continue  # skip unknown types quietly
        if not suffix:
            continue
        out.append(info)
    if not out:
        raise ValueError("zip contains no usable map files")
    return out


def _extract_to_temp(zf: zipfile.ZipFile, members: List[zipfile.ZipInfo], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for info in members:
        name = info.filename.replace("\\", "/")
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _collect_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _detect_package(extract_root: Path, map_id_hint: Optional[str]) -> Tuple[str, Path]:
    """Return (map_id, directory that holds yaml+image — may be occupancy or flat)."""
    files = _collect_files(extract_root)
    if not files:
        raise ValueError("empty zip")

    # Prefer .../<id>/occupancy/<yaml>
    for p in files:
        parts = p.relative_to(extract_root).parts
        if len(parts) >= 3 and parts[1].lower() == "occupancy":
            mid = parts[0]
            if map_id_hint and mid != map_id_hint:
                continue
            validate_map_id(mid)
            return mid, extract_root / mid / "occupancy"

    # .../<id>/<yaml> at one folder depth
    top_dirs = sorted({p.relative_to(extract_root).parts[0] for p in files if len(p.relative_to(extract_root).parts) >= 2})
    candidates: List[Tuple[str, Path]] = []
    for d in top_dirs:
        if d.startswith("."):
            continue
        try:
            validate_map_id(d)
        except ValueError:
            continue
        folder = extract_root / d
        if _find_yaml(folder, {}) is not None:
            candidates.append((d, folder))
        occ = folder / "occupancy"
        if occ.is_dir() and _find_yaml(occ, {}) is not None:
            candidates.append((d, occ))

    if map_id_hint:
        for mid, folder in candidates:
            if mid == map_id_hint:
                return mid, folder
        # Flat files under extract root with explicit map_id
        if _find_yaml(extract_root, {}) is not None:
            return validate_map_id(map_id_hint), extract_root

    if len(candidates) == 1:
        return candidates[0]

    if map_id_hint and _find_yaml(extract_root, {}) is not None:
        return validate_map_id(map_id_hint), extract_root

    if not candidates and _find_yaml(extract_root, {}) is not None:
        if not map_id_hint:
            raise ValueError("flat zip requires map_id (form field or ?map_id=)")
        return validate_map_id(map_id_hint), extract_root

    if len(candidates) > 1:
        raise ValueError(
            "zip has multiple map folders; pass map_id to pick one, or upload a single-map package"
        )
    raise ValueError("could not find ROS map yaml + image in zip")


def _ensure_valid_occupancy(occ_src: Path) -> Tuple[Path, Dict[str, Any]]:
    yaml_path = _find_yaml(occ_src, {})
    if yaml_path is None:
        raise ValueError("no ROS map yaml found in package")
    parsed = _parse_ros_yaml(yaml_path.read_text(encoding="utf-8"))
    if not parsed:
        raise ValueError(f"invalid ROS yaml: {yaml_path.name}")
    image_name = str(parsed["image"])
    image_path = occ_src / image_name
    if not image_path.is_file():
        # Sometimes image is basename-only referenced but file lives elsewhere under src
        matches = list(occ_src.rglob(Path(image_name).name))
        if len(matches) == 1:
            image_path = matches[0]
        else:
            raise ValueError(f"image missing for yaml image: {image_name}")
    return yaml_path, parsed


def _write_thumbnail(occ_dir: Path, image_path: Path, max_edge: int = 256) -> Optional[Path]:
    """Write occupancy/preview.png (downscaled) for Fleet catalog thumbnails."""
    try:
        img = Image.open(image_path)
        img = img.convert("L") if img.mode not in ("L", "RGB", "RGBA") else img
        w, h = img.size
        scale = min(1.0, float(max_edge) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.BILINEAR,
            )
        out = occ_dir / "preview.png"
        img.save(out, format="PNG")
        return out
    except OSError:
        return None


def _copy_occupancy_files(src: Path, dest: Path, yaml_path: Path, image_path: Path) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    # Always copy yaml + image (normalize image next to yaml)
    yaml_dest = dest / yaml_path.name
    shutil.copy2(yaml_path, yaml_dest)
    written.append(yaml_dest.name)

    image_dest = dest / Path(image_path).name
    if image_path.resolve() != image_dest.resolve():
        shutil.copy2(image_path, image_dest)
    written.append(image_dest.name)

    # Fix yaml image: field if basename differs from original relative path
    parsed = _parse_ros_yaml(yaml_dest.read_text(encoding="utf-8"))
    if parsed and Path(str(parsed["image"])).name != image_dest.name:
        text = yaml_dest.read_text(encoding="utf-8")
        # Replace image line
        lines = []
        for line in text.splitlines():
            if line.strip().startswith("image:"):
                lines.append(f"image: {image_dest.name}")
            else:
                lines.append(line)
        yaml_dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional extras from same folder
    extras = ("meta.json", "centerline.csv", "preview.png", "preview.jpg", "ATTRIBUTION.md", "README.md")
    for name in extras:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dest / name)
            written.append(name)

    # Also pick up *centerline*.csv / *_preview.*
    for p in src.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith("_centerline.csv") or low == "centerline.csv":
            if p.name not in written:
                dest_name = "centerline.csv" if low.endswith("centerline.csv") else p.name
                shutil.copy2(p, dest / dest_name)
                written.append(dest_name)
        if "preview" in low and low.endswith((".png", ".jpg", ".jpeg")):
            if p.name not in written:
                shutil.copy2(p, dest / p.name)
                written.append(p.name)

    return written


def install_map_zip(
    maps_root: Path,
    zip_bytes: bytes,
    *,
    map_id: Optional[str] = None,
    overwrite: bool = False,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate zip and install under maps_root/<id>/occupancy/. Does not generate mesh."""
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise ValueError(f"zip too large (>{MAX_ZIP_BYTES // (1024 * 1024)} MiB)")
    if not zip_bytes:
        raise ValueError("empty upload")

    hint = validate_map_id(map_id) if map_id else None

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("not a valid zip file") from exc

    with zf:
        members = _safe_members(zf)
        # Extract to a temp dir under maps_root
        staging = maps_root / ".upload_staging"
        if staging.exists():
            shutil.rmtree(staging)
        try:
            _extract_to_temp(zf, members, staging)
            mid, occ_src = _detect_package(staging, hint)
            yaml_path, _parsed = _ensure_valid_occupancy(occ_src)
            image_name = str(_parsed["image"])
            image_path = occ_src / image_name
            if not image_path.is_file():
                matches = list(occ_src.rglob(Path(image_name).name))
                if len(matches) != 1:
                    raise ValueError(f"image missing: {image_name}")
                image_path = matches[0]

            dest_map = maps_root / mid
            dest_occ = dest_map / "occupancy"
            if dest_map.exists() and not overwrite:
                raise ValueError(
                    f"map '{mid}' already exists — pass overwrite=true to replace occupancy"
                )
            if dest_occ.exists() and overwrite:
                # Keep mesh/ unless overwrite clears it? Keep mesh; replace occupancy only.
                for child in list(dest_occ.iterdir()):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)
            dest_occ.mkdir(parents=True, exist_ok=True)

            written = _copy_occupancy_files(occ_src, dest_occ, yaml_path, image_path)

            # meta.json
            meta_path = dest_occ / "meta.json"
            meta: Dict[str, Any] = {}
            if meta_path.is_file():
                try:
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        meta = raw
                except (OSError, json.JSONDecodeError):
                    meta = {}
            if label and label.strip():
                meta["label"] = label.strip()
            elif "label" not in meta:
                meta["label"] = mid.replace("_", " ").replace("-", " ").title()
            meta.setdefault("source", "upload")
            if "map_yaml" not in meta:
                meta["map_yaml"] = yaml_path.name
            # Preserve mesh status if mesh still present
            mesh_dir = dest_map / "mesh"
            if (mesh_dir / "track.obj").is_file() or (mesh_dir / "track_col.obj").is_file():
                mesh = meta.get("mesh")
                if not isinstance(mesh, dict):
                    meta["mesh"] = {"status": "ready"}
            else:
                mesh = meta.get("mesh")
                if isinstance(mesh, dict):
                    mesh["status"] = "none"
                else:
                    meta["mesh"] = {"status": "none"}
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            if "meta.json" not in written:
                written.append("meta.json")

            # Thumbnail if none present
            preview_names = ("preview.png", "preview.jpg", "Porto_preview.png")
            has_preview = any((dest_occ / n).is_file() for n in preview_names) or any(
                p.name.lower().endswith("_preview.png") or p.name.lower().endswith("_preview.jpg")
                for p in dest_occ.iterdir()
                if p.is_file()
            )
            if not has_preview:
                thumb = _write_thumbnail(dest_occ, dest_occ / Path(image_path).name)
                if thumb is not None:
                    written.append(thumb.name)

            return {
                "ok": True,
                "id": mid,
                "path": str(dest_occ),
                "files": sorted(set(written)),
                "overwrite": overwrite,
            }
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
