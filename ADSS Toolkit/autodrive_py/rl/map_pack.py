"""Map pack registry: cached generation, roles, and sealed holdouts.

One manifest (``maps/map_pack.json``) is the single source of truth for *which
maps a trainer is allowed to touch*:

* ``train_ok``   — free to train on
* ``validation`` — pinned for model selection; never trained on (one map)
* ``holdout``    — sealed for official scoring; hash-pinned, never trained on

Seals are content hashes taken at seal time, so a regenerated or edited holdout
fails :func:`verify_pack` instead of silently invalidating a pinned baseline.
Maps sealed in the frozen ``eval_protocol.yaml`` are always treated as holdouts,
even if the manifest is missing or stale.

API for siblings (no UI/trainer edits needed)::

    from .map_pack import (
        ensure_maps, train_safe_maps, validation_map, holdout_maps,
        is_holdout, assert_train_safe, verify_pack, pack_fingerprint,
    )

CLI::

    python -m rl.map_pack generate --num 5 [--seal|--validation] [--pgm]
    python -m rl.map_pack list [--role train_ok] [--train-safe|--holdout] [--ids|--json]
    python -m rl.map_pack seal map4 --reason "official holdout H2"
    python -m rl.map_pack pin-validation map3
    python -m rl.map_pack verify [--json]          # re-hash; nonzero exit if broken
    python -m rl.map_pack is-holdout map2          # exit 0 = holdout, 3 = not
    python -m rl.map_pack fingerprint              # provenance block for config.json

Regression smoke: ``python -m rl.test_map_pack``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

PACK_VERSION = 1
MANIFEST_NAME = "map_pack.json"

ROLE_TRAIN_OK = "train_ok"
ROLE_VALIDATION = "validation"
ROLE_HOLDOUT = "holdout"
ROLES = (ROLE_TRAIN_OK, ROLE_VALIDATION, ROLE_HOLDOUT)


class HoldoutViolation(RuntimeError):
    """Raised when a sealed holdout (or the validation pin) is used for training."""


class SealBroken(RuntimeError):
    """Raised when a sealed map's content no longer matches its recorded hash."""


# --------------------------------------------------------------------------- #
# paths / hashing
# --------------------------------------------------------------------------- #


def default_maps_root() -> Path:
    return Path(__file__).resolve().parent / "maps"


def _root(maps_root: Path | str | None) -> Path:
    return Path(maps_root) if maps_root else default_maps_root()


def manifest_path(maps_root: Path | str | None = None) -> Path:
    return _root(maps_root) / MANIFEST_NAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def map_yaml_for(map_id: str, maps_root: Path | str | None = None) -> Path | None:
    p = _root(maps_root) / str(map_id) / f"{map_id}.yaml"
    return p if p.is_file() else None


def map_parts(map_yaml: Path | str) -> list[Path]:
    """Canonical files that define a map: yaml, referenced image, centerline, start pose.

    Deliberately excludes extra twins (e.g. an added ``.pgm``) so a seal tracks
    geometry, not packaging.
    """
    map_yaml = Path(map_yaml)
    parts = [map_yaml]
    image = None
    try:
        meta = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
        if isinstance(meta, dict):
            image = meta.get("image")
    except (OSError, yaml.YAMLError):
        image = None
    if image:
        img = map_yaml.parent / str(image)
        if not img.is_file():
            alt = map_yaml.parent / (Path(str(image)).stem + ".png")
            img = alt if alt.is_file() else img
        parts.append(img)
    parts.append(map_yaml.parent / "centerline.csv")
    parts.append(map_yaml.parent / "start_pose.txt")
    return [p for p in parts if p.is_file()]


def map_content_hash(map_yaml: Path | str) -> str:
    """Digest over every canonical part — catches centerline/start-pose tampering."""
    h = hashlib.sha256()
    for p in map_parts(map_yaml):
        data = p.read_bytes()
        h.update(p.name.encode("utf-8"))
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()[:16]


def map_identity_hash(map_yaml: Path | str) -> str:
    """Protocol-compatible hash (yaml + image) as recorded in official eval rows."""
    from .eval_protocol import map_file_hash

    return map_file_hash(Path(map_yaml))


# --------------------------------------------------------------------------- #
# frozen protocol bridge
# --------------------------------------------------------------------------- #


def _protocol() -> dict | None:
    try:
        from .eval_protocol import load_protocol

        return load_protocol()
    except Exception:
        return None


def protocol_sealed_ids() -> set[str]:
    proto = _protocol()
    if not proto:
        return set()
    return {str(m["id"]) for m in (proto.get("maps") or []) if m.get("sealed")}


def protocol_train_ok_ids() -> set[str]:
    proto = _protocol()
    if not proto:
        return set()
    return {str(m) for m in (proto.get("train_ok_maps") or [])}


# --------------------------------------------------------------------------- #
# manifest load / save
# --------------------------------------------------------------------------- #


def _empty_pack() -> dict:
    from .contracts import CONTRACTS_VERSION

    return {
        "pack_version": PACK_VERSION,
        "contracts_version": CONTRACTS_VERSION,
        "updated_utc": None,
        "validation_map": None,
        "maps": {},
    }


def _blank_entry(map_id: str, role: str) -> dict:
    return {
        "id": str(map_id),
        "role": role,
        "sealed": role == ROLE_HOLDOUT,
        "map_hash": None,
        "content_hash": None,
        "seal_hash": None,
        "hashed_utc": None,
        "sealed_utc": None,
        "seal_reason": None,
        "gen": {"source": "adopted"},
    }


def _disk_map_ids(maps_root: Path) -> list[str]:
    if not maps_root.is_dir():
        return []
    ids = []
    for d in sorted(maps_root.iterdir()):
        if d.is_dir() and (d / f"{d.name}.yaml").is_file():
            ids.append(d.name)
    return ids


def load_pack(maps_root: Path | str | None = None, *, adopt: bool = True) -> dict:
    """Read the manifest (cheap — never hashes).

    With ``adopt``, maps present on disk but absent from the manifest appear as
    in-memory entries, and protocol-sealed maps are forced to sealed holdouts.
    Nothing is written; call :func:`adopt_disk_maps` to persist.
    """
    root = _root(maps_root)
    path = root / MANIFEST_NAME
    pack = _empty_pack()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("maps"), dict):
                pack.update(data)
        except (OSError, json.JSONDecodeError):
            pass
    if not adopt:
        return pack

    sealed = protocol_sealed_ids()
    train_ok = protocol_train_ok_ids()
    maps = pack["maps"]
    for mid in _disk_map_ids(root):
        if mid not in maps:
            if mid in sealed:
                role = ROLE_HOLDOUT
            elif mid in train_ok:
                role = ROLE_TRAIN_OK
            else:
                role = ROLE_TRAIN_OK
            maps[mid] = _blank_entry(mid, role)
    for mid in sealed:
        entry = maps.get(mid)
        if entry is None:
            # Not in this root: absence is reported by verify_pack, not invented here.
            continue
        # The frozen protocol outranks the manifest: never un-seal from here.
        entry["role"] = ROLE_HOLDOUT
        entry["sealed"] = True
        if not entry.get("seal_reason"):
            entry["seal_reason"] = "sealed in eval_protocol.yaml"
    for mid, entry in maps.items():
        entry["id"] = mid
        entry["exists"] = map_yaml_for(mid, root) is not None
    return pack


def save_pack(pack: dict, maps_root: Path | str | None = None) -> Path:
    """Atomically persist the manifest (transient ``exists`` flags stripped)."""
    root = _root(maps_root)
    root.mkdir(parents=True, exist_ok=True)
    out = dict(pack)
    out["pack_version"] = PACK_VERSION
    out["updated_utc"] = _utc_now()
    out["maps"] = {
        mid: {k: v for k, v in entry.items() if k != "exists"}
        for mid, entry in sorted((pack.get("maps") or {}).items())
    }
    path = root / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _refresh_hashes(entry: dict, map_yaml: Path) -> dict:
    entry["map_hash"] = map_identity_hash(map_yaml)
    entry["content_hash"] = map_content_hash(map_yaml)
    entry["hashed_utc"] = _utc_now()
    return entry


def _backfill_seal(entry: dict) -> dict:
    """Give a protocol-sealed map its seal baseline the first time we hash it."""
    if entry.get("sealed") and not entry.get("seal_hash") and entry.get("content_hash"):
        entry["seal_hash"] = entry["content_hash"]
        entry["sealed_utc"] = entry.get("sealed_utc") or _utc_now()
        entry["seal_reason"] = entry.get("seal_reason") or "sealed holdout"
    return entry


# --------------------------------------------------------------------------- #
# registration / generation
# --------------------------------------------------------------------------- #


def adopt_disk_maps(maps_root: Path | str | None = None) -> dict:
    """Persist every on-disk map into the manifest, hashing anything unhashed."""
    root = _root(maps_root)
    pack = load_pack(root)
    for mid, entry in pack["maps"].items():
        my = map_yaml_for(mid, root)
        if my is not None and not entry.get("content_hash"):
            _refresh_hashes(entry, my)
        _backfill_seal(entry)
    save_pack(pack, root)
    return pack


def register_specs(
    specs: Iterable[dict],
    *,
    maps_root: Path | str | None = None,
    role: str | None = None,
    seal: bool = False,
    seal_reason: str | None = None,
    validation: bool = False,
) -> dict:
    """Record generated/cached map specs (from ``trackgen``) in the manifest.

    ``role`` / ``seal`` / ``validation`` apply only to *newly generated* maps so
    re-running a cached ``generate`` never reclassifies an existing map.
    """
    root = _root(maps_root)
    pack = load_pack(root)
    sealed_by_protocol = protocol_sealed_ids()
    fresh: list[str] = []

    for spec in specs:
        mid = str(spec["id"])
        entry = pack["maps"].get(mid) or _blank_entry(mid, ROLE_TRAIN_OK)
        pack["maps"][mid] = entry
        is_new = not spec.get("cached")
        if is_new:
            fresh.append(mid)
            entry["gen"] = {
                "source": "trackgen",
                "seed": spec.get("gen_seed"),
                "index": spec.get("index"),
                "attempt": spec.get("gen_attempt"),
                "version": spec.get("gen_version"),
                "resolution": spec.get("resolution"),
                "scale": spec.get("scale"),
                "has_pgm": spec.get("has_pgm"),
                "n_centerline": spec.get("n_centerline"),
                "created_utc": _utc_now(),
            }
        my = map_yaml_for(mid, root)
        if my is not None and (is_new or not entry.get("content_hash")):
            _refresh_hashes(entry, my)
        if is_new and mid not in sealed_by_protocol:
            if role in ROLES:
                entry["role"] = role
            if validation:
                entry["role"] = ROLE_VALIDATION
            if seal:
                entry["role"] = ROLE_HOLDOUT
                entry["sealed"] = True
                entry["sealed_utc"] = _utc_now()
                entry["seal_reason"] = seal_reason or "sealed holdout"
                entry["seal_hash"] = entry.get("content_hash")
        _backfill_seal(entry)

    if validation and fresh:
        pack["validation_map"] = fresh[-1]
    save_pack(pack, root)
    return pack


def ensure_maps(
    num_maps: int = 3,
    *,
    seed: int = 123,
    prefix: str = "map",
    maps_root: Path | str | None = None,
    start_index: int = 0,
    role: str | None = None,
    seal: bool = False,
    seal_reason: str | None = None,
    validation: bool = False,
    force: bool = False,
    resolution: float = 0.05,
    scale: float = 1.0,
    write_pgm: bool = False,
) -> dict:
    """Ensure ``num_maps`` maps exist (cached: only missing ids are drawn).

    Returns ``{"created": [...], "cached": [...], "pack": {...}}``.
    Refuses to redraw a sealed map even with ``force``.
    """
    from .trackgen import generate_map_specs

    root = _root(maps_root)
    pack = load_pack(root)
    if force:
        blocked = [
            f"{prefix}{i}"
            for i in range(int(start_index), int(start_index) + int(num_maps))
            if (pack["maps"].get(f"{prefix}{i}") or {}).get("sealed")
        ]
        if blocked:
            raise HoldoutViolation(
                f"refusing to redraw sealed map(s): {', '.join(blocked)} "
                "(unseal explicitly first; regenerating invalidates pinned baselines)"
            )

    specs = generate_map_specs(
        root,
        num_maps=int(num_maps),
        seed=int(seed),
        prefix=str(prefix),
        start_index=int(start_index),
        skip_existing=not force,
        resolution=float(resolution),
        scale=float(scale),
        write_pgm=bool(write_pgm),
    )
    pack = register_specs(
        specs,
        maps_root=root,
        role=role,
        seal=seal,
        seal_reason=seal_reason,
        validation=validation,
    )
    return {
        "created": [s["id"] for s in specs if not s.get("cached")],
        "cached": [s["id"] for s in specs if s.get("cached")],
        "requested": int(num_maps),
        "pack": pack,
    }


# --------------------------------------------------------------------------- #
# queries (what siblings call)
# --------------------------------------------------------------------------- #


def list_maps(
    maps_root: Path | str | None = None,
    *,
    role: str | None = None,
    pack: dict | None = None,
    include_missing: bool = False,
) -> list[dict]:
    """Map picker / CLI rows: id, role, sealed, hashes, label. Never hashes on disk."""
    root = _root(maps_root)
    pack = pack or load_pack(root)
    validation = pack.get("validation_map")
    rows = []
    for mid, entry in sorted(pack.get("maps", {}).items()):
        if not entry.get("exists", True) and not include_missing:
            continue
        r = str(entry.get("role") or ROLE_TRAIN_OK)
        sealed = bool(entry.get("sealed"))
        if mid == validation and r == ROLE_TRAIN_OK and not sealed:
            r = ROLE_VALIDATION
        if role and r != role:
            continue
        tag = " [HOLDOUT]" if sealed else (" [VALIDATION]" if r == ROLE_VALIDATION else "")
        rows.append(
            {
                "id": mid,
                "role": r,
                "sealed": sealed,
                "train_safe": (not sealed) and r == ROLE_TRAIN_OK,
                "map_hash": entry.get("map_hash"),
                "content_hash": entry.get("content_hash"),
                "seal_hash": entry.get("seal_hash"),
                "seal_reason": entry.get("seal_reason"),
                "exists": bool(entry.get("exists", True)),
                "label": f"{mid}{tag}",
            }
        )
    return rows


def map_ids(maps_root: Path | str | None = None, **kwargs) -> list[str]:
    return [r["id"] for r in list_maps(maps_root, **kwargs)]


def train_safe_maps(maps_root: Path | str | None = None, *, pack: dict | None = None) -> list[str]:
    """Maps a trainer may use: not sealed, not the validation pin."""
    return [r["id"] for r in list_maps(maps_root, pack=pack) if r["train_safe"]]


def holdout_maps(maps_root: Path | str | None = None, *, pack: dict | None = None) -> list[str]:
    return [r["id"] for r in list_maps(maps_root, pack=pack) if r["sealed"]]


def validation_map(maps_root: Path | str | None = None, *, pack: dict | None = None) -> str | None:
    """The pinned selection map (used to promote best_model, never trained on)."""
    root = _root(maps_root)
    pack = pack or load_pack(root)
    mid = pack.get("validation_map")
    if mid and map_yaml_for(mid, root) is not None:
        return str(mid)
    for r in list_maps(root, pack=pack, role=ROLE_VALIDATION):
        return r["id"]
    return None


def is_holdout(
    map_id: str, maps_root: Path | str | None = None, *, pack: dict | None = None
) -> bool:
    """True if sealed in the manifest or in the frozen eval protocol."""
    mid = str(map_id)
    if mid in protocol_sealed_ids():
        return True
    pack = pack or load_pack(_root(maps_root))
    return bool((pack.get("maps", {}).get(mid) or {}).get("sealed"))


def is_train_safe(
    map_id: str, maps_root: Path | str | None = None, *, pack: dict | None = None
) -> bool:
    root = _root(maps_root)
    pack = pack or load_pack(root)
    return str(map_id) in set(train_safe_maps(root, pack=pack))


def map_info(
    map_id: str, maps_root: Path | str | None = None, *, pack: dict | None = None
) -> dict | None:
    root = _root(maps_root)
    pack = pack or load_pack(root)
    entry = pack.get("maps", {}).get(str(map_id))
    if entry is None:
        return None
    rows = {r["id"]: r for r in list_maps(root, pack=pack, include_missing=True)}
    info = dict(rows.get(str(map_id), {}))
    info["gen"] = entry.get("gen")
    info["sealed_utc"] = entry.get("sealed_utc")
    info["hashed_utc"] = entry.get("hashed_utc")
    my = map_yaml_for(map_id, root)
    info["map_yaml"] = str(my) if my else None
    return info


def assert_train_safe(
    ids: Iterable[str],
    *,
    maps_root: Path | str | None = None,
    allow_holdout: bool = False,
    allow_validation: bool = False,
    pack: dict | None = None,
) -> None:
    """Gate for trainers: raise :class:`HoldoutViolation` on sealed / validation maps."""
    root = _root(maps_root)
    pack = pack or load_pack(root)
    sealed, val_used = [], []
    val = validation_map(root, pack=pack)
    for mid in ids:
        mid = str(mid)
        if not allow_holdout and is_holdout(mid, root, pack=pack):
            sealed.append(mid)
        elif not allow_validation and val and mid == val:
            val_used.append(mid)
    problems = []
    if sealed:
        problems.append(
            f"sealed holdout(s) {', '.join(sealed)}: pass allow_holdout/--allow-holdout to override"
        )
    if val_used:
        problems.append(
            f"validation pin {', '.join(val_used)}: training on it breaks model selection"
        )
    if problems:
        raise HoldoutViolation("; ".join(problems))


# --------------------------------------------------------------------------- #
# seals
# --------------------------------------------------------------------------- #


def seal_maps(
    ids: Iterable[str],
    *,
    maps_root: Path | str | None = None,
    reason: str | None = None,
) -> dict:
    """Seal maps as holdouts, pinning their current content hash."""
    root = _root(maps_root)
    pack = load_pack(root)
    for mid in ids:
        mid = str(mid)
        my = map_yaml_for(mid, root)
        if my is None:
            raise FileNotFoundError(f"no map '{mid}' under {root}")
        entry = pack["maps"].setdefault(mid, _blank_entry(mid, ROLE_HOLDOUT))
        _refresh_hashes(entry, my)
        entry["role"] = ROLE_HOLDOUT
        entry["sealed"] = True
        entry["sealed_utc"] = _utc_now()
        entry["seal_reason"] = reason or entry.get("seal_reason") or "sealed holdout"
        entry["seal_hash"] = entry["content_hash"]
        if pack.get("validation_map") == mid:
            pack["validation_map"] = None
    save_pack(pack, root)
    return pack


def unseal_maps(
    ids: Iterable[str], *, maps_root: Path | str | None = None, force: bool = False
) -> dict:
    """Unseal maps (requires ``force``); protocol-sealed maps can never be unsealed."""
    root = _root(maps_root)
    if not force:
        raise HoldoutViolation("unsealing requires force=True: a broken seal invalidates claims")
    pack = load_pack(root)
    protocol_sealed = protocol_sealed_ids()
    for mid in ids:
        mid = str(mid)
        if mid in protocol_sealed:
            raise HoldoutViolation(
                f"'{mid}' is sealed by eval_protocol.yaml (frozen); cannot unseal via map pack"
            )
        entry = pack["maps"].get(mid)
        if entry is None:
            raise FileNotFoundError(f"map '{mid}' not in pack")
        entry["sealed"] = False
        entry["role"] = ROLE_TRAIN_OK
        entry["seal_hash"] = None
        entry["sealed_utc"] = None
        entry["seal_reason"] = f"unsealed {_utc_now()}"
    save_pack(pack, root)
    return pack


def pin_validation_map(map_id: str, *, maps_root: Path | str | None = None) -> dict:
    """Pin the selection map: exists, not sealed, and excluded from train-safe."""
    root = _root(maps_root)
    mid = str(map_id)
    if map_yaml_for(mid, root) is None:
        raise FileNotFoundError(f"no map '{mid}' under {root}")
    if is_holdout(mid, root):
        raise HoldoutViolation(f"'{mid}' is a sealed holdout; pick a non-sealed map for validation")
    pack = load_pack(root)
    previous = pack.get("validation_map")
    if previous and previous != mid and previous in pack["maps"]:
        prev = pack["maps"][previous]
        if prev.get("role") == ROLE_VALIDATION:
            prev["role"] = ROLE_TRAIN_OK
    entry = pack["maps"].setdefault(mid, _blank_entry(mid, ROLE_VALIDATION))
    entry["role"] = ROLE_VALIDATION
    if not entry.get("content_hash"):
        _refresh_hashes(entry, map_yaml_for(mid, root))
    pack["validation_map"] = mid
    save_pack(pack, root)
    return pack


def verify_pack(maps_root: Path | str | None = None, *, pack: dict | None = None) -> dict:
    """Re-hash every map; report missing files, drifted hashes, broken seals.

    Returns ``{"ok": bool, "maps": [{"id","ok","sealed","issues":[...]}], "issues": [...]}``.
    """
    root = _root(maps_root)
    pack = pack or load_pack(root)
    results: list[dict] = []
    pack_issues: list[str] = []

    for mid, entry in sorted(pack.get("maps", {}).items()):
        issues: list[str] = []
        my = map_yaml_for(mid, root)
        if my is None:
            issues.append("map files missing on disk")
            results.append(
                {"id": mid, "ok": False, "sealed": bool(entry.get("sealed")), "issues": issues}
            )
            continue
        parts = {p.name for p in map_parts(my)}
        if "centerline.csv" not in parts:
            issues.append("centerline.csv missing (lap timing would be undefined)")
        if "start_pose.txt" not in parts:
            issues.append("start_pose.txt missing")
        if len(parts) < 3:
            issues.append("map image missing")

        content = map_content_hash(my)
        identity = map_identity_hash(my)
        recorded_content = entry.get("content_hash")
        recorded_identity = entry.get("map_hash")
        if recorded_content and content != recorded_content:
            issues.append(f"content hash drift {recorded_content} -> {content}")
        if recorded_identity and identity != recorded_identity:
            issues.append(f"map hash drift {recorded_identity} -> {identity}")
        if entry.get("sealed"):
            seal = entry.get("seal_hash")
            if not seal:
                issues.append("sealed but no seal hash recorded")
            elif seal != content:
                issues.append(f"SEAL BROKEN: sealed {seal} but map is now {content}")
        results.append(
            {
                "id": mid,
                "ok": not issues,
                "sealed": bool(entry.get("sealed")),
                "role": entry.get("role"),
                "content_hash": content,
                "map_hash": identity,
                "issues": issues,
            }
        )

    val = pack.get("validation_map")
    if val:
        if map_yaml_for(val, root) is None:
            pack_issues.append(f"validation pin '{val}' is not on disk")
        elif is_holdout(val, root, pack=pack):
            pack_issues.append(f"validation pin '{val}' is also a sealed holdout")
        elif val in protocol_train_ok_ids():
            pack_issues.append(
                f"validation pin '{val}' is listed in eval_protocol train_ok_maps "
                "(selection would be biased)"
            )
    else:
        pack_issues.append("no validation map pinned (best_model selection has no clean map)")
    if not any(r["sealed"] for r in results):
        pack_issues.append("no sealed holdout in pack")
    for mid in protocol_sealed_ids():
        if map_yaml_for(mid, root) is None:
            pack_issues.append(f"protocol holdout '{mid}' missing on disk")

    return {
        "ok": all(r["ok"] for r in results) and not pack_issues,
        "maps": results,
        "issues": pack_issues,
        "maps_root": str(root),
    }


def assert_seals_intact(maps_root: Path | str | None = None) -> None:
    """Raise :class:`SealBroken` if any sealed map drifted from its seal hash."""
    report = verify_pack(maps_root)
    broken = [
        f"{r['id']}: {'; '.join(r['issues'])}"
        for r in report["maps"]
        if r["sealed"] and not r["ok"]
    ]
    if broken:
        raise SealBroken("sealed holdout integrity failure -> " + " | ".join(broken))


def pack_fingerprint(maps_root: Path | str | None = None, *, pack: dict | None = None) -> dict:
    """Compact provenance block for ``config.json`` / metrics fingerprints."""
    root = _root(maps_root)
    pack = pack or load_pack(root)
    rows = list_maps(root, pack=pack)
    return {
        "pack_version": pack.get("pack_version", PACK_VERSION),
        "updated_utc": pack.get("updated_utc"),
        "validation_map": validation_map(root, pack=pack),
        "train_ok": [r["id"] for r in rows if r["train_safe"]],
        "holdouts": {r["id"]: r.get("seal_hash") or r.get("content_hash") for r in rows if r["sealed"]},
        "map_hashes": {r["id"]: r.get("map_hash") for r in rows},
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _print_rows(rows: list[dict]) -> None:
    if not rows:
        print("(no maps)")
        return
    width = max(len(r["id"]) for r in rows)
    print(f"{'id'.ljust(width)}  {'role':<11} {'hash':<17} notes")
    for r in rows:
        note = r.get("seal_reason") if r["sealed"] else ""
        if not r["exists"]:
            note = "MISSING ON DISK"
        print(
            f"{r['id'].ljust(width)}  {r['role']:<11} "
            f"{(r.get('content_hash') or '-'):<17} {note or ''}".rstrip()
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rl.map_pack",
        description="Map pack: cached generation, roles, sealed holdouts",
    )
    parser.add_argument("--maps-root", type=str, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="ensure N maps exist (cached; only draws missing ids)")
    g.add_argument("--num", type=int, default=3)
    g.add_argument("--seed", type=int, default=123)
    g.add_argument("--prefix", type=str, default="map")
    g.add_argument("--start-index", type=int, default=0)
    g.add_argument("--role", choices=ROLES, default=None, help="role for newly created maps")
    g.add_argument("--seal", action="store_true", help="seal newly created maps as holdouts")
    g.add_argument("--reason", type=str, default=None, help="seal reason")
    g.add_argument("--validation", action="store_true", help="pin the new map as validation")
    g.add_argument("--resolution", type=float, default=0.05)
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--pgm", action="store_true", help="also write the ~40MB ROS .pgm twin")
    g.add_argument("--force", action="store_true", help="redraw existing (never sealed) maps")

    ls = sub.add_parser("list", help="list maps with roles/seals")
    ls.add_argument("--role", choices=ROLES, default=None)
    ls.add_argument("--train-safe", action="store_true", help="only train-safe ids")
    ls.add_argument("--holdout", action="store_true", help="only sealed holdout ids")
    ls.add_argument("--ids", action="store_true", help="print bare comma-separated ids")
    ls.add_argument("--json", action="store_true")

    sub.add_parser("adopt", help="register on-disk maps into the manifest (hash them)")

    se = sub.add_parser("seal", help="seal maps as sealed holdouts")
    se.add_argument("ids", nargs="+")
    se.add_argument("--reason", type=str, default=None)

    us = sub.add_parser("unseal", help="unseal maps (requires --force)")
    us.add_argument("ids", nargs="+")
    us.add_argument("--force", action="store_true")

    pv = sub.add_parser("pin-validation", help="pin the selection map")
    pv.add_argument("id")

    vf = sub.add_parser("verify", help="re-hash maps; check seals and pins")
    vf.add_argument("--json", action="store_true")

    inf = sub.add_parser("info", help="show one map's role/hashes/provenance")
    inf.add_argument("id")
    inf.add_argument("--json", action="store_true")

    ih = sub.add_parser("is-holdout", help="exit 0 if sealed holdout, 3 if not")
    ih.add_argument("id")

    fp = sub.add_parser("fingerprint", help="print pack fingerprint json")
    fp.add_argument("--json", action="store_true", default=True)

    args = parser.parse_args(argv)
    root = _root(args.maps_root)

    try:
        if args.cmd == "generate":
            res = ensure_maps(
                args.num,
                seed=args.seed,
                prefix=args.prefix,
                maps_root=root,
                start_index=args.start_index,
                role=args.role,
                seal=args.seal,
                seal_reason=args.reason,
                validation=args.validation,
                force=args.force,
                resolution=args.resolution,
                scale=args.scale,
                write_pgm=args.pgm,
            )
            print(f"created: {res['created'] or '-'}")
            print(f"cached : {res['cached'] or '-'}")
            _print_rows(list_maps(root))
            missing = res["requested"] - len(res["created"]) - len(res["cached"])
            if missing > 0:
                print(f"WARNING: {missing} map(s) could not be generated")
                return 1
            return 0

        if args.cmd == "list":
            rows = list_maps(root, role=args.role)
            if args.train_safe:
                rows = [r for r in rows if r["train_safe"]]
            if args.holdout:
                rows = [r for r in rows if r["sealed"]]
            if args.ids:
                print(",".join(r["id"] for r in rows))
            elif args.json:
                print(json.dumps(rows, indent=2))
            else:
                _print_rows(rows)
                print(f"\nvalidation pin: {validation_map(root) or '(none)'}")
            return 0

        if args.cmd == "adopt":
            adopt_disk_maps(root)
            _print_rows(list_maps(root))
            return 0

        if args.cmd == "seal":
            seal_maps(args.ids, maps_root=root, reason=args.reason)
            print(f"sealed: {', '.join(args.ids)}")
            _print_rows(list_maps(root))
            return 0

        if args.cmd == "unseal":
            unseal_maps(args.ids, maps_root=root, force=args.force)
            print(f"unsealed: {', '.join(args.ids)}")
            return 0

        if args.cmd == "pin-validation":
            pin_validation_map(args.id, maps_root=root)
            print(f"validation pin: {args.id}")
            return 0

        if args.cmd == "verify":
            report = verify_pack(root)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                for r in report["maps"]:
                    flag = "ok  " if r["ok"] else "FAIL"
                    seal = " sealed" if r["sealed"] else ""
                    print(f"[{flag}] {r['id']}{seal} {r.get('content_hash')}")
                    for issue in r["issues"]:
                        print(f"         - {issue}")
                for issue in report["issues"]:
                    print(f"[warn] {issue}")
                print("PACK OK" if report["ok"] else "PACK NOT OK")
            return 0 if report["ok"] else 1

        if args.cmd == "info":
            info = map_info(args.id, root)
            if info is None:
                print(f"unknown map '{args.id}'")
                return 2
            print(json.dumps(info, indent=2))
            return 0

        if args.cmd == "is-holdout":
            held = is_holdout(args.id, root)
            print(f"{args.id}: {'HOLDOUT (sealed)' if held else 'not sealed'}")
            return 0 if held else 3

        if args.cmd == "fingerprint":
            print(json.dumps(pack_fingerprint(root), indent=2))
            return 0
    except (HoldoutViolation, SealBroken, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 2

    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
