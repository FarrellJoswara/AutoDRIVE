"""Content-addressed provenance checks for official BC and PPO artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import marshal
import os
import re
import platform
import sys
import copy
import contextvars
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional


# The preserved official teacher run is the only legacy source accepted by the
# deterministic BC rebuild. Its older evaluation hash (b395...) belongs to a
# different, failed 4,000-transition run and is deliberately not accepted here.
OFFICIAL_BC_SOURCE = {
    "run_id": "1627565e7893",
    "checkpoint_sha256": "3e6707501d9d02ca48c6d2b11ca650df680e9da4d8048251ec7a00941f863e39",
    "dataset_sha256": "c1355f64aaf6af2e198ac239020582b717d61244d4298044145eadca7178bc36",
    "summary_sha256": "b093c1e07928a43c9549ad72d9fda3d554c0efc3e0b6bb2ff9179894ce15298e",
    "config_sha256": "69064503a785c8992ad3fd71446ed3ee3b7c8e603fe1df9b170644c51f99c39f",
    "teacher_report_sha256": "e74ee59d010d294b08963f0deb6f0ddc281f01a3dcc12b29b03a5b9d6bf1c4a0",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def resolve_checkpoint_path(value: str | Path) -> Path:
    """Resolve native Windows and WSL mount paths to the current host path."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("checkpoint path must not be empty")
    wsl_match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if os.name == "nt" and wsl_match:
        drive, rest = wsl_match.groups()
        raw = f"{drive.upper()}:/{rest}"
    elif os.name != "nt" and re.match(r"^[a-zA-Z]:[\\/]", raw):
        win = PureWindowsPath(raw)
        drive = win.drive[0].lower()
        raw = str(PurePosixPath("/mnt", drive, *win.parts[1:]))
    return Path(raw).expanduser().resolve()


def checkpoint_path_aliases(value: str | Path) -> dict[str, str]:
    """Return canonical host, Windows, and WSL spellings for provenance."""
    path = resolve_checkpoint_path(value)
    if os.name == "nt":
        windows = str(path)
        win = PureWindowsPath(path)
        wsl = str(PurePosixPath("/mnt", win.drive[0].lower(), *win.parts[1:]))
        host = windows
    else:
        host = str(path)
        posix = PurePosixPath(path)
        wsl_match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(posix))
        if wsl_match:
            drive, rest = wsl_match.groups()
            windows = str(PureWindowsPath(f"{drive.upper()}:/{rest}"))
            wsl = str(posix)
        else:
            windows = ""
            wsl = str(posix)
    return {"host": host, "windows": windows, "wsl": wsl}


class PPOVerificationContext:
    """Per-collection exact-hash and recursive-verification memo.

    A context is intentionally short lived. It trusts no timestamps across
    contexts, and ``seal`` re-reads every unique dependency before a caller
    accepts evidence gathered during the pass.
    """

    def __init__(self) -> None:
        self._hashes: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}
        self._json: dict[tuple[Path, str], Optional[dict[str, Any]]] = {}
        self._memo: dict[tuple[Any, ...], list[tuple[dict[Path, str], Any]]] = {}
        self._collectors: list[set[Path]] = []
        self.hash_reads = 0
        self.bytes_hashed = 0
        self.seal_reads = 0
        self.seal_bytes_hashed = 0
        self.memo_hits = 0
        self.memo_misses = 0
        self.stat_invalidations = 0
        self._sealed = False
        self._failed = False

    @staticmethod
    def _path(path: str | Path) -> Path:
        return Path(path).resolve()

    @staticmethod
    def _stat(path: Path) -> tuple[int, int, int, int, int]:
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    @staticmethod
    def _digest(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                count += len(block)
        return digest.hexdigest(), count

    def hash_file(self, path: str | Path) -> str:
        if self._sealed or self._failed:
            raise RuntimeError("verification context is sealed or failed")
        canonical = self._path(path)
        for collector in self._collectors:
            collector.add(canonical)
        try:
            stat_before = self._stat(canonical)
        except FileNotFoundError:
            if canonical in self._hashes:
                self._failed = True
            raise
        cached = self._hashes.get(canonical)
        if cached is not None and cached[0] == stat_before:
            return cached[1]
        if cached is not None:
            self.stat_invalidations += 1
            self._json = {key: value for key, value in self._json.items() if key[0] != canonical}
        digest, count = self._digest(canonical)
        stat_after = self._stat(canonical)
        if stat_before != stat_after:
            self._failed = True
            raise OSError(f"file changed while hashing: {canonical}")
        self._hashes[canonical] = (stat_after, digest)
        self.hash_reads += 1
        self.bytes_hashed += count
        return digest

    def read_json(self, path: str | Path) -> Optional[dict[str, Any]]:
        canonical = self._path(path)
        if not canonical.is_file():
            return None
        digest = self.hash_file(canonical)
        key = (canonical, digest)
        if key in self._json:
            return copy.deepcopy(self._json[key])
        try:
            data = canonical.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                self._failed = True
                raise OSError(f"JSON bytes changed during read: {canonical}")
            value = json.loads(data.decode("utf-8"))
            result = value if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError):
            result = None
        # Guard reads that race with a writer; seal provides the final pass-wide check.
        if self._stat(canonical) != self._hashes[canonical][0]:
            self._failed = True
            raise OSError(f"file changed while reading JSON: {canonical}")
        self._json[key] = copy.deepcopy(result)
        return result

    def memoized(self, key: tuple[Any, ...], callback: Any) -> Any:
        if self._sealed or self._failed:
            raise RuntimeError("verification context is sealed or failed")
        for closure, result in self._memo.get(key, []):
            try:
                if all(self.hash_file(path) == digest for path, digest in closure.items()):
                    self.memo_hits += 1
                    return copy.deepcopy(result)
            except (OSError, RuntimeError):
                self._failed = True
                return None
        self.memo_misses += 1
        dependencies: set[Path] = set()
        self._collectors.append(dependencies)
        try:
            result = callback()
            closure = {path: self.hash_file(path) for path in dependencies}
            rows = self._memo.setdefault(key, [])
            rows.append((closure, copy.deepcopy(result)))
            if len(rows) > 4:
                del rows[:-4]
            return result
        except (OSError, RuntimeError):
            return None
        finally:
            self._collectors.pop()

    def seal(self) -> bool:
        """Rehash all dependencies and reject any mid-pass content mutation."""
        if self._failed:
            return False
        for path, (_, expected) in list(self._hashes.items()):
            try:
                before = self._stat(path)
                actual, count = self._digest(path)
                after = self._stat(path)
            except OSError:
                self._failed = True
                return False
            self.seal_reads += 1
            self.seal_bytes_hashed += count
            if before != after or actual != expected:
                self._failed = True
                return False
        self._sealed = True
        return True

    @property
    def stats(self) -> dict[str, int]:
        return {
            "unique_files": len(self._hashes),
            "hash_reads": self.hash_reads,
            "bytes_hashed": self.bytes_hashed,
            "seal_reads": self.seal_reads,
            "seal_bytes_hashed": self.seal_bytes_hashed,
            "memo_hits": self.memo_hits,
            "memo_misses": self.memo_misses,
            "stat_invalidations": self.stat_invalidations,
        }


_ACTIVE_VERIFICATION_CONTEXT: contextvars.ContextVar[Optional[PPOVerificationContext]] = (
    contextvars.ContextVar("aicar_ppo_verification_context", default=None)
)


def sha256_file(path: str | Path) -> str:
    context = _ACTIVE_VERIFICATION_CONTEXT.get()
    if context is not None:
        return context.hash_file(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_ppo_lineage_sidecar(
    checkpoint: str | Path,
    *,
    parent_lineage: dict[str, Any],
    effective_config: str | Path,
    optimizer_updates: int,
    num_timesteps: int,
    training_context: dict[str, Any],
    run_id: str,
    kind: str = "official_ppo_training_checkpoint",
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a content-bound lineage record next to a frozen PPO archive."""
    path = resolve_checkpoint_path(checkpoint)
    config = resolve_checkpoint_path(effective_config)
    parent = parent_lineage.get("parent_checkpoint")
    ancestor = parent_lineage.get("bc_ancestor_checkpoint")
    if not path.is_file() or not config.is_file() or not isinstance(parent, dict) or not isinstance(ancestor, dict):
        raise ValueError("checkpoint, config, and authenticated parent lineage must exist")
    if int(optimizer_updates) < 1 or int(num_timesteps) < 0:
        raise ValueError("authenticated PPO lineage requires an optimizer update and non-negative steps")
    actor_change = _actor_parameter_change(
        _resolve_path_record(parent.get("path")), path
    )
    if actor_change is None or actor_change.get("actor_changed") is not True:
        raise ValueError("PPO checkpoint must contain an actor-parameter change from its parent")
    record: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "status": "frozen_after_optimizer_update",
        "post_optimizer_update": True,
        "run_id": str(run_id),
        "checkpoint_paths": checkpoint_path_aliases(path),
        "checkpoint_sha256": sha256_file(path),
        "parent_checkpoint": parent,
        "bc_ancestor_checkpoint": ancestor,
        "parent_origin": parent_lineage.get("verification"),
        "effective_config": {
            "path": checkpoint_path_aliases(config),
            "sha256": sha256_file(config),
        },
        "training_context": training_context,
        "num_timesteps": int(num_timesteps),
        "optimizer_updates": int(optimizer_updates),
        "actor_change": actor_change,
    }
    if extra:
        record.update(extra)
    sidecar = path.with_name(f"{path.stem}.lineage.json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def _actor_comparison_runtime_identity() -> str:
    """Identify Python/platform/serializer runtime that produced a comparison."""
    versions = {}
    for distribution in ("stable-baselines3", "torch"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = "missing"
    payload = {
        "python": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "packages": versions,
        "device": "cpu",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _actor_comparison_code_identity() -> str:
    """Fingerprint the loaded comparator bytecode, not a possibly newer file."""
    return hashlib.sha256(marshal.dumps(_actor_parameter_change_uncached.__code__)).hexdigest()


@lru_cache(maxsize=32)
def _actor_parameter_change_cached(
    parent_path: str,
    checkpoint_path: str,
    parent_sha256: str,
    checkpoint_sha256: str,
    comparison_code_sha256: str,
    runtime_identity: str,
) -> Optional[dict[str, Any]]:
    """Load/compare model tensors once per immutable archive/code/runtime key.

    Only the small JSON-safe comparison result is retained; archives, tensors,
    and model objects remain local to the cache miss.
    """
    del parent_sha256, checkpoint_sha256, comparison_code_sha256, runtime_identity
    result = _actor_parameter_change_uncached(Path(parent_path), Path(checkpoint_path))
    if not isinstance(result, dict):
        # Exceptions are not cached by functools.lru_cache; malformed or
        # transiently unreadable archives should be retried on the next call.
        raise ValueError("actor tensor comparison failed")
    return result


def _actor_parameter_change(parent: Optional[Path], checkpoint: Path) -> Optional[dict[str, Any]]:
    """Compare actor tensors, reusing only a content- and runtime-bound result.

    Archive bytes are hashed on every call, so the cache never substitutes for
    current checkpoint/parent/lineage/config verification performed by callers.
    """
    if parent is None or not parent.is_file() or not checkpoint.is_file():
        return None
    try:
        parent_hash = sha256_file(parent)
        checkpoint_hash = sha256_file(checkpoint)
        comparison_code_hash = _actor_comparison_code_identity()
        result = _actor_parameter_change_cached(
            str(parent.resolve()),
            str(checkpoint.resolve()),
            parent_hash,
            checkpoint_hash,
            comparison_code_hash,
            _actor_comparison_runtime_identity(),
        )
        if not isinstance(result, dict):
            return None
        copied = dict(result)
        for key, value in copied.items():
            if isinstance(value, list):
                copied[key] = list(value)
        return copied
    except Exception:
        return None


def _actor_parameter_change_uncached(parent: Path, checkpoint: Path) -> Optional[dict[str, Any]]:
    """Compare actor-side policy tensors in two SB3 archives without env creation."""
    try:
        import torch
        from stable_baselines3.common.save_util import load_from_zip_file

        _, parent_parameters, _ = load_from_zip_file(str(parent), device="cpu")
        _, child_parameters, _ = load_from_zip_file(str(checkpoint), device="cpu")
        parent_state = parent_parameters.get("policy") if isinstance(parent_parameters, dict) else None
        child_state = child_parameters.get("policy") if isinstance(child_parameters, dict) else None
        if not isinstance(parent_state, dict) or not isinstance(child_state, dict):
            return None
        keys = sorted(
            key for key, value in parent_state.items()
            if torch.is_tensor(value)
            and key in child_state
            and torch.is_tensor(child_state[key])
            and "value_net" not in key
        )
        if not keys:
            return None
        changed: list[str] = []
        max_delta = 0.0
        log_std_changed = False
        for key in keys:
            before = parent_state[key].detach().cpu()
            after = child_state[key].detach().cpu()
            if before.shape != after.shape:
                return None
            if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all()):
                return None
            delta = float(torch.max(torch.abs(after - before)).item())
            max_delta = max(max_delta, delta)
            if not torch.equal(before, after):
                changed.append(key)
                if key == "log_std":
                    log_std_changed = True
        actor_changed = any(key != "log_std" for key in changed)
        return {
            "compared_parameter_count": len(keys),
            "changed_parameter_keys": changed,
            "changed_parameter_count": len(changed),
            "log_std_changed": log_std_changed,
            "max_abs_delta": max_delta,
            "actor_changed": actor_changed,
        }
    except Exception:
        return None


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        context = _ACTIVE_VERIFICATION_CONTEXT.get()
        if context is not None:
            return context.read_json(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value.lower()) is not None


def _resolve_path_record(value: Any) -> Optional[Path]:
    """Resolve a path or alias object only when all declared aliases agree."""
    candidates = list(value.values()) if isinstance(value, dict) else [value]
    resolved = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        try:
            resolved.append(resolve_checkpoint_path(candidate))
        except (OSError, RuntimeError, ValueError):
            return None
    if not resolved or any(path != resolved[0] for path in resolved[1:]):
        return None
    return resolved[0]


def _path_record_matches(value: Any, path: Path) -> bool:
    resolved = _resolve_path_record(value)
    return resolved == path if resolved is not None else False


def _verify_rebuild_provenance(path: Path, summary: dict[str, Any]) -> bool:
    provenance = summary.get("provenance")
    if provenance is None:
        return True
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "rebuilt_from_preserved_demonstrations"
    ):
        return False
    if provenance.get("source_run_id") != OFFICIAL_BC_SOURCE["run_id"]:
        return False
    if not _path_record_matches(provenance.get("checkpoint_paths"), path):
        return False
    parents = provenance.get("parents")
    if not isinstance(parents, dict):
        return False
    expected = {
        "source_checkpoint": OFFICIAL_BC_SOURCE["checkpoint_sha256"],
        "dataset": OFFICIAL_BC_SOURCE["dataset_sha256"],
        "source_summary": OFFICIAL_BC_SOURCE["summary_sha256"],
        "source_config": OFFICIAL_BC_SOURCE["config_sha256"],
        "teacher_report": OFFICIAL_BC_SOURCE["teacher_report_sha256"],
    }
    for key, expected_hash in expected.items():
        record = parents.get(key)
        if not isinstance(record, dict):
            return False
        recorded_hash = str(record.get("sha256", "")).lower()
        if recorded_hash != expected_hash or not _valid_sha256(recorded_hash):
            return False
        parent_path = _resolve_path_record(record.get("path", ""))
        if parent_path is None:
            return False
        if not parent_path.is_file() or sha256_file(parent_path) != expected_hash:
            return False
    fitted = provenance.get("fitted_intermediate_checkpoint")
    if fitted is not None:
        if not isinstance(fitted, dict) or fitted.get("training_completed_before_reload_check_failed") is not True:
            return False
        fitted_hash = str(fitted.get("sha256", "")).lower()
        fitted_path = _resolve_path_record(fitted.get("path", ""))
        if fitted_path is None:
            return False
        if (
            not _valid_sha256(fitted_hash)
            or not fitted_path.is_file()
            or sha256_file(fitted_path) != fitted_hash
        ):
            return False
    artifact_config = provenance.get("artifact_config")
    if not isinstance(artifact_config, dict):
        return False
    artifact_config_hash = str(artifact_config.get("sha256", "")).lower()
    artifact_config_path = _resolve_path_record(artifact_config.get("path"))
    expected_config_path = path.parent / "config.json"
    if (
        artifact_config_path != expected_config_path.resolve()
        or not _valid_sha256(artifact_config_hash)
        or not expected_config_path.is_file()
        or sha256_file(expected_config_path) != artifact_config_hash
    ):
        return False
    config = _read_json(expected_config_path)
    if (
        not config
        or not str(config.get("runtime", "")).startswith("official")
        or config.get("observation_profile") != "official_sensors_camera"
        or not isinstance(config.get("actions"), dict)
        or not isinstance(config.get("behavior_cloning"), dict)
        or config["behavior_cloning"].get("status") != "ready"
        or config["behavior_cloning"].get("usable") is not True
        or config["behavior_cloning"].get("checkpoint_sha256") != sha256_file(path)
    ):
        return False
    if not bool(provenance.get("source_legacy_bytes_verified")):
        return False
    return True


def _verify_bc_checkpoint_impl(
    checkpoint: str | Path,
    *,
    expected_sha256: Optional[str] = None,
    allow_legacy: bool = False,
) -> Optional[dict[str, Any]]:
    """Verify a ready BC checkpoint, its summary, and any recorded parents.

    Historical summaries are not trusted as proof by themselves. A legacy
    checkpoint can only be accepted with a separately pinned SHA-256 and an
    explicit ``allow_legacy=True`` migration request.
    """
    try:
        path = resolve_checkpoint_path(checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not path.is_file():
        return None
    actual_hash = sha256_file(path)
    if expected_sha256 is not None:
        expected_hash = str(expected_sha256).lower()
        if not _valid_sha256(expected_hash) or actual_hash != expected_hash:
            return None
    summary_path = path.parent / "behavior_cloning_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        return None
    readiness = summary.get("readiness")
    if not (
        summary.get("status") == "ready"
        and summary.get("usable") is True
        and isinstance(readiness, dict)
        and readiness.get("ready") is True
        and _path_record_matches(summary.get("checkpoint"), path)
    ):
        return None
    recorded_hash = str(summary.get("checkpoint_sha256", "")).lower()
    if recorded_hash:
        if not _valid_sha256(recorded_hash) or actual_hash != recorded_hash:
            return None
        verification = "summary_hash_and_current_bytes"
    elif allow_legacy and expected_sha256 is not None:
        verification = "legacy_summary_plus_independent_expected_hash"
    else:
        return None
    if not _verify_rebuild_provenance(path, summary):
        return None
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": actual_hash,
        "verification": verification,
        "summary": summary,
    }


def _verify_ppo_warmstart_impl(checkpoint: str | Path) -> Optional[dict[str, Any]]:
    """Verify a derived PPO candidate and its exact, ready BC parent."""
    try:
        path = resolve_checkpoint_path(checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not path.is_file():
        return None
    metadata = _read_json(path.parent / "ppo_warmstart_provenance.json")
    if not metadata:
        # The existing preflight candidate predates adjacent provenance. Its
        # only accepted migration record is the exact checked-in readiness
        # manifest, which pins both candidate and original parent bytes.
        root = Path(__file__).resolve().parents[2]
        manifest = _read_json(root / "competition/iros2026/results/ppo_readiness_manifest_1627565e7893.json")
        if not manifest:
            return None
        recorded = manifest.get("preparation_checks", {}).get("separate_ppo_warmstart_candidate", {})
        source = manifest.get("source", {})
        try:
            candidate_path = resolve_checkpoint_path(root / str(recorded.get("path", "")))
            parent_path = resolve_checkpoint_path(root / str(source.get("checkpoint", "")))
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        expected_candidate_hash = str(recorded.get("sha256", "")).lower()
        expected_parent_hash = str(source.get("checkpoint_sha256", "")).lower()
        candidate_hash = sha256_file(path)
        verified_parent = verify_bc_checkpoint(
            parent_path,
            expected_sha256=expected_parent_hash,
            allow_legacy=True,
        )
        if (
            candidate_path != path
            or not _valid_sha256(expected_candidate_hash)
            or candidate_hash != expected_candidate_hash
            or not verified_parent
            or expected_parent_hash != OFFICIAL_BC_SOURCE["checkpoint_sha256"]
            or recorded.get("action_std_throttle_steering") != [0.01, 0.05]
            or float(recorded.get("deterministic_action_change_from_source", float("inf"))) > 1e-7
            or recorded.get("status") != "prepared; requires official evaluator diagnosis before PPO use"
        ):
            return None
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": candidate_hash,
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": expected_parent_hash,
            "verification": "pinned_readiness_manifest_and_legacy_parent_hash",
        }
    candidate_hash = sha256_file(path)
    expected_candidate_hash = str(metadata.get("checkpoint_sha256", "")).lower()
    if not _valid_sha256(expected_candidate_hash) or candidate_hash != expected_candidate_hash:
        return None
    if not _path_record_matches(metadata.get("checkpoint_paths"), path):
        return None
    parent_record = metadata.get("parent_checkpoint")
    if not isinstance(parent_record, dict):
        return None
    parent_hash = str(parent_record.get("sha256", "")).lower()
    if not _valid_sha256(parent_hash):
        return None
    parent = _resolve_path_record(parent_record.get("path", ""))
    if parent is None:
        return None
    verified_parent = verify_bc_checkpoint(parent, expected_sha256=parent_hash)
    if not verified_parent:
        return None
    dataset_record = metadata.get("parent_dataset")
    if not isinstance(dataset_record, dict):
        return None
    dataset_path = _resolve_path_record(dataset_record.get("path", ""))
    if dataset_path is None:
        return None
    dataset_hash = str(dataset_record.get("sha256", "")).lower()
    parent_dataset = (
        verified_parent.get("summary", {})
        .get("provenance", {})
        .get("parents", {})
        .get("dataset", {})
    )
    if (
        not _valid_sha256(dataset_hash)
        or not dataset_path.is_file()
        or sha256_file(dataset_path) != dataset_hash
        or not isinstance(parent_dataset, dict)
        or str(parent_dataset.get("sha256", "")).lower() != dataset_hash
        or not _path_record_matches(parent_dataset.get("path"), dataset_path)
    ):
        return None
    if metadata.get("action_std") != [0.01, 0.05]:
        return None
    if metadata.get("source_action_std") != [0.2, 0.2]:
        return None
    if metadata.get("parent_is_verified_rebuild") is not True:
        return None
    changed_keys = metadata.get("changed_policy_keys")
    deterministic_delta = metadata.get("deterministic_action_max_abs_delta")
    if changed_keys != ["log_std"] or not isinstance(deterministic_delta, (int, float)):
        return None
    if float(deterministic_delta) > 1e-7:
        return None
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": candidate_hash,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": parent_hash,
        "verification": "derived_candidate_and_verified_parent",
    }


def _verify_ppo_lineage_checkpoint_impl(
    checkpoint: str | Path, *, _seen: Optional[set[Path]] = None
) -> Optional[dict[str, Any]]:
    """Authenticate a frozen PPO checkpoint and its complete BC ancestry.

    The sibling lineage sidecar binds the current archive, effective run config,
    immediate parent and BC ancestor. Parent PPO archives are recursively
    verified; roots must be a pinned/rebuilt BC artifact or its exact warmstart.
    """
    try:
        path = resolve_checkpoint_path(checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not path.is_file():
        return None
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        return None
    seen.add(path)
    lineage = _read_json(path.with_name(f"{path.stem}.lineage.json"))
    if not isinstance(lineage, dict) or lineage.get("kind") not in {
        "official_ppo_training_checkpoint", "official_training_probe"
    }:
        return None
    actual_hash = sha256_file(path)
    try:
        updates = int(lineage.get("optimizer_updates", 0))
        timesteps = int(lineage.get("num_timesteps", -1))
    except (TypeError, ValueError):
        return None
    if (
        not _valid_sha256(lineage.get("checkpoint_sha256"))
        or lineage["checkpoint_sha256"] != actual_hash
        or not _path_record_matches(lineage.get("checkpoint_paths"), path)
        or lineage.get("status") != "frozen_after_optimizer_update"
        or lineage.get("post_optimizer_update") is not True
        or updates < 1
        or timesteps < 0
    ):
        return None
    config_record = lineage.get("effective_config")
    if not isinstance(config_record, dict):
        return None
    config_path = _resolve_path_record(config_record.get("path"))
    config_hash = str(config_record.get("sha256", "")).lower()
    if (
        config_path is None
        or not config_path.is_file()
        or not _valid_sha256(config_hash)
        or sha256_file(config_path) != config_hash
        or config_path.parent != path.parent.parent
    ):
        return None
    config = _read_json(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("ppo"), dict):
        return None
    context = lineage.get("training_context")
    if not isinstance(context, dict):
        return None
    packages = context.get("packages")
    code = context.get("code_sha256")
    if (
        not isinstance(packages, dict)
        or not packages
        or not isinstance(code, dict)
        or not code
        or any(not isinstance(name, str) or not name for name in packages)
        or any(not _valid_sha256(value) for value in code.values())
    ):
        return None
    parent_record = lineage.get("parent_checkpoint")
    ancestor_record = lineage.get("bc_ancestor_checkpoint")
    if not isinstance(parent_record, dict) or not isinstance(ancestor_record, dict):
        return None
    parent = _resolve_path_record(parent_record.get("path"))
    parent_hash = str(parent_record.get("sha256", "")).lower()
    if parent is None or not parent.is_file() or not _valid_sha256(parent_hash):
        return None
    if sha256_file(parent) != parent_hash:
        return None
    actor_record = lineage.get("actor_change")
    actual_actor_change = _actor_parameter_change(parent, path)
    if (
        not isinstance(actor_record, dict)
        or actual_actor_change is None
        or actual_actor_change.get("actor_changed") is not True
        or actor_record.get("actor_changed") is not True
        or actor_record.get("changed_parameter_keys") != actual_actor_change.get("changed_parameter_keys")
        or actor_record.get("changed_parameter_count") != actual_actor_change.get("changed_parameter_count")
        or actor_record.get("log_std_changed") != actual_actor_change.get("log_std_changed")
        or not math.isclose(
            float(actor_record.get("max_abs_delta", -1)),
            float(actual_actor_change.get("max_abs_delta", -2)),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        return None
    verified_parent: Optional[dict[str, Any]] = verify_bc_checkpoint(
        parent, expected_sha256=parent_hash
    )
    if not verified_parent:
        verified_parent = verify_bc_checkpoint(
            parent,
            expected_sha256=OFFICIAL_BC_SOURCE["checkpoint_sha256"],
            allow_legacy=True,
        )
    if not verified_parent:
        verified_parent = verify_ppo_warmstart(parent)
    if not verified_parent:
        verified_parent = verify_ppo_lineage_checkpoint(parent, _seen=seen)
    if not verified_parent:
        return None
    if "bc_ancestor_checkpoint_sha256" in verified_parent:
        expected_ancestor_path = Path(verified_parent["bc_ancestor_checkpoint"]).resolve()
        expected_ancestor_hash = str(
            verified_parent.get("bc_ancestor_checkpoint_sha256", "")
        ).lower()
    elif verified_parent.get("verification") == "derived_candidate_and_verified_parent":
        expected_ancestor_path = Path(verified_parent["parent_checkpoint"]).resolve()
        expected_ancestor_hash = str(verified_parent["parent_checkpoint_sha256"]).lower()
    else:
        expected_ancestor_path = parent
        expected_ancestor_hash = parent_hash
    ancestor_path = _resolve_path_record(ancestor_record.get("path"))
    ancestor_hash = str(ancestor_record.get("sha256", "")).lower()
    if (
        ancestor_path != expected_ancestor_path
        or ancestor_hash != expected_ancestor_hash
        or not _valid_sha256(ancestor_hash)
    ):
        return None
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": actual_hash,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": parent_hash,
        "bc_ancestor_checkpoint": str(ancestor_path),
        "bc_ancestor_checkpoint_sha256": ancestor_hash,
        "effective_config": str(config_path),
        "effective_config_sha256": config_hash,
        "optimizer_updates": updates,
        "num_timesteps": timesteps,
        "lineage": lineage,
        "verification": "ppo_checkpoint_and_authenticated_bc_ancestry",
    }


def _verify_training_probe_checkpoint_impl(checkpoint: str | Path) -> Optional[dict[str, Any]]:
    """Authenticate a frozen automatic-evaluation checkpoint from a PPO probe.

    Each snapshot has its own sibling ``<checkpoint-stem>.lineage.json`` so a
    queued evaluation remains independently verifiable after training moves on.
    The run config and each parent link are content-addressed. A parent chain
    may continue through PPO checkpoints, but must terminate at a pinned legacy
    BC artifact, a verified rebuilt BC artifact, or a verified lower-noise
    derivative of one of those artifacts.
    """
    try:
        path = resolve_checkpoint_path(checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not path.is_file():
        return None
    lineage_path = path.with_name(f"{path.stem}.lineage.json")
    lineage = _read_json(lineage_path)
    if not lineage or lineage.get("kind") != "official_training_probe":
        return None
    authenticated = verify_ppo_lineage_checkpoint(path)
    if not authenticated:
        return None
    actual_hash = sha256_file(path)
    try:
        optimizer_updates = int(lineage.get("optimizer_updates", 0))
        updates_at_trigger = int(lineage.get("optimizer_updates_at_trigger", -1))
    except (TypeError, ValueError):
        return None
    if (
        not _valid_sha256(lineage.get("checkpoint_sha256"))
        or lineage["checkpoint_sha256"] != actual_hash
        or not _path_record_matches(lineage.get("checkpoint_paths"), path)
        or lineage.get("status") != "frozen_after_optimizer_update"
        or optimizer_updates <= updates_at_trigger
    ):
        return None
    config_record = lineage.get("effective_config")
    if not isinstance(config_record, dict):
        return None
    config_path_record = config_record.get("path")
    config_hash = str(config_record.get("sha256", "")).lower()
    config_path = _resolve_path_record(config_path_record)
    if config_path is None:
        return None
    if (
        not _valid_sha256(config_hash)
        or not config_path.is_file()
        or sha256_file(config_path) != config_hash
        or config_path.parent != path.parent.parent
    ):
        return None
    config = _read_json(config_path)
    if not config or not isinstance(config.get("training_probe"), dict):
        return None
    evaluation = config.get("automatic_evaluation")
    if (
        config.get("automatic_evaluation_mode") != "training_probe"
        or not isinstance(evaluation, dict)
        or evaluation.get("enabled") is not True
        or evaluation.get("mode") != "training_probe"
        or evaluation.get("attempts_per_snapshot") != 1
    ):
        return None
    if config["training_probe"].get("enabled") is not True:
        return None
    probe = config["training_probe"]
    if (
        probe.get("ranking") is not False
        or float(probe.get("simulation_timeout_s", -1)) != 240.0
        or float(probe.get("wall_timeout_s", -1)) != 240.0
        or float(probe.get("frontier_stagnation_s", -1)) != 5.0
        or probe.get("lap_counts_by_environment") != [1, 3]
        or probe.get("stop_on_first_collision_including_warmup") is not True
    ):
        return None
    parent_record = lineage.get("parent_checkpoint")
    if not isinstance(parent_record, dict):
        return None
    parent = _resolve_path_record(parent_record.get("path", ""))
    if parent is None:
        return None
    parent_hash = str(parent_record.get("sha256", "")).lower()
    if not _valid_sha256(parent_hash) or not parent.is_file() or sha256_file(parent) != parent_hash:
        return None
    verified_parent = verify_bc_checkpoint(parent, expected_sha256=parent_hash)
    if not verified_parent:
        verified_parent = verify_bc_checkpoint(
            parent,
            expected_sha256=OFFICIAL_BC_SOURCE["checkpoint_sha256"],
            allow_legacy=True,
        )
    if not verified_parent:
        verified_parent = verify_ppo_warmstart(parent)
    if not verified_parent:
        verified_parent = verify_ppo_lineage_checkpoint(parent)
    if not verified_parent:
        return None
    ancestor_record = lineage.get("bc_ancestor_checkpoint")
    if not isinstance(ancestor_record, dict):
        return None
    ancestor_path = _resolve_path_record(ancestor_record.get("path"))
    ancestor_hash = str(ancestor_record.get("sha256", "")).lower()
    if "bc_ancestor_checkpoint_sha256" in verified_parent:
        expected_ancestor_path = Path(verified_parent["bc_ancestor_checkpoint"]).resolve()
        expected_ancestor_hash = str(verified_parent["bc_ancestor_checkpoint_sha256"]).lower()
    elif verified_parent.get("verification") == "derived_candidate_and_verified_parent":
        expected_ancestor_path = Path(verified_parent["parent_checkpoint"]).resolve()
        expected_ancestor_hash = str(verified_parent["parent_checkpoint_sha256"]).lower()
    else:
        expected_ancestor_path = parent
        expected_ancestor_hash = parent_hash
    if (
        ancestor_path != expected_ancestor_path
        or ancestor_hash != expected_ancestor_hash
        or not _valid_sha256(ancestor_hash)
    ):
        return None
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": actual_hash,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": parent_hash,
        "effective_config": str(config_path),
        "effective_config_sha256": config_hash,
        "lineage": lineage,
        "verification": "training_probe_snapshot_and_authenticated_parent",
    }


def _verification_call(
    kind: str,
    checkpoint: str | Path,
    verification_context: Optional[PPOVerificationContext],
    callback: Any,
    extra_key: tuple[Any, ...] = (),
) -> Optional[dict[str, Any]]:
    active = _ACTIVE_VERIFICATION_CONTEXT.get()
    context = verification_context or active
    owns_context = context is None
    if owns_context:
        context = PPOVerificationContext()
    assert context is not None
    try:
        path = resolve_checkpoint_path(checkpoint)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    token = _ACTIVE_VERIFICATION_CONTEXT.set(context)
    try:
        result = context.memoized((kind, path, *extra_key), callback)
        if owns_context and not context.seal():
            return None
        return result if isinstance(result, dict) else None
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        _ACTIVE_VERIFICATION_CONTEXT.reset(token)


def verify_bc_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: Optional[str] = None,
    allow_legacy: bool = False,
    verification_context: Optional[PPOVerificationContext] = None,
) -> Optional[dict[str, Any]]:
    """Verify BC bytes and provenance, optionally sharing a collection pass."""
    return _verification_call(
        "bc",
        checkpoint,
        verification_context,
        lambda: _verify_bc_checkpoint_impl(
            checkpoint, expected_sha256=expected_sha256, allow_legacy=allow_legacy
        ),
        (str(expected_sha256).lower() if expected_sha256 is not None else None, bool(allow_legacy)),
    )


def verify_ppo_warmstart(
    checkpoint: str | Path,
    *,
    verification_context: Optional[PPOVerificationContext] = None,
) -> Optional[dict[str, Any]]:
    """Verify warmstart and BC parent, sharing only per-context results."""
    return _verification_call(
        "warmstart", checkpoint, verification_context,
        lambda: _verify_ppo_warmstart_impl(checkpoint),
    )


def verify_ppo_lineage_checkpoint(
    checkpoint: str | Path,
    *,
    verification_context: Optional[PPOVerificationContext] = None,
    _seen: Optional[set[Path]] = None,
) -> Optional[dict[str, Any]]:
    """Verify a PPO lineage chain with optional per-collection hash sharing."""
    seen_key = tuple(sorted(str(path.resolve()) for path in (_seen or set())))
    return _verification_call(
        "ppo_lineage", checkpoint, verification_context,
        lambda: _verify_ppo_lineage_checkpoint_impl(checkpoint, _seen=_seen),
        (seen_key,),
    )


def verify_training_probe_checkpoint(
    checkpoint: str | Path,
    *,
    verification_context: Optional[PPOVerificationContext] = None,
) -> Optional[dict[str, Any]]:
    """Verify a training-probe snapshot with a shared context when supplied."""
    return _verification_call(
        "training_probe", checkpoint, verification_context,
        lambda: _verify_training_probe_checkpoint_impl(checkpoint),
    )
