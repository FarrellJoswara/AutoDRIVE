"""Official checkpoint discovery and replay contracts (no pickle loading)."""
import json
import hashlib
from pathlib import Path

ACTION_KEYS = (
    "throttle_mode", "negative_throttle_mode", "steering_mode",
    "steering_action_scale", "straight_throttle_gain", "straight_throttle_steering_threshold",
)


def checkpoint_settings(checkpoint: Path) -> dict:
    directory = checkpoint.parent
    if directory.name in {"ckpt", "checkpoints"}:
        directory = directory.parent
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Checkpoint needs its official training config.json to preserve observation/action semantics") from exc
    if not isinstance(config, dict) or not str(config.get("runtime", "")).startswith("official"):
        raise ValueError("Only checkpoints trained with the official runtime are supported here")
    actions = config.get("actions", {})
    if not isinstance(actions, dict) or any(key not in actions for key in ACTION_KEYS):
        raise ValueError("Checkpoint is missing its action mapping metadata")
    profile = config.get("observation_profile")
    if profile not in {"official_sensors", "official_sensors_history", "official_sensors_camera"}:
        raise ValueError("Checkpoint is missing a supported official observation profile")
    backend = config.get("simulator_backend", "docker")
    if backend not in {"docker", "windows_external", "macos_external"}:
        raise ValueError("Checkpoint has an unsupported simulator backend")
    return {"simulator_backend": backend, "observation_profile": profile,
            **{key: actions[key] for key in ACTION_KEYS}}


def list_official_models(root: Path) -> list[dict]:
    models = []
    if not root.is_dir():
        return models
    root = root.resolve()
    for run in root.iterdir():
        if not run.is_dir() or run.is_symlink():
            continue
        for directory in (run, run / "checkpoints", run / "ckpt"):
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in directory.glob("*.zip"):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    checkpoint_settings(path)
                    models.append({"id": path.relative_to(root).as_posix(),
                                   "label": path.relative_to(root).as_posix(),
                                   "modified": path.stat().st_mtime, "runtime": "official"})
                except (ValueError, OSError):
                    continue
    return sorted(models, key=lambda item: item["modified"], reverse=True)[:200]


def official_model_metadata(root: Path, model_id: str) -> dict:
    """Return metadata and a content hash for one selected official checkpoint.

    The list endpoint intentionally avoids hashing every archive.  This helper
    resolves only the requested catalog ID, enforces the same shallow catalog
    layout and symlink rules as discovery, then hashes that archive's bytes.
    """
    root = root.resolve()
    if not isinstance(model_id, str) or not model_id or "\\" in model_id:
        raise ValueError("invalid official checkpoint ID")
    parts = model_id.split("/")
    if len(parts) not in {2, 3} or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid official checkpoint ID")
    if len(parts) == 3 and parts[1] not in {"checkpoints", "ckpt"}:
        raise ValueError("invalid official checkpoint ID")

    candidate = root.joinpath(*parts)
    if any((root.joinpath(*parts[:index])).is_symlink() for index in range(1, len(parts))):
        raise ValueError("official checkpoint path cannot traverse a symbolic link")
    if candidate.is_symlink() or not candidate.is_file() or candidate.suffix.lower() != ".zip":
        raise ValueError("official checkpoint not found")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("official checkpoint path escapes the model catalog") from exc

    settings = checkpoint_settings(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"id": "/".join(parts), "sha256": digest.hexdigest(), **settings}
