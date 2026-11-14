#!/usr/bin/env python3
"""Install and relocate the public Porto 1,195,886-step checkpoint bundle.

Usage: python scripts/install_porto_pc_checkpoint.py BUNDLE.zip DESTINATION

The installer verifies every bundled file against artifact_manifest.json, copies
the bundle's repo-relative `logs/rl` tree, rewrites only JSON metadata paths,
and updates effective-config hashes in lineage sidecars. Model ZIP bytes are
copied verbatim. Run the repository's recursive verifier after installation.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rewrite(value, root: Path):
    if isinstance(value, dict):
        return {k: rewrite(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(v, root) for v in value]
    if isinstance(value, str):
        if value.startswith("$BUNDLE/"):
            return str(root / value[len("$BUNDLE/"):])
        return value
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    archive, destination = map(Path, sys.argv[1:])
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="porto-checkpoint-") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                path = Path(item.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise SystemExit(f"unsafe archive path: {item.filename}")
            zf.extractall(stage)
        manifest_path = stage / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for rel, expected in manifest["files"].items():
            path = stage / rel
            if not path.is_file() or digest(path) != expected:
                raise SystemExit(f"bundle integrity failure: {rel}")
        tree = stage / "logs" / "rl"
        if not tree.is_dir():
            raise SystemExit("bundle has no logs/rl tree")
        target = destination / "logs" / "rl"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}")
        shutil.copytree(tree, target)
        # Turn portable path tokens into destination paths. First update configs,
        # then bind each corresponding config's new hash in its lineage record.
        for path in target.rglob("*.json"):
            data = json.loads(path.read_text())
            data = rewrite(data, destination)
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        for sidecar in target.rglob("*.lineage.json"):
            data = json.loads(sidecar.read_text())
            cfg = data.get("effective_config", {})
            rec = cfg.get("path") if isinstance(cfg, dict) else None
            cfg_path = Path(rec.get("host", "")) if isinstance(rec, dict) else None
            if cfg_path and cfg_path.is_file():
                cfg["sha256"] = digest(cfg_path)
            sidecar.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Installed to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
