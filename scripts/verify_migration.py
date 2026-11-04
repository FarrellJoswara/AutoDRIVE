"""Verify a migration overlay without importing any training dependencies."""
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "migration-manifest.json").read_text(encoding="utf-8"))
    failed = []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        path.relative_to(root.resolve())
        if not path.is_file():
            failed.append(name)
            continue
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                failed.append(name)
    if failed:
        raise SystemExit("Missing/changed migration files: " + ", ".join(failed))
    print(f"Verified {len(manifest['files'])} files; base commit {manifest['base_commit']}")
    for name in manifest["deleted_tracked_files"]:
        if (root / name).exists():
            raise SystemExit(f"Apply original tracked deletion manually: {name}")


if __name__ == "__main__":
    main()
