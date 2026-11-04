"""Export the current working tree plus explicitly selected ignored run folders.

No commits, remote writes, credentials, simulator binaries, or runtime registry
are copied. The archive overlays a fresh clone at the recorded base commit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".codex", ".codex_work", ".agents", ".aws", "__pycache__", "node_modules", "dist"}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def collect(root: Path, run_paths: list[str]) -> list[Path]:
    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode("utf-8").split("\0")
    selected = set()
    for name in filter(None, names):
        path = root / name
        relative = path.relative_to(root)
        if (SKIP_PARTS.intersection(relative.parts) or relative.parts[0] in {"logs", "outputs", "simulator_win"}
                or relative.name.startswith(".env") or relative.suffix in {".log", ".tsbuildinfo"}
                or (relative.parts[0] == "simulator"
                    and relative.parts[1:2] not in {("maps",), ("unity",), ("README.md",)})):
            continue
        if path.is_file() and not path.is_symlink():
            selected.add(path)
    for name in run_paths:
        run = (root / name).resolve()
        run.relative_to((root / "logs" / "rl").resolve())
        if not run.is_dir():
            raise ValueError(f"Run directory not found: {name}")
        for path in run.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.resolve().relative_to(root.resolve())
                selected.add(path)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-run", action="append", default=[], metavar="logs/rl/RUN", help="Repeat for each complete run folder, including BC ancestry/demonstrations")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "migration" / "aicar-migration.tar.gz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = collect(ROOT, args.include_run)
    output = args.output.resolve()
    if output in paths:
        raise ValueError("Output overlaps an included input")
    size = sum(p.stat().st_size for p in paths)
    print(f"Selected {len(paths)} files, {size / 1024**3:.2f} GiB before compression")
    if args.dry_run:
        for path in paths:
            print(path.relative_to(ROOT).as_posix())
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--short").decode("utf-8"),
        "deleted_tracked_files": list(filter(None, git("ls-files", "--deleted", "-z").decode("utf-8").split("\0"))),
        "included_runs": args.include_run,
        "files": {},
    }
    # Read each file once so the manifest hashes exactly the archived bytes.
    # Stop training before export to keep multi-file checkpoints consistent.
    with tarfile.open(output, "w:gz") as archive:
        for path in paths:
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            archive.addfile(info, io.BytesIO(data))
            manifest["files"][relative] = hashlib.sha256(data).hexdigest()
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("migration-manifest.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"Created {output}")


if __name__ == "__main__":
    main()
