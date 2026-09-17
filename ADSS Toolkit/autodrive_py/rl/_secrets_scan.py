"""Thin secrets hygiene scan for tracked rl/ files (observe-only helper).

Usage (from ADSS Toolkit/autodrive_py):
  python -m rl._secrets_scan

Excludes .venv/, __pycache__, *.zip, and large binary map assets.
Does not print matched secret values — only path + pattern name.
Exit 0 = CLEAN, 1 = findings.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

RL_ROOT = Path(__file__).resolve().parent
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".pt", ".pth", ".pkl", ".bin"}
SKIP_DIR_PARTS = {".venv", "__pycache__", ".git", "runs", "logs"}

# Named patterns — report name only, never the match text.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret_access_key", re.compile(r"aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*\S+", re.I)),
    ("pem_private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("github_pat", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github_fine_grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("openai_sk", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt_like", re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("assignment_password", re.compile(r"""(?:password|passwd)\s*[:=]\s*['"][^'"]{4,}['"]""", re.I)),
    ("assignment_api_key", re.compile(r"""(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['"][^'"]{8,}['"]""", re.I)),
    ("db_url_creds", re.compile(r"(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|smtp)://[^\s/@]+:[^\s/@]+@", re.I)),
]


def _tracked_under_rl() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "ADSS Toolkit/autodrive_py/rl"],
            cwd=RL_ROOT.parents[3],  # repo root (…/AutoDRIVE)
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Fallback: walk local tree, skip junk dirs.
        paths: list[Path] = []
        for p in RL_ROOT.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIR_PARTS for part in p.parts):
                continue
            if p.suffix.lower() in SKIP_SUFFIXES:
                continue
            paths.append(p)
        return paths

    repo = RL_ROOT.parents[3]
    paths = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        p = repo / line
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIR_PARTS for part in p.parts):
            continue
        if p.is_file():
            paths.append(p)
    return paths


def scan() -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for path in _tracked_under_rl():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pat in PATTERNS:
            if pat.search(text):
                try:
                    rel = path.relative_to(RL_ROOT).as_posix()
                except ValueError:
                    rel = str(path)
                findings.append((rel, name))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        print("CLEAN - no secret-pattern hits in tracked rl/ text files.")
        return 0
    print(f"FINDINGS - {len(findings)} hit(s) (path + pattern only; values omitted):")
    for rel, name in findings:
        print(f"  {rel}  [{name}]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
