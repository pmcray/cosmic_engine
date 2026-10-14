"""Capture everything needed to re-run a render bit-for-bit (modulo GPU
nondeterminism). Written into each artifact directory as
`provenance.json` so a future reviewer can answer 'what was this?' just
by reading the file."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git(*args: str, cwd: str | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def capture_provenance(repo_dir: str | os.PathLike | None = None) -> dict:
    repo_dir = str(repo_dir) if repo_dir else None
    return {
        "captured_at": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {
            "sha": _git("rev-parse", "HEAD", cwd=repo_dir),
            "short_sha": _git("rev-parse", "--short", "HEAD", cwd=repo_dir),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_dir),
            "dirty": _git("status", "--porcelain", cwd=repo_dir) not in (None, ""),
            "last_commit": _git("log", "-1", "--oneline", cwd=repo_dir),
        },
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "modules": {
            name: _module_version(name)
            for name in ("numpy", "taichi", "imageio", "av", "torch", "PIL")
        },
    }


def write_provenance(path: Path, repo_dir: str | os.PathLike | None = None) -> dict:
    data = capture_provenance(repo_dir)
    Path(path).write_text(json.dumps(data, indent=2))
    return data
