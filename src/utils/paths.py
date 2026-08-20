r"""Shared filesystem paths for HCMAI.

The project is used from both Windows (`D:\HCMAI`) and WSL/Linux
(`/mnt/d/HCMAI`). Keep path resolution in one place so pipelines do not need
their own ad-hoc Windows-path converters.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_path(path: str | os.PathLike) -> str:
    r"""Convert a Windows `D:\...` path to the equivalent WSL/Linux path."""
    p = str(path)
    if os.name != "nt" and p.startswith("D:\\"):
        return p.replace("D:\\", "/mnt/d/").replace("\\", "/")
    return p


# Resolve from this module so the project remains portable after being moved.
# A legacy Windows path is still supported by resolve_path() for callers that
# explicitly provide one, but it must not define the active repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = REPO_ROOT / "results"
ENV_PATH = REPO_ROOT / ".env"


def load_env(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Load a simple KEY=VALUE .env file. Missing file returns {}."""
    env_path = Path(resolve_path(path or ENV_PATH))
    out: dict[str, str] = {}
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return out
