r"""Shared filesystem paths for HCMAI.

The project is used from both Windows (`D:\HCMAI`) and WSL/Linux
(`/mnt/d/HCMAI`). Keep path resolution in one place so pipelines do not need
their own ad-hoc Windows-path converters.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping, MutableMapping


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
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def dotenv_path(path: str | os.PathLike | None = None) -> Path:
    """Resolve an explicit dotenv path without importing/evaluating it."""
    chosen = path or os.environ.get("HCMAI_DOTENV") or ENV_PATH
    return Path(resolve_path(chosen))


def load_env(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Parse the shared dotenv file without evaluating shell syntax.

    Only ``KEY=VALUE`` and optional ``export KEY=VALUE`` records are accepted.
    Invalid lines are ignored rather than executed. Missing files are normal.
    """
    env_path = dotenv_path(path)
    out: dict[str, str] = {}
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            if ENV_NAME_RE.fullmatch(key):
                out[key] = _strip_optional_quotes(value)
    except FileNotFoundError:
        pass
    return out


def load_runtime_env(
    path: str | os.PathLike | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return shared dotenv values with real process environment overriding.

    This is the configuration precedence contract for all Python entrypoints:
    repository defaults live in code, `.env` holds local deploy/secrets, and
    the operating-system environment is the final explicit override.
    """
    values = load_env(path)
    source = os.environ if environ is None else environ
    values.update({str(key): str(value) for key, value in source.items()})
    return values


def activate_runtime_env(
    path: str | os.PathLike | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Populate missing process values from dotenv; never overwrite an export."""
    target = os.environ if environ is None else environ
    for key, value in load_env(path).items():
        target.setdefault(key, value)
    return load_runtime_env(path, target)
