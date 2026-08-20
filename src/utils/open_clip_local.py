"""Offline-safe OpenCLIP tokenizer resolution.

The persisted SigLIP2 cache contains tokenizer files and weights, but not
necessarily the Hub ``config.json`` that OpenCLIP's built-in tokenizer lookup
expects. Resolve the tokenizer directly from the local snapshot first; only
fall back to OpenCLIP's normal lookup for explicitly non-cached research
models.
"""
from __future__ import annotations

import os
from pathlib import Path


def _hub_cache_root() -> Path:
    explicit = os.getenv("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    return home / "hub"


def cached_snapshot(model_name: str, *, cache_root: str | Path | None = None) -> Path | None:
    """Return a complete local HF snapshot for one OpenCLIP model, if present."""

    repo_id = model_name if "/" in model_name else f"timm/{model_name}"
    root = Path(cache_root).expanduser() if cache_root is not None else _hub_cache_root()
    repo_cache = root / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        path for path in snapshots.iterdir()
        if path.is_dir()
        and (path / "tokenizer.json").is_file()
        and (path / "tokenizer_config.json").is_file()
    )
    return candidates[0] if candidates else None


def get_tokenizer(open_clip_module, model_name: str):
    """Build a tokenizer without a network request when the snapshot exists."""

    snapshot = cached_snapshot(model_name)
    if snapshot is not None:
        from open_clip.tokenizer import HFTokenizer

        return HFTokenizer(str(snapshot), context_length=64, clean="canonicalize")
    return open_clip_module.get_tokenizer(model_name)


__all__ = ["cached_snapshot", "get_tokenizer"]
