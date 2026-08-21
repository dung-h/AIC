"""Offline-safe OpenCLIP model and tokenizer resolution.

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
    """Return a tokenizer-complete local HF snapshot, if present.

    This deliberately does not contact Hugging Face.  It is useful for the
    tokenizer, whose only required artifacts are the tokenizer files.
    """

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


def cached_model_snapshot(model_name: str, *, cache_root: str | Path | None = None) -> Path | None:
    """Return a complete local OpenCLIP snapshot with config and weights.

    ``open_clip.create_model_and_transforms(name, pretrained="webli")`` uses
    the Hub resolver even when the matching snapshot is already cached.  The
    production retriever must instead load the snapshot via OpenCLIP's
    ``local-dir:`` schema so its model identity and offline behaviour are
    deterministic.
    """

    snapshot = cached_snapshot(model_name, cache_root=cache_root)
    if snapshot is None:
        return None
    required = ("open_clip_config.json", "open_clip_model.safetensors")
    if not all((snapshot / filename).is_file() for filename in required):
        return None
    return snapshot


def get_tokenizer(open_clip_module, model_name: str):
    """Build a tokenizer without a network request when the snapshot exists."""

    snapshot = cached_snapshot(model_name)
    if snapshot is not None:
        from open_clip.tokenizer import HFTokenizer

        return HFTokenizer(str(snapshot), context_length=64, clean="canonicalize")
    return open_clip_module.get_tokenizer(model_name)


def create_model_and_transforms_local(open_clip_module, model_name: str):
    """Load an OpenCLIP model only from its complete local snapshot.

    A missing snapshot is a configuration error.  Do not fall back to the Hub:
    silently downloading a new checkpoint makes offline runs non-reproducible
    and can change the model behind persisted embedding indexes.
    """

    snapshot = cached_model_snapshot(model_name)
    if snapshot is None:
        raise FileNotFoundError(
            "Local OpenCLIP snapshot is required for offline retrieval but is "
            f"missing or incomplete: {model_name}. Expected a Hugging Face cache "
            "snapshot containing open_clip_config.json and open_clip_model.safetensors."
        )
    return open_clip_module.create_model_and_transforms(f"local-dir:{snapshot}")


__all__ = [
    "cached_snapshot",
    "cached_model_snapshot",
    "create_model_and_transforms_local",
    "get_tokenizer",
]
