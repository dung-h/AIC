"""Read-only access to the canonical AIC2026 catalog."""

from .adapter import (
    CatalogAdapter,
    CatalogError,
    CatalogManifest,
    CanonicalFrame,
    EmbeddingShard,
    ManifestRecord,
    TextEvidence,
    UnknownFrameError,
    UnknownVideoError,
)

__all__ = [
    "CatalogAdapter",
    "CatalogError",
    "CatalogManifest",
    "CanonicalFrame",
    "EmbeddingShard",
    "ManifestRecord",
    "TextEvidence",
    "UnknownFrameError",
    "UnknownVideoError",
]
