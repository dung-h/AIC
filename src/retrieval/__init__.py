"""Model-agnostic retrieval fusion primitives.

This package deliberately operates on ranked retrieval records rather than
raw embeddings.  Encoders and indexes may therefore use different dimensions
and implementations as long as they emit :class:`ModalityResult` records.
"""

from .contracts import (
    FusedCandidate,
    ModalityHit,
    ModalityResult,
    RetrievalCandidate,
)
from .fusion import (
    collapse_to_videos,
    fuse_channels,
    normalize_scores,
    reciprocal_rank_fusion,
    video_level_rrf,
)
from .candidates import (
    dedupe_candidates,
    expand_temporal_neighbors,
    select_top_k,
)
from .evidence import EvidenceHit, validate_evidence_hit
from .query_plan import QueryPlan, normalize_modality, normalize_task

__all__ = [
    "FusedCandidate",
    "ModalityHit",
    "ModalityResult",
    "RetrievalCandidate",
    "collapse_to_videos",
    "dedupe_candidates",
    "expand_temporal_neighbors",
    "fuse_channels",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "video_level_rrf",
    "select_top_k",
    "EvidenceHit",
    "validate_evidence_hit",
    "QueryPlan",
    "normalize_modality",
    "normalize_task",
]
