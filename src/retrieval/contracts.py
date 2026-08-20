"""Stable records exchanged between retrieval layers.

The records contain no model, vector-index, database, or framework objects.
This is intentional: a visual encoder and an ASR encoder can have unrelated
dimensions and still be fused after each has produced ranked hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


def _finite(value: float, field_name: str) -> float:
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass(frozen=True)
class ModalityHit:
    """One result emitted by one modality channel.

    ``item_id`` is the channel's stable row/keyframe identifier.  It need not
    be a database integer.  ``video_id`` is kept separately because video
    collapse is a first-class operation in the competition pipeline.
    """

    item_id: str
    video_id: str
    score: float
    rank: Optional[int] = None
    kf_n: Optional[int] = None
    frame_idx: Optional[int] = None
    pts_time: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.item_id):
            raise ValueError("item_id must be non-empty")
        if not str(self.video_id):
            raise ValueError("video_id must be non-empty")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        if self.rank is not None and int(self.rank) < 1:
            raise ValueError("rank must be >= 1")
        if self.kf_n is not None and int(self.kf_n) < 0:
            raise ValueError("kf_n must be >= 0")
        if self.frame_idx is not None and int(self.frame_idx) < 0:
            raise ValueError("frame_idx must be >= 0")
        if self.pts_time is not None:
            object.__setattr__(self, "pts_time", _finite(self.pts_time, "pts_time"))


@dataclass(frozen=True)
class ModalityResult:
    """A ranked channel response plus its encoder/index provenance."""

    modality: str
    hits: Tuple[ModalityHit, ...]
    encoder_id: str = "unknown"
    index_id: str = "unknown"
    embedding_dim: Optional[int] = None
    metric: str = "score"
    higher_is_better: bool = True
    # A modality may have multiple independent channels (for example the two
    # visual encoders used by KIS).  Keep the old ``modality`` field for
    # routing, but use ``channel_id`` as the fusion/provenance key.
    channel_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.modality):
            raise ValueError("modality must be non-empty")
        if self.channel_id is not None and not str(self.channel_id).strip():
            raise ValueError("channel_id must be non-empty when provided")
        if self.embedding_dim is not None and int(self.embedding_dim) <= 0:
            raise ValueError("embedding_dim must be positive when provided")

    @property
    def channel_key(self) -> str:
        """Stable key for weights and per-channel evidence."""
        return str(self.channel_id or self.modality)


@dataclass(frozen=True)
class RetrievalCandidate:
    """Canonical candidate used after fusion and during frame allocation."""

    item_id: str
    video_id: str
    score: float
    kf_n: Optional[int] = None
    frame_idx: Optional[int] = None
    pts_time: Optional[float] = None
    source_modalities: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.item_id) or not str(self.video_id):
            raise ValueError("candidate item_id and video_id must be non-empty")
        object.__setattr__(self, "score", _finite(self.score, "score"))


@dataclass(frozen=True)
class FusedCandidate(RetrievalCandidate):
    """Candidate with explainable per-channel fusion evidence."""

    rrf_score: float = 0.0
    normalized_score: float = 0.0
    channel_ranks: Mapping[str, int] = field(default_factory=dict)
    channel_scores: Mapping[str, float] = field(default_factory=dict)


def hit_to_candidate(
    hit: ModalityHit,
    *,
    score: Optional[float] = None,
    source_modalities: Tuple[str, ...] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> RetrievalCandidate:
    """Convert a channel hit without coupling to a catalog implementation."""

    merged = dict(hit.metadata)
    if metadata:
        merged.update(metadata)
    return RetrievalCandidate(
        item_id=hit.item_id,
        video_id=hit.video_id,
        score=hit.score if score is None else score,
        kf_n=hit.kf_n,
        frame_idx=hit.frame_idx,
        pts_time=hit.pts_time,
        source_modalities=source_modalities,
        metadata=merged,
    )
