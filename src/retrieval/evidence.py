"""Immutable canonical evidence records for multimodal retrieval."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .query_plan import (
    _LEAKAGE_FIELDS,
    _contains_leakage,
    _finite,
    _freeze,
    _json,
    _plain,
    _text,
    normalize_modality,
    normalize_task,
)


def _canonical_video_id(value: Any) -> str:
    video_id = _text(value, "video_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", video_id):
        raise ValueError("video_id must be a canonical corpus identifier")
    return video_id


def _frame_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("frame_idx must be a canonical integer >= 0")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


def _optional_time(value: Any) -> float | None:
    if value is None:
        return None
    result = _finite(value, "pts_time")
    if result < 0:
        raise ValueError("pts_time must be >= 0")
    return result


def _normalize_span(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _text(value, "span")
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("span mapping must be non-empty")
        for key in value:
            if str(key).strip().lower() in _LEAKAGE_FIELDS:
                raise ValueError(f"answer leakage field is not allowed: {key}")
        return _freeze(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) != 2:
            raise ValueError("span sequence must contain [start, end]")
        start = _finite(value[0], "span.start")
        end = _finite(value[1], "span.end")
        if start < 0 or end < start:
            raise ValueError("span must satisfy 0 <= start <= end")
        return (start, end)
    raise ValueError("span must be text, a mapping, or [start, end]")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceHit:
    """One canonical frame-level hit emitted by a retrieval channel.

    ``frame_idx`` is deliberately mandatory and is the only submission-grade
    frame identity.  ``kf_n`` and ``pts_time`` are optional annotations, never
    substitutes for the canonical frame index.
    """

    task: str
    video_id: str
    frame_idx: int
    modality: str
    channel: str
    rank: int
    score: float
    model_id: str
    index_id: str
    kf_n: int | None = None
    pts_time: float | None = None
    text: str = ""
    span: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", normalize_task(self.task))
        object.__setattr__(self, "video_id", _canonical_video_id(self.video_id))
        object.__setattr__(self, "frame_idx", _frame_index(self.frame_idx))
        object.__setattr__(self, "modality", normalize_modality(self.modality))
        object.__setattr__(self, "channel", _text(self.channel, "channel").lower())
        if any(char.isspace() for char in self.channel) or any(ord(char) < 32 for char in self.channel):
            raise ValueError("channel must not contain whitespace/control characters")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be an integer >= 1")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "index_id", _text(self.index_id, "index_id"))
        object.__setattr__(self, "kf_n", _optional_nonnegative_int(self.kf_n, "kf_n"))
        object.__setattr__(self, "pts_time", _optional_time(self.pts_time))
        object.__setattr__(self, "text", _text(self.text, "text", required=False))
        object.__setattr__(self, "span", _normalize_span(self.span))
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        found = _contains_leakage(self.provenance)
        if found is not None:
            raise ValueError(f"answer leakage field is not allowed: {found}")
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def validate_for(self, plan: Any) -> "EvidenceHit":
        """Validate task compatibility with a QueryPlan and return itself."""

        if getattr(plan, "task", None) != self.task:
            raise ValueError(f"evidence task {self.task!r} does not match plan task {getattr(plan, 'task', None)!r}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "frame_idx": self.frame_idx,
            "index_id": self.index_id,
            "kf_n": self.kf_n,
            "modality": self.modality,
            "model_id": self.model_id,
            "provenance": _plain(self.provenance),
            "pts_time": self.pts_time,
            "rank": self.rank,
            "score": self.score,
            "span": _plain(self.span),
            "task": self.task,
            "text": self.text,
            "video_id": self.video_id,
        }

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceHit":
        if not isinstance(value, Mapping):
            raise ValueError("evidence hit must be a mapping")
        found = _contains_leakage(value)
        if found is not None:
            raise ValueError(f"answer leakage field is not allowed: {found}")
        allowed = {
            "task",
            "video_id",
            "frame_idx",
            "modality",
            "channel",
            "rank",
            "score",
            "model_id",
            "index_id",
            "kf_n",
            "pts_time",
            "text",
            "span",
            "provenance",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown evidence fields: {unknown}")
        return cls(**dict(value))


def validate_evidence_hit(hit: EvidenceHit, plan: Any) -> EvidenceHit:
    """Small functional adapter for retrieval lanes that prefer functions."""

    if not isinstance(hit, EvidenceHit):
        raise TypeError("hit must be an EvidenceHit")
    return hit.validate_for(plan)


__all__ = ["EvidenceHit", "validate_evidence_hit"]
