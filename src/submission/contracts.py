"""Stable internal records and canonical frame identity helpers.

The records deliberately contain provider/model metadata, but the submission
adapters only emit the fields accepted by the current AIC output contract.
This keeps local and remote providers interchangeable without leaking their
transport response shapes into ranking or serialization code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable, Mapping


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    value = " ".join(value.split())
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _frame_id(value: Any, field_name: str = "frame_id") -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be an integer") from exc
    # Reject lossy conversions such as 1.5 -> 1 while allowing numpy integers.
    if isinstance(value, float) and value != result:
        raise TypeError(f"{field_name} must be an integer")
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


@dataclass(frozen=True)
class QnAAnswerRecord:
    """Provider-neutral ranked Q&A result.

    ``score`` and provider metadata are internal ranking diagnostics. They are
    intentionally excluded from the external answer shape.
    """

    video_id: str
    frame_id: int
    answer: str
    score: float = 0.0
    grounding_score: float | None = None
    answer_confidence: float | None = None
    abstain: bool = False
    provider: str | None = None
    model_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _clean_text(self.video_id, "video_id"))
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        object.__setattr__(self, "answer", _clean_text(self.answer, "answer"))
        if not isfinite(float(self.score)):
            raise ValueError("score must be finite")
        for name in ("grounding_score", "answer_confidence"):
            value = getattr(self, name)
            if value is not None and (not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        if not isinstance(self.abstain, bool):
            raise TypeError("abstain must be bool")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QnAAnswerRecord":
        if not isinstance(value, Mapping):
            raise TypeError("Q&A answer must be a mapping or QnAAnswerRecord")
        return cls(
            video_id=value.get("video_id"),
            frame_id=value.get("frame_id", value.get("frame_idx")),
            answer=value.get("answer"),
            score=value.get("score", 0.0),
            grounding_score=value.get("grounding_score"),
            answer_confidence=value.get("answer_confidence"),
            abstain=value.get("abstain", False),
            provider=value.get("provider"),
            model_id=value.get("model_id"),
            metadata=value.get("metadata", {}),
        )

    def external(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "frame_id": self.frame_id, "answer": self.answer}


@dataclass(frozen=True)
class TrakeAnswerRecord:
    """Provider-neutral ranked TRAKE result."""

    video_id: str
    frame_ids: tuple[int, ...]
    score: float = 0.0
    provider: str | None = None
    model_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _clean_text(self.video_id, "video_id"))
        if isinstance(self.frame_ids, (str, bytes)):
            raise TypeError("frame_ids must be a sequence of integers")
        frames = tuple(_frame_id(frame, "frame_ids item") for frame in self.frame_ids)
        object.__setattr__(self, "frame_ids", frames)
        if not frames:
            raise ValueError("frame_ids must not be empty")
        if any(left >= right for left, right in zip(frames, frames[1:])):
            raise ValueError("frame_ids must be strictly increasing")
        if not isfinite(float(self.score)):
            raise ValueError("score must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrakeAnswerRecord":
        if not isinstance(value, Mapping):
            raise TypeError("TRAKE answer must be a mapping or TrakeAnswerRecord")
        return cls(
            video_id=value.get("video_id"),
            frame_ids=value.get("frame_ids", ()),
            score=value.get("score", 0.0),
            provider=value.get("provider"),
            model_id=value.get("model_id"),
            metadata=value.get("metadata", {}),
        )

    def external(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "frame_ids": list(self.frame_ids)}


class CanonicalFrameIndex:
    """Small immutable lookup for canonical ``video_id -> frame_idx`` pairs."""

    def __init__(self, frames: Mapping[str, Iterable[int]]) -> None:
        self._frames = {
            _clean_text(video_id, "video_id"): frozenset(_frame_id(frame) for frame in values)
            for video_id, values in frames.items()
        }

    def contains(self, video_id: str, frame_id: int) -> bool:
        return str(video_id) in self._frames and int(frame_id) in self._frames[str(video_id)]

    def __contains__(self, item: tuple[str, int]) -> bool:
        video_id, frame_id = item
        return self.contains(video_id, frame_id)


def canonical_frame_index(frames: Mapping[str, Iterable[int]]) -> CanonicalFrameIndex:
    """Build a canonical frame lookup used by submission validation."""
    return CanonicalFrameIndex(frames)
