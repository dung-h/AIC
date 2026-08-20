"""Validation for ranked Q&A and TRAKE submission records."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .contracts import CanonicalFrameIndex, QnAAnswerRecord, TrakeAnswerRecord

MAX_ANSWERS = 100
_PLACEHOLDER_ANSWERS = {
    "", "evidence-only", "evidence only", "unavailable", "unknown", "n/a", "na", "null",
    "none", "cannot determine", "cannot be determined", "i don't know",
}


def _canonical_contains(canonical_frames: Any, video_id: str, frame_id: int) -> bool:
    if canonical_frames is None:
        raise ValueError("canonical frame index is required for submission validation")
    if isinstance(canonical_frames, CanonicalFrameIndex):
        return canonical_frames.contains(video_id, frame_id)
    contains = getattr(canonical_frames, "contains", None)
    if callable(contains):
        return bool(contains(video_id, frame_id))
    if callable(canonical_frames):
        return bool(canonical_frames(video_id, frame_id))
    if isinstance(canonical_frames, Mapping):
        return int(frame_id) in {int(value) for value in canonical_frames.get(video_id, ())}
    try:
        return (video_id, int(frame_id)) in canonical_frames
    except TypeError as exc:
        raise TypeError("canonical_frames must provide contains(video_id, frame_id)") from exc


def _record(value: Any, cls: type) -> Any:
    return value if isinstance(value, cls) else cls.from_mapping(value)


def _check_ranked_length(answers: Sequence[Any]) -> None:
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        raise TypeError("ranked answers must be a sequence")
    if not answers:
        raise ValueError("ranked answers must not be empty")
    if len(answers) > MAX_ANSWERS:
        raise ValueError("at most 100 ranked answers are allowed per query")


def validate_qna_answers(
    answers: Sequence[QnAAnswerRecord | Mapping[str, Any]],
    *,
    canonical_frames: Any,
) -> list[QnAAnswerRecord]:
    """Validate and normalize one ranked Q&A list."""
    _check_ranked_length(answers)
    normalized: list[QnAAnswerRecord] = []
    seen: set[tuple[str, int]] = set()
    for rank, raw in enumerate(answers, 1):
        record = _record(raw, QnAAnswerRecord)
        if record.abstain:
            raise ValueError(f"Q&A rank {rank} is marked abstain")
        if record.answer.strip().casefold() in _PLACEHOLDER_ANSWERS:
            raise ValueError(f"Q&A rank {rank} has an empty/placeholder answer")
        if not _canonical_contains(canonical_frames, record.video_id, record.frame_id):
            raise ValueError(f"Q&A rank {rank} has non-canonical frame: {record.video_id}/{record.frame_id}")
        key = (record.video_id, record.frame_id)
        if key in seen:
            raise ValueError(f"Q&A rank {rank} duplicates an earlier answer")
        seen.add(key)
        normalized.append(record)
    return normalized


def validate_trake_answers(
    answers: Sequence[TrakeAnswerRecord | Mapping[str, Any]],
    *,
    event_count: int,
    canonical_frames: Any,
) -> list[TrakeAnswerRecord]:
    """Validate one ranked TRAKE list against its required event count."""
    _check_ranked_length(answers)
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count <= 0:
        raise ValueError("event_count must be a positive integer")
    normalized: list[TrakeAnswerRecord] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for rank, raw in enumerate(answers, 1):
        record = _record(raw, TrakeAnswerRecord)
        if len(record.frame_ids) != event_count:
            raise ValueError(
                f"TRAKE rank {rank} has {len(record.frame_ids)} frames; expected {event_count}"
            )
        for frame_id in record.frame_ids:
            if not _canonical_contains(canonical_frames, record.video_id, frame_id):
                raise ValueError(
                    f"TRAKE rank {rank} has non-canonical frame: {record.video_id}/{frame_id}"
                )
        key = (record.video_id, record.frame_ids)
        if key in seen:
            raise ValueError(f"TRAKE rank {rank} duplicates an earlier answer")
        seen.add(key)
        normalized.append(record)
    return normalized
