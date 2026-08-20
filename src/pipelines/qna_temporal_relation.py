"""Deterministic temporal-relation reasoning for Q&A evidence.

This module is intentionally standalone.  It does not alter the Q&A router or
submission adapter.  The caller supplies two event descriptions and evidence
records labelled ``event_index`` 0/1.  The resolver only emits a valid signal
when the two events have unambiguous, canonical evidence in one video.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


class TemporalRelationError(ValueError):
    """Base error for evidence that cannot safely support a relation."""


class MissingEvidenceError(TemporalRelationError):
    """Raised when one of the two events has no usable evidence."""


class AmbiguousEvidenceError(TemporalRelationError):
    """Raised when multiple evidence records are equally plausible."""


class CanonicalMappingError(TemporalRelationError):
    """Raised when evidence does not agree with the canonical frame map."""


@dataclass(frozen=True)
class TemporalEvidence:
    """A canonical frame supporting one event."""

    event_index: int
    video_id: str
    kf_n: int
    frame_idx: int
    pts_time: float
    score: float | None = None

    def key(self) -> tuple[str, int, int, float]:
        return (self.video_id, self.kf_n, self.frame_idx, self.pts_time)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_index": self.event_index,
            "video_id": self.video_id,
            "kf_n": self.kf_n,
            "frame_idx": self.frame_idx,
            "pts_time": self.pts_time,
        }
        if self.score is not None:
            result["score"] = self.score
        return result


@dataclass(frozen=True)
class TemporalRelationResult:
    """Contract-safe result returned after relation resolution."""

    event_a: str
    event_b: str
    relation: str
    video_id: str
    evidence_a: TemporalEvidence
    evidence_b: TemporalEvidence

    @property
    def answer(self) -> str:
        return f"{self.event_a} occurs {self.relation} {self.event_b}."

    def answer_signal(self) -> dict[str, Any]:
        """Return a non-empty, canonical signal suitable for an adapter.

        ``frame_ids`` follows semantic event order A then B.  The additional
        ``chronological_frame_ids`` field makes the actual temporal order
        explicit without silently changing event identity.
        """

        chronological = sorted(
            (self.evidence_a, self.evidence_b),
            key=lambda item: (item.pts_time, item.frame_idx, item.kf_n),
        )
        return {
            "status": "valid",
            "video_id": self.video_id,
            "answer": self.answer,
            "relation": self.relation,
            "frame_ids": [self.evidence_a.frame_idx, self.evidence_b.frame_idx],
            "chronological_frame_ids": [item.frame_idx for item in chronological],
            "evidence": [self.evidence_a.as_dict(), self.evidence_b.as_dict()],
        }


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return converted if isinstance(converted, Mapping) else None
    if hasattr(value, "_asdict"):
        converted = value._asdict()
        return converted if isinstance(converted, Mapping) else None
    return None


def _field(record: Any, *names: str) -> Any:
    mapping = _as_mapping(record)
    if mapping is not None:
        for name in names:
            if name in mapping:
                return mapping[name]
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return None


def _event_index(record: Any, event_a: str, event_b: str) -> int:
    value = _field(record, "event_index", "event_id", "event", "event_desc")
    if isinstance(value, bool):
        raise MissingEvidenceError("event label must identify event A or B")
    if isinstance(value, int) or (isinstance(value, str) and value.strip() in {"0", "1"}):
        index = int(value)
        if index in (0, 1):
            return index
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"a", "event_a", "event 0"}:
            return 0
        if normalized in {"b", "event_b", "event 1"}:
            return 1
        if normalized == event_a.strip().casefold():
            return 0
        if normalized == event_b.strip().casefold():
            return 1
    raise MissingEvidenceError(
        "each evidence candidate must identify event A/B via event_index, event_id, event, or event_desc"
    )


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalMappingError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise CanonicalMappingError(f"{field_name} must be finite")
    return result


def _finite_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalMappingError(f"{field_name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise CanonicalMappingError(f"{field_name} must be an integer")
    return result


def _canonical_rows(canonical_map: Any) -> list[dict[str, Any]]:
    """Normalize common map forms into rows without requiring pandas."""

    if canonical_map is None:
        return []
    if hasattr(canonical_map, "to_dict") and hasattr(canonical_map, "columns"):
        return [dict(row) for row in canonical_map.to_dict("records")]
    if isinstance(canonical_map, Mapping):
        rows: list[dict[str, Any]] = []
        for key, value in canonical_map.items():
            if isinstance(key, tuple) and len(key) == 2:
                video_id, kf_n = key
                if isinstance(value, Mapping):
                    row = dict(value)
                    row.setdefault("video_id", video_id)
                    row.setdefault("kf_n", kf_n)
                else:
                    row = {"video_id": video_id, "kf_n": kf_n, "frame_idx": value}
                rows.append(row)
            elif isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_key, tuple) and len(nested_key) == 2:
                        rows.append(dict(nested_value))
                    else:
                        row = dict(nested_value) if isinstance(nested_value, Mapping) else {"frame_idx": nested_value}
                        row.setdefault("video_id", key)
                        row.setdefault("kf_n", nested_key)
                        rows.append(row)
            else:
                raise CanonicalMappingError("unsupported canonical mapping shape")
        return rows
    if isinstance(canonical_map, Sequence) and not isinstance(canonical_map, (str, bytes)):
        rows = []
        for item in canonical_map:
            row = _as_mapping(item)
            if row is None:
                raise CanonicalMappingError("canonical map sequence must contain mappings or row objects")
            rows.append(dict(row))
        return rows
    raise CanonicalMappingError("unsupported canonical mapping type")


def _canonicalize(record: Any, index: int, canonical_map: Any) -> TemporalEvidence:
    video = _field(record, "video_id", "video")
    if video is None or not str(video).strip():
        raise CanonicalMappingError("evidence video_id is required")
    video_id = str(video)
    kf_raw = _field(record, "kf_n", "keyframe", "keyframe_idx")
    frame_raw = _field(record, "frame_idx", "frame_id")
    pts_raw = _field(record, "pts_time", "timestamp", "time")
    if pts_raw is None:
        raise CanonicalMappingError("evidence pts_time is required")
    pts_time = _finite_float(pts_raw, "pts_time")
    kf_n = _finite_int(kf_raw, "kf_n") if kf_raw is not None else None
    frame_idx = _finite_int(frame_raw, "frame_idx") if frame_raw is not None else None
    rows = _canonical_rows(canonical_map)
    if rows:
        matches = []
        for row in rows:
            if str(row.get("video_id")) != video_id:
                continue
            row_kf = row.get("kf_n")
            row_frame = row.get("frame_idx")
            if kf_n is not None and row_kf is not None and int(row_kf) != kf_n:
                continue
            if frame_idx is not None and row_frame is not None and int(row_frame) != frame_idx:
                continue
            matches.append(row)
        if len(matches) != 1:
            raise CanonicalMappingError(
                f"evidence does not resolve uniquely in canonical map: video_id={video_id!r}, kf_n={kf_n!r}, frame_idx={frame_idx!r}"
            )
        row = matches[0]
        if row.get("kf_n") is None or row.get("frame_idx") is None:
            raise CanonicalMappingError("canonical row must contain kf_n and frame_idx")
        kf_n = _finite_int(row["kf_n"], "canonical kf_n")
        frame_idx = _finite_int(row["frame_idx"], "canonical frame_idx")
        if row.get("pts_time") is not None:
            canonical_time = _finite_float(row["pts_time"], "canonical pts_time")
            if abs(canonical_time - pts_time) > 1e-3:
                raise CanonicalMappingError("evidence pts_time differs from canonical map")
    if kf_n is None or frame_idx is None:
        raise CanonicalMappingError("evidence must contain canonical kf_n and frame_idx")
    score_raw = _field(record, "score", "confidence", "relevance")
    score = None if score_raw is None else _finite_float(score_raw, "score")
    return TemporalEvidence(index, video_id, kf_n, frame_idx, pts_time, score)


def _select_event(candidates: list[TemporalEvidence], event_name: str) -> TemporalEvidence:
    if not candidates:
        raise MissingEvidenceError(f"missing evidence for {event_name}")
    unique = {candidate.key(): candidate for candidate in candidates}
    candidates = list(unique.values())
    if len(candidates) == 1:
        return candidates[0]
    scored = [candidate for candidate in candidates if candidate.score is not None]
    if len(scored) != len(candidates):
        raise AmbiguousEvidenceError(f"ambiguous evidence for {event_name}")
    best_score = max(candidate.score for candidate in scored)
    best = [candidate for candidate in scored if candidate.score == best_score]
    if len(best) != 1:
        raise AmbiguousEvidenceError(f"ambiguous evidence for {event_name}: tied scores")
    return best[0]


def resolve_temporal_relation(
    event_a: str,
    event_b: str,
    evidence_candidates: Iterable[Any],
    canonical_map: Any = None,
) -> TemporalRelationResult:
    """Resolve whether event A occurs before or after event B.

    Candidates must identify their event with ``event_index`` 0/1 (aliases
    ``event``, ``event_id`` and exact ``event_desc`` are accepted).  Every
    candidate is canonicalized before selection.  Multiple distinct candidates
    are accepted only when every one has a unique highest score.
    """

    if not isinstance(event_a, str) or not event_a.strip():
        raise MissingEvidenceError("event_a is required")
    if not isinstance(event_b, str) or not event_b.strip():
        raise MissingEvidenceError("event_b is required")
    if event_a.strip().casefold() == event_b.strip().casefold():
        raise AmbiguousEvidenceError("event descriptions must be distinct")
    grouped: dict[int, list[TemporalEvidence]] = {0: [], 1: []}
    for raw in evidence_candidates:
        index = _event_index(raw, event_a, event_b)
        grouped[index].append(_canonicalize(raw, index, canonical_map))
    evidence_a = _select_event(grouped[0], "event_a")
    evidence_b = _select_event(grouped[1], "event_b")
    if evidence_a.video_id != evidence_b.video_id:
        raise CanonicalMappingError("event evidence must belong to the same video")
    if evidence_a.pts_time == evidence_b.pts_time:
        raise AmbiguousEvidenceError("event evidence has identical timestamps")
    relation = "before" if evidence_a.pts_time < evidence_b.pts_time else "after"
    return TemporalRelationResult(event_a, event_b, relation, evidence_a.video_id, evidence_a, evidence_b)


__all__ = [
    "AmbiguousEvidenceError",
    "CanonicalMappingError",
    "MissingEvidenceError",
    "TemporalEvidence",
    "TemporalRelationError",
    "TemporalRelationResult",
    "resolve_temporal_relation",
]
