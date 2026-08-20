"""Deterministic, provenance-preserving listwise evidence reranking.

This module owns no model, index, or answer-generation lifecycle.  It accepts
already materialized evidence bundles and runs two injected scorers:

1. a cheap first-stage scorer over the full candidate list;
2. an expensive listwise scorer over the bounded shortlist.

The second scorer sees the shortlist as a list, so it can compare competing
evidence bundles.  Scores are intentionally opaque floats: this module does
not claim model accuracy or calibrate a score into a probability.

Every candidate must contain at least one canonical frame.  ``frame_idx`` is
the submission-facing identifier; ``kf_n`` is retained only as auxiliary
keyframe provenance.  Missing or malformed evidence fails closed before any
scorer is called.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any, Protocol, TypeAlias


class EvidenceContractError(ValueError):
    """The candidate bundle cannot be trusted as evidence."""


class MissingEvidenceError(EvidenceContractError):
    """A candidate is missing the minimum canonical evidence."""


class ScorerContractError(ValueError):
    """An injected scorer returned an invalid result."""


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceContractError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, field: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise EvidenceContractError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(number) or (non_negative and number < 0):
        raise EvidenceContractError(f"{field} must be a finite number")
    return number


def _non_negative_int(value: Any, field: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise EvidenceContractError(f"{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceContractError(f"{field} must be a non-negative integer") from exc
    try:
        exact = float(value) == float(number)
    except (TypeError, ValueError):
        exact = False
    if number < 0 or not exact:
        raise EvidenceContractError(f"{field} must be a non-negative integer")
    return number


def _row_value(row: Any, *names: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
        return default
    for name in names:
        try:
            return getattr(row, name)
        except AttributeError:
            continue
    return default


def _as_sequence(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise EvidenceContractError(f"{field} must be a sequence")
    return list(value)


@dataclass(frozen=True)
class CanonicalFrame:
    """One frame with canonical output identity and optional keyframe metadata."""

    video_id: str
    frame_idx: int
    kf_n: int | None = None
    pts_time: float | None = None
    frame_path: str | None = None
    role: str = "evidence"

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", _required_text(self.video_id, "frame.video_id"))
        object.__setattr__(self, "frame_idx", _non_negative_int(self.frame_idx, "frame.frame_idx"))
        object.__setattr__(self, "kf_n", _non_negative_int(self.kf_n, "frame.kf_n", optional=True))
        if self.pts_time is not None:
            object.__setattr__(self, "pts_time", _finite_number(
                self.pts_time, "frame.pts_time", non_negative=True))
        if self.frame_path is not None:
            object.__setattr__(self, "frame_path", _required_text(str(self.frame_path), "frame.frame_path"))
        object.__setattr__(self, "role", _required_text(self.role, "frame.role"))

    @classmethod
    def from_mapping(cls, row: Any, *, default_video_id: str | None = None) -> "CanonicalFrame":
        video_id = _row_value(row, "video_id", "vid", default=default_video_id)
        frame_idx = _row_value(row, "frame_idx", "frame_id", default=None)
        if frame_idx is None:
            raise MissingEvidenceError("frame evidence requires canonical frame_idx")
        if video_id is None:
            raise MissingEvidenceError("frame evidence requires video_id")
        return cls(
            video_id=str(video_id),
            frame_idx=frame_idx,
            kf_n=_row_value(row, "kf_n", "keyframe", default=None),
            pts_time=_row_value(row, "pts_time", "timestamp", "time", default=None),
            frame_path=_row_value(row, "frame_path", "path", default=None),
            role=str(_row_value(row, "role", default="evidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "role": self.role,
        }
        if self.kf_n is not None:
            result["kf_n"] = self.kf_n
        if self.pts_time is not None:
            result["pts_time"] = self.pts_time
        if self.frame_path is not None:
            result["frame_path"] = self.frame_path
        return result


@dataclass(frozen=True)
class TextEvidence:
    """Timestamped ASR/OCR text attached to one candidate video."""

    source: str
    text: str
    start_time: float | None = None
    end_time: float | None = None
    timestamp: float | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        source = _required_text(self.source, "text.source").casefold()
        if source not in {"asr", "ocr"}:
            raise EvidenceContractError("text.source must be 'asr' or 'ocr'")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "text", _required_text(self.text, "text.text"))
        for name in ("start_time", "end_time", "timestamp"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_number(
                    value, f"text.{name}", non_negative=True))
        if self.start_time is not None and self.end_time is not None and self.end_time < self.start_time:
            raise EvidenceContractError("text.end_time cannot precede text.start_time")

    @classmethod
    def from_mapping(cls, row: Any, *, source: str) -> "TextEvidence":
        text = _row_value(row, "text", "chunk", "transcript", "ocr_text", default="")
        if not isinstance(text, str) or not text.strip():
            raise MissingEvidenceError("text evidence requires non-empty text")
        metadata: list[tuple[str, str]] = []
        if isinstance(row, Mapping):
            for key in ("video_id", "kf_n", "frame_idx", "rank", "distance_s"):
                if key in row and row[key] is not None:
                    metadata.append((key, str(row[key])))
        return cls(
            source=source,
            text=text,
            start_time=_row_value(row, "start_time", "start", default=None),
            end_time=_row_value(row, "end_time", "end", default=None),
            timestamp=_row_value(row, "timestamp", "pts_time", "time", default=None),
            metadata=tuple(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source": self.source, "text": self.text}
        for name in ("start_time", "end_time", "timestamp"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


def _text_rows(raw: Any, *, source: str) -> tuple[TextEvidence, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (TextEvidence(source=source, text=raw),) if raw.strip() else ()
    rows = _as_sequence(raw, f"{source}_evidence")
    parsed: list[TextEvidence] = []
    for row in rows:
        if isinstance(row, TextEvidence):
            if row.source != source:
                raise EvidenceContractError("text evidence source mismatch")
            parsed.append(row)
        else:
            parsed.append(TextEvidence.from_mapping(row, source=source))
    return tuple(parsed)


@dataclass(frozen=True)
class EvidenceBundle:
    """Normalized candidate bundle consumed by the reranker and verifier."""

    candidate_id: str
    video_id: str
    frames: tuple[CanonicalFrame, ...]
    asr: tuple[TextEvidence, ...] = ()
    ocr: tuple[TextEvidence, ...] = ()
    sources: tuple[str, ...] = ()
    provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _required_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "video_id", _required_text(self.video_id, "video_id"))
        if not self.frames:
            raise MissingEvidenceError("evidence bundle requires at least one canonical frame")
        normalized_frames = tuple(self.frames)
        if any(not isinstance(frame, CanonicalFrame) for frame in normalized_frames):
            raise EvidenceContractError("frames must contain CanonicalFrame records")
        if any(frame.video_id != self.video_id for frame in normalized_frames):
            raise EvidenceContractError("all frames must belong to the candidate video")
        if len({(frame.video_id, frame.frame_idx) for frame in normalized_frames}) != len(normalized_frames):
            raise EvidenceContractError("evidence bundle cannot contain duplicate canonical frames")
        object.__setattr__(self, "frames", normalized_frames)
        for name in ("asr", "ocr"):
            rows = tuple(getattr(self, name))
            expected = name
            if any(not isinstance(row, TextEvidence) or row.source != expected for row in rows):
                raise EvidenceContractError(f"{name} must contain {expected} TextEvidence records")
            object.__setattr__(self, name, rows)
        normalized_sources = tuple(dict.fromkeys(
            str(source).strip().casefold() for source in self.sources if str(source).strip()))
        object.__setattr__(self, "sources", normalized_sources)
        normalized_provenance: list[Mapping[str, Any]] = []
        for item in self.provenance:
            if not isinstance(item, Mapping):
                raise EvidenceContractError("provenance entries must be mappings")
            normalized_provenance.append(dict(item))
        object.__setattr__(self, "provenance", tuple(normalized_provenance))

    @classmethod
    def from_mapping(cls, raw: Any) -> "EvidenceBundle":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise EvidenceContractError("candidate evidence must be a mapping or EvidenceBundle")
        video_id = raw.get("video_id")
        if video_id is None:
            raise MissingEvidenceError("candidate evidence requires video_id")
        video_id = str(video_id)
        raw_frames = raw.get("frames", raw.get("evidence_frames"))
        if raw_frames is None:
            raw_frames = [raw]
        frame_rows = _as_sequence(raw_frames, "frames")
        frames = tuple(CanonicalFrame.from_mapping(row, default_video_id=video_id) for row in frame_rows)
        if any(frame.video_id != video_id for frame in frames):
            raise EvidenceContractError("frame video_id differs from candidate video_id")
        asr_raw = raw.get("asr_chunks", raw.get("asr", raw.get("asr_text", ())))
        ocr_raw = raw.get("ocr_text", raw.get("ocr", ()))
        asr = _text_rows(asr_raw, source="asr")
        ocr = _text_rows(ocr_raw, source="ocr")
        candidate_id = raw.get("candidate_id")
        if candidate_id is None:
            candidate_id = f"{video_id}#frame_{frames[0].frame_idx}"
        supplied_sources = raw.get("sources", ()) or ()
        if isinstance(supplied_sources, str):
            supplied_sources = (supplied_sources,)
        sources = list(str(value) for value in supplied_sources if str(value).strip())
        if frames and "visual" not in {value.casefold() for value in sources}:
            sources.append("visual")
        if asr and "asr" not in {value.casefold() for value in sources}:
            sources.append("asr")
        if ocr and "ocr" not in {value.casefold() for value in sources}:
            sources.append("ocr")
        provenance_raw = raw.get("provenance", ()) or ()
        if isinstance(provenance_raw, Mapping):
            provenance_raw = (provenance_raw,)
        return cls(
            candidate_id=str(candidate_id),
            video_id=video_id,
            frames=frames,
            asr=asr,
            ocr=ocr,
            sources=tuple(sources),
            provenance=tuple(provenance_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "frames": [frame.to_dict() for frame in self.frames],
            "asr_chunks": [row.to_dict() for row in self.asr],
            "ocr_text": [row.to_dict() for row in self.ocr],
            "sources": list(self.sources),
            "provenance": [dict(item) for item in self.provenance],
        }


def coerce_evidence_bundle(raw: Any) -> EvidenceBundle:
    """Normalize a pipeline mapping while preserving supplied provenance."""
    return EvidenceBundle.from_mapping(raw)


ScoreValue: TypeAlias = float | Mapping[str, Any]


class ListwiseScorer(Protocol):
    def __call__(self, query: str, candidates: Sequence[EvidenceBundle]) -> Sequence[ScoreValue]: ...


def _score_value(value: Any, *, stage: str, index: int) -> float:
    if isinstance(value, Mapping):
        for key in ("score", "relevance", "value"):
            if key in value:
                value = value[key]
                break
        else:
            raise ScorerContractError(f"{stage} scorer result {index} has no score")
    if isinstance(value, bool):
        raise ScorerContractError(f"{stage} scorer result {index} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ScorerContractError(f"{stage} scorer result {index} must be numeric") from exc
    if not math.isfinite(score):
        raise ScorerContractError(f"{stage} scorer result {index} must be finite")
    return score


@dataclass(frozen=True)
class RankedEvidence:
    """One final candidate with both stage scores and intact provenance."""

    candidate: EvidenceBundle
    stage1_score: float
    stage2_score: float
    stage1_rank: int
    stage2_rank: int
    final_rank: int

    @property
    def final_score(self) -> float:
        return self.stage2_score

    def to_dict(self) -> dict[str, Any]:
        result = self.candidate.to_dict()
        result.update({
            "stage1_score": self.stage1_score,
            "stage2_score": self.stage2_score,
            "final_score": self.final_score,
            "stage1_rank": self.stage1_rank,
            "stage2_rank": self.stage2_rank,
            "final_rank": self.final_rank,
            "canonical_provenance": [frame.to_dict() for frame in self.candidate.frames],
        })
        return result


class ListwiseEvidenceReranker:
    """Run cheap candidate filtering followed by bounded listwise scoring."""

    def __init__(
        self,
        stage1_scorer: ListwiseScorer,
        stage2_scorer: ListwiseScorer,
        *,
        stage1_k: int = 24,
    ) -> None:
        if not callable(stage1_scorer) or not callable(stage2_scorer):
            raise TypeError("both stage scorers must be callable")
        if isinstance(stage1_k, bool) or not isinstance(stage1_k, int) or stage1_k < 1:
            raise ValueError("stage1_k must be a positive integer")
        self._stage1_scorer = stage1_scorer
        self._stage2_scorer = stage2_scorer
        self.stage1_k = stage1_k

    @staticmethod
    def _rank_indices(scores: Sequence[float], candidates: Sequence[EvidenceBundle]) -> list[int]:
        # Input position is the final deterministic tie-breaker.  It also
        # makes a scorer's listwise ordering reproducible when scores tie.
        return sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], candidates[index].candidate_id, index),
        )

    @staticmethod
    def _run_scorer(
        scorer: ListwiseScorer,
        query: str,
        candidates: Sequence[EvidenceBundle],
        *,
        stage: str,
    ) -> list[float]:
        try:
            raw_scores = scorer(query, candidates)
        except Exception as exc:  # noqa: BLE001 - scorer boundary must fail closed.
            raise ScorerContractError(f"{stage} scorer failed") from exc
        if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Sequence):
            raise ScorerContractError(f"{stage} scorer must return one score per candidate")
        if len(raw_scores) != len(candidates):
            raise ScorerContractError(f"{stage} scorer returned the wrong number of scores")
        return [_score_value(value, stage=stage, index=index)
                for index, value in enumerate(raw_scores)]

    def rerank(
        self,
        query: str,
        candidates: Iterable[EvidenceBundle | Mapping[str, Any]],
        *,
        top_k: int = 6,
    ) -> tuple[RankedEvidence, ...]:
        """Return a deterministic top-k list or fail closed on bad evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        normalized = tuple(coerce_evidence_bundle(item) for item in candidates)
        if not normalized:
            raise MissingEvidenceError("reranker requires at least one evidence bundle")
        candidate_ids = [candidate.candidate_id for candidate in normalized]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise EvidenceContractError("candidate_id values must be unique within a listwise batch")

        stage1_scores = self._run_scorer(
            self._stage1_scorer, query, normalized, stage="stage1")
        stage1_order = self._rank_indices(stage1_scores, normalized)
        shortlist_indices = stage1_order[:min(self.stage1_k, len(normalized))]
        shortlist = tuple(normalized[index] for index in shortlist_indices)
        stage2_scores_short = self._run_scorer(
            self._stage2_scorer, query, shortlist, stage="stage2")
        stage2_order_short = self._rank_indices(stage2_scores_short, shortlist)
        stage1_rank = {index: rank for rank, index in enumerate(stage1_order, 1)}
        stage2_rank = {shortlist_indices[index]: rank
                       for rank, index in enumerate(stage2_order_short, 1)}

        ranked: list[RankedEvidence] = []
        for final_rank, shortlist_position in enumerate(stage2_order_short[:top_k], 1):
            original_index = shortlist_indices[shortlist_position]
            ranked.append(RankedEvidence(
                candidate=normalized[original_index],
                stage1_score=stage1_scores[original_index],
                stage2_score=stage2_scores_short[shortlist_position],
                stage1_rank=stage1_rank[original_index],
                stage2_rank=stage2_rank[original_index],
                final_rank=final_rank,
            ))
        return tuple(ranked)


__all__ = [
    "CanonicalFrame",
    "EvidenceBundle",
    "EvidenceContractError",
    "ListwiseEvidenceReranker",
    "ListwiseScorer",
    "MissingEvidenceError",
    "RankedEvidence",
    "ScorerContractError",
    "TextEvidence",
    "coerce_evidence_bundle",
]
