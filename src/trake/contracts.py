"""Strict, model-agnostic contracts for TRAKE event alignment.

The competition output is intentionally smaller than the internal evidence
record: one video and exactly one canonical frame per event.  This module
keeps the internal event provenance while making it impossible for a partial
or unordered alignment to reach an output adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_MODALITIES = frozenset({"visual", "asr", "ocr", "audio"})


class TrakeContractError(ValueError):
    """Raised when an event, evidence row, or sequence violates TRAKE rules."""


def _clean_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TrakeContractError(f"{field_name} must be non-empty")
    return text


def _normalize_modalities(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        try:
            raw = list(value)
        except TypeError as exc:
            raise TrakeContractError("modalities must be a string or sequence") from exc
    result: list[str] = []
    for item in raw:
        modality = str(item).strip().lower()
        if not modality:
            continue
        if modality not in SUPPORTED_MODALITIES:
            raise TrakeContractError(
                f"unsupported TRAKE modality {modality!r}; "
                f"expected one of {sorted(SUPPORTED_MODALITIES)}"
            )
        if modality not in result:
            result.append(modality)
    return tuple(result)


@dataclass(frozen=True)
class TrakeEvent:
    """One ordered event in a TRAKE query."""

    index: int
    description: str
    modalities: tuple[str, ...] = ()
    event_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, Integral) or isinstance(self.index, bool) or self.index < 0:
            raise TrakeContractError("event index must be a non-negative integer")
        object.__setattr__(self, "description", _clean_text(self.description, "event description"))
        object.__setattr__(self, "modalities", _normalize_modalities(self.modalities))
        if self.event_id is not None:
            object.__setattr__(self, "event_id", _clean_text(self.event_id, "event_id"))


def _event_description(value: Mapping[str, Any]) -> str:
    for key in ("description", "desc", "caption", "event_desc", "text"):
        if key in value and str(value[key] or "").strip():
            return str(value[key])
    raise TrakeContractError("each TRAKE event needs description/desc/caption text")


def normalize_events(events: Sequence[Any] | Iterable[Any]) -> list[TrakeEvent]:
    """Normalize ordered event strings/mappings into a strict event list."""

    values = list(events)
    if not values:
        raise TrakeContractError("TRAKE events must be a non-empty sequence")
    normalized: list[TrakeEvent] = []
    for index, value in enumerate(values):
        if isinstance(value, TrakeEvent):
            if value.index != index:
                raise TrakeContractError(
                    f"event indices must be contiguous from zero; expected {index}, got {value.index}"
                )
            normalized.append(value)
            continue
        if isinstance(value, str):
            normalized.append(TrakeEvent(index=index, description=value))
            continue
        if not isinstance(value, Mapping):
            raise TrakeContractError(f"event {index} must be a string or mapping")
        explicit_index = value.get("event_index", value.get("index", index))
        if explicit_index != index:
            raise TrakeContractError(
                f"event indices must be contiguous from zero; expected {index}, got {explicit_index}"
            )
        modalities = value.get("modalities", value.get("required_modalities", value.get("modality")))
        normalized.append(
            TrakeEvent(
                index=index,
                description=_event_description(value),
                modalities=_normalize_modalities(modalities),
                event_id=value.get("event_id"),
            )
        )
    return normalized


def _int_field(value: Any, field_name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TrakeContractError(f"{field_name} must be an integer")
    value = int(value)
    if value < 0:
        raise TrakeContractError(f"{field_name} must be non-negative")
    return value


def _finite_time(value: Any, field_name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TrakeContractError(f"{field_name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise TrakeContractError(f"{field_name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class EventEvidence:
    """A single modality hit mapped to a canonical frame/timestamp."""

    event_index: int
    video_id: str
    modality: str
    score: float
    frame_idx: int
    pts_time: float
    kf_n: int | None = None
    source_id: str = ""
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_index, Integral) or isinstance(self.event_index, bool) or self.event_index < 0:
            raise TrakeContractError("evidence event_index must be a non-negative integer")
        object.__setattr__(self, "video_id", _clean_text(self.video_id, "evidence video_id"))
        modality = str(self.modality).strip().lower()
        if modality not in SUPPORTED_MODALITIES:
            raise TrakeContractError(f"unsupported evidence modality {modality!r}")
        object.__setattr__(self, "modality", modality)
        if not isinstance(self.score, Real) or isinstance(self.score, bool) or not math.isfinite(float(self.score)):
            raise TrakeContractError("evidence score must be finite numeric")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "frame_idx", _int_field(self.frame_idx, "evidence frame_idx"))
        object.__setattr__(self, "pts_time", _finite_time(self.pts_time, "evidence pts_time"))
        object.__setattr__(self, "kf_n", _int_field(self.kf_n, "evidence kf_n", allow_none=True))
        object.__setattr__(self, "source_id", str(self.source_id or "").strip())
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        event_index: int,
        modality: str | None = None,
    ) -> "EventEvidence":
        if not isinstance(value, Mapping):
            raise TrakeContractError("retriever evidence must be a mapping")
        explicit_event = value.get("event_index")
        if explicit_event is not None and int(explicit_event) != event_index:
            raise TrakeContractError(
                f"retriever returned evidence for event {explicit_event}, expected {event_index}"
            )
        resolved_modality = modality or value.get("modality")
        if not resolved_modality:
            raise TrakeContractError("retriever evidence must include modality")
        frame_idx = value.get("frame_idx", value.get("frame_id"))
        pts_time = value.get("pts_time", value.get("timestamp", value.get("start")))
        if frame_idx is None or pts_time is None:
            raise TrakeContractError("retriever evidence needs frame_idx and pts_time")
        return cls(
            event_index=event_index,
            video_id=value.get("video_id", value.get("vid")),
            modality=resolved_modality,
            score=value.get("score", value.get("similarity", 0.0)),
            frame_idx=frame_idx,
            pts_time=pts_time,
            kf_n=value.get("kf_n"),
            source_id=value.get("source_id", value.get("source", "")),
            text=value.get("text", value.get("chunk", "")),
            metadata=value.get("metadata", {}),
        )

    def key(self) -> tuple[str, int]:
        return self.video_id, self.frame_idx


def validate_sequence_path(
    path: Sequence[Mapping[str, Any] | EventEvidence],
    events: Sequence[TrakeEvent] | Sequence[Any] | int,
    *,
    video_id: str | None = None,
) -> list[EventEvidence]:
    """Validate exactly one canonical, strictly ordered frame per event."""

    normalized_events = normalize_events(events) if not isinstance(events, int) else None
    event_count = len(normalized_events) if normalized_events is not None else int(events)
    if event_count < 1:
        raise TrakeContractError("event_count must be positive")
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
        raise TrakeContractError("TRAKE sequence path must be a sequence")
    if len(path) != event_count:
        raise TrakeContractError(
            f"TRAKE path must contain exactly {event_count} steps, got {len(path)}"
        )
    result: list[EventEvidence] = []
    for index, item in enumerate(path):
        if isinstance(item, EventEvidence):
            evidence = item
        elif isinstance(item, Mapping):
            evidence = EventEvidence.from_mapping(item, event_index=index)
        else:
            raise TrakeContractError(f"TRAKE path step {index} must be a mapping")
        if evidence.event_index != index:
            raise TrakeContractError(
                f"TRAKE path event order is invalid at step {index}: {evidence.event_index}"
            )
        if video_id is not None and evidence.video_id != str(video_id):
            raise TrakeContractError("all TRAKE path steps must belong to one video_id")
        if result:
            if evidence.video_id != result[-1].video_id:
                raise TrakeContractError("TRAKE path cannot switch video_id")
            if evidence.frame_idx <= result[-1].frame_idx:
                raise TrakeContractError("TRAKE frame_idx must be strictly increasing")
            if evidence.pts_time <= result[-1].pts_time:
                raise TrakeContractError("TRAKE pts_time must be strictly increasing")
        result.append(evidence)
    return result


def validate_ranked_sequences(
    results: Sequence[Mapping[str, Any]],
    events: Sequence[TrakeEvent] | Sequence[Any] | int,
    *,
    max_answers: int = 100,
) -> list[dict[str, Any]]:
    """Validate ranked TRAKE answers before submission serialization."""

    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TrakeContractError("TRAKE ranked results must be a sequence")
    if len(results) == 0:
        raise TrakeContractError("TRAKE backend returned no ranked answers")
    if len(results) > max_answers:
        raise TrakeContractError(f"TRAKE ranked results exceed {max_answers} answers")
    event_count = len(normalize_events(events)) if not isinstance(events, int) else int(events)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for rank, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise TrakeContractError(f"TRAKE result {rank} must be a mapping")
        video_id = _clean_text(result.get("video_id"), "TRAKE result video_id")
        path = result.get("path")
        if path is not None:
            evidence = validate_sequence_path(path, event_count, video_id=video_id)
            frame_ids = [item.frame_idx for item in evidence]
        else:
            raw_frames = result.get("frame_ids")
            if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
                raise TrakeContractError("TRAKE result needs path or frame_ids")
            frame_ids = [_int_field(item, "TRAKE frame_id") for item in raw_frames]
            if len(frame_ids) != event_count:
                raise TrakeContractError("TRAKE frame_ids count must equal event count")
            if any(left >= right for left, right in zip(frame_ids, frame_ids[1:])):
                raise TrakeContractError("TRAKE frame_ids must be strictly increasing")
        identity = (video_id, tuple(frame_ids))
        if identity in seen:
            raise TrakeContractError("duplicate TRAKE ranked answer")
        seen.add(identity)
        normalized.append({**dict(result), "video_id": video_id, "frame_ids": frame_ids})
    return normalized
