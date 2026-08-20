"""Immutable query contract shared by Q&A and TRAKE retrieval.

The contract deliberately contains only routing inputs.  It does not know
about an encoder, ANN implementation, corpus, or model object, which makes it
safe to pass between retrieval lanes and straightforward to serialize in a
trace.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_TASK_ALIASES = {
    "qa": "qna",
    "q&a": "qna",
    "qna": "qna",
    "vqa": "qna",
    "trake": "trake",
}
_MODALITY_ALIASES = {
    "audio": "asr",
    "speech": "asr",
    "asr": "asr",
    "image": "visual",
    "vision": "visual",
    "visual": "visual",
    "ocr": "ocr",
    "text": "ocr",
}
_LEAKAGE_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "answer_text",
        "expected_answer",
        "reference_answer",
        "ground_truth",
        "ground_truth_answer",
        "gt_answer",
        "gold_answer",
        "correct_answer",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if required and not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def normalize_task(value: Any) -> str:
    task = _text(value, "task").lower()
    try:
        return _TASK_ALIASES[task]
    except KeyError as exc:
        raise ValueError(f"unsupported task: {value!r}") from exc


def normalize_modality(value: Any, name: str = "modality") -> str:
    modality = _text(value, name).lower()
    return _MODALITY_ALIASES.get(modality, modality)


def _validate_modality_list(values: Sequence[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of modalities")
    try:
        normalized = {normalize_modality(item, name) for item in values}
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of modalities") from exc
    return tuple(sorted(normalized))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= 1")
    if value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def _contains_leakage(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in _LEAKAGE_FIELDS:
                return str(key)
            found = _contains_leakage(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            found = _contains_leakage(child)
            if found is not None:
                return found
    return None


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-like metadata while retaining Mapping access."""

    if isinstance(value, Mapping):
        frozen = {
            str(key): _freeze(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(child) for child in value), key=repr))
    if isinstance(value, float):
        return _finite(value, "metadata value")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _json(mapping: Mapping[str, Any]) -> str:
    return json.dumps(
        _plain(mapping),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _reject_leakage(mapping: Mapping[str, Any]) -> None:
    found = _contains_leakage(mapping)
    if found is not None:
        raise ValueError(f"answer leakage field is not allowed: {found}")


@dataclass(frozen=True, slots=True, kw_only=True)
class QueryPlan:
    """Canonical routing request for either Q&A or TRAKE.

    ``required_modalities`` describes evidence the task needs; it is allowed
    to differ from ``available_modalities`` so a preflight can report missing
    coverage rather than silently changing the query semantics.
    """

    task: str
    request_id: str
    query: str
    question: str = ""
    events: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()
    available_modalities: tuple[str, ...] = ()
    channel_weights: Mapping[str, float] = field(default_factory=dict)
    top_k: int = 100
    budget: int = 100
    offline: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", normalize_task(self.task))
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        question = _text(self.question, "question", required=False)
        object.__setattr__(self, "question", question)

        if isinstance(self.events, (str, bytes)):
            raise ValueError("events must be a sequence of event descriptions")
        try:
            events = tuple(_text(event, "event") for event in self.events)
        except TypeError as exc:
            raise ValueError("events must be a sequence of event descriptions") from exc
        if self.task == "trake" and not events:
            raise ValueError("trake requires at least one event")
        if self.task == "qna" and not question:
            raise ValueError("qna requires a non-empty question")
        object.__setattr__(self, "events", events)

        required = _validate_modality_list(self.required_modalities, "required_modalities")
        available = _validate_modality_list(self.available_modalities, "available_modalities")
        object.__setattr__(self, "required_modalities", required)
        object.__setattr__(self, "available_modalities", available)

        if not isinstance(self.channel_weights, Mapping):
            raise ValueError("channel_weights must be a mapping")
        weights: dict[str, float] = {}
        for channel, weight in self.channel_weights.items():
            normalized_channel = normalize_modality(channel, "channel")
            if normalized_channel in weights:
                raise ValueError(f"duplicate normalized channel: {normalized_channel}")
            normalized_weight = _finite(weight, f"channel_weights[{channel!r}]")
            if normalized_weight < 0:
                raise ValueError("channel weights must be >= 0")
            weights[normalized_channel] = normalized_weight
        if weights and not any(weight > 0 for weight in weights.values()):
            raise ValueError("at least one channel weight must be positive")
        object.__setattr__(self, "channel_weights", MappingProxyType(dict(sorted(weights.items()))))

        object.__setattr__(self, "top_k", _positive_int(self.top_k, "top_k"))
        object.__setattr__(self, "budget", _positive_int(self.budget, "budget"))
        if not isinstance(self.offline, bool):
            raise ValueError("offline must be a boolean")
        _reject_leakage({"query": self.query, "question": self.question, "events": self.events})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, canonically ordered representation."""

        return {
            "available_modalities": list(self.available_modalities),
            "budget": self.budget,
            "channel_weights": _plain(self.channel_weights),
            "events": list(self.events),
            "offline": self.offline,
            "query": self.query,
            "question": self.question,
            "request_id": self.request_id,
            "required_modalities": list(self.required_modalities),
            "task": self.task,
            "top_k": self.top_k,
        }

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QueryPlan":
        if not isinstance(value, Mapping):
            raise ValueError("query plan must be a mapping")
        _reject_leakage(value)
        allowed = {
            "task",
            "request_id",
            "query",
            "question",
            "events",
            "required_modalities",
            "available_modalities",
            "channel_weights",
            "top_k",
            "budget",
            "offline",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown query plan fields: {unknown}")
        return cls(**dict(value))

    def accepts(self, evidence: Any) -> bool:
        """Return whether an EvidenceHit belongs to this task and request."""

        return getattr(evidence, "task", None) == self.task
