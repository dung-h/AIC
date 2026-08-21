"""Structured, answer-free hypothesis generation for VQA grounding.

The generator sits *before* retrieval.  It can make a hard factual/narrative
query searchable, but it never produces a submitted answer and its output
cannot directly alter local video ranks.  The only consumer is the external
grounding resolver, which must return source-backed text before that text is
allowed back into local ASR/OCR retrieval.

Keeping this boundary explicit prevents a capable LLM from turning an
unverified web fact into an apparently well-grounded video answer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
import time
from typing import Any, Protocol
import unicodedata

from .answer_provider import (
    AnswerProviderConfigurationError,
    AnswerProviderRequestError,
    RetryPolicy,
    _default_transport,
    _extract_chat_content,
    _extract_json_object,
    _run_with_retry,
)


class HypothesisGenerationError(RuntimeError):
    """A generator is unavailable or returned an unsafe structured result."""


def _text(value: object, field: str, *, limit: int = 360) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HypothesisGenerationError(f"{field} must be a non-empty string")
    return " ".join(value.split())[:limit]


def _items(value: object, field: str, *, limit: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HypothesisGenerationError(f"{field} must be a sequence")
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise HypothesisGenerationError(f"{field} must contain strings")
        normalized = " ".join(item.split())[:360]
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in seen:
            output.append(normalized)
            seen.add(key)
        if len(output) >= limit:
            break
    return tuple(output)


_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")


def _fold(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold()).replace("đ", "d")
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _source_numbers(request: "HypothesisRequest") -> set[str]:
    return {
        token.replace(",", ".")
        for token in _NUMBER.findall(f"{request.query} {request.question}")
    }


def _filter_ungrounded_hypotheses(
    data: Mapping[str, Any], request: "HypothesisRequest"
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Remove model-invented numeric/time constraints before search.

    The planner may paraphrase an entity, but it must never inject a date,
    amount, or episode number that the user did not supply.  Such additions
    are especially damaging to external search because they make a perfectly
    relevant source appear absent.  We drop an unsafe view rather than trying
    to edit it into a new model-generated fact.
    """

    normalized = dict(data)
    allowed_numbers = _source_numbers(request)
    dropped: list[str] = []
    queries = _items(normalized.get("retrieval_queries"), "retrieval_queries", limit=4)
    safe_queries: list[str] = []
    for value in queries:
        numbers = {token.replace(",", ".") for token in _NUMBER.findall(value)}
        if numbers.difference(allowed_numbers):
            dropped.append(value)
        else:
            safe_queries.append(value)
    if not safe_queries:
        raise HypothesisGenerationError(
            "hypothesis generator produced no retrieval query grounded in the input"
        )
    normalized["retrieval_queries"] = safe_queries

    source = _fold(f"{request.query} {request.question}")
    constraints = _items(
        normalized.get("temporal_constraints"), "temporal_constraints", limit=3
    )
    safe_constraints: list[str] = []
    for value in constraints:
        # A temporal constraint is a hard filter, so it must be a direct
        # normalized span of the user request, not an LLM interpretation.
        if _fold(value) in source:
            safe_constraints.append(value)
        else:
            dropped.append(value)
    normalized["temporal_constraints"] = safe_constraints
    return normalized, tuple(dropped)


_MODALITY_ALIASES = {
    "visual": "visual",
    "vision": "visual",
    "image": "visual",
    "frame": "visual",
    "video": "visual",
    "asr": "asr",
    "audio": "asr",
    "speech": "asr",
    "spoken": "asr",
    "transcript": "asr",
    "ocr": "ocr",
    "text": "ocr",
    "screen_text": "ocr",
    "screen text": "ocr",
    "on_screen_text": "ocr",
}


def _modalities(value: object) -> tuple[str, ...]:
    """Normalize the small, model-facing modality vocabulary once.

    A structured model commonly emits ``audio`` despite the prompt saying
    ``asr``.  That is a vocabulary difference, not a routing decision.  The
    normalization has no heuristic priority: an unknown phrase remains a
    fail-closed schema error.
    """

    output: list[str] = []
    for raw in _items(value, "expected_modalities", limit=3):
        # Models often return one composite label (``ASR/OCR`` or
        # ``audio and screen text``).  Split only explicit conjunctions, then
        # normalize each member; arbitrary prose still fails closed.
        members = re.split(r"\s*(?:/|,|&|\+|\band\b)\s*", raw, flags=re.IGNORECASE)
        for member in members:
            key = re.sub(r"[\s_-]+", "_", member.casefold()).strip("_")
            normalized = _MODALITY_ALIASES.get(key)
            if normalized is None:
                normalized = _MODALITY_ALIASES.get(member.casefold())
            if normalized is None:
                raise HypothesisGenerationError(
                    "expected_modalities must use visual/asr/ocr or a declared synonym"
                )
            if normalized not in output:
                output.append(normalized)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class HypothesisRequest:
    """Question context supplied to a hypothesis generator.

    There intentionally is no answer field.  ``query`` and ``question`` are
    both retained because a factual entity can occur in either half.
    """

    query: str
    question: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _text(self.query, "query", limit=1200))
        object.__setattr__(self, "question", _text(self.question, "question", limit=800))


@dataclass(frozen=True, slots=True)
class RetrievalHypothesisPlan:
    """Bounded, non-answer search hypotheses with auditable intent.

    The model may offer aliases or a compact fact/quotation retrieval query,
    but every field remains a retrieval hint.  It has no semantic authority
    until an allow-listed external source and then local timestamped evidence
    corroborate it.
    """

    intent: str
    answer_type: str
    retrieval_queries: tuple[str, ...]
    entities: tuple[str, ...] = ()
    expected_modalities: tuple[str, ...] = ()
    temporal_constraints: tuple[str, ...] = ()
    dropped_ungrounded_hypotheses: tuple[str, ...] = ()
    provider: str = "unknown"
    model_id: str = "unknown"
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _text(self.intent, "intent", limit=80).casefold())
        object.__setattr__(self, "answer_type", _text(self.answer_type, "answer_type", limit=80).casefold())
        queries = _items(self.retrieval_queries, "retrieval_queries", limit=4)
        if not queries:
            raise HypothesisGenerationError("retrieval_queries must contain at least one value")
        object.__setattr__(self, "retrieval_queries", queries)
        object.__setattr__(self, "entities", _items(self.entities, "entities", limit=6))
        modalities = _modalities(self.expected_modalities)
        object.__setattr__(self, "expected_modalities", modalities)
        object.__setattr__(self, "temporal_constraints", _items(
            self.temporal_constraints, "temporal_constraints", limit=3
        ))
        object.__setattr__(self, "dropped_ungrounded_hypotheses", _items(
            self.dropped_ungrounded_hypotheses,
            "dropped_ungrounded_hypotheses",
            limit=8,
        ))
        object.__setattr__(self, "provider", _text(self.provider, "provider", limit=80))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", limit=160))
        if self.latency_ms is not None:
            latency = float(self.latency_ms)
            if not math.isfinite(latency) or latency < 0:
                raise HypothesisGenerationError("latency_ms must be a non-negative finite number")
            object.__setattr__(self, "latency_ms", latency)

    def grounding_views(self) -> tuple[str, ...]:
        """Views usable by external search, in deterministic priority order."""

        return tuple(dict.fromkeys([*self.retrieval_queries, *self.entities]))[:4]

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "answer_type": self.answer_type,
            "retrieval_queries": list(self.retrieval_queries),
            "entities": list(self.entities),
            "expected_modalities": list(self.expected_modalities),
            "temporal_constraints": list(self.temporal_constraints),
            "dropped_ungrounded_hypotheses": list(self.dropped_ungrounded_hypotheses),
            "provider": self.provider,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
        }


class HypothesisGenerator(Protocol):
    enabled: bool

    def generate(self, request: HypothesisRequest) -> RetrievalHypothesisPlan: ...


class DisabledHypothesisGenerator:
    """Explicit default that cannot accidentally make a network request."""

    enabled = False

    def generate(self, request: HypothesisRequest) -> RetrievalHypothesisPlan:
        del request
        raise HypothesisGenerationError("hypothesis generation is disabled")


class CallableHypothesisGenerator:
    """Injection seam for local models and deterministic tests."""

    enabled = True

    def __init__(self, callback: Callable[[HypothesisRequest], RetrievalHypothesisPlan | Mapping[str, Any]]):
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callback = callback

    def generate(self, request: HypothesisRequest) -> RetrievalHypothesisPlan:
        result = self._callback(request)
        if isinstance(result, RetrievalHypothesisPlan):
            return result
        if isinstance(result, Mapping):
            return RetrievalHypothesisPlan(**dict(result))
        raise HypothesisGenerationError("hypothesis callback returned an invalid result")


class OpenAICompatibleHypothesisGenerator:
    """Remote, JSON-only hypothesis generator for an explicit online route."""

    enabled = True
    provider_name = "openai_compatible_hypothesis"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str,
        timeout_s: float = 20.0,
        retry_policy: RetryPolicy | None = None,
        transport: Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not re.match(
            r"^https?://[^/?#]+(?:/[^?#]*)?$", base_url.strip().rstrip("/"), flags=re.IGNORECASE
        ):
            raise AnswerProviderConfigurationError("base_url must be an http(s) URL")
        if not isinstance(model, str) or not model.strip():
            raise AnswerProviderConfigurationError("model must be non-empty")
        if not isinstance(api_key, str) or not api_key.strip():
            raise AnswerProviderConfigurationError("api_key is required for hypothesis generation")
        if not math.isfinite(float(timeout_s)) or not 0.1 <= float(timeout_s) <= 60.0:
            raise AnswerProviderConfigurationError("timeout_s must be in [0.1, 60]")
        self.base_url = base_url.strip().rstrip("/")
        self.model_id = model.strip()
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=2)
        self.transport = transport or _default_transport

    @property
    def endpoint(self) -> str:
        return self.base_url + "/chat/completions"

    @staticmethod
    def _prompt(request: HypothesisRequest) -> str:
        return "\n".join((
            "You are a retrieval planner for a closed video corpus.",
            "Do NOT answer the question. Do NOT state or infer its answer.",
            "Produce only compact hypotheses that can be sent to a search engine to find corroborating sources.",
            "Keep named entities/quoted phrases exact when present. Do not make up people, locations, dates, or quotes.",
            "Return ONLY JSON with exactly these fields:",
            '{"intent":"...","answer_type":"...","retrieval_queries":["..."],"entities":["..."],'
            '"expected_modalities":["visual|asr|ocr"],"temporal_constraints":["..."]}',
            "Use at most 4 retrieval_queries, 6 entities, 3 modalities, 3 temporal constraints.",
            f"Scene/query: {request.query}",
            f"Question: {request.question}",
        ))

    def generate(self, request: HypothesisRequest) -> RetrievalHypothesisPlan:
        if not isinstance(request, HypothesisRequest):
            raise HypothesisGenerationError("request must be HypothesisRequest")
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": self._prompt(request)}],
            "temperature": 0,
            "max_tokens": 320,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter()
        try:
            response = _run_with_retry(
                lambda: self.transport(self.endpoint, headers, payload, self.timeout_s),
                retry_policy=self.retry_policy,
                wrap_error=lambda exc: AnswerProviderRequestError(
                    f"hypothesis generation request failed ({type(exc).__name__})",
                    retryable=isinstance(exc, (TimeoutError, OSError)),
                ),
            )
            raw = _extract_chat_content(response)
            data = _extract_json_object(raw)
            allowed = {
                "intent", "answer_type", "retrieval_queries", "entities",
                "expected_modalities", "temporal_constraints",
            }
            if set(data) != allowed:
                raise HypothesisGenerationError("hypothesis response must contain exactly the declared fields")
            filtered, dropped = _filter_ungrounded_hypotheses(data, request)
            return RetrievalHypothesisPlan(
                **filtered,
                dropped_ungrounded_hypotheses=dropped,
                provider=self.provider_name,
                model_id=self.model_id,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except HypothesisGenerationError:
            raise
        except Exception as exc:
            # Do not include raw model output or request headers in the error.
            raise HypothesisGenerationError(
                f"hypothesis generation failed ({type(exc).__name__})"
            ) from None


__all__ = [
    "CallableHypothesisGenerator",
    "DisabledHypothesisGenerator",
    "HypothesisGenerationError",
    "HypothesisGenerator",
    "HypothesisRequest",
    "OpenAICompatibleHypothesisGenerator",
    "RetrievalHypothesisPlan",
]
