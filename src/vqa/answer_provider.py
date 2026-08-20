"""Provider boundary for grounded video-Q&A answers.

This module deliberately owns no pipeline, index, or model lifecycle.  It
defines the small object passed from retrieval/reranking to an answer model
and two adapters that implement the same contract:

* :class:`QwenLocalAnswerProvider` wraps an injected local Qwen-compatible
  model (or lazily constructs the repository's ``LocalVLM``).
* :class:`OpenAICompatibleAnswerProvider` calls a Chat Completions compatible
  endpoint using only the Python standard library.

The public response never contains raw prompts, model output, request
headers, or credentials.  Invalid/empty model answers fail closed as an
explicit abstention; malformed numeric fields remain schema errors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
import base64
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Generic, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_T = TypeVar("_T")


class AnswerProviderError(RuntimeError):
    """Base error for the answer-provider boundary."""


class AnswerProviderConfigurationError(AnswerProviderError):
    """Provider configuration or dependency is invalid."""


class AnswerProviderInputError(AnswerProviderError):
    """The request/evidence bundle cannot be sent to an answer provider."""


class AnswerProviderSchemaError(AnswerProviderError, ValueError):
    """A provider response does not satisfy the structured answer schema."""


class AnswerProviderRequestError(AnswerProviderError):
    """A provider request failed without exposing transport secrets."""

    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None):
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code


_PLACEHOLDER_ANSWERS = frozenset(
    {
        "",
        "null",
        "none",
        "n/a",
        "na",
        "unknown",
        "cannot determine",
        "cannot be determined",
        "i don't know",
        "i do not know",
        "unavailable",
        "evidence-only",
        "evidence only",
        "không xác định",
        "không thể trả lời",
        "không đủ thông tin",
    }
)
_REFUSAL_PREFIXES = (
    "i cannot answer",
    "i can't answer",
    "cannot answer",
    "can't answer",
    "not enough evidence",
    "insufficient evidence",
    "không thể",
    "không đủ",
)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerProviderInputError(f"{name} must be a non-empty string")
    return value.strip()


def _bounded_score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AnswerProviderSchemaError(f"{name} must be a number in [0, 1]")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise AnswerProviderSchemaError(f"{name} must be a number in [0, 1]") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise AnswerProviderSchemaError(f"{name} must be a finite number in [0, 1]")
    return score


def _is_non_answer(value: str) -> bool:
    normalized = " ".join(value.strip().casefold().split())
    return normalized in _PLACEHOLDER_ANSWERS or normalized.startswith(_REFUSAL_PREFIXES)


@dataclass(frozen=True)
class FrameEvidence:
    """One canonical frame that may be shown to an answer provider."""

    frame_id: int
    frame_path: str | Path | None = None
    pts_time: float | None = None
    modality: str = "visual"

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise AnswerProviderInputError("frame_id must be a non-negative canonical integer")
        if self.frame_path is not None:
            _required_text(str(self.frame_path), "frame_path")
        if self.pts_time is not None:
            if isinstance(self.pts_time, bool) or not math.isfinite(float(self.pts_time)) or float(self.pts_time) < 0:
                raise AnswerProviderInputError("pts_time must be a non-negative finite number")
        _required_text(self.modality, "modality")


@dataclass(frozen=True)
class EvidenceBundle:
    """Evidence for one answer candidate.

    ``frame_id`` values are canonical frame indices, never keyframe ordinals.
    ASR/OCR text is optional and is kept separate so an adapter can serialize
    it without making the retrieval layer depend on a specific model vendor.
    """

    candidate_id: str
    video_id: str
    frames: Sequence[FrameEvidence]
    asr_text: str = ""
    ocr_text: str = ""

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _required_text(self.video_id, "video_id")
        if isinstance(self.frames, (str, bytes)) or not isinstance(self.frames, Sequence) or not self.frames:
            raise AnswerProviderInputError("evidence bundle requires at least one frame")
        normalized = tuple(self.frames)
        if any(not isinstance(frame, FrameEvidence) for frame in normalized):
            raise AnswerProviderInputError("frames must contain FrameEvidence records")
        if len(normalized) > MAX_EVIDENCE_FRAMES:
            raise AnswerProviderInputError(
                f"evidence bundle cannot contain more than {MAX_EVIDENCE_FRAMES} frames"
            )
        if len({frame.frame_id for frame in normalized}) != len(normalized):
            raise AnswerProviderInputError("evidence bundle cannot contain duplicate frame_id values")
        object.__setattr__(self, "frames", normalized)
        if not isinstance(self.asr_text, str) or not isinstance(self.ocr_text, str):
            raise AnswerProviderInputError("asr_text and ocr_text must be strings")


@dataclass(frozen=True)
class AnswerProviderRequest:
    """Stable request passed to local and remote answer providers."""

    query: str
    question: str
    evidence: EvidenceBundle
    max_new_tokens: int = 160

    def __post_init__(self) -> None:
        _required_text(self.query, "query")
        _required_text(self.question, "question")
        if not isinstance(self.evidence, EvidenceBundle):
            raise AnswerProviderInputError("evidence must be an EvidenceBundle")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or not 1 <= self.max_new_tokens <= 4096:
            raise AnswerProviderInputError("max_new_tokens must be an integer in [1, 4096]")


@dataclass(frozen=True)
class AnswerProviderResponse:
    """Validated, transport-safe answer result.

    Abstention is represented by ``answer=None`` and ``abstain=True``.  The
    optional ``reason`` is a stable diagnostic label, never raw provider text.
    """

    candidate_id: str
    answer: str | None
    grounding_score: float
    answer_confidence: float
    abstain: bool
    provider: str
    model_id: str
    reason: str | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _required_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model_id", _required_text(self.model_id, "model_id"))
        object.__setattr__(self, "grounding_score", _bounded_score(self.grounding_score, "grounding_score"))
        object.__setattr__(self, "answer_confidence", _bounded_score(self.answer_confidence, "answer_confidence"))
        if not isinstance(self.abstain, bool):
            raise AnswerProviderSchemaError("abstain must be bool")
        if self.answer is not None and not isinstance(self.answer, str):
            raise AnswerProviderSchemaError("answer must be a string or None")
        if self.reason is not None:
            _required_text(self.reason, "reason")
        if self.latency_ms is not None:
            if isinstance(self.latency_ms, bool) or not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0:
                raise AnswerProviderSchemaError("latency_ms must be a non-negative finite number")
        normalized = self.answer.strip() if isinstance(self.answer, str) else ""
        if self.abstain or _is_non_answer(normalized):
            object.__setattr__(self, "answer", None)
            object.__setattr__(self, "abstain", True)
            if self.reason is None:
                object.__setattr__(self, "reason", "provider_abstained")
        elif not normalized:
            object.__setattr__(self, "answer", None)
            object.__setattr__(self, "abstain", True)
            object.__setattr__(self, "reason", self.reason or "empty_answer")
        else:
            object.__setattr__(self, "answer", normalized)

    @classmethod
    def abstained(
        cls,
        *,
        candidate_id: str,
        provider: str,
        model_id: str,
        reason: str,
        latency_ms: float | None = None,
    ) -> "AnswerProviderResponse":
        return cls(
            candidate_id=candidate_id,
            answer=None,
            grounding_score=0.0,
            answer_confidence=0.0,
            abstain=True,
            provider=provider,
            model_id=model_id,
            reason=reason,
            latency_ms=latency_ms,
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        candidate_id: str,
        provider: str,
        model_id: str,
        latency_ms: float | None = None,
    ) -> "AnswerProviderResponse":
        if not isinstance(payload, Mapping):
            raise AnswerProviderSchemaError("provider response must be a mapping")
        if "answer" not in payload:
            raise AnswerProviderSchemaError("provider response requires an answer field")
        payload_candidate_id = payload.get("candidate_id")
        if payload_candidate_id is not None:
            if not isinstance(payload_candidate_id, str) or payload_candidate_id.strip() != candidate_id:
                raise AnswerProviderSchemaError("provider response candidate_id does not match request")
        return cls(
            candidate_id=candidate_id,
            answer=payload.get("answer"),
            grounding_score=payload.get("grounding_score", 0.0),
            answer_confidence=payload.get("answer_confidence", payload.get("confidence", 0.0)),
            abstain=payload.get("abstain", False),
            provider=provider,
            model_id=model_id,
            reason=payload.get("reason"),
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return only safe structured output; never includes raw model text."""
        return {
            "candidate_id": self.candidate_id,
            "answer": self.answer,
            "grounding_score": self.grounding_score,
            "answer_confidence": self.answer_confidence,
            "abstain": self.abstain,
            "provider": self.provider,
            "model_id": self.model_id,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


_EVIDENCE_SOURCE_ALIASES = {
    "visual": "visual",
    "vision": "visual",
    "frame": "visual",
    "frames": "visual",
    "asr": "asr",
    "audio": "asr",
    "speech": "asr",
    "ocr": "ocr",
    "text": "ocr",
}
_VALID_EVIDENCE_SOURCES = frozenset({"visual", "asr", "ocr"})
MAX_EVIDENCE_FRAMES = 12


def _normalize_evidence_source(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return _EVIDENCE_SOURCE_ALIASES.get(" ".join(value.casefold().split()))


def _bundle_evidence_sources(bundle: EvidenceBundle) -> tuple[str, ...]:
    """Return validated sources in stable visual → ASR → OCR order.

    A frame is visual evidence even when its optional modality label is
    omitted.  Unknown labels are rejected instead of being silently treated
    as trustworthy evidence.  Text is only a source when it is non-empty.
    """
    sources: list[str] = []
    if bundle.frames:
        sources.append("visual")
    for frame in bundle.frames:
        if _normalize_evidence_source(frame.modality) is None:
            raise AnswerProviderInputError(
                f"unsupported evidence source: {frame.modality!r}"
            )
    if bundle.asr_text.strip():
        sources.append("asr")
    if bundle.ocr_text.strip():
        sources.append("ocr")
    return tuple(dict.fromkeys(source for source in sources if source in _VALID_EVIDENCE_SOURCES))


@dataclass(frozen=True)
class AnswerVerification:
    """Provider-agnostic, fail-closed verification result.

    This is deliberately separate from :class:`AnswerProviderResponse`:
    provider scores describe what a model claims, while this record describes
    whether the response is contract-safe for the exact candidate evidence
    that was supplied.  A rejected result never carries an answer forward.
    """

    candidate_id: str
    video_id: str
    frame_ids: tuple[int, ...]
    evidence_sources: tuple[str, ...]
    answer: str | None
    grounding_score: float
    answer_confidence: float
    abstain: bool
    accepted: bool
    reason: str
    response_candidate_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _required_text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "video_id", _required_text(self.video_id, "video_id"))
        if isinstance(self.frame_ids, (str, bytes)):
            raise AnswerProviderSchemaError("frame_ids must be a sequence of canonical integers")
        try:
            normalized_frames = tuple(self.frame_ids)
        except TypeError as exc:
            raise AnswerProviderSchemaError("frame_ids must be a sequence of canonical integers") from exc
        for frame_id in normalized_frames:
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                raise AnswerProviderSchemaError("frame_ids must contain non-negative canonical integers")
        if len(set(normalized_frames)) != len(normalized_frames):
            raise AnswerProviderSchemaError("frame_ids must be unique")
        object.__setattr__(self, "frame_ids", normalized_frames)

        sources = tuple(dict.fromkeys(str(source).strip().casefold() for source in self.evidence_sources))
        if any(source not in _VALID_EVIDENCE_SOURCES for source in sources):
            raise AnswerProviderSchemaError("evidence_sources contains an unsupported source")
        object.__setattr__(self, "evidence_sources", sources)
        object.__setattr__(self, "grounding_score", _bounded_score(self.grounding_score, "grounding_score"))
        object.__setattr__(self, "answer_confidence", _bounded_score(self.answer_confidence, "answer_confidence"))
        if not isinstance(self.abstain, bool) or not isinstance(self.accepted, bool):
            raise AnswerProviderSchemaError("abstain and accepted must be bool")
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        if self.response_candidate_id is not None:
            object.__setattr__(
                self,
                "response_candidate_id",
                _required_text(self.response_candidate_id, "response_candidate_id"),
            )
        if self.answer is not None:
            if not isinstance(self.answer, str):
                raise AnswerProviderSchemaError("answer must be a string or None")
            object.__setattr__(self, "answer", self.answer.strip())
        if self.accepted:
            if self.abstain or not self.evidence_sources or not isinstance(self.answer, str) or _is_non_answer(self.answer):
                raise AnswerProviderSchemaError("accepted verification requires a non-empty answer and evidence")

    @property
    def is_accepted(self) -> bool:
        """Readable alias for callers that treat verification as a decision."""
        return self.accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "frame_ids": list(self.frame_ids),
            "evidence_sources": list(self.evidence_sources),
            "answer": self.answer if self.accepted else None,
            "grounding_score": self.grounding_score,
            "answer_confidence": self.answer_confidence,
            "abstain": self.abstain,
            "accepted": self.accepted,
            "reason": self.reason,
            "response_candidate_id": self.response_candidate_id,
        }


class AnswerVerifier(Protocol):
    """Interface implemented by any answer verification policy."""

    def verify(
        self,
        request: AnswerProviderRequest,
        response: AnswerProviderResponse | Mapping[str, Any],
        *,
        required_sources: Sequence[str] = (),
    ) -> AnswerVerification:
        ...


class ContractAnswerVerifier:
    """Validate provider output against the exact request evidence.

    The verifier does not call a model, endpoint, or fallback provider.  It
    only checks identity, schema, answer safety, and declared evidence source;
    semantic support remains an injectable concern for a later verifier.
    """

    @staticmethod
    def _reject(
        request: AnswerProviderRequest,
        *,
        reason: str,
        response: AnswerProviderResponse | None = None,
    ) -> AnswerVerification:
        return AnswerVerification(
            candidate_id=request.evidence.candidate_id,
            video_id=request.evidence.video_id,
            frame_ids=tuple(frame.frame_id for frame in request.evidence.frames),
            evidence_sources=(),
            answer=None,
            grounding_score=(response.grounding_score if response is not None else 0.0),
            answer_confidence=(response.answer_confidence if response is not None else 0.0),
            abstain=True,
            accepted=False,
            reason=reason,
            response_candidate_id=(response.candidate_id if response is not None else None),
        )

    @staticmethod
    def _coerce_response(
        response: AnswerProviderResponse | Mapping[str, Any],
        *,
        candidate_id: str,
    ) -> AnswerProviderResponse:
        if isinstance(response, AnswerProviderResponse):
            return response
        if isinstance(response, Mapping):
            return AnswerProviderResponse.from_mapping(
                response,
                candidate_id=candidate_id,
                provider="mapping",
                model_id="mapping",
            )
        raise AnswerProviderSchemaError("response must be AnswerProviderResponse or a mapping")

    def verify(
        self,
        request: AnswerProviderRequest,
        response: AnswerProviderResponse | Mapping[str, Any],
        *,
        required_sources: Sequence[str] = (),
    ) -> AnswerVerification:
        if not isinstance(request, AnswerProviderRequest):
            raise AnswerProviderInputError("request must be an AnswerProviderRequest")
        try:
            bundle_sources = _bundle_evidence_sources(request.evidence)
        except AnswerProviderInputError:
            return self._reject(request, reason="invalid_evidence_source")

        try:
            normalized = self._coerce_response(response, candidate_id=request.evidence.candidate_id)
        except (AnswerProviderInputError, AnswerProviderSchemaError):
            return self._reject(request, reason="invalid_response")

        if normalized.candidate_id != request.evidence.candidate_id:
            return self._reject(request, reason="candidate_identity_mismatch", response=normalized)
        if not bundle_sources:
            return self._reject(request, reason="missing_evidence_source", response=normalized)
        if normalized.abstain:
            return self._reject(request, reason="provider_abstained", response=normalized)
        if normalized.answer is None or _is_non_answer(normalized.answer):
            return self._reject(request, reason="invalid_answer", response=normalized)

        required: list[str] = []
        for source in required_sources:
            normalized_source = _normalize_evidence_source(source)
            if normalized_source is None:
                return self._reject(request, reason="invalid_required_evidence_source", response=normalized)
            if normalized_source not in required:
                required.append(normalized_source)
        missing = tuple(source for source in required if source not in bundle_sources)
        if missing:
            return self._reject(request, reason="required_evidence_source_missing", response=normalized)

        return AnswerVerification(
            candidate_id=request.evidence.candidate_id,
            video_id=request.evidence.video_id,
            frame_ids=tuple(frame.frame_id for frame in request.evidence.frames),
            evidence_sources=bundle_sources,
            answer=normalized.answer,
            grounding_score=normalized.grounding_score,
            answer_confidence=normalized.answer_confidence,
            abstain=False,
            accepted=True,
            reason="verified_contract",
            response_candidate_id=normalized.candidate_id,
        )


# Explicit alias for callers that prefer a policy-shaped name.
DefaultAnswerVerifier = ContractAnswerVerifier


def verify_answer(
    request: AnswerProviderRequest,
    response: AnswerProviderResponse | Mapping[str, Any],
    *,
    required_sources: Sequence[str] = (),
    verifier: AnswerVerifier | None = None,
) -> AnswerVerification:
    """Verify one response without invoking another provider or network."""
    active_verifier = verifier or ContractAnswerVerifier()
    return active_verifier.verify(request, response, required_sources=required_sources)


def answer_with_verification(
    provider: Any,
    request: AnswerProviderRequest,
    *,
    required_sources: Sequence[str] = (),
    verifier: AnswerVerifier | None = None,
) -> tuple[AnswerProviderResponse, AnswerVerification]:
    """Run one injected provider and fail closed at the verification boundary.

    This helper intentionally has no fallback chain.  A timeout, exhausted
    retry policy, transport error, or unexpected provider exception becomes a
    structured abstention and is returned with a rejected verification record.
    Callers that need a remote provider must inject it explicitly; this
    function never constructs one or retries through another provider.
    """
    if not isinstance(request, AnswerProviderRequest):
        raise AnswerProviderInputError("request must be an AnswerProviderRequest")
    provider_name = str(getattr(provider, "provider_name", "provider") or "provider")
    model_id = str(getattr(provider, "model_id", "unknown") or "unknown")
    try:
        raw_response = provider.answer(request)
        if isinstance(raw_response, AnswerProviderResponse):
            response = raw_response
        elif isinstance(raw_response, Mapping):
            response = AnswerProviderResponse.from_mapping(
                raw_response,
                candidate_id=request.evidence.candidate_id,
                provider=provider_name,
                model_id=model_id,
            )
        else:
            raise AnswerProviderSchemaError("provider returned an unsupported response type")
    except Exception:
        response = AnswerProviderResponse.abstained(
            candidate_id=request.evidence.candidate_id,
            provider=provider_name,
            model_id=model_id,
            reason="provider_request_failed",
        )
    verification = verify_answer(
        request,
        response,
        required_sources=required_sources,
        verifier=verifier,
    )
    return response, verification


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry policy shared by both adapters."""

    max_attempts: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 8.0
    sleep: Callable[[float], None] = time.sleep
    retry_if: Callable[[BaseException], bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise AnswerProviderConfigurationError("max_attempts must be a positive integer")
        if self.initial_backoff_s < 0 or self.max_backoff_s < 0 or self.initial_backoff_s > self.max_backoff_s:
            raise AnswerProviderConfigurationError("backoff values must satisfy 0 <= initial <= max")
        if not callable(self.sleep):
            raise AnswerProviderConfigurationError("sleep must be callable")
        if self.retry_if is not None and not callable(self.retry_if):
            raise AnswerProviderConfigurationError("retry_if must be callable")

    def should_retry(self, error: BaseException) -> bool:
        if self.retry_if is not None:
            return bool(self.retry_if(error))
        return bool(getattr(error, "retryable", False)) or isinstance(error, (TimeoutError, OSError))

    def delay_for(self, retry_number: int) -> float:
        return min(self.max_backoff_s, self.initial_backoff_s * (2 ** max(0, retry_number - 1)))


TimeoutRunner = Callable[[Callable[[], _T], float], _T]
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def _default_timeout_runner(operation: Callable[[], _T], timeout_s: float) -> _T:
    """Run a local operation with a bounded wait.

    The worker cannot be forcibly killed by Python after a timeout, so local
    model implementations should still provide their own cancellation hook
    when available.  The hook is injectable for GPU-aware runtimes and tests.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vqa-answer")
    future: Future[_T] = executor.submit(operation)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError("local answer operation timed out") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_with_retry(
    operation: Callable[[], _T],
    *,
    retry_policy: RetryPolicy,
    wrap_error: Callable[[BaseException], AnswerProviderError],
) -> _T:
    last_error: AnswerProviderError | None = None
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            return operation()
        except (AnswerProviderSchemaError, AnswerProviderInputError, AnswerProviderConfigurationError):
            raise
        except AnswerProviderError as exc:
            last_error = exc
            retryable = retry_policy.should_retry(exc)
            if not retryable or attempt >= retry_policy.max_attempts:
                raise
        except Exception as exc:
            last_error = wrap_error(exc)
            if not retry_policy.should_retry(exc) or attempt >= retry_policy.max_attempts:
                raise last_error from None
        retry_policy.sleep(retry_policy.delay_for(attempt))
    assert last_error is not None
    raise last_error


def _extract_json_object(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if not isinstance(raw, str):
        raise AnswerProviderSchemaError("provider content must be JSON object text")
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidate = match.group(0) if match else text
    try:
        payload = json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AnswerProviderSchemaError("provider content is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise AnswerProviderSchemaError("provider JSON must be an object")
    return payload


def _extract_chat_content(payload: Mapping[str, Any]) -> Any:
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise AnswerProviderSchemaError("API response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise AnswerProviderSchemaError("API choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise AnswerProviderSchemaError("API choice has no message")
    content = message.get("content")
    if content:
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence):
            parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
            return "".join(str(part) for part in parts)
    reasoning = message.get("reasoning_content")
    if reasoning:
        return reasoning
    raise AnswerProviderSchemaError("API response has empty content")


def _build_prompt(request: AnswerProviderRequest) -> str:
    lines = [
        "Answer the question using only the supplied video evidence.",
        "Return ONLY a JSON object with exactly these fields:",
        '{"answer":"short answer", "grounding_score":0.0, "answer_confidence":0.0, "abstain":false}',
        "Scores must be numbers in [0, 1]. Set abstain=true when evidence is insufficient.",
        f"Scene/query: {request.query}",
        f"Question: {request.question}",
        f"Candidate video: {request.evidence.video_id}",
        "Canonical evidence frames: " + ", ".join(str(frame.frame_id) for frame in request.evidence.frames),
    ]
    if request.evidence.asr_text.strip():
        lines.append("ASR evidence:\n" + request.evidence.asr_text.strip())
    if request.evidence.ocr_text.strip():
        lines.append("OCR evidence:\n" + request.evidence.ocr_text.strip())
    return "\n".join(lines)


class AnswerProvider:
    """Abstract answer-provider interface for local and remote adapters."""

    provider_name: str
    model_id: str
    is_remote: bool = False

    def answer(self, request: AnswerProviderRequest) -> AnswerProviderResponse:
        raise NotImplementedError


class QwenLocalAnswerProvider(AnswerProvider):
    """Lazy adapter for a local Qwen2.5-VL-compatible model.

    A model object can be injected for tests or a custom runtime.  If omitted,
    the repository's existing ``LocalVLM`` is constructed lazily from
    ``model_path``; importing this adapter never loads torch or a model.
    """

    provider_name = "qwen_local"
    is_remote = False

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        model: Any = None,
        load_in_4bit: bool = False,
        timeout_s: float = 120.0,
        retry_policy: RetryPolicy | None = None,
        timeout_runner: TimeoutRunner | None = None,
        model_factory: Callable[[str, bool], Any] | None = None,
        model_id: str | None = None,
    ) -> None:
        if model is None and model_path is None:
            raise AnswerProviderConfigurationError("model_path or an injected model is required")
        if timeout_s <= 0 or not math.isfinite(float(timeout_s)):
            raise AnswerProviderConfigurationError("timeout_s must be a positive finite number")
        self.model_path = str(model_path) if model_path is not None else None
        self._model = model
        self.load_in_4bit = load_in_4bit
        self.timeout_s = float(timeout_s)
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_runner = timeout_runner or _default_timeout_runner
        self.model_factory = model_factory
        self.model_id = model_id or (Path(self.model_path).name if self.model_path else "qwen-local")

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        assert self.model_path is not None
        try:
            if self.model_factory is not None:
                self._model = self.model_factory(self.model_path, self.load_in_4bit)
            else:
                from src.core.local_vlm import LocalVLM

                self._model = LocalVLM(self.model_path, load_in_4bit=self.load_in_4bit)
        except ImportError as exc:
            raise AnswerProviderConfigurationError("local Qwen dependencies are unavailable") from exc
        except Exception as exc:
            raise AnswerProviderConfigurationError(f"could not initialize local Qwen model ({type(exc).__name__})") from None
        return self._model

    @staticmethod
    def _validate_frame_paths(request: AnswerProviderRequest) -> tuple[str, ...]:
        paths: list[str] = []
        for frame in request.evidence.frames:
            if frame.frame_path is None:
                raise AnswerProviderInputError("local Qwen requires frame_path for every frame")
            path = Path(frame.frame_path)
            if not path.is_file():
                raise AnswerProviderInputError(f"local evidence frame does not exist: {path.name}")
            paths.append(str(path))
        return tuple(paths)

    def _invoke(self, request: AnswerProviderRequest) -> Any:
        model = self._get_model()
        paths = self._validate_frame_paths(request)
        prompt = _build_prompt(request)

        def operation() -> Any:
            if len(paths) > 1 and callable(getattr(model, "answer_frames_with_metadata", None)):
                return model.answer_frames_with_metadata(
                    list(paths), prompt, max_new_tokens=request.max_new_tokens
                )
            if len(paths) == 1 and callable(getattr(model, "answer_with_metadata", None)):
                return model.answer_with_metadata(paths[0], prompt, max_new_tokens=request.max_new_tokens)
            if len(paths) > 1 and callable(getattr(model, "answer_frames", None)):
                return model.answer_frames(list(paths), prompt, max_new_tokens=request.max_new_tokens)
            answer_fn = getattr(model, "answer", None)
            if not callable(answer_fn):
                raise AnswerProviderConfigurationError("local model must expose answer or answer_with_metadata")
            return answer_fn(paths[0], prompt, max_new_tokens=request.max_new_tokens)

        return self.timeout_runner(operation, self.timeout_s)

    def answer(self, request: AnswerProviderRequest) -> AnswerProviderResponse:
        if not isinstance(request, AnswerProviderRequest):
            raise AnswerProviderInputError("request must be an AnswerProviderRequest")
        started = time.perf_counter()

        def operation() -> Any:
            return self._invoke(request)

        try:
            raw = _run_with_retry(
                operation,
                retry_policy=self.retry_policy,
                wrap_error=lambda exc: AnswerProviderRequestError(
                    f"local Qwen request failed ({type(exc).__name__})",
                    retryable=isinstance(exc, (TimeoutError, OSError)),
                ),
            )
        except (AnswerProviderConfigurationError, AnswerProviderInputError, AnswerProviderSchemaError):
            raise
        except AnswerProviderError:
            raise
        except Exception as exc:
            raise AnswerProviderRequestError(
                f"local Qwen request failed ({type(exc).__name__})",
                retryable=False,
            ) from None
        elapsed = (time.perf_counter() - started) * 1000
        if isinstance(raw, Mapping):
            payload = raw
        else:
            try:
                payload = _extract_json_object(raw)
            except AnswerProviderSchemaError:
                payload = {"answer": raw, "grounding_score": 0.0, "answer_confidence": 0.0}
        try:
            return AnswerProviderResponse.from_mapping(
                payload,
                candidate_id=request.evidence.candidate_id,
                provider=self.provider_name,
                model_id=self.model_id,
                latency_ms=elapsed,
            )
        except AnswerProviderSchemaError:
            # A provider that emits plain non-JSON text is still usable as an
            # answer generator; the structured scores remain conservative.
            if isinstance(raw, str) and raw.strip():
                return AnswerProviderResponse(
                    candidate_id=request.evidence.candidate_id,
                    answer=raw,
                    grounding_score=0.0,
                    answer_confidence=0.0,
                    abstain=False,
                    provider=self.provider_name,
                    model_id=self.model_id,
                    reason="unstructured_local_response",
                    latency_ms=elapsed,
                )
            raise


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # nosec B310 - configured provider endpoint
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        retryable = exc.code == 408 or exc.code == 429 or 500 <= exc.code <= 599
        raise AnswerProviderRequestError(
            f"remote API request failed with HTTP {exc.code}",
            retryable=retryable,
            status_code=exc.code,
        ) from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise AnswerProviderRequestError(
            f"remote API request failed ({type(exc).__name__})",
            retryable=True,
        ) from None
    if not isinstance(decoded, Mapping):
        raise AnswerProviderSchemaError("remote API response must be a JSON object")
    return decoded


class OpenAICompatibleAnswerProvider(AnswerProvider):
    """OpenAI Chat Completions-compatible answer adapter.

    ``transport`` is injectable with signature ``(url, headers, payload,
    timeout_s) -> mapping``.  This keeps tests offline and lets a deployment
    supply a pooled HTTP client without changing the provider contract.
    """

    provider_name = "openai_compatible"
    is_remote = True

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        require_api_key: bool = True,
        chat_path: str = "/chat/completions",
        timeout_s: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        transport: Transport | None = None,
    ) -> None:
        base_url = _required_text(base_url, "base_url").rstrip("/")
        model = _required_text(model, "model")
        if not re.match(r"^https?://[^/?#]+(?:/[^?#]*)?$", base_url, flags=re.IGNORECASE):
            raise AnswerProviderConfigurationError("base_url must be an http(s) URL")
        if "?" in base_url or "#" in base_url:
            raise AnswerProviderConfigurationError("base_url must not contain query parameters or fragments")
        if not isinstance(api_key, str):
            raise AnswerProviderConfigurationError("api_key must be a string")
        if require_api_key and not api_key.strip():
            raise AnswerProviderConfigurationError("api_key is required for this API provider")
        if timeout_s <= 0 or not math.isfinite(float(timeout_s)):
            raise AnswerProviderConfigurationError("timeout_s must be a positive finite number")
        chat_path = "/" + chat_path.strip("/")
        self.base_url = base_url
        self.model_id = model
        self.api_key = api_key
        self.require_api_key = require_api_key
        self.chat_path = chat_path
        self.timeout_s = float(timeout_s)
        self.retry_policy = retry_policy or RetryPolicy()
        self.transport = transport or _default_transport

    @property
    def endpoint(self) -> str:
        return self.base_url + self.chat_path

    def _image_part(self, frame: FrameEvidence) -> dict[str, Any] | None:
        if frame.frame_path is None:
            return None
        path = Path(frame.frame_path)
        if not path.is_file():
            raise AnswerProviderInputError(f"remote evidence frame does not exist: {path.name}")
        suffix = path.suffix.casefold()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
        if mime is None:
            raise AnswerProviderInputError(f"unsupported evidence image type: {suffix or '<none>'}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def _payload(self, request: AnswerProviderRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": _build_prompt(request)}]
        for frame in request.evidence.frames:
            image_part = self._image_part(frame)
            if image_part is not None:
                content.append(image_part)
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": request.max_new_tokens,
            "response_format": {"type": "json_object"},
        }

    def answer(self, request: AnswerProviderRequest) -> AnswerProviderResponse:
        if not isinstance(request, AnswerProviderRequest):
            raise AnswerProviderInputError("request must be an AnswerProviderRequest")
        started = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self._payload(request)

        def operation() -> Mapping[str, Any]:
            return self.transport(self.endpoint, headers, payload, self.timeout_s)

        try:
            response_payload = _run_with_retry(
                operation,
                retry_policy=self.retry_policy,
                wrap_error=lambda exc: AnswerProviderRequestError(
                    f"remote API request failed ({type(exc).__name__})",
                    retryable=isinstance(exc, (TimeoutError, OSError, URLError)),
                ),
            )
            content = _extract_chat_content(response_payload)
            structured = _extract_json_object(content)
            elapsed = (time.perf_counter() - started) * 1000
            return AnswerProviderResponse.from_mapping(
                structured,
                candidate_id=request.evidence.candidate_id,
                provider=self.provider_name,
                model_id=self.model_id,
                latency_ms=elapsed,
            )
        except (AnswerProviderConfigurationError, AnswerProviderInputError, AnswerProviderSchemaError):
            raise
        except AnswerProviderError:
            raise
        except Exception as exc:
            raise AnswerProviderRequestError(
                f"remote API request failed ({type(exc).__name__})",
                retryable=False,
            ) from None


__all__ = [
    "AnswerProvider",
    "AnswerProviderConfigurationError",
    "AnswerProviderError",
    "AnswerProviderInputError",
    "AnswerVerification",
    "AnswerVerifier",
    "AnswerProviderRequest",
    "AnswerProviderRequestError",
    "AnswerProviderResponse",
    "AnswerProviderSchemaError",
    "ContractAnswerVerifier",
    "DefaultAnswerVerifier",
    "EvidenceBundle",
    "FrameEvidence",
    "OpenAICompatibleAnswerProvider",
    "QwenLocalAnswerProvider",
    "RetryPolicy",
    "answer_with_verification",
    "verify_answer",
]
