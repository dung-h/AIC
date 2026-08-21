"""Independent semantic evidence judging for bounded VQA candidates.

This module deliberately judges an *already retrieved* candidate.  It cannot
search the corpus, create a video id/frame id, or answer the question.  That
makes it a precision component rather than a second uncontrolled retrieval
path.  A rejected judgement drops the candidate; a positive judgement only
adds auditable support to its existing local ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import base64
import math
from pathlib import Path
import re
import time
from typing import Any, Protocol

from .answer_provider import (
    AnswerProviderConfigurationError,
    AnswerProviderInputError,
    AnswerProviderRequestError,
    FrameEvidence,
    RetryPolicy,
    _default_transport,
    _extract_chat_content,
    _extract_json_object,
    _run_with_retry,
)


class SemanticEvidenceError(RuntimeError):
    """The evidence judge could not make a contract-safe judgement."""


def _text(value: object, field: str, *, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticEvidenceError(f"{field} must be a non-empty string")
    return " ".join(value.split())[:limit]


def _score(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise SemanticEvidenceError(f"{field} must be a number in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticEvidenceError(f"{field} must be a number in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SemanticEvidenceError(f"{field} must be a number in [0, 1]")
    return result


def _sources(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SemanticEvidenceError("expected_sources must be a sequence")
    normalized = tuple(dict.fromkeys(str(value).strip().casefold() for value in values if str(value).strip()))
    if any(value not in {"visual", "asr", "ocr"} for value in normalized):
        raise SemanticEvidenceError("expected_sources must use visual/asr/ocr")
    return normalized


@dataclass(frozen=True, slots=True)
class SemanticEvidenceRequest:
    """One candidate answer paired with its exact local evidence bundle."""

    query: str
    question: str
    candidate_id: str
    video_id: str
    answer: str
    frames: tuple[FrameEvidence, ...]
    asr_text: str = ""
    ocr_text: str = ""
    expected_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("query", "question", "candidate_id", "video_id", "answer"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if isinstance(self.frames, (str, bytes)) or not isinstance(self.frames, Sequence) or not self.frames:
            raise SemanticEvidenceError("frames must contain at least one canonical frame")
        frames = tuple(self.frames)
        if any(not isinstance(frame, FrameEvidence) for frame in frames):
            raise SemanticEvidenceError("frames must contain FrameEvidence records")
        if len(frames) > 12 or len({frame.frame_id for frame in frames}) != len(frames):
            raise SemanticEvidenceError("frames must contain at most 12 unique canonical frame ids")
        object.__setattr__(self, "frames", frames)
        if not isinstance(self.asr_text, str) or not isinstance(self.ocr_text, str):
            raise SemanticEvidenceError("asr_text and ocr_text must be strings")
        object.__setattr__(self, "asr_text", " ".join(self.asr_text.split())[:8000])
        object.__setattr__(self, "ocr_text", " ".join(self.ocr_text.split())[:8000])
        object.__setattr__(self, "expected_sources", _sources(self.expected_sources))


@dataclass(frozen=True, slots=True)
class SemanticEvidenceVerdict:
    """Structured judgement with no answer field and no hidden raw output."""

    candidate_id: str
    context_supported: bool
    answer_supported: bool
    temporal_consistent: bool
    contradicted: bool
    relevance_score: float
    abstain: bool
    reason: str
    provider: str
    model_id: str
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id", limit=240))
        for field in ("context_supported", "answer_supported", "temporal_consistent", "contradicted", "abstain"):
            if not isinstance(getattr(self, field), bool):
                raise SemanticEvidenceError(f"{field} must be bool")
        object.__setattr__(self, "relevance_score", _score(self.relevance_score, "relevance_score"))
        object.__setattr__(self, "reason", _text(self.reason, "reason", limit=160))
        object.__setattr__(self, "provider", _text(self.provider, "provider", limit=80))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", limit=160))
        if self.latency_ms is not None:
            latency = float(self.latency_ms)
            if not math.isfinite(latency) or latency < 0:
                raise SemanticEvidenceError("latency_ms must be a non-negative finite number")
            object.__setattr__(self, "latency_ms", latency)

    @property
    def accepted(self) -> bool:
        return bool(
            not self.abstain
            and not self.contradicted
            and self.context_supported
            and self.answer_supported
            and self.temporal_consistent
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "context_supported": self.context_supported,
            "answer_supported": self.answer_supported,
            "temporal_consistent": self.temporal_consistent,
            "contradicted": self.contradicted,
            "relevance_score": self.relevance_score,
            "abstain": self.abstain,
            "accepted": self.accepted,
            "reason": self.reason,
            "provider": self.provider,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
        }


class SemanticEvidenceJudge(Protocol):
    enabled: bool

    def judge(self, request: SemanticEvidenceRequest) -> SemanticEvidenceVerdict: ...


class DisabledSemanticEvidenceJudge:
    """Default object that makes the lack of a verifier explicit."""

    enabled = False

    def judge(self, request: SemanticEvidenceRequest) -> SemanticEvidenceVerdict:
        del request
        raise SemanticEvidenceError("semantic evidence judge is disabled")


class CallableSemanticEvidenceJudge:
    """Simple injection seam for a local VLM implementation and tests."""

    enabled = True

    def __init__(self, callback: Callable[[SemanticEvidenceRequest], SemanticEvidenceVerdict | Mapping[str, Any]]):
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._callback = callback

    def judge(self, request: SemanticEvidenceRequest) -> SemanticEvidenceVerdict:
        result = self._callback(request)
        if isinstance(result, SemanticEvidenceVerdict):
            return result
        if isinstance(result, Mapping):
            return SemanticEvidenceVerdict(**dict(result))
        raise SemanticEvidenceError("semantic evidence callback returned an invalid result")


class OpenAICompatibleSemanticEvidenceJudge:
    """JSON-only visual/text verifier for an explicitly enabled online route."""

    enabled = True
    provider_name = "openai_compatible_semantic_evidence"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str,
        timeout_s: float = 45.0,
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
            raise AnswerProviderConfigurationError("api_key is required for semantic evidence verification")
        if not math.isfinite(float(timeout_s)) or not 0.1 <= float(timeout_s) <= 120.0:
            raise AnswerProviderConfigurationError("timeout_s must be in [0.1, 120]")
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
    def _image_part(frame: FrameEvidence) -> dict[str, object] | None:
        if frame.frame_path is None:
            return None
        path = Path(frame.frame_path)
        if not path.is_file():
            raise AnswerProviderInputError(f"semantic evidence frame does not exist: {path.name}")
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.casefold())
        if mime is None:
            raise AnswerProviderInputError("semantic evidence frame has unsupported image type")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    @staticmethod
    def _prompt(request: SemanticEvidenceRequest) -> str:
        sources = ", ".join(request.expected_sources) or "visual/asr/ocr as available"
        return "\n".join((
            "You are an independent evidence verifier for a closed video corpus.",
            "Judge only the supplied candidate frames and timestamp-bounded ASR/OCR text.",
            "Do not use world knowledge. Do not repair a missing fact. Do not answer the question.",
            "Set answer_supported=true only if the proposed answer is directly entailed by the supplied evidence.",
            "Set context_supported=true only if this candidate actually matches the requested scene/context.",
            "Set temporal_consistent=true if the supplied timestamped bundle does not violate the requested order; for a non-temporal question use true unless contradicted.",
            "When uncertain, set abstain=true and all unsupported fields false. Return only JSON with exactly these fields:",
            '{"context_supported":false,"answer_supported":false,"temporal_consistent":false,'
            '"contradicted":false,"relevance_score":0.0,"abstain":true,"reason":"short_label"}',
            f"Scene/query: {request.query}",
            f"Question: {request.question}",
            f"Candidate video: {request.video_id}; canonical frames: {', '.join(str(frame.frame_id) for frame in request.frames)}",
            f"Expected evidence modalities: {sources}",
            f"Proposed answer to verify: {request.answer}",
            "ASR evidence:\n" + (request.asr_text or "(none)"),
            "OCR evidence:\n" + (request.ocr_text or "(none)"),
        ))

    def judge(self, request: SemanticEvidenceRequest) -> SemanticEvidenceVerdict:
        if not isinstance(request, SemanticEvidenceRequest):
            raise SemanticEvidenceError("request must be SemanticEvidenceRequest")
        content: list[dict[str, object]] = [{"type": "text", "text": self._prompt(request)}]
        for frame in request.frames:
            image = self._image_part(frame)
            if image is not None:
                content.append(image)
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 240,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter()
        try:
            response = _run_with_retry(
                lambda: self.transport(self.endpoint, headers, payload, self.timeout_s),
                retry_policy=self.retry_policy,
                wrap_error=lambda exc: AnswerProviderRequestError(
                    f"semantic evidence request failed ({type(exc).__name__})",
                    retryable=isinstance(exc, (TimeoutError, OSError)),
                ),
            )
            data = _extract_json_object(_extract_chat_content(response))
            allowed = {
                "context_supported", "answer_supported", "temporal_consistent", "contradicted",
                "relevance_score", "abstain", "reason",
            }
            if set(data) != allowed:
                raise SemanticEvidenceError("semantic evidence response must contain exactly the declared fields")
            return SemanticEvidenceVerdict(
                candidate_id=request.candidate_id,
                **dict(data),
                provider=self.provider_name,
                model_id=self.model_id,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except SemanticEvidenceError:
            raise
        except Exception as exc:
            raise SemanticEvidenceError(
                f"semantic evidence verification failed ({type(exc).__name__})"
            ) from None


__all__ = [
    "CallableSemanticEvidenceJudge",
    "DisabledSemanticEvidenceJudge",
    "OpenAICompatibleSemanticEvidenceJudge",
    "SemanticEvidenceError",
    "SemanticEvidenceJudge",
    "SemanticEvidenceRequest",
    "SemanticEvidenceVerdict",
]
