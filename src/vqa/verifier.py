"""Independent, fail-closed verification of generated VQA answers.

The verifier is deliberately separate from answer generation.  It accepts an
answer and a normalized evidence bundle, then checks support in timestamped
ASR/OCR text and/or through an injected visual checker.  The default text
check is deterministic and conservative; visual semantics require an
injected checker because this module must not pretend that a frame's pixels
were understood without a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any, Protocol, TypeAlias

from .reranker import EvidenceBundle, EvidenceContractError, TextEvidence, coerce_evidence_bundle


class VerifierContractError(ValueError):
    """The verifier input or injected checker result is invalid."""


_NON_ANSWERS = frozenset({
    "", "null", "none", "n/a", "na", "unknown", "not available",
    "cannot determine", "cannot be determined", "cannot answer",
    "i don't know", "i do not know", "evidence-only", "evidence only",
    "không xác định", "không thể trả lời", "không đủ thông tin",
})
_REFUSAL_PREFIXES = (
    "i cannot", "i can't", "cannot answer", "can't answer",
    "not enough evidence", "insufficient evidence", "unable to answer",
    "không thể", "không đủ",
)
_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "what", "which",
    "who", "where", "when", "how", "of", "to", "and", "or", "in",
    "on", "at", "for", "with", "the", "là", "cái", "gì", "nào",
})
_UNAVAILABLE_MODALITY_STATUSES = frozenset({
    "index_missing", "coverage_missing", "no_speech", "no_text",
    "no_speech_available", "no_text_available",
})
_VI_NUMBER_WORDS = {
    "khong": 0, "mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4,
    "nam": 5, "lam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9,
    "muoi": 10, "tram": 100, "nghin": 1000, "ngan": 1000,
    "trieu": 1_000_000,
}
_EN_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1_000_000,
}


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    for _ in range(2):
        def score(candidate: str) -> int:
            return (
                sum(candidate.count(marker) for marker in ("Ã", "Â", "â", "ð", "Ð", "Ñ", "ì", "í", "î", "ï", "ä", "å", "ç", "Æ", "æ", "ƒ", "„", "™", "»"))
                + sum(candidate.count(pair) for pair in ("â€", "â€™", "Ã", "Â", "ì", "í"))
                + 2 * sum(0x80 <= ord(char) <= 0x9F for char in candidate)
            )
        candidates = []
        for encoding in ("cp1252", "latin1"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "�" not in repaired:
                candidates.append(repaired)
        if not candidates:
            break
        repaired = min(candidates, key=score)
        if score(repaired) >= score(text):
            break
        text = repaired
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).strip()


def _fold_diacritics(value: Any) -> str:
    text = _normalize_text(value).replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _tokens(value: Any) -> set[str]:
    return {token for token in _TOKEN_RE.findall(_fold_diacritics(value))
            if token not in _STOPWORDS and len(token) > 1}


def _numbers(value: Any) -> set[str]:
    return {token.replace(",", ".") for token in _NUMBER_RE.findall(_fold_diacritics(value))}


def _parse_word_number(tokens: Sequence[str], words: Mapping[str, int]) -> set[str]:
    """Parse common Vietnamese/English number phrases conservatively."""
    values: set[str] = set()
    i = 0
    while i < len(tokens):
        if tokens[i] not in words:
            i += 1
            continue
        total = 0
        current = 0
        seen = False
        while i < len(tokens) and tokens[i] in words:
            token = tokens[i]
            value = words[token]
            seen = True
            if token in {"muoi", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}:
                current = max(current, 1) * value if token == "muoi" else current + value
            elif token in {"tram", "hundred"}:
                current = max(current, 1) * 100
            elif token in {"nghin", "ngan", "thousand", "trieu", "million"}:
                scale = value
                total += max(current, 1) * scale
                current = 0
            else:
                current += value
            i += 1
        if seen:
            values.add(str(total + current))
    return values


def _number_values(value: Any) -> set[str]:
    folded = _fold_diacritics(value).replace("-", " ")
    values = set(_numbers(folded))
    tokens = _TOKEN_RE.findall(folded)
    values.update(_parse_word_number(tokens, _VI_NUMBER_WORDS))
    values.update(_parse_word_number(tokens, _EN_NUMBER_WORDS))
    return values


def _answer_is_safe(answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    normalized = _normalize_text(answer)
    return bool(normalized) and normalized not in _NON_ANSWERS and not normalized.startswith(_REFUSAL_PREFIXES)


def _default_text_checker(answer: str, evidence: TextEvidence) -> Mapping[str, Any]:
    """Conservative extractive support for short facts and OCR labels."""
    answer_text = _fold_diacritics(answer)
    evidence_text = _fold_diacritics(evidence.text)
    if not answer_text or not evidence_text:
        return {"supported": False, "score": 0.0, "reason": "empty_text"}
    if answer_text in evidence_text:
        return {"supported": True, "score": 1.0, "reason": "exact_text_match"}
    answer_numbers = _number_values(answer_text)
    evidence_numbers = _number_values(evidence_text)
    if answer_numbers and answer_numbers.intersection(evidence_numbers):
        return {"supported": True, "score": 0.9, "reason": "numeric_fact_match"}
    answer_tokens = _tokens(answer_text)
    evidence_tokens = _tokens(evidence_text)
    if answer_tokens and answer_tokens.issubset(evidence_tokens):
        return {"supported": True, "score": 0.8, "reason": "token_match"}
    return {"supported": False, "score": 0.0, "reason": "no_text_match"}


CheckResult: TypeAlias = bool | float | Mapping[str, Any]


class TextChecker(Protocol):
    def __call__(self, answer: str, evidence: TextEvidence) -> CheckResult: ...


class FrameChecker(Protocol):
    def __call__(self, answer: str, frame: Any) -> CheckResult: ...


def _parse_check_result(value: Any) -> tuple[bool, float, str]:
    if isinstance(value, bool):
        return value, 1.0 if value else 0.0, "checker_bool"
    if isinstance(value, Mapping):
        supported = value.get("supported", value.get("match", False))
        score = value.get("score", 1.0 if supported else 0.0)
        reason = value.get("reason", "checker_mapping")
    else:
        supported = False
        score = value
        reason = "checker_score"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            supported = float(value) > 0.0
    if not isinstance(supported, bool):
        raise VerifierContractError("checker supported field must be bool")
    if isinstance(score, bool):
        raise VerifierContractError("checker score must be numeric")
    try:
        score_float = float(score)
    except (TypeError, ValueError) as exc:
        raise VerifierContractError("checker score must be numeric") from exc
    if not math.isfinite(score_float) or not 0.0 <= score_float <= 1.0:
        raise VerifierContractError("checker score must be finite and in [0, 1]")
    if not isinstance(reason, str) or not reason.strip():
        raise VerifierContractError("checker reason must be non-empty")
    return supported, score_float, reason.strip()


@dataclass(frozen=True)
class EvidenceCheck:
    source: str
    supported: bool
    score: float
    reason: str
    evidence_ref: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "supported": self.supported,
            "score": self.score,
            "reason": self.reason,
            "evidence_ref": dict(self.evidence_ref),
        }


@dataclass(frozen=True)
class VerificationResult:
    """Stable verification output suitable for ranking and trace serialization."""

    candidate_id: str | None
    video_id: str | None
    frame_ids: tuple[int, ...]
    checks: tuple[EvidenceCheck, ...]
    abstain: bool
    reason: str
    support_score: float
    canonical_provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.abstain, bool):
            raise VerifierContractError("abstain must be bool")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise VerifierContractError("reason must be non-empty")
        if isinstance(self.support_score, bool):
            raise VerifierContractError("support_score must be numeric")
        score = float(self.support_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise VerifierContractError("support_score must be finite and in [0, 1]")
        object.__setattr__(self, "support_score", score)

    @property
    def supported_sources(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(check.source for check in self.checks if check.supported))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "video_id": self.video_id,
            "frame_ids": list(self.frame_ids),
            "checks": [check.to_dict() for check in self.checks],
            "supported_sources": list(self.supported_sources),
            "abstain": self.abstain,
            "reason": self.reason,
            "support_score": self.support_score,
            "canonical_provenance": [dict(item) for item in self.canonical_provenance],
        }


class EvidenceVerifier:
    """Verify answer support without relying on the answer generator's scores."""

    def __init__(
        self,
        *,
        text_checker: TextChecker | None = None,
        frame_checker: FrameChecker | None = None,
    ) -> None:
        if text_checker is not None and not callable(text_checker):
            raise TypeError("text_checker must be callable")
        if frame_checker is not None and not callable(frame_checker):
            raise TypeError("frame_checker must be callable")
        self._text_checker = text_checker or _default_text_checker
        self._frame_checker = frame_checker

    @staticmethod
    def _invalid_result(*, candidate_id: str | None, video_id: str | None, reason: str) -> VerificationResult:
        return VerificationResult(
            candidate_id=candidate_id,
            video_id=video_id,
            frame_ids=(),
            checks=(),
            abstain=True,
            reason=reason,
            support_score=0.0,
        )

    def verify(
        self,
        answer: Any,
        candidate: EvidenceBundle | Mapping[str, Any],
        *,
        required_sources: Sequence[str] = (),
    ) -> VerificationResult:
        """Return an explicit abstention whenever support is insufficient."""
        raw_candidate_id = candidate.get("candidate_id") if isinstance(candidate, Mapping) else None
        raw_video_id = candidate.get("video_id") if isinstance(candidate, Mapping) else None
        candidate_id = str(raw_candidate_id) if raw_candidate_id else None
        video_id = str(raw_video_id) if raw_video_id else None
        if not _answer_is_safe(answer):
            return self._invalid_result(
                candidate_id=candidate_id, video_id=video_id, reason="invalid_answer")
        try:
            bundle = coerce_evidence_bundle(candidate)
        except (EvidenceContractError, TypeError, ValueError):
            return self._invalid_result(
                candidate_id=candidate_id, video_id=video_id, reason="missing_or_invalid_evidence")

        required = tuple(dict.fromkeys(str(source).strip().casefold() for source in required_sources if str(source).strip()))
        checks: list[EvidenceCheck] = []
        raw_status = {}
        if isinstance(candidate, Mapping):
            supplied_status = candidate.get("modality_status", {})
            if isinstance(supplied_status, Mapping):
                raw_status = {
                    str(source).strip().casefold(): value
                    for source, value in supplied_status.items()
                    if str(source).strip()
                }

        unavailable: dict[str, str] = {}
        for source, status_row in raw_status.items():
            if not isinstance(status_row, Mapping):
                continue
            status = str(status_row.get("status", "")).strip().casefold()
            if status in _UNAVAILABLE_MODALITY_STATUSES:
                unavailable[source] = status
                checks.append(EvidenceCheck(
                    source=source,
                    supported=False,
                    score=0.0,
                    reason=f"{source}_{status}",
                    evidence_ref={
                        "status": status,
                        "requested": bool(status_row.get("requested", True)),
                        "index_source": status_row.get("index_source"),
                        "row_count": status_row.get("row_count", 0),
                    },
                ))

        for source, rows in (("asr", bundle.asr), ("ocr", bundle.ocr)):
            for index, row in enumerate(rows):
                try:
                    supported, score, reason = _parse_check_result(self._text_checker(answer, row))
                except Exception as exc:  # noqa: BLE001 - checker boundary fails closed.
                    return self._invalid_result(
                        candidate_id=bundle.candidate_id,
                        video_id=bundle.video_id,
                        reason="text_checker_error")
                checks.append(EvidenceCheck(
                    source=source,
                    supported=supported,
                    score=score,
                    reason=reason,
                    evidence_ref={"index": index, "timestamp": row.timestamp,
                                  "start_time": row.start_time, "end_time": row.end_time},
                ))

        if self._frame_checker is not None:
            for index, frame in enumerate(bundle.frames):
                try:
                    supported, score, reason = _parse_check_result(self._frame_checker(answer, frame))
                except Exception as exc:  # noqa: BLE001 - checker boundary fails closed.
                    return self._invalid_result(
                        candidate_id=bundle.candidate_id,
                        video_id=bundle.video_id,
                        reason="frame_checker_error")
                checks.append(EvidenceCheck(
                    source="visual",
                    supported=supported,
                    score=score,
                    reason=reason,
                    evidence_ref={"frame_idx": frame.frame_idx, "kf_n": frame.kf_n,
                                  "pts_time": frame.pts_time, "index": index},
                ))

        supported_sources = {check.source for check in checks if check.supported}
        missing_unavailable = [source for source in required if source in unavailable]
        if missing_unavailable:
            return VerificationResult(
                candidate_id=bundle.candidate_id,
                video_id=bundle.video_id,
                frame_ids=tuple(frame.frame_idx for frame in bundle.frames),
                checks=tuple(checks),
                abstain=True,
                reason="required_evidence_unavailable",
                support_score=max((check.score for check in checks if check.supported), default=0.0),
                canonical_provenance=tuple(frame.to_dict() for frame in bundle.frames),
            )
        missing_required = [source for source in required if source not in supported_sources]
        if missing_required:
            reason = "required_evidence_not_supporting_answer"
            support_score = max((check.score for check in checks if check.supported), default=0.0)
            return VerificationResult(
                candidate_id=bundle.candidate_id,
                video_id=bundle.video_id,
                frame_ids=tuple(frame.frame_idx for frame in bundle.frames),
                checks=tuple(checks),
                abstain=True,
                reason=reason,
                support_score=support_score,
                canonical_provenance=tuple(frame.to_dict() for frame in bundle.frames),
            )

        support_score = max((check.score for check in checks if check.supported), default=0.0)
        if supported_sources:
            return VerificationResult(
                candidate_id=bundle.candidate_id,
                video_id=bundle.video_id,
                frame_ids=tuple(frame.frame_idx for frame in bundle.frames),
                checks=tuple(checks),
                abstain=False,
                reason="supported",
                support_score=support_score,
                canonical_provenance=tuple(frame.to_dict() for frame in bundle.frames),
            )

        if not checks:
            reason = "visual_checker_unavailable" if self._frame_checker is None else "no_supporting_evidence"
        elif self._frame_checker is None and not bundle.asr and not bundle.ocr:
            reason = "visual_checker_unavailable"
        elif unavailable and not supported_sources:
            reason = "modality_evidence_unavailable"
        else:
            reason = "evidence_does_not_support_answer"
        return VerificationResult(
            candidate_id=bundle.candidate_id,
            video_id=bundle.video_id,
            frame_ids=tuple(frame.frame_idx for frame in bundle.frames),
            checks=tuple(checks),
            abstain=True,
            reason=reason,
            support_score=0.0,
            canonical_provenance=tuple(frame.to_dict() for frame in bundle.frames),
        )


__all__ = [
    "EvidenceCheck",
    "EvidenceVerifier",
    "FrameChecker",
    "TextChecker",
    "VerificationResult",
    "VerifierContractError",
]
