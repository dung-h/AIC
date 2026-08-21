"""Constraint-aware, text-grounded verification for multi-modal VQA.

The regular answer verifier answers one narrow question: does some evidence
contain the generated answer?  That is insufficient for factual questions:
an answer can occur in an unrelated video which happens to share generic
words such as a province or a recipe ingredient.

This module adds a deliberately small, deterministic layer.  It extracts only
high-precision *query-side* anchors (named entities, acronyms, quoted spans,
and numeric descriptor phrases), then requires the selected video's evidence
to cover them.  It never derives an answer, rewrites a query, or contains
query-specific constants.  The caller remains responsible for retrieval and
for providing timestamped, same-video evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re
import unicodedata
from typing import Any


_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_UPPER_IDENTIFIER_RE = re.compile(r"(?<!\w)[A-ZĐ][A-ZĐ0-9_-]{1,}(?!\w)")
_PROPER_PHRASE_RE = re.compile(
    r"(?<![\wÀ-ỹ])([A-ZÀ-ÝĐ][a-zà-ỹđ]+(?:\s+[A-ZÀ-ÝĐ][a-zà-ỹđ]+){1,4})(?![\wÀ-ỹ])"
)
_QUOTED_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{4,240})[\"“”'‘’]")
_NUMBER_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?\s*(?:%|°\s*c|độ|do|g|gr|kg|ml|lít|lit)?)(?!\w)",
    flags=re.IGNORECASE,
)
_PROPER_PREFIX_STOPWORDS = frozenset({
    "cac", "cau", "chuong", "day", "hoi", "khi", "mot", "nhung",
    "phan", "phong", "sau", "theo", "trong", "video", "voi",
})
_IDENTIFIER_STOPWORDS = frozenset({"ASR", "OCR", "VQA", "KIS", "VKIS", "TRAKE"})
_DESCRIPTOR_STOPWORDS = frozenset({
    "cac", "co", "cua", "duoc", "gi", "la", "mot", "nay", "nhung",
    "tren", "trong", "va", "voi",
})


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return " ".join(
        "".join(char for char in decomposed if unicodedata.category(char) != "Mn").split()
    )


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(_normalise(value)))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw or "").split()).strip()
        key = _normalise(value)
        if value and key and key not in seen:
            result.append(value)
            seen.add(key)
    return tuple(result)


@dataclass(frozen=True)
class QueryClaim:
    """One query-side condition that selected evidence must cover."""

    text: str
    kind: str
    min_score: float

    def __post_init__(self) -> None:
        if self.kind not in {"acronym", "entity", "numeric_phrase", "quote"}:
            raise ValueError("unsupported query claim kind")
        text = " ".join(str(self.text or "").split()).strip()
        if not text:
            raise ValueError("query claim text must be non-empty")
        score = float(self.min_score)
        if not 0.0 < score <= 1.0:
            raise ValueError("query claim min_score must be in (0, 1]")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "min_score", score)

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "kind": self.kind, "min_score": self.min_score}


@dataclass(frozen=True)
class ClaimPolicy:
    """Bounded policy derived from the query, never from a known answer."""

    claims: tuple[QueryClaim, ...]
    active: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "claims": [claim.to_dict() for claim in self.claims],
        }


@dataclass(frozen=True)
class ClaimCheck:
    claim: QueryClaim
    supported: bool
    score: float
    source: str | None
    evidence_ref: Mapping[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim.to_dict(),
            "supported": self.supported,
            "score": self.score,
            "source": self.source,
            "evidence_ref": dict(self.evidence_ref or {}),
        }


@dataclass(frozen=True)
class ClaimVerificationResult:
    accepted: bool
    reason: str
    checks: tuple[ClaimCheck, ...]
    answer_sources: tuple[str, ...]
    role_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "checks": [check.to_dict() for check in self.checks],
            "answer_sources": list(self.answer_sources),
            "role_sources": list(self.role_sources),
        }


def derive_claim_policy(
    query: object,
    *,
    specialist_sources: Sequence[str] = (),
) -> ClaimPolicy:
    """Extract high-precision claims from the query description only.

    The question is deliberately excluded: it often contains answer-shaped
    wording, whereas the scene description is the stable retrieval contract.
    Bare numbers and generic words are ignored to avoid over-constraining a
    routine visual question.
    """
    sources = {
        str(source).strip().casefold() for source in specialist_sources
        if str(source).strip().casefold() in {"asr", "ocr"}
    }
    if not sources:
        return ClaimPolicy((), False)

    text = " ".join(str(query or "").split())
    claims: list[QueryClaim] = []

    for value in _unique(match.group(1) for match in _QUOTED_RE.finditer(text)):
        if len(_tokens(value)) >= 3:
            claims.append(QueryClaim(value, "quote", 0.80))

    for value in _unique(match.group(0) for match in _UPPER_IDENTIFIER_RE.finditer(text)):
        if value not in _IDENTIFIER_STOPWORDS:
            claims.append(QueryClaim(value, "acronym", 1.0))

    for value in _unique(match.group(1) for match in _PROPER_PHRASE_RE.finditer(text)):
        words = _tokens(value)
        if len(words) >= 2 and words[0] not in _PROPER_PREFIX_STOPWORDS:
            # ASR can substitute one syllable in a Vietnamese full name.
            # Requiring two thirds keeps entity evidence strict without
            # rejecting an otherwise corroborated, timestamped transcript.
            claims.append(QueryClaim(value, "entity", 2.0 / 3.0))

    for match in _NUMBER_RE.finditer(text):
        numeric = " ".join(match.group(1).split())
        # Counts such as "2 câu thơ" describe the question's structure, not
        # a retrievable scene condition.  Restrict hard numeric claims to a
        # measured value (200g, 25 độ, 10%, ...); visual count questions keep
        # their existing visual route instead of being over-constrained here.
        if not re.search(r"(?:%|°\s*c|độ|do|g|gr|kg|ml|lít|lit)\s*$", numeric, re.IGNORECASE):
            continue
        tail = re.split(r"[\n.;:!?]", text[match.end():], maxsplit=1)[0]
        descriptor = []
        for raw_token in _TOKEN_RE.findall(tail)[:5]:
            folded = _normalise(raw_token)
            if folded not in _DESCRIPTOR_STOPWORDS and len(folded) >= 2:
                descriptor.append(raw_token)
        # A bare number ("200g") is too common in cooking/news corpora.  A
        # numeric claim is activated only with at least two local descriptors.
        if len(descriptor) >= 2:
            claims.append(QueryClaim(f"{numeric} {' '.join(descriptor[:4])}", "numeric_phrase", 1.0))

    # Deterministic cap and de-duplication across overlapping extractors.
    deduped: list[QueryClaim] = []
    seen: set[tuple[str, str]] = set()
    for claim in claims:
        key = (claim.kind, _normalise(claim.text))
        if key not in seen:
            seen.add(key)
            deduped.append(claim)
        if len(deduped) >= 6:
            break
    return ClaimPolicy(tuple(deduped), bool(deduped))


def score_claim(claim: QueryClaim, evidence_text: object) -> float:
    """Return deterministic coverage of one claim by one evidence text."""
    evidence = _normalise(evidence_text)
    claim_text = _normalise(claim.text)
    if not evidence or not claim_text:
        return 0.0
    if claim.kind == "acronym":
        return 1.0 if re.search(rf"(?<!\w){re.escape(claim_text)}(?!\w)", evidence) else 0.0
    if claim.kind == "quote":
        claim_tokens = _tokens(claim_text)
        evidence_tokens = _tokens(evidence)
        if not claim_tokens or not evidence_tokens:
            return 0.0
        overlap = len(set(claim_tokens).intersection(evidence_tokens)) / len(set(claim_tokens))
        contiguous = 1.0 if claim_text in evidence else 0.0
        return max(overlap, contiguous)
    if claim.kind == "numeric_phrase":
        numeric = _NUMBER_RE.search(claim_text)
        if numeric is None or _normalise(numeric.group(1)) not in evidence:
            return 0.0
        descriptor = [token for token in _tokens(claim_text[numeric.end():]) if token]
        if not descriptor:
            return 1.0
        evidence_tokens = set(_tokens(evidence))
        return len(set(descriptor).intersection(evidence_tokens)) / len(set(descriptor))
    claim_tokens = tuple(dict.fromkeys(_tokens(claim_text)))
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(_tokens(evidence))
    return len(set(claim_tokens).intersection(evidence_tokens)) / len(set(claim_tokens))


def _packet_rows(packet: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, source in (("asr_chunks", "asr"), ("ocr_text", "ocr"), ("claim_evidence", None)):
        for raw in packet.get(key, ()) or ():
            if not isinstance(raw, Mapping):
                continue
            actual_source = str(raw.get("source", source or "")).strip().casefold()
            text = str(raw.get("text", raw.get("chunk", raw.get("ocr_text", ""))) or "").strip()
            if actual_source not in {"asr", "ocr"} or not text:
                continue
            rows.append({
                "source": actual_source,
                "text": text,
                "role": str(raw.get("role", "context")).strip().casefold() or "context",
                "timestamp": raw.get("timestamp", raw.get("pts_time")),
                "kf_n": raw.get("kf_n"),
                "frame_idx": raw.get("frame_idx"),
            })
    return rows


def _answer_score(answer: object, evidence_text: object) -> float:
    answer_text = _normalise(answer)
    evidence = _normalise(evidence_text)
    if not answer_text or not evidence:
        return 0.0
    if answer_text in evidence:
        return 1.0
    answer_tokens = set(_tokens(answer_text))
    evidence_tokens = set(_tokens(evidence))
    if answer_tokens and answer_tokens.issubset(evidence_tokens):
        return 0.8
    if len(answer_tokens) >= 5 and len(answer_tokens.intersection(evidence_tokens)) / len(answer_tokens) >= 0.80:
        return 0.8
    return 0.0


def verify_claim_roles(
    answer: object,
    packet: Mapping[str, Any],
    policy: ClaimPolicy,
    *,
    declared_sources: Sequence[str] = (),
    answer_sources: Sequence[str] = (),
) -> ClaimVerificationResult:
    """Verify query anchors and modality roles for one selected video.

    Claim-support rows may prove that the programme/entity is correct but do
    not themselves count as answer support.  This prevents an answer generator
    from turning an incidental name in a distant transcript into an answer.
    """
    rows = _packet_rows(packet)
    allowed_answers = {
        str(source).strip().casefold() for source in answer_sources
        if str(source).strip().casefold() in {"asr", "ocr"}
    }
    declared = tuple(dict.fromkeys(
        str(source).strip().casefold() for source in declared_sources
        if str(source).strip().casefold() in {"asr", "ocr"}
    ))

    checks: list[ClaimCheck] = []
    claim_sources: set[str] = set()
    for claim in policy.claims:
        best: tuple[float, dict[str, object] | None] = (0.0, None)
        for row in rows:
            score = score_claim(claim, row["text"])
            if score > best[0]:
                best = (score, row)
        score, row = best
        supported = score >= claim.min_score
        if supported and row is not None:
            claim_sources.add(str(row["source"]))
        checks.append(ClaimCheck(
            claim=claim,
            supported=supported,
            score=round(float(score), 6),
            source=(str(row["source"]) if row is not None else None),
            evidence_ref=(
                {key: row.get(key) for key in ("role", "timestamp", "kf_n", "frame_idx")}
                if row is not None else None
            ),
        ))

    accepted_answer_sources: set[str] = set()
    for row in rows:
        # Claim-only evidence identifies the video; it cannot by itself make
        # an arbitrary answer grounded.
        if row["role"] == "claim_support" or (
            allowed_answers and row["source"] not in allowed_answers
        ):
            continue
        if _answer_score(answer, row["text"]) > 0.0:
            accepted_answer_sources.add(str(row["source"]))

    role_sources = tuple(sorted(claim_sources.union(accepted_answer_sources)))
    if policy.active and any(not check.supported for check in checks):
        return ClaimVerificationResult(
            False, "query_claim_not_covered", tuple(checks),
            tuple(sorted(accepted_answer_sources)), role_sources,
        )
    if not accepted_answer_sources:
        return ClaimVerificationResult(
            False, "answer_not_supported_by_non_claim_evidence", tuple(checks),
            (), role_sources,
        )
    missing_declared = [source for source in declared if source not in role_sources]
    if missing_declared:
        return ClaimVerificationResult(
            False, "declared_modality_role_uncovered", tuple(checks),
            tuple(sorted(accepted_answer_sources)), role_sources,
        )
    return ClaimVerificationResult(
        True, "supported", tuple(checks),
        tuple(sorted(accepted_answer_sources)), role_sources,
    )


__all__ = [
    "ClaimCheck",
    "ClaimPolicy",
    "ClaimVerificationResult",
    "QueryClaim",
    "derive_claim_policy",
    "score_claim",
    "verify_claim_roles",
]
