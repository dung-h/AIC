"""Bounded, local video-level evidence-role joining for VQA.

This is deliberately *not* a retriever or a reranker.  A caller has already
selected one video and an evidence anchor in that video (for example, an OCR
hit for ``200g thịt nạc xay``).  For an explicit title/name question only, the
joiner can surface a small number of early ASR/OCR rows in the same video that
look like answer-supporting title evidence.  It never changes the retrieval
video, anchor frame, or submission coordinates; it only returns existing,
canonical local evidence.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
import re
import unicodedata
from typing import Any


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_TITLE_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dish_name", re.compile(r"\bten\s+mon\b")),
    ("program_name", re.compile(r"\bten\s+chuong\s+trinh\b")),
    ("title", re.compile(r"\btieu\s+de\b")),
    ("called_name", re.compile(r"\bgoi\s+la\s+gi\b")),
)
_TITLE_CUE_RE = re.compile(
    r"\b(?:mon(?:\s+an)?|chuong\s+trinh|tieu\s+de|tua(?:\s+de)?|ten)\b"
)
_TITLE_ASSERTION_RE = re.compile(
    r"\b(?:la|goi\s+la|mang\s+ten|co\s+ten|tua\s+la|"
    r"lam\s+mon|nau\s+mon|thuc\s+hien\s+mon|mon\s+an\s+hom\s+nay)\b"
)
_GENERIC_PROGRAM_RE = re.compile(r"\bmon\s+ngon\s+moi\s+ngay\b")
_SOURCE_TEXT_FIELDS: tuple[str, ...] = (
    "text",
    "chunk",
    "ocr_text",
    "transcript",
    "content",
)
_STRENGTH_FIELDS: tuple[str, ...] = (
    "evidence_score",
    "grounding_score",
    "match_score",
    "modality_score",
    "score",
)


def _normalise(value: object) -> str:
    """Vietnamese-aware folding used for activation and lexical cues only."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_RE.findall(_normalise(value)))


def _as_records(rows: Iterable[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]] | None:
    if hasattr(rows, "to_dict"):
        try:
            rows = rows.to_dict("records")
        except (TypeError, ValueError):
            return None
    try:
        materialised = list(rows)
    except TypeError:
        return None
    if not materialised or not all(isinstance(row, Mapping) for row in materialised):
        return None
    return materialised


@dataclass(frozen=True)
class _CanonicalEvidenceRow:
    video_id: str
    kf_n: int
    frame_idx: int
    pts_time: float
    source_text: str
    modality: str
    raw: Mapping[str, Any]


@dataclass
class VideoEvidenceRoleJoiner:
    """Find local title/name support for one strong anchor in one video.

    ``last_diagnostic`` distinguishes a deliberate non-activation from missing
    or inconsistent evidence.  The object has no model, network, index, or
    pipeline dependency and is consequently safe to invoke after retrieval.
    """

    min_anchor_strength: float = 0.80
    intro_window_s: float = 90.0
    pre_anchor_window_s: float = 120.0
    # Recipe videos commonly show an ingredient card just before the host
    # states the dish name.  Keep this finite but long enough to include that
    # hand-off; it is still same-video evidence and never a new retrieval.
    post_anchor_window_s: float = 60.0
    max_rows_scanned: int = 64
    max_support_rows: int = 3
    min_support_score: float = 0.55
    last_diagnostic: dict[str, Any] = field(default_factory=dict, init=False)

    def join(
        self,
        rows: Iterable[Mapping[str, Any]] | Any,
        query: str,
        question: str,
        *,
        anchor_candidate: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Return canonical answer-support rows or ``[]`` without fallback.

        ``rows`` must already be restricted to exactly the anchor's selected
        video.  Mixing videos, a non-canonical mapping, lack of title intent,
        or a weak/malformed anchor all fail closed.  The joiner intentionally
        does not return the anchor itself because that remains owned by the
        caller's frame allocator/submission flow.
        """
        self.last_diagnostic = {}
        if not self._valid_configuration():
            self.last_diagnostic = {"status": "invalid_configuration"}
            return []

        intent = self._title_intent(question)
        if intent is None:
            self.last_diagnostic = {"status": "inactive_non_title_intent"}
            return []

        anchor = self._canonical_anchor(anchor_candidate)
        if anchor is None:
            self.last_diagnostic = {"status": "missing_or_invalid_anchor"}
            return []
        anchor_strength, strength_field = self._anchor_strength(anchor_candidate)
        if anchor_strength is None or anchor_strength < self.min_anchor_strength:
            self.last_diagnostic = {
                "status": "weak_anchor",
                "anchor_strength": anchor_strength,
                "min_anchor_strength": self.min_anchor_strength,
            }
            return []

        records = _as_records(rows)
        if records is None:
            self.last_diagnostic = {"status": "invalid_evidence_rows"}
            return []
        canonical_rows = self._canonical_rows(records)
        if canonical_rows is None:
            self.last_diagnostic = {"status": "missing_or_invalid_canonical_fields"}
            return []
        if any(row.video_id != anchor.video_id for row in canonical_rows):
            self.last_diagnostic = {
                "status": "mixed_or_wrong_video_rows",
                "anchor_video_id": anchor.video_id,
            }
            return []

        bounded_rows = self._bounded_intro_rows(canonical_rows, anchor)
        if not bounded_rows:
            self.last_diagnostic = {"status": "no_rows_in_bounded_intro_window"}
            return []

        candidates = self._score_title_support(
            bounded_rows,
            anchor=anchor,
            query=query,
            question=question,
            intent=intent,
        )
        if not candidates:
            self.last_diagnostic = {
                "status": "no_title_support",
                "intent": intent,
                "bounded_rows": len(bounded_rows),
            }
            return []

        output = candidates[: self.max_support_rows]
        anchor_provenance = {
            "video_id": anchor.video_id,
            "kf_n": anchor.kf_n,
            "frame_idx": anchor.frame_idx,
            "pts_time": anchor.pts_time,
            "strength": anchor_strength,
            "strength_field": strength_field,
        }
        for rank, candidate in enumerate(output, 1):
            candidate["rank"] = rank
            candidate["role"] = "answer_support"
            candidate["provenance"] = {
                "joiner": "video_evidence_role_join_v1",
                "activation": {"intent": intent, "question_only": True},
                "anchor": anchor_provenance,
                "bounds": {
                    "intro_window_s": self.intro_window_s,
                    "pre_anchor_window_s": self.pre_anchor_window_s,
                    "post_anchor_window_s": self.post_anchor_window_s,
                },
                "score_components": candidate.pop("_score_components"),
            }

        self.last_diagnostic = {
            "status": "ok",
            "intent": intent,
            "anchor_strength": anchor_strength,
            "bounded_rows": len(bounded_rows),
            "support_rows": len(output),
        }
        return output

    def _valid_configuration(self) -> bool:
        return (
            0.0 <= self.min_anchor_strength <= 1.0
            and self.intro_window_s > 0
            and self.pre_anchor_window_s >= 0
            and self.post_anchor_window_s >= 0
            and self.max_rows_scanned >= 1
            and self.max_support_rows >= 1
            and 0.0 <= self.min_support_score <= 1.0
        )

    @staticmethod
    def _title_intent(question: str) -> str | None:
        folded = _normalise(question)
        for intent, pattern in _TITLE_INTENT_PATTERNS:
            if pattern.search(folded):
                return intent
        return None

    @staticmethod
    def _first_text(row: Mapping[str, Any]) -> str | None:
        for field in _SOURCE_TEXT_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def _canonical_row(cls, row: Mapping[str, Any]) -> _CanonicalEvidenceRow | None:
        text = cls._first_text(row)
        video_id = row.get("video_id")
        modality = row.get("modality") or row.get("source")
        if not isinstance(video_id, str) or not video_id.strip() or not text:
            return None
        if not isinstance(modality, str) or _normalise(modality) not in {"asr", "ocr"}:
            return None
        try:
            kf_n = int(row["kf_n"])
            frame_idx = int(row["frame_idx"])
            pts_time = float(row["pts_time"])
        except (KeyError, TypeError, ValueError):
            return None
        if kf_n < 0 or frame_idx < 0 or not math.isfinite(pts_time) or pts_time < 0:
            return None
        return _CanonicalEvidenceRow(
            video_id=video_id.strip(),
            kf_n=kf_n,
            frame_idx=frame_idx,
            pts_time=pts_time,
            source_text=text,
            modality=_normalise(modality),
            raw=row,
        )

    @classmethod
    def _canonical_rows(
        cls, rows: list[Mapping[str, Any]]
    ) -> list[_CanonicalEvidenceRow] | None:
        canonical: list[_CanonicalEvidenceRow] = []
        for row in rows:
            parsed = cls._canonical_row(row)
            if parsed is None:
                return None
            canonical.append(parsed)
        return canonical

    @classmethod
    def _canonical_anchor(cls, anchor: Mapping[str, Any] | None) -> _CanonicalEvidenceRow | None:
        return cls._canonical_row(anchor) if isinstance(anchor, Mapping) else None

    @staticmethod
    def _anchor_strength(anchor: Mapping[str, Any] | None) -> tuple[float | None, str | None]:
        if not isinstance(anchor, Mapping):
            return None, None
        if anchor.get("is_strong") is True:
            return 1.0, "is_strong"
        for field in _STRENGTH_FIELDS:
            try:
                value = float(anchor[field])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                return value, field
        return None, None

    def _bounded_intro_rows(
        self,
        rows: list[_CanonicalEvidenceRow],
        anchor: _CanonicalEvidenceRow,
    ) -> list[_CanonicalEvidenceRow]:
        """Bound local inspection by time and row count, never by an index scan."""
        lower = max(0.0, anchor.pts_time - self.pre_anchor_window_s)
        upper = min(self.intro_window_s, anchor.pts_time + self.post_anchor_window_s)
        bounded = [row for row in rows if lower <= row.pts_time <= upper]
        bounded.sort(key=lambda row: (row.pts_time, row.kf_n, row.frame_idx, row.modality))
        return bounded[: self.max_rows_scanned]

    def _score_title_support(
        self,
        rows: list[_CanonicalEvidenceRow],
        *,
        anchor: _CanonicalEvidenceRow,
        query: str,
        question: str,
        intent: str,
    ) -> list[dict[str, Any]]:
        query_terms = _tokens(f"{query} {question}")
        scored: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for row in rows:
            # The anchor keeps ownership of its frame.  Even if it contains a
            # title string, this joiner must not replace or duplicate it.
            if (row.kf_n, row.frame_idx) == (anchor.kf_n, anchor.frame_idx):
                continue
            folded = _normalise(row.source_text)
            has_title_cue = bool(_TITLE_CUE_RE.search(folded))
            has_assertion = bool(_TITLE_ASSERTION_RE.search(folded))
            if not has_title_cue:
                continue

            # "Món ngon mỗi ngày" is recurring programme boilerplate, not a
            # dish title.  A dish-name question needs a local statement that
            # actually introduces/names the dish ("làm món…", "tên là…",
            # etc.); otherwise an earlier teaser can override the recipe
            # currently identified by the strong ingredient anchor.
            if intent == "dish_name" and not has_assertion:
                continue
            if _GENERIC_PROGRAM_RE.search(folded) and not has_assertion:
                continue

            row_terms = _tokens(row.source_text)
            lexical_overlap = len(row_terms & query_terms) / max(1, len(query_terms))
            early_score = max(0.0, 1.0 - (row.pts_time / self.intro_window_s))
            before_anchor_score = 1.0 if row.pts_time <= anchor.pts_time else 0.0
            score = (
                0.42  # explicit title/domain cue; not generic text similarity
                + 0.18 * float(has_assertion)
                + 0.22 * early_score
                + 0.10 * before_anchor_score
                + 0.08 * min(1.0, lexical_overlap * 4.0)
            )
            if score < self.min_support_score:
                continue
            key = (row.video_id, row.kf_n, row.frame_idx)
            if key in seen:
                continue
            seen.add(key)
            scored.append(
                {
                    "video_id": row.video_id,
                    "kf_n": row.kf_n,
                    "frame_idx": row.frame_idx,
                    "pts_time": row.pts_time,
                    "modality": row.modality,
                    "source_text": row.source_text,
                    "score": round(score, 6),
                    "_score_components": {
                        "title_cue": has_title_cue,
                        "title_assertion": has_assertion,
                        "early_intro_score": round(early_score, 6),
                        "before_anchor": bool(before_anchor_score),
                        "query_overlap": round(lexical_overlap, 6),
                    },
                }
            )
        return sorted(
            scored,
            key=lambda item: (-float(item["score"]), float(item["pts_time"]), int(item["kf_n"])),
        )
