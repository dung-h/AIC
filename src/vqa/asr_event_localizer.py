"""Local, fail-closed verse-event localizer for Vietnamese video Q&A.

This module deliberately has no retriever, model, web, or external-answer
dependency. It operates only on canonical text rows that a caller has already
scoped to shortlisted videos. ASR rows may form a recitation while OCR rows
can provide nearby entity context in the same video. Its job is narrowly
bounded: when a question explicitly asks for verse (``câu thơ``/``bài thơ``),
surface consecutive ASR chunks that look like a recited verse and have a local
entity/context mention. It never creates an answer or a frame coordinate.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
import unicodedata
from typing import Any


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_POETRY_CUE_RE = re.compile(r"\b(?:cau tho|bai tho|hai cau tho)\b")
_VIETNAMESE_UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
_PROPER_PHRASE_RE = re.compile(
    rf"(?<![\wÀ-ỹ])([{_VIETNAMESE_UPPER}][\wÀ-ỹ]*(?:\s+[{_VIETNAMESE_UPPER}][\wÀ-ỹ]*)+)",
    re.UNICODE,
)

# These are deliberately conservative.  They remove query boilerplate while
# retaining proper-name tokens (for example ``nguyen trung truc kien giang``)
# that can anchor a local ASR context row.
_ENTITY_STOPWORDS = frozenset({
    "ai", "anh", "bai", "cau", "cho", "cua", "duoc", "hai", "hay", "la",
    "nao", "nhung", "noi", "o", "quoc", "su", "tai", "tho", "ve", "va",
    "viet", "voi", "what", "which", "the", "a", "an",
})
_CONTENT_FALLBACK_STOPWORDS = _ENTITY_STOPWORDS | frozenset({
    "anh", "bao", "ca", "doan", "dinh", "gi", "hung", "kien", "mot", "nha",
    "ngoi", "nguoi", "nhan", "nhung", "than", "trong", "video", "vi", "viem",
})
_SPOKEN_FILLERS = frozenset({
    "bay", "day", "la", "mot", "nhung", "se", "thi", "trong", "va", "voi",
    "xin", "duoc", "cua", "nhu", "cho", "den", "nay", "do", "tu", "ve",
})
_DISCOURSE_INTRO_PHRASES = (
    "sau day", "gioi thieu", "chuong trinh", "quy vi", "xin moi", "tiep theo",
)


def _normalise_text(value: object) -> str:
    """Case/diacritic-fold text without altering evidence retained in output."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    # Match the project's local-text normalization enough to tolerate common
    # Vietnamese ASR mojibake, while keeping this module independently usable.
    if any(marker in text for marker in ("Ã", "Â", "Æ", "Ä", "á»")):
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def _tokens(value: object) -> list[str]:
    return _TOKEN_RE.findall(_normalise_text(value))


def is_explicit_poetry_question(query: str, question: str) -> bool:
    """Whether a request explicitly asks for a verse/quotation event."""

    return bool(_POETRY_CUE_RE.search(_normalise_text(f"{query or ''} {question or ''}")))


def _as_records(rows: Iterable[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]] | None:
    """Accept ordinary mapping rows and DataFrame-like objects without pandas."""
    if hasattr(rows, "to_dict"):
        try:
            rows = rows.to_dict("records")
        except (TypeError, ValueError):
            return None
    try:
        materialized = list(rows)
    except TypeError:
        return None
    if not materialized or not all(isinstance(row, Mapping) for row in materialized):
        return None
    return materialized


@dataclass(frozen=True)
class _CanonicalASRRow:
    video_id: str
    kf_n: int
    frame_idx: int
    pts_time: float
    chunk: str
    source: Mapping[str, Any]
    evidence_modality: str = "asr"
    duplicate_count: int = 1


@dataclass(frozen=True)
class _EntityAnchors:
    """Bounded query anchors used only to support local ASR context."""

    proper_phrases: tuple[tuple[str, ...], ...]
    content_terms: tuple[str, ...]

    @property
    def strategy(self) -> str:
        return "proper_name_phrases" if self.proper_phrases else "bounded_content_fallback"

    @property
    def usable(self) -> bool:
        return bool(self.proper_phrases or len(self.content_terms) >= 2)

    def diagnostic(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "proper_phrases": [" ".join(phrase) for phrase in self.proper_phrases],
            "content_terms": list(self.content_terms),
        }


@dataclass
class ASRPoetryEventLocalizer:
    """Find locally evidenced poetry events after video shortlisting.

    ``last_diagnostic`` is intentionally structured so a caller can trace a
    fail-closed result rather than silently treating no candidate as a score
    failure.  It contains no external text and is safe to add to run traces.
    """

    verse_cluster_gap_s: float = 15.0
    context_window_s: float = 60.0
    min_cluster_size: int = 2
    # The v3 ASR index uses overlapping ~8-second word windows rather than
    # sentence snippets.  A real recitation can therefore share its evidence
    # window with a short presenter lead-in and contain 20--35 tokens.
    min_verse_score: float = 0.60
    last_diagnostic: dict[str, Any] = field(default_factory=dict, init=False)

    def localize(
        self,
        asr_rows: Iterable[Mapping[str, Any]] | Any,
        query: str,
        question: str,
        *,
        shortlisted_video_ids: Sequence[str] | None = None,
        max_candidates_per_video: int = 4,
    ) -> list[dict[str, Any]]:
        """Return ranked canonical ASR candidates, or ``[]`` on any unsafe input.

        Rows must already be local ASR evidence and carry the canonical mapping
        directly.  ``shortlisted_video_ids`` is an optional defensive scope;
        rows outside it are ignored and it also determines the stable video
        tie-break order.  A normal spoken fact deliberately never activates.
        """
        self.last_diagnostic = {}
        if max_candidates_per_video < 1:
            self.last_diagnostic = {"status": "invalid_max_candidates"}
            return []
        if self.min_cluster_size < 2 or self.verse_cluster_gap_s <= 0 or self.context_window_s <= 0:
            self.last_diagnostic = {"status": "invalid_configuration"}
            return []

        activation_text = f"{query or ''} {question or ''}"
        cues = _POETRY_CUE_RE.findall(_normalise_text(activation_text))
        if not cues:
            self.last_diagnostic = {"status": "inactive_non_poetry"}
            return []

        records = _as_records(asr_rows)
        if records is None:
            self.last_diagnostic = {"status": "invalid_asr_rows"}
            return []
        canonical_rows = self._canonical_rows(records)
        if canonical_rows is None:
            # Never select a partly canonical result; the caller must repair
            # the ASR index/mapping before this capability can be used.
            self.last_diagnostic = {"status": "missing_or_invalid_canonical_columns"}
            return []

        canonical_rows = self._deduplicate_canonical_rows(canonical_rows)
        if canonical_rows is None:
            self.last_diagnostic = {"status": "conflicting_canonical_duplicate"}
            return []
        ordered_video_ids = self._ordered_video_ids(shortlisted_video_ids, canonical_rows)
        if not ordered_video_ids:
            self.last_diagnostic = {"status": "empty_shortlist"}
            return []
        allowed = set(ordered_video_ids)
        scoped = [row for row in canonical_rows if row.video_id in allowed]
        if not scoped:
            self.last_diagnostic = {"status": "no_asr_rows_in_shortlist", "cues": cues}
            return []

        anchors = self._entity_anchors(query)
        if not anchors.usable:
            # A verse event without a local entity/context anchor is too broad
            # to be a safe replacement for a general ASR retriever.
            self.last_diagnostic = {
                "status": "insufficient_entity_context",
                "cues": cues,
                "entity_anchors": anchors.diagnostic(),
            }
            return []

        candidates = self._score_scoped_rows(scoped, ordered_video_ids, anchors, cues)
        if not candidates:
            self.last_diagnostic = {
                "status": "no_supported_verse_cluster",
                "cues": cues,
                "entity_anchors": anchors.diagnostic(),
            }
            return []

        output: list[dict[str, Any]] = []
        for video_rank, video_id in enumerate(ordered_video_ids, 1):
            video_candidates = [item for item in candidates if item["video_id"] == video_id]
            # The candidate list represents a local recitation event, so
            # retain the sequence order inside a video.  ``modality_score``
            # remains available to a later cross-video reranker; sorting two
            # poem lines by independent confidence would destroy their local
            # temporal meaning.
            video_candidates.sort(key=lambda item: (float(item["pts_time"]), int(item["kf_n"])))
            for within_video_rank, item in enumerate(video_candidates[:max_candidates_per_video], 1):
                item["video_shortlist_rank"] = video_rank
                item["rank_within_video"] = within_video_rank
                output.append(item)

        output.sort(
            key=lambda item: (
                int(item["video_shortlist_rank"]),
                int(item["rank_within_video"]),
            )
        )
        for rank, item in enumerate(output, 1):
            item["rank"] = rank
        self.last_diagnostic = {
            "status": "ok",
            "cues": cues,
            "entity_anchors": anchors.diagnostic(),
            "candidate_count": len(output),
        }
        return output

    @staticmethod
    def _deduplicate_canonical_rows(rows: Sequence[_CanonicalASRRow]) -> list[_CanonicalASRRow] | None:
        """Collapse repeated same-modality chunks at one canonical frame.

        ASR/OCR providers can emit duplicate or overlapping chunks carrying
        the same keyframe/time. A frame can appear once in a Q&A candidate
        list; keep the text with the richest local evidence, then use lexical
        order as a stable tie-breaker.
        """
        # ASR carries a recital while OCR can carry nearby entity context.
        # They may share one canonical keyframe but remain separate evidence.
        by_coordinate: dict[tuple[str, int, str], _CanonicalASRRow] = {}
        for row in rows:
            key = (row.video_id, row.kf_n, row.evidence_modality)
            existing = by_coordinate.get(key)
            if existing is None:
                by_coordinate[key] = row
                continue
            # One canonical keyframe must map to exactly one frame/time.  A
            # duplicate transcript may be merged, but conflicting coordinates
            # must never be guessed or fabricated.
            if existing.frame_idx != row.frame_idx or not math.isclose(
                existing.pts_time, row.pts_time, rel_tol=0.0, abs_tol=1e-3
            ):
                return None
            candidate_key = (
                ASRPoetryEventLocalizer._verse_likeness(row.chunk),
                len(_tokens(row.chunk)),
                row.chunk.casefold(),
            )
            existing_key = (
                ASRPoetryEventLocalizer._verse_likeness(existing.chunk),
                len(_tokens(existing.chunk)),
                existing.chunk.casefold(),
            )
            chosen = row if candidate_key > existing_key else existing
            by_coordinate[key] = _CanonicalASRRow(
                video_id=chosen.video_id,
                kf_n=chosen.kf_n,
                frame_idx=chosen.frame_idx,
                pts_time=chosen.pts_time,
                chunk=chosen.chunk,
                source=chosen.source,
                evidence_modality=chosen.evidence_modality,
                duplicate_count=existing.duplicate_count + row.duplicate_count,
            )
        return sorted(by_coordinate.values(), key=lambda row: (row.video_id, row.pts_time, row.kf_n, row.frame_idx))

    def _canonical_rows(self, records: list[Mapping[str, Any]]) -> list[_CanonicalASRRow] | None:
        required = ("video_id", "kf_n", "frame_idx", "pts_time")
        output: list[_CanonicalASRRow] = []
        for source in records:
            if any(column not in source for column in required):
                return None
            try:
                video_id = str(source["video_id"]).strip()
                kf_n = int(source["kf_n"])
                frame_idx = int(source["frame_idx"])
                pts_time = float(source["pts_time"])
                # The public merged-ASR artifact uses ``text`` while the
                # router exposes its normalized ``chunk`` alias.  Accept both
                # at this narrow read boundary so the localizer has one
                # canonical behavior in direct diagnostics and production.
                chunk = str(source.get("chunk", source.get("text", "")) or "").strip()
                evidence_modality = str(source.get("evidence_modality", "asr")).strip().lower()
            except (TypeError, ValueError):
                return None
            if (not video_id or kf_n < 0 or frame_idx < 0 or not math.isfinite(pts_time)
                    or not chunk or evidence_modality not in {"asr", "ocr"}):
                return None
            output.append(_CanonicalASRRow(
                video_id, kf_n, frame_idx, pts_time, chunk, source,
                evidence_modality=evidence_modality,
            ))
        return output

    @staticmethod
    def _ordered_video_ids(
        shortlisted_video_ids: Sequence[str] | None,
        rows: Sequence[_CanonicalASRRow],
    ) -> list[str]:
        if shortlisted_video_ids is None:
            values = [row.video_id for row in rows]
        else:
            values = [str(value).strip() for value in shortlisted_video_ids]
        return list(dict.fromkeys(value for value in values if value))

    @staticmethod
    def _entity_anchors(query: str) -> _EntityAnchors:
        """Extract high-precision entity phrases, then a small safe fallback.

        The question text often consists entirely of boilerplate (for example
        ``Hai câu thơ đó là gì?``), so it must not dilute a source entity in
        the query.  Capitalized multi-token spans are preferred because they
        retain names such as ``Nguyễn Trung Trực`` exactly.  If none exists,
        only a bounded set of non-generic query tokens can act as context.
        """
        phrases: list[tuple[str, ...]] = []
        seen_phrases: set[tuple[str, ...]] = set()
        for match in _PROPER_PHRASE_RE.finditer(str(query or "")):
            phrase = tuple(_tokens(match.group(1)))
            if len(phrase) >= 2 and phrase not in seen_phrases:
                phrases.append(phrase)
                seen_phrases.add(phrase)
        if phrases:
            return _EntityAnchors(tuple(phrases), ())

        # This fallback is intentionally narrow.  It is not a bag-of-words
        # copy of the whole query, and generic verse language is excluded so
        # an underspecified "hai câu thơ" query remains fail-closed.
        terms: list[str] = []
        seen_terms: set[str] = set()
        for term in _tokens(query):
            if len(term) < 4 or term in _CONTENT_FALLBACK_STOPWORDS or term in seen_terms:
                continue
            terms.append(term)
            seen_terms.add(term)
            if len(terms) == 4:
                break
        return _EntityAnchors((), tuple(terms))

    def _score_scoped_rows(
        self,
        rows: Sequence[_CanonicalASRRow],
        ordered_video_ids: Sequence[str],
        anchors: _EntityAnchors,
        cues: Sequence[str],
    ) -> list[dict[str, Any]]:
        by_video: dict[str, list[_CanonicalASRRow]] = {video_id: [] for video_id in ordered_video_ids}
        for row in rows:
            by_video.setdefault(row.video_id, []).append(row)

        output: list[dict[str, Any]] = []
        for video_id in ordered_video_ids:
            video_rows = sorted(by_video.get(video_id, ()), key=lambda row: (row.pts_time, row.kf_n))
            if not video_rows:
                continue
            context_rows = self._context_rows(video_rows, anchors)
            # A context row may be short too; explicitly exclude entity-heavy
            # rows from verse candidates, so it supports but cannot displace
            # the recited lines.
            verse_rows = [
                row for row in video_rows
                if row.evidence_modality == "asr"
                and self._anchor_support(row.chunk, anchors)["score"] < 0.60
                and self._verse_likeness(row.chunk) >= self.min_verse_score
            ]
            for cluster in self._clusters(verse_rows):
                if len(cluster) < self.min_cluster_size:
                    continue
                if not self._word_window_cluster_has_event_cue(cluster):
                    continue
                support = self._bounded_context_support(cluster, context_rows, anchors)
                if support is None:
                    continue
                cluster_density = self._cluster_density(cluster)
                # Long v3 word windows must themselves preserve the recital
                # cue. A nearby generic narration line belongs to the same
                # temporal cluster, but is not a safe representative frame
                # for an answer about the poem.
                long_window = any(len(_tokens(item.chunk)) > 20 for item in cluster)
                output_rows = (
                    [item for item in cluster if self._row_has_event_cue(item)]
                    if long_window else list(cluster)
                )
                for position, row in enumerate(output_rows):
                    verse_score = self._verse_likeness(row.chunk)
                    # Every output is part of an increasing-time cluster; the
                    # score still exposes this as an explicit component.
                    temporal_order = 1.0
                    final_score = (
                        0.55 * verse_score
                        + 0.20 * min(1.0, (len(cluster) - 1) / 2.0)
                        + 0.15 * cluster_density
                        + 0.10 * float(support["score"])
                    )
                    output.append({
                        "video_id": row.video_id,
                        "kf_n": row.kf_n,
                        "frame_idx": row.frame_idx,
                        "pts_time": row.pts_time,
                        "modality": "asr",
                        "modality_score": float(final_score),
                        "score_mode": "local_asr_poetry_event",
                        "text": row.chunk,
                        "evidence": {"modality": "asr", "text": row.chunk},
                        "provenance": {
                            "activation": {"kind": "explicit_poetry_cue", "cues": list(cues)},
                        "score_components": {
                            "verse_likeness": float(verse_score),
                                "cluster_size": len(cluster),
                                "cluster_density": float(cluster_density),
                                "temporal_order": temporal_order,
                                "context_support": float(support["score"]),
                            },
                            "cluster": {
                                "position": position + 1,
                                "kf_ns": [candidate.kf_n for candidate in cluster],
                                "frame_idxs": [candidate.frame_idx for candidate in cluster],
                                "time_range_s": [cluster[0].pts_time, cluster[-1].pts_time],
                            },
                            "supporting_context": support,
                            "source": "local_canonical_asr",
                            "canonical_dedup": {
                                "key": [row.video_id, row.kf_n],
                                "input_row_count": row.duplicate_count,
                                "strategy": "strongest_verse_text",
                            },
                        },
                    })
        return output

    def _context_rows(
        self,
        rows: Sequence[_CanonicalASRRow],
        anchors: _EntityAnchors,
    ) -> list[_CanonicalASRRow]:
        # A query with several proper phrases often contains both a person and
        # a broad location (e.g. ``Nguyễn Trung Trực`` and ``Kiên Giang``).
        # The location alone recurs throughout a news/travel video and cannot
        # safely justify an unrelated verse-like passage.  Scores for proper
        # phrases already discount shorter phrases by specificity; require a
        # high score here so context must match the most specific entity
        # rather than merely its surrounding province/city.
        minimum_support = 0.80 if anchors.proper_phrases else 0.40
        output = []
        for row in rows:
            match = self._anchor_support(row.chunk, anchors)
            # A named entity is a high-precision context proof.  Partial
            # token overlap (for example ``nguyên liệu`` matching ``Nguyễn``)
            # must not link an unrelated lesson or recipe to a verse event.
            if anchors.proper_phrases:
                if bool(match.get("exact")):
                    output.append(row)
            elif float(match["score"]) >= minimum_support:
                output.append(row)
        return output

    @staticmethod
    def _word_window_cluster_has_event_cue(cluster: Sequence[_CanonicalASRRow]) -> bool:
        """Require a local recital cue when v3 word windows are long.

        Two ordinary adjacent narration windows can otherwise look like a
        couplet once the tokenizer no longer splits at sentence boundaries.
        Short legacy sentence rows keep their historical behavior; a long
        word-window cluster must retain an explicit recital cue or delimiter.
        """

        if not any(len(_tokens(row.chunk)) > 20 for row in cluster):
            return True
        return any(ASRPoetryEventLocalizer._row_has_event_cue(row) for row in cluster)

    @staticmethod
    def _row_has_event_cue(row: _CanonicalASRRow) -> bool:
        return bool(_POETRY_CUE_RE.search(_normalise_text(row.chunk))) or ":" in row.chunk

    def _bounded_context_support(
        self,
        cluster: Sequence[_CanonicalASRRow],
        context_rows: Sequence[_CanonicalASRRow],
        anchors: _EntityAnchors,
    ) -> dict[str, Any] | None:
        start = cluster[0].pts_time
        matches: list[tuple[float, _CanonicalASRRow, float, dict[str, Any]]] = []
        for row in context_rows:
            # Narration may introduce the person immediately before a verse
            # or identify them in the sentence immediately after it.  Both
            # orders are local evidence; an unbounded mention elsewhere in
            # the video is not.  Keep the prior order slightly preferred but
            # never infer a missing entity from the verse itself.
            signed_gap = row.pts_time - start
            gap = abs(signed_gap)
            if gap > self.context_window_s:
                continue
            match = self._anchor_support(row.chunk, anchors)
            coverage = float(match["score"])
            proximity = max(0.0, 1.0 - gap / self.context_window_s)
            direction_weight = 1.0 if signed_gap <= 0.0 else 0.92
            score = direction_weight * (0.65 * coverage + 0.35 * proximity)
            matches.append((score, row, coverage, match))
        if not matches:
            return None
        score, row, coverage, anchor_match = max(
            matches, key=lambda item: (item[0], item[1].pts_time, item[1].kf_n)
        )
        return {
            "video_id": row.video_id,
            "kf_n": row.kf_n,
            "frame_idx": row.frame_idx,
            "pts_time": row.pts_time,
            "text": row.chunk,
            "modality": row.evidence_modality,
            "entity_coverage": float(coverage),
            "anchor_match": anchor_match,
            "gap_before_cluster_s": float(max(0.0, start - row.pts_time)),
            "gap_after_cluster_s": float(max(0.0, row.pts_time - start)),
            "context_direction": "before" if row.pts_time <= start else "after",
            "score": float(score),
        }

    def _clusters(self, rows: Sequence[_CanonicalASRRow]) -> list[list[_CanonicalASRRow]]:
        clusters: list[list[_CanonicalASRRow]] = []
        current: list[_CanonicalASRRow] = []
        for row in sorted(rows, key=lambda item: (item.pts_time, item.kf_n)):
            if current and row.pts_time - current[-1].pts_time > self.verse_cluster_gap_s:
                clusters.append(current)
                current = []
            current.append(row)
        if current:
            clusters.append(current)
        return clusters

    def _cluster_density(self, cluster: Sequence[_CanonicalASRRow]) -> float:
        if len(cluster) < 2:
            return 0.0
        gaps = [
            max(0.0, cluster[index].pts_time - cluster[index - 1].pts_time)
            for index in range(1, len(cluster))
        ]
        mean_gap = sum(gaps) / len(gaps)
        return max(0.0, min(1.0, 1.0 - mean_gap / self.verse_cluster_gap_s))

    @staticmethod
    def _anchor_support(text: str, anchors: _EntityAnchors) -> dict[str, Any]:
        """Score a context row by phrase match before bounded token fallback."""
        document = _tokens(text)
        document_set = set(document)
        if anchors.proper_phrases:
            phrase_scores: list[dict[str, Any]] = []
            max_length = max(len(phrase) for phrase in anchors.proper_phrases)
            for phrase in anchors.proper_phrases:
                phrase_text = " ".join(phrase)
                exact = phrase_text in " ".join(document)
                coverage = len(set(phrase) & document_set) / float(len(phrase))
                # A three-token name is more diagnostic than a two-token
                # location when both are present in the query.
                specificity = len(phrase) / float(max_length)
                phrase_scores.append({
                    "phrase": phrase_text,
                    "exact": exact,
                    "coverage": coverage,
                    "specificity": specificity,
                    "score": (1.0 if exact else coverage) * specificity,
                })
            best = max(phrase_scores, key=lambda item: (item["score"], item["specificity"], item["phrase"]))
            return {
                "strategy": "proper_name_phrases",
                "score": float(best["score"]),
                "matched_phrase": best["phrase"],
                "exact": bool(best["exact"]),
                "phrase_coverage": float(best["coverage"]),
                "phrase_scores": phrase_scores,
            }
        if anchors.content_terms:
            matches = [term for term in anchors.content_terms if term in document_set]
            return {
                "strategy": "bounded_content_fallback",
                "score": len(matches) / float(len(anchors.content_terms)),
                "matched_terms": matches,
                "content_terms": list(anchors.content_terms),
            }
        return {"strategy": "none", "score": 0.0}

    @staticmethod
    def _verse_likeness(text: str) -> float:
        tokens = _tokens(text)
        length = len(tokens)
        # Word-window ASR may include a compact verse plus a few words of
        # lead-in/outro.  The prior 20-token ceiling silently made those true
        # events invisible after v3 chunking; retain a hard upper bound so a
        # whole paragraph can never become "verse" evidence.
        if length < 4 or length > 42:
            return 0.0
        # Recited verse lines are normally compact and content-heavy.  We do
        # not claim to recognize poetry semantically: this is only a
        # transparent shape feature used after an explicit poetry cue.
        ideal_length = 16.0
        length_score = max(0.0, 1.0 - abs(length - ideal_length) / 26.0)
        filler_ratio = sum(token in _SPOKEN_FILLERS for token in tokens) / float(length)
        content_score = max(0.0, 1.0 - 1.5 * filler_ratio)
        normalized = " ".join(tokens)
        # ASR often puts a short presenter transition immediately before a
        # recital.  It has the same length as a verse line but is not evidence
        # of the verse itself, so penalize explicit discourse-intro phrases.
        # This is a bounded lexical shape rule, not an inferred answer.
        intro_penalty = 0.45 if any(phrase in normalized for phrase in _DISCOURSE_INTRO_PHRASES) else 0.0
        punctuation_bonus = 0.12 if any(mark in str(text) for mark in (":", ",", ";", "/", "…")) else 0.0
        return max(0.0, min(
            1.0,
            0.50 * length_score + 0.50 * content_score + punctuation_bonus - intro_penalty,
        ))


def localize_asr_poetry_events(
    asr_rows: Iterable[Mapping[str, Any]] | Any,
    query: str,
    question: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convenience wrapper for one-shot local, canonical ASR localization."""
    return ASRPoetryEventLocalizer().localize(asr_rows, query, question, **kwargs)
