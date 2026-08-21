"""Deterministic query views for grounded video Q&A retrieval.

This is deliberately a *planner*, not an answer generator. It never adds an
answer, calls a remote service, or changes the submission contract. Its job is
to keep the original request intact while exposing independently useful
retrieval views (description, question, and exact anchors) to ASR/OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Mapping

from .grounding import GroundingEvidence


_QUOTED = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,160})[\"“”'‘’]")
_NUMERIC_ANCHOR = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:%|°\s*c|độ|do|g|gr|kg|ml|lít|lit)?(?!\w)",
    flags=re.IGNORECASE,
)
_KNOWLEDGE_CUES = re.compile(
    r"\b(?:bài thơ|câu thơ|trích|tác giả|lịch sử|nhân vật|câu lạc bộ|tổ chức|địa phương|xã|huyện)\b",
    flags=re.IGNORECASE,
)
_OCR_SUPPORT_CUES = re.compile(
    r"\b(?:có thể thấy|nhìn thấy|trên bảng|biển|biển hiệu|địa chỉ|tên xã|tên địa phương|tên là gì|tên gì|tiêu đề|công thức)\b",
    flags=re.IGNORECASE,
)
_CONTEXTUAL_ACRONYM = re.compile(
    r"\b((?i:câu\s+lạc\s+bộ|clb|tổ\s+chức|chương\s+trình|dự\s+án)"
    r"(?:\s+[a-zà-ỹđ][\wÀ-ỹ-]{1,}){0,4}\s+[A-ZĐ][A-ZĐ0-9_-]{1,})\b",
)

# Evidence is untrusted retrieval material. These caps prevent one verbose
# external page from becoming a query-expansion dump that dominates local RRF.
_MAX_EVIDENCE_ITEMS = 5
_MAX_QUOTES = 12
_MAX_ALIASES = 12
_MAX_ATOMS_PER_QUOTE = 3
_MAX_HYPOTHESIS_TEXT = 240
_QUOTED_SPANS = (
    re.compile(r'“([^”]{2,400})”'),
    re.compile(r'"([^"\n]{2,400})"'),
    re.compile(r'‘([^’]{2,400})’'),
    re.compile(r"'([^'\n]{2,400})'"),
    re.compile(r'«([^»]{2,400})»'),
)
_CITED_POETRY = re.compile(
    r"(?:\b(?:hai\s+)?câu\s+thơ\b|\bbài\s+thơ\b|\btrích\s+(?:dẫn|đoạn)\b)"
    r"[^:\n]{0,120}:\s*([^.!?\n]{8,400})",
    flags=re.IGNORECASE,
)
_PROPER_NAME = re.compile(
    r"(?<![\wÀ-ỹ])(?:[A-ZÀ-ÝĐ][a-zà-ỹđ]+(?:[-'][A-ZÀ-ÝĐ][a-zà-ỹđ]+)?)(?:\s+[A-ZÀ-ÝĐ][a-zà-ỹđ]+(?:[-'][A-ZÀ-ÝĐ][a-zà-ỹđ]+)?){1,4}(?![\wÀ-ỹ])"
)
_UPPER_IDENTIFIER = re.compile(r"(?<!\w)[A-ZĐ][A-ZĐ0-9_-]{1,}(?!\w)")
_HISTORIC_VIETNAMESE = (("nhựt", "nhật"),)


def _clean(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _unique(values: list[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _clean(value)
        key = normalized.casefold()
        if normalized and key not in seen:
            output.append(normalized)
            seen.add(key)
    return tuple(output)


def _unaccented_vietnamese(value: str) -> str:
    """Return a deterministic Vietnamese ASR/OCR-friendly spelling view."""

    decomposed = unicodedata.normalize("NFD", value)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def _vietnamese_variants(value: str) -> tuple[str, ...]:
    """Produce only lossless/known-spelling variants; never infer an answer."""

    text = _clean(value)[:_MAX_HYPOTHESIS_TEXT]
    variants = [text]
    unaccented = _unaccented_vietnamese(text)
    if unaccented != text:
        variants.append(unaccented)
    folded = text
    for old, new in _HISTORIC_VIETNAMESE:
        folded = re.sub(old, new, folded, flags=re.IGNORECASE)
    if folded != text:
        variants.append(folded)
        folded_unaccented = _unaccented_vietnamese(folded)
        if folded_unaccented != folded:
            variants.append(folded_unaccented)
    return _unique(variants)


def _atomic_quote_views(value: str) -> tuple[str, ...]:
    """Keep a quote and, for a two-line quote, independently searchable lines."""

    quote = _clean(value).strip(" .;:!?")
    if len(quote) < 3:
        return ()
    atoms = [quote]
    # Vietnamese verse is commonly serialized as two clauses separated by a
    # comma. Keeping each clause makes a noisy ASR chunk retrievable without
    # replacing the exact source quote.
    pieces = [
        _clean(piece).strip(" .;:!?")
        for piece in re.split(r"(?:\r?\n|\s*[;—–]\s*|,\s*)", quote)
    ]
    for piece in pieces:
        if len(piece) >= 8 and piece.casefold() != quote.casefold():
            atoms.append(piece)
        if len(atoms) >= _MAX_ATOMS_PER_QUOTE:
            break
    return _unique(atoms)


def _quoted_spans(value: str) -> tuple[str, ...]:
    spans: list[str] = []
    for pattern in _QUOTED_SPANS:
        spans.extend(match.group(1) for match in pattern.finditer(value))
    return _unique(spans)


def _source_fact_spans(value: str) -> tuple[str, ...]:
    """Extract a cited quotation even when a search snippet drops quotes.

    Search engines commonly remove typography around the exact phrase, e.g.
    ``hai câu thơ: Hỏa hồng …, Kiếm bạt …``.  The cue and colon are source
    text, and the bounded tail is still only a retrieval hypothesis; it is
    never treated as a submitted answer.
    """

    spans = list(_quoted_spans(value))
    spans.extend(match.group(1) for match in _CITED_POETRY.finditer(value))
    return _unique(spans)


def _numeric_fact_anchors(value: str) -> tuple[str, ...]:
    """Keep a supplied numeric fact with its immediate descriptor.

    A bare ``200g`` has hundreds of OCR matches in a recipe corpus.  The
    contiguous user-provided phrase ``200g thịt nạc xay`` is a materially
    different retrieval signal: it is still not an answer, but it can route
    the OCR index to the right recipe card before a VLM reads its title.
    """

    anchors: list[str] = []
    for match in _NUMERIC_ANCHOR.finditer(value):
        numeric = _clean(match.group(0))
        if not numeric:
            continue
        anchors.append(numeric)
        # Never cross a sentence/list boundary: only the contiguous local
        # description is reliable enough to act as an exact evidence view.
        remainder = re.split(r"[\n.;:!?]", value[match.end():], maxsplit=1)[0]
        descriptor = re.findall(r"[\wÀ-ỹ]+", remainder, flags=re.UNICODE)[:4]
        if descriptor:
            anchors.append(f"{numeric} {' '.join(descriptor)}")
    return _unique(anchors)


def _identifier_views(value: str) -> tuple[str, ...]:
    """Extract source-backed entity aliases, excluding one-word prose starts."""

    identifiers: list[str] = []
    for match in _PROPER_NAME.finditer(value):
        phrase = match.group(0)
        tokens = phrase.split()
        identifiers.append(phrase)
        # A sentence-initial preposition such as "Tại Kiên Giang" is not the
        # entity alias we want to retrieve. Add bounded two+ token suffixes,
        # retaining the original span for provenance while exposing Kiên Giang.
        for start in range(1, len(tokens) - 1):
            identifiers.append(" ".join(tokens[start:]))
    identifiers.extend(match.group(0) for match in _UPPER_IDENTIFIER.finditer(value))
    return _unique(identifiers)


def _query_fact_views(value: str) -> tuple[str, ...]:
    """Extract source-text facts that are safe as independent retrieval views.

    The full scene/question view remains the default.  This narrower view is
    only for user-supplied specific entities and contextual identifiers such
    as ``Câu lạc bộ FANA``.  It never reads an answer or invents an alias, so
    it is suitable for both offline BM25 and dense retrieval.  Bare channel
    labels (``TV``, ``HTV7``) are deliberately excluded: they identify many
    videos but almost never identify the requested event.
    """

    values: list[str] = []
    # Unlike external snippets, a user query has no provenance for truncated
    # suffix aliases (``City University``).  Keep the exact multiword span
    # only; broad aliases would turn an evidence view into a generic topic
    # query and can destabilize global RRF.
    for match in _PROPER_NAME.finditer(value):
        phrase = match.group(0)
        # Two-word locations occur in thousands of unrelated news clips;
        # they remain in the full semantic query but are not independently
        # amplified. Three-word names such as Nguyễn Trung Trực retain enough
        # specificity to serve as a standalone evidence view.
        if len(re.findall(r"[\wÀ-ỹ]+", phrase, flags=re.UNICODE)) >= 3:
            values.append(phrase)
    values.extend(match.group(1) for match in _CONTEXTUAL_ACRONYM.finditer(value))
    return _unique(values)


@dataclass(frozen=True, slots=True)
class RetrievalHypothesis:
    """One bounded, source-backed local retrieval hypothesis.

    The object intentionally contains no answer field. Its sole purpose is to
    make later ASR/OCR retrieval explainable and reproducible.
    """

    text: str
    kind: str
    source_url: str
    source_title: str
    provider: str
    evidence_index: int
    source_field: str
    normalized_from: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"quote", "alias"}:
            raise ValueError("kind must be 'quote' or 'alias'")
        object.__setattr__(self, "text", _clean(self.text)[:_MAX_HYPOTHESIS_TEXT])
        if not self.text:
            raise ValueError("hypothesis text must be non-empty")
        if self.source_field not in {"source_title", "source_snippet", "query_variant"}:
            raise ValueError("invalid hypothesis source_field")
        object.__setattr__(self, "normalized_from", (
            _clean(self.normalized_from)[:_MAX_HYPOTHESIS_TEXT]
            if self.normalized_from else None
        ))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "provider": self.provider,
            "evidence_index": self.evidence_index,
            "source_field": self.source_field,
            "normalized_from": self.normalized_from,
        }


def _evidence_texts(item: GroundingEvidence) -> tuple[tuple[str, str], ...]:
    """Return source fields in stable order, retaining compatibility records."""

    values: list[tuple[str, str]] = [("source_title", item.source_title)]
    if item.source_snippet:
        values.append(("source_snippet", item.source_snippet))
    values.extend(("query_variant", variant) for variant in item.query_variants)
    return tuple(values)


def _grounding_hypotheses(
    evidence: tuple[GroundingEvidence, ...],
) -> tuple[RetrievalHypothesis, ...]:
    """Compile source titles/snippets into atomic quote and alias views."""

    output: list[RetrievalHypothesis] = []
    seen: set[tuple[str, str]] = set()
    counts = {"quote": 0, "alias": 0}
    for evidence_index, item in enumerate(evidence[:_MAX_EVIDENCE_ITEMS]):
        for source_field, source_text in _evidence_texts(item):
            quote_candidates = [
                atom
                for span in _source_fact_spans(source_text)
                for atom in _atomic_quote_views(span)
            ]
            candidates = (
                ("quote", quote_candidates),
                ("alias", _identifier_views(source_text)),
            )
            for kind, values in candidates:
                limit = _MAX_QUOTES if kind == "quote" else _MAX_ALIASES
                for base_text in values:
                    for variant in _vietnamese_variants(base_text):
                        key = (kind, variant.casefold())
                        if key in seen or counts[kind] >= limit:
                            continue
                        seen.add(key)
                        output.append(RetrievalHypothesis(
                            text=variant,
                            kind=kind,
                            source_url=item.source_url,
                            source_title=item.source_title,
                            provider=item.provider,
                            evidence_index=evidence_index,
                            source_field=source_field,
                            normalized_from=(base_text if variant != base_text else None),
                        ))
                        counts[kind] += 1
    return tuple(output)


@dataclass(frozen=True, slots=True)
class VQAQueryPlan:
    """Auditable retrieval views derived only from user-provided text."""

    query: str
    question: str
    question_type: str
    modality_queries: Mapping[str, tuple[str, ...]]
    support_modalities: tuple[str, ...]
    exact_anchors: tuple[str, ...]
    external_grounding_eligible: bool
    external_grounding_reasons: tuple[str, ...]
    external_grounding_enabled: bool
    external_grounding_attempted: bool
    external_evidence: tuple[GroundingEvidence, ...]
    quote_views: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    alias_views: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    query_fact_views: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    hypothesis_provenance: tuple[RetrievalHypothesis, ...] = ()

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "question": self.question,
            "question_type": self.question_type,
            "modality_queries": {
                key: list(values)
                for key, values in sorted(self.modality_queries.items())
            },
            "support_modalities": list(self.support_modalities),
            "exact_anchors": list(self.exact_anchors),
            "quote_views": {
                key: list(values) for key, values in sorted(self.quote_views.items())
            },
            "alias_views": {
                key: list(values) for key, values in sorted(self.alias_views.items())
            },
            "query_fact_views": {
                key: list(values) for key, values in sorted(self.query_fact_views.items())
            },
            "hypothesis_provenance": [
                hypothesis.to_dict() for hypothesis in self.hypothesis_provenance
            ],
            "external_grounding_eligible": self.external_grounding_eligible,
            "external_grounding_reasons": list(self.external_grounding_reasons),
            "external_grounding_status": (
                "used" if self.external_evidence else
                ("disabled_by_default" if not self.external_grounding_enabled else
                 ("no_allowlisted_evidence" if self.external_grounding_attempted
                  else "not_eligible"))
            ),
            "external_evidence": [item.to_dict() for item in self.external_evidence],
        }


def build_vqa_query_plan(
    query: object,
    question: object,
    *,
    question_type: object = "unknown",
    modalities: tuple[str, ...] | list[str] = (),
    external_evidence: tuple[GroundingEvidence, ...] | list[GroundingEvidence] = (),
    external_grounding_enabled: bool = False,
    external_grounding_attempted: bool = False,
) -> VQAQueryPlan:
    """Build bounded modality-specific retrieval views without answer leakage."""

    scene = _clean(query)
    asked = _clean(question)
    if not scene or not asked:
        raise ValueError("query and question must be non-empty")
    full = f"{scene}\n{asked}"
    lowered = full.casefold()
    anchors = _unique([
        *(_clean(match.group(1)) for match in _QUOTED.finditer(full)),
        *_numeric_fact_anchors(full),
    ])
    requested = tuple(dict.fromkeys(
        str(modality).strip().lower() for modality in modalities
        if str(modality).strip().lower() in {"asr", "ocr"}
    ))
    evidence = tuple(external_evidence)
    if any(not isinstance(item, GroundingEvidence) for item in evidence):
        raise TypeError("external_evidence must contain GroundingEvidence records")
    hypotheses = _grounding_hypotheses(evidence)
    external_views = [variant for item in evidence for variant in item.query_variants]
    support_modalities = list(requested)
    # The annotation declares the modality that must *verify* the answer.
    # This support-lane heuristic only broadens retrieval/evidence packaging:
    # a spoken query asking for a visible place/name commonly needs OCR to
    # localize the sign while ASR identifies the programme/entity.
    if "ocr" not in support_modalities and _OCR_SUPPORT_CUES.search(lowered):
        support_modalities.append("ocr")
    support_modalities = list(dict.fromkeys(support_modalities))
    query_fact_values = _query_fact_views(scene)
    query_fact_by_modality = {
        modality: query_fact_values for modality in support_modalities
    }
    quote_values = _unique([
        item.text for item in hypotheses if item.kind == "quote"
    ])
    alias_values = _unique([
        item.text for item in hypotheses if item.kind == "alias"
    ])
    quote_views = {modality: quote_values for modality in support_modalities}
    alias_views = {modality: alias_values for modality in support_modalities}
    views = {
        # Keep the historical query/question/anchor views. Hypotheses extend,
        # rather than replace, that contract; raw provider variants remain
        # last for compatibility with existing consumers.
        modality: _unique([
            full, scene, asked, *anchors,
            *quote_views.get(modality, ()),
            *alias_views.get(modality, ()),
            *external_views,
        ])
        for modality in support_modalities
    }
    proper_token_count = sum(
        token[:1].isupper()
        for token in re.findall(r"\b[\wÀ-ỹ]+\b", scene)
        if len(token) > 2
    )
    reasons: list[str] = []
    if _KNOWLEDGE_CUES.search(lowered):
        reasons.append("knowledge_or_quote_cue")
    if proper_token_count >= 2:
        reasons.append("named_entity_like_scene")
    return VQAQueryPlan(
        query=scene,
        question=asked,
        question_type=_clean(question_type).casefold() or "unknown",
        modality_queries=views,
        support_modalities=tuple(support_modalities),
        exact_anchors=anchors,
        external_grounding_eligible=bool(reasons),
        external_grounding_reasons=tuple(reasons),
        external_grounding_enabled=bool(external_grounding_enabled),
        external_grounding_attempted=bool(external_grounding_attempted),
        external_evidence=evidence,
        quote_views=quote_views,
        alias_views=alias_views,
        query_fact_views=query_fact_by_modality,
        hypothesis_provenance=hypotheses,
    )


__all__ = ["RetrievalHypothesis", "VQAQueryPlan", "build_vqa_query_plan"]
