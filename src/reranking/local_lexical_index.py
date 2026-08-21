"""Small, dependency-light BM25 index for local ASR/OCR text."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata

import numpy as np


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def normalise_text(value: object) -> str:
    """Return stable Unicode/casefolded text, repairing common mojibake."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    if any(marker in text for marker in ("Ã", "Â", "Æ", "Ä", "á»")):
        try:
            repaired = text.encode("latin-1").decode("utf-8")
            if repaired.count("�") <= text.count("�"):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    # ASR often varies Vietnamese tone marks in names and Hán--Việt
    # quotations. Folding marks makes lexical retrieval robust to that
    # variation while original text remains untouched in evidence metadata.
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def tokenize(value: object) -> list[str]:
    return _TOKEN_RE.findall(normalise_text(value))


def quote_match(anchor: object, document: object) -> dict[str, float | bool]:
    """Measure an exact or near quoted-text match using folded Vietnamese.

    This is deliberately a lexical evidence feature, not an answer generator.
    Accents, case, punctuation and the common ``đ``/``d`` ASR variation are
    normalised by :func:`tokenize`, while the original source text is left
    untouched for the caller to retain as evidence.  A near match combines
    multiset token coverage with the longest contiguous token run so that a
    long ASR chunk containing a slightly misrecognised quotation remains
    useful without inventing a missing phrase.
    """
    anchor_tokens = tokenize(anchor)
    document_tokens = tokenize(document)
    if not anchor_tokens or not document_tokens:
        return {
            "exact": False,
            "coverage": 0.0,
            "contiguous": 0.0,
            "score": 0.0,
        }

    anchor_phrase = " ".join(anchor_tokens)
    document_phrase = " ".join(document_tokens)
    exact = anchor_phrase in document_phrase
    if exact:
        return {
            "exact": True,
            "coverage": 1.0,
            "contiguous": 1.0,
            "score": 1.0,
        }

    anchor_counts = Counter(anchor_tokens)
    document_counts = Counter(document_tokens)
    overlap = sum(min(count, document_counts[token]) for token, count in anchor_counts.items())
    coverage = float(overlap) / float(len(anchor_tokens))

    # Chunks and explicit anchors are short enough for the direct scan.  It
    # preserves repeated-token semantics unlike a set-intersection heuristic.
    longest = 0
    for anchor_start, anchor_token in enumerate(anchor_tokens):
        for document_start, document_token in enumerate(document_tokens):
            if anchor_token != document_token:
                continue
            run = 0
            while (anchor_start + run < len(anchor_tokens)
                   and document_start + run < len(document_tokens)
                   and anchor_tokens[anchor_start + run] == document_tokens[document_start + run]):
                run += 1
            longest = max(longest, run)
    contiguous = float(longest) / float(len(anchor_tokens))
    return {
        "exact": False,
        "coverage": coverage,
        "contiguous": contiguous,
        # Contiguous text is a stronger quote indicator than scattered words,
        # but a long ASR chunk with one transcription error still retains its
        # high coverage signal.
        "score": max(coverage, contiguous),
    }


class BM25Index:
    """In-memory BM25 with deterministic scores and preserved row alignment."""

    def __init__(self, documents: list[object], *, k1: float = 1.2, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths = np.zeros(len(documents), dtype=np.float32)
        for doc_id, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self._lengths[doc_id] = sum(counts.values())
            for token, frequency in counts.items():
                self._postings[token].append((doc_id, frequency))
        self._avgdl = float(self._lengths.mean()) if len(self._lengths) else 0.0

    def scores(self, query: object) -> np.ndarray:
        """Return one BM25 score per document."""
        output = np.zeros(len(self._lengths), dtype=np.float32)
        query_terms = Counter(tokenize(query))
        if not query_terms or not len(self._lengths) or self._avgdl <= 0:
            return output
        document_count = len(self._lengths)
        for token, query_frequency in query_terms.items():
            postings = self._postings.get(token)
            if not postings:
                continue
            document_frequency = len(postings)
            idf = math.log1p((document_count - document_frequency + 0.5) /
                             (document_frequency + 0.5))
            for doc_id, term_frequency in postings:
                length_norm = 1.0 - self.b + self.b * float(self._lengths[doc_id]) / self._avgdl
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * length_norm
                output[doc_id] += float(query_frequency) * idf * numerator / denominator
        return output

    def coverage_scores(self, query: object) -> np.ndarray:
        """IDF-weighted query-term coverage, independent of document length.

        BM25 correctly penalizes a very long transcript for broad retrieval,
        but that penalty is harmful for a quoted fact embedded inside a long
        Deepgram utterance.  Coverage supplies a second, bounded lexical
        signal: how much of the query's distinctive vocabulary is present in a
        row, regardless of surrounding narration.  It is a rank feature only;
        callers retain raw BM25 separately for auditability.
        """
        output = np.zeros(len(self._lengths), dtype=np.float32)
        query_terms = set(tokenize(query))
        if not query_terms or not len(self._lengths):
            return output
        document_count = len(self._lengths)
        weights: dict[str, float] = {}
        for token in query_terms:
            postings = self._postings.get(token)
            if not postings:
                continue
            weights[token] = math.log1p(
                (document_count - len(postings) + 0.5) / (len(postings) + 0.5)
            )
        total = sum(weights.values())
        if total <= 0:
            return output
        for token, weight in weights.items():
            for doc_id, _ in self._postings[token]:
                output[doc_id] += weight
        return output / float(total)
