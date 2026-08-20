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
    return text.casefold()


def tokenize(value: object) -> list[str]:
    return _TOKEN_RE.findall(normalise_text(value))


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
