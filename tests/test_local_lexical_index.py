from __future__ import annotations

import numpy as np

from src.reranking.local_lexical_index import BM25Index, tokenize


def test_bm25_prefers_exact_vietnamese_match_and_preserves_rows():
    index = BM25Index([
        "Quảng trường Gendarmenmarkt ở trung tâm Berlin",
        "Một quảng trường ở Hà Nội",
        "Nữ sinh chạy xe máy không đội mũ bảo hiểm",
    ])
    scores = index.scores("Berlin Gendarmenmarkt")
    assert scores.shape == (3,)
    assert int(np.argmax(scores)) == 0
    assert scores[0] > 0
    assert scores[2] == 0


def test_tokenizer_casefolds_unicode_without_dropping_diacritics():
    assert "phú" in tokenize("Phú Quốc")
    assert tokenize("PHÚ QUỐC") == tokenize("phú quốc")
