from __future__ import annotations

import numpy as np
import pandas as pd

from src.reranking.local_lexical_index import quote_match
from src.reranking.qna_modality_router import QNAModalityRouter


class _ContextEmbedder:
    """Make broad historical context score higher than the exact quote."""

    def embed(self, texts, batch_size=1, normalize=True):
        vectors = []
        for text in texts:
            vectors.append(
                [1.0, 0.0] if "đình thần" in str(text).casefold() else [0.2, 0.98]
            )
        return np.asarray(vectors, dtype=np.float32)


def _router_with_synthetic_asr() -> QNAModalityRouter:
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    router.text_mode = "dense"
    router.embedder = _ContextEmbedder()
    router._lexical_indexes = {}
    router._indexes = {
        "asr": (
            np.asarray([[1.0, 0.0], [0.0, 1.0], [0.4, 0.1]], dtype=np.float32),
            pd.DataFrame([
                {
                    "video_id": "L28_V020", "kf_n": 130, "frame_idx": 6006,
                    "pts_time": 240.24,
                    "chunk": "Đình thần Nguyễn Trung Trực ở Kiên Giang.",
                    "start": 239.0, "end": 241.0, "raw_chunk_id": "context",
                },
                {
                    "video_id": "L28_V020", "kf_n": 143, "frame_idx": 7020,
                    "pts_time": 280.8,
                    # ASR drops the leading consonant in "nhược". The local matcher
                    # must preserve the raw text but still localize this
                    # near-quote anchor ahead of generic context.
                    "chunk": "Tĩnh kiên nược ngọa vô dung địa, bão hận thâm cừu bất hối thiên.",
                    "start": 279.0, "end": 282.0, "raw_chunk_id": "quote",
                },
                {
                    "video_id": "L28_V020", "kf_n": 150, "frame_idx": np.nan,
                    "pts_time": 300.0, "chunk": "Một dòng không có frame canonical.",
                    "start": 299.0, "end": 301.0, "raw_chunk_id": "invalid",
                },
            ]),
        )
    }
    return router


def test_localize_evidence_promotes_near_quote_after_context_video_shortlist():
    router = _router_with_synthetic_asr()

    rows = router.localize_evidence(
        "asr",
        ["L28_V020"],
        ["Nguyễn Trung Trực ở đình thần Kiên Giang"],
        quote_anchors=["Tĩnh kiên nhược ngọa vô dung địa bão hận thâm cừu bất hối thiên"],
        per_video=3,
    )

    # Explicit quotation localization is verification, so unverified nearby
    # context is not returned merely because it mentions the same entity.
    assert [row["frame_idx"] for row in rows] == [7020]
    quote = rows[0]
    assert quote["rank_within_video"] == 1
    assert quote["source_row"]["raw_chunk_id"] == "quote"
    assert quote["text"].startswith("Tĩnh kiên nược")
    assert quote["quote_anchor_provenance"][0]["exact"] is False
    assert quote["quote_anchor_provenance"][0]["score"] > 0.9
    assert {entry["score_mode"] for entry in quote["view_provenance"]} == {
        "embedding", "bm25_coverage"
    }


def test_localize_evidence_never_fabricates_missing_canonical_frame():
    router = _router_with_synthetic_asr()

    rows = router.localize_evidence(
        "asr", ["L28_V020"], ["Nguyễn Trung Trực"], per_video=5
    )

    assert len(rows) == 2
    assert all(row["frame_idx"] in {6006, 7020} for row in rows)
    assert all("source_row" in row and "view_provenance" in row for row in rows)


def test_localize_evidence_rejects_related_but_unverified_external_quote():
    """A wrong famous verse must not label generic local context as grounded."""
    router = _router_with_synthetic_asr()

    rows = router.localize_evidence(
        "asr",
        ["L28_V020"],
        ["Nguyễn Trung Trực ở đình thần Kiên Giang"],
        quote_anchors=["Hỏa hồng Nhật Tảo oanh thiên địa kiếm bạt Kiên Giang khấp quỷ thần"],
        per_video=3,
    )

    assert rows == []


def test_quote_match_is_exact_after_vietnamese_normalization():
    match = quote_match(
        "Tĩnh kiên nhược ngọa vô dung địa",
        "TĨNH, KIÊN NHƯỢC NGỌA VÔ DUNG ĐỊA.",
    )

    assert match == {"exact": True, "coverage": 1.0, "contiguous": 1.0, "score": 1.0}
