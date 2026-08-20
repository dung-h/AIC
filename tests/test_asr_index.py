from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reranking.asr_index import (
    ASRIndex,
    ASRIndexPreflightError,
    map_timestamp_to_canonical,
    normalize_transcript,
    preflight_asr_metadata,
)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame({
        "video_id": ["K01_V001", "K01_V001", "K01_V001"],
        "kf_n": [1, 2, 3],
        "frame_idx": [100, 200, 300],
        "pts_time": [1.0, 5.0, 10.0],
    })


def _metadata() -> pd.DataFrame:
    return pd.DataFrame({
        "vid": ["K01_V001", "K01_V001"],
        "kf_n": [2, 3],
        "frame_idx": [200, 300],
        "start": [4.0, 9.0],
        "end": [6.0, 11.0],
        "chunk": ["Nhiá»‡t Ä‘á»™ Nha Trang là hai mÆ°Æ¡i lÄƒm Ä‘á»™.", "MÆ°a tá»›i vào buổi tối."],
    })


def test_normalize_transcript_repairs_mojibake_but_preserves_valid_vietnamese():
    assert "Nhiệt độ" in normalize_transcript("Nhiá»‡t Ä‘á»™")
    assert normalize_transcript("Nhiệt độ ở Nha Trang") == "Nhiệt độ ở Nha Trang"


def test_timestamp_maps_to_canonical_frame():
    mapped = map_timestamp_to_canonical("K01_V001", 5.7, _canonical())
    assert mapped["kf_n"] == 2
    assert mapped["frame_idx"] == 200
    assert mapped["distance_seconds"] == pytest.approx(0.7)


def test_preflight_accepts_vid_alias_and_repairable_mojibake():
    report = preflight_asr_metadata(_metadata(), _canonical())
    assert report["passed"], report["errors"]
    assert report["text_quality"]["repairable_mojibake_rows"] == 2
    assert report["timestamp_mapping"]["mapped_rows"] == 2


def test_preflight_rejects_replacement_and_mapping_mismatch():
    metadata = _metadata().copy()
    metadata.loc[0, "chunk"] = "bad\ufffd transcript"
    metadata.loc[1, "frame_idx"] = 100
    report = preflight_asr_metadata(metadata, _canonical())
    codes = {item["code"] for item in report["errors"]}
    assert "text_replacement_char" in codes
    assert "canonical_mapping_mismatch" in codes


def test_bm25_index_returns_grounded_nonempty_evidence():
    index = ASRIndex.from_frames(_metadata(), _canonical(), mode="bm25")
    results = index.search("Nhiệt độ Nha Trang", topk=2)
    assert results
    assert results[0]["video_id"] == "K01_V001"
    assert results[0]["canonical_frame_idx"] == 200
    assert results[0]["chunk"]
    assert results[0]["score_mode"] == "bm25"


def test_null_source_mapping_is_recomputed_from_timestamp():
    metadata = _metadata().copy()
    metadata["kf_n"] = np.nan
    metadata["frame_idx"] = np.nan
    report = preflight_asr_metadata(metadata, _canonical())
    assert report["passed"], report["errors"]
    assert report["timestamp_mapping"]["recomputed_mapping_rows"] == 2
    index = ASRIndex.from_frames(metadata, _canonical(), mode="bm25")
    result = index.search("Nhiệt độ Nha Trang", topk=1)[0]
    assert result["kf_n"] == 2
    assert result["frame_idx"] == 200


def test_dense_index_rejects_misaligned_embeddings():
    with pytest.raises(ASRIndexPreflightError):
        ASRIndex.from_frames(_metadata(), _canonical(), mode="dense", embeddings=np.zeros((1, 4), dtype=np.float32))
