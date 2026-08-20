from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.eval.materialize_local_ocr_corpus_v1 import (
    _join_canonical,
    _pack_coverage,
    _pick_pilot_frames,
    _select_frames,
    _select_videos,
    parse_packs,
)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "pts_time": 0.0, "frame_idx": 10},
        {"video_id": "K01_V001", "kf_n": 2, "pts_time": 5.0, "frame_idx": 20},
        {"video_id": "L21_V001", "kf_n": 1, "pts_time": 0.0, "frame_idx": 30},
    ])


def test_pilot_selection_is_canonical_and_representative():
    selected = _pick_pilot_frames(_canonical(), ["K01_V001", "L21_V001"])
    assert list(selected["video_id"]) == ["K01_V001", "L21_V001"]
    assert list(selected["kf_n"]) == [2, 1]


def test_join_canonical_preserves_frame_idx_and_rejects_unknown_frame(tmp_path: Path):
    output = tmp_path / "ocr.jsonl"
    output.write_text('{"video_id":"K01_V001","kf_n":2,"pts_time":5.0,"ocr_text":"25 độ"}\n', encoding="utf-8")
    joined = _join_canonical(output, _canonical())
    assert joined.iloc[0]["frame_idx"] == 20
    assert joined.iloc[0]["ocr_text"] == "25 độ"

    output.write_text('{"video_id":"K01_V001","kf_n":99,"pts_time":5.0,"ocr_text":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="outside canonical map"):
        _join_canonical(output, _canonical())


def test_pack_and_video_selection_is_sorted_and_intersected():
    canonical = pd.DataFrame([
        {"video_id": "L21_V001", "kf_n": 1, "pts_time": 0.0, "frame_idx": 30},
        {"video_id": "K01_V002", "kf_n": 1, "pts_time": 0.0, "frame_idx": 20},
        {"video_id": "K01_V001", "kf_n": 1, "pts_time": 0.0, "frame_idx": 10},
    ])
    assert parse_packs("k01, L21") == {"K01", "L21"}
    assert _select_videos(canonical, packs="K01", video_limit=1) == ["K01_V001"]
    assert _select_videos(canonical, packs="K01", video_ids=["K01_V002"]) == ["K01_V002"]
    selected = _select_frames(canonical, mode="full", packs="K01")
    assert list(selected["video_id"]) == ["K01_V001", "K01_V002"]


def test_pack_coverage_is_explicit_and_does_not_claim_unselected_packs():
    canonical = _canonical()
    selected = _select_frames(canonical, mode="full", packs="K01")
    output = selected.iloc[[0]].copy()
    coverage = _pack_coverage(canonical, selected, output)
    assert sorted(coverage) == ["K01"]
    assert coverage["K01"]["selected_videos"] == 1
    assert coverage["K01"]["output_videos"] == 1
    # The fixture selects two canonical frames but the synthetic OCR output
    # contains one; coverage must expose that partial materialization.
    assert coverage["K01"]["frame_coverage"] == 0.5


def test_unknown_pack_and_video_limit_fail_closed():
    with pytest.raises(ValueError, match="unknown pack"):
        parse_packs("K99")
    with pytest.raises(ValueError, match="video_limit"):
        _select_videos(_canonical(), video_limit=-1)
