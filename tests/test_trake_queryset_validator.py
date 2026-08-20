import pandas as pd
import pytest

from src.eval.validate_trake_queryset import validate


def test_rejects_synthetic_provenance(tmp_path):
    path = tmp_path / "q.parquet"
    pd.DataFrame({
        "query_id": ["q"], "video_id": ["v"], "step": [0], "kf_n": [1],
        "frame_idx": [2], "pts_time": [1.0], "caption": ["event"],
        "provenance": ["vlm_generated"],
    }).to_parquet(path)
    with pytest.raises(ValueError, match="missing columns"):
        validate(path)


def test_accepts_valid_human_authored_query(tmp_path):
    path = tmp_path / "q.parquet"
    pd.DataFrame({
        "query_id": ["q", "q", "q"], "video_id": ["v", "v", "v"], "step": [0, 1, 2],
        "kf_n": [1, 2, 3], "frame_idx": [2, 3, 4], "pts_time": [1.0, 2.0, 3.0],
        "caption": ["first", "second", "third"], "provenance": ["human_authored"] * 3,
        "split": ["dev"] * 3, "annotator_id": ["a"] * 3, "reviewer_id": ["b"] * 3,
        "confidence": ["high"] * 3, "authoring_method": ["human_timeline_review"] * 3,
        "target_selection_method": ["human_timestamp_review"] * 3,
    }).to_parquet(path)
    assert validate(path)["queries"] == 1


def test_rejects_non_monotonic_event_time(tmp_path):
    path = tmp_path / "q.parquet"
    pd.DataFrame({
        "query_id": ["q", "q", "q"], "video_id": ["v", "v", "v"], "step": [0, 1, 2],
        "kf_n": [1, 2, 3], "frame_idx": [2, 3, 4], "pts_time": [2.0, 1.0, 3.0],
        "caption": ["first", "second", "third"], "provenance": ["human_authored"] * 3,
        "split": ["dev"] * 3, "annotator_id": ["a"] * 3, "reviewer_id": ["b"] * 3,
        "confidence": ["high"] * 3, "authoring_method": ["human_timeline_review"] * 3,
        "target_selection_method": ["human_timestamp_review"] * 3,
    }).to_parquet(path)
    with pytest.raises(ValueError, match="not increasing"):
        validate(path)
