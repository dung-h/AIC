import json

import pandas as pd
import pytest

from src.queryset.normalize_vqa_kf_windows_v1 import normalize


def test_normalize_unions_primary_target_without_overwriting_source(tmp_path):
    canonical = tmp_path / "canonical.parquet"
    pd.DataFrame([
        {"video_id": "V1", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
        {"video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
    ]).to_parquet(canonical, index=False)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "annotation_id": "q1", "video_id": "V1", "kf_n": 2,
        "frame_idx": 20, "acceptable_kf_n": "3", "status": "valid",
    }) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    report = tmp_path / "report.json"

    result = normalize(source, output, report, canonical_path=canonical)

    assert result["changed_rows"] == 1
    assert result["production_eligible"] is False
    assert json.loads(source.read_text(encoding="utf-8"))["acceptable_kf_n"] == "3"
    assert json.loads(output.read_text(encoding="utf-8"))["acceptable_kf_n"] == "2,3"


def test_normalize_fails_on_primary_frame_mismatch(tmp_path):
    canonical = tmp_path / "canonical.parquet"
    pd.DataFrame([
        {"video_id": "V1", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
    ]).to_parquet(canonical, index=False)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "annotation_id": "q1", "video_id": "V1", "kf_n": 2,
        "frame_idx": 99, "acceptable_kf_n": "2", "status": "valid",
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frame_idx disagrees"):
        normalize(source, tmp_path / "output.jsonl", tmp_path / "report.json", canonical_path=canonical)
