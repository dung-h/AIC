from __future__ import annotations

import json

import pandas as pd
import pytest

import src.pipelines.vqa_pipeline_v3 as vqa_module
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


def _pipeline() -> VQAPipelineV3:
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._asr = None
    pipeline._asr_by_video = None
    pipeline._asr_context_source = None
    pipeline._context_cache_stats = {
        "asr_video_hits": 0, "asr_video_misses": 0,
        "ocr_video_hits": 0, "ocr_video_misses": 0,
    }
    return pipeline


def _write_merged_index(root, *, status="ready"):
    merged = root / "modality_global_v2" / "asr_global_merged_v2"
    merged.mkdir(parents=True)
    (merged / "asr_global_merge_v2_manifest.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    pd.DataFrame([
        {
            "video_id": "L27_V010",
            "text": "Hóa hồng Nhật Tảo quanh thiên địa Kiếm bạt Kiên Giang khắp vị thần",
            "start": 218.125,
            "end": 226.045,
        }
    ]).to_parquet(merged / "retrieval.parquet", index=False)


def test_asr_context_uses_ready_merged_snapshot_not_legacy_shards(tmp_path, monkeypatch):
    _write_merged_index(tmp_path)
    # This incompatible value demonstrates that a ready global snapshot has
    # exclusive ownership of context, just like global retrieval.
    pd.DataFrame([
        {"chunk": "wrong legacy transcript", "vid": "L27_V010", "start": 218.0, "end": 226.0}
    ]).to_parquet(tmp_path / "asr_chunks_l27_ts.parquet", index=False)
    monkeypatch.setattr(vqa_module, "IDX", str(tmp_path))

    pipeline = _pipeline()
    context = pipeline._asr_context("l27_v010", 221.4, window=3.0)

    assert "Hóa hồng Nhật Tảo" in context
    assert "wrong legacy" not in context
    assert pipeline._asr_context_source == "merged_global_asr"


def test_asr_context_fails_closed_when_visible_merged_index_is_not_ready(tmp_path, monkeypatch):
    _write_merged_index(tmp_path, status="partial")
    pd.DataFrame([
        {"chunk": "legacy fallback must not win", "vid": "L27_V010", "start": 218.0, "end": 226.0}
    ]).to_parquet(tmp_path / "asr_chunks_l27_ts.parquet", index=False)
    monkeypatch.setattr(vqa_module, "IDX", str(tmp_path))

    with pytest.raises(RuntimeError, match="not ready"):
        _pipeline()._asr_context("L27_V010", 221.4)
