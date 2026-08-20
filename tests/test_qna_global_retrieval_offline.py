"""Focused tests for the Q&A retrieval benchmark's strict offline visual seam."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval import benchmark_qna_global_retrieval_v3 as benchmark
from src.eval.qna_materialized_visual import MaterializedVisualRetriever


def _materialized_inputs(tmp_path: Path) -> tuple[Path, Path]:
    keyframe_map = tmp_path / "global_keyframes.parquet"
    pd.DataFrame([
        {"video_id": "L21_V001", "kf_n": 1, "frame_idx": 100, "pts_time": 4.0},
        {"video_id": "L21_V001", "kf_n": 2, "frame_idx": 200, "pts_time": 8.0},
        {"video_id": "L21_V002", "kf_n": 1, "frame_idx": 300, "pts_time": 12.0},
    ]).to_parquet(keyframe_map)
    report = tmp_path / "visual_report.json"
    report.write_text(json.dumps({
        "per_query": [{
            "query_id": "q0",
            "retrieved_video_ids_top100": ["L21_V001", "L21_V002"],
            "candidate_trace": {"retrieval_top100": [
                {"video_id": "L21_V001", "frame_idx": 100, "kf_n": 1,
                 "pts_time": 4.0, "base_score": 0.9, "video_rank": 0},
                {"video_id": "L21_V001", "frame_idx": 200, "kf_n": 2,
                 "pts_time": 8.0, "base_score": 0.8, "video_rank": 0},
                {"video_id": "L21_V002", "frame_idx": 300, "kf_n": 1,
                 "pts_time": 12.0, "base_score": 0.7, "video_rank": 1},
            ]}
        }]
    }), encoding="utf-8")
    return report, keyframe_map


def test_materialized_visual_retriever_requires_explicit_query_binding(tmp_path: Path):
    report, keyframe_map = _materialized_inputs(tmp_path)
    retriever = MaterializedVisualRetriever(report, keyframe_map)

    with pytest.raises(RuntimeError, match="set_query_id.*lexical fallback"):
        retriever.search("this text must not select a query")

    retriever.set_query_id("q0")
    assert [row[0] for row in retriever.search("ignored", topk=2)] == [
        "L21_V001", "L21_V002"
    ]
    assert retriever.materialized_candidates(["L21_V001"], 2) == [
        ("L21_V001", 100, 1, 0.9),
        ("L21_V001", 200, 2, 0.8),
    ]


def test_materialized_pipeline_does_not_construct_siglip(monkeypatch, tmp_path: Path):
    report, keyframe_map = _materialized_inputs(tmp_path)
    retriever = MaterializedVisualRetriever(report, keyframe_map)

    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("SigLIP/KIS model must not be constructed")

    monkeypatch.setattr("src.pipelines.vqa_pipeline_v3.KISFusionRetriever", fail_if_constructed)
    from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3

    pipeline = VQAPipelineV3(translate=False, kis_retriever=retriever)
    retriever.set_query_id("q0")
    assert pipeline.kis is retriever
    assert pipeline._local_candidates("ignored", ["L21_V001"], 2) == [
        ("L21_V001", 100, 1, 0.9),
        ("L21_V001", 200, 2, 0.8),
    ]


def test_missing_local_visual_dependency_fails_closed_with_exact_models(monkeypatch):
    def fail_constructor(*args, **kwargs):
        raise FileNotFoundError("tokenizer cache is absent")

    monkeypatch.setattr(benchmark, "VQAPipelineV3", fail_constructor)
    with pytest.raises(RuntimeError, match=(
        r"offline visual baseline unavailable.*"
        r"ViT-L-16-SigLIP2-256.*ViT-SO400M-16-SigLIP2-384.*"
        r"tokenizer cache is absent"
    )):
        benchmark._build_pipeline(
            visual_candidates=None,
            keyframe_map="unused.parquet",
        )
