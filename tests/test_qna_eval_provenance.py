"""Focused contract tests for Q&A retrieval report provenance."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.benchmark_qna_ranked_offline import _retrieval_candidate_pool
from src.eval.benchmark_qna_global_retrieval_v3 import (
    attach_report_provenance,
    build_report_provenance,
    load_qna_retrieval_report,
)
from src.eval.validate_qna_bottlenecks_v1 import _check_source_parity
from src.eval.qna_materialized_visual import (
    MaterializedVisualRetriever,
    candidate_pool_contract,
    candidate_pool_digest,
)


def _current_inputs(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    annotation = tmp_path / "vqa_eval_v3.jsonl"
    annotation.write_text('{"annotation_id":"q1"}\n', encoding="utf-8")
    canonical = tmp_path / "global_keyframes.parquet"
    canonical.write_bytes(b"canonical-map-v1")
    visual = tmp_path / "visual_candidates.json"
    visual.write_text('{"per_query":[]}', encoding="utf-8")

    asr_dir = tmp_path / "asr_global_v2"
    ocr_dir = tmp_path / "ocr_global_v2"
    asr_dir.mkdir(exist_ok=True)
    ocr_dir.mkdir(exist_ok=True)
    (asr_dir / "asr_global_manifest.json").write_text(
        json.dumps({"scope_digest": "asr-scope-v1"}), encoding="utf-8"
    )
    (ocr_dir / "ocr_global_manifest.json").write_text(
        json.dumps({"scope_digest": "ocr-scope-v1"}), encoding="utf-8"
    )
    preflight = {
        "passed": True,
        "index_dir": str(tmp_path),
        "expected_packs": ["l21"],
        "active_modalities": ["asr", "ocr"],
        "sources": {"asr": "global_merged_v2", "ocr": "global_merged_v2"},
        "coverage": {"asr_observed": ["l21"], "ocr_observed": ["l21"]},
        "asr": [{"pack": "l21", "metadata_rows": 1}],
        "ocr": [{"pack": "l21", "metadata_rows": 1}],
    }
    visual_source = {
        "mode": "materialized",
        "identity": "materialized_visual_candidate_report",
        "report": str(visual),
        "keyframe_map": str(canonical),
        "artifact_paths": [str(visual), str(canonical)],
    }
    return annotation, canonical, preflight, visual_source


def _build_current_report(tmp_path: Path) -> dict:
    annotation, canonical, preflight, visual_source = _current_inputs(tmp_path)
    provenance = build_report_provenance(
        annotation_path=annotation,
        source_split="holdout",
        query_ids=["q1"],
        canonical_map_path=canonical,
        visual_source=visual_source,
        active_modalities=["asr", "ocr"],
        modality_index_dir=tmp_path,
        preflight=preflight,
        config={
            "weights": {"visual": 1.0, "asr": 0.5, "ocr": 0.25},
            "selector_policy": "balanced",
        },
        code_path=__file__,
    )
    return attach_report_provenance(
        {"experiment": "test", "split": "holdout", "status": "complete"},
        provenance,
    )


def test_current_report_emits_complete_deterministic_provenance(tmp_path: Path):
    report = _build_current_report(tmp_path)
    provenance = report["provenance"]

    assert report["schema_version"] == "hcmai.qna_global_retrieval_v3"
    assert report["source_split"] == "holdout"
    assert report["run_id"].startswith("qna-v3-")
    assert report["run_fingerprint"] == provenance["run_fingerprint"]
    assert report["promotion_eligible"] is True
    assert provenance["annotation"]["sha256"]
    assert provenance["canonical_map"]["sha256"]
    assert provenance["visual_source"]["source_digest"]
    assert provenance["modality_indexes"]["asr"]["manifest_id"]
    assert provenance["modality_indexes"]["asr"]["index_id"]
    assert provenance["modality_indexes"]["ocr"]["manifest_id"]
    assert provenance["config"]["weights"]["asr"] == 0.5
    assert provenance["config"]["selector_policy"] == "balanced"

    second = _build_current_report(tmp_path)
    assert second["run_id"] == report["run_id"]
    assert second["run_fingerprint"] == report["run_fingerprint"]


def test_loader_marks_old_report_legacy_and_non_promotable(tmp_path: Path):
    path = tmp_path / "old_v31.json"
    path.write_text(
        json.dumps({
            "experiment": "old v31",
            "split": "holdout",
            "per_query": [{"query_id": "q1"}],
        }),
        encoding="utf-8",
    )

    loaded = load_qna_retrieval_report(path)

    assert loaded["provenance_status"] == "legacy"
    assert loaded["promotion_eligible"] is False
    assert "current_report_schema_missing_or_mismatched" in loaded["promotion_blockers"]


def test_current_schema_without_provenance_is_blocked(tmp_path: Path):
    path = tmp_path / "incomplete_v3.json"
    path.write_text(
        json.dumps({
            "schema_version": "hcmai.qna_global_retrieval_v3",
            "status": "complete",
            "per_query": [],
        }),
        encoding="utf-8",
    )

    loaded = load_qna_retrieval_report(path)

    assert loaded["provenance_status"] == "legacy"
    assert loaded["promotion_eligible"] is False
    assert "provenance_block_missing" in loaded["promotion_blockers"]


def test_candidate_pool_digest_is_order_sensitive_and_contract_is_explicit():
    first = [
        {"video_id": "v1", "frame_idx": 10, "kf_n": 1, "video_rank": 0},
        {"video_id": "v2", "frame_idx": 20, "kf_n": 2, "video_rank": 1},
    ]
    second = list(reversed(first))

    assert candidate_pool_digest(first) != candidate_pool_digest(second)
    contract = candidate_pool_contract(first, query_id="q1")
    assert contract["version"] == "hcmai.qna.candidate_pool.v1"
    assert contract["candidate_count"] == 2
    assert contract["query_id"] == "q1"
    assert contract["digest"] == candidate_pool_digest(first)


def test_materialized_retriever_exposes_the_same_source_pool_contract(tmp_path: Path):
    keyframe_map = tmp_path / "keyframes.parquet"
    pd.DataFrame([
        {"video_id": "v1", "kf_n": 1, "frame_idx": 10, "pts_time": 1.0},
        {"video_id": "v2", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
    ]).to_parquet(keyframe_map)
    source = tmp_path / "visual.json"
    source.write_text(json.dumps({
        "per_query": [{
            "query_id": "q1",
            "retrieved_video_ids_top100": ["v1", "v2"],
            "candidate_trace": {"retrieval_top100": [
                {"video_id": "v1", "frame_idx": 10, "kf_n": 1,
                 "pts_time": 1.0, "base_score": 0.9, "video_rank": 0},
                {"video_id": "v2", "frame_idx": 20, "kf_n": 2,
                 "pts_time": 2.0, "base_score": 0.8, "video_rank": 1},
            ]},
        }],
    }), encoding="utf-8")

    retriever = MaterializedVisualRetriever(source, keyframe_map)
    with pytest.raises(RuntimeError, match="set_query_id"):
        retriever.candidate_pool_contract()
    retriever.set_query_id("q1")
    actual = retriever.candidate_pool_contract()
    expected = candidate_pool_contract([
        {"video_id": "v1", "frame_idx": 10, "kf_n": 1, "video_rank": 0},
        {"video_id": "v2", "frame_idx": 20, "kf_n": 2, "video_rank": 1},
    ], query_id="q1")
    assert actual == expected


def test_ranked_evaluator_rejects_selector_truncated_retrieval_pool():
    with pytest.raises(RuntimeError, match="candidate-pool contract missing"):
        _retrieval_candidate_pool(
            {"candidates": [{"video_id": "v1", "kf_n": 1}]},
            query_id="q1",
        )


def test_materialized_prepare_returns_source_pool_before_selector(tmp_path: Path, monkeypatch):
    keyframe_map = tmp_path / "keyframes.parquet"
    pd.DataFrame([
        {"video_id": "v1", "kf_n": 1, "frame_idx": 10, "pts_time": 1.0},
        {"video_id": "v1", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
        {"video_id": "v2", "kf_n": 1, "frame_idx": 30, "pts_time": 3.0},
    ]).to_parquet(keyframe_map)
    source = tmp_path / "visual.json"
    source.write_text(json.dumps({
        "per_query": [{
            "query_id": "q1",
            "retrieved_video_ids_top100": ["v1", "v2"],
            "candidate_trace": {"retrieval_top100": [
                {"video_id": "v1", "frame_idx": 10, "kf_n": 1,
                 "pts_time": 1.0, "base_score": 0.9, "video_rank": 0},
                {"video_id": "v1", "frame_idx": 20, "kf_n": 2,
                 "pts_time": 2.0, "base_score": 0.8, "video_rank": 0},
                {"video_id": "v2", "frame_idx": 30, "kf_n": 1,
                 "pts_time": 3.0, "base_score": 0.7, "video_rank": 1},
            ]},
        }],
    }), encoding="utf-8")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not decoded; existence is the contract under test")

    from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3

    monkeypatch.setattr(VQAPipelineV3, "_frame_path", lambda self, *_: str(frame))
    retriever = MaterializedVisualRetriever(source, keyframe_map)
    retriever.set_query_id("q1")
    pipeline = VQAPipelineV3(translate=False, kis_retriever=retriever)
    prepared = pipeline.prepare_ranked_candidates(
        "scene", "What is shown?", top_videos=2, frames_per_video=2,
        max_vlm_candidates=100, required_modalities="visual",
        return_candidate_pool=True,
    )

    pool = prepared["_candidate_pool"]
    assert prepared["candidate_pool_count"] == 3
    assert candidate_pool_digest(pool) == retriever.candidate_pool_contract()["digest"]
    assert len(prepared["candidates"]) == len(pool)


def _ranked_parity_fixture(tmp_path: Path, *, mode: str, runtime_video: str) -> dict:
    canonical = tmp_path / f"canonical_{mode}.parquet"
    pd.DataFrame([
        {"video_id": "v1", "kf_n": 1, "frame_idx": 10},
        {"video_id": "v2", "kf_n": 2, "frame_idx": 20},
    ]).to_parquet(canonical)
    source = tmp_path / f"visual_{mode}.json"
    source.write_text(json.dumps({
        "per_query": [{
            "query_id": "q1",
            "candidate_trace": {"retrieval_top100": [
                {"video_id": "v1", "frame_idx": 10, "kf_n": 1,
                 "video_rank": 0},
            ]},
        }],
    }), encoding="utf-8")
    runtime = [{
        "video_id": runtime_video,
        "frame_idx": 10 if runtime_video == "v1" else 20,
        "kf_n": 1 if runtime_video == "v1" else 2,
        "video_rank": 0,
    }]
    contract = candidate_pool_contract(runtime, query_id="q1")
    source_sha = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    return {
        "provenance": {
            "visual_source": {"path": str(source), "sha256": source_sha},
            "canonical_index": {"path": str(canonical)},
            "candidate_pool": {
                "mode": mode,
                "query_digests": {"q1": contract["digest"]},
            },
        },
        "per_query": [{
            "query_id": "q1",
            "candidate_trace": {
                "retrieval_top100": runtime,
                "retrieval_pool_contract": contract,
            },
            "candidate_pool_contract": {
                "retrieval": {
                    "provenance_mode": mode,
                    "count": len(runtime),
                    "digest": contract["digest"],
                },
            },
        }],
    }


def test_visual_only_requires_exact_frozen_source_parity(tmp_path: Path):
    report = _ranked_parity_fixture(
        tmp_path, mode="visual_only", runtime_video="v2",
    )
    findings = []
    result = _check_source_parity(report, findings)
    assert result["mode"] == "visual_only"
    assert result["mismatch_queries"] == 1
    assert any(item["id"] == "S6_candidate_pool_provenance_mismatch" for item in findings)


def test_routed_runtime_uses_self_contract_not_visual_source_frame_parity(tmp_path: Path):
    report = _ranked_parity_fixture(
        tmp_path, mode="routed_runtime", runtime_video="v2",
    )
    findings = []
    result = _check_source_parity(report, findings)
    assert result["mode"] == "routed_runtime"
    assert result["baseline_parity_skipped"] is True
    assert result["mismatch_queries"] == 0
    assert result["self_contract"]["valid"] is True
    assert findings == []
