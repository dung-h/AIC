"""Regression tests for the ranked Q&A candidate-pool protocol."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.eval.benchmark_qna_ranked_offline as ranked


def _candidate(video: str, kf: int) -> dict:
    return {
        "video_id": video,
        "kf_n": kf,
        "frame_idx": kf * 10,
        "base_score": 1.0,
    }


def test_retrieval_trace_uses_untruncated_pool_not_selector_output():
    selector_output = [_candidate("v1", 1), _candidate("v2", 1)]
    materialized_pool = [_candidate("v1", 1), _candidate("v1", 2),
                         _candidate("v2", 1), _candidate("v2", 2)]

    prepared = {
        "candidates": selector_output,
        "_candidate_pool": materialized_pool,
        "candidate_pool_count": len(materialized_pool),
    }

    assert ranked._retrieval_candidate_pool(prepared, query_id="q1") == materialized_pool
    assert ranked._retrieval_candidate_pool(prepared, query_id="q1") != selector_output


def test_retrieval_pool_contract_fails_closed_when_private_pool_is_missing():
    with pytest.raises(RuntimeError, match="return_candidate_pool=True"):
        ranked._retrieval_candidate_pool({"candidates": [_candidate("v1", 1)]}, query_id="q1")


def test_retrieval_pool_contract_fails_closed_on_declared_count_mismatch():
    with pytest.raises(RuntimeError, match="declared=1, materialized=2"):
        ranked._retrieval_candidate_pool(
            {
                "_candidate_pool": [_candidate("v1", 1), _candidate("v1", 2)],
                "candidate_pool_count": 1,
            },
            query_id="q1",
        )


def test_retrieval_pool_contract_fails_closed_when_declared_count_is_missing():
    with pytest.raises(RuntimeError, match="candidate_pool_count is missing"):
        ranked._retrieval_candidate_pool(
            {"_candidate_pool": [_candidate("v1", 1)]},
            query_id="q1",
        )


def test_retrieval_pool_contract_uses_explicit_top_k_prefix_for_large_pool():
    pool = [_candidate("v1", index) for index in range(3)]
    assert ranked._retrieval_candidate_pool(
        {"_candidate_pool": pool, "candidate_pool_count": len(pool)},
        query_id="q1",
        requested_limit=2,
    ) == pool[:2]


def test_ranked_evaluator_traces_the_pool_used_by_retrieval(monkeypatch, tmp_path: Path):
    """The integration seam must request and trace the pre-selector pool."""
    source_path = Path("results/exp_qna_p0_p1_v31_holdout_visual.json")
    if not source_path.is_file():
        pytest.skip("frozen visual source report is not available")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_row = source["per_query"][0]
    source_pool = source_row["candidate_trace"]["retrieval_top100"]
    video_ids = source_row["retrieved_video_ids_top100"]

    annotation = tmp_path / "one_query.jsonl"
    annotation.write_text(
        json.dumps({
            "annotation_id": source_row["query_id"],
            "video_id": source_row["gt_video"],
            "query": "scene",
            "question": "What is shown?",
            "answer": "answer",
            "status": "valid",
            "split": "holdout",
            "question_type": "action",
            "acceptable_kf_n": str(source_pool[0]["kf_n"]),
        }) + "\n",
        encoding="utf-8",
    )

    class FakePipeline:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def prepare_ranked_candidates(self, *args, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs.get("return_candidate_pool"):
                return {
                    "_candidate_pool": list(source_pool),
                    "candidate_pool_count": len(source_pool),
                    "candidates": list(source_pool[:2]),
                    "retrieved_video_ids": list(video_ids),
                }
            return {
                "candidates": list(source_pool[:2]),
                "retrieved_video_ids": list(video_ids[:20]),
            }

        def release_retrieval_models(self):
            return None

        def answer_ranked_candidates(self, prepared, **kwargs):
            first = prepared["candidates"][0]
            return {
                "answers": [{
                    "video_id": first["video_id"],
                    "frame_id": first["frame_idx"],
                    "answer": "answer",
                }],
                "answer_trace": [],
                "candidate_count": len(prepared["candidates"]),
                "vlm_candidate_count": len(prepared["candidates"]),
            }

    monkeypatch.setattr("src.pipelines.vqa_pipeline_v3.VQAPipelineV3", FakePipeline)
    report = ranked.evaluate(
        annotation,
        staged=True,
        max_vlm_candidates=12,
        max_answers=1,
        visual_candidates=source_path,
    )

    assert any(call.get("return_candidate_pool") is True for call in FakePipeline.calls)
    assert sum(call.get("return_candidate_pool") is True for call in FakePipeline.calls) == 1
    row = report["per_query"][0]
    assert [(item["video_id"], item["kf_n"])
            for item in row["candidate_trace"]["retrieval_top100"]] == [
                (item["video_id"], item["kf_n"]) for item in source_pool
            ]
    assert row["candidate_pool_contract"]["retrieval"]["count"] == len(source_pool)
