import json

import pandas as pd

from src.eval import benchmark_qna_ranked_offline as evaluator
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


def _write_fixture(tmp_path):
    path = tmp_path / "qna.jsonl"
    row = {
        "query_id": "q1",
        "video_id": "v50",
        "query": "a scene",
        "question": "What is visible?",
        "answer": "A fork.",
        "status": "valid",
        "split": "dev",
        "question_type": "place",
        "required_modalities": "visual,asr",
        "acceptable_kf_n": "1",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _patch_keyframes(monkeypatch):
    keyframes = pd.DataFrame([
        {"video_id": "v50", "kf_n": 1, "frame_idx": 500},
        {"video_id": "v1", "kf_n": 1, "frame_idx": 1},
    ])

    def read_parquet(path, *args, **kwargs):
        return keyframes

    monkeypatch.setattr(evaluator.pd, "read_parquet", read_parquet)


def test_max_answers_truncates_after_pipeline_rerank(monkeypatch, tmp_path):
    _patch_keyframes(monkeypatch)
    calls = []

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def vqa_ranked(self, *args, **kwargs):
            calls.append(kwargs["max_answers"])
            return {
                "answers": [
                    {"video_id": "v50", "frame_id": 500, "answer": "A fork."},
                    {"video_id": "v1", "frame_id": 1, "answer": "A table."},
                ],
                "candidate_count": 2,
                "vlm_candidate_count": 2,
            }

    monkeypatch.setattr(evaluator, "HCMAIPipeline", FakePipeline)
    result = evaluator.evaluate(_write_fixture(tmp_path), staged=False, max_answers=1)
    row = result["per_query"][0]

    assert calls == [20]
    assert row["answers"] == 1
    assert row["answers_before_output_truncation"] == 2
    assert len(row["answer_trace"]) == 2
    assert row["top_answer_match"] is True


def test_staged_evaluation_has_independent_top100_and_answer_stage(monkeypatch, tmp_path):
    _patch_keyframes(monkeypatch)
    prepare_calls = []
    answer_calls = []

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def prepare_ranked_candidates(self, query, question, **kwargs):
            top_videos = kwargs["top_videos"]
            prepare_calls.append(top_videos)
            if top_videos >= 100:
                return {
                    "retrieved_video_ids": [f"v{i}" for i in range(1, 101)],
                    "candidates": [{
                        "video_id": "v50", "frame_idx": 500, "kf_n": 1,
                        "pts_time": 1.0, "base_score": 0.9,
                        "video_rank": 50, "source": "visual",
                    }],
                    # Explicit retrieval-pool seam: ``candidates`` is the
                    # bounded/answer-facing field, while retrieval metrics
                    # must consume this untruncated pool contract.
                    "_candidate_pool": [{
                        "video_id": "v50", "frame_idx": 500, "kf_n": 1,
                        "pts_time": 1.0, "base_score": 0.9,
                        "video_rank": 50, "source": "visual",
                    }, {
                        "video_id": "v1", "frame_idx": 1, "kf_n": 1,
                        "pts_time": 1.0, "base_score": 0.8,
                        "video_rank": 1, "source": "visual",
                    }],
                    "candidate_pool_count": 2,
                    "vlm_candidate_count": 1,
                }
            return {
                "retrieved_video_ids": [f"v{i}" for i in range(1, 21)],
                "candidates": [{
                    "video_id": "v1", "frame_idx": 1, "kf_n": 1,
                    "pts_time": 1.0, "base_score": 0.9,
                    "video_rank": 1, "source": "visual",
                }],
                "vlm_candidate_count": 1,
            }

        def release_retrieval_models(self):
            pass

        def answer_ranked_candidates(self, prepared, **kwargs):
            answer_calls.append(kwargs["max_answers"])
            assert prepared["required_sources"] == ["asr"]
            return {
                "answers": [{"video_id": "v1", "frame_id": 1, "answer": "A table."}],
                "candidate_count": 1,
                "vlm_candidate_count": 1,
            }

    import src.pipelines.vqa_pipeline_v3 as v3
    monkeypatch.setattr(v3, "VQAPipelineV3", FakePipeline)
    result = evaluator.evaluate(
        _write_fixture(tmp_path), staged=True, max_answers=1,
        allow_missing_modalities=True,
    )
    row = result["per_query"][0]
    metrics = result["metrics"]["all"]

    assert prepare_calls == [100, 20]
    assert answer_calls == [20]
    assert row["retrieval_top100_video_rank"] == 50
    assert row["answer_stage_video_rank"] is None
    assert row["retrieval_top100_frame_hit"] is True
    assert row["answer_stage_frame_hit"] is False
    assert metrics["video_r20"] == 0.0
    assert metrics["video_r100"] == 1.0
    assert metrics["frame_recall"] == 0.0
    assert metrics["retrieval_frame_recall_top100"] == 1.0
    assert metrics["answer_stage_frame_recall"] == 0.0
    assert row["retrieval_latency_ms"] is not None
    assert row["answer_latency_ms"] is not None
    assert row["end_to_end_latency_ms"] >= row["answer_latency_ms"]


def test_normalized_match_is_diagnostic_only():
    row = type("Row", (), {"answer": "A fork."})()
    assert evaluator._answer_match(row, "fork") is False
    assert evaluator._normalized_answer_match_diagnostic(row, "fork") is True


def test_materialized_retriever_release_does_not_require_torch():
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = object()
    pipeline.release_retrieval_models()


def test_routed_answer_rejects_missing_modality_evidence(monkeypatch):
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = object()
    pipeline.answer_provider = None

    def missing_evidence(*args, **kwargs):
        raise ValueError("modality evidence requires asr and/or ocr")

    monkeypatch.setattr(pipeline, "_build_evidence_packet", missing_evidence)
    result = pipeline.answer_ranked_candidates({
        "query": "a scene",
        "question": "what was said?",
        "route_active": True,
        "evidence_fusion": True,
        "required_sources": ("asr",),
        "candidate_state": "candidate_available",
        "candidates": [{
            "video_id": "v1", "frame_idx": 10, "kf_n": 1, "pts_time": 1.0,
            "base_score": 0.5, "video_rank": 1, "frame_path": "/missing.jpg",
            "source": "visual",
        }],
    }, max_answers=1)

    assert result["answers"] == []
    assert result["status"] == "no_valid_local_answer"
    assert result["answer_trace"][0]["status"] == "rejected_missing_modality_evidence"


def test_routed_answer_rejects_visual_only_packet_before_vlm(monkeypatch):
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = object()
    pipeline.answer_provider = None

    monkeypatch.setattr(
        pipeline,
        "_build_evidence_packet",
        lambda *args, **kwargs: {"sources": ["visual"], "asr_chunks": [], "ocr_text": []},
    )
    result = pipeline.answer_ranked_candidates({
        "query": "a scene",
        "question": "what was said?",
        "route_active": True,
        "evidence_fusion": True,
        "required_sources": ("asr",),
        "candidate_state": "candidate_available",
        "candidates": [{
            "video_id": "v1", "frame_idx": 10, "kf_n": 1, "pts_time": 1.0,
            "base_score": 0.5, "video_rank": 1, "frame_path": "/missing.jpg",
            "source": "visual",
        }],
    }, max_answers=1)

    assert result["answers"] == []
    assert result["answer_trace"][0]["status"] == "rejected_missing_modality_evidence"
