"""Regression tests for Q&A evaluator metric semantics.

These tests intentionally use small synthetic records.  They do not load a
retrieval model or VLM; the purpose is to prevent evaluator-only leakage and
mislabeling from returning to benchmark reports.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.eval import benchmark_qna_global_retrieval_v3 as global_eval
from src.eval import benchmark_qna_ranked_offline as ranked_eval
from src.eval.qna_answer_metrics import normalize_answer


def test_global_metrics_separate_preselector_postselector_and_rescue():
    records = [
        {
            "arm": "baseline", "video_rank": 30, "baseline_video_rank": 30,
            "frame_hit": False, "preselector_frame_hit": False,
            "postselector_frame_hit": False, "latency_ms": 1.0,
        },
        {
            "arm": "routed", "video_rank": 10, "baseline_video_rank": 30,
            "frame_hit": False, "preselector_frame_hit": True,
            "postselector_frame_hit": False, "video_rescue_top20": True,
            "video_rescue_top100": True, "latency_ms": 1.0,
        },
    ]

    metrics = global_eval._metrics(records, "routed")

    assert metrics["frame_recall_pre_selector"] == 1.0
    assert metrics["frame_recall_post_selector_budget_12"] == 0.0
    assert metrics["video_rescue_top20"] == 1.0
    assert metrics["video_rescue_top100"] == 1.0
    # The historical candidate_rescue field intentionally remains the old
    # top-100/None definition; it is not the corrected top-20 rescue metric.
    assert metrics["candidate_rescue"] == 0.0


def test_oracle_report_loader_uses_independent_exact_match(tmp_path: Path):
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"rows": [{"annotation_id": "q1", "exact_match": True}]}),
                    encoding="utf-8")

    assert ranked_eval._load_oracle_answers(path) == {"q1": True}


def test_ranked_evaluator_does_not_call_retrieved_answer_an_oracle(tmp_path: Path):
    input_path = tmp_path / "qna.jsonl"
    input_path.write_text(json.dumps({
        "query_id": "q1", "video_id": "v50", "query": "a scene",
        "question": "What is visible?", "answer": "A fork.",
        "status": "valid", "split": "dev", "question_type": "place",
        "acceptable_kf_n": "1",
    }) + "\n", encoding="utf-8")
    keyframes = pd.DataFrame([{"video_id": "v50", "kf_n": 1, "frame_idx": 500}])

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def vqa_ranked(self, *args, **kwargs):
            # The retrieved GT frame is deliberately answered incorrectly.
            return {
                "answers": [{"video_id": "v50", "frame_id": 500, "answer": "A table."}],
                "candidate_count": 1, "vlm_candidate_count": 1,
            }

    with patch.object(ranked_eval.pd, "read_parquet", return_value=keyframes), \
            patch.object(ranked_eval, "HCMAIPipeline", FakePipeline):
        result = ranked_eval.evaluate(
            input_path, staged=False, max_answers=1,
            oracle_answers={"q1": True},
        )

    row = result["per_query"][0]
    metrics = result["metrics"]["all"]
    assert row["retrieval_frame_hit"] is False
    assert row["legacy_retrieval_frame_hit"] is False
    assert row["retrieved_gt_frame_answer_match_legacy"] is False
    assert row["oracle_gt_frame_answer_match"] is True
    assert row["gt_frame_answer_match"] is True
    assert metrics["answer_accuracy_on_gt_frame"] == 1.0
    assert metrics["answer_accuracy_retrieved_gt_frame_legacy"] == 0.0


def test_ranked_evaluator_marks_oracle_unavailable_instead_of_zero(tmp_path: Path):
    input_path = tmp_path / "qna.jsonl"
    input_path.write_text(json.dumps({
        "query_id": "q1", "video_id": "v50", "query": "a scene",
        "question": "What is visible?", "answer": "A fork.",
        "status": "valid", "split": "dev", "question_type": "place",
        "acceptable_kf_n": "1",
    }) + "\n", encoding="utf-8")
    keyframes = pd.DataFrame([{"video_id": "v50", "kf_n": 1, "frame_idx": 500}])

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def vqa_ranked(self, *args, **kwargs):
            return {"answers": [], "candidate_count": 0, "vlm_candidate_count": 0}

    with patch.object(ranked_eval.pd, "read_parquet", return_value=keyframes), \
            patch.object(ranked_eval, "HCMAIPipeline", FakePipeline):
        result = ranked_eval.evaluate(input_path, staged=False, max_answers=1)

    row = result["per_query"][0]
    assert row["oracle_gt_frame_answer_available"] is False
    assert result["metrics"]["all"]["answer_accuracy_on_gt_frame"] is None
    assert "not_run" in result["oracle_gt_frame_answer_status"]


def test_grounded_answer_requires_gt_video_and_accepted_frame():
    row = type("Row", (), {
        "video_id": "v50",
        "answer": "Red.",
    })()
    accepted = {500}
    assert ranked_eval._grounded_answer_match(
        row, {"video_id": "wrong", "frame_id": 500, "answer": "red"}, accepted
    ) is False
    assert ranked_eval._grounded_answer_match(
        row, {"video_id": "v50", "frame_id": 501, "answer": "red"}, accepted
    ) is False
    assert ranked_eval._grounded_answer_match(
        row, {"video_id": "v50", "frame_id": 500, "answer": "red"}, accepted
    ) is True


def test_shared_normalizer_handles_punctuation_articles_and_numbers():
    assert normalize_answer("Two.") == normalize_answer("2")
    assert normalize_answer("A face mask.") == normalize_answer("Face mask")


def test_oracle_loader_prefers_normalized_and_skips_non_frame_rows(tmp_path: Path):
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps({"rows": [
        {"annotation_id": "visual", "exact_match": False, "normalized_exact_match": True},
        {"annotation_id": "spoken", "exact_match": True, "normalized_exact_match": True,
         "frame_oracle_eligible": False},
    ]}), encoding="utf-8")
    assert ranked_eval._load_oracle_answers(path) == {"visual": True}
