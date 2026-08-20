from __future__ import annotations

from src.eval.run_architecture_benchmark import report_to_json, run_architecture_benchmark


def test_qna_metrics_are_reproducible_with_injected_latency_and_per_type_failures() -> None:
    rows = [
        {
            "task": "qna",
            "query_id": "q1",
            "question_type": "screen_text",
            "ground_truth_video_id": "V1",
            "relevant_frame_keys": [["V1", 10]],
            "ground_truth_answer": "25 độ",
        },
        {
            "task": "qna",
            "query_id": "q2",
            "question_type": "spoken_fact",
            "ground_truth_video_id": "V2",
            "relevant_frame_keys": [["V2", 20]],
            "ground_truth_answer": "Nha Trang",
        },
    ]

    def retrieve(row):
        if row["query_id"] == "q1":
            return {
                "latency_ms": 10,
                "baseline_candidates": [{"video_id": "V1", "frame_idx": 9}],
                "candidates": [
                    {"video_id": "V9", "frame_idx": 99},
                    {"video_id": "V1", "frame_idx": 10},
                ],
                "selected": [{"video_id": "V1", "frame_idx": 10}],
            }
        return {
            "latency_ms": 30,
            "baseline_candidates": [{"video_id": "V2", "frame_idx": 20}],
            "candidates": [{"video_id": "V2", "frame_idx": 20}],
            "selected": [],
        }

    def answer(row, _retrieval):
        return {
            "latency_ms": 5 if row["query_id"] == "q1" else 7,
            "answers": [
                {
                    "video_id": row["ground_truth_video_id"],
                    "frame_idx": 10 if row["query_id"] == "q1" else 20,
                    "answer": "25 độ" if row["query_id"] == "q1" else "wrong",
                }
            ],
        }

    report = run_architecture_benchmark(
        rows,
        retriever=retrieve,
        answerer=answer,
        config={"selector_budget": 1},
        model_provenance={"answer": "fake-local"},
        index_provenance={"visual": "synthetic-v1"},
        split_provenance={"split": "holdout"},
    )

    assert report["schema_version"] == "hcmai.architecture_benchmark_v1"
    assert report["status"] == "complete"
    qna = report["qna"]
    assert qna["metrics"]["video_r1"]["value"] == 0.5
    assert qna["metrics"]["video_r100"]["value"] == 1.0
    assert qna["metrics"]["frame_recall"]["value"] == 1.0
    assert qna["metrics"]["candidate_rescue"]["value"] == 0.5
    assert qna["metrics"]["selector_recall"]["value"] == 0.5
    assert qna["metrics"]["answer_exact"]["value"] == 0.5
    assert qna["metrics"]["answer_grounded"]["value"] == 0.5
    assert qna["metrics"]["latency"]["p50_ms"] == 26.0
    assert qna["metrics"]["latency"]["p95_ms"] == 35.9
    assert qna["per_question_type"]["screen_text"]["metrics"]["answer_exact"]["value"] == 1.0
    assert qna["per_question_type"]["spoken_fact"]["failure_taxonomy"]["counts"]["selector_miss"] == 0
    assert qna["failure_taxonomy"]["counts"]["answer_miss"] == 1

    # JSON is stable, and the callable objects never enter the artifact.
    assert report_to_json(report) == report_to_json(report)
    assert "function" not in report_to_json(report)


def test_qna_missing_oracle_blocks_without_fabricating_zero() -> None:
    report = run_architecture_benchmark(
        [{"task": "qna", "query_id": "q1", "question_type": "action", "question": "what?"}],
        retriever=lambda _row: [{"video_id": "V1", "kf_n": 7}],
    )
    assert report["status"] == "blocked"
    qna = report["qna"]
    assert qna["metrics"]["video_r1"]["value"] is None
    assert qna["metrics"]["frame_recall"]["value"] is None
    assert qna["metrics"]["answer_exact"]["value"] is None
    assert any("oracle" in reason for reason in report["blocked_reasons"])
    assert qna["per_query"][0]["failure_types"] == []


def test_trake_metrics_use_timestamped_oracle_and_explicit_final_only() -> None:
    rows = [
        {
            "task": "trake",
            "query_id": "t1",
            "question_type": "temporal_relation",
            "ground_truth_video_id": "V1",
            "ground_truth_events": [{"timestamp": 10.0}, {"timestamp": 20.0}],
        },
        {
            "task": "trake",
            "query_id": "t2",
            "question_type": "temporal_relation",
            "ground_truth_video_id": "V2",
            "ground_truth_events": [{"timestamp": 5.0}, {"timestamp": 8.0}],
        },
    ]

    def solve(row):
        if row["query_id"] == "t1":
            return {
                "final_score": 0.7,
                "results": [{"video_id": "V1", "frame_ids": [100, 200], "timestamps": [10.5, 19.5]}],
            }
        return {"results": [{"video_id": "V2", "frame_ids": [80, 70], "timestamps": [5.0, 8.0]}]}

    report = run_architecture_benchmark(rows, trake=solve)
    assert report["status"] == "complete"
    trake = report["trake"]
    assert trake["metrics"]["video_r1"]["value"] == 1.0
    assert trake["metrics"]["event_hit_2s"]["value"] == 1.0
    assert trake["metrics"]["full_sequence"]["value"] == 0.5
    assert trake["metrics"]["temporal_order"]["value"] == 0.5
    assert trake["metrics"]["final_score"]["value"] == 0.7
    assert trake["metrics"]["final_score"]["source"] == "explicit_injected_output"
    assert trake["official_score_status"] == "available_explicit"
    assert trake["failure_taxonomy"]["counts"]["alignment_miss"] == 0


def test_trake_missing_timestamp_oracle_blocks_event_scores_and_does_not_derive_final() -> None:
    report = run_architecture_benchmark(
        [
            {
                "task": "trake",
                "query_id": "t1",
                "ground_truth_video_id": "V1",
                "ground_truth_events": [{"frame_idx": 10}, {"frame_idx": 20}],
            }
        ],
        trake=lambda _row: [{"video_id": "V1", "frame_ids": [10, 20]}],
    )
    assert report["status"] == "blocked"
    trake = report["trake"]
    assert trake["metrics"]["video_r1"]["value"] == 1.0
    assert trake["metrics"]["event_hit_5s"]["value"] is None
    assert trake["metrics"]["final_score"]["value"] is None
    assert trake["metrics"]["final_score"]["source"] is None
    assert any("timestamped event oracle" in reason for reason in report["blocked_reasons"])


def test_mapping_input_and_empty_benchmark_are_explicit() -> None:
    report = run_architecture_benchmark({"qna": [], "trake": []})
    assert report["status"] == "blocked"
    assert report["qna"]["status"] == "not_run"
    assert report["trake"]["status"] == "not_run"
    assert report["protocol"]["network_called"] is False
