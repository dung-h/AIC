import pytest

from src.eval.eval_trake_end_to_end import TrakeEvaluatorError, evaluate_trake_records
from src.trake.paired_benchmark import _failure_taxonomy, build_paired_report


def test_end_to_end_separates_video_retrieval_from_alignment_and_zeroes_wrong_video():
    result = evaluate_trake_records(
        [
            {
                "query_id": "q_correct",
                "ground_truth_video_id": "v1",
                "ranked_video_ids": ["v1", "other"],
                "gt_intervals": [[10.0, 12.0], [20.0, 20.0]],
                "predicted_video_id": "v1",
                "pred_times": [11.0, 20.0],
            },
            {
                "query_id": "q_wrong_video",
                "ground_truth_video_id": "v2",
                "ranked_video_ids": ["wrong", "v2"],
                "gt_times": [30.0, 40.0],
                "predicted_video_id": "wrong",
                # This looks close to the GT event times, but must not become
                # an oracle score because the selected video is wrong.
                "pred_times": [30.0, 40.0],
            },
        ]
    )

    assert result["video_retrieval"]["video_r1"] == 0.5
    assert result["video_retrieval"]["video_r5"] == 1.0
    end_to_end = result["event_alignment"]["end_to_end"]
    assert end_to_end["event_hit_2s"] == 0.5
    assert end_to_end["full_sequence_2s"] == 0.5
    assert end_to_end["wrong_video_queries"] == 1
    assert result["event_alignment"]["correct_video_diagnostic"]["event_hit_2s"] == 1.0
    assert result["failure_taxonomy"]["wrong_video_queries"] == 1
    assert result["failure_taxonomy"]["oracle_event_scores_used"] is False
    wrong_row = result["per_query"][1]
    assert wrong_row["alignment"]["event_hit_10s"] == 0
    assert wrong_row["failure"]["stage"] == "wrong_video"


def test_interval_metrics_expand_interval_by_tolerance():
    result = evaluate_trake_records(
        [
            {
                "query_id": "q_interval",
                "ground_truth_video_id": "v1",
                "predicted_video_id": "v1",
                "video_rank": 1,
                "event_intervals": [
                    {"start": 10.0, "end": 12.0},
                    {"start": 20.0, "end": 22.0},
                ],
                "pred_times": [14.0, 25.0],
            }
        ]
    )

    end_to_end = result["event_alignment"]["end_to_end"]
    interval = result["event_alignment"]["interval_metrics"]
    assert end_to_end["event_hit_2s"] == 0.5
    assert end_to_end["event_hit_5s"] == 1.0
    assert interval["available"] is True
    assert interval["event_hit_2s"] == 0.5
    assert interval["event_hit_5s"] == 1.0
    assert result["per_query"][0]["alignment"]["metric_semantics"] == "interval_tolerance"


def test_failure_taxonomy_uses_rank_buckets_not_generic_video_miss():
    rows = []
    for query_id, rank in (("q1", 1), ("q2", 8), ("q3", 80), ("q4", 101)):
        rows.append(
            {
                "query_id": query_id,
                "ground_truth_video_id": f"v-{query_id}",
                "predicted_video_id": f"v-{query_id}",
                "video_rank": rank,
                "gt_times": [1.0, 2.0],
                "pred_times": [1.0, 2.0],
            }
        )
    taxonomy = evaluate_trake_records(rows)["failure_taxonomy"]
    assert taxonomy["rank_buckets"]["r1_5"]["count"] == 1
    assert taxonomy["rank_buckets"]["r6_20"]["count"] == 1
    assert taxonomy["rank_buckets"]["r21_100"]["count"] == 1
    assert taxonomy["rank_buckets"]["gt_not_in_r100"]["count"] == 1
    assert taxonomy["stages"]["candidate_miss"]["count"] == 2
    assert taxonomy["stages"]["video_miss"]["count"] == 1


def test_strict_mode_fails_closed_when_video_rank_is_not_provable():
    with pytest.raises(TrakeEvaluatorError, match="video_rank or ranked_video_ids"):
        evaluate_trake_records(
            [
                {
                    "query_id": "q_missing_rank",
                    "ground_truth_video_id": "v1",
                    "predicted_video_id": "v1",
                    "gt_times": [1.0],
                    "pred_times": [1.0],
                }
            ]
        )


def test_paired_report_exposes_rank_buckets_and_blocks_oracle_event_claims():
    ids = ["q1"]
    rows = [
        {
            "query_id": "q1",
            "split": "holdout",
            "n_events": 2,
            "video_id": "v1",
            "video_rank": 80,
            "failure_taxonomy": {"alignment_miss": True},
            "metric_semantics": "oracle_gt_video_alignment",
        }
    ]
    report = {
        "benchmark_id": "b1",
        "holdout_id": "h1",
        "holdout_query_ids": ids,
        "per_query": rows,
        "metric_semantics": {"event_hit": "oracle_gt_video_alignment"},
        "metrics": {
            "holdout": {
                "video_r1": 0.0,
                "video_r5": 0.0,
                "video_r20": 0.0,
                "video_r50": 0.0,
                "video_r100": 1.0,
            }
        },
    }
    paired = build_paired_report(
        report,
        {**report, "metric_semantics": {"event_hit": "oracle_gt_video_alignment"}},
        manifest={"holdout_query_ids": ids},
    )
    visual_taxonomy = paired["failure_taxonomy"]["visual"]
    assert visual_taxonomy["rank_buckets"]["r21_100"]["count"] == 1
    assert paired["event_alignment_status"] == "blocked"
    assert paired["event_alignment"]["visual"]["official_end_to_end_eligible"] is False
    assert paired["contract"]["wrong_video_event_scores_are_zero"] is True


def test_failure_taxonomy_exposes_tolerance_specific_sequence_loss():
    rows = {
        "q1": {
            "query_id": "q1", "n_events": 3, "video_rank": 1,
            "event_hit_2s": 1, "event_hit_5s": 2, "event_hit_10s": 3,
            "failure_type": "alignment_loss",
            "failure_taxonomy": {
                "video_miss": False, "candidate_miss": False, "alignment_miss": True,
            },
        },
        "q2": {
            "query_id": "q2", "n_events": 3, "video_rank": 4,
            "event_hit_2s": 2, "event_hit_5s": 3, "event_hit_10s": 3,
            "failure_type": "stable_hit",
            "failure_taxonomy": {
                "video_miss": False, "candidate_miss": False, "alignment_miss": True,
            },
        },
    }
    taxonomy = _failure_taxonomy(rows)

    assert taxonomy["alignment_tolerance"]["2s"]["miss_queries"] == 2
    assert taxonomy["alignment_tolerance"]["5s"]["miss_queries"] == 1
    assert taxonomy["alignment_tolerance"]["10s"]["miss_queries"] == 0
    assert taxonomy["source_failure_type_counts"] == {
        "alignment_loss": 1,
        "stable_hit": 1,
    }
