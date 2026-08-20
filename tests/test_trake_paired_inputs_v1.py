import pandas as pd

from src.eval.build_trake_paired_inputs_v1 import (
    _failure_flags,
    _standardize_visual_report,
    _summary,
)


def _queryset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q_dev",
                "video_id": "v_dev",
                "split": "dev",
                "step": 0,
                "caption": "event one",
                "pts_time": 1.0,
            },
            {
                "query_id": "q_dev",
                "video_id": "v_dev",
                "split": "dev",
                "step": 1,
                "caption": "event two",
                "pts_time": 2.0,
            },
            {
                "query_id": "q_holdout",
                "video_id": "v_holdout",
                "split": "holdout",
                "step": 0,
                "caption": "event one",
                "pts_time": 3.0,
            },
            {
                "query_id": "q_holdout",
                "video_id": "v_holdout",
                "split": "holdout",
                "step": 1,
                "caption": "event two",
                "pts_time": 4.0,
            },
        ]
    )


def test_standardize_visual_report_deduplicates_event_level_holdout_ids():
    result = _standardize_visual_report(
        {
            "per_query": {
                "0.0": [
                    {
                        "query_id": "q_dev",
                        "frozen_lambda0_video_rank": 1,
                        "pred_times": [1.0, 2.0],
                        "oracle_video_event_2s": 2,
                        "oracle_video_event_5s": 2,
                        "oracle_video_event_10s": 2,
                    },
                    {
                        "query_id": "q_holdout",
                        "frozen_lambda0_video_rank": 3,
                        "pred_times": [3.0, 4.0],
                        "oracle_video_event_2s": 1,
                        "oracle_video_event_5s": 2,
                        "oracle_video_event_10s": 2,
                    },
                ]
            }
        },
        _queryset(),
        benchmark_id="b1",
        holdout_id="h1",
    )

    assert result["holdout_query_ids"] == ["q_holdout"]
    assert result["dev_query_ids"] == ["q_dev"]
    assert result["metrics"]["holdout"]["events"] == 2
    assert result["metrics"]["holdout"]["event_hit_2s"] == 0.5
    assert result["per_query"][1]["failure_taxonomy"]["alignment_miss"] is True


def test_failure_flags_prioritize_video_miss_then_candidate_miss():
    assert _failure_flags(rank=None, path_available=True, event_hit_2s=2, n_events=2) == {
        "video_miss": True,
        "candidate_miss": False,
        "alignment_miss": False,
    }
    assert _failure_flags(rank=4, path_available=False, event_hit_2s=0, n_events=2) == {
        "video_miss": False,
        "candidate_miss": True,
        "alignment_miss": False,
    }


def test_summary_uses_event_weighted_hits_and_query_weighted_sequence():
    result = _summary(
        [
            {
                "n_events": 2,
                "video_r1": True,
                "video_r5": True,
                "video_r20": True,
                "video_r50": True,
                "video_r100": True,
                "event_hit_2s": 1,
                "event_hit_5s": 2,
                "event_hit_10s": 2,
                "full_sequence": False,
                "temporal_order_validity": True,
            },
            {
                "n_events": 1,
                "video_r1": False,
                "video_r5": True,
                "video_r20": True,
                "video_r50": True,
                "video_r100": True,
                "event_hit_2s": 1,
                "event_hit_5s": 1,
                "event_hit_10s": 1,
                "full_sequence": True,
                "temporal_order_validity": True,
            },
        ]
    )

    assert result["events"] == 3
    assert result["event_hit_2s"] == 2 / 3
    assert result["full_sequence"] == 0.5
    assert result["final_score"] == 0.9
