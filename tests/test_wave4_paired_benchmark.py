import pytest

from src.trake.paired_benchmark import PairedBenchmarkError, build_paired_report


def _report(
    ids=("q1", "q2"),
    *,
    final_score=None,
    provenance=None,
    event_counts=None,
    videos=None,
    metrics=None,
    failure_types=None,
):
    event_counts = event_counts or {query_id: 2 for query_id in ids}
    videos = videos or {query_id: f"video-{index}" for index, query_id in enumerate(ids)}
    rows = []
    for index, query_id in enumerate(ids):
        row = {
            "query_id": query_id,
            "split": "holdout",
            "n_events": event_counts[query_id],
            "ground_truth_video_id": videos[query_id],
        }
        if failure_types:
            row["failure_type"] = failure_types[index]
        rows.append(row)
    values = {
        "video_r1": 0.4,
        "video_r5": 0.6,
        "video_r20": 0.8,
        "video_r50": 0.9,
        "video_r100": 1.0,
        "event_hit_2s": 0.2,
        "event_hit_5s": 0.4,
        "event_hit_10s": 0.6,
        "full_sequence": 0.3,
        "temporal_order_validity": 0.75,
    }
    if metrics is not None:
        values = metrics
    if final_score is not None:
        values["final_score"] = final_score
    report = {
        "benchmark_id": "bench-v4",
        "holdout_id": "holdout-v4",
        "holdout_query_ids": list(ids),
        "per_query": rows,
        "holdout_metrics": values,
    }
    if provenance is not None:
        report["provenance"] = provenance
    return report


def test_three_way_pair_checks_identity_extracts_metrics_and_selects_best_path():
    visual = _report(final_score=0.70, failure_types=["video_retrieval_loss", "stable_hit"])
    asr = _report(final_score=0.72, failure_types=["candidate_missing", "stable_hit"])
    multimodal = _report(final_score=0.82, failure_types=["alignment_loss", "stable_hit"])

    result = build_paired_report(
        visual,
        asr,
        multimodal,
        manifest={
            "benchmark_id": "bench-v4",
            "holdout_id": "holdout-v4",
            "holdout_query_ids": ["q1", "q2"],
        },
    )

    assert result["paths"] == ["visual", "asr", "multimodal"]
    assert result["selection"]["selected_path"] == "multimodal"
    assert result["official_score_status"] == "available"
    assert result["visual_metrics"]["video_r1"] == 0.4
    assert result["visual_metrics"]["temporal_order_validity"] == 0.75
    assert result["multimodal_metrics"]["final_score"] == 0.82
    assert result["failure_taxonomy"]["visual"]["status"] == "available"
    assert result["failure_taxonomy"]["visual"]["counts"]["video_miss"] == 1
    assert result["failure_taxonomy"]["asr"]["counts"]["candidate_miss"] == 1
    assert result["failure_taxonomy"]["multimodal"]["counts"]["alignment_miss"] == 1


def test_visual_is_preferred_when_within_five_percentage_points_of_best():
    visual = _report(final_score=0.80)
    asr = _report(final_score=0.84)
    multimodal = _report(final_score=0.849)

    result = build_paired_report(visual, asr, multimodal)

    assert result["selection"]["selected_path"] == "visual"
    assert result["selection"]["final_score_gap_to_best"]["visual"] == pytest.approx(0.049)


def test_final_score_can_be_derived_from_complete_holdout_video_ranks_but_is_labelled():
    ranks = {
        "video_r1": 0.2,
        "video_r5": 0.4,
        "video_r20": 0.6,
        "video_r50": 0.8,
        "video_r100": 1.0,
    }
    result = build_paired_report(_report(metrics=ranks), _report(metrics=ranks))

    assert result["visual_metrics"]["final_score"] == pytest.approx(0.6)
    assert result["metric_status"]["visual"]["final_score_source"] == "derived_from_holdout_video_ranks"
    assert result["official_score_status"] == "derived_official_style"
    assert result["selection"]["selected_path"] == "visual"


def test_proxy_score_is_visible_for_audit_but_cannot_select_a_production_path():
    visual = _report(final_score=0.99, provenance="synthetic proxy diagnostic")
    asr = _report(final_score=0.70)

    result = build_paired_report(visual, asr)

    assert result["selection"]["status"] == "blocked"
    assert result["selection"]["selected_path"] is None
    assert result["visual_metrics"]["final_score"] is None
    assert result["metric_status"]["visual"]["reported_final_score"] == 0.99
    assert any("proxy" in reason for reason in result["selection"]["blocked_reasons"])


def test_partial_failure_taxonomy_does_not_invent_missing_fields_or_rates():
    report = _report()
    report["per_query"][0]["video_miss"] = True
    report["per_query"][1]["video_miss"] = False
    result = build_paired_report(report, _report())

    taxonomy = result["failure_taxonomy"]["visual"]
    assert taxonomy["status"] == "partial"
    assert taxonomy["counts"]["video_miss"] == 1
    assert taxonomy["rates"]["video_miss"] == 0.5
    assert taxonomy["counts"]["candidate_miss"] is None
    assert taxonomy["rates"]["candidate_miss"] is None


def test_exact_parity_rejects_extra_holdout_row_and_multimodal_event_mismatch():
    extra = _report()
    extra["per_query"].append({
        "query_id": "q-extra",
        "split": "holdout",
        "n_events": 2,
        "ground_truth_video_id": "video-extra",
    })
    with pytest.raises(PairedBenchmarkError, match="not declared"):
        build_paired_report(extra, _report())

    multimodal = _report(event_counts={"q1": 2, "q2": 3})
    with pytest.raises(PairedBenchmarkError, match="event count mismatch"):
        build_paired_report(_report(), _report(), multimodal)


def test_missing_metrics_are_reported_explicitly_without_fake_score():
    visual = _report(metrics={"video_r20": 0.5})
    asr = _report(metrics={"video_r20": 0.6})

    result = build_paired_report(visual, asr)

    assert result["selection"]["status"] == "blocked"
    assert "final_score" in result["metric_status"]["visual"]["missing_metrics"]
    assert result["metric_status"]["visual"]["blocked_reasons"]
