import json
import sys

import pytest

from src.eval.run_trake_paired_benchmark import main as run_paired_main
from src.trake.paired_benchmark import PairedBenchmarkError, build_paired_report


def _report(ids, *, benchmark_id="b1", holdout_id="h1", final_score=None):
    rows = [
        {"query_id": query_id, "split": "holdout", "n_events": 2, "video_id": f"video-{i}"}
        for i, query_id in enumerate(ids)
    ]
    metrics = {"video_r20": 0.5}
    if final_score is not None:
        metrics["final_score"] = final_score
    return {
        "benchmark_id": benchmark_id,
        "holdout_id": holdout_id,
        "holdout_query_ids": list(ids),
        "per_query": rows,
        "metrics": metrics,
    }


def test_paired_runner_requires_same_explicit_holdout_and_keeps_missing_score_blocked():
    ids = ["q1", "q2"]
    report = build_paired_report(
        _report(ids),
        _report(ids),
        manifest={"holdout_query_ids": ids},
    )
    assert report["status"] == "paired"
    assert report["holdout_queries"] == 2
    assert report["selection"]["status"] == "blocked"
    assert report["official_score_status"] == "blocked"


def test_paired_runner_rejects_query_id_mismatch():
    with pytest.raises(PairedBenchmarkError, match="same holdout query IDs"):
        build_paired_report(
            _report(["q1", "q2"]),
            _report(["q1", "q3"]),
            manifest={"holdout_query_ids": ["q1", "q2"]},
        )


def test_paired_runner_rejects_benchmark_or_holdout_mismatch():
    with pytest.raises(PairedBenchmarkError, match="benchmark_id/holdout_id"):
        build_paired_report(_report(["q1"]), _report(["q1"], holdout_id="h2"))


def test_paired_runner_rejects_manifest_without_explicit_holdout_ids():
    with pytest.raises(PairedBenchmarkError, match="explicitly expose"):
        build_paired_report(
            _report(["q1"]),
            _report(["q1"]),
            manifest={"eligible_sources": [{"query_ids": ["q1"]}]},
        )


def test_paired_runner_only_selects_with_two_explicit_final_scores():
    report = build_paired_report(
        _report(["q1"], final_score=0.70),
        _report(["q1"], final_score=0.69),
        manifest={"holdout_query_ids": ["q1"]},
    )
    assert report["selection"]["status"] == "selected"
    assert report["selection"]["selected_path"] == "visual"


def test_legacy_reports_can_use_explicit_trusted_pair_metadata():
    ids = ["q1", "q2"]
    visual = _report(ids, final_score=0.70)
    asr = _report(ids, final_score=0.69)
    visual.pop("benchmark_id")
    visual.pop("holdout_id")
    asr.pop("benchmark_id")
    asr.pop("holdout_id")

    result = build_paired_report(
        visual,
        asr,
        manifest={"holdout_query_ids": ids},
        benchmark_id="b1",
        holdout_id="h1",
    )

    assert result["status"] == "paired"
    assert result["benchmark_id"] == "b1"
    assert result["holdout_id"] == "h1"
    assert result["selection"]["status"] == "selected"


def test_trusted_pair_metadata_rejects_genuine_report_mismatch():
    with pytest.raises(PairedBenchmarkError, match="benchmark_id does not match"):
        build_paired_report(
            _report(["q1"], benchmark_id="wrong"),
            _report(["q1"]),
            benchmark_id="b1",
            holdout_id="h1",
        )


def test_missing_explicit_scores_remains_blocked_after_metadata_is_bound():
    ids = ["q1", "q2"]
    visual = _report(ids)
    asr = _report(ids)
    visual.pop("benchmark_id")
    visual.pop("holdout_id")
    asr.pop("benchmark_id")
    asr.pop("holdout_id")

    result = build_paired_report(
        visual,
        asr,
        manifest={
            "benchmark_id": "b1",
            "holdout_id": "h1",
            "holdout_query_ids": ids,
        },
    )

    assert result["status"] == "paired"
    assert result["selection"]["status"] == "blocked"
    assert result["official_score_status"] == "blocked"


def test_missing_metadata_exposes_precise_fail_closed_diagnostic():
    visual = _report(["q1"])
    visual.pop("benchmark_id")
    visual.pop("holdout_id")

    with pytest.raises(PairedBenchmarkError) as caught:
        build_paired_report(visual, _report(["q1"]))

    error = caught.value.as_dict()
    assert error["code"] == "missing_paired_metadata"
    assert error["report"] == "visual"
    assert error["missing_fields"] == ["benchmark_id", "holdout_id"]
    assert "benchmark_id and holdout_id" in error["message"]


def test_query_id_mismatch_diagnostic_contains_missing_and_extra_ids():
    with pytest.raises(PairedBenchmarkError) as caught:
        build_paired_report(_report(["q1", "q2"]), _report(["q1", "q3"]))

    error = caught.value.as_dict()
    assert error["code"] == "holdout_query_id_mismatch"
    assert error["report"] == "asr"
    assert error["details"]["missing_from_report"] == ["q2"]
    assert error["details"]["extra_in_report"] == ["q3"]


def test_event_count_mismatch_diagnostic_is_query_scoped():
    visual = _report(["q1"], final_score=0.5)
    asr = _report(["q1"], final_score=0.5)
    asr["per_query"][0]["n_events"] = 3

    with pytest.raises(PairedBenchmarkError) as caught:
        build_paired_report(visual, asr)

    error = caught.value.as_dict()
    assert error["code"] == "event_count_mismatch"
    assert error["details"] == {"query_id": "q1", "event_counts": {"asr": 3, "visual": 2}}


def test_ground_truth_video_mismatch_diagnostic_is_query_scoped():
    visual = _report(["q1"], final_score=0.5)
    asr = _report(["q1"], final_score=0.5)
    asr["per_query"][0]["video_id"] = "different-video"

    with pytest.raises(PairedBenchmarkError) as caught:
        build_paired_report(visual, asr)

    error = caught.value.as_dict()
    assert error["code"] == "ground_truth_video_mismatch"
    assert error["details"]["query_id"] == "q1"
    assert error["details"]["ground_truth_video_ids"] == {
        "asr": "different-video",
        "visual": "video-0",
    }


def test_valid_minimal_pair_emits_explicit_parity_and_path_metrics():
    ids = ["q1", "q2"]
    result = build_paired_report(
        _report(ids, final_score=0.70),
        _report(ids, final_score=0.69),
        manifest={
            "benchmark_id": "b1",
            "holdout_id": "h1",
            "holdout_query_ids": ids,
        },
    )

    assert result["benchmark_id"] == "b1"
    assert result["holdout_id"] == "h1"
    assert result["holdout_query_ids"] == ids
    assert result["query_parity"]["q1"]["event_counts"] == {"asr": 2, "visual": 2}
    assert result["query_parity"]["q1"]["ground_truth_video_ids"] == {
        "asr": "video-0",
        "visual": "video-0",
    }
    assert result["path_metrics"]["visual"]["final_score"] == 0.70
    assert result["path_metrics"]["asr"]["final_score"] == 0.69
    assert result["official_score_status"] == "available"


def test_runner_cli_returns_precise_nonzero_diagnostic_for_missing_metadata(tmp_path, monkeypatch):
    visual = _report(["q1"])
    visual.pop("benchmark_id")
    visual.pop("holdout_id")
    visual_path = tmp_path / "visual.json"
    asr_path = tmp_path / "asr.json"
    output_path = tmp_path / "paired.json"
    visual_path.write_text(json.dumps(visual), encoding="utf-8")
    asr_path.write_text(json.dumps(_report(["q1"])), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_trake_paired_benchmark",
            "--visual", str(visual_path),
            "--asr", str(asr_path),
            "--output", str(output_path),
        ],
    )
    assert run_paired_main() == 2
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["error"]["code"] == "missing_paired_metadata"
    assert output["error"]["report"] == "visual"
    assert output["error"]["missing_fields"] == ["benchmark_id", "holdout_id"]
