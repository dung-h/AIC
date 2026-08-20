"""Focused tests for the model-free Q&A selector decomposition."""

from __future__ import annotations

import json

from src.eval.qna_selector_benchmark import (
    SCHEMA_VERSION,
    evaluate_selector,
    report_to_json,
)


def _candidate_lattice():
    return {
        "q1": [
            {"video_id": "V1", "frame_idx": 10, "question_type": "place"},
            {"video_id": "V2", "frame_idx": 20, "question_type": "place"},
            {"video_id": "V3", "frame_idx": 30, "question_type": "place"},
        ],
    }


def test_perfect_lattice_and_selector_report_full_recall_and_stable_json():
    lattice = _candidate_lattice()
    selected = {"q1": lattice["q1"][:2]}
    report = evaluate_selector(
        lattice,
        selected,
        relevant_keys={"q1": [("V1", 10), ("V2", 20)]},
        relevant_video_ids={"q1": ["V1", "V2"]},
        budget={"q1": 2},
        question_types={"q1": "place"},
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "complete"
    assert report["lattice_video_recall"] == 1.0
    assert report["lattice_frame_recall"] == 1.0
    assert report["selector_video_recall"] == 1.0
    assert report["selector_frame_recall"] == 1.0
    assert report["candidate_miss"]["frame"]["count"] == 0
    assert report["selector_miss"]["frame"]["count"] == 0
    assert report["budget"]["selected_count"] == 2
    assert report["budget"]["within_budget"] is None  # per-query budgets are reported per query
    assert report["per_question_type"]["place"]["query_count"] == 1
    assert json.loads(report_to_json(report))["schema_version"] == SCHEMA_VERSION
    assert report_to_json(report) == report_to_json(json.loads(report_to_json(report)))


def test_partial_case_separates_candidate_miss_from_selector_miss():
    report = evaluate_selector(
        _candidate_lattice(),
        {"q1": [{"video_id": "V1", "frame_idx": 10}]},
        relevant_keys={"q1": [("V1", 10), ("V2", 20), ("V9", 90)]},
        relevant_video_ids={"q1": ["V1", "V2", "V9"]},
        budget={"q1": 1},
    )

    assert report["status"] == "complete"
    assert report["lattice_video_recall"] == 2 / 3
    assert report["lattice_frame_recall"] == 2 / 3
    assert report["selector_video_recall"] == 1 / 3
    assert report["selector_frame_recall"] == 1 / 3
    assert report["candidate_miss"]["video"]["count"] == 1
    assert report["candidate_miss"]["frame"]["count"] == 1
    assert report["selector_miss"]["video"]["count"] == 1
    assert report["selector_miss"]["frame"]["count"] == 1
    assert report["selector_miss"]["frame"]["keys"] == [{"video_id": "V2", "frame_idx": 20}]
    assert report["budget"]["within_budget"] is None
    assert report["per_query"][0]["coverage"]["selector_frame_coverage_of_lattice"] == 1 / 3


def test_missing_oracle_blocks_without_fabricating_zero_recall():
    report = evaluate_selector(
        _candidate_lattice(),
        {"q1": _candidate_lattice()["q1"][:1]},
        budget={"q1": 1},
    )

    assert report["status"] == "blocked"
    assert report["official_score_claim"] is False
    assert any("oracle" in reason.lower() for reason in report["blocked_reasons"])
    assert report["lattice_video_recall"] is None
    assert report["lattice_frame_recall"] is None
    assert report["selector_video_recall"] is None
    assert report["selector_frame_recall"] is None
    assert report["candidate_miss"]["frame"]["count"] == 0
    assert report["candidate_miss"]["frame"]["rate"] is None
    assert report["per_query"][0]["coverage"]["lattice"]["raw_count"] == 3
