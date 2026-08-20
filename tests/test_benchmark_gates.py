"""Tests for the isolated Q&A benchmark decomposition gate.

These tests use only synthetic artifacts.  They intentionally do not import
or execute the production pipeline, retrieval models, VLMs, or providers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.report_qna_decomposition import BenchmarkGateError, build_report


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _retrieval_rows() -> list[dict]:
    return [
        {
            "query_id": "q1", "split": "holdout", "question_type": "spoken_fact",
            "arm": "baseline", "video_rank": 30, "preselector_frame_hit": True,
            "postselector_frame_hit": False,
        },
        {
            "query_id": "q2", "split": "holdout", "question_type": "screen_text",
            "arm": "baseline", "video_rank": 2, "preselector_frame_hit": True,
            "postselector_frame_hit": True,
        },
    ]


def _candidate_retrieval_rows() -> list[dict]:
    return [
        {
            "query_id": "q1", "split": "holdout", "question_type": "spoken_fact",
            "arm": "routed", "video_rank": 4, "preselector_frame_hit": True,
            "postselector_frame_hit": True, "video_rescue_top20": True,
            "video_rescue_top100": True,
        },
        {
            "query_id": "q2", "split": "holdout", "question_type": "screen_text",
            "arm": "routed", "video_rank": 2, "preselector_frame_hit": True,
            "postselector_frame_hit": False, "video_rescue_top20": False,
            "video_rescue_top100": False,
        },
    ]


def _answer_rows() -> list[dict]:
    return [
        {
            "query_id": "q1", "split": "holdout", "top_answer_match": True,
            "answer_stage_frame_hit": True,
            # This historical field must not substitute for an oracle.
            "gt_frame_answer_match": True,
        },
        {
            "query_id": "q2", "split": "holdout", "top_answer_match": False,
            "answer_stage_frame_hit": False,
            "gt_frame_answer_match": True,
        },
    ]


def _build_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "base_retrieval": _write(tmp_path / "base_retrieval.json", {"per_query": _retrieval_rows()}),
        "cand_retrieval": _write(tmp_path / "cand_retrieval.json", {"per_query": _candidate_retrieval_rows()}),
        "base_answers": _write(tmp_path / "base_answers.json", {"per_query": _answer_rows()}),
        "cand_answers": _write(tmp_path / "cand_answers.json", {"per_query": _answer_rows()}),
        "oracle": _write(tmp_path / "oracle.json", {"rows": [
            {"query_id": "q1", "split": "holdout", "exact_match": True},
            {"query_id": "q2", "split": "holdout", "exact_match": False},
        ]}),
    }


def _call(paths: dict[str, Path], **overrides):
    args = {
        "baseline_retrieval": paths["base_retrieval"],
        "candidate_retrieval": paths["cand_retrieval"],
        "baseline_arm": "baseline",
        "candidate_arm": "routed",
        "baseline_answers": paths["base_answers"],
        "candidate_answers": paths["cand_answers"],
        "gt_frame_oracle": paths["oracle"],
        "split": "holdout",
    }
    args.update(overrides)
    return build_report(**args)


def test_decomposition_separates_all_stages_and_rescue(tmp_path: Path):
    report = _call(_build_paths(tmp_path))

    assert report["status"] == "complete"
    assert report["gate"]["promotion_allowed"] is True
    candidate = report["metrics"]["candidate"]
    assert candidate["video_recall"]["r20"]["value"] == 1.0
    assert candidate["frame_lattice_recall"]["value"] == 1.0
    assert candidate["selector_recall"]["value"] == 0.5
    assert candidate["answer_on_gt_frame"]["value"] == 0.5
    assert candidate["answer_on_retrieved_frame"]["value"] == 0.5
    assert candidate["end_to_end_grounded_answer"]["value"] == 0.5
    assert report["modality_rescue"]["top20"]["value"] == 0.5
    assert report["protocol"]["holdout_used_for_tuning"] is False


def test_retrieved_gt_frame_field_cannot_be_used_as_oracle(tmp_path: Path):
    paths = _build_paths(tmp_path)
    report = _call(paths, gt_frame_oracle=None)

    assert report["status"] == "blocked"
    assert report["gate"]["promotion_allowed"] is False
    assert report["metrics"]["candidate"]["answer_on_gt_frame"]["value"] is None
    assert report["metrics"]["candidate"]["answer_on_gt_frame"]["present"] == 0
    assert "independent GT-frame oracle is absent" in report["blocking_reasons"]


def test_missing_required_artifact_fails_closed(tmp_path: Path):
    paths = _build_paths(tmp_path)
    with pytest.raises(BenchmarkGateError, match="missing benchmark artifact"):
        _call(paths, candidate_answers=tmp_path / "missing.json")


def test_split_mismatch_fails_closed(tmp_path: Path):
    paths = _build_paths(tmp_path)
    paths["cand_retrieval"].write_text(json.dumps({
        "split": "dev", "per_query": _candidate_retrieval_rows(),
    }), encoding="utf-8")
    with pytest.raises(BenchmarkGateError, match="does not match requested split"):
        _call(paths)


def test_query_set_mismatch_fails_closed(tmp_path: Path):
    paths = _build_paths(tmp_path)
    payload = {"per_query": _candidate_retrieval_rows()[:-1]}
    paths["cand_retrieval"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkGateError, match="query sets differ"):
        _call(paths)
