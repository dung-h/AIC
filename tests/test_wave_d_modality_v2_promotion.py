"""Offline contract tests for global modality-index v2 promotion.

These tests use only synthetic manifests, preflight payloads, and benchmark
reports.  They never import a model, open a network connection, or mutate the
existing index/artifact files.
"""
from __future__ import annotations

from copy import deepcopy

from src.eval.modality_v2_promotion import (
    evaluate_modality_v2_promotion,
    validate_modality_v2_manifest,
    validate_v2_preflight,
)


PACKS = ("k01", "l21")
VIDEO_IDS = ("K01_V001", "L21_V001")


def _manifest(*, status: str = "ready", scope: str = "global", rows_ok: bool = True) -> dict:
    return {
        "schema": {"name": "hcmai.modality_index_manifest", "version": 2},
        "index_id": "modality_global_v2",
        "scope": {
            "name": "full_corpus",
            "video_count": 2,
            "video_ids": list(VIDEO_IDS),
            "packs": list(PACKS),
        },
        "canonical": {
            "validated": True,
            "video_count": 2,
            "mapping_errors": 0,
        },
        "modalities": {
            modality: {
                "status": status,
                "scope": scope,
                "video_count": 2,
                "expected_video_count": 2,
                "missing_video_ids": [],
                "missing_packs": [],
                "metadata_rows": 4,
                "embedding_rows": 4 if rows_ok else 3,
                "embedding_dim": 1024,
                "canonical_mapping_errors": 0,
            }
            for modality in ("asr", "ocr")
        },
        "benchmark": {
            "benchmark_id": "vqa_eval_v3",
            "holdout_id": "vqa-v3-holdout-20260818",
            "split": "holdout",
            "query_ids": ["q1", "q2"],
        },
    }


def _pack(pack: str) -> dict:
    return {
        "pack": pack,
        "errors": [],
        "embedding_rows": 2,
        "metadata_rows": 2,
        "canonical_missing_rows": 0,
        "mapping_mismatch_rows": 0,
    }


def _preflight(*, coverage: float = 1.0, passed: bool = True) -> dict:
    observed = 2 if coverage == 1.0 else 1
    missing_videos = [] if coverage == 1.0 else ["L21_V001"]
    missing_packs = [] if coverage == 1.0 else ["l21"]
    def report(modality: str) -> dict:
        return {
            "passed": passed,
            "scope": {
                "name": "full_corpus",
                "is_full_corpus": True,
                "active_packs": list(PACKS),
            },
            "coverage": {
                f"{modality}_observed_video_count": observed,
                "canonical_expected_video_count": 2,
                f"{modality}_video_coverage_ratio": coverage,
                f"{modality}_missing_videos": missing_videos,
                f"{modality}_missing": missing_packs,
            },
            "errors": [] if passed and coverage == 1.0 else [{"code": "coverage_incomplete"}],
            modality: [_pack("k01"), _pack("l21")],
        }
    return {"asr": report("asr"), "ocr": report("ocr")}


def _benchmark(*, holdout_id: str = "vqa-v3-holdout-20260818", query_ids=None, specialist: float = 0.5, global_score: float = 0.8) -> dict:
    return {
        "schema_version": "hcmai.architecture_benchmark_v1",
        "status": "complete",
        "provenance": {
            "split": {
                "benchmark_id": "vqa_eval_v3",
                "holdout_id": holdout_id,
                "split": "holdout",
            }
        },
        "qna": {
            "per_query": [{"query_id": item} for item in (query_ids or ["q1", "q2"])],
            "metrics": {
                "video_r20": {"value": global_score},
                "frame_recall": {"value": 0.7},
                "answer_grounded": {"value": 0.6},
            },
            "per_question_type": {
                "spoken_fact": {"metrics": {"answer_exact": {"value": specialist}}},
                "screen_text": {"metrics": {"answer_exact": {"value": specialist}}},
            },
        },
    }


def test_complete_manifest_and_preflight_pass_only_with_full_aligned_scope():
    manifest_report = validate_modality_v2_manifest(
        _manifest(), expected_video_ids=VIDEO_IDS, expected_packs=PACKS, expected_video_count=2
    )
    preflight_report = validate_v2_preflight(
        _preflight(), expected_packs=PACKS, expected_video_count=2
    )
    assert manifest_report["passed"] is True, manifest_report["errors"]
    assert preflight_report["passed"] is True, preflight_report["errors"]


def test_partial_diagnostic_or_misaligned_manifest_is_blocked():
    partial = validate_modality_v2_manifest(
        _manifest(status="diagnostic"), expected_video_ids=VIDEO_IDS, expected_packs=PACKS, expected_video_count=2
    )
    misaligned = validate_modality_v2_manifest(
        _manifest(rows_ok=False), expected_video_ids=VIDEO_IDS, expected_packs=PACKS, expected_video_count=2
    )
    assert partial["passed"] is False
    assert any(error["code"] == "modality_not_ready" for error in partial["errors"])
    assert misaligned["passed"] is False
    assert any(error["code"] == "embedding_metadata_row_mismatch" for error in misaligned["errors"])


def test_partial_preflight_remains_blocked_even_when_pack_files_exist():
    report = validate_v2_preflight(
        _preflight(coverage=0.5, passed=False), expected_packs=PACKS, expected_video_count=2
    )
    assert report["passed"] is False
    codes = {error["code"] for error in report["errors"]}
    assert "preflight_not_passed" in codes
    assert "preflight_video_coverage_incomplete" in codes


def test_promotion_requires_same_v3_holdout_and_specialist_gain():
    manifest = _manifest()
    baseline = _benchmark(specialist=0.4)
    candidate = {
        "asr": _benchmark(specialist=0.5),
        "ocr": _benchmark(specialist=0.5),
    }
    report = evaluate_modality_v2_promotion(
        manifest,
        _preflight(),
        baseline,
        candidate,
        expected_video_ids=VIDEO_IDS,
        expected_packs=PACKS,
        expected_video_count=2,
    )
    assert report["promotion_allowed"] is True, report["errors"]
    assert report["modalities"]["asr"]["promotion_allowed"] is True
    assert report["modalities"]["ocr"]["promotion_allowed"] is True


def test_promotion_blocks_mismatched_holdout_and_insufficient_gain():
    manifest = _manifest()
    baseline = _benchmark(specialist=0.4)
    candidate = {
        "asr": _benchmark(holdout_id="wrong-holdout", specialist=0.5),
        "ocr": _benchmark(specialist=0.42),
    }
    report = evaluate_modality_v2_promotion(
        manifest,
        _preflight(),
        baseline,
        candidate,
        expected_video_ids=VIDEO_IDS,
        expected_packs=PACKS,
        expected_video_count=2,
    )
    assert report["promotion_allowed"] is False
    assert any(error["code"] == "benchmark_holdout_mismatch" for error in report["errors"])
    assert any(error["code"] == "specialist_gain_below_gate" for error in report["errors"])


def test_promotion_does_not_infer_missing_v3_provenance_or_metrics():
    manifest = _manifest()
    baseline = _benchmark()
    candidate = {"asr": deepcopy(_benchmark()), "ocr": deepcopy(_benchmark())}
    for report in candidate.values():
        report.pop("provenance")
    result = evaluate_modality_v2_promotion(
        manifest,
        _preflight(),
        baseline,
        candidate,
        expected_video_ids=VIDEO_IDS,
        expected_packs=PACKS,
        expected_video_count=2,
    )
    assert result["promotion_allowed"] is False
    assert all(item["promotion_allowed"] is False for item in result["modalities"].values())
    assert any(error["code"] == "benchmark_holdout_mismatch" for error in result["errors"])


def test_promotion_rejects_non_architecture_benchmark_schema():
    manifest = _manifest()
    baseline = _benchmark(specialist=0.4)
    candidate = {"asr": _benchmark(specialist=0.5), "ocr": _benchmark(specialist=0.5)}
    candidate["asr"]["schema_version"] = "legacy_report"
    result = evaluate_modality_v2_promotion(
        manifest,
        _preflight(),
        baseline,
        candidate,
        expected_video_ids=VIDEO_IDS,
        expected_packs=PACKS,
        expected_video_count=2,
    )
    assert result["promotion_allowed"] is False
    assert any(error["code"] == "benchmark_schema_mismatch" for error in result["errors"])
