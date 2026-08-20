from __future__ import annotations

import socket
from urllib import request as urllib_request

import pytest

from src.eval.qna_cross_modal_contract import (
    evaluate_predictions,
    offline_network_guard,
    validate_benchmark_rows,
)


def _canonical():
    return [
        {"video_id": "K01_V001", "kf_n": 10, "frame_idx": 100, "pts_time": 10.0},
        {"video_id": "K01_V001", "kf_n": 11, "frame_idx": 110, "pts_time": 12.0},
    ]


def _spoken(**overrides):
    row = {
        "annotation_id": "q-spoken",
        "question_type": "spoken_fact",
        "video_id": "K01_V001",
        "kf_n": 10,
        "frame_idx": 100,
        "acceptable_kf_n": "10,11",
        "query": "A weather report shows an anchor beside a city map.",
        "question": "What temperature is reported?",
        "answer": "25 degrees",
        "answer_start_time": 10.0,
        "answer_end_time": 12.0,
        "required_modalities": "visual,asr",
        "status": "valid",
        "asr_evidence": [{"start": 10.0, "end": 12.0, "text": "The temperature is 25 degrees."}],
    }
    row.update(overrides)
    return row


def _screen(**overrides):
    row = {
        "annotation_id": "q-screen",
        "question_type": "screen_text",
        "video_id": "K01_V001",
        "kf_n": 11,
        "frame_idx": 110,
        "acceptable_kf_n": "11",
        "query": "A poster appears beside a presenter.",
        "question": "What year is printed?",
        "answer": "2025",
        "answer_start_time": 12.0,
        "answer_end_time": 12.0,
        "required_modalities": "visual,ocr",
        "status": "valid",
        "ocr_evidence": [{"pts_time": 12.0, "text": "2025"}],
    }
    row.update(overrides)
    return row


def test_spoken_contract_requires_asr_and_canonical_mapping():
    report = validate_benchmark_rows([_spoken()], canonical_rows=_canonical(), require_sidecar_evidence=True)
    assert report["passed"], report["issues"]

    bad = validate_benchmark_rows([_spoken(required_modalities="visual")], canonical_rows=_canonical(), require_sidecar_evidence=True)
    assert "missing_required_modality" in {item["code"] for item in bad["issues"]}

    no_visual = validate_benchmark_rows([_spoken(required_modalities="asr")], canonical_rows=_canonical())
    assert "missing_visual_modality" in {item["code"] for item in no_visual["issues"]}


def test_screen_text_sidecar_is_not_fabricated():
    report = validate_benchmark_rows([_screen(ocr_evidence=[])], canonical_rows=_canonical(), require_sidecar_evidence=True)
    assert not report["passed"]
    assert "missing_evidence_sidecar" in {item["code"] for item in report["issues"]}


def test_benchmark_rejects_answer_leak_and_placeholder():
    report = validate_benchmark_rows(
        [_spoken(query="The report says 25 degrees.", answer="25 degrees"), _screen(answer="unknown")],
        canonical_rows=_canonical(),
    )
    codes = {item["code"] for item in report["issues"]}
    assert "answer_leak" in codes
    assert "empty_or_placeholder_answer" in codes


def test_prediction_timestamp_tolerance_and_canonical_frame():
    rows = [_spoken()]
    predictions = [{
        "annotation_id": "q-spoken",
        "video_id": "K01_V001",
        "frame_id": 100,
        "answer": "25 degrees",
        "evidence": {"asr": [{"start": 13.5, "end": 14.0, "text": "The temperature is 25 degrees."}]},
    }]
    result = evaluate_predictions(rows, predictions, canonical_rows=_canonical(), timestamp_tolerances=(2.0, 5.0))
    metrics = result["metrics"]
    assert metrics["canonical_frame_valid"] == 1.0
    assert metrics["answer_accuracy"] == 1.0
    assert metrics["asr_timestamp_hit_2.0s"] == 1.0
    assert metrics["asr_timestamp_hit_5.0s"] == 1.0

    miss = evaluate_predictions(rows, [{**predictions[0], "evidence": {"asr": [{"start": 30, "end": 31, "text": "25 degrees"}]} }], canonical_rows=_canonical())
    assert miss["metrics"]["asr_timestamp_hit_2.0s"] == 0.0


def test_prediction_requires_evidence_for_grounded_answer_and_ocr_path():
    rows = [_screen()]
    no_evidence = evaluate_predictions(rows, [{"annotation_id": "q-screen", "video_id": "K01_V001", "frame_id": 110, "answer": "2025"}], canonical_rows=_canonical())
    assert no_evidence["metrics"]["answer_accuracy"] == 1.0
    assert no_evidence["metrics"]["evidence_presence"] == 0.0
    assert no_evidence["metrics"]["end_to_end_grounded_answer"] == 0.0

    grounded = evaluate_predictions(rows, [{
        "annotation_id": "q-screen", "video_id": "K01_V001", "frame_id": 110,
        "answer": "2025", "evidence": {"ocr": [{"text": "2025", "pts_time": 12.0}]},
    }], canonical_rows=_canonical())
    assert grounded["metrics"]["end_to_end_grounded_answer"] == 1.0


def test_invalid_frame_is_fail_closed():
    result = evaluate_predictions([_spoken()], [{
        "annotation_id": "q-spoken", "video_id": "K01_V001", "frame_id": 999,
        "answer": "25 degrees", "evidence": {"asr": [{"start": 10, "end": 12, "text": "25 degrees"}]},
    }], canonical_rows=_canonical())
    assert result["metrics"]["canonical_frame_valid"] == 0.0
    assert result["metrics"]["end_to_end_grounded_answer"] == 0.0


def test_offline_guard_blocks_socket_and_urlopen():
    with offline_network_guard():
        with pytest.raises(RuntimeError, match="network disabled"):
            socket.create_connection(("example.com", 443), timeout=0.01)
        with pytest.raises(RuntimeError, match="network disabled"):
            urllib_request.urlopen("https://example.com", timeout=0.01)
