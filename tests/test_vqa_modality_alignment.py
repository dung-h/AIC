from __future__ import annotations

from src.eval.audit_vqa_modality_alignment import audit_asr_row, audit_ocr_row


def _annotation(annotation_id: str = "q1", *, question_type: str = "spoken_fact") -> dict:
    return {
        "annotation_id": annotation_id,
        "split": "dev",
        "question_type": question_type,
        "video_id": "K01_V001",
        "kf_n": 2,
        "pts_time": 10.0,
        "acceptable_kf_n": "2,3",
        "canonical_evidence": [
            {"kf_n": 2, "pts_time": 10.0},
            {"kf_n": 3, "pts_time": 20.0},
        ],
    }


def test_asr_interval_overlap_is_target_aligned() -> None:
    row = _annotation()
    chunks = {
        "K01_V001": [
            {"chunk_index": 4, "start": 8.0, "end": 12.0, "text": "evidence"},
        ]
    }

    result = audit_asr_row(row, chunks, set(), target_tolerance_seconds=0.0)

    assert result["status"] == "target_aligned"
    assert result["target_aligned"] is True
    assert result["acceptable_aligned"] is True
    assert result["matched_target_chunks"] == [{"chunk_index": 4, "start": 8.0, "end": 12.0}]


def test_asr_interval_can_hit_acceptable_timestamp_only() -> None:
    row = _annotation()
    chunks = {
        "K01_V001": [
            {"chunk_index": 5, "start": 19.0, "end": 21.0, "text": "later evidence"},
        ]
    }

    result = audit_asr_row(row, chunks, set(), target_tolerance_seconds=0.0, acceptable_tolerance_seconds=0.0)

    assert result["status"] == "acceptable_only"
    assert result["target_aligned"] is False
    assert result["acceptable_aligned"] is True


def test_asr_missing_and_declared_no_speech_are_distinct() -> None:
    row = _annotation()

    missing = audit_asr_row(row, {}, set())
    no_speech = audit_asr_row(row, {}, {"K01_V001"})

    assert missing["status"] == "missing_global_chunks"
    assert no_speech["status"] == "no_speech_available"


def _ocr_row(annotation_id: str = "q-ocr") -> dict:
    row = _annotation(annotation_id, question_type="screen_text")
    row["video_id"] = "L26_V257"
    return row


def _ocr_sidecar(source: str, text: str = "TITLE", *, kf_n: int = 25, pts_time: float = 35.84) -> dict:
    return {
        "annotation_id": "q-ocr",
        "ocr_evidence": [
            {"kf_n": kf_n, "pts_time": pts_time, "text": text, "source": source}
        ],
    }


def test_ocr_classifies_retrievable_on_exact_key_text_and_time() -> None:
    result = audit_ocr_row(
        _ocr_row(),
        _ocr_sidecar("local_ocr_index"),
        {("L26_V257", 25): [{"pts_time": 35.84, "ocr_text": "TITLE"}]},
    )

    assert result["status"] == "retrievable"
    assert result["observations"][0]["exact_text_and_time_match"] is True


def test_ocr_classifies_manual_only_without_global_exact_evidence() -> None:
    result = audit_ocr_row(
        _ocr_row(),
        _ocr_sidecar("manual_frame_read", "VISIBLE TITLE"),
        {("L26_V257", 25): [{"pts_time": 35.84, "ocr_text": "channel logo"}]},
    )

    assert result["status"] == "manual-only"


def test_ocr_classifies_missing_when_nonmanual_key_is_absent() -> None:
    result = audit_ocr_row(
        _ocr_row(),
        _ocr_sidecar("local_ocr_index"),
        {},
    )

    assert result["status"] == "missing"
    assert result["observations"][0]["global_key_present"] is False


def test_ocr_classifies_mismatch_when_key_exists_but_text_differs() -> None:
    result = audit_ocr_row(
        _ocr_row(),
        _ocr_sidecar("local_ocr_index", "VISIBLE TITLE"),
        {("L26_V257", 25): [{"pts_time": 35.84, "ocr_text": "channel logo"}]},
    )

    assert result["status"] == "mismatch"
    assert result["observations"][0]["global_key_present"] is True
    assert result["observations"][0]["exact_text_match_any_time"] is False
