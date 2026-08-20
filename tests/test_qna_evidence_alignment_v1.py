from __future__ import annotations

from src.vqa.evidence_fusion import build_evidence_packet
from src.vqa.verifier import EvidenceVerifier


def _candidate(**overrides):
    value = {
        "candidate_id": "q1",
        "video_id": "V1",
        "kf_n": 10,
        "frame_idx": 100,
        "pts_time": 100.0,
    }
    value.update(overrides)
    return value


def test_modality_status_distinguishes_index_missing_no_speech_and_no_match():
    missing = build_evidence_packet(_candidate(), asr_rows=None)
    assert missing["modality_status"]["asr"]["status"] == "index_missing"

    no_speech = build_evidence_packet(
        _candidate(),
        asr_rows=[{"video_id": "V1", "start": 99.0, "end": 101.0, "chunk": ""}],
    )
    assert no_speech["modality_status"]["asr"]["status"] == "no_speech"

    no_match = build_evidence_packet(
        _candidate(),
        asr_rows=[{"video_id": "V1", "start": 150.0, "end": 151.0, "chunk": "valid transcript"}],
    )
    assert no_match["modality_status"]["asr"]["status"] == "no_match"


def test_required_no_speech_is_unavailable_not_a_no_text_match():
    packet = build_evidence_packet(
        _candidate(),
        asr_rows=[{"video_id": "V1", "start": 99.0, "end": 101.0, "chunk": ""}],
    )
    result = EvidenceVerifier().verify("25 độ", packet, required_sources=("asr",))
    assert result.abstain is True
    assert result.reason == "required_evidence_unavailable"
    assert result.checks[0].reason == "asr_no_speech"


def test_asr_interval_overlap_is_selected_even_when_start_is_before_anchor_window():
    packet = build_evidence_packet(
        _candidate(pts_time=100.0),
        asr_rows=[{
            "video_id": "V1", "start": 80.0, "end": 105.0,
            "chunk": "Nha Trang hai mươi lăm độ",
        }],
        asr_window=1.0,
    )
    assert len(packet["asr_chunks"]) == 1
    assert packet["asr_chunks"][0]["distance_s"] == 0.0


def test_ocr_adjacent_canonical_keyframe_is_a_bounded_rescue():
    packet = build_evidence_packet(
        _candidate(kf_n=10, pts_time=100.0),
        ocr_rows=[{
            "video_id": "V1", "kf_n": 9, "frame_idx": 90,
            "pts_time": 103.0, "ocr_text": "Nhãn sản phẩm",
        }],
        ocr_window=1.0,
        ocr_adjacent_kf_radius=2,
        ocr_adjacent_kf_window=5.0,
    )
    assert [row["kf_n"] for row in packet["ocr_text"]] == [9]
    assert packet["ocr_text"][0]["frame_idx"] == 90
    assert packet["modality_status"]["ocr"]["status"] == "matched"


def test_canonical_map_rejects_frame_idx_mismatch():
    try:
        build_evidence_packet(
            _candidate(),
            canonical_map={("V1", 10): (101, 100.0)},
        )
    except ValueError as exc:
        assert "canonical mapping mismatch" in str(exc)
    else:  # pragma: no cover - assertion documents the fail-closed contract.
        raise AssertionError("invalid canonical mapping was accepted")


def test_verifier_normalizes_mojibake_diacritics_and_number_words():
    result = EvidenceVerifier().verify(
        "25 độ",
        {
            "candidate_id": "q1",
            "video_id": "V1",
            "frames": [{"video_id": "V1", "frame_idx": 100}],
            "asr_chunks": [{
                "start": 99.0, "end": 101.0,
                "chunk": "Nha Trang hai mÆ°Æ¡i lÄƒm Ä‘á»™",
            }],
        },
        required_sources=("asr",),
    )
    assert result.abstain is False
    assert result.checks[0].reason == "numeric_fact_match"
