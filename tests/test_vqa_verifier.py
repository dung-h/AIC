from __future__ import annotations

from src.vqa.verifier import EvidenceVerifier


def _candidate(**overrides):
    candidate = {
        "candidate_id": "c1",
        "video_id": "V1",
        "frames": [{"video_id": "V1", "frame_idx": 120, "kf_n": 12,
                    "pts_time": 8.0}],
        "provenance": [{"source": "visual", "rank": 1}],
    }
    candidate.update(overrides)
    return candidate


def test_verifier_accepts_answer_supported_by_timestamped_asr():
    result = EvidenceVerifier().verify(
        "25 độ",
        _candidate(asr_chunks=[{"start": 8.0, "end": 10.0, "chunk": "Nha Trang hôm nay 25 độ"}]),
        required_sources=("asr",),
    )

    assert result.abstain is False
    assert result.reason == "supported"
    assert result.supported_sources == ("asr",)
    assert result.frame_ids == (120,)
    assert result.canonical_provenance[0]["frame_idx"] == 120


def test_verifier_accepts_answer_supported_by_ocr():
    result = EvidenceVerifier().verify(
        "Nha Trang",
        _candidate(ocr_text=[{"pts_time": 8.0, "ocr_text": "Dự báo Nha Trang"}]),
        required_sources=("ocr",),
    )
    assert result.abstain is False
    assert result.supported_sources == ("ocr",)


def test_verifier_accepts_long_asr_quote_with_bounded_transcription_noise():
    result = EvidenceVerifier().verify(
        "Hỏa hồng Nhật Tảo oanh thiên địa Kiếm bạt Kiên Giang khấp quỷ thần",
        _candidate(asr_chunks=[{
            "start": 8.0, "end": 12.0,
            "chunk": "Hóa hồng Nhật Tảo quanh thiên địa Kiếm bạt Kiên Giang khắp vị thần",
        }]),
        required_sources=("asr",),
    )
    assert result.abstain is False
    assert result.checks[0].reason == "long_asr_ocr_overlap"


def test_verifier_uses_injected_visual_checker_for_frame_only_evidence():
    seen = []

    def checker(answer, frame):
        seen.append((answer, frame.frame_idx))
        return {"supported": answer == "red", "score": 0.75, "reason": "fake_visual_match"}

    result = EvidenceVerifier(frame_checker=checker).verify("red", _candidate())
    assert result.abstain is False
    assert result.supported_sources == ("visual",)
    assert seen == [("red", 120)]


def test_verifier_abstains_when_visual_semantics_have_no_checker():
    result = EvidenceVerifier().verify("red", _candidate())
    assert result.abstain is True
    assert result.reason == "visual_checker_unavailable"


def test_verifier_abstains_when_answer_is_not_supported_or_required_source_missing():
    result = EvidenceVerifier().verify(
        "18 độ",
        _candidate(asr_chunks=[{"start": 8.0, "end": 10.0, "chunk": "Nha Trang hôm nay 25 độ"}]),
        required_sources=("asr",),
    )
    assert result.abstain is True
    assert result.reason == "required_evidence_not_supporting_answer"
    assert result.support_score == 0.0

    missing = EvidenceVerifier().verify(
        "25 độ",
        {"candidate_id": "bad", "video_id": "V1", "frames": []},
    )
    assert missing.abstain is True
    assert missing.reason == "missing_or_invalid_evidence"


def test_verifier_rejects_non_answer_and_checker_failure_fail_closed():
    empty = EvidenceVerifier().verify("", _candidate())
    assert empty.abstain is True
    assert empty.reason == "invalid_answer"

    def broken(answer, frame):
        raise RuntimeError("fake failure")

    failed = EvidenceVerifier(frame_checker=broken).verify("red", _candidate())
    assert failed.abstain is True
    assert failed.reason == "frame_checker_error"
