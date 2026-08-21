from __future__ import annotations

from src.vqa.video_evidence_join import VideoEvidenceRoleJoiner
from src.vqa.selector import deduplicate_candidates
from src.reranking.qna_modality_router import QNAModalityRouter
from src.vqa.evidence_fusion import build_evidence_packet


def _p1_22_like_rows() -> list[dict]:
    return [
        {
            "video_id": "L26_V178",
            "kf_n": 13,
            "frame_idx": 300,
            "pts_time": 12.0,
            "modality": "asr",
            "chunk": "Món ngon mỗi ngày Bánh ít rừng mềm mịn dẻo dai.",
        },
        {
            "video_id": "L26_V178",
            "kf_n": 27,
            "frame_idx": 1346,
            "pts_time": 53.84,
            "modality": "asr",
            "chunk": "Hôm nay cô làm món bánh ít trần.",
        },
        {
            "video_id": "L26_V178",
            "kf_n": 19,
            "frame_idx": 768,
            "pts_time": 30.72,
            "modality": "ocr",
            "ocr_text": "Thịt nạc dăm xay 200g",
        },
        {
            "video_id": "L26_V178",
            "kf_n": 35,
            "frame_idx": 1490,
            "pts_time": 55.0,
            "modality": "asr",
            "chunk": "Tiếp theo, chúng ta sơ chế phần nhân.",
        },
    ]


def _strong_ingredient_anchor() -> dict:
    return {
        "video_id": "L26_V178",
        "kf_n": 19,
        "frame_idx": 768,
        "pts_time": 30.72,
        "modality": "ocr",
        "ocr_text": "Thịt nạc dăm xay 200g",
        "evidence_score": 0.98,
    }


def test_p1_22_like_anchor_joins_early_title_in_same_video():
    joiner = VideoEvidenceRoleJoiner()

    rows = joiner.join(
        _p1_22_like_rows(),
        query="Nguyên liệu có 200g thịt nạc dăm xay.",
        question="Tên món là gì?",
        anchor_candidate=_strong_ingredient_anchor(),
    )

    assert len(rows) == 1
    result = rows[0]
    assert (result["video_id"], result["kf_n"], result["frame_idx"]) == (
        "L26_V178",
        27,
        1346,
    )
    assert result["source_text"] == "Hôm nay cô làm món bánh ít trần."
    assert result["modality"] == "asr"
    assert result["role"] == "answer_support"
    assert result["provenance"]["anchor"]["frame_idx"] == 768
    assert result["provenance"]["joiner"] == "video_evidence_role_join_v1"
    assert result["provenance"]["score_components"]["title_cue"] is True
    assert joiner.last_diagnostic["status"] == "ok"


def test_rejects_mixed_or_wrong_video_evidence_instead_of_cross_video_join():
    joiner = VideoEvidenceRoleJoiner()
    rows = _p1_22_like_rows()
    rows[0] = {**rows[0], "video_id": "L26_V177"}

    result = joiner.join(
        rows,
        query="Nguyên liệu có 200g thịt nạc dăm xay.",
        question="Tên món là gì?",
        anchor_candidate=_strong_ingredient_anchor(),
    )

    assert result == []
    assert joiner.last_diagnostic["status"] == "mixed_or_wrong_video_rows"


def test_rejects_missing_canonical_mapping():
    joiner = VideoEvidenceRoleJoiner()
    rows = _p1_22_like_rows()
    del rows[0]["frame_idx"]

    result = joiner.join(
        rows,
        query="Nguyên liệu có 200g thịt nạc dăm xay.",
        question="Tên món là gì?",
        anchor_candidate=_strong_ingredient_anchor(),
    )

    assert result == []
    assert joiner.last_diagnostic["status"] == "missing_or_invalid_canonical_fields"


def test_no_title_intent_or_weak_anchor_does_not_trigger_broad_search():
    joiner = VideoEvidenceRoleJoiner()

    no_intent = joiner.join(
        _p1_22_like_rows(),
        query="Nguyên liệu có 200g thịt nạc dăm xay.",
        question="Nguyên liệu này nặng bao nhiêu?",
        anchor_candidate=_strong_ingredient_anchor(),
    )
    assert no_intent == []
    assert joiner.last_diagnostic["status"] == "inactive_non_title_intent"

    weak_anchor = {**_strong_ingredient_anchor(), "evidence_score": 0.3}
    weak = joiner.join(
        _p1_22_like_rows(),
        query="Nguyên liệu có 200g thịt nạc dăm xay.",
        question="Tên món là gì?",
        anchor_candidate=weak_anchor,
    )
    assert weak == []
    assert joiner.last_diagnostic["status"] == "weak_anchor"


def test_visual_specialist_same_frame_merge_preserves_role_join_anchor():
    merged = deduplicate_candidates([
        {
            "video_id": "L26_V178", "kf_n": 19, "frame_idx": 768,
            "pts_time": 30.72, "source": "visual", "base_score": 0.9,
        },
        {
            "video_id": "L26_V178", "kf_n": 19, "frame_idx": 768,
            "pts_time": 30.72, "source": "ocr", "modality_score": 0.98,
            "text": "Thịt nạc dăm xay 200g",
            "evidence": {"modality": "ocr", "text": "Thịt nạc dăm xay 200g"},
            "view_provenance": [{"score_mode": "bm25_coverage", "score": 1.0}],
        },
    ])[0]

    assert merged["source"] == "visual"
    assert QNAModalityRouter._strong_role_join_anchor(
        merged,
        "Nguyên liệu có 200g thịt nạc dăm xay.",
        "Tên món là gì?",
    ) is True
    recovered = QNAModalityRouter._specialist_role_join_anchor(merged)
    assert recovered is not None
    assert recovered["source"] == "ocr"
    assert recovered["text"] == "Thịt nạc dăm xay 200g"
    assert recovered["frame_idx"] == 768


def test_generic_distant_evidence_frame_cannot_expand_answer_context():
    packet = build_evidence_packet(
        {
            "video_id": "V1", "frame_idx": 10, "kf_n": 1, "pts_time": 1.0,
            "evidence_frames": [{
                "video_id": "V1", "frame_idx": 100, "kf_n": 9, "pts_time": 90.0,
                "modality": "asr", "relation": "unbounded_same_video",
            }],
        },
        asr_rows=[{
            "video_id": "V1", "kf_n": 9, "frame_idx": 100,
            "start": 89.0, "end": 91.0, "chunk": "a distant unrelated title",
        }],
        query="scene", question="What is the answer?",
    )

    assert len(packet["frames"]) == 1
    assert packet["frames"][0]["frame_idx"] == 10
    assert packet["asr_chunks"] == []
