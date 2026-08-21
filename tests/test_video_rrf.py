"""Focused contract tests for evidence-aware video RRF rescue."""

from src.reranking.query_routing_policy import (
    RoutingConfig,
    build_routing_plan,
    route_video_candidates,
)
from src.reranking.video_rrf import weighted_video_rrf


def _ranked(*video_ids):
    return [
        {"video_id": video_id, "kf_n": rank, "frame_idx": rank}
        for rank, video_id in enumerate(video_ids, 1)
    ]


def test_routing_off_is_the_unchanged_visual_baseline():
    plan = build_routing_plan("screen_text", RoutingConfig.baseline(enabled=False))
    visual = _ranked("V1", "V2", "V3")

    rows = route_video_candidates(
        {"visual": visual, "ocr": _ranked("OCR_ONLY")}, plan,
    )

    assert rows == weighted_video_rrf(
        {"visual": visual}, {"visual": 1.0}, topk=20,
        specialist_rescue_enabled=False,
    )
    assert [row["video_id"] for row in rows] == ["V1", "V2", "V3"]
    assert all("rrf_guard_reason" not in row for row in rows)


def test_noisy_rank_16_specialist_without_strong_provenance_is_rejected():
    visual = _ranked(*[f"V{rank:02d}" for rank in range(1, 21)])
    ocr = _ranked(*[f"OCR_{rank:02d}" for rank in range(1, 16)], "NOISY")
    ocr[-1].update({
        "text": "generic recipe narration",
        "evidence": {"text": "generic recipe narration"},
        "view_provenance": [{"score_mode": "bm25_coverage", "rank": 9, "score": 1.0}],
    })

    rows = weighted_video_rrf(
        {"visual": visual, "ocr": ocr}, {"visual": 1.0, "ocr": 1.0},
        topk=20, require_specialist_evidence=True,
    )

    assert [row["video_id"] for row in rows] == [f"V{rank:02d}" for rank in range(1, 21)]
    assert all(row["rrf_guard"] == "visual_boundary_rrf" for row in rows)


def test_rank_16_exact_lexical_evidence_can_rescue_video_deterministically():
    visual = _ranked(*[f"V{rank:02d}" for rank in range(1, 21)])
    ocr = _ranked(*[f"OCR_{rank:02d}" for rank in range(1, 16)], "TARGET")
    ocr[-1].update({
        "text": "NGUYÊN LIỆU: Thịt nạc dăm xay 200g",
        "evidence": {"modality": "ocr", "text": "NGUYÊN LIỆU: Thịt nạc dăm xay 200g"},
        "view_provenance": [{"score_mode": "bm25_coverage", "rank": 1, "score": 1.0}],
    })

    rows = weighted_video_rrf(
        {"visual": visual, "ocr": ocr}, {"visual": 1.0, "ocr": 1.0},
        topk=20, require_specialist_evidence=True,
    )

    target = next(row for row in rows if row["video_id"] == "TARGET")
    assert target["ocr_rank"] == 16
    assert target["rrf_guard"] == "evidence_aware_specialist_rescue"
    assert target["rrf_guard_reason"] == "evidence_aware:ocr:bm25_coverage:rank_1:score_1.000"


def test_gated_specialist_vote_reorders_an_already_visual_admitted_video():
    visual = _ranked("V1", "V2", "V3")
    asr = _ranked("V3")
    asr[0].update({
        "text": "Nhiệt độ tại Nha Trang là 25 độ C",
        "evidence": {"modality": "asr", "text": "Nhiệt độ tại Nha Trang là 25 độ C"},
        "view_provenance": [{"score_mode": "bm25_coverage", "rank": 1, "score": 1.0}],
    })

    rows = weighted_video_rrf(
        {"visual": visual, "asr": asr}, {"visual": 1.0, "asr": 1.0},
        topk=3, require_specialist_evidence=True,
    )

    assert [row["video_id"] for row in rows] == ["V3", "V1", "V2"]
    assert rows[0]["rrf_guard"] == "visual_boundary_rrf"
