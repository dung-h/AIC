from src.reranking.video_rrf import weighted_video_rrf


def _ranked(video_ids, *, evidence=False, start_score=1.0):
    rows = []
    for rank, video_id in enumerate(video_ids, 1):
        row = {"video_id": video_id, "kf_n": rank, "score": start_score / rank}
        if evidence:
            row["evidence"] = f"support for {video_id}"
        rows.append(row)
    return rows


def test_evidence_gate_blocks_rank_strong_specialist_without_payload():
    rows = weighted_video_rrf(
        {"visual": _ranked(["V1", "V2", "V3"]), "asr": _ranked(["TARGET"])},
        {"visual": 1.0, "asr": 1.0},
        topk=3,
        require_specialist_evidence=True,
        specialist_strong_rank=5,
    )
    assert [row["video_id"] for row in rows] == ["V1", "V2", "V3"]
    assert all(row["rrf_guard"] == "visual_baseline" for row in rows)


def test_evidence_and_rank_gate_allow_single_specialist_rescue():
    rows = weighted_video_rrf(
        {"visual": _ranked(["V1", "V2", "V3"]), "asr": _ranked(["TARGET"], evidence=True)},
        {"visual": 1.0, "asr": 1.0},
        topk=3,
        require_specialist_evidence=True,
        specialist_strong_rank=5,
    )
    assert "TARGET" in [row["video_id"] for row in rows]
    assert all(row["rrf_guard"] == "strong_specialist_rescue" for row in rows)


def test_score_gate_is_explicit_and_channel_specific():
    rows = weighted_video_rrf(
        {"visual": _ranked(["V1", "V2"]),
         "asr": _ranked(["TARGET"], evidence=True, start_score=0.1)},
        {"visual": 1.0, "asr": 1.0},
        topk=2,
        require_specialist_evidence=True,
        specialist_min_scores={"asr": 0.5},
    )
    assert [row["video_id"] for row in rows] == ["V1", "V2"]


def test_rescue_can_be_disabled_without_changing_unrestricted_fusion():
    rows = weighted_video_rrf(
        {"visual": _ranked(["V1", "V2"]), "asr": _ranked(["TARGET"])},
        {"visual": 1.0, "asr": 1.0},
        topk=3,
        specialist_rescue_enabled=False,
    )
    assert [row["video_id"] for row in rows] == ["TARGET", "V1", "V2"]


def test_consensus_gate_can_reject_single_strong_specialist():
    rows = weighted_video_rrf(
        {
            "visual": _ranked(["V1", "V2"]),
            "asr": _ranked(["TARGET"], evidence=True),
            "ocr": _ranked(["OCR_OTHER"], evidence=True),
        },
        {"visual": 1.0, "asr": 1.0, "ocr": 1.0},
        topk=3,
        require_specialist_evidence=True,
        min_specialist_channels=2,
        allow_single_strong_rescue=False,
    )
    assert "TARGET" not in [row["video_id"] for row in rows]
