from src.reranking.video_rrf import weighted_video_rrf


def _ids(rows):
    return [row["video_id"] for row in rows]


def _ranked(*video_ids):
    return [{"video_id": video_id, "kf_n": index + 1}
            for index, video_id in enumerate(video_ids)]


def test_visual_only_is_unchanged_and_does_not_mark_a_guard():
    rows = weighted_video_rrf(
        {"visual": _ranked("V1", "V2", "V3")},
        {"visual": 1.0, "asr": 0.5},
        topk=2,
    )

    assert _ids(rows) == ["V1", "V2"]
    assert [row["video_rank"] for row in rows] == [1, 2]
    assert all(row["rrf_guard"] == "none" for row in rows)


def test_weak_specialist_cannot_displace_visual_topk():
    visual = _ranked(*[f"V{index:02d}" for index in range(1, 31)])
    weak_asr = _ranked(
        *[f"V{index:02d}" for index in range(1, 6)],
        "NOISY",
        *[f"V{index:02d}" for index in range(31, 100)],
    )

    rows = weighted_video_rrf(
        {"visual": visual, "asr": weak_asr},
        {"visual": 1.0, "asr": 0.5},
        topk=20,
        specialist_strong_rank=5,
    )

    assert _ids(rows) == [f"V{index:02d}" for index in range(1, 21)]
    assert all(row["rrf_guard"] == "visual_baseline" for row in rows)


def test_strong_specialist_can_rescue_a_video_outside_visual_topk():
    visual = _ranked(*[f"V{index:02d}" for index in range(1, 31)])
    asr = _ranked("TARGET", "V01", "V02")

    rows = weighted_video_rrf(
        {"visual": visual, "asr": asr},
        {"visual": 1.0, "asr": 1.0},
        topk=3,
        specialist_strong_rank=5,
    )

    assert "TARGET" in _ids(rows)
    assert rows[0]["rrf_guard"] == "strong_specialist_rescue"


def test_two_moderate_specialists_count_as_strong_consensus():
    visual = _ranked(*[f"V{index:02d}" for index in range(1, 31)])
    rows = weighted_video_rrf(
        {
            "visual": visual,
            "asr": _ranked("TARGET"),
            "ocr": _ranked("TARGET"),
        },
        {"visual": 1.0, "asr": 0.5, "ocr": 0.5},
        topk=3,
        specialist_strong_rank=1,
        specialist_support_rank=20,
        min_specialist_channels=2,
    )

    assert "TARGET" in _ids(rows)
    assert all(row["rrf_guard"] == "strong_specialist_rescue" for row in rows)


def test_guard_does_not_mutate_input_candidates():
    visual = _ranked("V1", "V2", "V3")
    asr = _ranked("NOISY")
    original_visual = [dict(row) for row in visual]
    original_asr = [dict(row) for row in asr]

    weighted_video_rrf(
        {"visual": visual, "asr": asr},
        {"visual": 1.0, "asr": 0.5},
        topk=2,
    )

    assert visual == original_visual
    assert asr == original_asr
