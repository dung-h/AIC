"""Wave 2B contract tests for the unified video-fusion owner."""

from dataclasses import dataclass

from src.reranking.video_rrf import collapse_video_ranks, weighted_video_rrf


@dataclass(frozen=True)
class EvidenceHit:
    video_id: str
    frame_idx: int
    rank: int
    evidence: str = ""


def _ids(rows):
    return [row["video_id"] for row in rows]


def _ranked(*video_ids):
    return [
        {"video_id": video_id, "frame_idx": index, "kf_n": index}
        for index, video_id in enumerate(video_ids, 1)
    ]


def test_rank_best_collapse_accepts_dataclass_evidence_hits():
    collapsed = collapse_video_ranks(
        [
            EvidenceHit("V1", frame_idx=50, rank=5),
            EvidenceHit("V1", frame_idx=20, rank=2, evidence="best"),
            EvidenceHit("V2", frame_idx=1, rank=1),
        ]
    )

    assert list(collapsed) == ["V1", "V2"]
    assert collapsed["V1"]["rank"] == 2
    assert collapsed["V1"]["frame_idx"] == 20
    assert collapsed["V1"]["evidence"] == "best"


def test_weighted_rrf_uses_channel_union_and_best_rank_once():
    rows = weighted_video_rrf(
        {
            "visual": _ranked("V1", "V2"),
            "asr": _ranked("V2", "TARGET"),
        },
        {"visual": 1.0, "asr": 1.0},
        topk=3,
    )

    assert _ids(rows)[0] == "V2"
    assert set(_ids(rows)) == {"V1", "V2", "TARGET"}
    v2 = rows[0]
    assert v2["visual_rank"] == 2
    assert v2["asr_rank"] == 1
    assert v2["rrf_score"] > rows[-1]["rrf_score"]


def test_specialist_channel_can_rescue_video_into_bounded_topk():
    rows = weighted_video_rrf(
        {
            "visual": _ranked("V1", "V2", "V3"),
            "asr": _ranked("TARGET"),
            "ocr": _ranked("TARGET"),
        },
        {"visual": 1.0, "asr": 1.0, "ocr": 1.0},
        topk=2,
        min_specialist_channels=2,
        allow_single_strong_rescue=False,
    )

    assert len(rows) == 2
    assert "TARGET" in _ids(rows)
    assert all(row["rrf_guard"] == "strong_specialist_rescue" for row in rows)


def test_duplicate_frames_do_not_create_extra_channel_votes():
    rows = weighted_video_rrf(
        {
            "visual": [
                {"video_id": "V1", "frame_idx": 10},
                {"video_id": "V1", "frame_idx": 11},
                {"video_id": "V2", "frame_idx": 20},
            ]
        },
        {"visual": 1.0},
        topk=2,
    )

    assert _ids(rows) == ["V1", "V2"]
    assert rows[0]["visual_rank"] == 1
    assert rows[0]["visual_candidate"]["frame_idx"] == 10
    assert rows[0]["rrf_score"] == 1 / 61


def test_missing_or_empty_specialist_channel_is_not_negative_evidence():
    visual = _ranked("V1", "V2")
    omitted = weighted_video_rrf(
        {"visual": visual},
        {"visual": 1.0, "asr": 1.0, "ocr": 1.0},
        topk=2,
    )
    empty = weighted_video_rrf(
        {"visual": visual, "asr": [], "ocr": []},
        {"visual": 1.0, "asr": 1.0, "ocr": 1.0},
        topk=2,
    )

    assert _ids(omitted) == _ids(empty) == ["V1", "V2"]
    assert [row["rrf_score"] for row in omitted] == [row["rrf_score"] for row in empty]


def test_deterministic_tie_break_is_video_id_not_input_order():
    rows = weighted_video_rrf(
        {
            "visual": [
                {"video_id": "V2", "rank": 1},
                {"video_id": "V1", "rank": 1},
            ],
            "asr": [],
        },
        {"visual": 1.0, "asr": 1.0},
        topk=2,
    )

    assert _ids(rows) == ["V1", "V2"]
    assert [row["video_rank"] for row in rows] == [1, 2]


def test_max_output_budget_is_enforced_even_for_large_channel_union():
    rows = weighted_video_rrf(
        {
            "visual": _ranked(*[f"V{index:03d}" for index in range(100)]),
            "asr": _ranked(*[f"A{index:03d}" for index in range(100)]),
            "ocr": _ranked(*[f"O{index:03d}" for index in range(100)]),
        },
        {"visual": 1.0, "asr": 0.5, "ocr": 0.5},
        topk=7,
        specialist_rescue_enabled=False,
    )

    assert len(rows) == 7
    assert [row["video_rank"] for row in rows] == list(range(1, 8))
    assert weighted_video_rrf({"visual": _ranked("V1")}, {"visual": 1.0}, topk=0) == []


def test_missing_specialist_evidence_cannot_trigger_rescue_when_required():
    rows = weighted_video_rrf(
        {"visual": _ranked("V1", "V2"), "asr": _ranked("TARGET")},
        {"visual": 1.0, "asr": 1.0},
        topk=2,
        require_specialist_evidence=True,
        specialist_strong_rank=1,
    )

    assert _ids(rows) == ["V1", "V2"]
    assert all(row["rrf_guard"] == "visual_baseline" for row in rows)
