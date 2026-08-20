"""Minimal P0 regression tests for Q&A video-level modality rescue.

These tests deliberately stay below the full benchmark: the visual channel
misses the target video, while the specialist channels contain it in their
global top-100 lists.  They verify the safety boundary around video-level RRF.
"""

from src.reranking.video_rrf import weighted_video_rrf


def _ranked(video_ids):
    return [
        {"video_id": video_id, "kf_n": rank, "frame_idx": rank}
        for rank, video_id in enumerate(video_ids, 1)
    ]


def test_strong_asr_ocr_rescue_video_missed_by_visual_top20():
    visual_ids = [f"VISUAL_{rank:02d}" for rank in range(1, 21)]
    asr_ids = ["GROUND_TRUTH"] + [f"ASR_NOISE_{rank:02d}" for rank in range(1, 100)]
    ocr_ids = ["GROUND_TRUTH"] + [f"OCR_NOISE_{rank:02d}" for rank in range(1, 100)]

    rows = weighted_video_rrf(
        {
            "visual": _ranked(visual_ids),
            "asr": _ranked(asr_ids),
            "ocr": _ranked(ocr_ids),
        },
        {"visual": 1.0, "asr": 1.0, "ocr": 1.0},
        topk=20,
        specialist_strong_rank=5,
        specialist_support_rank=20,
        min_specialist_channels=2,
    )

    assert "GROUND_TRUTH" not in visual_ids
    assert len(asr_ids) == len(ocr_ids) == 100
    assert "GROUND_TRUTH" in [row["video_id"] for row in rows]
    rescued = next(row for row in rows if row["video_id"] == "GROUND_TRUTH")
    assert rescued["rrf_guard"] == "strong_specialist_rescue"


def test_weak_specialist_cannot_displace_visual_top20():
    visual_ids = [f"VISUAL_{rank:02d}" for rank in range(1, 21)]
    # Specialist lists agree with the visual anchors first; the target only
    # appears at weak ranks (50 and 79), so no specialist-only strong rescue
    # is present to activate the rescue branch.
    asr_ids = (
        visual_ids
        + [f"ASR_NOISE_{rank:02d}" for rank in range(1, 30)]
        + ["GROUND_TRUTH"]
        + [f"ASR_TAIL_{rank:02d}" for rank in range(1, 51)]
    )
    ocr_ids = (
        visual_ids
        + [f"OCR_NOISE_{rank:02d}" for rank in range(1, 59)]
        + ["GROUND_TRUTH"]
        + [f"OCR_TAIL_{rank:02d}" for rank in range(1, 22)]
    )

    rows = weighted_video_rrf(
        {
            "visual": _ranked(visual_ids),
            "asr": _ranked(asr_ids),
            "ocr": _ranked(ocr_ids),
        },
        # Deliberately large specialist weights ensure this test exercises the
        # guard, rather than merely passing because weak RRF scores are small.
        {"visual": 1.0, "asr": 10.0, "ocr": 10.0},
        topk=20,
        specialist_strong_rank=5,
        specialist_support_rank=20,
        min_specialist_channels=2,
    )

    assert len(asr_ids) == len(ocr_ids) == 100
    assert [row["video_id"] for row in rows] == visual_ids
    assert all(row["rrf_guard"] == "visual_baseline" for row in rows)
