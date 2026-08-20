from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.vqa.evidence_fusion import build_evidence_packet
from src.vqa.selector import (
    allocate_recall_preserving_candidates,
    deduplicate_candidates,
    selector_metrics,
)


def _candidates(video_ids):
    rows = []
    for video_id in video_ids:
        rows.extend([
            {"video_id": video_id, "kf_n": 1, "source": "visual", "base_score": 1.0},
            {"video_id": video_id, "kf_n": 2, "source": "ocr", "base_score": 0.9},
            {"video_id": video_id, "kf_n": 3, "source": "temporal", "base_score": 0.8},
        ])
    return rows


def test_allocator_covers_ranked_videos_before_second_frame():
    video_ids = [f"V{i}" for i in range(1, 9)]
    selected = VQAPipelineV3._allocate_routed_candidates(
        _candidates(video_ids), video_ids, max_vlm_candidates=12,
    )

    assert len(selected) == 12
    assert [item["video_id"] for item in selected[:8]] == video_ids
    assert [item["video_id"] for item in selected[8:]] == ["V1", "V2", "V3", "V4"]
    assert all(item["kf_n"] == 1 for item in selected[:8])
    assert all(item["kf_n"] == 2 for item in selected[8:])


def test_allocator_keeps_source_priority_and_deduplicates_keyframes():
    video_ids = ["V1"]
    candidates = [
        {"video_id": "V1", "kf_n": 1, "source": "asr", "base_score": 0.99},
        {"video_id": "V1", "kf_n": 1, "source": "visual", "base_score": 0.10},
        {"video_id": "V1", "kf_n": 2, "source": "temporal", "base_score": 0.95},
        {"video_id": "V1", "kf_n": 3, "source": "ocr", "base_score": 0.20},
    ]

    selected = VQAPipelineV3._allocate_routed_candidates(candidates, video_ids, 2)

    assert [(item["kf_n"], item["source"]) for item in selected] == [
        (1, "visual"), (3, "ocr"),
    ]
    assert len({(item["video_id"], item["kf_n"]) for item in selected}) == 2


def test_allocator_uses_specialist_or_temporal_when_visual_is_missing():
    video_ids = ["V1", "V2", "V3"]
    candidates = [
        {"video_id": "V1", "kf_n": 4, "source": "temporal", "base_score": 0.1},
        {"video_id": "V2", "kf_n": 5, "source": "asr", "base_score": 0.2},
        {"video_id": "V3", "kf_n": 6, "source": "ocr", "base_score": 0.3},
    ]

    selected = VQAPipelineV3._allocate_routed_candidates(candidates, video_ids, 3)

    assert [item["video_id"] for item in selected] == video_ids
    assert [item["source"] for item in selected] == ["temporal", "asr", "ocr"]


def test_allocator_budget_smaller_than_video_count_preserves_rank_order():
    video_ids = ["V1", "V2", "V3", "V4"]
    selected = VQAPipelineV3._allocate_routed_candidates(
        _candidates(video_ids), video_ids, max_vlm_candidates=3,
    )

    assert [item["video_id"] for item in selected] == ["V1", "V2", "V3"]
    assert all(item["kf_n"] == 1 for item in selected)


def test_allocator_deduplicates_duplicate_ranked_video_ids():
    video_ids = ["V1", "V1", "V2"]
    selected = VQAPipelineV3._allocate_routed_candidates(
        _candidates(["V1", "V2"]), video_ids, max_vlm_candidates=2,
    )

    assert [item["video_id"] for item in selected] == ["V1", "V2"]


def test_allocator_reserves_specialist_frames_for_routed_query():
    video_ids = [f"V{i}" for i in range(1, 9)]
    selected = VQAPipelineV3._allocate_routed_candidates(
        _candidates(video_ids), video_ids, max_vlm_candidates=12,
        specialist_modalities=["ocr"],
    )

    assert len(selected) == 12
    assert sum(item["source"] == "ocr" for item in selected) >= 3
    assert all(item["source"] in {"visual", "ocr"} for item in selected)
    assert len({item["video_id"] for item in selected if item["source"] == "ocr"}) >= 3


def test_routed_compatibility_allocator_keeps_required_specialist_anchor():
    """A routed query must expose its required specialist modality.

    This reproduces the old failure mode: specialist scores are much higher
    than visual scores for every video, but the 12-frame budget can only cover
    the first 12 ranked videos.  The allocator must keep those video ranks
    while choosing the specialist frame for a spoken/text query.
    """
    video_ids = [f"V{i:02d}" for i in range(1, 21)]
    candidates = []
    for rank, video_id in enumerate(video_ids):
        candidates.extend([
            {
                "video_id": video_id,
                "kf_n": rank * 10 + 1,
                "frame_idx": rank * 100 + 1,
                "source": "visual",
                "base_score": 0.01,
            },
            {
                "video_id": video_id,
                "kf_n": rank * 10 + 2,
                "frame_idx": rank * 100 + 2,
                "source": "asr",
                "modality_score": 0.99,
                "text": "spoken evidence",
            },
        ])

    selected = VQAPipelineV3._allocate_routed_candidates(
        candidates,
        video_ids,
        max_vlm_candidates=12,
        specialist_modalities=["asr"],
    )

    assert [row["video_id"] for row in selected] == video_ids[:12]
    assert all(row["source"] == "asr" for row in selected)


def test_routed_compatibility_allocator_deduplicates_before_budgeting():
    """A duplicate specialist hit cannot consume a second budget slot."""
    candidates = [
        {"video_id": "V1", "kf_n": 1, "frame_idx": 10,
         "source": "visual", "base_score": 0.2},
        {"video_id": "V1", "kf_n": 1, "frame_idx": 10,
         "source": "asr", "modality_score": 0.99,
         "text": "spoken evidence"},
        {"video_id": "V1", "kf_n": 2, "frame_idx": 20,
         "source": "visual", "base_score": 0.8},
    ]

    selected = VQAPipelineV3._allocate_routed_candidates(
        candidates, ["V1"], max_vlm_candidates=2,
        specialist_modalities=["asr"],
    )

    assert [(row["video_id"], row["kf_n"]) for row in selected] == [
        ("V1", 1), ("V1", 2),
    ]
    assert selected[0]["sources"] == ["visual", "asr"]


def test_allocator_does_not_compare_raw_scores_across_modalities():
    """Channel scale differences cannot reorder the ranked video anchors."""
    video_ids = ["V1", "V2", "V3", "V4", "V5"]
    candidates = []
    for rank, video_id in enumerate(video_ids, start=1):
        candidates.extend([
            {"video_id": video_id, "kf_n": rank * 10 + 1,
             "source": "visual", "base_score": 1000.0 - rank},
        {"video_id": video_id, "kf_n": rank * 10 + 2,
             "source": "asr", "modality_score": 0.001 * rank,
             "retrieval_rank": rank, "text": "spoken evidence"},
        ])

    result = allocate_recall_preserving_candidates(
        candidates, video_ids, max_vlm_candidates=4,
        specialist_modalities=["asr"], specialist_reservation=0,
        temporal_reservation=0,
    )

    assert [row["video_id"] for row in result] == video_ids[:4]
    assert all(row["source"] == "asr" for row in result)
    assert result.diagnostics["selection_stages"] == {"specialist_anchor": 4}


def test_adaptive_depth_is_ranked_video_round_robin_within_cap():
    """Adaptive depth cannot let a high raw score starve earlier video ranks."""
    video_ids = [f"V{i}" for i in range(1, 6)]
    candidates = []
    for rank, video_id in enumerate(video_ids, start=1):
        candidates.extend([
            {"video_id": video_id, "kf_n": 1, "source": "visual",
             "base_score": 0.9},
            {"video_id": video_id, "kf_n": 2, "source": "visual",
             "base_score": 100.0 if video_id == "V5" else 0.1},
            {"video_id": video_id, "kf_n": 3, "source": "temporal_neighbor",
             "base_score": 1000.0},
        ])

    result = allocate_recall_preserving_candidates(
        candidates, video_ids, max_vlm_candidates=5,
        specialist_reservation=0, temporal_reservation=0,
        selection_policy="adaptive", per_video_cap=2,
    )

    assert [(row["video_id"], row["kf_n"]) for row in result] == [
        ("V1", 1), ("V2", 1), ("V3", 1), ("V4", 1), ("V1", 2),
    ]
    assert max(sum(row["video_id"] == video_id for row in result)
               for video_id in video_ids) <= 2

    repeat = allocate_recall_preserving_candidates(
        candidates, video_ids, max_vlm_candidates=5,
        specialist_reservation=0, temporal_reservation=0,
        selection_policy="adaptive", per_video_cap=2,
    )
    assert [dict(row) for row in result] == [dict(row) for row in repeat]


def test_dedupe_preserves_multimodal_provenance_for_same_frame():
    pool = deduplicate_candidates([
        {"video_id": "V1", "kf_n": 7, "frame_idx": 70,
         "source": "visual", "base_score": 0.8, "video_rank": 0},
        {"video_id": "V1", "kf_n": 7, "frame_idx": 70,
         "source": "asr", "modality_score": 0.95, "video_rank": 0},
    ])

    assert len(pool) == 1
    assert pool[0]["source"] == "visual"
    assert pool[0]["sources"] == ["visual", "asr"]
    assert {item["source"] for item in pool[0]["provenance"]} == {"visual", "asr"}

    selected = VQAPipelineV3._allocate_routed_candidates(
        pool, ["V1"], max_vlm_candidates=1, specialist_modalities=["asr"],
    )
    assert len(selected) == 1
    assert selected[0]["sources"] == ["visual", "asr"]


def test_selector_metrics_reports_coverage_and_recall_without_fabrication():
    pool = [
        {"video_id": "V1", "kf_n": 1, "source": "visual"},
        {"video_id": "V2", "kf_n": 2, "source": "asr"},
        {"video_id": "V3", "kf_n": 3, "source": "ocr"},
    ]
    selected = [pool[0], pool[2]]
    report = selector_metrics(
        pool, selected, ["V1", "V2", "V3"],
        relevant_keys=[("V1", 1), ("V3", 3)],
    )

    assert report["selected_video_count"] == 2
    assert report["video_coverage"] == 2 / 3
    assert report["relevant_recall"] == 1.0
    assert report["source_counts"] == {"ocr": 1, "visual": 1}
    assert report["dedupe_collisions"] == 0


def test_evidence_packet_does_not_fabricate_cross_video_or_out_of_window_text():
    candidate = {
        "video_id": "V1", "frame_idx": 70, "kf_n": 7,
        "pts_time": 10.0, "frame_path": "frame.jpg",
    }
    packet = build_evidence_packet(
        candidate,
        asr_rows=[
            {"vid": "V2", "start": 10.0, "end": 11.0, "chunk": "wrong video"},
            {"vid": "V1", "start": 100.0, "end": 101.0, "chunk": "wrong time"},
        ],
        ocr_rows=[
            {"video_id": "V2", "pts_time": 10.0, "ocr_text": "wrong video"},
        ],
        query="weather", question="What is spoken?",
    )

    assert packet["sources"] == ["visual"]
    assert packet["asr_chunks"] == []
    assert packet["ocr_text"] == []
    assert packet["timestamps"] == [{
        "source": "visual", "start_time": 10.0,
        "end_time": 10.0, "frame_idx": 70,
    }]


def test_adaptive_allocator_never_compares_raw_scores_across_channels():
    """Depth follows video/channel rank, not incomparable score magnitudes."""
    candidates = [
        {"video_id": "V1", "kf_n": 1, "source": "visual", "base_score": 0.20},
        {"video_id": "V1", "kf_n": 2, "source": "visual", "base_score": 0.19},
        {"video_id": "V2", "kf_n": 1, "source": "visual", "base_score": 0.90},
        {"video_id": "V2", "kf_n": 2, "source": "visual", "base_score": 0.89},
        {"video_id": "V3", "kf_n": 1, "source": "visual", "base_score": 0.80},
    ]

    result = allocate_recall_preserving_candidates(
        candidates, ["V1", "V2", "V3"], max_candidates=4,
        selection_policy="adaptive",
    )

    assert [(row["video_id"], row["kf_n"]) for row in result.selected] == [
        ("V1", 1), ("V2", 1), ("V3", 1), ("V1", 2),
    ]
    assert result.diagnostics["allocator"] == "adaptive_quality_v1"


def test_required_specialist_evidence_wins_anchor_without_gt_oracle():
    """A real ASR/OCR evidence row can replace visual only for that route."""
    candidates = [
        {"video_id": "V1", "kf_n": 1, "source": "visual", "base_score": 0.99},
        {"video_id": "V1", "kf_n": 2, "source": "asr", "modality_score": 0.01,
         "text": "Nha Trang 25 độ"},
        {"video_id": "V2", "kf_n": 1, "source": "visual", "base_score": 0.80},
        {"video_id": "V3", "kf_n": 1, "source": "visual", "base_score": 0.70},
    ]

    result = allocate_recall_preserving_candidates(
        candidates, ["V1", "V2", "V3"], max_candidates=2,
        specialist_modalities=["asr"],
    )

    assert [(row["video_id"], row["kf_n"]) for row in result.selected] == [
        ("V1", 2), ("V2", 1),
    ]
    assert result.diagnostics["selection_stages"]["specialist_anchor"] == 1


def test_selector_dedupes_temporal_neighbor_and_honors_per_video_cap():
    candidates = [
        {"video_id": "V1", "kf_n": 5, "source": "visual", "base_score": 0.9},
        {"video_id": "V1", "kf_n": 5, "source": "temporal", "base_score": 0.2},
        {"video_id": "V1", "kf_n": 6, "source": "neighbor", "base_score": 0.8},
        {"video_id": "V1", "kf_n": 7, "source": "visual", "base_score": 0.7},
        {"video_id": "V2", "kf_n": 1, "source": "visual", "base_score": 0.6},
    ]

    result = allocate_recall_preserving_candidates(
        candidates, ["V1", "V2"], max_candidates=4, per_video_cap=2,
    )

    keys = [(row["video_id"], row["kf_n"]) for row in result.selected]
    assert len(keys) == len(set(keys))
    assert keys[:2] == [("V1", 5), ("V2", 1)]
    assert sum(row["video_id"] == "V1" for row in result.selected) == 2


def test_selector_order_is_deterministic_for_reversed_mixed_input():
    candidates = [
        {"video_id": "V1", "kf_n": 1, "source": "visual", "base_score": 0.4},
        {"video_id": "V1", "kf_n": 2, "source": "ocr", "modality_score": 99.0,
         "text": "weather"},
        {"video_id": "V2", "kf_n": 1, "source": "visual", "base_score": 0.8},
        {"video_id": "V2", "kf_n": 2, "source": "ocr", "modality_score": 1.0,
         "text": "temperature"},
    ]

    first = allocate_recall_preserving_candidates(
        candidates, ["V1", "V2"], max_candidates=3,
        specialist_modalities=["ocr"], selection_policy="adaptive",
    )
    second = allocate_recall_preserving_candidates(
        list(reversed(candidates)), ["V1", "V2"], max_candidates=3,
        specialist_modalities=["ocr"], selection_policy="adaptive",
    )

    assert [(row["video_id"], row["kf_n"]) for row in first.selected] == [
        (row["video_id"], row["kf_n"]) for row in second.selected
    ]
