from src.vqa.selector import (
    allocate_recall_preserving_candidates,
    deduplicate_candidates,
)


def _lattice(video_ids, frames=3, *, source="visual"):
    rows = []
    for video_rank, video_id in enumerate(video_ids):
        for offset in range(frames):
            rows.append({
                "video_id": video_id,
                "kf_n": video_rank * 100 + offset,
                "frame_idx": video_rank * 1000 + offset,
                "base_score": 1.0 - offset * 0.01,
                "source": source,
            })
    return rows


def test_allocator_covers_each_eligible_video_before_depth():
    videos = ["V1", "V2", "V3", "V4"]
    result = allocate_recall_preserving_candidates(
        _lattice(videos, frames=3), videos, max_candidates=8,
    )

    assert [row["video_id"] for row in result.selected[:4]] == videos
    assert {row["video_id"] for row in result.selected} == set(videos)
    assert all(
        sum(row["video_id"] == video_id for row in result.selected) <= 2
        for video_id in videos
    )
    assert result.diagnostics["coverage_guaranteed"] is True
    assert result.diagnostics["uncovered_eligible_videos"] == []


def test_allocator_deduplicates_canonical_key_and_preserves_real_provenance():
    candidates = [
        {"video_id": "V1", "kf_n": 7, "frame_idx": 70,
         "source": "visual", "base_score": 0.7},
        {"video_id": "V1", "kf_n": 7, "frame_idx": 70,
         "source": "asr", "modality_score": 0.9},
        {"video_id": "V1", "kf_n": 8, "frame_idx": 80,
         "source": "temporal", "base_score": 0.5},
    ]
    result = allocate_recall_preserving_candidates(
        candidates, ["V1"], max_candidates=2,
        specialist_modalities=["asr"], temporal_reservation=1,
    )

    assert len(result.selected) == 2
    assert len({(row["video_id"], row["kf_n"]) for row in result.selected}) == 2
    merged = next(row for row in result.selected if row["kf_n"] == 7)
    assert merged["sources"] == ["visual", "asr"]
    assert {item["source"] for item in merged["provenance"]} == {"visual", "asr"}
    assert all("frame_idx" in row for row in result.selected)
    assert all(row["frame_idx"] in {70, 80} for row in result.selected)


def test_allocator_reserves_specialist_and_temporal_evidence_without_starvation():
    videos = ["V1", "V2", "V3", "V4"]
    candidates = _lattice(videos, frames=1)
    candidates.extend([
        {"video_id": "V2", "kf_n": 21, "frame_idx": 210,
         "source": "asr", "modality_score": 0.95},
        {"video_id": "V3", "kf_n": 31, "frame_idx": 310,
         "source": "ocr", "modality_score": 0.94},
        {"video_id": "V4", "kf_n": 41, "frame_idx": 410,
         "source": "temporal", "base_score": 0.1},
    ])

    result = allocate_recall_preserving_candidates(
        candidates, videos, max_candidates=8,
        specialist_modalities=["asr", "ocr"],
        specialist_reservation=1,
        temporal_reservation=1,
    )
    selected_videos = [row["video_id"] for row in result.selected]
    sources = {source for row in result.selected for source in row["sources"]}

    assert set(videos).issubset(selected_videos)
    assert {"asr", "ocr", "temporal"}.issubset(sources)
    assert result.diagnostics["specialist_reserved"] == {"asr": 1, "ocr": 1}
    assert result.diagnostics["temporal_reserved"] == 1
    assert max(selected_videos.count(video_id) for video_id in videos) <= 2


def test_allocator_reports_impossible_budget_explicitly():
    videos = ["V1", "V2", "V3", "V4", "V5"]
    result = allocate_recall_preserving_candidates(
        _lattice(videos, frames=2), videos, max_candidates=3,
    )

    assert [row["video_id"] for row in result.selected] == ["V1", "V2", "V3"]
    assert result.impossible_budget_reason is not None
    assert "max_candidates=3" in result.impossible_budget_reason
    assert "eligible_video_count=5" in result.impossible_budget_reason
    assert result.diagnostics["uncovered_eligible_videos"] == ["V4", "V5"]
    assert result.diagnostics["coverage_guaranteed"] is False


def test_allocator_is_deterministic_and_exposes_recall_metrics():
    videos = ["V1", "V2", "V3"]
    candidates = _lattice(videos, frames=3)
    relevant = [("V1", 0), ("V2", 101)]
    first = allocate_recall_preserving_candidates(
        candidates, videos, max_candidates=3, relevant_keys=relevant,
    )
    second = allocate_recall_preserving_candidates(
        list(reversed(candidates)), videos, max_candidates=3,
        relevant_keys=relevant,
    )

    assert [(row["video_id"], row["kf_n"]) for row in first.selected] == [
        (row["video_id"], row["kf_n"]) for row in second.selected
    ]
    assert first.diagnostics["video_coverage"] == 1.0
    assert first.diagnostics["relevant_pool_count"] == 2
    assert first.diagnostics["relevant_recall"] == 0.5
    assert first.diagnostics["dedupe_collisions"] == 0


def test_allocator_never_fabricates_frame_idx_and_dedup_helper_stays_compatible():
    candidates = [
        {"video_id": "V1", "kf_n": 1, "source": "visual"},
        {"video_id": "V2", "kf_n": 2, "source": "visual"},
    ]
    result = allocate_recall_preserving_candidates(
        candidates, ["V1", "V2"], max_candidates=2,
    )

    assert all("frame_idx" not in row for row in result.selected)
    deduped = deduplicate_candidates(candidates)
    assert [(row["video_id"], row["kf_n"]) for row in deduped] == [("V1", 1), ("V2", 2)]
    assert all("frame_idx" not in row for row in deduped)


def test_visual_anchor_is_preserved_before_specialist_with_budget_12():
    """A specialist hit cannot replace the only visual frame of a video."""
    videos = [f"V{i}" for i in range(1, 21)]
    candidates = []
    for rank, video_id in enumerate(videos):
        candidates.extend([
            {
                "video_id": video_id,
                "kf_n": rank * 10 + 1,
                "frame_idx": rank * 100 + 1,
                "source": "visual",
                "base_score": 0.10,
            },
            {
                "video_id": video_id,
                "kf_n": rank * 10 + 2,
                "frame_idx": rank * 100 + 2,
                "source": "asr",
                "modality_score": 0.99,
            },
        ])

    result = allocate_recall_preserving_candidates(
        candidates,
        videos,
        max_candidates=12,
        specialist_modalities=["asr"],
    )

    assert len(result.selected) == 12
    assert [row["video_id"] for row in result.selected] == videos[:12]
    assert [row["kf_n"] for row in result.selected] == [
        rank * 10 + 1 for rank in range(12)
    ]
    assert all("visual" in row["sources"] for row in result.selected)
    assert result.diagnostics["visual_anchor_preservation_rate"] == 1.0
    assert result.diagnostics["visual_anchor_selected_video_ids"] == videos[:12]
    assert result.diagnostics["specialist_reserved"] == {}


def test_visual_anchor_uses_visual_score_after_multimodal_dedupe():
    """A high ASR score on one keyframe cannot beat a stronger visual frame."""
    candidates = [
        {
            "video_id": "V1", "kf_n": 1, "frame_idx": 10,
            "source": "visual", "base_score": 0.10,
        },
        {
            "video_id": "V1", "kf_n": 1, "frame_idx": 10,
            "source": "asr", "modality_score": 0.99,
        },
        {
            "video_id": "V1", "kf_n": 2, "frame_idx": 20,
            "source": "visual", "base_score": 0.80,
        },
    ]

    result = allocate_recall_preserving_candidates(
        candidates,
        ["V1"],
        max_candidates=1,
        specialist_modalities=["asr"],
    )

    assert [(row["video_id"], row["kf_n"]) for row in result.selected] == [("V1", 2)]
    assert result.selected[0]["sources"] == ["visual"]
    assert result.diagnostics["visual_anchor_preservation_rate"] == 1.0


def test_adaptive_policy_spends_surplus_slot_on_strong_within_video_frame():
    candidates = [
        {"video_id": "V1", "kf_n": 1, "frame_idx": 10,
         "source": "visual", "base_score": 0.90},
        {"video_id": "V1", "kf_n": 2, "frame_idx": 20,
         "source": "visual", "base_score": 0.80},
        {"video_id": "V2", "kf_n": 1, "frame_idx": 30,
         "source": "visual", "base_score": 0.80},
        {"video_id": "V3", "kf_n": 1, "frame_idx": 40,
         "source": "visual", "base_score": 0.70},
        {"video_id": "V4", "kf_n": 1, "frame_idx": 50,
         "source": "visual", "base_score": 0.60},
    ]

    result = allocate_recall_preserving_candidates(
        candidates, ["V1", "V2", "V3", "V4"], max_candidates=4,
        selection_policy="adaptive",
    )

    assert [(row["video_id"], row["kf_n"]) for row in result.selected] == [
        ("V1", 1), ("V2", 1), ("V3", 1), ("V1", 2)
    ]
    assert result.diagnostics["selection_policy"] == "adaptive"
    assert result.diagnostics["coverage_floor"] == 3
    assert result.diagnostics["coverage_floor_guaranteed"] is True
    assert result.diagnostics["selection_stages"]["adaptive_utility"] == 1


def test_specialist_is_added_only_after_visual_coverage_has_spare_budget():
    videos = [f"V{i}" for i in range(1, 5)]
    candidates = _lattice(videos, frames=1)
    candidates.extend([
        {
            "video_id": video_id,
            "kf_n": index * 100 + 10,
            "frame_idx": index * 1000 + 10,
            "source": "asr",
            "modality_score": 0.99,
        }
        for index, video_id in enumerate(videos, start=1)
    ])

    result = allocate_recall_preserving_candidates(
        candidates,
        videos,
        max_candidates=6,
        specialist_modalities=["asr"],
    )

    assert [row["video_id"] for row in result.selected[:4]] == videos
    assert all("visual" in row["sources"] for row in result.selected[:4])
    assert len(result.selected) == 6
    assert sum("asr" in row["sources"] for row in result.selected) >= 1
    assert result.diagnostics["selection_stages"]["visual_anchor"] == 4
