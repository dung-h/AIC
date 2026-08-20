"""Focused contracts for the Q&A anchor-preserving candidate allocator."""

import pandas as pd

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


def _pipeline(rows):
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.km = pd.DataFrame(rows)
    return pipeline


def _row(video_id, kf_n, frame_idx):
    return {
        "video_id": video_id,
        "kf_n": kf_n,
        "frame_idx": frame_idx,
        "pts_time": float(kf_n),
    }


def _visual(video_id, kf_n, frame_idx, score=0.9):
    return {
        "video_id": video_id,
        "kf_n": kf_n,
        "frame_idx": frame_idx,
        "source": "visual",
        "base_score": score,
    }


def test_anchor_policy_spends_budget_on_competitive_videos_before_depth():
    video_ids = [f"V{index:02d}" for index in range(1, 21)]
    rows = []
    candidates = []
    for index, video_id in enumerate(video_ids, start=1):
        rows.extend([
            _row(video_id, 1, index * 100 + 1),
            _row(video_id, 2, index * 100 + 2),
        ])
        candidates.extend([
            _visual(video_id, 1, index * 100 + 1, 0.9),
            _visual(video_id, 2, index * 100 + 2, 0.8),
        ])

    result = _pipeline(rows)._allocate_anchor_preserving_candidates(
        candidates, video_ids, 12,
    )

    assert [(item["video_id"], item["kf_n"]) for item in result.selected] == [
        (video_id, 1) for video_id in video_ids[:12]
    ]
    assert sum(entry["selected"] for entry in result.diagnostics["selection_trace"]) == 12
    assert result.diagnostics["anchor_video_count"] == 12


def test_specialist_reservation_keeps_visual_anchor_and_real_evidence():
    rows = [
        _row("V1", 1, 101), _row("V1", 2, 102),
        _row("V2", 1, 201), _row("V2", 2, 202),
        _row("V3", 1, 301),
    ]
    candidates = [
        _visual("V1", 1, 101),
        {"video_id": "V1", "kf_n": 2, "frame_idx": 102, "source": "asr",
         "modality_score": 0.01, "text": "Nha Trang 25 độ"},
        _visual("V2", 1, 201),
        _visual("V2", 2, 202, 0.8),
        _visual("V3", 1, 301),
    ]

    result = _pipeline(rows)._allocate_anchor_preserving_candidates(
        candidates, ["V1", "V2", "V3"], 3,
        specialist_modalities=["asr"],
    )

    assert [(item["video_id"], item["kf_n"]) for item in result.selected] == [
        ("V1", 1), ("V2", 1), ("V1", 2),
    ]
    assert result.diagnostics["specialist_reservation_status"] == [{
        "modality": "asr", "fulfilled": True, "selected_count": 1,
    }]
    selected_trace = {
        (entry["video_id"], entry["kf_n"]): entry
        for entry in result.diagnostics["selection_trace"] if entry["selected"]
    }
    assert selected_trace[("V1", 1)]["reason"] == "visual_anchor"
    assert selected_trace[("V1", 2)]["reason"] == "asr_evidence_reservation"


def test_canonical_map_remaps_stale_frame_and_rejects_unknown_keyframe():
    pipeline = _pipeline([_row("V1", 1, 101)])
    result = pipeline._allocate_anchor_preserving_candidates(
        [
            _visual("V1", 1, 999),
            _visual("V1", 2, 202),
        ],
        ["V1"],
        2,
    )

    assert [(item["video_id"], item["kf_n"], item["frame_idx"])
            for item in result.selected] == [("V1", 1, 101)]
    trace = result.diagnostics["selection_trace"]
    assert any(entry["canonical_frame_remapped"] for entry in trace if entry["selected"])
    assert any(entry["reason"] == "canonical_key_not_found" for entry in trace)
    assert result.diagnostics["canonical_map_checked"] is True


def test_specialist_without_payload_does_not_evict_visual_anchor():
    rows = [_row("V1", 1, 101), _row("V1", 2, 102), _row("V2", 1, 201)]
    result = _pipeline(rows)._allocate_anchor_preserving_candidates(
        [
            _visual("V1", 1, 101),
            {"video_id": "V1", "kf_n": 2, "frame_idx": 102, "source": "ocr",
             "modality_score": 1.0},
            _visual("V2", 1, 201),
        ],
        ["V1", "V2"],
        2,
        specialist_modalities=["ocr"],
    )

    assert [(item["video_id"], item["kf_n"]) for item in result.selected] == [
        ("V1", 1), ("V2", 1),
    ]
    assert any(
        entry["reason"] == "specialist_payload_missing"
        for entry in result.diagnostics["selection_trace"]
    )


def test_anchor_policy_selection_and_trace_are_deterministic():
    rows = [
        _row("V1", 1, 101), _row("V1", 2, 102),
        _row("V2", 1, 201), _row("V2", 2, 202),
    ]
    candidates = [
        _visual("V1", 1, 101, 0.5),
        {"video_id": "V1", "kf_n": 2, "frame_idx": 102, "source": "ocr",
         "modality_score": 99.0, "text": "Nhiệt độ"},
        _visual("V2", 1, 201, 0.8),
        _visual("V2", 2, 202, 0.7),
    ]
    pipeline = _pipeline(rows)
    first = pipeline._allocate_anchor_preserving_candidates(
        candidates, ["V1", "V2"], 3, specialist_modalities=["ocr"],
    )
    second = pipeline._allocate_anchor_preserving_candidates(
        list(reversed(candidates)), ["V1", "V2"], 3, specialist_modalities=["ocr"],
    )

    assert [dict(item) for item in first.selected] == [dict(item) for item in second.selected]
    assert first.diagnostics["selection_trace"] == second.diagnostics["selection_trace"]
