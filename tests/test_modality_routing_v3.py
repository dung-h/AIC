from src.reranking.video_rrf import collapse_video_ranks, weighted_video_rrf
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


def test_video_rrf_collapses_to_first_rank_per_video():
    rows = [
        {"video_id": "V2", "kf_n": 2},
        {"video_id": "V1", "kf_n": 1},
        {"video_id": "V2", "kf_n": 3},
    ]
    collapsed = collapse_video_ranks(rows)
    assert collapsed["V2"]["rank"] == 1
    assert collapsed["V2"]["kf_n"] == 2
    assert len(collapsed) == 2


def test_video_rrf_can_rescue_wrong_visual_video():
    fused = weighted_video_rrf(
        {"visual": [{"video_id": "wrong"}, {"video_id": "shared"}],
         "asr": [{"video_id": "target"}, {"video_id": "shared"}]},
        {"visual": 1.0, "asr": 1.0}, topk=3,
    )
    assert [row["video_id"] for row in fused][:2] == ["shared", "target"]


def test_routed_allocator_limits_two_frames_per_video():
    candidates = []
    for video_id in ("V1", "V2", "V3"):
        candidates.extend([
            {"video_id": video_id, "kf_n": 1, "source": "visual", "base_score": 1.0},
            {"video_id": video_id, "kf_n": 2, "source": "ocr", "base_score": 0.9},
            {"video_id": video_id, "kf_n": 3, "source": "temporal", "base_score": 0.8},
        ])
    selected = VQAPipelineV3._allocate_routed_candidates(candidates, ["V1", "V2", "V3"], 12)
    assert len(selected) == 6
    assert max(sum(item["video_id"] == video for item in selected) for video in ("V1", "V2", "V3")) <= 2
    assert {item["source"] for item in selected} == {"visual", "ocr"}


def test_routed_allocator_rejects_empty_budget():
    assert VQAPipelineV3._allocate_routed_candidates([], [], 0) == []
