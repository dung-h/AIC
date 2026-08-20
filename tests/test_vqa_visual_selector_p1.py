import unittest

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


def _candidates(video_ids, frames=3):
    rows = []
    for video_index, video_id in enumerate(video_ids):
        for frame_index in range(frames):
            rows.append({
                "video_id": video_id,
                "kf_n": video_index * 10 + frame_index,
                "source": "visual",
                "base_score": 1.0 - frame_index * 0.1,
            })
    return rows


class VisualSelectorTests(unittest.TestCase):
    def test_balanced_covers_ranked_videos_before_depth(self):
        video_ids = [f"V{i}" for i in range(1, 9)]
        selected = VQAPipelineV3._allocate_visual_candidates(
            _candidates(video_ids), video_ids, 12,
        )
        self.assertEqual([item["video_id"] for item in selected[:8]], video_ids)
        self.assertEqual([item["video_id"] for item in selected[8:]], ["V1", "V2", "V3", "V4"])

    def test_legacy_policy_is_preserved_for_ab(self):
        video_ids = ["V1", "V2", "V3"]
        selected = VQAPipelineV3._allocate_visual_candidates(
            _candidates(video_ids), video_ids, 4, policy="legacy",
        )
        self.assertEqual([item["video_id"] for item in selected], ["V1", "V1", "V1", "V2"])

    def test_full_materialization_is_not_truncated_by_balanced_policy(self):
        video_ids = ["V1", "V2"]
        candidates = _candidates(video_ids)
        selected = VQAPipelineV3._allocate_visual_candidates(
            candidates, video_ids, len(candidates),
        )
        self.assertEqual(len(selected), len(candidates))


if __name__ == "__main__":
    unittest.main()
