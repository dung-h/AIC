"""Contract tests for the opt-in same-video multi-candidate TRAKE policy."""

import numpy as np
import pandas as pd

from src.pipelines.trake_visual import VisualTrakeDante


def _two_video_aligner():
    """V1 wins retrieval but has reversed evidence; V2 has the true sequence."""
    metadata = pd.DataFrame(
        {
            "video_id": ["V1", "V1", "V2", "V2"],
            "pts_time": [1.0, 2.0, 1.0, 2.0],
            "kf_n": [1, 2, 1, 2],
            "frame_idx": [10, 20, 30, 40],
        }
    )
    features = np.asarray(
        [
            [0.10, 1.00],  # V1: event 2 appears first
            [1.00, 0.10],  # V1: event 1 appears second
            [0.70, 0.00],  # V2: event 1 then event 2
            [0.00, 0.70],
        ],
        dtype=np.float32,
    )
    event_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return VisualTrakeDante(metadata, features, lambda _: event_vectors)


def test_multivideo_policy_retains_lower_ranked_video_and_can_promote_sequence_evidence():
    result = _two_video_aligner().align(
        ["event one", "event two"],
        alignment_policy="multi_video_v1",
        candidate_video_limit=2,
        top_k_videos=2,
        lam=0.0,
        video_relevance_weight=0.25,
        alignment_evidence_weight=0.75,
    )

    assert [item["video_id"] for item in result["results"]] == ["V2", "V1"]
    assert [item["video_id"] for item in result["diagnostics"]["candidate_videos"]] == ["V1", "V2"]
    assert result["diagnostics"]["candidate_count"] == 2
    assert result["diagnostics"]["ranking"] == "normalized_video_relevance_plus_alignment_evidence"
    provenance = result["results"][0]["provenance"]
    assert provenance["alignment_policy"] == "multi_video_v1"
    assert provenance["candidate_video_rank"] == 2
    assert provenance["video_relevance_normalized"] == 0.0
    assert provenance["alignment_score_normalized"] == 1.0
    assert provenance["ranking_weights"] == {
        "video_relevance": 0.25,
        "alignment_evidence": 0.75,
    }


def test_multivideo_answers_never_mix_videos_and_keep_full_canonical_order():
    result = _two_video_aligner().align(
        ["event one", "event two"],
        alignment_policy="multi_video_v1",
        candidate_video_limit=2,
        top_k_videos=2,
        lam=0.0,
    )

    for answer in result["results"]:
        assert len(answer["path"]) == 2
        assert [step["video_id"] for step in answer["path"]] == [answer["video_id"]] * 2
        assert answer["frame_ids"] == [step["frame_idx"] for step in answer["path"]]
        assert all(
            left < right for left, right in zip(answer["frame_ids"], answer["frame_ids"][1:])
        )
        assert all(
            left["kf_n"] < right["kf_n"] for left, right in zip(answer["path"], answer["path"][1:])
        )


def test_multivideo_default_keeps_the_legacy_three_x_candidate_budget():
    video_ids = [f"V{index:02d}" for index in range(40)]
    metadata = pd.DataFrame(
        {
            "video_id": [video_id for video_id in video_ids for _ in range(2)],
            "pts_time": [time for _ in video_ids for time in (1.0, 2.0)],
            "kf_n": [keyframe for _ in video_ids for keyframe in (1, 2)],
            "frame_idx": [
                100 * index + frame
                for index in range(40)
                for frame in (1, 2)
            ],
        }
    )
    features = np.tile(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        (40, 1),
    )
    aligner = VisualTrakeDante(
        metadata,
        features,
        lambda _: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    result = aligner.align(
        ["event one", "event two"],
        alignment_policy="multi_video_v1",
        top_k_videos=10,
        lam=0.0,
    )

    assert result["diagnostics"]["candidate_video_limit"] == 30
    assert result["diagnostics"]["candidate_count"] == 30


def test_legacy_policy_is_unchanged_when_multivideo_flag_is_off():
    aligner = _two_video_aligner()
    implicit = aligner.align(["event one", "event two"], top_k_videos=1, lam=0.0)
    explicit = aligner.align(
        ["event one", "event two"],
        top_k_videos=1,
        lam=0.0,
        alignment_policy="legacy",
    )

    assert implicit["results"] == explicit["results"]
    for key in ("mode", "candidate_count", "scored_count", "lattice_enabled"):
        assert implicit["diagnostics"][key] == explicit["diagnostics"][key]
    assert "provenance" not in implicit["results"][0]
    assert "frame_ids" not in implicit["results"][0]
