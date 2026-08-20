import pytest

from src.trake.multimodal import (
    COVERAGE_COHERENT_ALIGNMENT_POLICY,
    EventLevelMultimodalDante,
)


class SyntheticRetriever:
    def __init__(self, modality, rows):
        self.modality = modality
        self.rows = rows

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        allowed = set(map(str, candidate_videos or ())) if candidate_videos is not None else None
        output = []
        for row in self.rows.get(event.index, []):
            if allowed is not None and str(row["video_id"]) not in allowed:
                continue
            output.append({**row, "event_index": event.index, "modality": self.modality})
        return output[:top_k]


def _row(video_id, frame_idx, pts_time, score, *, kf_n=None, source_id=""):
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "pts_time": pts_time,
        "score": score,
        "kf_n": kf_n,
        "source_id": source_id,
    }


def test_opt_in_policy_supports_mixed_modality_per_event_and_strict_order():
    visual = SyntheticRetriever(
        "visual",
        {
            0: [
                _row("v1", 10, 1.0, 0.90, kf_n=1, source_id="visual-strong"),
                # Same canonical frame: this must be provenance only, not a
                # second timeline position.
                _row("v1", 10, 1.0, 0.80, kf_n=1, source_id="visual-duplicate"),
            ]
        },
    )
    asr = SyntheticRetriever(
        "asr",
        {1: [_row("v1", 20, 2.0, 0.95, kf_n=2, source_id="asr-1")]},
    )

    result = EventLevelMultimodalDante(
        {"visual": visual, "asr": asr},
        alignment_policy=COVERAGE_COHERENT_ALIGNMENT_POLICY,
    ).align(
        [
            {"description": "visible event", "required_modalities": ["visual"]},
            {"description": "spoken event", "required_modalities": ["asr"]},
        ]
    )

    assert result["diagnostics"]["alignment_policy"] == COVERAGE_COHERENT_ALIGNMENT_POLICY
    assert [item["frame_idx"] for item in result["results"][0]["path"]] == [10, 20]
    assert result["results"][0]["modalities"] == ["visual", "asr"]
    assert result["results"][0]["coverage"] == pytest.approx(1.0)
    assert result["results"][0]["policy_diagnostics"]["selected_modality_support"] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    assert result["diagnostics"]["fused_candidate_count"] == 2


def test_new_policy_drops_video_with_missing_event_support():
    visual = SyntheticRetriever(
        "visual",
        {
            0: [
                _row("partial", 10, 1.0, 0.95),
                _row("complete", 11, 1.1, 0.90),
            ],
            1: [_row("complete", 20, 2.0, 0.90)],
        },
    )

    result = EventLevelMultimodalDante(
        {"visual": visual},
        alignment_policy=COVERAGE_COHERENT_ALIGNMENT_POLICY,
    ).align(["event one", "event two"])

    assert [item["video_id"] for item in result["results"]] == ["complete"]
    assert result["diagnostics"]["coverage_by_video"]["partial"]["coverage"] == pytest.approx(0.5)
    assert result["diagnostics"]["candidate_video_count"] == 1


def test_new_policy_preserves_monotonic_path_when_raw_best_frames_conflict():
    visual = SyntheticRetriever(
        "visual",
        {
            0: [
                _row("v1", 20, 2.0, 0.99),
                _row("v1", 10, 1.0, 0.70),
            ],
            1: [
                _row("v1", 10, 1.0, 0.99),
                _row("v1", 30, 3.0, 0.70),
            ],
        },
    )

    result = EventLevelMultimodalDante(
        {"visual": visual},
        alignment_policy=COVERAGE_COHERENT_ALIGNMENT_POLICY,
    ).align(["first event", "second event"])

    assert result["results"]
    frame_ids = result["results"][0]["frame_ids"]
    assert frame_ids == sorted(frame_ids)
    assert all(left < right for left, right in zip(frame_ids, frame_ids[1:]))
    assert result["results"][0]["path"][0]["pts_time"] < result["results"][0]["path"][1]["pts_time"]


def test_legacy_policy_remains_default_and_does_not_emit_new_score_fields():
    visual = SyntheticRetriever(
        "visual",
        {
            0: [_row("v1", 10, 1.0, 0.90)],
            1: [_row("v1", 20, 2.0, 0.80)],
        },
    )

    result = EventLevelMultimodalDante({"visual": visual}).align(["one", "two"])

    assert result["diagnostics"]["alignment_policy"] == "legacy"
    assert set(result["results"][0]) == {
        "video_id",
        "score",
        "path",
        "frame_ids",
        "modalities",
    }
