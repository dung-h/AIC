"""Wave 5A tests for the public multimodal TRAKE composition root.

All retrievers are deterministic fakes.  The tests exercise only routing and
the canonical output boundary; they do not load models, indexes, or network
clients.
"""

from __future__ import annotations

import pytest

from src.pipelines.hcmai_pipeline import HCMAIPipeline
from src.runtime_policy import RuntimePolicy


class _FakeEventRetriever:
    def __init__(self, modality: str):
        self.modality = modality
        self.calls = []

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        self.calls.append((event.index, top_k, tuple(candidate_videos or ())))
        allowed = set(map(str, candidate_videos)) if candidate_videos is not None else None
        video_id = "V1"
        if allowed is not None and video_id not in allowed:
            return []
        frame_idx = 10 + 10 * event.index
        return [{
            "event_index": event.index,
            "video_id": video_id,
            "modality": self.modality,
            "score": 1.0 - event.index * 0.01,
            "frame_idx": frame_idx,
            "kf_n": event.index + 1,
            "pts_time": float(event.index + 1),
        }][:top_k]


def _pipeline(retrievers=None, **kwargs):
    return HCMAIPipeline(
        policy=RuntimePolicy(),
        trake_multimodal_retrievers=retrievers,
        **kwargs,
    )


def test_public_trake_routes_explicit_multimodal_to_injected_retrievers():
    visual = _FakeEventRetriever("visual")
    asr = _FakeEventRetriever("asr")
    pipeline = _pipeline({"visual": visual, "asr": asr})

    result = pipeline.trake(
        ["first event", "second event"],
        topk=2,
        mode="multimodal",
    )

    assert result["task"] == "TRAKE"
    assert result["winner"] == "trake_multimodal"
    assert result["mode"] == "multimodal"
    assert result["results"][0]["video_id"] == "V1"
    assert result["results"][0]["frame_ids"] == [10, 20]
    assert [step["frame_idx"] for step in result["results"][0]["path"]] == [10, 20]
    assert [call[0] for call in visual.calls] == [0, 1]
    assert [call[0] for call in asr.calls] == [0, 1]
    assert pipeline._trake is None
    assert pipeline._trake_visual is None
    assert pipeline._trake_multimodal is not None


def test_multimodal_can_be_the_explicit_default_without_widening_runtime_policy():
    visual = _FakeEventRetriever("visual")
    asr = _FakeEventRetriever("asr")
    pipeline = _pipeline(
        {"visual": visual, "asr": asr},
        default_trake_mode="multimodal",
    )

    assert pipeline.policy.trake_mode == "visual"
    result = pipeline.trake(["one event"], topk=1)
    assert result["mode"] == "multimodal"
    assert result["results"][0]["frame_ids"] == [10]


def test_multimodal_missing_retrievers_fails_closed_without_legacy_fallback():
    pipeline = _pipeline()
    pipeline._ensure_trake_visual = lambda: pytest.fail("visual fallback was invoked")
    pipeline._ensure_trake = lambda: pytest.fail("ASR fallback was invoked")

    with pytest.raises(RuntimeError, match="multimodal TRAKE requires injected"):
        pipeline.trake(["one event"], topk=1, mode="multimodal")


def test_multimodal_rejects_non_increasing_or_mismatched_frame_ids():
    pipeline = _pipeline({"visual": _FakeEventRetriever("visual")})

    class _BadBackend:
        def align(self, events, **kwargs):
            return [{
                "video_id": "V1",
                "frame_ids": [10, 30],
                "path": [
                    {"frame_idx": 10, "pts_time": 1.0},
                    {"frame_idx": 20, "pts_time": 2.0},
                ],
            }]

    pipeline._trake_multimodal = _BadBackend()
    with pytest.raises(RuntimeError, match="disagree with its canonical path"):
        pipeline.trake(["first", "second"], topk=1, mode="multimodal")


def test_topk_validation_applies_to_explicit_multimodal_path():
    pipeline = _pipeline({"visual": _FakeEventRetriever("visual")})
    with pytest.raises(ValueError, match="between 1 and 100"):
        pipeline.trake(["one event"], topk=101, mode="multimodal")


def test_legacy_visual_shape_and_owner_remain_unchanged_with_injection():
    pipeline = _pipeline({"visual": _FakeEventRetriever("visual")})

    class _FakeVisual:
        def search(self, events, **kwargs):
            return {
                "results": [{
                    "video_id": "V1",
                    "path": [
                        {"frame_idx": 10 + index, "pts_time": float(index + 1)}
                        for index, _ in enumerate(events)
                    ],
                }],
                "diagnostics": {"fake": True},
            }

    pipeline._trake_visual = _FakeVisual()
    result = pipeline.trake(["one event"], topk=1, mode="visual")

    assert result["winner"] == "trake_visual"
    assert result["mode"] == "visual"
    assert result["results"][0]["path"][0]["frame_idx"] == 10
    assert "frame_ids" not in result["results"][0]
    assert pipeline._trake_multimodal is None
