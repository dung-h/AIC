"""P0/P1 audit tests for the offline Q&A vertical slice.

This file intentionally contains only contract tests.  It does not patch or
modify production behavior.  A failing test is an actionable production
blocker and should be reported to the owner of the corresponding pipeline.
"""

from __future__ import annotations

from pathlib import Path
import socket
import urllib.request

import pandas as pd

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


class _FakeStructuredVLM:
    """Deterministic local-only VLM used to expose control-flow contracts."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer_with_metadata(self, frame_path, prompt, *, max_new_tokens=128):
        self.calls.append(str(frame_path))
        stem = Path(frame_path).stem
        return {
            "answer": f"answer-{stem}",
            "grounding_score": 0.9,
            "answer_confidence": 0.9,
            "abstain": False,
            "parse_failed": False,
        }


def _answer_pipeline(candidates: list[dict]) -> tuple[VQAPipelineV3, _FakeStructuredVLM]:
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    fake_vlm = _FakeStructuredVLM()
    pipeline._local_vlm = fake_vlm
    prepared = {
        "query": "a test scene",
        "question": "What is visible?",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "vlm_candidate_count": len(candidates),
        "route_active": False,
    }
    return pipeline, fake_vlm


def _candidate(video_id: str, frame_idx: int, kf_n: int) -> dict:
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "kf_n": kf_n,
        "pts_time": float(kf_n),
        "base_score": 1.0 / (kf_n + 1),
        "video_rank": kf_n,
        "frame_path": f"fake_{video_id}_{kf_n}.jpg",
        "source": "visual",
    }


def test_max_answers_does_not_stop_before_rerank():
    """max_answers must truncate after all candidates have been scored.

    This is deliberately an audit assertion.  The current implementation is
    expected to fail because it breaks from the candidate loop as soon as
    ``len(answered) >= max_answers``.
    """

    candidates = [_candidate("V1", 101, 1), _candidate("V2", 202, 2), _candidate("V3", 303, 3)]
    pipeline, fake_vlm = _answer_pipeline(candidates)

    result = pipeline.answer_ranked_candidates(
        {
            "query": "a test scene",
            "question": "What is visible?",
            "candidates": candidates,
            "candidate_count": len(candidates),
            "vlm_candidate_count": len(candidates),
            "route_active": False,
        },
        max_answers=1,
        use_context=False,
        structured_vlm=True,
    )

    assert len(fake_vlm.calls) == len(candidates), (
        "P0 blocker: max_answers truncated before candidate scoring/rerank; "
        f"expected {len(candidates)} VLM calls, got {len(fake_vlm.calls)}"
    )
    assert len(result["answers"]) == 1


def test_answer_result_contains_nonempty_trace():
    """Every answer run must expose a non-empty audit trace."""

    candidates = [_candidate("V1", 101, 1)]
    pipeline, _ = _answer_pipeline(candidates)
    result = pipeline.answer_ranked_candidates(
        {
            "query": "a test scene",
            "question": "What is visible?",
            "candidates": candidates,
            "candidate_count": 1,
            "vlm_candidate_count": 1,
            "route_active": False,
        },
        max_answers=1,
        use_context=False,
        structured_vlm=True,
    )

    trace = result.get("answer_trace")
    assert isinstance(trace, list) and trace, (
        "P0 blocker: answer result has no non-empty answer_trace"
    )


def test_local_vlm_metadata_is_default_when_backend_supports_it():
    candidates = [_candidate("V1", 101, 1)]
    pipeline, _ = _answer_pipeline(candidates)
    result = pipeline.answer_ranked_candidates(
        {
            "query": "a test scene",
            "question": "What is visible?",
            "candidates": candidates,
            "candidate_count": 1,
            "vlm_candidate_count": 1,
            "route_active": False,
        },
        max_answers=1,
        use_context=False,
    )

    assert result["structured_vlm"] is True
    assert result["answers"][0]["grounding_score"] == 0.9
    assert result["answers"][0]["answer_confidence"] == 0.9


def test_routed_budget_covers_distinct_videos_before_second_frames():
    """A 12-frame P1 budget must cover 12 ranked videos first."""

    video_ids = [f"V{i:02d}" for i in range(1, 21)]
    candidates = []
    for index, video_id in enumerate(video_ids, 1):
        candidates.append(_candidate(video_id, index * 10, index))
        candidates.append({
            **_candidate(video_id, index * 10 + 1, index + 100),
            "source": "asr",
            "base_score": 0.99,
        })

    selected = VQAPipelineV3._allocate_routed_candidates(candidates, video_ids, 12)
    selected_videos = [item["video_id"] for item in selected]

    assert len(selected) == 12
    assert len(set(selected_videos)) == 12, (
        "P1 blocker: 12-frame selector does not provide one frame per ranked "
        f"video; selected {len(set(selected_videos))} videos"
    )
    assert all(selected_videos.count(video_id) <= 2 for video_id in set(selected_videos))


def test_route_off_preserves_visual_retrieval_and_does_not_call_specialist():
    """Feature-off behavior must remain visual-only and deterministic."""

    class FakeKIS:
        def search(self, query, topk):
            return [
                ("V1", 101, 1, 0.9),
                ("V2", 202, 2, 0.8),
                ("V3", 303, 3, 0.7),
            ][:topk]

    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = FakeKIS()
    pipeline.km = pd.DataFrame([
        {"video_id": "V1", "kf_n": 1, "frame_idx": 101, "pts_time": 1.0},
        {"video_id": "V2", "kf_n": 2, "frame_idx": 202, "pts_time": 2.0},
        {"video_id": "V3", "kf_n": 3, "frame_idx": 303, "pts_time": 3.0},
    ])
    pipeline._local_candidates = lambda *args, **kwargs: [
        ("V1", 101, 1, 0.9),
        ("V2", 202, 2, 0.8),
        ("V3", 303, 3, 0.7),
    ]
    pipeline._frame_path = lambda video_id, kf_n: f"fake_{video_id}_{kf_n}.jpg"

    class ForbiddenSpecialist:
        def global_candidates(self, *args, **kwargs):
            raise AssertionError("route-off must not call the specialist router")

    baseline = pipeline.prepare_ranked_candidates(
        "scene", "question", top_videos=3, frames_per_video=1,
        max_vlm_candidates=3, required_modalities=None,
    )
    route_off = pipeline.prepare_ranked_candidates(
        "scene", "question", top_videos=3, frames_per_video=1,
        max_vlm_candidates=3, required_modalities="asr",
        global_modality_router=None,
    )

    assert baseline["retrieved_video_ids"] == route_off["retrieved_video_ids"]
    assert baseline["candidates"] == route_off["candidates"]
    assert route_off["route_active"] is False


def test_answer_output_has_canonical_fields_and_nonempty_answer():
    """Local answers must be submission-shaped and grounded to a frame."""

    candidates = [_candidate("V1", 101, 1)]
    pipeline, _ = _answer_pipeline(candidates)
    result = pipeline.answer_ranked_candidates(
        {
            "query": "a test scene",
            "question": "What is visible?",
            "candidates": candidates,
            "candidate_count": 1,
            "vlm_candidate_count": 1,
            "route_active": False,
        },
        max_answers=1,
        use_context=False,
        structured_vlm=True,
    )

    assert result["answers"]
    for answer in result["answers"]:
        assert answer["video_id"]
        assert isinstance(answer["frame_id"], int)
        assert str(answer["answer"]).strip()
        assert str(answer["answer"]).strip().casefold() not in {"evidence-only", "unknown", "n/a"}


def test_offline_answer_path_makes_no_network_call(monkeypatch):
    """The local answer path must stay offline even when network is forbidden."""

    def forbidden(*args, **kwargs):
        raise AssertionError("offline Q&A attempted a network call")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    candidates = [_candidate("V1", 101, 1)]
    pipeline, _ = _answer_pipeline(candidates)
    result = pipeline.answer_ranked_candidates(
        {
            "query": "a test scene",
            "question": "What is visible?",
            "candidates": candidates,
            "candidate_count": 1,
            "vlm_candidate_count": 1,
            "route_active": False,
        },
        max_answers=1,
        use_context=False,
        structured_vlm=True,
    )
    assert result["answers"]
