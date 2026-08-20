"""Wave 1 submission-boundary contract tests.

These tests deliberately exercise only the submission policy and adapters;
they do not load models, instantiate a pipeline, or make network calls.
"""
from __future__ import annotations

import pytest

from src.pipelines.codabench_submit import _submission_policy
from src.runtime_policy import RuntimePolicy
from src.submission.adapters import serialize_public_result
from src.submission.contracts import CanonicalFrameIndex


def test_offline_ranked_submission_locks_provider_and_remote_flags() -> None:
    policy = RuntimePolicy(
        kis_remote_translation=True,
        trake_remote_embeddings=True,
        vqa_answer_provider="openai",
    )

    locked = _submission_policy(policy, "qa", offline=True)

    assert locked.vqa_answer_provider == "local"
    assert locked.kis_remote_translation is False
    assert locked.trake_remote_embeddings is False
    assert policy.vqa_answer_provider == "openai"
    assert policy.kis_remote_translation is True


def test_non_ranked_or_non_offline_policy_is_not_rewritten() -> None:
    policy = RuntimePolicy(vqa_answer_provider="openai")

    assert _submission_policy(policy, "kis", offline=True) is policy
    assert _submission_policy(policy, "qa", offline=False) is policy


def test_public_qna_result_uses_stable_shape_and_canonical_validation() -> None:
    output = serialize_public_result(
        "Q&A",
        {
            "answers": [
                {
                    "video_id": "v1",
                    "frame_id": 100,
                    "answer": "25 degrees",
                    "provider": "local",
                }
            ],
            "trace": {"provider": "local"},
        },
        query_id="q1",
        canonical_frames=CanonicalFrameIndex({"v1": [100]}),
    )

    assert output == {
        "task": "qa",
        "queries": {
            "q1": [{"video_id": "v1", "frame_id": 100, "answer": "25 degrees"}]
        },
    }


def test_public_trake_path_result_is_converted_and_validated() -> None:
    output = serialize_public_result(
        "TRAKE",
        {
            "results": [
                {
                    "video_id": "v1",
                    "path": [
                        {"frame_idx": 100, "pts_time": 1.0},
                        {"frame_idx": 200, "pts_time": 2.0},
                    ],
                    "score": 0.9,
                }
            ]
        },
        query_id="t1",
        event_count=2,
        canonical_frames=CanonicalFrameIndex({"v1": [100, 200]}),
    )

    assert output == {
        "task": "trake",
        "queries": {"t1": [{"video_id": "v1", "frame_ids": [100, 200]}]},
    }


def test_public_result_fails_closed_on_missing_canonical_frame() -> None:
    with pytest.raises(ValueError, match="non-canonical"):
        serialize_public_result(
            "qa",
            {
                "answers": [
                    {"video_id": "v1", "frame_id": 999, "answer": "answer"}
                ]
            },
            query_id="q1",
            canonical_frames=CanonicalFrameIndex({"v1": [100]}),
        )
