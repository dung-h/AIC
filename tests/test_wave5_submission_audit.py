"""Wave 5B submission-boundary audit tests.

These tests stay at the serializer/CLI boundary: no models, corpus indexes or
network providers are loaded.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from src.pipelines.codabench_submit import (
    _canonical_pairs,
    _submission_policy,
    format_ranked_submission_rows,
    format_submission_rows,
)
from src.runtime_policy import RuntimePolicy
from src.submission.adapters import (
    audit_submission,
    serialize_public_result,
    serialize_submission,
)
from src.submission.contracts import CanonicalFrameIndex


@pytest.fixture
def canonical():
    return CanonicalFrameIndex({"v1": [10, 20, 30, 40], "v2": [5, 15, 25, 35]})


def test_qna_external_contract_preserves_rank_and_strips_diagnostics(canonical):
    payload = serialize_submission(
        "qa",
        {
            "q1": [
                {"video_id": "v2", "frame_id": 15, "answer": "second", "score": 0.2},
                {"video_id": "v1", "frame_id": 10, "answer": "first", "provider": "local"},
            ]
        },
        canonical_frames=canonical,
    )

    assert payload["queries"]["q1"] == [
        {"video_id": "v2", "frame_id": 15, "answer": "second"},
        {"video_id": "v1", "frame_id": 10, "answer": "first"},
    ]
    assert all(set(answer) == {"video_id", "frame_id", "answer"}
               for answer in payload["queries"]["q1"])
    assert "audit" not in payload


@pytest.mark.parametrize("answer", [None, "", "   ", "unknown", "evidence-only", "null"])
def test_qna_rejects_null_empty_and_placeholder_answers(canonical, answer):
    with pytest.raises((TypeError, ValueError), match="answer"):
        serialize_submission(
            "qa",
            {"q1": [{"video_id": "v1", "frame_id": 10, "answer": answer}]},
            canonical_frames=canonical,
        )


def test_qna_rejects_invalid_frame_duplicate_and_missing_index(canonical):
    with pytest.raises(ValueError, match="non-canonical"):
        serialize_submission(
            "qa",
            {"q1": [{"video_id": "v1", "frame_id": 999, "answer": "answer"}]},
            canonical_frames=canonical,
        )
    with pytest.raises(ValueError, match="duplicates"):
        serialize_submission(
            "qa",
            {
                "q1": [
                    {"video_id": "v1", "frame_id": 10, "answer": "a"},
                    {"video_id": "v1", "frame_id": 10, "answer": "b"},
                ]
            },
            canonical_frames=canonical,
        )
    with pytest.raises(ValueError, match="canonical frame index"):
        serialize_submission(
            "qa",
            {"q1": [{"video_id": "v1", "frame_id": 10, "answer": "a"}]},
            canonical_frames=None,
        )


def test_trake_requires_exact_event_count_order_and_only_external_fields(canonical):
    payload = serialize_submission(
        "trake",
        {"t1": [{"video_id": "v1", "frame_ids": [10, 30], "score": 0.8}]},
        event_counts={"t1": 2},
        canonical_frames=canonical,
    )
    assert payload["queries"]["t1"] == [{"video_id": "v1", "frame_ids": [10, 30]}]
    assert set(payload["queries"]["t1"][0]) == {"video_id", "frame_ids"}

    for frame_ids, message in [([10], "expected 2"), ([30, 10], "strictly increasing"),
                               ([10, 999], "non-canonical")]:
        with pytest.raises(ValueError, match=message):
            serialize_submission(
                "trake",
                {"t1": [{"video_id": "v1", "frame_ids": frame_ids}]},
                event_counts={"t1": 2},
                canonical_frames=canonical,
            )


def test_both_tasks_reject_more_than_100_answers(canonical):
    qna = [{"video_id": "v1", "frame_id": frame, "answer": str(frame)}
           for frame in range(101)]
    # The canonical index is deliberately broad for this budget-only check.
    broad = CanonicalFrameIndex({"v1": range(101)})
    with pytest.raises(ValueError, match="100"):
        serialize_submission("qa", {"q1": qna}, canonical_frames=broad)

    trake = [{"video_id": "v1", "frame_ids": [0, 1]} for _ in range(101)]
    with pytest.raises(ValueError, match="100"):
        serialize_submission(
            "trake", {"t1": trake}, event_counts={"t1": 2}, canonical_frames=broad
        )


def test_audit_metadata_is_explicit_but_separate_from_default_payload(canonical):
    queries = {"q1": [{"video_id": "v1", "frame_id": 10, "answer": "answer"}]}
    audit = audit_submission(
        "qa", queries, canonical_frames=canonical, metadata={"entrypoint": "test"}
    )
    assert audit["schema"] == "hcmai.submission_audit.v1"
    assert audit["ranked_order_preserved"] is True
    assert audit["canonical_frames_validated"] is True
    assert audit["external_fields"] == ["video_id", "frame_id", "answer"]
    assert audit["diagnostics"]["entrypoint"] == "test"

    with_audit = serialize_submission(
        "qa", queries, canonical_frames=canonical, include_audit=True
    )
    assert with_audit["audit"]["answer_count"] == 1


def test_public_result_and_csv_adapter_share_contract(canonical, tmp_path):
    public = serialize_public_result(
        "qa",
        {"answers": [{"video_id": "v1", "frame_id": 20, "answer": "rain"}]},
        query_id="q1",
        canonical_frames=canonical,
    )
    csv_frame = format_ranked_submission_rows("qa", public)
    assert list(csv_frame.columns) == ["query_id", "video_id", "frame_id", "answer", "rank"]
    assert csv_frame.iloc[0].to_dict() == {
        "query_id": "q1", "video_id": "v1", "frame_id": 20,
        "answer": "rain", "rank": 1,
    }

    trake = serialize_public_result(
        "trake",
        {"results": [{"video_id": "v1", "path": [{"frame_idx": 10}, {"frame_idx": 30}]}]},
        query_id="t1",
        event_count=2,
        canonical_frames=canonical,
    )
    trake_csv = format_ranked_submission_rows("trake", trake)
    assert json.loads(trake_csv.iloc[0]["frame_ids"]) == [10, 30]


def test_kis_legacy_formatter_is_strict_and_does_not_invent_placeholder_rows():
    valid = format_submission_rows(
        [{"query_id": "q1", "video_name": "v1", "frame_idx": 10}]
    )
    assert list(valid.columns) == ["query_id", "video_name", "frame_idx"]
    with pytest.raises(ValueError, match="missing required columns"):
        format_submission_rows([{"query_id": "q1"}])
    with pytest.raises(ValueError, match="at least one"):
        format_submission_rows([])


def test_offline_submission_policy_locks_provider_and_network_boundary():
    policy = RuntimePolicy(
        execution_mode="research",
        vqa_answer_provider="openai",
        kis_remote_translation=True,
        trake_remote_embeddings=True,
    )
    locked = _submission_policy(policy, "qa", offline=True)
    assert locked.vqa_answer_provider == "local"
    assert locked.network_mode == "offline"
    assert locked.kis_remote_translation is False
    assert locked.trake_remote_embeddings is False
    assert policy.network_mode == "online"


def test_canonical_lookup_fails_closed_when_provider_or_index_is_missing():
    pipe = SimpleNamespace(_vqa_ranked=None, _vqa=None)
    with pytest.raises(RuntimeError, match="provider.*canonical"):
        _canonical_pairs(pipe, "qa")

    pipe = SimpleNamespace(
        _ensure_trake_visual=lambda: SimpleNamespace(km=SimpleNamespace(video_id=[], frame_idx=[]))
    )
    with pytest.raises(RuntimeError, match="canonical frame index is empty"):
        _canonical_pairs(pipe, "trake", "visual")


def test_codabench_rejects_mixed_tasks_without_creating_output(monkeypatch, tmp_path):
    from src.pipelines import codabench_submit

    input_path = tmp_path / "mixed.csv"
    input_path.write_text(
        "query_id,query,question,task_type\n"
        "q1,scene,what?,VQA\n"
        "q2,scene,,KIS\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "submission.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["codabench_submit.py", "--input", str(input_path), "--output", str(output_path), "--offline"],
    )
    with pytest.raises(ValueError, match="mixed task_type"):
        codabench_submit.main()
    assert not output_path.exists()
