"""Wave 0 characterization of trace and public output boundaries.

These tests intentionally use fakes only.  They freeze the evidence that a
future composition root must expose without loading a model, reading an
index, or making a network call.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.flow import FlowTrace, decide_specialist_flow
from src.runtime_context import RuntimeContext
from src.runtime_policy import RuntimePolicy
from src.submission.adapters import serialize_submission
from src.submission.contracts import CanonicalFrameIndex


@dataclass(frozen=True)
class _FakeRetrievalHit:
    video_id: str
    frame_idx: int
    score: float


class _FakeModalityRetriever:
    """Deterministic channel fake used to characterize retrieval tracing."""

    def __init__(self, modality: str, hits: list[_FakeRetrievalHit]) -> None:
        self.modality = modality
        self.hits = list(hits)
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, *, topk: int) -> list[_FakeRetrievalHit]:
        self.calls.append({"query": query, "topk": topk})
        return self.hits[:topk]


class _FakeTraceRuntime:
    """Small fake composition root for Wave 0 trace characterization."""

    def __init__(self, channels: dict[str, _FakeModalityRetriever]) -> None:
        self.channels = channels

    def retrieve(self, query: str, *, topk: int, context: RuntimeContext) -> dict:
        trace = FlowTrace("Q&A", context, "fake.qna")
        trace.event("request", query=query, topk=topk)
        results: dict[str, list[_FakeRetrievalHit]] = {}
        for modality, retriever in self.channels.items():
            hits = retriever.search(query, topk=topk)
            results[modality] = hits
            trace.event(
                "modality_retrieval",
                modality=modality,
                status="ok",
                requested_topk=topk,
                hit_count=len(hits),
                query=query,
                index_version=f"fake-{modality}-v1",
            )
        trace.finish()
        return {"results": results, "trace": trace.to_dict()}


def _strict_context(*, request_id: str | None = None) -> RuntimeContext:
    return RuntimeContext.from_policy(
        RuntimePolicy(execution_mode="benchmark_strict"),
        mode="benchmark_strict",
        request_id=request_id,
        split="dev",
    )


def test_each_request_trace_has_a_non_empty_distinct_request_id() -> None:
    runtime = _FakeTraceRuntime(
        {"visual": _FakeModalityRetriever("visual", [_FakeRetrievalHit("v1", 10, 0.9)])}
    )

    first = runtime.retrieve("a presenter", topk=5, context=_strict_context())
    second = runtime.retrieve("a presenter", topk=5, context=_strict_context())

    first_trace = first["trace"]
    second_trace = second["trace"]
    assert isinstance(first_trace["request_id"], str) and first_trace["request_id"]
    assert isinstance(second_trace["request_id"], str) and second_trace["request_id"]
    assert first_trace["request_id"] != second_trace["request_id"]
    assert first_trace["context"]["request_id"] == first_trace["request_id"]
    assert second_trace["context"]["request_id"] == second_trace["request_id"]


def test_modality_retrieval_trace_contains_channel_provenance_fields() -> None:
    visual = _FakeModalityRetriever(
        "visual", [_FakeRetrievalHit("v-visual", 101, 0.91)]
    )
    asr = _FakeModalityRetriever(
        "asr", [_FakeRetrievalHit("v-spoken", 202, 0.88)]
    )
    runtime = _FakeTraceRuntime({"visual": visual, "asr": asr})

    result = runtime.retrieve("what temperature was spoken", topk=7, context=_strict_context())

    events = [event for event in result["trace"]["events"] if event["name"] == "modality_retrieval"]
    assert {event["modality"] for event in events} == {"visual", "asr"}
    for event in events:
        assert event["status"] == "ok"
        assert event["requested_topk"] == 7
        assert event["hit_count"] == 1
        assert event["query"] == "what temperature was spoken"
        assert event["index_version"] == f"fake-{event['modality']}-v1"

    assert visual.calls == [{"query": "what temperature was spoken", "topk": 7}]
    assert asr.calls == [{"query": "what temperature was spoken", "topk": 7}]


@pytest.mark.parametrize(
    ("requested", "available", "hit", "expected"),
    [
        ((), ("visual",), False, "baseline_success"),
        (("asr",), ("visual", "asr"), True, "specialist_success"),
        (("asr",), ("visual", "asr"), False, "failed"),
    ],
)
def test_strict_route_state_has_explicit_semantics(
    requested: tuple[str, ...],
    available: tuple[str, ...],
    hit: bool,
    expected: str,
) -> None:
    decision = decide_specialist_flow(
        _strict_context(),
        owner="fake.qna",
        required_modalities=requested,
        available_modalities=available,
        specialist_hit=hit,
    )

    assert decision.state == expected
    if requested and not hit:
        assert decision.state != "baseline_success"
        assert decision.error == "specialist_returned_no_hit"


def test_interactive_route_state_marks_degraded_fallback() -> None:
    context = RuntimeContext.from_policy(
        RuntimePolicy(
            execution_mode="interactive_safe",
            vqa_fallback_policy="visual_with_trace",
        ),
        mode="interactive_safe",
    )

    decision = decide_specialist_flow(
        context,
        owner="fake.qna",
        required_modalities=("asr",),
        available_modalities=("visual", "asr"),
        specialist_hit=False,
    )

    assert decision.state == "baseline_degraded"
    assert decision.used_fallback
    assert decision.fallback_reason == "specialist_returned_no_hit"


def test_qna_output_uses_universal_external_shape() -> None:
    canonical = CanonicalFrameIndex({"v1": [100]})

    output = serialize_submission(
        "Q&A",
        {
            "q1": [
                {
                    "video_id": "v1",
                    "frame_id": 100,
                    "answer": "25 degrees",
                    "provider": "fake-vlm",
                }
            ]
        },
        canonical_frames=canonical,
    )

    assert output == {
        "task": "qa",
        "queries": {
            "q1": [{"video_id": "v1", "frame_id": 100, "answer": "25 degrees"}]
        },
    }
    assert set(output["queries"]["q1"][0]) == {"video_id", "frame_id", "answer"}


def test_trake_output_uses_universal_external_shape() -> None:
    canonical = CanonicalFrameIndex({"v1": [100, 200, 300]})

    output = serialize_submission(
        "TRAKE",
        {
            "t1": [
                {
                    "video_id": "v1",
                    "frame_ids": [100, 200, 300],
                    "score": 0.8,
                }
            ]
        },
        event_counts={"t1": 3},
        canonical_frames=canonical,
    )

    assert output == {
        "task": "trake",
        "queries": {"t1": [{"video_id": "v1", "frame_ids": [100, 200, 300]}]},
    }
    assert set(output["queries"]["t1"][0]) == {"video_id", "frame_ids"}
