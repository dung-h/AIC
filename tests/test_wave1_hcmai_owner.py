"""Wave 1 checks for the HCMAIPipeline orchestration boundary.

These tests use fakes only.  They verify ownership, request-scoped tracing and
strict offline behavior without loading a model, an index or a network client.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.pipelines.hcmai_pipeline import HCMAIPipeline
from src.runtime_context import RuntimeContext
from src.runtime_policy import RuntimePolicy


def _bare_pipeline(policy: RuntimePolicy) -> HCMAIPipeline:
    pipeline = HCMAIPipeline.__new__(HCMAIPipeline)
    pipeline._policy = policy
    pipeline.policy = policy
    pipeline.context = RuntimeContext.from_policy(policy, artifact_snapshot={"index": "test"})
    pipeline.last_trace = None
    pipeline._vqa = None
    pipeline._vqa_ranked = None
    pipeline._vqa_ranked_spec = None
    pipeline._vqa_answer_providers = {}
    pipeline._trake = None
    pipeline._trake_visual = None
    pipeline._kis_remote_translation = policy.kis_remote_translation
    pipeline._vqa_modality_routing = policy.vqa_modality_routing
    pipeline._vqa_modality_routers = {}
    pipeline._vqa_modality_router = None
    pipeline._default_trake_mode = policy.trake_mode
    return pipeline


def _unsafe_strict_policy(**changes) -> RuntimePolicy:
    """Build a policy fixture with a violation for boundary-level testing."""
    policy = RuntimePolicy(execution_mode="benchmark_strict")
    for name, value in changes.items():
        object.__setattr__(policy, name, value)
    return policy


def test_public_vqa_is_compatibility_facade_for_ranked_owner(monkeypatch):
    pipeline = _bare_pipeline(RuntimePolicy())
    calls = []

    def ranked(query, question, **kwargs):
        calls.append((query, question, kwargs))
        return {
            "answers": [{"video_id": "V1", "frame_id": 42, "answer": "25 degrees"}],
            "status": "ok",
            "trace": {"request_id": "req-1"},
        }

    monkeypatch.setattr(pipeline, "vqa_ranked", ranked)
    monkeypatch.setattr(
        pipeline,
        "_ensure_vqa",
        lambda: pytest.fail("legacy interactive VQA path was used"),
    )

    out = pipeline.vqa("weather report", "What temperature is mentioned?", topk=1)

    assert calls[0][2]["max_answers"] == 1
    assert out["best"] == {
        "video": "V1",
        "frame_idx": 42,
        "answer": "25 degrees",
    }


def test_ranked_vqa_reuses_pipeline_kis_owner(monkeypatch):
    pipeline = _bare_pipeline(RuntimePolicy())
    shared_kis = object()
    provider = object()
    pipeline._kis = None
    pipeline._ensure_kis = lambda: shared_kis
    pipeline._ensure_vqa_answer_provider = lambda **kwargs: provider
    calls = []

    class FakeVQA:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_module = ModuleType("vqa_pipeline_v3")
    fake_module.VQAPipelineV3 = FakeVQA
    monkeypatch.setitem(sys.modules, "vqa_pipeline_v3", fake_module)

    pipeline._ensure_vqa_ranked()

    assert calls == [{
        "translate": False,
        "answer_provider": provider,
        "kis_retriever": shared_kis,
    }]


def test_each_public_qna_request_gets_a_new_context_and_trace():
    pipeline = _bare_pipeline(RuntimePolicy())

    class FakeVQA:
        _local_vlm = object()

        def ranked_answers(self, *args, **kwargs):
            return {
                "answers": [{"video_id": "V1", "frame_id": 42, "answer": "ok"}],
                "status": "ok",
            }

    pipeline._vqa = FakeVQA()
    first = pipeline.vqa_ranked("scene", "What is shown?", max_answers=1)
    second = pipeline.vqa_ranked("scene", "What is shown?", max_answers=1)

    assert first["trace"]["request_id"] != second["trace"]["request_id"]
    assert first["trace"]["request_id"] != pipeline.context.request_id
    assert second["trace"]["context"]["request_id"] == second["trace"]["request_id"]


def test_each_public_trake_request_gets_a_new_context_and_trace():
    pipeline = _bare_pipeline(RuntimePolicy())

    class FakeVisualTrake:
        def search(self, events, **kwargs):
            return {
                "results": [{
                    "video_id": "V1",
                    "path": [
                        {"frame_idx": 10 + i, "pts_time": float(i)}
                        for i in range(len(events))
                    ],
                }],
                "diagnostics": {},
            }

    pipeline._trake_visual = FakeVisualTrake()
    first = pipeline.trake(["first"], topk=1)
    second = pipeline.trake(["first"], topk=1)

    assert first["trace"]["request_id"] != second["trace"]["request_id"]
    assert first["trace"]["context"]["request_id"] == first["trace"]["request_id"]


def test_strict_offline_rejects_remote_qna_provider_before_construction(monkeypatch):
    policy = _unsafe_strict_policy(vqa_answer_provider="openai")
    pipeline = _bare_pipeline(policy)
    provider_for_called = []

    import src.core.providers as providers

    monkeypatch.setattr(providers, "provider_for", lambda *args, **kwargs: provider_for_called.append(1))

    with pytest.raises(RuntimeError, match="offline.*local|remote"):
        pipeline._ensure_vqa_answer_provider()

    assert provider_for_called == []


def test_strict_offline_qna_construction_disables_translation(monkeypatch):
    policy = _unsafe_strict_policy(kis_remote_translation=True)
    pipeline = _bare_pipeline(policy)
    calls = []

    class FakeVQA:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_module = ModuleType("vqa_pipeline_v3")
    fake_module.VQAPipelineV3 = FakeVQA
    monkeypatch.setitem(sys.modules, "vqa_pipeline_v3", fake_module)

    pipeline._ensure_vqa()

    assert calls == [{"translate": False}]


def test_strict_offline_trake_rejects_remote_embeddings_before_construction(monkeypatch):
    policy = _unsafe_strict_policy(trake_remote_embeddings=True)
    pipeline = _bare_pipeline(policy)
    constructed = []

    class ForbiddenOnlineTrake:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

    fake_module = ModuleType("trake_pipeline")
    fake_module.TrakePipeline = ForbiddenOnlineTrake
    monkeypatch.setitem(sys.modules, "trake_pipeline", fake_module)

    with pytest.raises(RuntimeError, match="offline.*remote"):
        pipeline._ensure_trake()

    assert constructed == []


def test_ranked_qna_does_not_fallback_to_legacy_provider_seam(monkeypatch):
    policy = RuntimePolicy(execution_mode="benchmark_strict")
    pipeline = _bare_pipeline(policy)
    pipeline._ensure_vqa_answer_provider = lambda **kwargs: object()
    calls = []

    class FakeVQA:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "answer_provider" in kwargs:
                raise TypeError("unexpected keyword argument 'answer_provider'")

    fake_module = ModuleType("vqa_pipeline_v3")
    fake_module.VQAPipelineV3 = FakeVQA
    monkeypatch.setitem(sys.modules, "vqa_pipeline_v3", fake_module)

    with pytest.raises(RuntimeError, match="legacy provider fallback|answer_provider"):
        pipeline._ensure_vqa_ranked()

    assert len(calls) == 1
    assert "answer_provider" in calls[0]
