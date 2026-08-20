"""Wave 0 / owner B: strict offline and provider-boundary guardrails.

This module is intentionally source-free: it characterizes the control-plane
contracts at the production boundaries.  Invalid strict policies are rejected
at construction, while valid strict policies are verified at lazy provider
construction without loading models.

The tests use injected fakes and module seams, so they do not load models,
read the corpus, or make network calls.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from src.pipelines.codabench_submit import _submission_policy
from src.runtime_policy import RuntimePolicy


def _bare_pipeline(policy: RuntimePolicy):
    """Build only the control-plane state needed by a lazy ensure method."""

    from src.pipelines.hcmai_pipeline import HCMAIPipeline

    pipeline = HCMAIPipeline.__new__(HCMAIPipeline)
    pipeline._policy = policy
    pipeline.policy = policy
    pipeline._vqa = None
    pipeline._vqa_ranked = None
    pipeline._vqa_ranked_spec = None
    pipeline._vqa_answer_providers = {}
    pipeline._trake = None
    pipeline._kis_remote_translation = policy.kis_remote_translation
    return pipeline


def test_strict_submission_policy_forces_local_qna_provider() -> None:
    """An offline Q&A/TRAKE submission must not inherit an API provider."""

    policy = RuntimePolicy(
        kis_remote_translation=True,
        trake_remote_embeddings=True,
        vqa_answer_provider="openai",
        execution_mode="research",
    )

    locked = _submission_policy(policy, "qa", offline=True)

    assert locked.vqa_answer_provider == "local"
    assert locked.kis_remote_translation is False
    assert locked.trake_remote_embeddings is False


def test_strict_offline_qna_rejects_remote_provider_before_instantiation(monkeypatch) -> None:
    """The policy boundary rejects a remote provider before any pipeline exists."""

    with pytest.raises(ValueError, match=r"offline|remote|strict"):
        RuntimePolicy(vqa_answer_provider="openai", execution_mode="benchmark_strict")


def test_strict_offline_qna_constructs_no_translating_vqa(monkeypatch) -> None:
    """Strict offline Q&A must construct its VQA path with translation off."""

    policy = RuntimePolicy(execution_mode="benchmark_strict")
    pipeline = _bare_pipeline(policy)
    calls: list[dict] = []

    class FakeVQA:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))

    fake_module = ModuleType("vqa_pipeline_v3")
    fake_module.VQAPipelineV3 = FakeVQA
    monkeypatch.setitem(sys.modules, "vqa_pipeline_v3", fake_module)

    pipeline._ensure_vqa()

    assert calls == [{"translate": False}]


def test_strict_offline_trake_rejects_online_embedder_before_instantiation(monkeypatch) -> None:
    """The policy boundary rejects remote TRAKE embeddings before instantiation."""

    with pytest.raises(ValueError, match=r"offline|remote|strict"):
        RuntimePolicy(trake_remote_embeddings=True, execution_mode="benchmark_strict")


def test_strict_offline_trake_selects_only_local_embedder(monkeypatch) -> None:
    """The current safe TRAKE policy path must pass ``online=False``."""

    policy = RuntimePolicy(
        trake_remote_embeddings=False,
        execution_mode="benchmark_strict",
    )
    pipeline = _bare_pipeline(policy)
    calls: list[bool] = []

    class FakeOfflineTrake:
        def __init__(self, *, online=False):
            calls.append(bool(online))
            if online:
                raise AssertionError("offline TRAKE selected online embedding")

    fake_module = ModuleType("trake_pipeline")
    fake_module.TrakePipeline = FakeOfflineTrake
    monkeypatch.setitem(sys.modules, "trake_pipeline", fake_module)

    assert pipeline._ensure_trake() is not None
    assert calls == [False]


def test_strict_offline_qna_does_not_silently_switch_provider_seam(monkeypatch) -> None:
    """Provider-constructor incompatibility must fail closed in strict mode."""

    policy = RuntimePolicy(
        vqa_answer_provider="local",
        execution_mode="benchmark_strict",
    )
    pipeline = _bare_pipeline(policy)
    pipeline._ensure_vqa_answer_provider = lambda **kwargs: object()
    calls: list[dict] = []

    class FakeVQA:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            if "answer_provider" in kwargs:
                raise TypeError("unexpected keyword argument 'answer_provider'")

    fake_module = ModuleType("vqa_pipeline_v3")
    fake_module.VQAPipelineV3 = FakeVQA
    monkeypatch.setitem(sys.modules, "vqa_pipeline_v3", fake_module)

    with pytest.raises(RuntimeError, match=r"provider|strict|offline"):
        pipeline._ensure_vqa_ranked()

    assert len(calls) == 1
    assert "answer_provider" in calls[0]
