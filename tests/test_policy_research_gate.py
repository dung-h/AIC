import pytest

from src.runtime_policy import RuntimePolicy


def test_policy_defaults_are_production_and_fail_closed(monkeypatch):
    for key in ("HCMAI_EXECUTION_MODE", "HCMAI_ENABLE_RESEARCH_ROUTES", "HCMAI_VQA_FALLBACK_POLICY"):
        monkeypatch.delenv(key, raising=False)
    policy = RuntimePolicy.from_env()
    assert policy.execution_mode == "production"
    assert policy.research_routes_enabled is False
    assert policy.vqa_fallback_policy == "fail_closed"


def test_strict_mode_rejects_visual_fallback():
    with pytest.raises(ValueError):
        RuntimePolicy(execution_mode="benchmark_strict", vqa_fallback_policy="visual_with_trace")


def test_interactive_mode_can_opt_into_trace_fallback():
    policy = RuntimePolicy(execution_mode="interactive_safe", vqa_fallback_policy="visual_with_trace")
    assert policy.vqa_fallback_policy == "visual_with_trace"
