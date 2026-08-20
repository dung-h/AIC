from src.flow import FlowTrace, decide_specialist_flow
from src.runtime_context import RuntimeContext
from src.runtime_policy import RuntimePolicy


def test_strict_missing_modality_fails_closed():
    ctx = RuntimeContext.from_policy(RuntimePolicy(), mode="benchmark_strict")
    decision = decide_specialist_flow(
        ctx, owner="qna", required_modalities=["visual", "asr"],
        available_modalities=["visual"], specialist_hit=False,
    )
    assert decision.state == "failed"
    assert decision.error == "missing_required_modality:asr"


def test_interactive_missing_modality_is_visible_degradation():
    ctx = RuntimeContext.from_policy(
        RuntimePolicy(execution_mode="interactive_safe", vqa_fallback_policy="visual_with_trace"),
        mode="interactive_safe",
    )
    decision = decide_specialist_flow(
        ctx, owner="qna", required_modalities=["visual", "ocr"],
        available_modalities=["visual"], specialist_hit=False,
    )
    assert decision.state == "baseline_degraded"
    assert decision.used_fallback


def test_strict_specialist_no_hit_is_not_visual_rescue():
    ctx = RuntimeContext.from_policy(RuntimePolicy(), mode="benchmark_strict")
    decision = decide_specialist_flow(
        ctx, owner="qna", required_modalities=["asr"],
        available_modalities=["asr"], specialist_hit=False,
    )
    assert decision.state == "failed"
    assert decision.error == "specialist_returned_no_hit"


def test_trace_is_json_safe():
    ctx = RuntimeContext.from_policy(RuntimePolicy(), mode="production", request_id="r1")
    trace = FlowTrace("Q&A", ctx, "qna")
    trace.event("retrieval", topk=20)
    trace.finish()
    payload = trace.to_dict()
    assert payload["request_id"] == "r1"
    assert payload["events"][0]["name"] == "retrieval"
