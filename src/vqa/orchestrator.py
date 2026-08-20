"""Experimental parallel VQA orchestration with bounded, isolated tools."""
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
import time
from typing import Any

from .tools import ToolContext, ToolResult, VQATool, classify_question, QuestionType


@dataclass
class EvidenceAnswer:
    answer: Any = None
    selected_frame: Any = None
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    latencies_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class ParallelVQAOrchestrator:
    def __init__(self, tools: list[VQATool], timeout_s: float = 10.0, max_workers: int | None = None):
        self.tools, self.timeout_s = tools, timeout_s
        self.max_workers = max_workers or max(1, len(tools))

    def run(self, question: str, candidates=None, frame=None, context=None) -> EvidenceAnswer:
        ctx = ToolContext(question, list(candidates or []), frame, dict(context or {}))
        result = EvidenceAnswer(selected_frame=frame)
        started = time.perf_counter()
        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            futures = {pool.submit(tool.run, ctx): tool.name for tool in self.tools}
            try:
                completed = as_completed(futures, timeout=self.timeout_s)
                for future in completed:
                    name = futures[future]
                    try:
                        item = future.result()
                    except Exception as exc:
                        item = ToolResult(name, error=f"{type(exc).__name__}: {exc}")
                    result.latencies_ms[name] = item.latency_ms
                    if item.error:
                        result.errors[name] = item.error
                    else:
                        result.tool_outputs[name] = {"output": item.output, "evidence": item.evidence,
                                                     "confidence": item.confidence}
                        result.confidence = max(result.confidence, item.confidence)
            except TimeoutError:
                for future, name in futures.items():
                    if not future.done():
                        future.cancel()
                        result.errors.setdefault(name, "tool timeout")
        finally:
            # Running Python threads cannot be force-killed. Do not wait for a
            # timed-out provider here; the result must respect the VQA budget.
            pool.shutdown(wait=False, cancel_futures=True)
        # as_completed timeout is converted to a stable result instead of crashing callers
        elapsed = time.perf_counter() - started
        if elapsed > self.timeout_s:
            result.errors.setdefault("orchestrator", "overall timeout")
        return result
