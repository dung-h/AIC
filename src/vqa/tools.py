"""Small experimental VQA tools. Tools never own model/API lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import Any, Protocol


class QuestionType(str, Enum):
    VISUAL = "visual"
    OCR_TEXT = "ocr_text"
    ASR_SPEECH = "asr_speech"
    MATH_NUMERIC = "math_numeric"
    TABLE_COUNT = "table_count"
    MIXED = "mixed"


def classify_question(question: str) -> QuestionType:
    q = question.lower()
    groups = {
        QuestionType.OCR_TEXT: r"text|chữ|viết|đọc|tên|tiêu đề|caption|logo|màn hình",
        QuestionType.ASR_SPEECH: r"nói|phát biểu|lời|giọng|audio|nghe|nhắc đến|bảo rằng",
        QuestionType.MATH_NUMERIC: r"bao nhiêu|số nào|tổng|cộng|trừ|nhân|chia|%|phần trăm|tính|how many|calculate",
        QuestionType.TABLE_COUNT: r"đếm|bao nhiêu|mấy|số lượng|bảng|hàng|cột|count|table|number of",
    }
    hits = [kind for kind, pattern in groups.items() if re.search(pattern, q)]
    if QuestionType.ASR_SPEECH in hits and re.search(r"hình|ảnh|frame|visual|thấy", q):
        return QuestionType.MIXED
    # Speech questions often mention a number; that does not make them math.
    if QuestionType.ASR_SPEECH in hits and set(hits) <= {QuestionType.ASR_SPEECH, QuestionType.MATH_NUMERIC}:
        return QuestionType.ASR_SPEECH
    if QuestionType.TABLE_COUNT in hits:
        if set(hits) <= {QuestionType.TABLE_COUNT, QuestionType.MATH_NUMERIC}:
            return QuestionType.TABLE_COUNT
        return QuestionType.MIXED
    visual_signal = re.search(r"hình|ảnh|frame|visual|thấy|what is", q)
    if len(hits) > 1 or (visual_signal and not hits):
        return QuestionType.MIXED
    return hits[0] if hits else QuestionType.VISUAL


@dataclass
class ToolContext:
    question: str
    candidates: list[Any] = field(default_factory=list)
    frame: Any = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    output: Any = None
    evidence: list[Any] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None
    latency_ms: float = 0.0


class VQATool(Protocol):
    name: str
    def run(self, ctx: ToolContext) -> ToolResult: ...


class _TimedTool:
    def _run_timed(self, fn, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        try:
            result = fn(ctx)
            if not isinstance(result, ToolResult):
                result = ToolResult(self.name, result, [result], 0.5)
        except Exception as exc:  # tools are failure-isolated by contract
            result = ToolResult(self.name, error=f"{type(exc).__name__}: {exc}")
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result


class OCRTool(_TimedTool):
    name = "ocr"
    def __init__(self, provider=None): self.provider = provider
    def run(self, ctx):
        return self._run_timed(lambda c: self.provider(c.frame, c) if self.provider else
            ToolResult(self.name, error="OCR provider unavailable"), ctx)


class ASRContextTool(_TimedTool):
    name = "asr"
    def __init__(self, provider=None): self.provider = provider
    def run(self, ctx):
        return self._run_timed(lambda c: self.provider(c.frame, c) if self.provider else
            ToolResult(self.name, error="ASR provider unavailable"), ctx)


class CalculatorTool(_TimedTool):
    name = "calculator"
    def run(self, ctx):
        def calculate(c):
            expr = re.search(r"[\d\s()+\-*/.,%]+", c.question)
            if not expr or not re.search(r"\d", expr.group()):
                return ToolResult(self.name, error="No numeric expression")
            raw = expr.group().replace(",", ".").replace("%", "/100")
            try:
                import sympy
                value = sympy.sympify(raw, evaluate=True)
                if not value.is_number: raise ValueError("not numeric")
                return ToolResult(self.name, str(value), [raw], 0.95)
            except ImportError:
                return ToolResult(self.name, error="SymPy unavailable")
            except Exception as exc:
                return ToolResult(self.name, error=f"Invalid expression: {exc}")
        return self._run_timed(calculate, ctx)


class VLMTool(_TimedTool):
    name = "vlm"
    def __init__(self, provider=None): self.provider = provider
    def run(self, ctx):
        return self._run_timed(lambda c: self.provider(c.frame, c.question, c) if self.provider else
            ToolResult(self.name, error="VLM provider unavailable"), ctx)
