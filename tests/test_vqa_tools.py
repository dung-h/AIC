import time
from src.vqa import CalculatorTool, ParallelVQAOrchestrator, QuestionType, ToolContext, ToolResult
from src.vqa.tools import classify_question

def test_classifier():
    assert classify_question("Chữ trên màn hình là gì?") == QuestionType.OCR_TEXT
    assert classify_question("Người dẫn nói con số nào?") == QuestionType.ASR_SPEECH
    assert classify_question("Có bao nhiêu người?") == QuestionType.TABLE_COUNT
    assert classify_question("Tính 12 + 3") == QuestionType.MATH_NUMERIC
    assert classify_question("Trong ảnh ai nói gì?") == QuestionType.MIXED

def test_calculator():
    r = CalculatorTool().run(ToolContext("Tính 12 + 3 * 2"))
    assert r.output == "18"
    assert r.error is None

class T:
    def __init__(self, name, delay, error=False): self.name, self.delay, self.error = name, delay, error
    def run(self, ctx):
        time.sleep(self.delay)
        return ToolResult(self.name, error="failed") if self.error else ToolResult(self.name, self.name, [self.name], .8)

def test_parallel_is_faster_and_isolates_failure():
    tools = [T("a", .08), T("b", .08), T("bad", .08, True)]
    start = time.perf_counter(); [t.run(None) for t in tools]; sequential = time.perf_counter() - start
    start = time.perf_counter(); out = ParallelVQAOrchestrator(tools).run("q"); parallel = time.perf_counter() - start
    assert parallel < sequential * .75
    assert out.tool_outputs["a"]["output"] == "a"
    assert out.errors["bad"] == "failed"

def test_unavailable_tools_are_evidence_errors():
    out = ParallelVQAOrchestrator([CalculatorTool()], timeout_s=1).run("What is visible?")
    assert "calculator" in out.errors

def test_timeout_is_not_an_exception():
    started = time.perf_counter()
    out = ParallelVQAOrchestrator([T("slow", .2)], timeout_s=.03).run("q")
    elapsed = time.perf_counter() - started
    assert "orchestrator" in out.errors
    assert elapsed < .12
