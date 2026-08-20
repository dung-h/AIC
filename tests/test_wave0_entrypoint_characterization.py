"""Wave 0 characterization for public entrypoint ownership.

These tests intentionally describe the Wave 1 ownership contract: Web UI,
service, and Codabench must delegate task execution through public
``HCMAIPipeline`` methods.  The fakes keep the tests independent of models,
indexes, network access, and the real submission data.

Failures are useful evidence in Wave 0: they identify an entrypoint that
still reaches a private child pipeline or legacy public method directly.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from src.runtime_policy import RuntimePolicy


class _FakeTrace:
    def __init__(self, task, context, owner):
        self.task = task
        self.context = context
        self.owner = owner
        self.events = []

    def event(self, name, **payload):
        self.events.append((name, payload))

    def decision(self, decision):
        self.events.append(("decision", getattr(decision, "state", None)))

    def finish(self):
        return None

    def to_dict(self):
        return {"task": self.task, "owner": self.owner, "events": self.events}


class _FakeDecision:
    state = "baseline_success"
    error = None


class _FakeRankedVQA:
    def __init__(self, calls):
        self._calls = calls

    def ranked_answers(self, query, question, **kwargs):
        self._calls.append(("ranked_answers", query, question, kwargs))
        return {
            "query": query,
            "question": question,
            "answers": [{"video_id": "V1", "frame_id": 42, "answer": "25 degrees"}],
            "status": "ok",
            "route_state": None,
        }


class _FakeVisualTrake:
    def __init__(self, calls):
        self._calls = calls
        self.last_timings_ms = {"total": 0.1}

    def search(self, events, **kwargs):
        self._calls.append(("visual_search", events, kwargs))
        return {
            "results": [{
                "video_id": "V1",
                "path": [{"frame_idx": 42, "pts_time": 1.0} for _ in events],
            }],
        }


class _FakePipeline:
    instances = []

    def __init__(self, policy=None):
        self.policy = policy if policy is not None else RuntimePolicy()
        self.context = object()
        self.calls = []
        self._vqa_ranked = None
        self._trake_visual = None
        type(self).instances.append(self)

    def vqa_ranked(self, query, question, **kwargs):
        self.calls.append(("vqa_ranked", query, question, kwargs))
        return {
            "query": query,
            "question": question,
            "answers": [{"video_id": "V1", "frame_id": 42, "answer": "25 degrees"}],
            "status": "ok",
            "winner": "ranked",
        }

    def vqa(self, query, question, **kwargs):
        self.calls.append(("vqa", query, question, kwargs))
        return {"best": None}

    def trake(self, events, **kwargs):
        self.calls.append(("trake", events, kwargs))
        return {
            "mode": "visual",
            "winner": "trake_visual",
            "results": [{
                "video_id": "V1",
                "path": [{"frame_idx": 10 + i, "pts_time": float(i)}
                         for i in range(len(events))],
            }],
        }

    # These private seams deliberately remain available so the test can
    # record a bypass rather than failing with an uninformative AttributeError.
    def _ensure_vqa_ranked(self, **kwargs):
        self.calls.append(("_ensure_vqa_ranked", kwargs))
        self._vqa_ranked = _FakeRankedVQA(self.calls)
        return self._vqa_ranked

    def _ensure_trake_visual(self):
        self.calls.append(("_ensure_trake_visual",))
        self._trake_visual = _FakeVisualTrake(self.calls)
        return self._trake_visual


def _method_names(fake_pipeline):
    return [call[0] for call in fake_pipeline.calls]


def test_web_ui_vqa_uses_the_public_ranked_pipeline(monkeypatch):
    """Interactive UI must share the public ranked Q&A owner."""
    from src.pipelines import web_ui

    _FakePipeline.instances.clear()
    fake = _FakePipeline()
    monkeypatch.setattr(web_ui, "get_pipe", lambda: fake)

    asyncio.run(web_ui.api_vqa({
        "query": "weather report",
        "question": "What temperature is mentioned?",
        "mode": "interactive",
    }))

    assert _method_names(fake) == ["vqa_ranked"]


def test_web_ui_trake_uses_the_public_pipeline(monkeypatch):
    from src.pipelines import web_ui

    _FakePipeline.instances.clear()
    fake = _FakePipeline()
    monkeypatch.setattr(web_ui, "get_pipe", lambda: fake)

    asyncio.run(web_ui.api_trake({"events": ["first event"]}))

    assert _method_names(fake) == ["trake"]


def test_service_vqa_delegates_to_hcmai_public_method(monkeypatch):
    """Service must not own a second Q&A orchestration path."""
    import src.pipelines.hcmai_pipeline as pipeline_module
    import src.service.runtime as runtime_module

    _FakePipeline.instances.clear()
    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", _FakePipeline)
    monkeypatch.setattr(runtime_module, "FlowTrace", _FakeTrace)
    monkeypatch.setattr(
        runtime_module,
        "decide_specialist_flow",
        lambda *args, **kwargs: _FakeDecision(),
    )

    runtime = runtime_module.RetrievalRuntime(policy=RuntimePolicy())
    runtime.search_vqa("weather report", "What temperature is mentioned?", 1, 1, 1, 1)

    fake = _FakePipeline.instances[-1]
    assert _method_names(fake) == ["vqa_ranked"]


def test_service_trake_delegates_to_hcmai_public_method(monkeypatch):
    """Service must not bypass HCMAIPipeline.trake through a backend child."""
    import src.pipelines.hcmai_pipeline as pipeline_module
    import src.service.runtime as runtime_module

    _FakePipeline.instances.clear()
    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", _FakePipeline)
    monkeypatch.setattr(runtime_module, "FlowTrace", _FakeTrace)
    monkeypatch.setattr(
        runtime_module,
        "decide_specialist_flow",
        lambda *args, **kwargs: _FakeDecision(),
    )

    runtime = runtime_module.RetrievalRuntime(policy=RuntimePolicy())
    runtime.search_trake(["first event"], 1, False)

    fake = _FakePipeline.instances[-1]
    assert _method_names(fake) == ["trake"]


def test_compatibility_cli_uses_the_public_ranked_pipeline(monkeypatch, tmp_path):
    """The opt-in compatibility CLI still delegates to the public owner."""
    from src.pipelines import codabench_submit

    calls = []

    class FakeCliPipeline:
        def __init__(self, policy=None):
            self.policy = policy if policy is not None else RuntimePolicy()
            self._vqa_ranked = SimpleNamespace()

        def vqa_ranked(self, query, question, **kwargs):
            calls.append(("vqa_ranked", query, question, kwargs))
            return {
                "answers": [{"video_id": "V1", "frame_id": 42, "answer": "25 degrees"}],
            }

    fake_module = types.ModuleType("hcmai_pipeline")
    fake_module.HCMAIPipeline = FakeCliPipeline
    monkeypatch.setitem(sys.modules, "hcmai_pipeline", fake_module)
    monkeypatch.setattr(codabench_submit, "_canonical_pairs", lambda *args, **kwargs: {("V1", 42)})

    input_path = tmp_path / "queries.csv"
    input_path.write_text(
        "query_id,query,question,task_type\n"
        "q1,weather report,What temperature is mentioned?,VQA\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "submission.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codabench_submit.py",
            "--compatibility-only",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--offline",
        ],
    )

    codabench_submit.main()

    assert [call[0] for call in calls] == ["vqa_ranked"]
    assert output_path.exists()
