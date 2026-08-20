"""Wave 1 contract: service uses HCMAIPipeline public owners."""
from __future__ import annotations

from src.runtime_policy import RuntimePolicy


class _FakePipeline:
    instances = []

    def __init__(self, policy=None):
        self.policy = policy or RuntimePolicy()
        self.context = object()
        self.calls = []
        type(self).instances.append(self)

    def vqa_ranked(self, query, question, **kwargs):
        self.calls.append(("vqa_ranked", query, question, kwargs))
        return {
            "query": query,
            "question": question,
            "winner": "ranked",
            "status": "answered_local",
            "answers": [{
                "video_id": "V1",
                "frame_id": 42,
                "kf_n": 7,
                "pts_time": 1.5,
                "answer": "25 degrees",
                "grounding_score": 0.9,
            }],
            "trace": {"owner": "qna"},
        }

    def trake(self, events, **kwargs):
        self.calls.append(("trake", events, kwargs))
        return {
            "task": "TRAKE",
            "mode": "visual",
            "winner": "trake_visual",
            "results": [{
                "video_id": "V1",
                "path": [
                    {"frame_idx": 10 + i, "pts_time": float(i)}
                    for i, _ in enumerate(events)
                ],
            }],
            "trace": {"owner": "trake"},
        }

    def vqa(self, *args, **kwargs):  # pragma: no cover - guard against bypass
        raise AssertionError("legacy interactive VQA path must not be called")

    def _ensure_vqa_ranked(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("service must not reach the ranked child pipeline")

    def _ensure_trake_visual(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("service must not reach the visual TRAKE child")


def _last_fake():
    assert _FakePipeline.instances
    return _FakePipeline.instances[-1]


def test_service_qna_delegates_to_public_hcmai_method(monkeypatch):
    import src.pipelines.hcmai_pipeline as pipeline_module
    import src.service.runtime as runtime_module

    _FakePipeline.instances.clear()
    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", _FakePipeline)

    runtime = runtime_module.RetrievalRuntime(policy=RuntimePolicy())
    result = runtime.search_vqa(
        "weather report", "What temperature is mentioned?",
        max_answers=3, top_videos=5, frames_per_video=2,
        max_vlm_candidates=4, question_type="spoken_fact",
        required_modalities="visual,asr",
    )

    fake = _last_fake()
    assert [call[0] for call in fake.calls] == ["vqa_ranked"]
    assert fake.calls[0][3]["question_type"] == "spoken_fact"
    assert fake.calls[0][3]["required_modalities"] == "visual,asr"
    assert result["answers"][0]["video_id"] == "V1"
    assert runtime.last_trace == {"owner": "qna"}


def test_service_trake_delegates_to_public_hcmai_method(monkeypatch):
    import src.pipelines.hcmai_pipeline as pipeline_module
    import src.service.runtime as runtime_module

    _FakePipeline.instances.clear()
    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", _FakePipeline)

    runtime = runtime_module.RetrievalRuntime(policy=RuntimePolicy())
    result = runtime.search_trake(["first event", "second event"], 3, False)

    fake = _last_fake()
    assert [call[0] for call in fake.calls] == ["trake"]
    assert fake.calls[0][2] == {"topk": 3}
    assert len(result["results"][0]["path"]) == 2
    assert runtime.last_trace == {"owner": "trake"}
