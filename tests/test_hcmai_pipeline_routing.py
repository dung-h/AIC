from __future__ import annotations

from unittest.mock import patch

import pytest

from src.pipelines.hcmai_pipeline import HCMAIPipeline
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3, infer_question_type
from src.runtime_policy import RuntimePolicy


class _FakeVQA:
    _local_vlm = object()

    def __init__(self):
        self.kwargs = None

    def ranked_answers(self, query, question, **kwargs):
        self.kwargs = kwargs
        return {"answers": []}


def test_enabled_routing_infers_one_specialist_without_type_label():
    pipeline = HCMAIPipeline()
    fake_vqa = _FakeVQA()
    pipeline._vqa = fake_vqa
    fake_router = object()

    with patch.object(pipeline, "_ensure_vqa_modality_router", return_value=fake_router):
        pipeline.vqa_ranked("weather report", "What temperature is mentioned?",
                            modality_routing=True)

    assert fake_vqa.kwargs["required_modalities"] == "visual,asr"
    assert fake_vqa.kwargs["global_modality_router"] is fake_router
    assert fake_vqa.kwargs["visual_selector_policy"] == "adaptive"


@pytest.mark.parametrize(
    ("query", "question", "expected"),
    [
        ("weather report", "According to the presenter, what temperature was mentioned?", "spoken_fact"),
        ("a storefront", "What is written on the signboard?", "screen_text"),
        ("một vật thể", "Vật thể có màu gì?", "color"),
        ("people in a room", "What are they doing?", "action"),
    ],
)
def test_live_question_type_inference_is_conservative(query, question, expected):
    assert infer_question_type(query, question) == expected


def test_unlabeled_ambiguous_question_stays_visual():
    pipeline = HCMAIPipeline()
    fake_vqa = _FakeVQA()
    pipeline._vqa = fake_vqa

    with patch.object(pipeline, "_ensure_vqa_modality_router") as make_router:
        pipeline.vqa_ranked(
            "people in a kitchen", "What are they doing?", modality_routing=True,
        )

    assert fake_vqa.kwargs["required_modalities"] is None
    assert fake_vqa.kwargs["question_type"] == "action"
    make_router.assert_not_called()


def test_explicit_visual_type_does_not_enable_text_channels():
    pipeline = HCMAIPipeline()
    fake_vqa = _FakeVQA()
    pipeline._vqa = fake_vqa

    with patch.object(pipeline, "_ensure_vqa_modality_router") as make_router:
        pipeline.vqa_ranked("a red object", "What color is it?",
                            question_type="color", modality_routing=True)

    assert fake_vqa.kwargs["required_modalities"] is None
    assert fake_vqa.kwargs["global_modality_router"] is None
    assert fake_vqa.kwargs["visual_selector_policy"] == "adaptive"
    make_router.assert_not_called()


def test_strict_qna_does_not_rescue_specialist_no_hit_with_visual():
    pipeline = HCMAIPipeline()
    fake_vqa = _FakeVQA()
    fake_vqa.ranked_answers = lambda *args, **kwargs: {
        "answers": [], "route_state": "specialist_no_hit"
    }
    pipeline._vqa = fake_vqa
    with patch.object(pipeline, "_ensure_vqa_modality_router", return_value=object()):
        with pytest.raises(RuntimeError, match="specialist_returned_no_hit"):
            pipeline.vqa_ranked(
                "weather report", "What temperature is mentioned?",
                modality_routing=True,
            )


def test_interactive_qna_records_visual_degradation():
    policy = RuntimePolicy(
        execution_mode="interactive_safe",
        vqa_fallback_policy="visual_with_trace",
    )
    pipeline = HCMAIPipeline(policy=policy)
    fake_vqa = _FakeVQA()
    fake_vqa.ranked_answers = lambda *args, **kwargs: {
        "answers": [], "route_state": "specialist_no_hit"
    }
    pipeline._vqa = fake_vqa
    with patch.object(pipeline, "_ensure_vqa_modality_router", return_value=object()):
        result = pipeline.vqa_ranked(
            "weather report", "What temperature is mentioned?",
            modality_routing=True,
        )
    assert result["trace"]["events"][-1]["state"] == "baseline_degraded"


def test_global_specialist_is_called_when_visual_retrieval_is_empty():
    class EmptyKIS:
        def search(self, *args, **kwargs):
            return []

    class CountingRouter:
        def __init__(self):
            self.calls = 0

        def global_candidates(self, *args, **kwargs):
            self.calls += 1
            return []

    pipeline = object.__new__(VQAPipelineV3)
    pipeline.kis = EmptyKIS()
    router = CountingRouter()
    out = pipeline.prepare_ranked_candidates(
        "weather report", "What temperature is mentioned?",
        required_modalities="asr", global_modality_router=router,
    )
    assert router.calls == 1
    assert out["route_state"] == "specialist_no_hit"
    assert out["status"] == "no_retrieval"
