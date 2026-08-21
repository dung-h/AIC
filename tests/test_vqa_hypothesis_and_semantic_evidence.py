"""Contracts for the answer-free planner and independent precision gate."""

from __future__ import annotations

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.vqa.answer_provider import AnswerProviderResponse
from src.vqa.hypothesis_generator import (
    HypothesisRequest,
    OpenAICompatibleHypothesisGenerator,
    RetrievalHypothesisPlan,
)
from src.vqa.semantic_evidence import (
    CallableSemanticEvidenceJudge,
    OpenAICompatibleSemanticEvidenceJudge,
    SemanticEvidenceRequest,
    SemanticEvidenceVerdict,
)
from src.vqa.answer_provider import FrameEvidence


def test_hypothesis_generator_is_answer_free_and_bounded():
    seen = {}

    def transport(_url, _headers, payload, _timeout):
        seen["payload"] = payload
        return {
            "choices": [{"message": {"content": (
                '{"intent":"spoken fact","answer_type":"location",'
                '"retrieval_queries":["câu lạc bộ FANA Khánh Hòa"],'
                '"entities":["FANA"],"expected_modalities":["asr","ocr"],'
                '"temporal_constraints":[]}'
            )}}]
        }

    generator = OpenAICompatibleHypothesisGenerator(
        "https://api.example/v1", "vision-test", api_key="test-key", transport=transport
    )
    plan = generator.generate(HypothesisRequest(
        "Phóng sự về câu lạc bộ FANA tại Khánh Hòa.", "Xã được nhắc đến là gì?"
    ))

    prompt = seen["payload"]["messages"][0]["content"]
    assert "Do NOT answer" in prompt
    assert "answer" not in plan.to_dict()
    assert plan.grounding_views() == ("câu lạc bộ FANA Khánh Hòa", "FANA")
    assert plan.expected_modalities == ("asr", "ocr")


def test_hypothesis_generator_drops_ungrounded_numeric_or_temporal_constraints():
    def transport(_url, _headers, _payload, _timeout):
        return {
            "choices": [{"message": {"content": (
                '{"intent":"find location","answer_type":"location",'
                '"retrieval_queries":["FANA Khánh Hòa", "FANA Khánh Hòa 2023"],'
                '"entities":["FANA"],"expected_modalities":["asr"],'
                '"temporal_constraints":["2023"]}'
            )}}]
        }

    generator = OpenAICompatibleHypothesisGenerator(
        "https://api.example/v1", "vision-test", api_key="test-key", transport=transport
    )
    plan = generator.generate(HypothesisRequest(
        "Phóng sự về câu lạc bộ FANA tại Khánh Hòa.", "Xã được nhắc đến là gì?"
    ))

    assert plan.retrieval_queries == ("FANA Khánh Hòa",)
    assert plan.temporal_constraints == ()
    assert set(plan.dropped_ungrounded_hypotheses) == {"FANA Khánh Hòa 2023", "2023"}


def test_hypothesis_plan_normalizes_model_modality_synonyms_once():
    plan = RetrievalHypothesisPlan(
        intent="spoken fact",
        answer_type="location",
        retrieval_queries=("FANA Khánh Hòa",),
        expected_modalities=("vision", "audio and screen text"),
        provider="test",
        model_id="test",
    )

    assert plan.expected_modalities == ("visual", "asr", "ocr")


def test_semantic_evidence_judge_uses_only_candidate_evidence(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    seen = {}

    def transport(_url, _headers, payload, _timeout):
        seen["payload"] = payload
        return {
            "choices": [{"message": {"content": (
                '{"context_supported":true,"answer_supported":true,'
                '"temporal_consistent":true,"contradicted":false,'
                '"relevance_score":0.92,"abstain":false,"reason":"asr_entails_answer"}'
            )}}]
        }

    judge = OpenAICompatibleSemanticEvidenceJudge(
        "https://api.example/v1", "vision-test", api_key="test-key", transport=transport
    )
    verdict = judge.judge(SemanticEvidenceRequest(
        query="Bản tin thời tiết.",
        question="Nhiệt độ được nói là bao nhiêu?",
        candidate_id="L30_V072#14",
        video_id="L30_V072",
        answer="25 độ",
        frames=(FrameEvidence(frame_id=420, frame_path=frame),),
        asr_text="Nhiệt độ tại Nha Trang là 25 độ.",
        expected_sources=("visual", "asr"),
    ))

    assert verdict.accepted is True
    content = seen["payload"]["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert "Do not use world knowledge" in content[0]["text"]
    assert "25 độ" in content[0]["text"]


def test_semantic_evidence_rejection_is_a_fail_closed_candidate_gate():
    class Provider:
        provider_name = "fake"
        model_id = "fake-v1"

        def answer(self, request):
            return AnswerProviderResponse(
                candidate_id=request.evidence.candidate_id,
                answer="Trường Sa",
                grounding_score=0.99,
                answer_confidence=0.99,
                abstain=False,
                provider=self.provider_name,
                model_id=self.model_id,
            )

    def reject(request):
        assert request.answer == "Trường Sa"
        return SemanticEvidenceVerdict(
            candidate_id=request.candidate_id,
            context_supported=False,
            answer_supported=False,
            temporal_consistent=True,
            contradicted=True,
            relevance_score=0.01,
            abstain=True,
            reason="candidate_context_mismatch",
            provider="test",
            model_id="test",
        )

    pipeline = object.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    pipeline.answer_provider = Provider()
    result = pipeline.answer_ranked_candidates(
        {
            "query": "Một phóng sự thiện nguyện ở Khánh Hòa.",
            "question": "Xã được nhắc đến là gì?",
            "candidates": [{
                "video_id": "K01_V001", "frame_idx": 120, "kf_n": 4,
                "pts_time": 5.0, "frame_path": "frame.jpg", "source": "visual",
                "video_rank": 0, "base_score": 0.8,
            }],
        },
        use_context=False,
        semantic_evidence_judge=CallableSemanticEvidenceJudge(reject),
    )

    assert result["answers"] == []
    assert result["semantic_evidence_verifier"]["enabled"] is True
    assert result["answer_trace"][0]["status"] == "rejected_semantic_evidence"
