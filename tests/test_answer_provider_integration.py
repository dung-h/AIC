from types import SimpleNamespace

import pytest

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.vqa.answer_provider import AnswerProviderResponse


class FakeProvider:
    provider_name = "fake"
    model_id = "fake-v1"

    def __init__(self):
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return AnswerProviderResponse(
            candidate_id=request.evidence.candidate_id,
            answer="hai mươi lăm độ",
            grounding_score=0.9,
            answer_confidence=0.8,
            abstain=False,
            provider=self.provider_name,
            model_id=self.model_id,
        )


class _StaticEvidenceProvider:
    def __init__(self, packet):
        self.packet = packet

    def evidence_packet_for_candidate(self, *_args, **_kwargs):
        return self.packet


def test_ranked_qna_accepts_structured_answer_provider_without_local_vlm():
    provider = FakeProvider()
    pipeline = object.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    pipeline.answer_provider = provider

    result = pipeline.answer_ranked_candidates(
        {
            "query": "một bản tin thời tiết",
            "question": "Nhiệt độ là bao nhiêu?",
            "candidates": [{
                "video_id": "K01_V001",
                "frame_idx": 120,
                "kf_n": 4,
                "pts_time": 5.0,
                "frame_path": "frame.jpg",
                "source": "asr",
                "video_rank": 0,
                "base_score": 0.8,
            }],
            "candidate_count": 1,
            "vlm_candidate_count": 1,
        },
        use_context=False,
    )

    assert result["status"] == "answered_local"
    assert result["answers"][0]["answer"] == "hai mươi lăm độ"
    assert result["answers"][0]["provider"] == "fake"
    assert provider.requests[0].evidence.frames[0].frame_id == 120
    assert provider.requests[0].evidence.video_id == "K01_V001"


def test_screen_text_uses_pixels_as_primary_answer_evidence(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = FakeProvider()
    pipeline = object.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    pipeline.answer_provider = provider
    pipeline.evidence_verifier = None
    packet = {
        "asr_chunks": (),
        # A sampled OCR row can contain a ticker while the requested large
        # sign is still visibly readable by the VLM in the same frame.
        "ocr_text": ({"text": "unrelated news ticker", "start": 1.0, "end": 1.0},),
        "frames": ({
            "video_id": "K01_V001", "frame_idx": 120, "kf_n": 4,
            "pts_time": 1.0, "frame_path": str(frame), "source": "visual",
        },),
        "sources": ("ocr",),
    }

    result = pipeline.answer_ranked_candidates({
        "query": "a large sign", "question": "What is written on the sign?",
        "question_type": "screen_text",
        "candidates": [{
            "video_id": "K01_V001", "frame_idx": 120, "kf_n": 4,
            "pts_time": 1.0, "frame_path": str(frame), "source": "visual",
            "video_rank": 0, "base_score": 0.8,
        }],
        "candidate_count": 1, "vlm_candidate_count": 1,
        "route_active": True, "evidence_fusion": True,
        "required_sources": ["ocr"],
        "_evidence_provider": _StaticEvidenceProvider(packet),
    }, use_context=False)

    assert result["status"] == "answered_local"
    assert result["answers"][0]["verification_policy"] == "visual_frame_primary"
    assert result["answers"][0]["verification"]["accepted"] is True


def test_ranked_answers_allows_explicit_remote_provider_only_in_online_mode():
    pipeline = object.__new__(VQAPipelineV3)
    pipeline.answer_provider = SimpleNamespace(is_remote=True)
    pipeline.prepare_ranked_candidates = lambda *_args, **_kwargs: {
        "query": "scene", "question": "what?", "candidates": [],
    }
    pipeline.answer_ranked_candidates = lambda prepared, **_kwargs: {
        "answers": [], "status": "no_valid_remote_answer",
    }

    result = pipeline.ranked_answers("scene", "what?", offline=False)
    assert result["status"] == "no_valid_remote_answer"

    with pytest.raises(ValueError, match="offline.*remote"):
        pipeline.ranked_answers("scene", "what?", offline=True)
