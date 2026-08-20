"""Wave 3B contract tests for provider-agnostic answer verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.vqa.answer_provider import (
    AnswerProviderRequest,
    AnswerProviderResponse,
    AnswerProviderSchemaError,
    AnswerVerification,
    ContractAnswerVerifier,
    EvidenceBundle,
    FrameEvidence,
    OpenAICompatibleAnswerProvider,
    QwenLocalAnswerProvider,
    RetryPolicy,
    answer_with_verification,
)


def _request(tmp_path: Path, *, include_asr: bool = True, include_ocr: bool = True):
    frame_a = tmp_path / "frame-a.jpg"
    frame_b = tmp_path / "frame-b.jpg"
    frame_a.write_bytes(b"fake-jpeg-a")
    frame_b.write_bytes(b"fake-jpeg-b")
    return AnswerProviderRequest(
        query="A weather presenter is speaking.",
        question="What temperature is mentioned?",
        evidence=EvidenceBundle(
            candidate_id="K01_V001#42",
            video_id="K01_V001",
            frames=(
                FrameEvidence(frame_id=4200, frame_path=frame_a, pts_time=12.0),
                FrameEvidence(frame_id=4210, frame_path=frame_b, pts_time=12.4),
            ),
            asr_text="Nha Trang hôm nay khoảng 25 độ." if include_asr else "",
            ocr_text="DỰ BÁO THỜI TIẾT" if include_ocr else "",
        ),
    )


def _response(request, **overrides):
    values = {
        "candidate_id": request.evidence.candidate_id,
        "answer": "25 độ",
        "grounding_score": 0.91,
        "answer_confidence": 0.87,
        "abstain": False,
        "provider": "fake-local",
        "model_id": "fake-v1",
    }
    values.update(overrides)
    return AnswerProviderResponse(**values)


def test_structured_response_normalization_and_candidate_identity_are_strict(tmp_path):
    request = _request(tmp_path)
    response = AnswerProviderResponse.from_mapping(
        {
            "candidate_id": request.evidence.candidate_id,
            "answer": "25 độ",
            "grounding_score": "0.91",
            "answer_confidence": 0.87,
            "abstain": False,
        },
        candidate_id=request.evidence.candidate_id,
        provider="fake",
        model_id="fake-v1",
    )
    assert response.grounding_score == 0.91
    assert response.answer_confidence == 0.87

    with pytest.raises(AnswerProviderSchemaError, match="candidate_id"):
        AnswerProviderResponse.from_mapping(
            {"candidate_id": "other#1", "answer": "25 độ"},
            candidate_id=request.evidence.candidate_id,
            provider="fake",
            model_id="fake-v1",
        )


def test_verifier_accepts_multiframe_response_and_reports_all_sources(tmp_path):
    request = _request(tmp_path)
    result = ContractAnswerVerifier().verify(
        request,
        _response(request),
        required_sources=("visual", "speech", "ocr"),
    )

    assert isinstance(result, AnswerVerification)
    assert result.accepted is True
    assert result.is_accepted is True
    assert result.abstain is False
    assert result.frame_ids == (4200, 4210)
    assert result.evidence_sources == ("visual", "asr", "ocr")
    assert result.to_dict()["answer"] == "25 độ"


@pytest.mark.parametrize("answer", ["", "unknown", "evidence-only", "không đủ thông tin"])
def test_verifier_rejects_empty_evidence_only_and_non_answers(tmp_path, answer):
    request = _request(tmp_path)
    result = ContractAnswerVerifier().verify(request, _response(request, answer=answer))
    assert result.accepted is False
    assert result.abstain is True
    assert result.answer is None
    assert result.reason in {"invalid_answer", "provider_abstained"}


def test_verifier_rejects_identity_and_required_source_mismatch(tmp_path):
    request = _request(tmp_path, include_asr=False, include_ocr=False)
    mismatch = _response(request)
    mismatch = AnswerProviderResponse(
        candidate_id="K01_V001#wrong",
        answer=mismatch.answer,
        grounding_score=mismatch.grounding_score,
        answer_confidence=mismatch.answer_confidence,
        abstain=False,
        provider=mismatch.provider,
        model_id=mismatch.model_id,
    )
    verifier = ContractAnswerVerifier()
    assert verifier.verify(request, mismatch).reason == "candidate_identity_mismatch"
    assert verifier.verify(request, _response(request), required_sources=("asr",)).reason == "required_evidence_source_missing"


def test_verifier_rejects_unknown_frame_source(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    request = AnswerProviderRequest(
        query="scene",
        question="what is shown?",
        evidence=EvidenceBundle(
            candidate_id="c1",
            video_id="v1",
            frames=(FrameEvidence(frame_id=1, frame_path=frame, modality="mystery"),),
        ),
    )
    result = ContractAnswerVerifier().verify(
        request,
        AnswerProviderResponse(
            candidate_id="c1", answer="a person", grounding_score=0.5,
            answer_confidence=0.5, abstain=False, provider="fake", model_id="v1",
        ),
    )
    assert result.accepted is False
    assert result.reason == "invalid_evidence_source"


def test_local_provider_forwards_every_frame_to_injected_multiframe_model(tmp_path):
    request = _request(tmp_path)
    calls = []

    class FakeQwen:
        def answer_frames(self, image_paths, prompt, max_new_tokens):
            calls.append((list(image_paths), prompt, max_new_tokens))
            return {
                "answer": "25 độ",
                "grounding_score": 0.8,
                "answer_confidence": 0.7,
                "abstain": False,
            }

    provider = QwenLocalAnswerProvider(
        model=FakeQwen(),
        model_id="qwen-test",
        retry_policy=RetryPolicy(max_attempts=1),
        timeout_runner=lambda operation, _timeout: operation(),
    )
    response = provider.answer(request)
    assert response.answer == "25 độ"
    assert [Path(path).name for path in calls[0][0]] == ["frame-a.jpg", "frame-b.jpg"]
    assert "4200, 4210" in calls[0][1]


def test_remote_provider_forwards_every_frame_without_network(tmp_path):
    request = _request(tmp_path)
    seen = {}

    def transport(url, headers, payload, timeout_s):
        seen.update(url=url, headers=headers, payload=payload, timeout_s=timeout_s)
        return {
            "choices": [{"message": {"content": (
                '{"answer":"25 độ","grounding_score":0.8,'
                '"answer_confidence":0.7,"abstain":false}'
            )}}]
        }

    provider = OpenAICompatibleAnswerProvider(
        "https://api.example/v1", "remote-test", api_key="secret",
        retry_policy=RetryPolicy(max_attempts=1), transport=transport,
    )
    response = provider.answer(request)
    content = seen["payload"]["messages"][0]["content"]
    assert response.answer == "25 độ"
    assert sum(part.get("type") == "image_url" for part in content) == 2
    assert "4200, 4210" in content[0]["text"]
    assert seen["url"].endswith("/chat/completions")


def test_timeout_and_exhausted_retry_fail_closed_without_remote_fallback(tmp_path):
    request = _request(tmp_path)
    attempts = []
    sleeps = []

    class TimeoutProvider:
        provider_name = "local-timeout"
        model_id = "local-v1"

        def answer(self, _request):
            attempts.append(1)
            raise TimeoutError("test timeout")

    response, verification = answer_with_verification(
        TimeoutProvider(), request,
        verifier=ContractAnswerVerifier(),
    )
    assert len(attempts) == 1
    assert response.abstain is True
    assert response.reason == "provider_request_failed"
    assert verification.accepted is False
    assert verification.reason == "provider_abstained"

    # The actual adapter's bounded retry policy remains injectable and never
    # performs a provider switch after retries are exhausted.
    def transport(*_args):
        attempts.append(1)
        raise TimeoutError("transient")

    remote = OpenAICompatibleAnswerProvider(
        "https://api.example/v1", "remote-test", api_key="secret",
        retry_policy=RetryPolicy(max_attempts=2, initial_backoff_s=0.1, sleep=sleeps.append),
        transport=transport,
    )
    response, verification = answer_with_verification(remote, request)
    assert response.abstain is True
    assert verification.accepted is False
    assert len(sleeps) == 1
    assert sleeps == [0.1]
