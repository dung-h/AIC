"""Focused offline tests for the isolated VQA AnswerProvider boundary."""

from pathlib import Path

import pytest

from src.vqa.answer_provider import (
    AnswerProviderInputError,
    AnswerProviderRequest,
    AnswerProviderRequestError,
    AnswerProviderResponse,
    AnswerProviderSchemaError,
    EvidenceBundle,
    FrameEvidence,
    OpenAICompatibleAnswerProvider,
    QwenLocalAnswerProvider,
    RetryPolicy,
)


def _request(frame_path: Path) -> AnswerProviderRequest:
    return AnswerProviderRequest(
        query="A weather presenter is speaking.",
        question="What temperature is mentioned?",
        evidence=EvidenceBundle(
            candidate_id="K01_V001:42",
            video_id="K01_V001",
            frames=(FrameEvidence(frame_id=4200, frame_path=frame_path),),
            asr_text="The temperature in Nha Trang is twenty-five degrees.",
        ),
    )


def test_response_normalizes_empty_and_evidence_only_to_abstention():
    for answer in (
        "", "unknown", "evidence-only", "I cannot answer from this evidence",
        "Không tìm thấy hai câu thơ nào.",
    ):
        response = AnswerProviderResponse(
            candidate_id="candidate-1",
            answer=answer,
            grounding_score=0.4,
            answer_confidence=0.3,
            abstain=False,
            provider="test",
            model_id="fake",
        )
        assert response.abstain is True
        assert response.answer is None


def test_response_rejects_invalid_schema_scores_and_does_not_expose_raw_fields():
    with pytest.raises(AnswerProviderSchemaError):
        AnswerProviderResponse.from_mapping(
            {"answer": "25 degrees", "grounding_score": 2.0, "answer_confidence": 0.9},
            candidate_id="candidate-1",
            provider="test",
            model_id="fake",
        )
    safe = AnswerProviderResponse.abstained(
        candidate_id="candidate-1", provider="test", model_id="fake", reason="invalid_response"
    ).to_dict()
    assert "raw" not in safe
    assert "prompt" not in safe
    assert "api_key" not in safe


def test_evidence_requires_canonical_unique_frame_ids(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    with pytest.raises(AnswerProviderInputError):
        EvidenceBundle(
            candidate_id="c",
            video_id="v",
            frames=(FrameEvidence(frame_id=7, frame_path=frame), FrameEvidence(frame_id=7, frame_path=frame)),
        )

    with pytest.raises(AnswerProviderInputError, match="more than 12 frames"):
        EvidenceBundle(
            candidate_id="c",
            video_id="v",
            frames=tuple(FrameEvidence(frame_id=index, frame_path=frame) for index in range(13)),
        )


def test_local_qwen_adapter_uses_structured_model_and_timeout_hook(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    calls = []

    class FakeQwen:
        def answer_with_metadata(self, image_path, prompt, max_new_tokens):
            calls.append((image_path, prompt, max_new_tokens))
            return {
                "answer": "25 degrees",
                "grounding_score": 0.91,
                "answer_confidence": 0.88,
                "abstain": False,
            }

    timeout_calls = []

    def timeout_runner(operation, timeout_s):
        timeout_calls.append(timeout_s)
        return operation()

    provider = QwenLocalAnswerProvider(
        model=FakeQwen(),
        model_id="Qwen2.5-VL-3B",
        timeout_s=7.0,
        timeout_runner=timeout_runner,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    response = provider.answer(_request(frame))
    assert response.answer == "25 degrees"
    assert response.provider == "qwen_local"
    assert timeout_calls == [7.0]
    assert calls and "25 degrees" not in calls[0][1]


def test_local_qwen_adapter_parses_json_from_plain_model_output(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")

    class PlainQwen:
        def answer(self, *_args, **_kwargs):
            return '```json\n{"answer":"25 degrees","grounding_score":0.6,"answer_confidence":0.5}\n```'

    provider = QwenLocalAnswerProvider(
        model=PlainQwen(),
        model_id="Qwen2.5-VL-3B",
        retry_policy=RetryPolicy(max_attempts=1),
        timeout_runner=lambda operation, _timeout: operation(),
    )
    response = provider.answer(_request(frame))
    assert response.answer == "25 degrees"
    assert response.grounding_score == 0.6


def test_api_adapter_uses_openai_payload_and_redacts_secret_from_failure(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    seen = {}

    def transport(url, headers, payload, timeout_s):
        seen.update(url=url, headers=dict(headers), payload=payload, timeout_s=timeout_s)
        return {
            "choices": [{"message": {"content": '{"answer":"25 degrees","grounding_score":0.8,"answer_confidence":0.7,"abstain":false}'}}]
        }

    provider = OpenAICompatibleAnswerProvider(
        "https://api.example/v1/",
        "gpt-test",
        api_key="super-secret-key",
        timeout_s=3.5,
        retry_policy=RetryPolicy(max_attempts=1),
        transport=transport,
    )
    response = provider.answer(_request(frame))
    assert response.answer == "25 degrees"
    assert seen["url"] == "https://api.example/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer super-secret-key"
    assert seen["timeout_s"] == 3.5
    assert seen["payload"]["response_format"] == {"type": "json_object"}

    def failing_transport(*_args):
        raise RuntimeError("server echoed super-secret-key")

    failing = OpenAICompatibleAnswerProvider(
        "https://api.example/v1",
        "gpt-test",
        api_key="super-secret-key",
        retry_policy=RetryPolicy(max_attempts=1),
        transport=failing_transport,
    )
    with pytest.raises(AnswerProviderRequestError) as error:
        failing.answer(_request(frame))
    assert "super-secret-key" not in str(error.value)


def test_api_adapter_retries_transport_with_bounded_backoff(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    attempts = []
    sleeps = []

    def transport(*_args):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("transient failure")
        return {
            "choices": [{"message": {"content": '{"answer":"25 degrees","grounding_score":0.8,"answer_confidence":0.7}'}}]
        }

    provider = OpenAICompatibleAnswerProvider(
        "https://api.example/v1",
        "gpt-test",
        api_key="secret",
        retry_policy=RetryPolicy(max_attempts=2, initial_backoff_s=0.25, sleep=sleeps.append),
        transport=transport,
    )
    response = provider.answer(_request(frame))
    assert response.answer == "25 degrees"
    assert len(attempts) == 2
    assert sleeps == [0.25]


def test_api_adapter_retries_transient_empty_choice_content(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    attempts = []
    sleeps = []

    def transport(*_args):
        attempts.append(1)
        if len(attempts) == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return {
            "choices": [{"message": {"content": (
                '{"answer":"25 degrees","grounding_score":0.8,'
                '"answer_confidence":0.7,"abstain":false}'
            )}}]
        }

    provider = OpenAICompatibleAnswerProvider(
        "https://api.example/v1", "gpt-test", api_key="secret",
        retry_policy=RetryPolicy(max_attempts=2, initial_backoff_s=0.25, sleep=sleeps.append),
        transport=transport,
    )

    response = provider.answer(_request(frame))

    assert response.answer == "25 degrees"
    assert len(attempts) == 2
    assert sleeps == [0.25]


def test_api_adapter_can_abstain_on_structured_response_without_network(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fake-jpeg")
    provider = OpenAICompatibleAnswerProvider(
        "http://localhost:8000/v1",
        "local-vlm",
        require_api_key=False,
        retry_policy=RetryPolicy(max_attempts=1),
        transport=lambda *_: {
            "choices": [{"message": {"content": '{"answer":"","grounding_score":0,"answer_confidence":0,"abstain":true}'}}]
        },
    )
    response = provider.answer(_request(frame))
    assert response.abstain is True
    assert response.answer is None
