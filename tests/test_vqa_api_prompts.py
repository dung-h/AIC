"""Offline contract tests for the remote/API VLM prompt adapter.

These tests inspect payloads through a fake transport.  They never call a
remote endpoint or consume API credits.
"""

from pathlib import Path

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3


def _pipeline_with_capture():
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    calls = []

    def fake_chat(payload, timeout):
        calls.append((payload, timeout))
        return {"choices": [{"message": {"content": "Blue"}}]}

    pipeline._vision_chat = fake_chat
    return pipeline, calls


def _image(tmp_path: Path) -> str:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"not-a-real-image-but-sufficient-for-base64-payload-test")
    return str(path)


def test_remote_answer_prompt_uses_question_language_and_repairs_context(tmp_path):
    pipeline, calls = _pipeline_with_capture()

    answer = pipeline._vlm_answer_with_context(
        _image(tmp_path),
        "What color is the shirt?",
        "MÃ u Ã¡o lÃ  xanh.",
        "",
        query="A presenter appears in a studio.",
    )

    assert answer == "Blue"
    prompt = calls[0][0]["messages"][0]["content"][0]["text"]
    assert "in English" in prompt
    assert "Visual description: A presenter appears in a studio." in prompt
    assert "Question: What color is the shirt?" in prompt
    assert "Màu áo là xanh." in prompt
    assert "MÃ u" not in prompt
    assert "Blue" not in prompt
    assert "TIẾNG VIỆT" not in prompt


def test_remote_answer_prompt_can_pin_language_without_answer_leak(tmp_path):
    pipeline, calls = _pipeline_with_capture()

    pipeline._vlm_answer_with_context(
        _image(tmp_path),
        "Màu áo là gì?",
        "",
        "",
        query="Một người đứng trong trường quay.",
        answer_language="en",
    )

    prompt = calls[0][0]["messages"][0]["content"][0]["text"]
    assert "in English" in prompt
    assert "Màu áo là gì?" in prompt
    assert "Một người đứng trong trường quay." in prompt
    assert "blue" not in prompt.lower()
    assert "ground truth" not in prompt.lower()


def test_verify_prompt_repairs_mojibake_without_network(tmp_path):
    pipeline, calls = _pipeline_with_capture()
    pipeline._vision_chat = lambda payload, timeout: (
        calls.append((payload, timeout)) or
        {"choices": [{"message": {"content": "0"}}]}
    )

    assert pipeline._vlm_verify(_image(tmp_path), "MÃ u xanh") == 0.0
    prompt = calls[0][0]["messages"][0]["content"][0]["text"]
    assert "Màu xanh" in prompt
    assert "MÃ u" not in prompt


def test_nonanswer_detection_handles_utf8_and_mojibake():
    assert VQAPipelineV3._looks_like_nonanswer("Không thể xác định từ frame này.")
    assert VQAPipelineV3._looks_like_nonanswer("KhÃ´ng thá»ƒ xÃ¡c Ä‘á»‹nh.")
    assert not VQAPipelineV3._looks_like_nonanswer("Một chiếc áo màu xanh.")
