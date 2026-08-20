from src.core.local_vlm import LocalVLM


def test_local_vlm_is_lazy():
    model = LocalVLM("/nonexistent/model")
    assert model.model is None
    assert model.processor is None


def test_local_vlm_structured_parser_handles_json_empty_and_malformed():
    parsed = LocalVLM._parse_metadata(
        '```json\n{"answer":"25 degrees","grounding_score":0.7,'
        '"answer_confidence":0.8,"abstain":false}\n```'
    )
    assert parsed["answer"] == "25 degrees"
    assert parsed["grounding_score"] == 0.7
    assert parsed["abstain"] is False

    empty = LocalVLM._parse_metadata(
        {"answer": "", "grounding_score": 0, "answer_confidence": 0, "abstain": False}
    )
    assert empty["abstain"] is True

    malformed = LocalVLM._parse_metadata("not-json")
    assert malformed["parse_failed"] is True
    assert malformed["answer"] == "not-json"


def test_local_vlm_structured_prompt_is_idempotent():
    prompt = 'Return ONLY valid JSON with exactly these fields: {"answer":"x"}'
    assert LocalVLM._structured_prompt(prompt) == prompt
    assert '"grounding_score"' in LocalVLM._structured_prompt("Answer this")
