"""P0 contract tests for the local Q&A answer layer only."""

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.eval.qna_materialized_visual import candidate_pool_digest


class _FakeLocalVLM:
    def __init__(self, answer):
        self.answer_text = answer
        self.calls = []

    def answer(self, frame_path, prompt, max_new_tokens=128):
        self.calls.append((frame_path, prompt, max_new_tokens))
        return self.answer_text


def _pipeline(answer):
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = _FakeLocalVLM(answer)
    return pipeline


def _prepared(question="What utensil is visible?"):
    return {
        "query": "A plate is shown.",
        "question": question,
        "candidate_count": 1,
        "vlm_candidate_count": 1,
        "candidates": [{
            "video_id": "K01_V001",
            "frame_idx": 42,
            "kf_n": 3,
            "pts_time": 1.5,
            "frame_path": "unused-frame.jpg",
            "video_rank": 1,
            "base_score": 0.8,
            "source": "visual",
        }],
    }


def test_prompt_requests_question_grammar_without_answer_leak():
    prompt = VQAPipelineV3._build_answer_prompt(
        "A person holds a utensil.", "What utensil is visible?", "", "",
    )
    assert "one answer" in prompt
    assert "noun phrase" in prompt
    assert "complete sentence" in prompt
    assert "bare label" in prompt
    assert "They are" not in prompt
    assert "fork" not in prompt.lower()
    assert '"grounding_score"' in prompt
    assert '"answer_confidence"' in prompt
    assert "Return exactly one answer" not in prompt


def test_extract_answer_removes_transport_wrappers_and_preserves_text():
    cases = {
        "```json\n{\"answer\": \" A fork. \", \"grounding_score\": 1}\n```": "A fork.",
        "Final answer: They are inspecting the bins.": "They are inspecting the bins.",
        "- It is on fire.\nReason: flames are visible.": "It is on fire.",
        "  Nước tương 2M  ": "Nước tương 2M",
        "": "",
        None: "",
    }
    for raw, expected in cases.items():
        assert VQAPipelineV3._extract_answer_text(raw) == expected


def test_answer_ranked_candidates_uses_cleaned_answer_not_json_wrapper():
    pipeline = _pipeline('{"answer": " A fork. ", "grounding_score": 0.9}')
    result = pipeline.answer_ranked_candidates(
        _prepared(), max_answers=1, use_context=False, structured_vlm=False,
    )
    assert result["answers"][0]["answer"] == "A fork."
    assert result["answer_trace"][0]["raw_answer"].startswith("{")
    assert result["answers"][0]["answer"] not in {"unknown", "evidence-only"}


def test_answer_ranked_candidates_rejects_cleaned_refusal():
    pipeline = _pipeline("Answer: I cannot answer from this image.")
    result = pipeline.answer_ranked_candidates(
        _prepared(), max_answers=1, use_context=False, structured_vlm=False,
    )
    assert result["answers"] == []
    assert result["status"] == "no_valid_local_answer"


class _FakeMultiFrameVLM:
    def __init__(self, answer="25 degrees"):
        self.answer_text = answer
        self.calls = []

    def answer_frames_with_metadata(self, image_paths, prompt, max_new_tokens=128):
        self.calls.append((list(image_paths), prompt, max_new_tokens))
        return {
            "answer": self.answer_text,
            "grounding_score": 0.9,
            "answer_confidence": 0.8,
            "abstain": False,
        }


def _routed_prepared(candidate, packet):
    return {
        "query": "A weather presenter is shown.",
        "question": "What temperature is mentioned?",
        "candidate_count": 1,
        "vlm_candidate_count": 1,
        "candidates": [candidate],
        "evidence_fusion": True,
        "route_active": True,
        "_packet": packet,
    }


def test_structured_local_answer_sees_bounded_packet_frames(monkeypatch):
    candidate = {
        "video_id": "K01_V001", "frame_idx": 42, "kf_n": 3,
        "pts_time": 1.5, "frame_path": "anchor.jpg", "video_rank": 1,
        "base_score": 0.8, "source": "visual",
    }
    packet = {
        "video_id": "K01_V001",
        "frames": [
            {"video_id": "K01_V001", "frame_idx": 42, "frame_path": "anchor.jpg"},
            {"video_id": "K01_V001", "frame_idx": 43, "frame_path": "specialist.jpg"},
            {"video_id": "K01_V001", "frame_idx": 44, "frame_path": "neighbor.jpg"},
        ],
        "asr_chunks": [], "ocr_text": [], "sources": ["visual"],
    }
    model = _FakeMultiFrameVLM()
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = model
    pipeline.answer_provider = None
    monkeypatch.setattr(pipeline, "_build_evidence_packet", lambda *args, **kwargs: packet)
    result = pipeline.answer_ranked_candidates(
        _routed_prepared(candidate, packet), max_answers=1,
        use_context=False, structured_vlm=True,
    )
    assert result["answers"][0]["answer"] == "25 degrees"
    assert model.calls and model.calls[0][0] == ["anchor.jpg", "specialist.jpg", "neighbor.jpg"]
    assert len(model.calls[0][0]) <= 12


def test_single_image_metadata_fallback_calls_each_frame_and_is_deterministic():
    class SingleImageModel:
        def __init__(self):
            self.calls = []

        def answer_with_metadata(self, image_path, prompt, max_new_tokens=128):
            self.calls.append(image_path)
            confidence = 0.4 if image_path == "anchor.jpg" else 0.9
            return {
                "answer": "25 degrees" if image_path != "bad.jpg" else "",
                "grounding_score": 0.5,
                "answer_confidence": confidence,
                "abstain": image_path == "bad.jpg",
            }

    model = SingleImageModel()
    record = VQAPipelineV3._local_answer_with_evidence(
        model, ["anchor.jpg", "specialist.jpg", "bad.jpg"],
        "Return structured answer", max_new_tokens=32,
    )
    assert model.calls == ["anchor.jpg", "specialist.jpg", "bad.jpg"]
    assert record["answer"] == "25 degrees"
    assert record["answer_confidence"] == 0.9


def test_structured_local_answer_rejects_empty_or_malformed_records():
    class BrokenModel:
        def answer_with_metadata(self, image_path, prompt, max_new_tokens=128):
            return "not a structured answer"

    record = VQAPipelineV3._local_answer_with_evidence(
        BrokenModel(), ["anchor.jpg"], "prompt", max_new_tokens=32,
    )
    assert record["answer"] == ""
    assert record["parse_failed"] is True

    class EmptyModel:
        def answer_with_metadata(self, image_path, prompt, max_new_tokens=128):
            return {"answer": "", "grounding_score": 0, "answer_confidence": 0, "abstain": True}

    empty = VQAPipelineV3._local_answer_with_evidence(
        EmptyModel(), ["anchor.jpg"], "prompt", max_new_tokens=32,
    )
    assert empty["answer"] == ""
    assert empty["abstain"] is True


def test_materialized_retrieval_pool_keeps_frozen_lattice_before_selector(tmp_path):
    frozen = [
        {"video_id": "V1", "frame_idx": 101, "kf_n": 1,
         "pts_time": 1.0, "base_score": 0.9, "video_rank": 0},
        {"video_id": "V1", "frame_idx": 102, "kf_n": 2,
         "pts_time": 2.0, "base_score": 0.8, "video_rank": 0},
        {"video_id": "V2", "frame_idx": 201, "kf_n": 1,
         "pts_time": 3.0, "base_score": 0.7, "video_rank": 1},
        {"video_id": "V2", "frame_idx": 202, "kf_n": 2,
         "pts_time": 4.0, "base_score": 0.6, "video_rank": 1},
    ]

    class FrozenKIS:
        materialized = True

        def _active_candidates(self):
            return list(frozen)

        def search(self, _query, topk=20):
            return [
                ("V1", 101, 1, 0.9),
                ("V2", 201, 1, 0.7),
            ][:topk]

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = FrozenKIS()
    pipeline.km = __import__("pandas").DataFrame(frozen)
    pipeline._frame_path = lambda *_args: str(frame)

    prepared = pipeline.prepare_ranked_candidates(
        "scene", "What is shown?", top_videos=2, frames_per_video=1,
        max_vlm_candidates=2, return_candidate_pool=True,
    )

    pool = prepared["_candidate_pool"]
    assert prepared["candidate_pool_count"] == len(frozen)
    assert candidate_pool_digest(pool) == candidate_pool_digest(frozen)
    assert prepared["candidate_count"] == len(frozen)
    assert len(prepared["candidates"]) <= 2


def test_answer_selector_uses_one_pool_and_keeps_local_ocr_fallback():
    pipeline = _pipeline("text from frame")
    prepared = {
        "query": "A sign is shown.",
        "question": "What word is on the sign?",
        "retrieved_video_ids": ["V1"],
        "route_active": True,
        "evidence_fusion": False,
        "_candidate_pool": [
            {
                "video_id": "V1", "frame_idx": 1, "kf_n": 1,
                "pts_time": 1.0, "video_rank": 0, "base_score": 0.9,
                "source": "ocr", "sources": ["ocr"],
                "text": "unrelated global text", "frame_path": "ocr.jpg",
            },
            {
                "video_id": "V1", "frame_idx": 2, "kf_n": 2,
                "pts_time": 2.0, "video_rank": 0, "base_score": 0.8,
                "source": "visual", "sources": ["visual"],
                "frame_path": "target.jpg",
            },
        ],
    }
    selected = pipeline.select_prepared_candidates(
        prepared, top_videos=1, max_vlm_candidates=2,
        required_modalities="visual,ocr", visual_selector_policy="adaptive",
    )
    assert all(item["video_id"] == "V1" for item in selected["candidates"])
    assert any(item.get("source") == "ocr" for item in selected["candidates"])
    assert any(item.get("_local_ocr_fallback") for item in selected["candidates"])
    assert selected["answer_selection"]["source"] == "retrieval_candidate_pool"
