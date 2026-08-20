"""Contract tests for Q&A's multi-modal evidence packet."""

from src.vqa.evidence_fusion import (
    answer_is_submission_safe,
    build_evidence_packet,
    evidence_support_score,
    render_evidence_prompt,
)
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
import pandas as pd


def _candidate():
    return {
        "video_id": "K01_V001", "kf_n": 8, "frame_idx": 800,
        "pts_time": 12.0, "frame_path": "anchor.jpg",
    }


def test_packet_keeps_same_video_and_timestamped_multimodal_evidence():
    packet = build_evidence_packet(
        _candidate(),
        query="weather forecast in Nha Trang",
        question="What temperature is mentioned?",
        asr_rows=[
            {"vid": "K01_V001", "chunk": "Nha Trang hôm nay khoảng 25 độ.", "start": 11.0, "end": 13.0},
            {"vid": "K01_V001", "chunk": "A later unrelated sentence.", "start": 40.0, "end": 42.0},
            {"vid": "K01_V999", "chunk": "Nha Trang khoảng 99 độ.", "start": 11.0, "end": 13.0},
        ],
        ocr_rows=[
            {"video_id": "K01_V001", "ocr_text": "DỰ BÁO THỜI TIẾT", "pts_time": 12.5},
        ],
    )

    assert packet["video_id"] == "K01_V001"
    assert packet["anchor"] == {"frame_idx": 800, "kf_n": 8, "pts_time": 12.0}
    assert len(packet["frames"]) == 1
    assert [row["text"] for row in packet["asr_chunks"]] == ["Nha Trang hôm nay khoảng 25 độ."]
    assert packet["ocr_text"][0]["timestamp"] == 12.5
    assert packet["timestamps"][1]["start_time"] == 11.0
    assert packet["sources"] == ["visual", "asr", "ocr"]


def test_packet_accepts_pipeline_dataframes():
    packet = build_evidence_packet(
        _candidate(),
        question="What temperature is mentioned?",
        asr_rows=pd.DataFrame([{
            "vid": "K01_V001", "chunk": "25 độ", "start": 11.0, "end": 13.0,
        }]),
        ocr_rows=pd.DataFrame([{
            "video_id": "K01_V001", "ocr_text": "weather", "pts_time": 12.0,
        }]),
    )
    assert len(packet["asr_chunks"]) == 1
    assert len(packet["ocr_text"]) == 1


def test_packet_uses_same_video_specialist_frame_as_text_anchor():
    packet = build_evidence_packet(
        {**_candidate(), "evidence_frames": [{
            "video_id": "K01_V001", "kf_n": 20, "frame_idx": 2000,
            "pts_time": 100.0, "role": "evidence", "modality": "asr",
        }]},
        query="weather forecast in Nha Trang",
        question="What temperature is mentioned?",
        asr_rows=[{
            "vid": "K01_V001", "chunk": "Nha Trang khoảng 25 độ",
            "start": 99.0, "end": 101.0,
        }],
    )

    assert packet["has_spoken_evidence"] is True
    assert packet["asr_chunks"][0]["text"] == "Nha Trang khoảng 25 độ"
    assert packet["asr_chunks"][0]["start_time"] == 99.0


def test_spoken_answer_is_grounded_even_when_frame_has_no_answer_text():
    packet = build_evidence_packet(
        _candidate(), question="What temperature is mentioned?",
        asr_rows=[{"vid": "K01_V001", "chunk": "Nha Trang khoảng 25 độ", "start": 11, "end": 13}],
    )
    assert evidence_support_score("25 độ", packet) == 1.0
    assert "25 độ" not in "weather forecast"  # the frame/query need not contain the answer
    prompt = render_evidence_prompt(packet)
    assert "11.00s-13.00s" in prompt
    assert "not visible in the anchor frame" in prompt


def test_packet_rejects_cross_video_temporal_frames_and_nonanswers():
    packet = build_evidence_packet(
        {**_candidate(), "evidence_frames": [
            {"video_id": "K01_V999", "kf_n": 9, "frame_idx": 900, "pts_time": 13},
            {"video_id": "K01_V001", "kf_n": 9, "frame_idx": 900, "pts_time": 13},
        ]},
    )
    assert [frame["kf_n"] for frame in packet["frames"]] == [8, 9]
    assert not answer_is_submission_safe("evidence-only")
    assert not answer_is_submission_safe("I cannot answer from this frame")
    assert answer_is_submission_safe("25 degrees")


class _StructuredSpeechVLM:
    def __init__(self):
        self.prompts = []

    def answer_with_metadata(self, frame_path, prompt, *, max_new_tokens=128):
        self.prompts.append(prompt)
        return {"answer": "25 độ", "grounding_score": 0.8,
                "answer_confidence": 0.9, "abstain": False, "parse_failed": False}


def test_answerer_emits_canonical_frame_for_speech_only_answer(monkeypatch):
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    vlm = _StructuredSpeechVLM()
    pipeline._local_vlm = vlm
    candidate = {**_candidate(), "video_rank": 0, "base_score": 0.8, "source": "asr"}
    packet = build_evidence_packet(
        candidate, question="What temperature is mentioned?",
        asr_rows=[{"vid": "K01_V001", "chunk": "Nha Trang khoảng 25 độ", "start": 11, "end": 13}],
    )
    monkeypatch.setattr(pipeline, "_build_evidence_packet", lambda *args: packet)
    result = pipeline.answer_ranked_candidates({
        "query": "weather forecast in Nha Trang", "question": "What temperature is mentioned?",
        "candidates": [candidate], "candidate_count": 1, "vlm_candidate_count": 1,
        "evidence_fusion": True, "route_active": True,
    }, max_answers=1, structured_vlm=True)

    assert result["answers"][0]["video_id"] == "K01_V001"
    assert result["answers"][0]["frame_id"] == 800
    assert result["answers"][0]["answer"] == "25 độ"
    assert result["answers"][0]["evidence_sources"] == ["visual", "asr"]
    assert "25 độ" in vlm.prompts[0]
    assert "11.00s-13.00s" in vlm.prompts[0]
