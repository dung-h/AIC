from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.pipelines.vqa_pipeline_v3 as vqa_module
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.reranking.qna_modality_router import QNAModalityRouter
from src.vqa.answer_provider import AnswerProviderResponse


class _FakeKIS:
    def search(self, query, topk=20):
        return [
            ("V1", 100, 1, 0.95),
            ("V2", 200, 1, 0.85),
        ][:topk]


class _FakeGlobalRouter:
    def global_candidates(self, text, modality, topk=100):
        assert text == "weather forecast\nWhat temperature is spoken?"
        assert topk == 100
        if modality == "asr":
            return [{
                "video_id": "V1", "kf_n": 2, "frame_idx": 110,
                "pts_time": 2.0, "modality": "asr", "modality_score": 0.99,
                "rank": 1, "score_mode": "dense", "text": "Nha Trang 25 độ",
                "evidence": {"modality": "asr", "text": "Nha Trang 25 độ"},
            }]
        return []

    def evidence_packet_for_candidate(self, candidate, query, question, *, modalities=None):
        return {
            "index_source": "global_modality_registry",
            "index_modalities": list(modalities or ("asr",)),
            "video_id": candidate["video_id"],
            "anchor": {"frame_idx": candidate["frame_idx"], "kf_n": candidate["kf_n"],
                        "pts_time": candidate["pts_time"]},
            "frames": [],
            "asr_chunks": [{"source": "asr", "text": "Nha Trang 25 độ",
                            "start_time": 1.5, "end_time": 2.5,
                            "timestamp": 1.5, "rank": 1, "distance_s": 0.0}],
            "ocr_text": [],
            "timestamps": [],
            "sources": ["visual", "asr"],
        }


def _prepare_pipeline(tmp_path: Path) -> VQAPipelineV3:
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = _FakeKIS()
    pipeline.km = pd.DataFrame([
        {"video_id": "V1", "kf_n": 1, "frame_idx": 100, "pts_time": 1.0},
        {"video_id": "V1", "kf_n": 2, "frame_idx": 110, "pts_time": 2.0},
        {"video_id": "V1", "kf_n": 3, "frame_idx": 120, "pts_time": 3.0},
        {"video_id": "V2", "kf_n": 1, "frame_idx": 200, "pts_time": 1.0},
        {"video_id": "V2", "kf_n": 2, "frame_idx": 210, "pts_time": 2.0},
    ])
    for video_id, kf_n in (("V1", 1), ("V1", 2), ("V1", 3), ("V2", 1), ("V2", 2)):
        (tmp_path / f"{video_id}_{kf_n}.jpg").write_bytes(b"frame")
    pipeline._frame_path = lambda video_id, kf_n: str(tmp_path / f"{video_id}_{int(kf_n)}.jpg")
    pipeline._local_candidates = lambda query, video_ids, frames_per_video, **kwargs: [
        ("V1", 100, 1, 0.90),
        ("V1", 120, 3, 0.80),
        ("V2", 200, 1, 0.70),
    ][: len(video_ids) * frames_per_video]
    pipeline._asr = None
    pipeline._ocr = None
    pipeline._asr_by_video = None
    pipeline._ocr_by_video = None
    pipeline._context_cache_stats = {
        "asr_video_hits": 0, "asr_video_misses": 0,
        "ocr_video_hits": 0, "ocr_video_misses": 0,
    }
    return pipeline


def test_routed_prepare_calls_wave2_rrf_owner_and_preserves_provenance(tmp_path, monkeypatch):
    pipeline = _prepare_pipeline(tmp_path)
    calls = []

    def fake_rrf(channels, weights, **kwargs):
        calls.append((channels, dict(weights), kwargs))
        assert set(channels) == {"visual", "asr"}
        return [
            {"video_id": "V1", "video_rank": 1, "rrf_score": 0.9,
             "rrf_guard": "none", "visual_rank": 1, "asr_rank": 1},
            {"video_id": "V2", "video_rank": 2, "rrf_score": 0.2,
             "rrf_guard": "visual_baseline", "visual_rank": 2},
        ]

    monkeypatch.setattr(vqa_module, "weighted_video_rrf", fake_rrf)
    prepared = pipeline.prepare_ranked_candidates(
        "weather forecast", "What temperature is spoken?",
        top_videos=2, frames_per_video=2, max_vlm_candidates=4,
        required_modalities=["asr"], question_type="spoken_fact",
        global_modality_router=_FakeGlobalRouter(),
    )

    assert len(calls) == 1
    assert prepared["route_active"] is True
    assert prepared["route_state"] == "specialist_success"
    assert prepared["candidate_state"] == "candidate_available"
    assert prepared["wrong_video_state"] == "not_evaluated"
    assert prepared["rrf_videos"][0]["video_id"] == "V1"
    assert prepared["candidate_source_counts"]["asr"] >= 1
    v1 = next(item for item in prepared["candidates"] if item["video_id"] == "V1")
    assert v1["video_fusion"]["channel_ranks"] == {"asr": 1, "visual": 1}
    # The selector keeps the visual anchor as the canonical answer frame.
    # Specialist evidence remains a separate candidate/evidence frame and
    # must not evict that visual anchor under the bounded budget.
    assert v1["source"] == "visual"
    assert {record["source"] for record in v1["provenance"]} == {"visual", "temporal"}
    asr_candidate = next(item for item in prepared["candidates"]
                         if item["video_id"] == "V1" and item["source"] == "asr")
    assert asr_candidate["evidence"]["text"] == "Nha Trang 25 độ"
    assert all(frame["video_id"] == "V1" for frame in v1["evidence_frames"])


def test_global_router_builds_packet_from_its_loaded_metadata():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    router.active_modalities = ("asr", "ocr")
    asr = pd.DataFrame([{
        "video_id": "V1", "kf_n": 2, "frame_idx": 110, "pts_time": 2.0,
        "start": 1.5, "end": 2.5, "chunk": "Nha Trang 25 độ",
    }])
    ocr = pd.DataFrame([{
        "video_id": "V1", "kf_n": 2, "frame_idx": 110, "pts_time": 2.0,
        "ocr_text": "DỰ BÁO THỜI TIẾT",
    }])
    router._load = lambda modality: ((None, asr) if modality == "asr" else (None, ocr))

    packet = router.evidence_packet_for_candidate(
        {"video_id": "V1", "kf_n": 2, "frame_idx": 110, "pts_time": 2.0},
        "weather forecast", "What temperature is spoken?", modalities=("asr", "ocr"),
    )

    assert packet["index_source"] == "global_modality_registry"
    assert packet["has_spoken_evidence"] is True
    assert packet["has_screen_text_evidence"] is True
    assert packet["asr_chunks"][0]["text"] == "Nha Trang 25 độ"
    assert packet["ocr_text"][0]["text"] == "DỰ BÁO THỜI TIẾT"


class _RecordingProvider:
    provider = "fake-provider"

    def __init__(self, response):
        self.response = response
        self.requests = []

    def answer(self, request):
        self.requests.append(request)
        return self.response


def _answer_pipeline(provider, asr_rows, ocr_rows):
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    pipeline.answer_provider = provider
    pipeline.evidence_verifier = None
    pipeline._asr = asr_rows
    pipeline._ocr = ocr_rows
    pipeline._asr_by_video = {"V1": asr_rows}
    pipeline._ocr_by_video = {"V1": ocr_rows}
    pipeline._context_cache_stats = {
        "asr_video_hits": 0, "asr_video_misses": 0,
        "ocr_video_hits": 0, "ocr_video_misses": 0,
    }
    return pipeline


def test_route_active_answer_uses_same_video_packet_and_multiframe_provider_request(tmp_path):
    paths = []
    for frame_id in (100, 110, 120):
        path = tmp_path / f"{frame_id}.jpg"
        path.write_bytes(b"frame")
        paths.append(str(path))
    provider = _RecordingProvider(AnswerProviderResponse(
        candidate_id="V1#1", answer="25 độ", grounding_score=0.9,
        answer_confidence=0.8, abstain=False, provider="fake-provider",
        model_id="fake-model",
    ))
    asr_rows = pd.DataFrame([{
        "vid": "V1", "start": 1.5, "end": 2.5, "chunk": "Nha Trang 25 độ",
    }])
    ocr_rows = pd.DataFrame([{
        "video_id": "V1", "pts_time": 2.0, "ocr_text": "DỰ BÁO THỜI TIẾT",
    }])
    pipeline = _answer_pipeline(provider, asr_rows, ocr_rows)

    def forbidden_context(*args, **kwargs):
        raise AssertionError("legacy global context must not be used in route-active mode")

    pipeline._asr_context = forbidden_context
    pipeline._ocr_context = forbidden_context
    candidate = {
        "video_id": "V1", "frame_idx": 100, "kf_n": 1, "pts_time": 2.0,
        "frame_path": paths[0], "video_rank": 0, "base_score": 0.9,
        "source": "visual", "sources": ["visual", "asr"],
        "evidence_frames": [
            {"video_id": "V1", "frame_idx": 110, "kf_n": 2,
             "pts_time": 2.0, "frame_path": paths[1], "role": "evidence",
             "modality": "asr"},
            {"video_id": "V1", "frame_idx": 120, "kf_n": 3,
             "pts_time": 3.0, "frame_path": paths[2], "role": "neighbor",
             "modality": "temporal"},
            {"video_id": "V2", "frame_idx": 999, "kf_n": 9,
             "pts_time": 2.0, "frame_path": paths[2], "role": "wrong-video"},
        ],
    }
    result = pipeline.answer_ranked_candidates({
        "query": "weather forecast", "question": "What temperature is spoken?",
        "candidates": [candidate], "candidate_count": 1,
        "candidate_pool_count": 3, "vlm_candidate_count": 1,
        "route_active": True, "evidence_fusion": True,
        "route_state": "specialist_success", "candidate_state": "candidate_available",
        "wrong_video_state": "not_evaluated", "required_sources": [],
    }, use_context=False, structured_vlm=True, max_answers=1)

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [frame.frame_id for frame in request.evidence.frames] == [100, 110, 120]
    assert all(frame.frame_id != 999 for frame in request.evidence.frames)
    assert all(frame.frame_path for frame in request.evidence.frames)
    assert request.evidence.video_id == "V1"
    assert "Nha Trang 25 độ" in request.evidence.asr_text
    assert "DỰ BÁO" in request.evidence.ocr_text
    assert result["answers"][0]["frame_id"] == 100
    assert result["route_state"] == "specialist_success"
    assert result["wrong_video_state"] == "not_evaluated"


def test_provider_abstain_is_excluded_and_candidate_state_remains_explicit():
    provider = _RecordingProvider(AnswerProviderResponse.abstained(
        candidate_id="V1#1", provider="fake-provider", model_id="fake-model",
        reason="insufficient_evidence",
    ))
    pipeline = _answer_pipeline(provider, pd.DataFrame(), pd.DataFrame())
    result = pipeline.answer_ranked_candidates({
        "query": "q", "question": "What is shown?",
        "candidates": [{
            "video_id": "V1", "frame_idx": 100, "kf_n": 1, "pts_time": 1.0,
            "frame_path": "anchor.jpg", "video_rank": 0, "base_score": 1.0,
            "source": "visual",
        }],
        "candidate_count": 1, "vlm_candidate_count": 1,
        "route_active": False, "evidence_fusion": False,
        "route_state": "baseline_success", "candidate_state": "candidate_available",
        "wrong_video_state": "not_evaluated",
    }, use_context=False, structured_vlm=True, max_answers=1)

    assert result["answers"] == []
    assert result["status"] == "no_valid_local_answer"
    assert result["candidate_state"] == "candidate_available"
    assert result["candidate_miss"] is False
    assert result["wrong_video_state"] == "not_evaluated"
    assert result["answer_trace"][0]["status"] == "rejected_provider_abstain"
