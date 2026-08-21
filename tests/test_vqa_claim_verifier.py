from __future__ import annotations

import pandas as pd

from src.reranking.qna_modality_router import QNAModalityRouter
from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.vqa.answer_provider import AnswerProviderResponse
from src.vqa.claim_verifier import derive_claim_policy, verify_claim_roles


def _packet(*, asr=(), ocr=(), claims=()):
    return {
        "asr_chunks": list(asr),
        "ocr_text": list(ocr),
        "claim_evidence": list(claims),
    }


def test_fana_like_claim_rejects_generic_khanh_hoa_charity_and_accepts_role_join():
    policy = derive_claim_policy(
        "Phóng sự về Câu lạc bộ FANA tặng quà tại một xã ở Khánh Hòa.",
        specialist_sources=("asr", "ocr"),
    )
    assert {claim.text for claim in policy.claims} >= {"FANA", "Khánh Hòa"}

    wrong = verify_claim_roles(
        "Trường Sa",
        _packet(asr=[{"source": "asr", "text": "Tặng quà tại huyện Trường Sa tỉnh Khánh Hòa."}]),
        policy,
        declared_sources=("asr",), answer_sources=("asr", "ocr"),
    )
    assert wrong.accepted is False
    assert wrong.reason == "query_claim_not_covered"

    right = verify_claim_roles(
        "Giang Ly",
        _packet(
            ocr=[{"source": "ocr", "text": "UBND Xã Giang Ly"}],
            claims=[
                {"source": "asr", "role": "claim_support", "text": "Câu lạc bộ FANA đến Khánh Hòa."},
            ],
        ),
        policy,
        declared_sources=("asr",), answer_sources=("asr", "ocr"),
    )
    assert right.accepted is True
    assert right.answer_sources == ("ocr",)
    assert right.role_sources == ("asr", "ocr")


def test_numeric_phrase_requires_all_descriptors_not_just_same_weight_and_meat():
    policy = derive_claim_policy(
        "Nguyên liệu có 200g thịt nạc xay.", specialist_sources=("ocr", "asr")
    )
    numeric = next(claim for claim in policy.claims if claim.kind == "numeric_phrase")
    assert numeric.text == "200g thịt nạc xay"

    wrong = verify_claim_roles(
        "Bánh rán burger bò",
        _packet(
            ocr=[{"source": "ocr", "text": "Thịt bò xay 200g"}],
            asr=[{"source": "asr", "text": "Hôm nay làm món bánh rán burger bò."}],
        ),
        policy,
        declared_sources=("ocr", "asr"), answer_sources=("ocr", "asr"),
    )
    assert wrong.accepted is False
    assert wrong.reason == "query_claim_not_covered"

    right = verify_claim_roles(
        "Bánh ít trần",
        _packet(
            ocr=[{"source": "ocr", "text": "Thịt nạc dăm xay 200g"}],
            asr=[{"source": "asr", "text": "Hôm nay cô làm món bánh ít trần."}],
        ),
        policy,
        declared_sources=("ocr", "asr"), answer_sources=("ocr", "asr"),
    )
    assert right.accepted is True
    assert right.role_sources == ("asr", "ocr")


def test_router_claim_evidence_is_same_video_and_canonical_only():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    policy = derive_claim_policy("Câu lạc bộ FANA tại Khánh Hòa.", specialist_sources=("asr",))
    packet = {}
    router._attach_claim_evidence(
        packet,
        asr_rows=pd.DataFrame([
            {"video_id": "V1", "kf_n": 7, "frame_idx": 70, "pts_time": 18.0,
             "start": 17.0, "end": 19.0, "chunk": "Câu lạc bộ FANA đến Khánh Hòa."},
        ]),
        ocr_rows=pd.DataFrame(),
        claim_policy=policy,
    )
    assert len(packet["claim_evidence"]) == 2
    assert {(item["kf_n"], item["frame_idx"]) for item in packet["claim_evidence"]} == {(7, 70)}
    assert all(item["role"] == "claim_support" for item in packet["claim_evidence"])


def test_global_claim_channel_requires_every_anchor_in_one_video():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    policy = derive_claim_policy("Câu lạc bộ FANA tại Khánh Hòa.", specialist_sources=("asr",))

    def fake_global(texts, modality, topk=100):
        text = tuple(texts)[0]
        if text == "FANA":
            return [{"video_id": "RIGHT", "kf_n": 7, "frame_idx": 70, "pts_time": 7.0,
                     "rank": 4, "modality_score": 0.5, "text": "Câu lạc bộ FANA"}]
        return [
            {"video_id": "WRONG", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0,
             "rank": 1, "modality_score": 0.8, "text": "Khánh Hòa"},
            {"video_id": "RIGHT", "kf_n": 9, "frame_idx": 90, "pts_time": 9.0,
             "rank": 3, "modality_score": 0.6, "text": "Khánh Hòa"},
        ]

    router.global_candidates_multi = fake_global
    metadata = pd.DataFrame([
        {"video_id": "RIGHT", "kf_n": 7, "frame_idx": 70, "pts_time": 7.0,
         "chunk": "Câu lạc bộ FANA"},
        {"video_id": "RIGHT", "kf_n": 9, "frame_idx": 90, "pts_time": 9.0,
         "chunk": "Khánh Hòa"},
        {"video_id": "WRONG", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0,
         "chunk": "Khánh Hòa"},
    ])
    router._load = lambda _modality: (None, metadata)
    rows = router.global_claim_candidates(policy, ("asr",))

    assert [row["video_id"] for row in rows] == ["RIGHT"]
    assert rows[0]["rank"] == 1
    assert len(rows[0]["claim_coverage"]) == 2


def test_global_claim_channel_fails_closed_for_non_discriminative_anchor():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    policy = derive_claim_policy("A TV broadcast shows a channel logo.", specialist_sources=("ocr",))

    def fake_global(_texts, _modality, topk=100):
        return [
            {"video_id": f"V{index:02d}", "kf_n": index, "frame_idx": index * 10,
             "pts_time": float(index), "rank": index, "modality_score": 1.0,
             "text": "TV channel logo"}
            for index in range(1, 14)
        ]

    router.global_candidates_multi = fake_global
    router._load = lambda _modality: (None, pd.DataFrame())

    assert router.global_claim_candidates(policy, ("ocr",)) == []


class _Provider:
    provider_name = "test"
    model_id = "test"
    is_remote = False

    def __init__(self, answer: str):
        self.answer_text = answer

    def answer(self, request):
        return AnswerProviderResponse(
            candidate_id=request.evidence.candidate_id,
            answer=self.answer_text,
            grounding_score=1.0,
            answer_confidence=1.0,
            abstain=False,
            provider=self.provider_name,
            model_id=self.model_id,
        )


class _EvidenceProvider:
    def __init__(self, packet):
        self.packet = packet

    def evidence_packet_for_candidate(self, *_args, **_kwargs):
        return self.packet


def _prepared(provider):
    return {
        "query": "Phóng sự về Câu lạc bộ FANA tặng quà tại một xã ở Khánh Hòa.",
        "question": "Xã đó tên là gì?",
        "route_active": True,
        "evidence_fusion": True,
        "required_sources": ("asr",),
        "declared_sources": ("asr",),
        "support_sources": ("asr", "ocr"),
        "evidence_provider": provider,
        "candidates": [{
            "video_id": "V1", "frame_idx": 100, "kf_n": 10, "pts_time": 20.0,
            "frame_path": "unused.jpg", "video_rank": 1, "base_score": 0.8,
            "source": "asr",
        }],
    }


def test_pipeline_rejects_answer_from_generic_wrong_video_even_when_provider_is_confident():
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    pipeline.answer_provider = _Provider("Trường Sa")
    pipeline.evidence_verifier = None
    packet = {
        "frames": [{"video_id": "V1", "frame_idx": 100, "kf_n": 10, "pts_time": 20.0}],
        "sources": ["visual", "asr"],
        "asr_chunks": [{"source": "asr", "text": "Tặng quà tại huyện Trường Sa tỉnh Khánh Hòa."}],
        "ocr_text": [],
        "claim_evidence": [],
    }

    result = pipeline.answer_ranked_candidates(_prepared(_EvidenceProvider(packet)), max_answers=1)

    assert result["answers"] == []
    assert result["answer_trace"][0]["status"] == "rejected_claim_verification"
    assert result["answer_trace"][0]["verification"]["claim_verification"]["reason"] == "query_claim_not_covered"


def test_pipeline_accepts_answer_when_entity_and_answer_have_distinct_roles():
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    pipeline.answer_provider = _Provider("Giang Ly")
    pipeline.evidence_verifier = None
    packet = {
        "frames": [{"video_id": "V1", "frame_idx": 100, "kf_n": 10, "pts_time": 20.0}],
        "sources": ["visual", "asr", "ocr"],
        "asr_chunks": [{"source": "asr", "text": "Phóng sự thiện nguyện tại Khánh Hòa."}],
        "ocr_text": [{"source": "ocr", "text": "UBND Xã Giang Ly"}],
        "claim_evidence": [{
            "source": "asr", "role": "claim_support",
            "text": "Câu lạc bộ FANA đến Khánh Hòa.",
        }],
    }

    result = pipeline.answer_ranked_candidates(_prepared(_EvidenceProvider(packet)), max_answers=1)

    assert result["answers"][0]["answer"] == "Giang Ly"
    claim = result["answers"][0]["verification"]["claim_verification"]
    assert claim["accepted"] is True
    assert claim["role_sources"] == ["asr", "ocr"]
