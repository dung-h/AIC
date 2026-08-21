"""Regression tests for P0 Q&A evidence routing failures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.reranking.qna_modality_router import QNAModalityRouter
from src.vqa.grounding import GroundingEvidence
from src.vqa.query_planner import build_vqa_query_plan
from src.vqa.selector import allocate_recall_preserving_candidates


class _Embedder:
    def embed(self, texts, batch_size=1, normalize=True):
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _candidate(video_id: str, kf_n: int, source: str) -> dict:
    return {
        "video_id": video_id,
        "frame_idx": kf_n * 100,
        "kf_n": kf_n,
        "pts_time": float(kf_n),
        "video_rank": kf_n - 1,
        "base_score": 1.0,
        "source": source,
        "frame_path": f"{video_id}_{kf_n}.jpg",
    }


def test_query_plan_keeps_independent_description_question_and_exact_anchor_views():
    plan = build_vqa_query_plan(
        "Chương trình nấu ăn có công thức thịt nạc xay.",
        "Tên món nào dùng 200g thịt nạc xay?",
        question_type="screen_text",
        modalities=("ocr", "asr"),
    )

    assert plan.modality_queries["ocr"][0].startswith("Chương trình nấu ăn")
    assert "Tên món nào dùng 200g thịt nạc xay?" in plan.modality_queries["ocr"]
    assert any("200g" in anchor for anchor in plan.exact_anchors)
    assert "200g thịt nạc xay" in plan.exact_anchors
    assert "200g thịt nạc xay" in plan.modality_queries["ocr"]
    assert plan.to_dict()["external_grounding_status"] == "disabled_by_default"


def test_query_plan_adds_only_source_grounded_discriminative_fact_views():
    plan = build_vqa_query_plan(
        "Phóng sự về Câu lạc bộ FANA tại Khánh Hòa và anh hùng Nguyễn Trung Trực.",
        "Địa phương hoặc nhân vật được nhắc đến là ai?",
        question_type="spoken_fact",
        modalities=("asr", "ocr"),
    )

    assert "Câu lạc bộ FANA" in plan.query_fact_views["asr"]
    assert "Nguyễn Trung Trực" in plan.query_fact_views["ocr"]
    # Query facts are diagnostic/external hypotheses. They must pass the
    # router's corpus-selectivity gate before they can influence video RRF.
    assert "Câu lạc bộ FANA" not in plan.modality_queries["asr"]
    assert "Khánh Hòa" not in plan.query_fact_views["asr"]

    generic = build_vqa_query_plan(
        "An HTV7 TV broadcast shows a clock.", "What time is displayed?",
        question_type="screen_text", modalities=("ocr",),
    )
    assert generic.query_fact_views["ocr"] == ()


def test_multi_view_retrieval_rescues_rare_quote_with_local_lexical_evidence():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    router.text_mode = "dense"
    router.embedder = _Embedder()
    router._lexical_indexes = {}
    router._indexes = {
        "asr": (
            np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
            pd.DataFrame([
                {"video_id": "WRONG", "kf_n": 1, "frame_idx": 100,
                 "pts_time": 1.0, "chunk": "Một bản tin không liên quan."},
                {"video_id": "RIGHT", "kf_n": 2, "frame_idx": 200,
                 "pts_time": 2.0,
                 "chunk": "Hóa hồng Nhật Tảo quanh thiên địa, kiếm bạt Kiên Giang khắp vị thần."},
            ]),
        )
    }

    rows = router.global_candidates_multi(
        ["Nguyễn Trung Trực", "Hỏa hồng Nhựt Tảo oanh thiên địa"],
        "asr",
        topk=2,
    )

    assert rows[0]["video_id"] == "RIGHT", rows[0]["view_provenance"]
    assert rows[0]["score_mode"] == "multi_view_rrf"
    assert {entry["score_mode"] for entry in rows[0]["view_provenance"]} >= {
        "bm25_coverage"
    }


def test_parallel_selector_does_not_drop_ocr_hit_when_asr_is_absent_on_frame():
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline._local_vlm = None
    prepared = {
        "_candidate_pool": [
            _candidate("ASR_VIDEO", 1, "asr"),
            _candidate("OCR_VIDEO", 2, "ocr"),
        ],
        "retrieved_video_ids": ["ASR_VIDEO", "OCR_VIDEO"],
        "evidence_fusion": False,
    }

    selected = pipeline.select_prepared_candidates(
        prepared,
        top_videos=2,
        max_vlm_candidates=2,
        required_modalities="asr,ocr",
        visual_selector_policy="balanced",
    )

    assert selected["candidate_count"] == 2
    assert {item["video_id"] for item in selected["candidates"]} == {"ASR_VIDEO", "OCR_VIDEO"}
    assert selected["answer_selection"]["modality_policy"] == "parallel_union_primary_verified"


def test_selector_prefers_authoritative_ocr_over_asr_support_within_one_video():
    result = allocate_recall_preserving_candidates(
        [
            {
                **_candidate("V1", 1, "visual"),
                "sources": ["visual", "ocr"],
                "modality_score": 0.1,
                "text": "weak OCR text on a visual neighbour",
            },
            {
                **_candidate("V1", 2, "ocr"),
                "modality_score": 0.9,
                "text": "recipe title",
            },
        ],
        ["V1"],
        max_vlm_candidates=1,
        specialist_modalities=("ocr", "asr"),
        specialist_reservation=0,
        temporal_reservation=0,
        prefer_specialist_anchors=True,
        selection_policy="coverage",
    )

    assert result.selected[0]["source"] == "ocr"
    assert result.selected[0]["kf_n"] == 2


def test_primary_evidence_is_type_specific_and_ambiguous_multimodal_request_fails_closed():
    assert VQAPipelineV3._primary_evidence_sources(
        "screen_text", ("asr", "ocr")
    ) == ("ocr",)
    assert VQAPipelineV3._primary_evidence_sources(
        "spoken_fact", ("ocr", "asr")
    ) == ("asr",)
    with pytest.raises(ValueError, match="multiple specialist"):
        VQAPipelineV3._primary_evidence_sources("action", ("asr", "ocr"))


def test_grounded_query_variants_are_retrieval_only_and_can_add_ocr_support():
    evidence = GroundingEvidence(
        source_url="https://example.org/fana",
        source_title="FANA đến xã Giang Ly",
        query_variants=("Câu lạc bộ Nắng Ấm Yêu Thương FANA xã Giang Ly",),
        provider="test",
    )
    plan = build_vqa_query_plan(
        "Phóng sự về hoạt động của Câu lạc bộ FANA tại Khánh Hòa.",
        "Trên biển hiệu có thể thấy tên xã nào?",
        question_type="spoken_fact",
        modalities=("asr",),
        external_evidence=(evidence,),
        external_grounding_attempted=True,
    )

    assert plan.support_modalities == ("asr", "ocr")
    assert "FANA xã Giang Ly" in plan.modality_queries["asr"][-1]
    serialised = plan.to_dict()
    assert serialised["external_grounding_status"] == "used"
    assert serialised["external_evidence"][0]["source_url"] == "https://example.org/fana"


def test_quote_grounding_compiles_atomic_asr_and_ocr_views_with_provenance():
    evidence = GroundingEvidence(
        source_url="https://example.org/nguyen-trung-truc",
        source_title="Ý nghĩa ngày giỗ Nguyễn Trung Trực",
        source_snippet=(
            "Tại Kiên Giang lưu truyền hai câu thơ: “Hỏa hồng Nhựt Tảo oanh thiên địa, "
            "Kiếm bạt Kiên Giang khấp quỷ thần.”"
        ),
        query_variants=("Đình thần Nguyễn Trung Trực tại Kiên Giang",),
        provider="fixture",
    )

    plan = build_vqa_query_plan(
        "Phóng sự giới thiệu về Nguyễn Trung Trực tại Kiên Giang.",
        "Hai câu thơ được đọc trong phóng sự là gì?",
        question_type="spoken_fact",
        modalities=("asr", "ocr"),
        external_evidence=(evidence,),
        external_grounding_enabled=True,
        external_grounding_attempted=True,
    )

    exact_quote = "Hỏa hồng Nhựt Tảo oanh thiên địa, Kiếm bạt Kiên Giang khấp quỷ thần"
    assert plan.modality_queries["asr"][:3] == (
        "Phóng sự giới thiệu về Nguyễn Trung Trực tại Kiên Giang. Hai câu thơ được đọc trong phóng sự là gì?",
        "Phóng sự giới thiệu về Nguyễn Trung Trực tại Kiên Giang.",
        "Hai câu thơ được đọc trong phóng sự là gì?",
    )
    assert exact_quote in plan.quote_views["asr"]
    assert "Hỏa hồng Nhựt Tảo oanh thiên địa" in plan.quote_views["asr"]
    assert "Kiếm bạt Kiên Giang khấp quỷ thần" in plan.quote_views["ocr"]
    assert "Hoa hong Nhut Tao oanh thien dia" in plan.quote_views["asr"]
    assert "Nguyễn Trung Trực" in plan.alias_views["asr"]
    assert "Kiên Giang" in plan.alias_views["ocr"]
    assert plan.modality_queries["asr"][-1] == "Đình thần Nguyễn Trung Trực tại Kiên Giang"
    quote_trace = [
        item for item in plan.hypothesis_provenance
        if item.kind == "quote" and item.text == exact_quote
    ]
    assert len(quote_trace) == 1
    assert quote_trace[0].source_field == "source_snippet"
    assert quote_trace[0].source_url == evidence.source_url
    assert all("answer" not in item.to_dict() for item in plan.hypothesis_provenance)


def test_grounding_extracts_cited_poetry_when_search_snippet_omits_quote_marks():
    evidence = GroundingEvidence(
        source_url="https://example.org/nguyen-trung-truc",
        source_title="Anh hùng Nguyễn Trung Trực",
        source_snippet=(
            "Nhà văn ca tụng chiến công bằng hai câu thơ: Hỏa hồng Nhật Tảo "
            "oanh thiên địa, Kiếm bạt Kiên Giang khấp quỷ thần."
        ),
        query_variants=("Nguyễn Trung Trực Kiên Giang",),
        provider="fixture",
    )

    plan = build_vqa_query_plan(
        "Phóng sự về Nguyễn Trung Trực.", "Hai câu thơ được đọc là gì?",
        modalities=("asr",), external_evidence=(evidence,),
    )

    assert "Hỏa hồng Nhật Tảo oanh thiên địa" in plan.quote_views["asr"]
    assert "Kiếm bạt Kiên Giang khấp quỷ thần" in plan.quote_views["asr"]


def test_grounding_hypotheses_are_bounded_and_ignore_evidence_after_budget():
    evidence = tuple(
        GroundingEvidence(
            source_url=f"https://example.org/{index}",
            source_title=f"Nhân vật Lịch Sử {index}",
            source_snippet=(
                f"Nguồn {index} viết: “Câu thơ thứ nhất của nhân vật {index}, "
                f"câu thơ thứ hai của nhân vật {index}.”"
            ),
            query_variants=(f"Nhân vật Lịch Sử {index}",),
            provider="fixture",
        )
        for index in range(8)
    )

    plan = build_vqa_query_plan(
        "Phóng sự lịch sử.", "Câu thơ được đọc là gì?",
        modalities=("asr", "ocr"), external_evidence=evidence,
    )

    assert len(plan.hypothesis_provenance) <= 24  # 12 quote + 12 alias.
    assert {item.evidence_index for item in plan.hypothesis_provenance} <= set(range(5))
    assert plan.quote_views["asr"] == plan.quote_views["ocr"]
    assert plan.alias_views["asr"] == plan.alias_views["ocr"]


def test_ranked_answers_rejects_external_grounding_in_offline_mode_before_models_load():
    class _EnabledResolver:
        enabled = True

    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.answer_provider = None
    with pytest.raises(ValueError, match="offline ranked_answers cannot use external grounding"):
        pipeline.ranked_answers(
            "phóng sự", "Tên địa phương là gì?",
            grounding_resolver=_EnabledResolver(),
            offline=True,
        )


def test_external_resolver_is_not_called_for_routine_visual_question():
    class _EmptyKIS:
        def search(self, *_args, **_kwargs):
            return []

    class _CountingResolver:
        enabled = True

        def __init__(self):
            self.calls = 0

        def resolve(self, _request):
            self.calls += 1
            return ()

    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = _EmptyKIS()
    resolver = _CountingResolver()
    prepared = pipeline.prepare_ranked_candidates(
        "Một chiếc xe màu đỏ đang chạy trên đường.",
        "Xe có màu gì?",
        grounding_resolver=resolver,
    )

    assert resolver.calls == 0
    assert prepared["query_plan"]["external_grounding_status"] == "not_eligible"


def test_external_resolver_is_called_for_entity_fact_and_preserves_source_trace():
    class _EmptyKIS:
        def search(self, *_args, **_kwargs):
            return []

    class _FactResolver:
        enabled = True

        def __init__(self):
            self.calls = 0

        def resolve(self, _request):
            self.calls += 1
            return (GroundingEvidence(
                source_url="https://example.org/fana",
                source_title="FANA tại Giang Ly",
                query_variants=("FANA Giang Ly Khánh Hòa",),
                provider="test",
            ),)

    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = _EmptyKIS()
    resolver = _FactResolver()
    prepared = pipeline.prepare_ranked_candidates(
        "Phóng sự về câu lạc bộ thiện nguyện FANA tại Khánh Hòa.",
        "Trên biển hiệu có thể thấy tên xã nào?",
        grounding_resolver=resolver,
    )

    assert resolver.calls == 1
    assert prepared["query_plan"]["external_grounding_status"] == "used"
    assert prepared["query_plan"]["external_evidence"][0]["source_title"] == "FANA tại Giang Ly"
