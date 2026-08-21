"""Contract tests for web-image references that retrieve only local frames."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.pipelines.vqa_pipeline_v3 import VQAPipelineV3
from src.vqa.image_grounding import (
    ImageGroundingCandidate,
    ImageGroundingRequest,
    ImageGroundingResult,
    SearxNGImageGroundingProvider,
    image_grounding_eligibility,
)


def _image_bytes() -> bytes:
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (16, 16), (160, 40, 200)).save(stream, "JPEG")
    return stream.getvalue()


def test_image_grounder_uses_allowlisted_web_image_only_as_local_vkis_seed():
    requests = []

    class _FakeVKIS:
        def search_image(self, path, topk):
            assert path.endswith(".jpg")
            assert topk == 3
            return [
                ("RIGHT", 200, 8.0, 0.95),
                ("OTHER", 300, 12.0, 0.70),
            ]

    def search_transport(url, _timeout):
        requests.append(url)
        return {"results": [{
            "url": "https://example.org/labubu",
            "img_src": "https://cdn.example.org/labubu.jpg",
            "title": "Labubu reference image",
        }]}

    provider = SearxNGImageGroundingProvider(
        "https://search.example.org",
        allowed_domains=("example.org",),
        allow_any_image_host=False,
        vkis_factory=_FakeVKIS,
        max_references=1,
        search_transport=search_transport,
        image_transport=lambda _url, _timeout, _max_bytes: _image_bytes(),
    )

    result = provider.retrieve(ImageGroundingRequest("Labubu toy", "What is shown?"), topk=3)

    assert "categories=images" in requests[0]
    assert result.status == "candidates_ready"
    assert result.candidates[0].video_id == "RIGHT"
    assert result.candidates[0].frame_idx == 200
    assert result.candidates[0].reference_urls == ("https://example.org/labubu",)


def test_image_grounding_gate_avoids_fact_and_text_contract_queries():
    assert image_grounding_eligibility(
        "Phóng sự về câu lạc bộ FANA.", "Tên xã nào được nhắc tới?",
        question_type="spoken_fact", specialist_modalities=("asr",),
    ) == (False, ("specialist_text_contract",))

    eligible, reasons = image_grounding_eligibility(
        "Một Labubu toy đặt trên bàn.", "Nhân vật nào xuất hiện?",
        question_type="action", specialist_modalities=(),
    )
    assert eligible is True
    assert "visual_entity_cue" in reasons


def test_vqa_image_grounding_adds_only_canonical_local_seed_frame(tmp_path):
    class _KIS:
        materialized = False

        def search(self, _query, topk):
            assert topk == 100
            return [("BASE", 100, 4.0, 0.8)]

    class _Provider:
        enabled = True

        def retrieve(self, _request, *, topk):
            assert topk == 100
            return ImageGroundingResult(
                status="candidates_ready",
                candidates=(ImageGroundingCandidate(
                    video_id="RIGHT", frame_idx=200, pts_time=8.0,
                    score=0.03, rank=1,
                    reference_urls=("https://example.org/labubu",),
                ),),
            )

    base_path = tmp_path / "base.jpg"
    right_path = tmp_path / "right.jpg"
    base_path.write_bytes(b"base")
    right_path.write_bytes(b"right")
    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.kis = _KIS()
    pipeline.km = pd.DataFrame([
        {"video_id": "BASE", "kf_n": 1, "frame_idx": 100, "pts_time": 4.0},
        {"video_id": "RIGHT", "kf_n": 2, "frame_idx": 200, "pts_time": 8.0},
    ])
    pipeline._frame_path = lambda video_id, _kf_n: str(
        base_path if video_id == "BASE" else right_path
    )
    pipeline._local_candidates = lambda _query, _video_ids, _budget, **_kwargs: [
        ("BASE", 100, 1, 0.8),
    ]

    prepared = pipeline.prepare_ranked_candidates(
        "Một Labubu toy đặt trên bàn.", "Nhân vật nào xuất hiện?",
        top_videos=2, max_vlm_candidates=2,
        image_grounding_provider=_Provider(),
        visual_selector_policy="balanced",
        return_candidate_pool=True,
    )

    right = [item for item in prepared["_candidate_pool"] if item["video_id"] == "RIGHT"]
    assert prepared["image_grounding"]["status"] == "candidates_ready"
    assert right[0]["frame_idx"] == 200
    assert right[0]["kf_n"] == 2
    assert "external_image" in right[0]["sources"]
    assert all(item["frame_path"] != "https://example.org/labubu" for item in prepared["candidates"])


def test_image_provider_requires_explicit_host_policy():
    with pytest.raises(ValueError, match="allowed_domains"):
        SearxNGImageGroundingProvider(
            "https://search.example.org",
            allowed_domains=(),
            allow_any_image_host=False,
            vkis_factory=lambda: object(),
        )


def test_ranked_answers_rejects_image_grounding_in_offline_mode_before_models_load():
    class _EnabledProvider:
        enabled = True

    pipeline = VQAPipelineV3.__new__(VQAPipelineV3)
    pipeline.answer_provider = None
    with pytest.raises(ValueError, match="offline ranked_answers cannot use external grounding"):
        pipeline.ranked_answers(
            "Một Labubu toy.", "Nhân vật nào xuất hiện?",
            image_grounding_provider=_EnabledProvider(),
            offline=True,
        )
