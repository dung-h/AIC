import pytest

from src.pipelines.trake_pipeline import MultimodalTrakePipeline, TrakePipeline


class _Retriever:
    def __init__(self, modality):
        self.modality = modality

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        if candidate_videos is not None and "v1" not in set(candidate_videos):
            return []
        return [{
            "event_index": event.index,
            "video_id": "v1",
            "modality": self.modality,
            "score": 0.9 - 0.01 * event.index,
            "frame_idx": 10 + 10 * event.index,
            "kf_n": 1 + event.index,
            "pts_time": 1.0 + event.index,
            "source_id": f"{self.modality}-test",
        }]


def _retrievers():
    return {"visual": _Retriever("visual"), "asr": _Retriever("asr")}


def test_multimodal_alignment_policy_is_explicit_and_wired():
    pipeline = MultimodalTrakePipeline(
        _retrievers(), alignment_policy="coverage_coherent_v1"
    )

    assert pipeline.aligner.alignment_policy == "coverage_coherent_v1"
    result = pipeline.align(["first", "second"], top_k_videos=1)
    assert result["diagnostics"]["alignment_policy"] == "coverage_coherent_v1"
    assert result["results"][0]["frame_ids"] == [10, 20]


def test_trake_pipeline_passes_explicit_alignment_policy():
    pipeline = TrakePipeline(
        mode="multimodal",
        retrievers=_retrievers(),
        alignment_policy="coverage_coherent_v1",
    )

    assert pipeline._multimodal.aligner.alignment_policy == "coverage_coherent_v1"


def test_unknown_alignment_policy_fails_closed():
    with pytest.raises(ValueError, match="alignment_policy"):
        MultimodalTrakePipeline(_retrievers(), alignment_policy="unknown")
