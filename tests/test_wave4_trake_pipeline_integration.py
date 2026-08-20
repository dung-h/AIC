"""Integration tests for explicit ASR and multimodal TRAKE modes."""

import json
import numpy as np
import pandas as pd
import pytest

from src.pipelines import trake_pipeline as module
from src.pipelines.trake_asr_index import ASRGlobalIndexError, SharedASRGlobalIndex
from src.pipelines.trake_pipeline import TrakePipeline


class FakeEventRetriever:
    def __init__(self, modality, rows):
        self.modality = modality
        self.rows = rows
        self.calls = []

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        self.calls.append((event.index, tuple(candidate_videos or ())))
        allowed = set(map(str, candidate_videos)) if candidate_videos is not None else None
        return [
            {
                **row,
                "event_index": event.index,
                "modality": self.modality,
            }
            for row in self.rows.get(event.index, [])
            if allowed is None or str(row["video_id"]) in allowed
        ][:top_k]


def _rows(video_id="video-1"):
    return {
        0: [{"video_id": video_id, "score": 0.92, "frame_idx": 10,
             "kf_n": 1, "pts_time": 1.0}],
        1: [{"video_id": video_id, "score": 0.91, "frame_idx": 20,
             "kf_n": 2, "pts_time": 2.0}],
    }


def _write_merged_asr(tmp_path, *, bad_canonical=False):
    """Create the persisted global-index contract used by production ASR."""
    asr_dir = tmp_path / "asr_global_merged_v2"
    asr_dir.mkdir()
    canonical = pd.DataFrame([
        {"video_id": "VIDEO-1", "kf_n": 1, "frame_idx": 10, "pts_time": 1.0},
        {"video_id": "VIDEO-1", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
    ])
    canonical_path = tmp_path / "global_keyframes.parquet"
    canonical.to_parquet(canonical_path, index=False)
    frame_values = [10, 20] if not bad_canonical else [10, 999]
    metadata = pd.DataFrame([
        {
            "embedding_row": 0,
            "video_id": "VIDEO-1",
            "chunk_index": 0,
            "start": 1.0,
            "end": 2.0,
            "text": "a spoken event",
            "kf_n": 1,
            "frame_idx": frame_values[0],
            "pts_time": 1.0,
        },
        {
            "embedding_row": 1,
            "video_id": "VIDEO-1",
            "chunk_index": 1,
            "start": 2.0,
            "end": 3.0,
            "text": "a second spoken event",
            "kf_n": 2,
            "frame_idx": frame_values[1],
            "pts_time": 2.0,
        },
    ])
    metadata.to_parquet(asr_dir / "retrieval.parquet", index=False)
    np.save(
        asr_dir / "embeddings.npy",
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    manifest = {
        "status": "ready",
        "index_id": "test-asr-global",
        "artifacts": {"retrieval": "retrieval.parquet", "embeddings": "embeddings.npy"},
        "canonical": {"path": str(canonical_path), "validated": True},
        "scope": {"video_ids": ["VIDEO-1", "SILENT-VIDEO"]},
        "packs": {"TEST": {"no_speech_videos": ["SILENT-VIDEO"]}},
    }
    (asr_dir / "asr_global_merge_v2_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return asr_dir


def test_explicit_asr_reads_shared_merged_index_and_preserves_no_speech(monkeypatch, tmp_path):
    asr_dir = _write_merged_asr(tmp_path)

    class FakeEmbedder:
        def embed(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(module, "get_text_embedder", lambda mode: FakeEmbedder())

    pipeline = TrakePipeline(asr_index_dir=asr_dir)

    assert pipeline.mode == "asr"
    assert pipeline._multimodal is None
    assert pipeline.index_diagnostics["index_id"] == "test-asr-global"
    assert pipeline.no_speech_videos == {"SILENT-VIDEO"}
    result = pipeline.align(["a spoken event"], top_k_videos=1)
    assert result[0]["video_id"] == "VIDEO-1"
    assert result[0]["path"][0]["frame_idx"] == 10
    assert pipeline.align(["a spoken event"], video_id="SILENT-VIDEO") == []


def test_asr_does_not_fallback_to_legacy_shards(tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    pd.DataFrame([{"vid": "VIDEO-1", "start": 1.0, "end": 2.0}]).to_parquet(
        legacy_dir / "asr_chunks_demo_ts.parquet"
    )
    with pytest.raises(RuntimeError, match="merged ASR global index.*legacy-shard fallback"):
        TrakePipeline(asr_index_dir=legacy_dir)


def test_asr_loader_rejects_noncanonical_frame_mapping(tmp_path):
    with pytest.raises(ASRGlobalIndexError, match="frame_idx disagrees"):
        SharedASRGlobalIndex(_write_merged_asr(tmp_path, bad_canonical=True))


def test_asr_alignment_returns_canonical_strictly_ordered_frames(monkeypatch, tmp_path):
    asr_dir = _write_merged_asr(tmp_path)

    class FakeEmbedder:
        def embed(self, texts):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(module, "get_text_embedder", lambda mode: FakeEmbedder())
    pipeline = TrakePipeline(asr_index_dir=asr_dir)
    results = pipeline.align(["first", "second"], top_k_videos=1)
    path = results[0]["path"]
    assert [row["frame_idx"] for row in path] == [10, 20]
    assert [row["pts_time"] for row in path] == [1.0, 2.0]


def test_asr_mode_selection_is_explicit():
    with pytest.raises(ValueError, match="choose 'asr' or 'multimodal'"):
        TrakePipeline(mode="visual")


def test_multimodal_flag_routes_both_modalities_and_returns_ordered_path():
    visual = FakeEventRetriever("visual", _rows())
    asr = FakeEventRetriever("asr", _rows())

    pipeline = TrakePipeline(
        mode="multimodal",
        retrievers={"visual": visual, "asr": asr},
    )
    results = pipeline.align([
        {"description": "first event"},
        {"description": "second event"},
    ], top_k_videos=100)

    assert len(results) == 1
    assert results[0]["video_id"] == "video-1"
    assert results[0]["frame_ids"] == [10, 20]
    assert [step["frame_idx"] for step in results[0]["path"]] == [10, 20]
    assert [call[0] for call in visual.calls] == [0, 1]
    assert [call[0] for call in asr.calls] == [0, 1]
    assert pipeline._multimodal.available_modalities == ("asr", "visual")


def test_requested_multimodal_fails_closed_without_retrievers():
    with pytest.raises(RuntimeError, match="requires injected .*retrievers"):
        TrakePipeline(mode="multimodal")


def test_multimodal_fails_closed_without_required_asr_retriever():
    with pytest.raises(RuntimeError, match="requires visual and ASR retrievers.*asr"):
        TrakePipeline(
            mode="multimodal",
            retrievers={"visual": FakeEventRetriever("visual", _rows())},
        )


def test_multimodal_path_respects_video_restriction_and_top_100():
    visual = FakeEventRetriever("visual", _rows("video-1"))
    asr = FakeEventRetriever("asr", _rows("video-1"))
    pipeline = TrakePipeline(
        mode="multimodal",
        retrievers={"visual": visual, "asr": asr},
    )

    result = pipeline.align(
        ["first event", "second event"],
        video_id="video-1",
        top_k_videos=100,
    )
    assert result[0]["frame_ids"] == [10, 20]
    assert all(call[1] == ("video-1",) for call in visual.calls + asr.calls)

    with pytest.raises(ValueError, match="between 1 and 100"):
        pipeline.align(["first event", "second event"], top_k_videos=101)
