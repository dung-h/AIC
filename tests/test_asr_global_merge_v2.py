from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.asr_global_merge_v2 import ASRGlobalMergeError, MergeConfig, merge_global


PACKS = tuple([f"K{i:02d}" for i in range(1, 21)] + [f"L{i:02d}" for i in range(21, 31)])


def _canonical(path: Path) -> None:
    rows = []
    for pack in PACKS:
        video_id = f"{pack}_V001"
        rows.extend(
            [
                {"video_id": video_id, "pack": pack, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0},
                {"video_id": video_id, "pack": pack, "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
            ]
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_merge_normalizes_legacy_k_and_new_l_contracts(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    _canonical(canonical_path)
    legacy = tmp_path / "legacy"
    l_dir = tmp_path / "l"
    legacy.mkdir()
    l_dir.mkdir()
    for pack in PACKS:
        if pack.startswith("K"):
            frame = pd.DataFrame(
                [{"chunk": "K text", "vid": f"{pack}_V001", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10}]
            )
            frame.to_parquet(legacy / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
            np.save(legacy / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))
        else:
            frame = pd.DataFrame(
                [{"video_id": f"{pack}_V001", "chunk_index": 0, "text": "L text", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "distance_seconds": 0.5}]
            )
            frame.to_parquet(l_dir / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
            np.save(l_dir / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))

    report = merge_global(MergeConfig(canonical_path, legacy, l_dir, tmp_path / "out"))
    assert report["status"] == "ready"
    assert report["scope"]["video_count"] == 30
    metadata = pd.read_parquet(tmp_path / "out" / "retrieval.parquet")
    matrix = np.load(tmp_path / "out" / "embeddings.npy")
    assert len(metadata) == 30
    assert matrix.shape == (30, 1024)
    assert set(metadata["source_provenance"]) == {"validated_legacy_k", "deepgram_local_bge_l"}
    assert metadata["frame_idx"].tolist() == [10] * 30


def test_merge_fails_closed_when_one_pack_is_missing(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    _canonical(canonical_path)
    with pytest.raises(ASRGlobalMergeError, match="missing"):
        merge_global(MergeConfig(canonical_path, tmp_path / "legacy", tmp_path / "l", tmp_path / "out"))


def test_merge_fails_closed_if_normalization_would_stale_embeddings(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    _canonical(canonical_path)
    legacy = tmp_path / "legacy"
    l_dir = tmp_path / "l"
    legacy.mkdir()
    l_dir.mkdir()
    for pack in PACKS:
        if pack.startswith("K"):
            text = "Nhiá»‡t Ä‘á»™" if pack == "K01" else "K text"
            frame = pd.DataFrame(
                [{"chunk": text, "vid": f"{pack}_V001", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10}]
            )
            frame.to_parquet(legacy / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
            np.save(legacy / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))
        else:
            frame = pd.DataFrame(
                [{"video_id": f"{pack}_V001", "chunk_index": 0, "text": "L text", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "distance_seconds": 0.5}]
            )
            frame.to_parquet(l_dir / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
            np.save(l_dir / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))

    with pytest.raises(ASRGlobalMergeError, match="embeddings would be stale"):
        merge_global(MergeConfig(canonical_path, legacy, l_dir, tmp_path / "out"))


def test_merge_time_maps_legacy_chunks_when_mapping_columns_are_missing(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    _canonical(canonical_path)
    legacy = tmp_path / "legacy"
    l_dir = tmp_path / "l"
    legacy.mkdir()
    l_dir.mkdir()
    for pack in PACKS:
        if pack.startswith("K"):
            frame = pd.DataFrame(
                [{"chunk": "K text", "vid": f"{pack}_V001", "start": 0.25, "end": 0.75, "kf_n": None, "frame_idx": None}]
            )
            frame.to_parquet(legacy / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
            np.save(legacy / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))
        else:
            frame = pd.DataFrame(
                [{"video_id": f"{pack}_V001", "chunk_index": 0, "text": "L text", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "distance_seconds": 0.5}]
            )
            frame.to_parquet(l_dir / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
            np.save(l_dir / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))

    report = merge_global(MergeConfig(canonical_path, legacy, l_dir, tmp_path / "out"))
    assert report["status"] == "ready"
    assert report["packs"]["K01"]["provenance"] == ["validated_legacy_k_time_mapped"]
    metadata = pd.read_parquet(tmp_path / "out" / "retrieval.parquet")
    assert metadata.loc[metadata["source_pack"] == "K01", "kf_n"].tolist() == [1]


def test_merge_accepts_uniform_k_materializer_when_legacy_k_is_absent(tmp_path: Path) -> None:
    canonical_path = tmp_path / "canonical.parquet"
    _canonical(canonical_path)
    legacy = tmp_path / "legacy"
    materialized = tmp_path / "materialized"
    legacy.mkdir()
    materialized.mkdir()
    for pack in PACKS:
        target = materialized if pack == "K01" or pack.startswith("L") else legacy
        if pack == "K01" or pack.startswith("L"):
            frame = pd.DataFrame(
                [{"video_id": f"{pack}_V001", "chunk_index": 0, "text": f"{pack} text", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "distance_seconds": 0.5}]
            )
        else:
            frame = pd.DataFrame(
                [{"chunk": "K text", "vid": f"{pack}_V001", "start": 0.0, "end": 1.0, "kf_n": 1, "frame_idx": 10}]
            )
        frame.to_parquet(target / f"asr_chunks_{pack.lower()}_ts.parquet", index=False)
        np.save(target / f"emb_cache_asr_{pack.lower()}_chunks.npy", np.ones((1, 1024), dtype=np.float32))

    report = merge_global(MergeConfig(canonical_path, legacy, materialized, tmp_path / "out"))

    assert report["packs"]["K01"]["source_kind"] == "materialized"
    metadata = pd.read_parquet(tmp_path / "out" / "retrieval.parquet")
    assert metadata.loc[metadata["source_pack"] == "K01", "source_provenance"].tolist() == ["deepgram_local_bge_k"]
