from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.reranking.modality_index_registry import (
    ModalityIndexRegistry,
    ModalityIndexRegistryError,
)
from src.reranking.modality_index_preflight import run_modality_index_preflight
from src.reranking.qna_modality_router import QNAModalityRouter


class _Embedder:
    def embed(self, texts, batch_size=1, normalize=True):
        vector = np.zeros((1, 1024), dtype=np.float32)
        vector[0, 0] = 1.0
        return vector


def _write_global_fixture(root: Path, *, with_no_speech: bool = False):
    canonical = pd.DataFrame(
        {
            "video_id": ["K01_V001", "K01_V001", "K01_V002"],
            "pack": ["K01", "K01", "K01"],
            "kf_n": [1, 2, 1],
            "frame_idx": [10, 20, 30],
            "pts_time": [1.0, 2.0, 3.0],
        }
    )
    canonical.to_parquet(root / "global_keyframes.parquet", index=False)
    directory = root / "modality_global_v2" / "asr_global_merged_v2"
    directory.mkdir(parents=True)
    metadata = pd.DataFrame(
        {
            "embedding_row": [0, 1],
            "video_id": ["K01_V001", "K01_V001"],
            "chunk_index": [0, 1],
            "text": ["Nhiệt độ Nha Trang 25 độ", "Trời có mưa"],
            "start": [0.5, 1.5],
            "end": [1.5, 2.5],
            "kf_n": [1, 2],
            "frame_idx": [10, 20],
            "pts_time": [1.0, 2.0],
            "distance_seconds": [0.0, 0.0],
            "source_pack": ["K01", "K01"],
            "source_provenance": ["fixture", "fixture"],
        }
    )
    metadata.to_parquet(directory / "retrieval.parquet", index=False)
    embeddings = np.zeros((2, 1024), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 0.9
    embeddings[1, 1] = 0.1
    np.save(directory / "embeddings.npy", embeddings)
    no_speech = ["K01_V002"] if with_no_speech else []
    manifest = {
        "status": "ready",
        "index_id": "fixture-asr",
        "scope": {"packs": ["K01"], "video_count": 2, "video_ids": ["K01_V001", "K01_V002"]},
        "embedding": {"dim": 1024, "shape": [2, 1024]},
        "artifacts": {"retrieval": "retrieval.parquet", "embeddings": "embeddings.npy"},
        "packs": {"K01": {"rows": 2, "no_speech_videos": no_speech}},
    }
    (directory / "asr_global_merge_v2_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class ModalityIndexRegistryTests(unittest.TestCase):
    def test_loads_global_asr_and_preserves_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_global_fixture(root, with_no_speech=True)
            registry = ModalityIndexRegistry(root)
            embeddings, metadata, info = registry.load_asr(
                expected_packs=("k01",)
            )

        self.assertEqual(embeddings.shape, (2, 1024))
        self.assertEqual(metadata["chunk"].tolist(), ["Nhiệt độ Nha Trang 25 độ", "Trời có mưa"])
        self.assertEqual(metadata["frame_idx"].tolist(), [10, 20])
        self.assertEqual(info["source"], "global_merged_v2")
        self.assertEqual(info["no_speech_videos"], ["K01_V002"])

    def test_rejects_global_frame_mapping_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_global_fixture(root)
            retrieval = root / "modality_global_v2" / "asr_global_merged_v2" / "retrieval.parquet"
            table = pd.read_parquet(retrieval)
            table.loc[0, "frame_idx"] = 999
            table.to_parquet(retrieval, index=False)
            with self.assertRaises(ModalityIndexRegistryError):
                ModalityIndexRegistry(root).load_asr(expected_packs=("k01",))

    def test_router_prefers_global_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_global_fixture(root, with_no_speech=True)
            router = QNAModalityRouter(
                index_dir=root,
                embedder=_Embedder(),
                strict=True,
                expected_packs=("k01",),
                active_modalities=("asr",),
            )
            rows = router.global_candidates("nhiệt độ", "asr", topk=1)

        self.assertEqual(rows[0]["video_id"], "K01_V001")
        self.assertEqual(rows[0]["frame_idx"], 10)
        self.assertEqual(rows[0]["text"], "Nhiệt độ Nha Trang 25 độ")
        self.assertEqual(router.preflight_report["sources"]["asr"], "global_merged_v2")

    def test_preflight_counts_no_speech_as_global_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_global_fixture(root, with_no_speech=True)
            report = run_modality_index_preflight(
                root,
                expected_packs=("k01",),
                active_modalities=("asr",),
                expected_dim=1024,
            )

        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["sources"]["asr"], "global_merged_v2")
        self.assertEqual(report["coverage"]["asr_missing"], [])
        self.assertEqual(report["coverage"]["asr_observed_video_count"], 2)
        self.assertEqual(report["asr"][0]["no_speech_videos"], ["K01_V002"])

    def test_strict_router_refuses_legacy_shards_without_global_manifest(self):
        """Production routing must not silently substitute per-pack indexes."""
        with tempfile.TemporaryDirectory() as directory:
            router = QNAModalityRouter.__new__(QNAModalityRouter)
            router.index_dir = Path(directory)
            router.registry = ModalityIndexRegistry(router.index_dir)
            router.strict = True
            router.text_mode = "dense"
            router._indexes = {}
            router.expected_packs = ("k01",)

            with self.assertRaisesRegex(RuntimeError, "global ASR manifest"):
                router._load("asr")
            with self.assertRaisesRegex(RuntimeError, "global OCR manifest"):
                router._load("ocr")


if __name__ == "__main__":
    unittest.main()
