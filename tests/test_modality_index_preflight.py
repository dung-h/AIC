"""Unit tests for the fail-closed ASR/OCR index gate.

The tests inject parquet/NumPy readers so they stay offline and do not need a
parquet engine or the multi-gigabyte production indexes.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.reranking.modality_index_preflight import (
    BENCHMARK_K_PACKS,
    ModalityIndexPreflightError,
    require_modality_index,
    run_modality_index_preflight,
)
from src.reranking.modality_index_preflight import _text_quality


def _fixture(root: Path, *, bad_rows: bool = False, bad_text: bool = False):
    expected = ("k01",)
    canonical = pd.DataFrame(
        {
            "video_id": ["K01_V001", "K01_V001"],
            "kf_n": [1, 2],
            "frame_idx": [10, 20],
            "pts_time": [1.0, 2.0],
        }
    )
    asr = pd.DataFrame(
        {
            "video_id": ["K01_V001", "K01_V001"],
            "kf_n": [1, 2],
            "frame_idx": [10, 20],
            "start": [0.5, 1.5],
            "end": [1.5, 2.5],
            "chunk": ["một câu nói", "câu thứ hai"],
        }
    )
    ocr = pd.DataFrame(
        {
            "video_id": ["K01_V001", "K01_V001"],
            "kf_n": [1, 2],
            "pts_time": [1.0, 2.0],
            "ocr_text": ["TITLE", "PRICE"],
        }
    )
    if bad_rows:
        asr = asr.iloc[:1].copy()
    if bad_text:
        asr.loc[0, "chunk"] = "bad\ufffd text Ã"

    files = [
        "global_keyframes.parquet",
        "asr_chunks_k01_ts.parquet",
        "emb_cache_asr_k01_chunks.npy",
        "ocr_k01.parquet",
        "emb_cache_ocr_k01.npy",
    ]
    for name in files:
        (root / name).touch()
    frames = {
        "global_keyframes.parquet": canonical,
        "asr_chunks_k01_ts.parquet": asr,
        "ocr_k01.parquet": ocr,
    }
    arrays = {
        "emb_cache_asr_k01_chunks.npy": np.zeros((len(asr), 1024), dtype=np.float32),
        "emb_cache_ocr_k01.npy": np.zeros((len(ocr), 1024), dtype=np.float32),
    }
    return expected, frames, arrays


class ModalityIndexPreflightTests(unittest.TestCase):
    def test_text_quality_does_not_flag_valid_vietnamese_as_mojibake(self):
        quality = _text_quality([
            "Khí hậu và chương trình hôm nay",
            "NÃO CHUẨN ĂN LÀNH",
        ])
        self.assertEqual(quality["marker_rows"], 1)
        self.assertEqual(quality["mojibake_token_count"], 0)

    def test_text_quality_flags_only_provable_round_trip_corruption(self):
        quality = _text_quality(["Nhiá»‡t Ä‘á»™ Nha Trang"])
        self.assertEqual(quality["mojibake_token_count"], 1)

    def _run(self, root: Path, *, bad_rows=False, bad_text=False, expected=("k01",)):
        _, frames, arrays = _fixture(root, bad_rows=bad_rows, bad_text=bad_text)
        return run_modality_index_preflight(
            root,
            expected_packs=expected,
            read_parquet=lambda path: frames[path.name],
            load_npy=lambda path: arrays[path.name],
        )

    def test_complete_pack_passes_schema_and_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._run(Path(directory))
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["coverage"]["asr_observed"], ["k01"])
        self.assertEqual(report["coverage"]["ocr_observed"], ["k01"])
        self.assertEqual(report["canonical"]["videos"], 1)
        self.assertEqual(report["asr"][0]["canonical_missing_rows"], 0)
        self.assertEqual(report["ocr"][0]["mapping_mismatch_rows"], 0)

    def test_missing_pack_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root)
            report = run_modality_index_preflight(
                root,
                expected_packs=("k01", "l21"),
                read_parquet=lambda path: {"global_keyframes.parquet": _fixture(root)[1]["global_keyframes.parquet"],
                                            "asr_chunks_k01_ts.parquet": _fixture(root)[1]["asr_chunks_k01_ts.parquet"],
                                            "ocr_k01.parquet": _fixture(root)[1]["ocr_k01.parquet"]}[path.name],
                load_npy=lambda path: np.zeros((2, 1024), dtype=np.float32),
            )
        self.assertFalse(report["passed"])
        self.assertIn("l21", report["coverage"]["asr_missing"])
        self.assertIn("l21", report["coverage"]["ocr_missing"])
        with self.assertRaises(ModalityIndexPreflightError):
            require_modality_index(report)

    def test_embedding_rows_and_text_replacement_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, frames, arrays = _fixture(root, bad_rows=True, bad_text=True)
            arrays["emb_cache_asr_k01_chunks.npy"] = np.zeros((2, 1024), dtype=np.float32)
            report = run_modality_index_preflight(
                root,
                expected_packs=("k01",),
                read_parquet=lambda path: frames[path.name],
                load_npy=lambda path: arrays[path.name],
            )
        self.assertFalse(report["passed"])
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("row_count_mismatch", codes)
        self.assertIn("text_replacement_char", codes)

    def test_canonical_frame_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, frames, arrays = _fixture(root)
            frames["ocr_k01.parquet"] = frames["ocr_k01.parquet"].copy()
            frames["ocr_k01.parquet"].loc[0, "pts_time"] = 99.0
            report = run_modality_index_preflight(
                root,
                expected_packs=("k01",),
                read_parquet=lambda path: frames[path.name],
                load_npy=lambda path: arrays[path.name],
            )
        self.assertFalse(report["passed"])
        self.assertIn("canonical_mapping_mismatch", {item["code"] for item in report["errors"]})

    def test_benchmark_scope_checks_only_active_k_packs(self):
        """A K-only benchmark can pass while the default full corpus fails."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, frames, arrays = _fixture(root)
            source_frames = {
                "asr": frames["asr_chunks_k01_ts.parquet"],
                "ocr": frames["ocr_k01.parquet"],
            }
            source_arrays = {
                "asr": arrays["emb_cache_asr_k01_chunks.npy"],
                "ocr": arrays["emb_cache_ocr_k01.npy"],
            }
            for pack in BENCHMARK_K_PACKS[1:]:
                asr_name = f"asr_chunks_{pack}_ts.parquet"
                asr_emb_name = f"emb_cache_asr_{pack}_chunks.npy"
                ocr_name = f"ocr_{pack}.parquet"
                ocr_emb_name = f"emb_cache_ocr_{pack}.npy"
                for name in (asr_name, asr_emb_name, ocr_name, ocr_emb_name):
                    (root / name).touch()
                frames[asr_name] = source_frames["asr"]
                frames[ocr_name] = source_frames["ocr"]
                arrays[asr_emb_name] = source_arrays["asr"]
                arrays[ocr_emb_name] = source_arrays["ocr"]

            def read(path):
                return frames[path.name]

            def load(path):
                return arrays[path.name]

            report = run_modality_index_preflight(
                root,
                expected_packs=tuple(pack.upper() for pack in BENCHMARK_K_PACKS),
                read_parquet=read,
                load_npy=load,
            )
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["scope"]["name"], "custom")
            self.assertFalse(report["scope"]["is_full_corpus"])
            self.assertEqual(report["coverage"]["asr_missing"], [])
            self.assertEqual(report["coverage"]["ocr_missing"], [])

            full_report = run_modality_index_preflight(
                root,
                read_parquet=read,
                load_npy=load,
            )
            self.assertFalse(full_report["passed"])
            self.assertIn("l21", full_report["coverage"]["asr_missing"])

    def test_active_modality_scope_does_not_require_other_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, frames, arrays = _fixture(root)
            for name in ("asr_chunks_k01_ts.parquet", "emb_cache_asr_k01_chunks.npy"):
                (root / name).unlink()

            report = run_modality_index_preflight(
                root,
                expected_packs=("k01",),
                active_modalities=("ocr",),
                read_parquet=lambda path: frames[path.name],
                load_npy=lambda path: arrays[path.name],
            )
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["coverage"]["asr_missing"], [])
        self.assertEqual(report["coverage"]["ocr_observed"], ["k01"])

    def test_pack_files_do_not_hide_partial_video_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, frames, arrays = _fixture(root)
            frames["global_keyframes.parquet"] = pd.concat([
                frames["global_keyframes.parquet"],
                pd.DataFrame({
                    "video_id": ["K01_V002"],
                    "kf_n": [1],
                    "frame_idx": [30],
                    "pts_time": [3.0],
                }),
            ], ignore_index=True)
            report = run_modality_index_preflight(
                root,
                expected_packs=("k01",),
                read_parquet=lambda path: frames[path.name],
                load_npy=lambda path: arrays[path.name],
            )

        self.assertFalse(report["passed"])
        codes = {item["code"] for item in report["errors"]}
        self.assertIn("asr_video_coverage_incomplete", codes)
        self.assertIn("ocr_video_coverage_incomplete", codes)
        self.assertEqual(report["coverage"]["canonical_expected_video_count"], 2)
        self.assertEqual(report["coverage"]["asr_observed_video_count"], 1)
        self.assertEqual(report["coverage"]["asr_video_coverage_ratio"], 0.5)
        self.assertEqual(report["coverage"]["ocr_video_coverage_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
