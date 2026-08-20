from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval.qna_modality_scope import (
    derive_active_modality_packs,
    pack_from_video_id,
    run_scoped_preflight,
)


class QNAModalityScopeEvaluatorTests(unittest.TestCase):
    def test_pack_derivation_and_modality_scope(self):
        rows = [
            {"video_id": "K01_V001", "required_modalities": ["asr"]},
            {"video_id": "L25_V007", "required_modalities": "ocr"},
            {"video_id": "K03_V002", "required_modalities": "visual"},
            {"video_id": "bad_video", "required_modalities": "asr"},
        ]
        scope = derive_active_modality_packs("benchmark.jsonl", rows=rows)
        self.assertEqual(pack_from_video_id("K01_V001"), "k01")
        self.assertEqual(scope["active_packs"], ["k01", "l25"])
        self.assertEqual(scope["by_modality"], {"asr": ["k01"], "ocr": ["l25"]})
        self.assertEqual(scope["unrecognised_video_ids"], ["bad_video"])

    def test_jsonl_split_filter_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"split": "dev", "video_id": "K01_V001", "required_modalities": "asr"}),
                    json.dumps({"split": "holdout", "video_id": "L25_V001", "required_modalities": "ocr"}),
                ]) + "\n",
                encoding="utf-8",
            )
            scope = derive_active_modality_packs(path, split="dev")
        self.assertEqual(scope["rows_considered"], 1)
        self.assertEqual(scope["active_packs"], ["k01"])
        self.assertEqual(scope["by_modality"]["ocr"], [])

    @patch("src.eval.qna_modality_scope.run_modality_index_preflight")
    def test_preflight_projection_ignores_inactive_modality_errors(self, preflight):
        preflight.return_value = {
            "errors": [
                {"code": "missing_embedding", "message": "missing ASR", "modality": "asr", "pack": "k01"},
                {"code": "missing_embedding", "message": "missing OCR", "modality": "ocr", "pack": "k01"},
                {"code": "canonical_read_error", "message": "canonical unavailable"},
            ],
            "warnings": [],
            "coverage": {"asr_observed": [], "ocr_observed": ["k01"]},
            "passed": False,
        }
        scope = {
            "active_packs": ["k01"],
            "by_modality": {"asr": ["k01"], "ocr": []},
            "unrecognised_video_ids": [],
        }
        report = run_scoped_preflight("index", scope)
        self.assertEqual({item["code"] for item in report["errors"]}, {"missing_embedding", "canonical_read_error"})
        self.assertEqual(report["coverage"]["asr_missing"], ["k01"])
        self.assertEqual(report["coverage"]["ocr_missing"], [])
        preflight.assert_called_once_with(
            "index",
            expected_packs=["k01"],
            active_modalities=["asr"],
            require_embeddings=True,
        )


if __name__ == "__main__":
    unittest.main()
