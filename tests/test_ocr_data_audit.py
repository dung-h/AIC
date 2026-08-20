from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.audit_ocr_data import _keyframe_inventory, _ocr_inventory


def test_keyframe_inventory_detects_missing_frame(tmp_path: Path):
    root = tmp_path / "frames"
    (root / "K01_V001").mkdir(parents=True)
    (root / "K01_V001" / "001.jpg").write_bytes(b"frame")
    canonical = pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "pts_time": 0.0},
        {"video_id": "K01_V001", "kf_n": 2, "pts_time": 1.0},
    ])
    report = _keyframe_inventory(root, canonical)
    assert report["canonical_videos"] == 1
    assert report["sampled_frame_rows_missing"] == 1
    assert report["missing_canonical_frame_files"] == 0
    assert not report["all_canonical_frames_available"]


def test_existing_ocr_is_diagnostic_and_checks_embedding_alignment(tmp_path: Path):
    index = tmp_path / "index"
    index.mkdir()
    pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "pts_time": 0.0, "ocr_text": "Nha Trang"},
    ]).to_parquet(index / "ocr_k01.parquet", index=False)
    import numpy as np
    np.save(index / "emb_cache_ocr_k01.npy", np.zeros((1, 1024), dtype=np.float32))
    canonical = pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "pts_time": 0.0},
        {"video_id": "K01_V002", "kf_n": 1, "pts_time": 0.0},
    ])
    report = _ocr_inventory(index, canonical)
    artifact = report["artifacts"][0]
    assert artifact["status"] == "diagnostic_existing"
    assert artifact["videos"] == 1
    assert artifact["missing_canonical_videos"] == 1
    assert artifact["embedding"]["rows_match"]
