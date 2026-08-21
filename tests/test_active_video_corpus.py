from __future__ import annotations

import numpy as np

from src.pipelines.kis_fusion_retriever import KISFusionRetriever, _normalise_active_video_prefixes
from src.pipelines.vqa_pipeline_v3 import _filter_to_active_videos


def test_normalise_active_video_prefixes_is_stable(monkeypatch):
    monkeypatch.setenv("HCMAI_ACTIVE_VIDEO_PREFIXES", " l ,K,L ")

    assert _normalise_active_video_prefixes(None) == ("L", "K")


def test_kis_maxvec_ignores_uninstalled_pack_rows():
    # No model construction: exercise the ranking boundary directly.  The K
    # frame has the largest raw score but must not consume an L-only top-k.
    retriever = object.__new__(KISFusionRetriever)
    retriever.all_vids = np.array(["L21_V001", "L22_V001"])
    retriever._active_row_idx = np.array([0, 2, 3], dtype=np.int64)
    retriever._active_vid_idx_arr = np.array([0, 1, 1], dtype=np.int32)

    scores = np.array([0.2, 0.99, 0.4, 0.3], dtype=np.float32)

    assert retriever._maxvec(scores).tolist() == [0.2, 0.4]


def test_specialist_rows_cannot_rescue_an_uninstalled_pack():
    rows = [
        {"video_id": "K01_V001", "kf_n": 3},
        {"video_id": "L21_V001", "kf_n": 4},
        ("L22_V001", 100, 5, 0.3),
    ]

    assert _filter_to_active_videos(rows, {"L21_V001", "L22_V001"}) == rows[1:]
