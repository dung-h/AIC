import numpy as np
import pandas as pd
import pytest

from src.pipelines.vkis_temporal import VKISTemporalAligner


def make_index():
    x = np.eye(5, dtype=np.float32)
    rows = pd.DataFrame({"global_id": range(5), "video_id": ["v"] * 5,
                         "kf_n": range(5), "frame_idx": range(5),
                         "pts_time": [0., 1., 2., 3., 4.]})
    return x, rows


def test_alignment_preserves_order_and_mapping():
    x, rows = make_index()
    out = VKISTemporalAligner(x, rows, max_gap=3).search(x[[1, 3]])
    assert [m.kf_n for m in out[0]["matches"]] == [1, 3]
    assert out[0]["diagnostics"]["monotonic"]


def test_empty_and_short_inputs():
    x, rows = make_index()
    aligner = VKISTemporalAligner(x, rows)
    assert aligner.search(x[[2]])[0]["matches"][0].frame_idx == 2
    with pytest.raises(ValueError):
        aligner.search(np.empty((0, 5)))


def test_stable_video_lookup_and_candidate_limit():
    x, rows = make_index()
    rows = pd.concat([rows, rows.assign(global_id=range(5, 10), video_id="other")], ignore_index=True)
    out = VKISTemporalAligner(np.vstack([x, x]), rows, candidate_count=1).search(x[[0, 2]])
    assert len(out) == 1 and out[0]["video_id"] == "v"
