import pandas as pd
import numpy as np

from src.pipelines.trake_visual import build_event_candidate_lattice


def test_event_lattice_keeps_top_k_per_event():
    metadata = pd.DataFrame({
        "kf_n": [1, 2, 3], "frame_idx": [10, 20, 30], "pts_time": [1.0, 2.0, 3.0]
    })
    lattice = build_event_candidate_lattice(
        metadata, np.asarray([[.1, .9, .8], [.7, .2, .6]], dtype=np.float32), top_k=2
    )
    assert [row["kf_n"] for row in lattice[0]] == [2, 3]
    assert [row["kf_n"] for row in lattice[1]] == [1, 3]
