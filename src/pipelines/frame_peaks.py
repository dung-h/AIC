"""Temporal peak selection for interactive KIS frame inspection."""
from __future__ import annotations

import numpy as np


def select_temporal_peaks(rows, scores, *, count=5, min_gap_kf=3, min_gap_s=2.0):
    """Return diverse high-scoring rows from one video's ordered keyframes.

    ``rows`` must be ordered by ``kf_n`` or ``pts_time`` and expose the fields
    used below. Selection is greedy by score, followed by temporal diversity.
    The first peak is always the highest-scoring frame.
    """
    if count < 1 or len(rows) != len(scores):
        raise ValueError("count must be positive and rows/scores must align")
    if not len(rows):
        return []
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    chosen = []
    for index in order:
        row = rows.iloc[int(index)]
        if all(
            abs(int(row.kf_n) - int(prev.kf_n)) >= min_gap_kf
            and abs(float(row.pts_time) - float(prev.pts_time)) >= min_gap_s
            for _, prev in chosen
        ):
            chosen.append((int(index), row))
            if len(chosen) >= count:
                break
    return [
        {"frame_idx": int(row.frame_idx), "kf_n": int(row.kf_n),
         "pts_time": float(row.pts_time), "score": float(scores[index])}
        for index, row in chosen
    ]
