"""Experimental VKIS temporal alignment.

This module is deliberately independent from ``vkis_pipeline.VKISPipeline``.
It expects normalized query and index embeddings, making the temporal algorithm
testable without a model or an API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Match:
    video_id: str
    global_id: int
    kf_n: int
    frame_idx: int
    pts_time: float
    similarity: float


class VKISTemporalAligner:
    """Two-stage video retrieval followed by monotonic subsequence alignment."""

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                 candidate_count: int = 20, max_gap: int | None = None,
                 gap_penalty: float = 0.0):
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) != len(metadata):
            raise ValueError("embeddings and metadata must have the same row count")
        self.embeddings = x
        self.metadata = metadata.reset_index(drop=True).copy()
        required = {"video_id", "kf_n", "frame_idx", "pts_time"}
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"metadata missing columns: {sorted(missing)}")
        if "global_id" not in self.metadata:
            self.metadata["global_id"] = np.arange(len(self.metadata))
        self.candidate_count = candidate_count
        self.max_gap = max_gap
        self.gap_penalty = gap_penalty
        # Arrays are materialized once; candidate scoring never calls pandas.
        self._groups = {}
        for video, frame in self.metadata.groupby("video_id", sort=False):
            ids = frame.index.to_numpy(dtype=np.int64)
            order = np.argsort(self.metadata.loc[ids, "pts_time"].to_numpy())
            self._groups[str(video)] = ids[order]

    @staticmethod
    def _normalise(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 2 or q.shape[0] == 0:
            raise ValueError("query embeddings must be a non-empty 2-D array")
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        return q / np.maximum(norms, 1e-12)

    def _video_scores(self, q: np.ndarray, aggregate: str) -> dict[str, float]:
        sims = self.embeddings @ q.T
        scores = {}
        for video, ids in self._groups.items():
            values = sims[ids]
            if aggregate == "max":
                scores[video] = float(values.max())
            elif aggregate == "mean":
                scores[video] = float(values.mean())
            else:  # top3 query-frame evidence, robust to one uninformative frame
                per_query = values.max(axis=0)
                scores[video] = float(np.sort(per_query)[-min(3, len(per_query)):].mean())
        return scores

    def _align(self, q: np.ndarray, ids: np.ndarray) -> tuple[list[int], float]:
        sims = self.embeddings[ids] @ q.T  # video frames x query frames
        n, m = sims.shape
        dp = np.full((m, n), -np.inf, dtype=np.float32)
        prev = np.full((m, n), -1, dtype=np.int32)
        dp[0] = sims[:, 0]
        for qi in range(1, m):
            for vi in range(n):
                lo = 0 if self.max_gap is None else max(0, vi - self.max_gap)
                if vi == 0 or lo >= vi:
                    continue
                choices = dp[qi - 1, lo:vi] - self.gap_penalty * (vi - np.arange(lo, vi) - 1)
                pj = int(np.argmax(choices)) + lo
                dp[qi, vi] = sims[vi, qi] + choices[pj - lo]
                prev[qi, vi] = pj
        end = int(np.argmax(dp[-1]))
        if not np.isfinite(dp[-1, end]):
            return [], float("-inf")
        path = [end]
        for qi in range(m - 1, 0, -1):
            path.append(int(prev[qi, path[-1]]))
        path.reverse()
        return [int(ids[i]) for i in path], float(dp[-1, end] / m)

    def search(self, query_embeddings: np.ndarray, aggregate: str = "top3") -> list[dict]:
        q = self._normalise(query_embeddings)
        if aggregate not in {"max", "mean", "top3"}:
            raise ValueError("aggregate must be max, mean, or top3")
        stage1 = self._video_scores(q, aggregate)
        videos = sorted(stage1, key=stage1.get, reverse=True)[:self.candidate_count]
        results = []
        for video in videos:
            matched, score = self._align(q, self._groups[video])
            if matched:
                rows = self.metadata.iloc[matched]
                matches = [Match(str(r.video_id), int(r.global_id), int(r.kf_n),
                                 int(r.frame_idx), float(r.pts_time),
                                 float(self.embeddings[int(i)] @ q[j]))
                           for j, (i, r) in enumerate(zip(matched, rows.itertuples()))]
                results.append({"video_id": video, "score": score,
                                "stage1_score": stage1[video], "matches": matches,
                                "diagnostics": {"query_count": len(q), "candidate": True,
                                                 "monotonic": all(a.pts_time < b.pts_time for a, b in zip(matches, matches[1:]))}})
        return sorted(results, key=lambda r: r["score"], reverse=True)
