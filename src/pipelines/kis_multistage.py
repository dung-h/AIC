"""Experimental multi-stage KIS retrieval.

This module is deliberately independent of :class:`HCMAIPipeline`.  It accepts
precomputed query vectors, which makes experiments reproducible and permits a
caller to encode each query once.  Scores and ordering are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from indexing.canonical import FRAME_KEY, canonicalize_frame_map


@dataclass(frozen=True)
class Candidate:
    video_id: str
    frame_idx: int
    kf_n: int
    pts_time: float
    score: float
    stage: str
    encoder_scores: tuple[float, ...]

    def as_tuple(self):
        return (self.video_id, self.frame_idx, self.kf_n, self.pts_time, self.score,
                {"stage": self.stage, "encoder_scores": self.encoder_scores})


@dataclass(frozen=True)
class SearchTrace:
    """All stages of a vector search, before final one-per-video truncation."""
    stage1_union_videos: tuple[str, ...]
    pre_dedup_frame_candidates: tuple[Candidate, ...]
    final_results: tuple[Candidate, ...]


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map descending values to [0, 1], with stable ordinal tie breaking."""
    order = np.argsort(-values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float32)
    out[order] = 1.0 - np.arange(len(values), dtype=np.float32) / max(len(values) - 1, 1)
    return out


class KISMultiStage:
    def __init__(self, feature_maps, feature_matrices, *, video_top_n=100,
                 frame_top_m=20, encoder_weights=None):
        if not feature_matrices:
            raise ValueError("at least one encoder is required")
        self.maps = []
        self.features = []
        canonical_keys = None
        canonical_pts = None
        for name, matrix in feature_matrices.items():
            if name not in feature_maps:
                raise ValueError(f"missing map for encoder {name}")
            km, issues = canonicalize_frame_map(feature_maps[name], name)
            errors = [i.message for i in issues if i.severity == "error"]
            if errors:
                raise ValueError("; ".join(errors))
            if km.duplicated(list(FRAME_KEY)).any():
                raise ValueError(f"{name}: duplicate FRAME_KEY")
            keys = list(km.loc[:, list(FRAME_KEY)].itertuples(index=False, name=None))
            if canonical_keys is None:
                canonical_keys, canonical_pts = keys, km.pts_time.to_numpy(float)
            else:
                if set(keys) != set(canonical_keys):
                    raise ValueError(f"{name}: map key set differs from canonical map")
                if keys != canonical_keys:
                    raise ValueError(f"{name}: map row order differs; positional matrices are unsafe")
                if not np.allclose(km.pts_time.to_numpy(float), canonical_pts, atol=1e-3, rtol=0,
                                   equal_nan=True):
                    raise ValueError(f"{name}: pts_time differs from canonical map beyond 1ms")
            matrix = np.asarray(matrix)
            if len(matrix) != len(km):
                raise ValueError(f"{name}: feature/map length mismatch")
            self.maps.append(km.reset_index(drop=True))
            self.features.append(matrix)
        self.names = tuple(feature_matrices)
        self.video_top_n = int(video_top_n)
        self.frame_top_m = int(frame_top_m)
        self.weights = np.asarray(encoder_weights if encoder_weights is not None
                                 else [1.0] * len(self.features), dtype=np.float32)
        if len(self.weights) != len(self.features) or self.weights.sum() <= 0:
            raise ValueError("encoder_weights must match encoders and have positive sum")
        self.weights /= self.weights.sum()
        # Row groups avoid a pandas scan for every candidate.
        self.groups = []
        self.identity_rows = []
        self.identity_lookup = []
        for km in self.maps:
            groups = {}
            for i, video in enumerate(km.video_id.astype(str)):
                groups.setdefault(video, []).append(i)
            self.groups.append({v: np.asarray(ix, dtype=np.int64) for v, ix in groups.items()})
            identities = [tuple(x) for x in km.loc[:, list(FRAME_KEY)].itertuples(index=False, name=None)]
            self.identity_rows.append(identities)
            self.identity_lookup.append({key: i for i, key in enumerate(identities)})

    def search_vectors(self, query_vectors: dict[str, np.ndarray], *, topk=20):
        return list(self.search_trace(query_vectors, topk=topk).final_results)

    def search_trace(self, query_vectors: dict[str, np.ndarray], *, topk=20):
        scores = []
        for name, matrix in zip(self.names, self.features):
            if name not in query_vectors:
                scores.append(None)
                continue
            q = np.asarray(query_vectors[name], dtype=np.float32)
            scores.append(np.asarray(matrix @ q, dtype=np.float32))
        present = [i for i, s in enumerate(scores) if s is not None]
        if not present:
            raise ValueError("no query vector supplied for available encoder")

        # Stage 1: independent video max-pools, then union of video candidates.
        video_scores = []
        for i in present:
            vs = {v: float(np.max(scores[i][ix])) for v, ix in self.groups[i].items()}
            video_scores.append(vs)
        union = set()
        for vs in video_scores:
            union.update(v for v, _ in sorted(vs.items(), key=lambda x: (-x[1], x[0]))[:self.video_top_n])

        candidates = {}
        for enc_i in present:
            for video in union:
                ix = self.groups[enc_i].get(video)
                if ix is None:
                    continue
                local = ix[np.argsort(-scores[enc_i][ix], kind="mergesort")[:self.frame_top_m]]
                for row in local:
                    key = self.identity_rows[enc_i][row]
                    candidates.setdefault(key, {})[enc_i] = float(scores[enc_i][row])

        # Stage 2: global candidate-rank normalization (not per-video).
        normalized = {}
        for i in present:
            keys = list(candidates)
            vals = np.array([candidates[k].get(i, -np.inf) for k in keys], dtype=np.float32)
            finite = np.isfinite(vals)
            ranked = np.zeros(len(keys), dtype=np.float32)
            ranked[finite] = _rank_normalize(vals[finite])
            normalized[i] = dict(zip(keys, ranked))
        rows = []
        for key, raw in candidates.items():
            enc_norm = [float(normalized[i][key]) for i in present]
            present_weight = self.weights[present]
            present_weight = present_weight / present_weight.sum()
            score = float(sum(present_weight[p] * enc_norm[p] for p in range(len(present))))
            map_i = present[0]
            r = self.maps[map_i].iloc[self.identity_lookup[map_i][key]]
            rows.append(Candidate(str(key[0]), int(key[2]), int(key[1]), float(r.pts_time), score, "multistage", tuple(enc_norm)))
        rows.sort(key=lambda r: (-r.score, r.video_id, r.frame_idx, r.kf_n))
        pre_dedup = tuple(rows)
        seen = set(); out = []
        for row in rows:
            if row.video_id not in seen:
                out.append(row); seen.add(row.video_id)
            if len(out) >= topk: break
        return SearchTrace(tuple(sorted(union)), pre_dedup, tuple(out))
