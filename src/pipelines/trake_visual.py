"""Visual TRAKE alignment with injectable embeddings for offline evaluation.

``VisualTrakePipeline`` owns the production model/index lifecycle and uses
this module for the same-video DANTE stage.  Injectable vectors keep unit
tests and controlled offline experiments independent of a model/API.
"""
from dataclasses import dataclass
import time
import numpy as np

from ..utils.dante import dante_align, normalize_event_scores, sequence_quality
from ..trake.contracts import normalize_events, validate_sequence_path


LEGACY_ALIGNMENT_POLICY = "legacy"
MULTI_VIDEO_ALIGNMENT_POLICY = "multi_video_v1"


def _normalize_video_scores(scores):
    """Min/max-normalize finite video scores without depending on input order."""
    if not scores:
        return {}
    values = {str(video_id): float(score) for video_id, score in scores.items()}
    if not all(np.isfinite(score) for score in values.values()):
        raise ValueError("video scores must be finite")
    low, high = min(values.values()), max(values.values())
    if high <= low:
        return {video_id: 1.0 for video_id in values}
    return {
        video_id: float((score - low) / (high - low))
        for video_id, score in values.items()
    }


def build_event_candidate_lattice(metadata, score_matrix, *, top_k=5,
                                  temporal_neighbor_radius=0):
    """Retain multiple local frame hypotheses for each event before DANTE.

    ``score_matrix`` is shaped ``(n_events, n_frames)`` and metadata must be
    ordered by the same frame axis. Candidates are ranked within each event;
    no scores are compared across videos or events. When
    ``temporal_neighbor_radius`` is positive, the frames immediately around
    each top-k seed are added to the event's local lattice. The default radius
    is zero so the historical helper behavior is unchanged.
    """
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2 or len(metadata) != scores.shape[1]:
        raise ValueError("score_matrix must be (events, frames) and match metadata")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if temporal_neighbor_radius < 0:
        raise ValueError("temporal_neighbor_radius must be non-negative")
    lattice = []
    for event_scores in scores:
        finite_indices = np.flatnonzero(np.isfinite(event_scores))
        ranked = finite_indices[
            np.argsort(-event_scores[finite_indices], kind="mergesort")
        ]
        seed_indices = [int(index) for index in ranked[:top_k]]
        candidate_indices = set(seed_indices)
        for seed in seed_indices:
            start = max(0, seed - int(temporal_neighbor_radius))
            end = min(len(metadata), seed + int(temporal_neighbor_radius) + 1)
            candidate_indices.update(range(start, end))
        # Keep score order deterministic. The original top-k order is kept
        # when radius=0; neighbors are interleaved by their own score only
        # after the seeds have been selected.
        order = sorted(candidate_indices,
                       key=lambda index: (-float(event_scores[index]), int(index)))
        seed_rank = {index: rank for rank, index in enumerate(seed_indices)}
        rows = []
        for rank, idx in enumerate(order):
            row = metadata.iloc[int(idx)]
            rows.append({
                "position": int(idx),
                "kf_n": int(row.kf_n),
                "frame_idx": int(row.frame_idx),
                "pts_time": float(row.pts_time),
                "score": float(event_scores[idx]),
                "candidate_rank": int(rank),
                "source": "seed" if idx in seed_rank else "temporal_neighbor",
                "seed_rank": int(seed_rank[idx]) if idx in seed_rank else None,
            })
        lattice.append(rows)
    return lattice


@dataclass
class _Group:
    rows: object
    indices: np.ndarray


class VisualTrakeDante:
    def __init__(self, metadata, features, embed_provider=None, mode="visual",
                 lattice_enabled=False, alignment_policy=LEGACY_ALIGNMENT_POLICY):
        if mode not in {"visual", "asr", "hybrid"}:
            raise ValueError("mode must be visual, asr, or hybrid")
        self.mode = mode
        self.embed_provider = embed_provider
        self.lattice_enabled = bool(lattice_enabled)
        alignment_policy = str(alignment_policy).strip().lower()
        if alignment_policy not in {
            LEGACY_ALIGNMENT_POLICY,
            MULTI_VIDEO_ALIGNMENT_POLICY,
        }:
            raise ValueError(
                "alignment_policy must be 'legacy' or 'multi_video_v1'"
            )
        self.alignment_policy = alignment_policy
        self.groups = {}
        metadata = metadata.reset_index(drop=True)
        self.features = features
        if len(metadata) != len(features):
            raise ValueError("metadata and features must have equal length")
        if mode != "visual":
            raise ValueError(f"{mode} signal is unavailable: provide a visual-only experiment explicitly")
        for vid, g in metadata.groupby("video_id", sort=False):
            order = np.argsort(g["pts_time"].to_numpy())
            idx = g.index.to_numpy()[order]
            self.groups[vid] = _Group(g.iloc[order].reset_index(drop=True), idx)

    def _features(self, group):
        return np.asarray(self.features[group.indices], dtype=np.float32)

    def _events(self, events):
        return [{"desc": e} if isinstance(e, str) else e for e in events]

    def _rank_candidate_videos(self, vectors, candidates, event_count, limit):
        """Rank a same-corpus video shortlist before expensive alignment.

        The score is a retrieval-stage upper bound: for every event it takes
        the strongest keyframe in a video, then sums across events.  It is
        deliberately separate from DANTE: candidate generation can favour a
        video with individually strong but temporally incompatible events,
        while DANTE later supplies the sequence evidence needed to rank a
        complete answer.  Ties use ``video_id`` so provenance is stable.
        """
        if limit < 1:
            raise ValueError("candidate_video_limit must be positive")
        scored = []
        for raw_video_id in dict.fromkeys(str(value) for value in candidates):
            group = self.groups.get(raw_video_id)
            if group is None or len(group.indices) < event_count:
                continue
            similarities = self._features(group) @ vectors.T
            relevance = float(np.max(similarities, axis=0).sum())
            if np.isfinite(relevance):
                scored.append((raw_video_id, relevance))
        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = scored[:limit]
        normalized = _normalize_video_scores(dict(selected))
        return [
            {
                "video_id": video_id,
                "candidate_video_rank": rank,
                "video_relevance_raw": float(score),
                "video_relevance_normalized": float(normalized[video_id]),
                "candidate_source": "global_visual_event_upper_bound",
            }
            for rank, (video_id, score) in enumerate(selected, start=1)
        ]

    def align(self, events, video_id=None, lam=0.001, top_k_videos=10,
              candidate_videos=None, adaptive_lambda=None, use_lattice=None,
              lattice_top_k=10, temporal_neighbor_radius=1,
              sequence_coverage_weight=0.25,
              sequence_coherence_weight=0.10,
              alignment_policy=None, candidate_video_limit=None,
              video_relevance_weight=0.50,
              alignment_evidence_weight=0.50):
        events = normalize_events(events)
        if self.embed_provider is None:
            raise ValueError("visual mode requires an injected embed_provider")
        policy = self.alignment_policy if alignment_policy is None else str(alignment_policy).strip().lower()
        if policy not in {LEGACY_ALIGNMENT_POLICY, MULTI_VIDEO_ALIGNMENT_POLICY}:
            raise ValueError("alignment_policy must be 'legacy' or 'multi_video_v1'")
        for name, value in {
            "video_relevance_weight": video_relevance_weight,
            "alignment_evidence_weight": alignment_evidence_weight,
        }.items():
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        weight_total = float(video_relevance_weight) + float(alignment_evidence_weight)
        if weight_total <= 0:
            raise ValueError("at least one multi-video ranking weight must be positive")
        t0 = time.perf_counter()
        vec = np.asarray(self.embed_provider([event.description for event in events]), dtype=np.float32)
        if vec.shape[0] != len(events): raise ValueError("provider returned wrong event count")
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
        if video_id is not None:
            candidates = [video_id]
        elif candidate_videos is not None:
            candidates = list(candidate_videos)
        else:
            candidates = list(self.groups)
        candidate_provenance = {}
        if policy == MULTI_VIDEO_ALIGNMENT_POLICY:
            if candidate_video_limit is None:
                # Preserve the established legacy candidate budget.  The old
                # path aligned 3x the requested answer count before it kept
                # the final top-k; using only 20 here regressed R@20 on the
                # fixed holdout by excluding a video that legacy retained.
                candidate_video_limit = max(int(top_k_videos) * 3, 20)
            if (
                not isinstance(candidate_video_limit, int)
                or isinstance(candidate_video_limit, bool)
                or candidate_video_limit < 1
            ):
                raise ValueError("candidate_video_limit must be a positive integer")
            ranked_candidates = self._rank_candidate_videos(
                vec,
                candidates,
                len(events),
                int(candidate_video_limit),
            )
            candidates = [item["video_id"] for item in ranked_candidates]
            candidate_provenance = {
                item["video_id"]: item for item in ranked_candidates
            }
        elif candidate_videos is None and video_id is None:
            # Cheap upper bound for candidate generation; alignment is run only on these.
            bounds = [(float(np.max(self._features(self.groups[v]) @ vec.T, axis=0).sum()), v)
                      for v in candidates if len(self.groups[v].indices) >= len(events)]
            candidates = [v for _, v in sorted(bounds, reverse=True)[:max(top_k_videos, 1) * 3]]
        lattice_enabled = self.lattice_enabled if use_lattice is None else bool(use_lattice)
        if lattice_top_k < 1:
            raise ValueError("lattice_top_k must be positive")
        if temporal_neighbor_radius < 0:
            raise ValueError("temporal_neighbor_radius must be non-negative")
        scored, align_ms = [], 0.0
        for vid in candidates:
            group = self.groups.get(vid)
            if group is None or len(group.indices) < len(events): continue
            S = (self._features(group) @ vec.T).T
            chosen_lam = adaptive_lambda(S) if adaptive_lambda else lam
            candidate_lattice = None
            normalized_S = None
            lattice_mask = None
            scoring_mode = "legacy"
            raw_dante_score = None
            lattice_raw_score = None
            a0 = time.perf_counter()
            if lattice_enabled:
                candidate_lattice = build_event_candidate_lattice(
                    group.rows,
                    S,
                    top_k=lattice_top_k,
                    temporal_neighbor_radius=temporal_neighbor_radius,
                )
                lattice_mask = np.zeros(S.shape, dtype=bool)
                for event_index, event_candidates in enumerate(candidate_lattice):
                    for candidate in event_candidates:
                        lattice_mask[event_index, candidate["position"]] = True
                normalized_S = normalize_event_scores(S, valid_mask=lattice_mask)
                _, path = dante_align(
                    normalized_S,
                    lam=chosen_lam,
                    valid_mask=lattice_mask,
                )
                scoring_mode = "lattice"
                if path is None:
                    # The lattice is a recall optimization, not a reason to
                    # drop a video. If its strict ordered candidates are
                    # infeasible, fall back to the proven full-frame DANTE
                    # path and make the fallback observable.
                    raw_dante_score, path = dante_align(S, lam=chosen_lam)
                    score = raw_dante_score
                    normalized_S = normalize_event_scores(S)
                    scoring_mode = "legacy_fallback"
                else:
                    # Candidate-lattice scores are per-event normalized and
                    # therefore are not calibrated across videos. Use the
                    # proven full-frame DANTE score for video ranking while
                    # retaining the lattice path for frame localization.
                    # This preserves video-retrieval non-regression and makes
                    # the new policy a safe second-stage path selector.
                    lattice_raw_score, _ = dante_align(
                        S,
                        lam=chosen_lam,
                        valid_mask=lattice_mask,
                    )
            else:
                score, path = dante_align(S, lam=chosen_lam)
            if lattice_enabled and raw_dante_score is None:
                raw_dante_score, _ = dante_align(S, lam=chosen_lam)
            if lattice_enabled and scoring_mode == "lattice":
                score = raw_dante_score
                scoring_mode = "lattice_path_baseline_rank"
            align_ms += (time.perf_counter() - a0) * 1000
            if path is None: continue
            rows = group.rows.iloc[path]
            frame_ids = [int(r.frame_idx) for r in rows.itertuples()]
            pts_times = [float(r.pts_time) for r in rows.itertuples()]
            if any(a >= b for a, b in zip(frame_ids, frame_ids[1:])):
                continue
            if any(a >= b for a, b in zip(pts_times, pts_times[1:])):
                continue
            per_event = [float(S[i, path[i]]) for i in range(len(path))]
            quality = None
            if lattice_enabled:
                if normalized_S is None:
                    normalized_S = normalize_event_scores(S)
                quality = sequence_quality(
                    normalized_S,
                    path,
                    pts_times,
                    coverage_weight=sequence_coverage_weight,
                    coherence_weight=sequence_coherence_weight,
                )
                if scoring_mode == "lattice":
                    score = quality["score"]
            sequence = [{"event_index": events[i].index,
                         "event_desc": events[i].description,
                         "video_id": str(vid),
                         "modality": "visual",
                         "kf_n": int(r.kf_n),
                         "frame_idx": int(r.frame_idx),
                         "pts_time": float(r.pts_time),
                         "score": float(S[i, path[i]])}
                        for i, r in enumerate(rows.itertuples())]
            validate_sequence_path(sequence, events, video_id=str(vid))
            result = {"video_id": vid, "score": float(score), "lambda": float(chosen_lam),
                      "per_event_scores": per_event,
                      "path": sequence}
            if lattice_enabled:
                result.update({
                    "scoring_mode": scoring_mode,
                    "raw_dante_score": float(raw_dante_score),
                    "lattice_raw_score": (
                        None if lattice_raw_score is None else float(lattice_raw_score)
                    ),
                    "candidate_lattice": candidate_lattice,
                    "normalized_sequence_score": float(quality["score"]),
                    "coverage": float(quality["coverage"]),
                    "coherence": float(quality["coherence"]),
                })
            scored.append(result)
        if policy == MULTI_VIDEO_ALIGNMENT_POLICY:
            alignment_normalized = _normalize_video_scores(
                {str(item["video_id"]): float(item["score"]) for item in scored}
            )
            relevance_weight = float(video_relevance_weight) / weight_total
            evidence_weight = float(alignment_evidence_weight) / weight_total
            for item in scored:
                video_id = str(item["video_id"])
                retrieval = candidate_provenance[video_id]
                raw_alignment = float(item["score"])
                normalized_alignment = float(alignment_normalized[video_id])
                final_score = (
                    relevance_weight * float(retrieval["video_relevance_normalized"])
                    + evidence_weight * normalized_alignment
                )
                item["score"] = float(final_score)
                item["frame_ids"] = [int(step["frame_idx"]) for step in item["path"]]
                item["provenance"] = {
                    "alignment_policy": MULTI_VIDEO_ALIGNMENT_POLICY,
                    "candidate_source": retrieval["candidate_source"],
                    "candidate_video_rank": int(retrieval["candidate_video_rank"]),
                    "video_relevance_raw": float(retrieval["video_relevance_raw"]),
                    "video_relevance_normalized": float(retrieval["video_relevance_normalized"]),
                    "alignment_score_raw": raw_alignment,
                    "alignment_score_normalized": normalized_alignment,
                    "ranking_weights": {
                        "video_relevance": relevance_weight,
                        "alignment_evidence": evidence_weight,
                    },
                }
        scored.sort(key=lambda x: (-x["score"], str(x["video_id"])))
        diagnostics = {
            "mode": self.mode, "candidate_count": len(candidates), "scored_count": len(scored),
            "lattice_enabled": lattice_enabled,
            "lattice_top_k": int(lattice_top_k) if lattice_enabled else None,
            "temporal_neighbor_radius": int(temporal_neighbor_radius) if lattice_enabled else None,
            "retrieval_ms": (time.perf_counter() - t0) * 1000 - align_ms,
            "alignment_ms": align_ms}
        if policy == MULTI_VIDEO_ALIGNMENT_POLICY:
            diagnostics.update({
                "alignment_policy": MULTI_VIDEO_ALIGNMENT_POLICY,
                "candidate_video_limit": int(candidate_video_limit),
                "candidate_videos": [candidate_provenance[video_id] for video_id in candidates],
                "ranking": "normalized_video_relevance_plus_alignment_evidence",
            })
        return {"results": scored[:top_k_videos], "diagnostics": diagnostics}


def adaptive_lambda(scores):
    """Conservative default: do not penalize sparse visual event spacing."""
    return 0.0 if scores.shape[1] > scores.shape[0] * 8 else 0.001
