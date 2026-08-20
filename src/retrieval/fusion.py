"""Deterministic, model-agnostic score and rank fusion."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import FusedCandidate, ModalityHit, ModalityResult


def _ordered_hits(channel: ModalityResult) -> Tuple[ModalityHit, ...]:
    """Canonicalize a channel's order; input rank fields are advisory only."""

    indexed = list(enumerate(channel.hits))
    indexed.sort(
        key=lambda pair: (
            -pair[1].score if channel.higher_is_better else pair[1].score,
            pair[1].item_id,
            pair[0],
        )
    )
    return tuple(hit for _, hit in indexed)


def _rank_map(channel: ModalityResult) -> Dict[str, Tuple[int, ModalityHit]]:
    ranks: Dict[str, Tuple[int, ModalityHit]] = {}
    for rank, hit in enumerate(_ordered_hits(channel), start=1):
        # Duplicate rows in one channel must not award multiple ranks.
        ranks.setdefault(hit.item_id, (rank, hit))
    return ranks


def _video_rank_map(channel: ModalityResult) -> Dict[str, Tuple[int, ModalityHit]]:
    """Collapse one channel to its best-ranked representative per video."""
    ranks: Dict[str, Tuple[int, ModalityHit]] = {}
    for rank, hit in enumerate(_ordered_hits(channel), start=1):
        # The first hit is the channel's strongest evidence for that video.
        ranks.setdefault(str(hit.video_id), (rank, hit))
    return ranks


def normalize_scores(
    scores: Sequence[float],
    *,
    method: str = "minmax",
    higher_is_better: bool = True,
) -> Tuple[float, ...]:
    """Normalize scores into comparable ``[0, 1]`` values.

    ``rank`` is deliberately included because it is robust across unrelated
    encoder score scales.  Ties and constant-score channels are deterministic.
    """

    values = tuple(float(value) for value in scores)
    if not values:
        return ()
    if method not in {"minmax", "zscore", "rank"}:
        raise ValueError("method must be one of: minmax, zscore, rank")
    oriented = tuple(value if higher_is_better else -value for value in values)
    if method == "rank":
        order = sorted(range(len(values)), key=lambda i: (-oriented[i], i))
        out = [0.0] * len(values)
        denominator = max(len(values), 1)
        for rank, index in enumerate(order, start=1):
            out[index] = (denominator - rank + 1) / denominator
        return tuple(out)
    low, high = min(oriented), max(oriented)
    if high == low:
        return tuple(1.0 for _ in values)
    minmax = tuple((value - low) / (high - low) for value in oriented)
    if method == "minmax":
        return minmax
    mean = sum(oriented) / len(oriented)
    variance = sum((value - mean) ** 2 for value in oriented) / len(oriented)
    std = variance ** 0.5
    if std == 0:
        return tuple(1.0 for _ in values)
    # Squash z-scores to [0, 1] while retaining ordering.
    z = tuple((value - mean) / std for value in oriented)
    z_low, z_high = min(z), max(z)
    if z_high == z_low:
        return tuple(1.0 for _ in values)
    return tuple((value - z_low) / (z_high - z_low) for value in z)


def reciprocal_rank_fusion(
    channels: Sequence[ModalityResult],
    *,
    weights: Optional[Mapping[str, float]] = None,
    rrf_k: int = 60,
    top_k: Optional[int] = None,
) -> Tuple[FusedCandidate, ...]:
    """Fuse channel hits using weighted RRF, preserving all wrong-video hits.

    This is the frame/item-level primitive and remains backward compatible.
    Video-level callers must use :func:`video_level_rrf` so independent ASR,
    OCR, and visual frame IDs can reinforce the same video.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0")
    weights = dict(weights or {})
    aggregate: Dict[str, Dict[str, object]] = {}
    for channel in channels:
        weight = float(weights.get(channel.channel_key,
                                   weights.get(channel.modality, 1.0)))
        if weight < 0:
            raise ValueError("channel weights must be non-negative")
        rank_map = _rank_map(channel)
        ordered = list(rank_map.items())
        normalized = normalize_scores(
            [hit.score for _, (_, hit) in ordered],
            method="minmax",
            higher_is_better=channel.higher_is_better,
        )
        for position, (item_id, (rank, hit)) in enumerate(ordered):
            row = aggregate.setdefault(
                item_id,
                {
                    "hit": hit,
                    "rrf": 0.0,
                    "norm": 0.0,
                    "ranks": {},
                    "scores": {},
                    "modalities": set(),
                },
            )
            row["rrf"] = float(row["rrf"]) + weight / (rrf_k + rank)
            row["norm"] = float(row["norm"]) + weight * normalized[position]
            row["ranks"][channel.channel_key] = rank
            row["scores"][channel.channel_key] = hit.score
            row["modalities"].add(channel.channel_key)
    rows = []
    for item_id, row in aggregate.items():
        hit = row["hit"]
        rows.append(
            FusedCandidate(
                item_id=item_id,
                video_id=hit.video_id,
                score=float(row["rrf"]),
                kf_n=hit.kf_n,
                frame_idx=hit.frame_idx,
                pts_time=hit.pts_time,
                source_modalities=tuple(sorted(row["modalities"])),
                metadata=dict(hit.metadata),
                rrf_score=float(row["rrf"]),
                normalized_score=float(row["norm"]),
                channel_ranks=dict(row["ranks"]),
                channel_scores=dict(row["scores"]),
            )
        )
    rows.sort(key=lambda candidate: (-candidate.rrf_score, candidate.item_id))
    return tuple(rows[:top_k] if top_k is not None else rows)


def video_level_rrf(
    channels: Sequence[ModalityResult],
    *,
    weights: Optional[Mapping[str, float]] = None,
    rrf_k: int = 60,
    top_k: Optional[int] = None,
) -> Tuple[FusedCandidate, ...]:
    """Fuse independent channels after collapsing each one to video.

    ASR/OCR rows and visual frames normally have different IDs.  Therefore
    their evidence must be joined on ``video_id`` before RRF.  The returned
    candidate keeps one canonical representative frame for downstream use and
    stores the best representative from every channel in ``metadata``.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0")
    weights = dict(weights or {})
    aggregate: Dict[str, Dict[str, object]] = {}
    for channel in channels:
        key = channel.channel_key
        weight = float(weights.get(key, weights.get(channel.modality, 1.0)))
        if weight < 0:
            raise ValueError("channel weights must be non-negative")
        rank_map = _video_rank_map(channel)
        ordered = list(rank_map.items())
        normalized = normalize_scores(
            [hit.score for _, (_, hit) in ordered],
            method="minmax",
            higher_is_better=channel.higher_is_better,
        )
        for position, (video_id, (rank, hit)) in enumerate(ordered):
            row = aggregate.setdefault(
                video_id,
                {
                    "representative": hit,
                    "rrf": 0.0,
                    "norm": 0.0,
                    "ranks": {},
                    "scores": {},
                    "modalities": set(),
                    "evidence": {},
                },
            )
            row["rrf"] = float(row["rrf"]) + weight / (rrf_k + rank)
            row["norm"] = float(row["norm"]) + weight * normalized[position]
            row["ranks"][key] = rank
            row["scores"][key] = hit.score
            row["modalities"].add(key)
            row["evidence"][key] = {
                "item_id": hit.item_id,
                "video_id": hit.video_id,
                "rank": rank,
                "score": hit.score,
                "kf_n": hit.kf_n,
                "frame_idx": hit.frame_idx,
                "pts_time": hit.pts_time,
                "metadata": dict(hit.metadata),
            }
            current = row["representative"]
            if (rank, str(key), str(hit.item_id)) < (
                int(row["ranks"].get("__representative_rank", 10**9)),
                str(row["ranks"].get("__representative_channel", "~")),
                str(getattr(current, "item_id", "~")),
            ):
                row["representative"] = hit
                row["ranks"]["__representative_rank"] = rank
                row["ranks"]["__representative_channel"] = key

    rows = []
    for video_id, row in aggregate.items():
        hit = row["representative"]
        ranks = {k: v for k, v in row["ranks"].items() if not k.startswith("__")}
        rows.append(
            FusedCandidate(
                item_id=str(hit.item_id),
                video_id=str(video_id),
                score=float(row["rrf"]),
                kf_n=hit.kf_n,
                frame_idx=hit.frame_idx,
                pts_time=hit.pts_time,
                source_modalities=tuple(sorted(row["modalities"])),
                metadata={"channel_evidence": row["evidence"]},
                rrf_score=float(row["rrf"]),
                normalized_score=float(row["norm"]),
                channel_ranks=ranks,
                channel_scores=dict(row["scores"]),
            )
        )
    rows.sort(key=lambda candidate: (-candidate.rrf_score, candidate.video_id, candidate.item_id))
    return tuple(rows[:top_k] if top_k is not None else rows)


def collapse_to_videos(
    candidates: Iterable[FusedCandidate],
    *,
    top_k: Optional[int] = None,
) -> Tuple[FusedCandidate, ...]:
    """Collapse frame hits to one deterministic best representative per video."""

    best: Dict[str, FusedCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.video_id)
        if current is None or (
            candidate.rrf_score,
            candidate.normalized_score,
            candidate.item_id,
        ) > (current.rrf_score, current.normalized_score, current.item_id):
            best[candidate.video_id] = candidate
    rows = sorted(best.values(), key=lambda c: (-c.rrf_score, c.video_id, c.item_id))
    return tuple(rows[:top_k] if top_k is not None else rows)


def fuse_channels(
    channels: Sequence[ModalityResult],
    *,
    weights: Optional[Mapping[str, float]] = None,
    rrf_k: int = 60,
    top_k: Optional[int] = None,
    collapse_videos: bool = False,
) -> Tuple[FusedCandidate, ...]:
    """Convenience entry point for frame-level or video-level fusion."""

    fused = (
        video_level_rrf(channels, weights=weights, rrf_k=rrf_k, top_k=top_k)
        if collapse_videos
        else reciprocal_rank_fusion(channels, weights=weights, rrf_k=rrf_k)
    )
    if collapse_videos:
        return fused
    return fused[:top_k] if top_k is not None else fused
