"""Deterministic candidate selection and catalog-independent temporal hooks."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

from .contracts import RetrievalCandidate


def _candidate_key(candidate: RetrievalCandidate) -> Tuple[str, object]:
    """Prefer canonical frame identity, with a safe item fallback."""

    if candidate.frame_idx is not None:
        return candidate.video_id, ("frame_idx", candidate.frame_idx)
    if candidate.kf_n is not None:
        return candidate.video_id, ("kf_n", candidate.kf_n)
    return candidate.video_id, ("item_id", candidate.item_id)


def _ordered(candidates: Iterable[RetrievalCandidate]) -> Tuple[RetrievalCandidate, ...]:
    return tuple(sorted(candidates, key=lambda c: (-c.score, c.video_id, c.item_id)))


def dedupe_candidates(
    candidates: Iterable[RetrievalCandidate],
    *,
    key_fn: Optional[Callable[[RetrievalCandidate], object]] = None,
) -> Tuple[RetrievalCandidate, ...]:
    """Dedupe by canonical frame identity, retaining the strongest candidate."""

    key_fn = key_fn or _candidate_key
    best = {}
    for candidate in candidates:
        key = key_fn(candidate)
        current = best.get(key)
        if current is None or (candidate.score, candidate.item_id) > (
            current.score,
            current.item_id,
        ):
            best[key] = candidate
    return _ordered(best.values())


def select_top_k(
    candidates: Iterable[RetrievalCandidate],
    k: int,
    *,
    dedupe: bool = True,
) -> Tuple[RetrievalCandidate, ...]:
    """Select a stable top-k list; ``k=0`` returns an empty tuple."""

    if k < 0:
        raise ValueError("k must be >= 0")
    rows = dedupe_candidates(candidates) if dedupe else _ordered(candidates)
    return rows[:k]


NeighborFn = Callable[[RetrievalCandidate, int], Iterable[RetrievalCandidate]]


def expand_temporal_neighbors(
    candidates: Sequence[RetrievalCandidate],
    neighbor_fn: NeighborFn,
    *,
    radius: int = 2,
    max_candidates: Optional[int] = None,
) -> Tuple[RetrievalCandidate, ...]:
    """Add catalog-provided temporal neighbors without knowing catalog details.

    ``neighbor_fn`` owns timestamp/keyframe lookup.  This function only
    orchestrates expansion, deduplication, and deterministic truncation.
    """

    if radius < 0:
        raise ValueError("radius must be >= 0")
    expanded = list(candidates)
    for candidate in candidates:
        expanded.extend(neighbor_fn(candidate, radius))
    result = dedupe_candidates(expanded)
    if max_candidates is not None:
        if max_candidates < 0:
            raise ValueError("max_candidates must be >= 0")
        result = result[:max_candidates]
    return result
