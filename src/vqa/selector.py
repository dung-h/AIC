"""Pure helpers for bounded, provenance-aware VQA candidate selection.

The selector is deliberately independent of retriever/model implementations.
It only manipulates already materialized candidates and never creates an
evidence hit.  A duplicate ``(video_id, kf_n)`` is merged so that visual,
ASR/OCR, and temporal provenance survive the VLM budget allocation.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


# ``external_image`` is not a web-frame source: it is a local VKIS hit seeded
# by a web reference image. Keep it adjacent to visual provenance, while the
# original visual channel still wins stable anchor ties.
SOURCE_PRIORITY = {
    "visual": 0, "external_image": 1, "asr": 1, "ocr": 1, "temporal": 2,
}
SPECIALIST_SOURCES = frozenset({"asr", "ocr"})
TEMPORAL_SOURCES = frozenset({"temporal", "temporal_neighbor", "neighbor"})


def candidate_key(candidate: Mapping[str, Any]) -> tuple[str, int]:
    """Return the canonical selector key without inventing frame evidence."""
    video_id = str(candidate.get("video_id", ""))
    kf_n = candidate.get("kf_n")
    if not video_id or kf_n is None:
        raise ValueError("candidate requires video_id and kf_n")
    return video_id, int(kf_n)


def _source_values(candidate: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    source = candidate.get("source")
    if source:
        values.append(str(source).strip().lower())
    for value in candidate.get("sources", ()) or ():
        if value:
            values.append(str(value).strip().lower())
    for record in candidate.get("provenance", ()) or ():
        if isinstance(record, Mapping) and record.get("source"):
            values.append(str(record["source"]).strip().lower())
    return list(dict.fromkeys(value for value in values if value))


def candidate_sources(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """List real retrieval sources attached to a candidate."""
    return tuple(_source_values(candidate))


def has_source(candidate: Mapping[str, Any], source: str) -> bool:
    return str(source).strip().lower() in set(candidate_sources(candidate))


def _provenance_record(candidate: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Copy retrieval facts only; do not synthesize text/frame evidence."""
    record: dict[str, Any] = {"source": source}
    for key in ("video_id", "frame_idx", "kf_n", "pts_time", "video_rank"):
        if key in candidate:
            record[key] = candidate[key]
    # Keep the score tied to the source that produced this provenance record.
    # A merged row may carry both fields; using modality_score for a visual
    # record would corrupt visual-anchor selection.
    if source == "visual":
        score = candidate.get("base_score", candidate.get("modality_score"))
    elif source in SPECIALIST_SOURCES:
        score = candidate.get("modality_score", candidate.get("base_score"))
    else:
        score = candidate.get("base_score", candidate.get("modality_score"))
    if score is not None:
        record["score"] = float(score)
    rank = candidate.get("retrieval_rank", candidate.get("rank"))
    if rank is not None:
        record["retrieval_rank"] = int(rank)
    # When a visual and specialist row share one canonical frame, the visual
    # row remains the stable selector anchor. Preserve the specialist's real
    # text/provenance separately so downstream evidence-role joins do not
    # mistake that merged candidate for a visual-only frame.
    if source in SPECIALIST_SOURCES:
        for key in ("text", "evidence", "view_provenance", "score_mode"):
            value = candidate.get(key)
            if value not in (None, "", (), []):
                record[key] = value
    return record


def _records(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    supplied = candidate.get("provenance", ()) or ()
    for item in supplied:
        if not isinstance(item, Mapping) or not item.get("source"):
            continue
        records.append(dict(item))
    known_sources = set(_source_values(candidate))
    recorded_sources = {str(item.get("source")).lower() for item in records}
    for source in sorted(known_sources - recorded_sources, key=lambda x: SOURCE_PRIORITY.get(x, 99)):
        records.append(_provenance_record(candidate, source))
    return records


def _merge(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Merge one duplicate while keeping actual evidence and stable fields."""
    left = dict(existing)
    for key, value in incoming.items():
        left.setdefault(key, value)
    left_sources = list(candidate_sources(left))
    for source in candidate_sources(incoming):
        if source not in left_sources:
            left_sources.append(source)
    left_sources.sort(key=lambda value: (SOURCE_PRIORITY.get(value, 99), value))
    left["sources"] = left_sources
    left["source"] = left_sources[0] if left_sources else left.get("source", "visual")

    records: list[dict[str, Any]] = []
    seen_records: set[tuple[Any, ...]] = set()
    for item in [*_records(existing), *_records(incoming)]:
        key = tuple(sorted((str(k), repr(v)) for k, v in item.items()))
        if key not in seen_records:
            seen_records.add(key)
            records.append(item)
    left["provenance"] = records

    # Scores are retrieval facts, not answer evidence. Preserve the strongest
    # observed score while retaining each channel score in provenance.
    for key in ("base_score", "modality_score"):
        values = [value.get(key) for value in (existing, incoming)
                  if value.get(key) is not None]
        if values:
            left[key] = max(float(value) for value in values)
    return left


def deduplicate_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Stable dedupe by canonical key, merging all real source provenance."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []
    for raw in candidates:
        item = dict(raw)
        key = candidate_key(item)
        if key not in merged:
            merged[key] = _merge({}, item)
            order.append(key)
        else:
            merged[key] = _merge(merged[key], item)
    return [merged[key] for key in order]


def selector_metrics(candidate_pool: Iterable[Mapping[str, Any]],
                     selected: Iterable[Mapping[str, Any]],
                     ranked_video_ids: Iterable[str], *,
                     relevant_keys: Iterable[tuple[str, int]] = ()) -> dict[str, Any]:
    """Return auditable coverage/dedupe metrics for a selector stage."""
    pool = list(candidate_pool)
    chosen = list(selected)
    ranked = list(dict.fromkeys(str(value) for value in ranked_video_ids))
    pool_keys = {candidate_key(item) for item in pool}
    selected_keys = {candidate_key(item) for item in chosen}
    relevant = { (str(video), int(kf)) for video, kf in relevant_keys }
    source_counts = Counter(
        source for item in chosen for source in candidate_sources(item)
    )
    selected_videos = {candidate_key(item)[0] for item in chosen}
    return {
        "candidate_pool_count": len(pool),
        "candidate_unique_count": len(pool_keys),
        "selected_count": len(chosen),
        "selected_unique_count": len(selected_keys),
        "dedupe_collisions": len(pool) - len(pool_keys),
        "ranked_video_count": len(ranked),
        "selected_video_count": len(selected_videos),
        "video_coverage": (len(selected_videos) / len(ranked)) if ranked else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
        "relevant_pool_count": len(pool_keys.intersection(relevant)),
        "relevant_selected_count": len(selected_keys.intersection(relevant)),
        "relevant_recall": (
            len(selected_keys.intersection(relevant)) / len(pool_keys.intersection(relevant))
            if pool_keys.intersection(relevant) else None
        ),
    }


@dataclass(frozen=True)
class AllocationResult(Sequence[Mapping[str, Any]]):
    """Result of the bounded, recall-preserving candidate allocator.

    ``selected`` contains only rows from the materialized candidate pool.  It
    is a tuple so callers cannot accidentally mutate the audited allocation;
    the object is also sequence-compatible for callers that previously
    expected a list-like return value.  ``diagnostics`` is deliberately plain
    JSON-compatible data so it can be attached to a flow trace.
    """

    selected: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]
    impossible_budget_reason: str | None = None

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Alias used by benchmark/reporting callers."""
        return self.diagnostics

    def __len__(self) -> int:
        return len(self.selected)

    def __getitem__(self, index):
        return self.selected[index]

    def __iter__(self):
        return iter(self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [dict(item) for item in self.selected],
            "diagnostics": dict(self.diagnostics),
            "impossible_budget_reason": self.impossible_budget_reason,
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _candidate_score(candidate: Mapping[str, Any],
                     preferred_sources: Iterable[str] = ()) -> float:
    """Score a candidate using a requested source's score when available.

    A canonical frame may be returned by visual and ASR/OCR retrieval.  After
    deduplication the top-level score is the strongest score across channels,
    which is useful for generic ordering but wrong for choosing the visual
    anchor: a high specialist score must not make a weak visual frame look
    like the best visual frame.  Provenance retains per-source scores.
    """
    preferred = {
        str(source).strip().lower() for source in preferred_sources
        if str(source).strip()
    }
    if preferred:
        scores = [
            _safe_float(record.get("score"))
            for record in _records(candidate)
            if str(record.get("source", "")).strip().lower() in preferred
            and record.get("score") is not None
        ]
        if scores:
            return max(scores)

        # Preserve compatibility with raw rows that have not yet received a
        # provenance list.  Do not use max(base_score, modality_score) here.
        if "visual" in preferred and candidate.get("base_score") is not None:
            return _safe_float(candidate.get("base_score"))
        if preferred.intersection(SPECIALIST_SOURCES) and candidate.get("modality_score") is not None:
            return _safe_float(candidate.get("modality_score"))

    return max(
        _safe_float(candidate.get("base_score")),
        _safe_float(candidate.get("modality_score")),
    )


def _channel_score(candidate: Mapping[str, Any], source: str) -> float:
    """Score a candidate only inside one retrieval channel.

    Visual, ASR, and OCR scores are produced by different indexes and are not
    comparable.  This helper is used only to rank candidates inside one
    ``(video, source)`` group.
    """
    return _candidate_score(candidate, (str(source).strip().lower(),))


def _channel_rank_maps(
    grouped: Mapping[str, list[dict[str, Any]]],
) -> dict[tuple[str, int, str], int]:
    """Build deterministic 1-based ranks inside each video/channel."""
    ranks: dict[tuple[str, int, str], int] = {}
    for video_id, values in grouped.items():
        sources = sorted({
            source
            for candidate in values
            for source in candidate_sources(candidate)
        })
        for source in sources:
            channel_values = [
                candidate for candidate in values
                if has_source(candidate, source)
            ]
            channel_values.sort(key=lambda candidate: (
                -_channel_score(candidate, source),
                _safe_int(candidate.get("retrieval_rank", candidate.get("rank"))),
                int(candidate_key(candidate)[1]),
            ))
            for rank, candidate in enumerate(channel_values, start=1):
                ranks[(video_id, candidate_key(candidate)[1], source)] = rank
    return ranks


def _safe_int(value: Any, default: int = 2**31 - 1) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _preferred_source_order(preferred_sources: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        str(source).strip().lower()
        for source in preferred_sources
        if str(source).strip()
    ))


def _candidate_priority(candidate: Mapping[str, Any], *, video_rank: int,
                        preferred_sources: Iterable[str] = (),
                        channel_ranks: Mapping[tuple[str, int, str], int] | None = None) -> tuple[Any, ...]:
    """Build a total, deterministic order without creating candidate facts."""
    preferred_order = _preferred_source_order(preferred_sources)
    sources = set(candidate_sources(candidate))
    preferred_source_rank = min(
        (index for index, source in enumerate(preferred_order)
         if source in sources),
        default=len(preferred_order),
    )
    preferred_rank = 0 if preferred_source_rank < len(preferred_order) else 1
    # Once a caller explicitly asks for a primary specialist, source priority
    # must be evaluated *inside that lane*.  A weak OCR hit that happens to
    # share a visual frame must not beat a stronger OCR-only recipe/sign frame
    # solely because ``visual`` has global priority 0.
    prioritized_sources = sources.intersection(preferred_order) or sources
    source_rank = min(
        (SOURCE_PRIORITY.get(source, 3) for source in prioritized_sources),
        default=3,
    )
    # Localized rows are not another retrieval channel and cannot rescue a
    # video. They are, however, an explicit local proof (quote, named fact,
    # numeric event, or compact recital) inside a video already admitted by
    # global RRF. Prefer that proof over a broad global context row from the
    # same channel/video; otherwise the answer stage can receive a frame far
    # from the fact even though its timestamp was materialized correctly.
    # This is query-agnostic: localization never manufactures a frame or
    # changes the video shortlist.
    localization_rank = 0 if bool(candidate.get("localized_evidence", False)) else 1
    key = candidate_key(candidate)
    channel_ranks = channel_ranks or {}
    candidate_channel_ranks = [
        channel_ranks.get((key[0], key[1], source), 2**31 - 1)
        for source in sources
    ]
    preferred_channel_ranks = [
        channel_ranks.get((key[0], key[1], source), 2**31 - 1)
        for source in preferred_order
        if source in sources
    ]
    # Ranks are comparable only within one video/channel.  Cross-video order
    # is controlled by ``video_rank`` below, so a high ASR/OCR score cannot
    # leapfrog a better-ranked video or distort visual ranking.
    within_rank = min(preferred_channel_ranks or candidate_channel_ranks,
                      default=2**31 - 1)
    retrieval_rank = _safe_int(candidate.get("retrieval_rank", candidate.get("rank")))
    return (
        int(video_rank),
        preferred_rank,
        preferred_source_rank,
        source_rank,
        localization_rank,
        within_rank,
        retrieval_rank,
        int(candidate_key(candidate)[1]),
        tuple(sorted(sources)),
    )


def _source_match(candidate: Mapping[str, Any], sources: Iterable[str]) -> bool:
    wanted = {str(source).strip().lower() for source in sources}
    return bool(wanted.intersection(candidate_sources(candidate)))


def _has_specialist_evidence(
    candidate: Mapping[str, Any],
    sources: Iterable[str],
) -> bool:
    """Whether a specialist row carries usable text/evidence payload.

    ``source=asr``/``source=ocr`` alone is not enough to replace a visual
    anchor: synthetic or diagnostic rows often contain only a score.  The
    production global router attaches ``text``/``evidence`` to specialist
    rows, so this check keeps the required-modality preference grounded and
    preserves compatibility with score-only legacy fixtures.
    """
    wanted = {str(source).strip().lower() for source in sources}
    if not wanted.intersection(candidate_sources(candidate)):
        return False
    evidence_fields = ("text", "evidence", "chunk", "ocr_text", "transcript")
    for key in evidence_fields:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return True
    for record in candidate.get("provenance", ()) or ():
        if not isinstance(record, Mapping):
            continue
        if str(record.get("source", "")).strip().lower() not in wanted:
            continue
        for key in evidence_fields:
            value = record.get(key)
            if value is not None and str(value).strip():
                return True
    return False


def _select_from_video(
    grouped: Mapping[str, list[dict[str, Any]]],
    video_id: str,
    *,
    video_rank: int,
    seen: set[tuple[str, int]],
    selected: list[dict[str, Any]],
    counts: Counter[str],
    per_video_cap: int,
    preferred_sources: Iterable[str] = (),
    require_sources: Iterable[str] | None = None,
    channel_ranks: Mapping[tuple[str, int, str], int] | None = None,
) -> dict[str, Any] | None:
    """Take one real candidate from a video, respecting identity and cap."""
    if counts[video_id] >= per_video_cap:
        return None
    ordered = sorted(
        grouped.get(video_id, ()),
        key=lambda candidate: _candidate_priority(
            candidate,
            video_rank=video_rank,
            preferred_sources=preferred_sources,
            channel_ranks=channel_ranks,
        ),
    )
    for candidate in ordered:
        if require_sources is not None and not _source_match(candidate, require_sources):
            continue
        key = candidate_key(candidate)
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
        counts[video_id] += 1
        return candidate
    return None


def allocate_recall_preserving_candidates(
    candidates: Iterable[Mapping[str, Any]],
    ranked_video_ids: Iterable[str],
    max_candidates: int | None = None,
    *,
    max_vlm_candidates: int | None = None,
    specialist_modalities: Iterable[str] = (),
    specialist_reservation: int = 1,
    temporal_reservation: int = 1,
    per_video_cap: int = 2,
    prefer_specialist_anchors: bool = False,
    relevant_keys: Iterable[tuple[str, int]] = (),
    selection_policy: str = "coverage",
) -> AllocationResult:
    """Allocate VLM candidates with coverage before depth.

    The allocator is intentionally independent of retrieval and pipeline
    classes.  It accepts only materialized candidates and therefore cannot
    fabricate a frame, ``frame_idx``, timestamp, or evidence record.

    Allocation policy:

    1. Canonically deduplicate by ``(video_id, kf_n)`` while merging source
       provenance.
    2. Cover each eligible ranked video once before selecting a second frame.
       If the budget is smaller than the eligible-video count, the result is
       still deterministic and reports the exact impossible-budget reason.
    3. Preserve a visual anchor for visual-only queries.  For a routed query,
       prefer a specialist candidate only when it carries actual text/evidence
       payload; score-only diagnostic rows cannot evict the visual anchor.
    4. Fill remaining capacity by ranked video and within-video channel rank,
       with a hard per-video cap.

    ``selection_policy="coverage"`` is the explicit balanced A/B policy.  The
    ``adaptive`` policy is the locked production policy for a bounded VLM budget: it
        keeps a deterministic coverage floor, then spends the remaining slots on
        the next ranked within-video frames (with a hard per-video cap).  It never
    fabricates a candidate and reports the relaxed coverage floor in the
    diagnostics, so it cannot be mistaken for full top-video coverage.

    ``max_vlm_candidates`` is a named compatibility alias for callers that use
    the VLM terminology.  Exactly one of the two budget arguments is needed;
    omitting both returns an explicit ``ValueError`` rather than silently
    selecting an unbounded number of frames.
    """
    if max_candidates is None:
        max_candidates = max_vlm_candidates
    elif max_vlm_candidates is not None and max_candidates != max_vlm_candidates:
        raise ValueError("max_candidates and max_vlm_candidates disagree")
    if max_candidates is None:
        raise ValueError("max_candidates is required")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 0:
        raise ValueError("max_candidates must be an integer >= 0")
    if isinstance(per_video_cap, bool) or not isinstance(per_video_cap, int) or per_video_cap < 1:
        raise ValueError("per_video_cap must be an integer >= 1")
    if isinstance(specialist_reservation, bool) or not isinstance(specialist_reservation, int) or specialist_reservation < 0:
        raise ValueError("specialist_reservation must be an integer >= 0")
    if isinstance(temporal_reservation, bool) or not isinstance(temporal_reservation, int) or temporal_reservation < 0:
        raise ValueError("temporal_reservation must be an integer >= 0")
    if not isinstance(prefer_specialist_anchors, bool):
        raise ValueError("prefer_specialist_anchors must be a boolean")
    selection_policy = str(selection_policy).strip().lower()
    if selection_policy not in {"coverage", "adaptive"}:
        raise ValueError("selection_policy must be 'coverage' or 'adaptive'")

    raw_candidates = [dict(candidate) for candidate in candidates]
    pool = deduplicate_candidates(raw_candidates)
    ranked = list(dict.fromkeys(str(video_id) for video_id in ranked_video_ids))
    rank_by_video = {video_id: rank for rank, video_id in enumerate(ranked)}

    grouped: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in ranked}
    for candidate in pool:
        video_id = candidate_key(candidate)[0]
        if video_id in grouped:
            grouped[video_id].append(candidate)
    eligible = [video_id for video_id in ranked if grouped[video_id]]
    channel_ranks = _channel_rank_maps(grouped)
    for video_id in eligible:
        grouped[video_id].sort(
            key=lambda item: _candidate_priority(
                item,
                video_rank=rank_by_video[video_id],
                preferred_sources=("visual",),
                channel_ranks=channel_ranks,
            )
        )

    specialists = tuple(dict.fromkeys(
        str(source).strip().lower()
        for source in specialist_modalities
        if str(source).strip().lower() in SPECIALIST_SOURCES
    ))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    specialist_reserved: Counter[str] = Counter()
    temporal_reserved_videos: set[str] = set()

    # A reservation is distributed across videos instead of letting the
    # highest-scoring first video consume all specialist/neighbor slots.
    specialist_targets = {
        source: min(specialist_reservation, len(eligible))
        for source in specialists
    }

    def take(
        video_id: str,
        *,
        stage: str,
        preferred_sources: Iterable[str] = (),
        require_sources: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        effective_preferred = tuple(preferred_sources)
        if not effective_preferred:
            # No specialist modality means visual is the base channel.  When
            # a specialist is required, it is preferred for depth/coverage
            # too, but only if a real candidate carries that provenance.
            effective_preferred = specialists or ("visual",)
        item = _select_from_video(
            grouped,
            video_id,
            video_rank=rank_by_video[video_id],
            seen=seen,
            selected=selected,
            counts=counts,
            per_video_cap=per_video_cap,
            preferred_sources=effective_preferred,
            require_sources=require_sources,
            channel_ranks=channel_ranks,
        )
        if item is not None:
            stages[stage] += 1
            for source in specialists:
                if has_source(item, source):
                    specialist_reserved[source] += 1
            if _source_match(item, TEMPORAL_SOURCES):
                temporal_reserved_videos.add(video_id)
        return item

    # Coverage pass is a hard ranked-video invariant. Visual is the base
    # channel, but a requested ASR/OCR channel gets first refusal inside each
    # covered video. This prevents the old visual-first rule from discarding
    # the only frame that can answer a spoken/text question.
    if selection_policy == "adaptive":
        # Keep a majority of the budget as top-ranked video anchors, then use
        # the remaining slots for depth.  This is deliberately not an oracle
        # policy: it uses only retrieval scores/provenance already present in
        # the candidate pool.  The floor is exposed below so a benchmark can
        # measure the video-coverage trade-off explicitly.
        coverage_floor = min(
            len(eligible),
            # Keep most budget slots on distinct ranked videos.  The old 2/3
            # floor dropped four video anchors at budget=12 and caused a
            # measurable video-recall regression.  Reserve only a bounded
            # one-sixth depth slice for within-video recovery.
            max(1, max_candidates - max(1, max_candidates // 6))
            if max_candidates else 0,
        )
    else:
        coverage_floor = min(max_candidates, len(eligible))
    coverage_video_ids = eligible[:coverage_floor]
    specialist_anchor_enabled = bool(specialists) and (
        # A declared primary ASR/OCR contract is not a mere support lane. Its
        # real evidence must anchor the answer candidate even when the budget
        # covers every ranked video. Otherwise a screen-text query can select
        # a generic visual neighbour and discard the recipe/sign frame that
        # actually proves the answer.
        prefer_specialist_anchors
        or coverage_floor < len(eligible)
        or max_candidates < len(eligible)
    )
    for video_id in coverage_video_ids:
        specialist_candidates = [
            candidate for candidate in grouped[video_id]
            if any(
                _has_specialist_evidence(candidate, (source,))
                for source in specialists
            )
        ]
        # If both channels point to the same canonical frame, keeping that
        # merged row as the anchor does not sacrifice visual evidence and is
        # safe even when the budget covers every ranked video.
        has_multimodal_anchor = any(
            has_source(candidate, "visual")
            and any(has_source(candidate, source) for source in specialists)
            for candidate in specialist_candidates
        )
        has_specialist = bool(specialist_candidates) and (
            specialist_anchor_enabled or has_multimodal_anchor
        )
        has_visual = any(
            has_source(candidate, "visual")
            for candidate in grouped[video_id]
        )
        if has_specialist:
            preferred_sources = specialists
            require_sources = specialists
            stage = "specialist_anchor"
        elif has_visual:
            preferred_sources = ("visual",)
            require_sources = ("visual",)
            stage = "visual_anchor"
        else:
            preferred_sources = ()
            require_sources = None
            stage = "coverage_fallback"
        item = take(
            video_id,
            stage=stage,
            preferred_sources=preferred_sources,
            require_sources=require_sources,
        )
        if item is None:
            # This branch is only possible for a malformed pool after grouping;
            # it is reported rather than replaced with a fabricated sentinel.
            continue

    if selection_policy == "adaptive":
        # The adaptive branch intentionally skips specialist/temporal
        # reservation before depth.  A reservation would consume the very
        # slots that are meant to repair within-video frame misses.  Remaining
        # candidates are ordered by ranked video first, then by the requested
        # channel's within-video rank.  Raw ASR/OCR scores never compete with
        # raw visual scores here.
        depth_sources = specialists or ("visual",)
        remaining = [
            candidate for candidate in pool
            if candidate_key(candidate) not in seen
            and candidate_key(candidate)[0] in rank_by_video
        ]
        remaining.sort(key=lambda candidate: (
            *_candidate_priority(
                candidate,
                video_rank=rank_by_video[candidate_key(candidate)[0]],
                preferred_sources=depth_sources,
                channel_ranks=channel_ranks,
            ),
        ))
        for candidate in remaining:
            if len(selected) >= max_candidates:
                break
            video_id = candidate_key(candidate)[0]
            if counts[video_id] >= per_video_cap:
                continue
            selected.append(candidate)
            seen.add(candidate_key(candidate))
            counts[video_id] += 1
            stages["adaptive_utility"] += 1
            for source in specialists:
                if has_source(candidate, source):
                    specialist_reserved[source] += 1
            if _source_match(candidate, TEMPORAL_SOURCES):
                temporal_reserved_videos.add(video_id)
    if selection_policy != "adaptive":
        # If multiple specialist modalities are requested, finish their
        # explicit reservation across different videos before generic depth.
        # Adaptive owns its remaining slots exclusively through utility order.
        for source in specialists:
            while specialist_reserved[source] < specialist_targets[source] and len(selected) < max_candidates:
                picked = None
                for video_id in eligible:
                    if counts[video_id] >= per_video_cap:
                        continue
                    picked = take(
                        video_id,
                        stage="specialist_reservation",
                        preferred_sources=(source,),
                        require_sources=(source,),
                    )
                    if picked is not None:
                        break
                if picked is None:
                    break

        # Temporal-neighbor reservation follows specialist evidence. It is a
        # separate diagnostic quota, not a reason to replace a required
        # coverage frame when the budget is already exhausted.
        while len(temporal_reserved_videos) < temporal_reservation and len(selected) < max_candidates:
            picked = None
            temporal_order = [
                video_id for video_id in eligible
                if video_id not in temporal_reserved_videos
            ] + [
                video_id for video_id in eligible
                if video_id in temporal_reserved_videos
            ]
            for video_id in temporal_order:
                if counts[video_id] >= per_video_cap:
                    continue
                picked = take(
                    video_id,
                    stage="temporal_reservation",
                    preferred_sources=TEMPORAL_SOURCES,
                    require_sources=TEMPORAL_SOURCES,
                )
                if picked is not None:
                    break
            if picked is None:
                break

    # Depth pass: round-robin, hard-capped at two by default. This is the
    # critical anti-starvation property: no video can consume depth slots
    # until every eligible video has received its first candidate.
    while len(selected) < max_candidates:
        added = False
        for video_id in eligible:
            if len(selected) >= max_candidates:
                break
            if counts[video_id] >= per_video_cap:
                continue
            if take(
                video_id,
                stage="depth",
                preferred_sources=specialists or ("visual",),
            ) is not None:
                added = True
        if not added:
            break

    selected_video_ids = {candidate_key(item)[0] for item in selected}
    uncovered = [video_id for video_id in eligible if video_id not in selected_video_ids]
    coverage_target = min(max_candidates, len(eligible))
    impossible_reason = None
    if max_candidates < len(eligible):
        impossible_reason = (
            "budget_too_small_for_full_video_coverage: "
            f"max_candidates={max_candidates} < eligible_video_count={len(eligible)}; "
            f"{len(eligible) - max_candidates} ranked eligible video(s) cannot receive a frame"
        )

    reservation_unmet: list[str] = []
    for source, target in specialist_targets.items():
        if specialist_reserved[source] < target:
            reservation_unmet.append(
                f"{source}:{specialist_reserved[source]}/{target}"
            )
    if len(temporal_reserved_videos) < min(temporal_reservation, len(eligible)):
        reservation_unmet.append(
            f"temporal:{len(temporal_reserved_videos)}/{min(temporal_reservation, len(eligible))}"
        )

    diagnostics = selector_metrics(
        pool,
        selected,
        ranked,
        relevant_keys=relevant_keys,
    )
    diagnostics.update({
        "allocator": "recall_preserving_v1" if selection_policy == "coverage" else "adaptive_quality_v1",
        "selection_policy": selection_policy,
        "max_candidates": max_candidates,
        "per_video_cap": per_video_cap,
        "eligible_video_count": len(eligible),
        "eligible_video_ids": list(eligible),
        "coverage_target": coverage_target,
        "coverage_floor": coverage_floor,
        "coverage_guaranteed": not uncovered and max_candidates >= len(eligible),
        "coverage_floor_guaranteed": not any(
            video_id not in selected_video_ids for video_id in coverage_video_ids
        ),
        "uncovered_eligible_videos": uncovered,
        "impossible_budget_reason": impossible_reason,
        "specialist_modalities": list(specialists),
        "specialist_target": dict(specialist_targets),
        "specialist_reserved": dict(specialist_reserved),
        "temporal_target": min(temporal_reservation, len(eligible)),
        "temporal_reserved": len(temporal_reserved_videos),
        "reservation_unmet": reservation_unmet,
        "budget_unused": max_candidates - len(selected),
        "selection_stages": dict(sorted(stages.items())),
        "score_policy": "within_video_channel_rank_v1",
        "specialist_anchor_enabled": specialist_anchor_enabled,
        "prefer_specialist_anchors": prefer_specialist_anchors,
        "visual_anchor_policy": (
            "specialist_evidence_first_then_visual_when_budget_constrained"
            if specialists
            else "visual_first"
        ),
        "specialist_anchor_target_video_ids": [
            video_id for video_id in coverage_video_ids
            if any(
                _has_specialist_evidence(candidate, specialists)
                for candidate in grouped[video_id]
            )
        ],
        "specialist_anchor_selected_video_ids": [
            video_id for video_id in coverage_video_ids
            if any(
                candidate_key(item)[0] == video_id
                and _has_specialist_evidence(item, specialists)
                for item in selected
            )
        ],
        "visual_anchor_target_video_ids": list(coverage_video_ids),
        "visual_anchor_available_video_ids": [
            video_id for video_id in coverage_video_ids
            if any(has_source(candidate, "visual") for candidate in grouped[video_id])
        ],
        "visual_anchor_missing_video_ids": [
            video_id for video_id in coverage_video_ids
            if not any(has_source(candidate, "visual") for candidate in grouped[video_id])
        ],
        "visual_anchor_selected_video_ids": [
            video_id for video_id in coverage_video_ids
            if any(
                candidate_key(item)[0] == video_id and has_source(item, "visual")
                for item in selected
            )
        ],
        "out_of_ranked_pool_count": sum(
            1 for item in pool if candidate_key(item)[0] not in rank_by_video
        ),
    })
    visual_anchor_available = diagnostics["visual_anchor_available_video_ids"]
    visual_anchor_selected = diagnostics["visual_anchor_selected_video_ids"]
    diagnostics["visual_anchor_preservation_rate"] = (
        len(visual_anchor_selected) / len(visual_anchor_available)
        if visual_anchor_available else None
    )
    return AllocationResult(
        selected=tuple(selected),
        diagnostics=diagnostics,
        impossible_budget_reason=impossible_reason,
    )


def allocate_candidates(*args, **kwargs) -> AllocationResult:
    """Compatibility name for the recall-preserving allocator."""
    return allocate_recall_preserving_candidates(*args, **kwargs)


def stage_record(stage: str, candidate: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """Create a traceable stage record from candidate facts only."""
    key = candidate_key(candidate)
    record = {"stage": str(stage), "candidate_key": [key[0], key[1]],
              "sources": list(candidate_sources(candidate)),
              "provenance": [dict(item) for item in candidate.get("provenance", ()) or ()]}
    record.update(extra)
    return record
