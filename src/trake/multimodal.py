"""Event-level multimodal TRAKE retrieval and DANTE alignment.

The class in this module owns orchestration and evidence fusion. Model/index
lifecycle remains in the duck-typed ``search_event`` retrievers supplied by
the caller.

The key invariant is that DANTE receives a fused event-by-timeline matrix:
visual/ASR/OCR evidence for the same canonical frame contributes positively,
and an unavailable modality is unknown rather than a negative score.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.utils.dante import dante_align

from .contracts import (
    SUPPORTED_MODALITIES,
    EventEvidence,
    TrakeContractError,
    TrakeEvent,
    normalize_events,
    validate_sequence_path,
)


class MissingModalityRetriever(TrakeContractError):
    """Raised when an event explicitly requires an unavailable modality."""


LEGACY_ALIGNMENT_POLICY = "legacy"
COVERAGE_COHERENT_ALIGNMENT_POLICY = "coverage_coherent_v1"


@dataclass(frozen=True)
class _ModalityPolicy:
    required: tuple[str, ...]
    optional: tuple[str, ...]

    @property
    def requested(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.required + self.optional)))


@dataclass(frozen=True)
class _NormalizedEvidence:
    evidence: EventEvidence
    normalized_score: float
    rank: int


@dataclass(frozen=True)
class _FusedCandidate:
    evidence: tuple[EventEvidence, ...]
    fused_score: float
    modality_scores: Mapping[str, float]


class EventLevelMultimodalDante:
    """Retrieve event modalities in parallel, fuse evidence, then align.

    ``modalities`` on an event keeps its historical meaning: required
    modalities. ``required_modalities`` is an explicit alias. New
    ``optional_modalities`` are best effort. An event without a declaration
    searches all configured retrievers as optional channels.
    """

    def __init__(
        self,
        retrievers: Mapping[str, object],
        *,
        modality_weights: Mapping[str, float] | None = None,
        missing_score: float = -1e6,
        max_workers: int = 3,
        executor_factory: Callable[..., Any] | None = None,
        alignment_policy: str = LEGACY_ALIGNMENT_POLICY,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one event-level retriever is required")
        self.retrievers: dict[str, object] = {}
        for raw_modality, retriever in retrievers.items():
            modality = str(raw_modality).strip().lower()
            if not modality:
                raise ValueError("retriever modality must be non-empty")
            if modality not in SUPPORTED_MODALITIES:
                raise ValueError(f"unsupported retriever modality {modality!r}")
            if not hasattr(retriever, "search_event"):
                raise TypeError(f"retriever {modality!r} must expose search_event()")
            if modality in self.retrievers:
                raise ValueError(f"duplicate retriever modality: {modality}")
            self.retrievers[modality] = retriever

        self.modality_weights = {
            modality: float((modality_weights or {}).get(modality, 1.0))
            for modality in self.retrievers
        }
        if any(
            not np.isfinite(weight) or weight <= 0
            for weight in self.modality_weights.values()
        ):
            raise ValueError("modality weights must be finite and positive")
        if not np.isfinite(float(missing_score)):
            raise ValueError("missing_score must be finite")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        alignment_policy = str(alignment_policy).strip().lower()
        if alignment_policy not in {
            LEGACY_ALIGNMENT_POLICY,
            COVERAGE_COHERENT_ALIGNMENT_POLICY,
        }:
            raise ValueError(
                "alignment_policy must be one of "
                f"{LEGACY_ALIGNMENT_POLICY!r}, {COVERAGE_COHERENT_ALIGNMENT_POLICY!r}"
            )
        self.missing_score = float(missing_score)
        self.max_workers = int(max_workers)
        self._executor_factory = executor_factory or ThreadPoolExecutor
        self.alignment_policy = alignment_policy
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _as_modality_tuple(value: Any) -> tuple[str, ...]:
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            values = value.replace(";", ",").split(",")
        else:
            try:
                values = list(value)
            except TypeError as exc:
                raise TrakeContractError(
                    "event modality declaration must be a string or sequence"
                ) from exc
        result: list[str] = []
        for value in values:
            modality = str(value).strip().lower()
            if not modality:
                continue
            if modality not in SUPPORTED_MODALITIES:
                raise TrakeContractError(f"unsupported TRAKE modality {modality!r}")
            if modality not in result:
                result.append(modality)
        return tuple(sorted(result))

    def _policies_for_events(
        self,
        raw_events: Sequence[Any],
        normalized: Sequence[TrakeEvent],
    ) -> dict[int, _ModalityPolicy]:
        policies: dict[int, _ModalityPolicy] = {}
        for event in normalized:
            raw = raw_events[event.index] if event.index < len(raw_events) else event
            if isinstance(raw, Mapping):
                if "required_modalities" in raw:
                    required = self._as_modality_tuple(raw.get("required_modalities"))
                elif "modalities" in raw:
                    required = self._as_modality_tuple(raw.get("modalities"))
                elif "modality" in raw:
                    required = self._as_modality_tuple(raw.get("modality"))
                else:
                    required = ()
                optional = self._as_modality_tuple(raw.get("optional_modalities"))
            elif event.modalities:
                required = tuple(sorted(event.modalities))
                optional = ()
            else:
                required = ()
                optional = tuple(sorted(self.retrievers))
            optional = tuple(item for item in optional if item not in required)
            policies[event.index] = _ModalityPolicy(required, optional)
        return policies

    def _modalities_for_event(
        self,
        event: TrakeEvent,
        policy: _ModalityPolicy | None = None,
    ) -> tuple[str, ...]:
        if policy is None:
            if event.modalities:
                policy = _ModalityPolicy(tuple(sorted(event.modalities)), ())
            else:
                policy = _ModalityPolicy((), tuple(sorted(self.retrievers)))
        missing = [item for item in policy.required if item not in self.retrievers]
        if missing:
            raise MissingModalityRetriever(
                f"event {event.index} requires unavailable modality(s): {sorted(missing)}"
            )
        return tuple(item for item in policy.requested if item in self.retrievers)

    @staticmethod
    def _canonical_sort_key(item: EventEvidence) -> tuple[Any, ...]:
        return (
            item.video_id,
            int(item.frame_idx),
            float(item.pts_time),
            -float(item.score),
            item.modality,
            int(item.kf_n) if item.kf_n is not None else -1,
            item.source_id,
        )

    def _validate_raw_hits(
        self,
        raw_hits: Sequence[Any] | None,
        *,
        event: TrakeEvent,
        modality: str,
        allowed_videos: set[str] | None,
    ) -> list[EventEvidence]:
        evidence_rows: list[EventEvidence] = []
        for raw in raw_hits or []:
            evidence = (
                raw
                if isinstance(raw, EventEvidence)
                else EventEvidence.from_mapping(raw, event_index=event.index, modality=modality)
            )
            if evidence.event_index != event.index:
                raise TrakeContractError("retriever returned evidence for the wrong event")
            if evidence.modality != modality:
                raise TrakeContractError(
                    f"retriever {modality} returned modality {evidence.modality}"
                )
            if allowed_videos is not None and evidence.video_id not in allowed_videos:
                continue
            evidence_rows.append(evidence)
        evidence_rows.sort(key=self._canonical_sort_key)
        return evidence_rows

    def retrieve_events(
        self,
        events: Sequence[Any],
        *,
        top_k_per_event: int = 100,
        candidate_videos: Sequence[str] | None = None,
    ) -> dict[int, list[EventEvidence]]:
        """Run requested searches via bounded parallel fanout."""

        raw_events = list(events)
        normalized = normalize_events(raw_events)
        if top_k_per_event < 1 or top_k_per_event > 1000:
            raise ValueError("top_k_per_event must be between 1 and 1000")
        allowed_videos = set(map(str, candidate_videos)) if candidate_videos is not None else None
        policies = self._policies_for_events(raw_events, normalized)

        jobs: list[tuple[TrakeEvent, str, bool]] = []
        for event in normalized:
            policy = policies[event.index]
            self._modalities_for_event(event, policy)
            required = set(policy.required)
            for modality in policy.requested:
                if modality in self.retrievers:
                    jobs.append((event, modality, modality in required))

        worker_count = min(self.max_workers, max(1, len(jobs)))
        output: dict[int, list[EventEvidence]] = {event.index: [] for event in normalized}
        modality_counts = {item: 0 for item in sorted(self.retrievers)}
        modality_calls = {item: 0 for item in sorted(self.retrievers)}
        optional_skipped: list[dict[str, Any]] = []
        modality_errors: list[dict[str, Any]] = []

        futures: list[Any] = []
        for event in normalized:
            policy = policies[event.index]
            for modality in policy.optional:
                if modality not in self.retrievers:
                    optional_skipped.append(
                        {
                            "event_index": event.index,
                            "modality": modality,
                            "reason": "unavailable_retriever",
                        }
                    )
        with self._executor_factory(max_workers=worker_count) as executor:
            for event, modality, _required in jobs:
                modality_calls[modality] += 1
                futures.append(
                    executor.submit(
                        self.retrievers[modality].search_event,
                        event,
                        top_k=top_k_per_event,
                        candidate_videos=candidate_videos,
                    )
                )
            for (event, modality, required), future in zip(jobs, futures):
                try:
                    raw_hits = future.result()
                    hits = self._validate_raw_hits(
                        raw_hits,
                        event=event,
                        modality=modality,
                        allowed_videos=allowed_videos,
                    )
                except Exception as exc:
                    if required:
                        raise TrakeContractError(
                            f"required {modality} retrieval failed for event {event.index}: {exc}"
                        ) from exc
                    optional_skipped.append(
                        {"event_index": event.index, "modality": modality, "reason": "retriever_error"}
                    )
                    modality_errors.append(
                        {
                            "event_index": event.index,
                            "modality": modality,
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue
                output[event.index].extend(hits)
                modality_counts[modality] += len(hits)

        for event in normalized:
            output[event.index].sort(
                key=lambda item: (
                    item.modality,
                    -float(item.score),
                    item.video_id,
                    int(item.frame_idx),
                    float(item.pts_time),
                    item.source_id,
                )
            )
        self.last_diagnostics = {
            "modality_counts": modality_counts,
            "modality_calls": modality_calls,
            "optional_skipped": optional_skipped,
            "modality_errors": modality_errors,
            "parallel_workers": worker_count,
            "parallel_job_count": len(jobs),
            "event_order": [event.index for event in normalized],
            "event_modality_order": [
                {"event_index": event.index, "modalities": list(policies[event.index].requested)}
                for event in normalized
            ],
            "missing_is_unknown": True,
        }
        return output

    @staticmethod
    def _rank_score(rank: int, count: int) -> float:
        return float(count - rank) / float(max(1, count))

    @staticmethod
    def _score_normalization(scores: Sequence[float]) -> list[float]:
        values = [float(score) for score in scores]
        if not values:
            return []
        low, high = min(values), max(values)
        if not math.isfinite(low) or not math.isfinite(high):
            raise TrakeContractError("retriever scores must be finite")
        if high <= low:
            return [1.0 for _ in values]
        return [(value - low) / (high - low) for value in values]

    def _normalize_event_hits(self, hits: Sequence[EventEvidence]) -> list[_NormalizedEvidence]:
        """Normalize rank and score independently for each event/modality."""

        by_modality: dict[str, list[EventEvidence]] = defaultdict(list)
        for hit in hits:
            by_modality[hit.modality].append(hit)
        normalized: list[_NormalizedEvidence] = []
        for modality in sorted(by_modality):
            modality_hits = sorted(
                by_modality[modality],
                key=lambda item: (-float(item.score), self._canonical_sort_key(item)),
            )
            score_values = self._score_normalization([item.score for item in modality_hits])
            count = len(modality_hits)
            for rank, (item, score_norm) in enumerate(zip(modality_hits, score_values)):
                normalized.append(
                    _NormalizedEvidence(
                        evidence=item,
                        normalized_score=max(
                            0.0,
                            0.5 * (float(score_norm) + self._rank_score(rank, count)),
                        ),
                        rank=rank,
                    )
                )
        return normalized

    def _fuse_event_hits(
        self,
        events: Sequence[TrakeEvent],
        event_hits: Mapping[int, Sequence[EventEvidence]],
    ) -> dict[str, dict[tuple[str, int], dict[int, _FusedCandidate]]]:
        """Fuse positive evidence at ``(video_id, canonical frame_idx)``."""

        fused: dict[str, dict[tuple[str, int], dict[int, _FusedCandidate]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for event in events:
            normalized_hits = self._normalize_event_hits(event_hits.get(event.index, ()))
            by_key_modality: dict[tuple[str, int], dict[str, list[_NormalizedEvidence]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for item in normalized_hits:
                by_key_modality[item.evidence.key()][item.evidence.modality].append(item)

            for key, modality_map in by_key_modality.items():
                modality_scores: dict[str, float] = {}
                all_evidence: list[EventEvidence] = []
                for modality in sorted(modality_map):
                    modality_items = modality_map[modality]
                    # Repeated hits from one channel remain in provenance but
                    # only the strongest one can contribute to its score.
                    modality_scores[modality] = max(
                        item.normalized_score for item in modality_items
                    )
                    all_evidence.extend(item.evidence for item in modality_items)
                all_evidence.sort(
                    key=lambda item: (
                        item.modality,
                        -float(item.score),
                        item.source_id,
                        int(item.frame_idx),
                    )
                )
                fused[key[0]][key][event.index] = _FusedCandidate(
                    evidence=tuple(all_evidence),
                    fused_score=float(
                        sum(
                            self.modality_weights.get(modality, 1.0) * value
                            for modality, value in modality_scores.items()
                        )
                    ),
                    modality_scores=dict(modality_scores),
                )
        return {video_id: dict(frame_map) for video_id, frame_map in fused.items()}

    def _build_video_candidates(
        self,
        events: Sequence[TrakeEvent],
        event_hits: Mapping[int, Sequence[EventEvidence]],
    ) -> dict[str, dict[tuple[str, int], list[EventEvidence]]]:
        """Return only videos with observed evidence for every event."""

        by_video: dict[str, dict[tuple[str, int], list[EventEvidence]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in events:
            for evidence in event_hits.get(event.index, ()):
                by_video[evidence.video_id][evidence.key()].append(evidence)
        return {
            video_id: dict(frame_map)
            for video_id, frame_map in by_video.items()
            if all(
                any(
                    item.event_index == event.index
                    for values in frame_map.values()
                    for item in values
                )
                for event in events
            )
        }

    def _alignment_diagnostics(
        self,
        fused: Mapping[str, Mapping[tuple[str, int], Mapping[int, _FusedCandidate]]],
        event_hits: Mapping[int, Sequence[EventEvidence]],
        events: Sequence[TrakeEvent],
    ) -> dict[str, Any]:
        diagnostics = dict(self.last_diagnostics)
        diagnostics.update(
            {
                "fused_candidate_count": int(sum(len(value) for value in fused.values())),
                "fused_candidate_count_by_video": {
                    video_id: len(frame_map) for video_id, frame_map in sorted(fused.items())
                },
                "fused_candidate_count_by_event": {
                    str(event.index): int(
                        sum(
                            1
                            for frame_map in fused.values()
                            for candidate_map in frame_map.values()
                            if event.index in candidate_map
                        )
                    )
                    for event in events
                },
                "raw_hit_count": int(sum(len(value) for value in event_hits.values())),
                "event_count": len(events),
                "missing_is_unknown": True,
            }
        )
        return diagnostics

    @staticmethod
    def _event_score_quality(
        candidate: _FusedCandidate,
        values: Sequence[float],
        *,
        candidate_value: float | None = None,
    ) -> float:
        """Return a bounded, event-balanced quality score.

        ``_normalize_event_hits`` already makes scores comparable inside a
        modality.  A second normalization over the fused candidates prevents
        an event with two corroborating modalities from dominating an event
        with one modality merely because its raw sums are larger.  Rank is
        blended in so a flat or tiny score range remains useful.
        """

        if not values:
            return 0.0
        ordered = sorted((float(value) for value in values), reverse=True)
        score_norm = EventLevelMultimodalDante._score_normalization(ordered)
        try:
            target = float(
                candidate.fused_score if candidate_value is None else candidate_value
            )
            value_rank = ordered.index(target)
        except ValueError:
            return 0.0
        score_component = float(score_norm[value_rank])
        rank_component = EventLevelMultimodalDante._rank_score(value_rank, len(ordered))
        return float(max(0.0, min(1.0, 0.7 * score_component + 0.3 * rank_component)))

    def _coverage_coherent_path_score(
        self,
        *,
        events: Sequence[TrakeEvent],
        policies: Mapping[int, _ModalityPolicy],
        event_score_pools: Mapping[int, Sequence[float]],
        frame_map: Mapping[tuple[str, int], Mapping[int, _FusedCandidate]],
        timeline: Sequence[tuple[str, int]],
        path: Sequence[int],
        dante_score: float,
    ) -> tuple[float, dict[str, Any]]:
        """Score a complete path with conservative coverage/coherence terms.

        This policy is deliberately opt-in.  It does not invent penalties for
        large but valid temporal gaps: DANTE's order constraint and the strict
        contract remain authoritative.  Coherence means that event qualities
        are consistent along the selected path, while coverage measures how
        much candidate support each event has in the video.
        """

        event_values: dict[int, list[float]] = {event.index: [] for event in events}
        event_key_counts: dict[int, int] = {event.index: 0 for event in events}
        for key in timeline:
            candidates = frame_map[key]
            for event in events:
                candidate = candidates.get(event.index)
                if candidate is not None:
                    event_values[event.index].append(float(candidate.fused_score))
                    event_key_counts[event.index] += 1

        selected_quality: list[float] = []
        selected_modalities: list[float] = []
        for event in events:
            key = timeline[int(path[event.index])]
            candidate = frame_map[key].get(event.index)
            if candidate is None:
                # The caller only invokes this method for paths that pass the
                # complete-path gate, but fail closed if that invariant ever
                # regresses.
                return -math.inf, {
                    "coverage": 0.0,
                    "support_coverage": 0.0,
                    "score_coherence": 0.0,
                    "selected_event_quality": [],
                    "dante_score": float(dante_score),
                }
            selected_quality.append(
                self._event_score_quality(
                    candidate,
                    event_score_pools.get(event.index, event_values[event.index]),
                    candidate_value=self._candidate_base_score(candidate),
                )
            )
            # Required channels are a hard coverage condition below. Optional
            # channels are unknown when absent, so they are excluded from the
            # denominator rather than silently becoming a negative score.
            policy = policies[event.index]
            observed = set(candidate.modality_scores)
            expected = set(policy.required)
            expected.update(observed.intersection(policy.optional))
            selected_modalities.append(
                len(observed.intersection(expected)) / max(1.0, len(expected))
            )

        event_count = max(1, len(events))
        coverage = sum(bool(event_values[event.index]) for event in events) / event_count
        # One candidate is enough for correctness; additional independent
        # candidates provide robustness but saturate quickly to avoid rewarding
        # dense/generic videos disproportionately.
        support_coverage = sum(
            min(1.0, event_key_counts[event.index] / 3.0) for event in events
        ) / event_count
        mean_quality = float(np.mean(selected_quality)) if selected_quality else 0.0
        min_quality = float(min(selected_quality)) if selected_quality else 0.0
        score_coherence = float(max(0.0, 1.0 - np.std(selected_quality)))
        modality_support = float(np.mean(selected_modalities)) if selected_modalities else 0.0

        # Coverage is a gate in the caller; these modest weights keep the
        # policy from overturning a materially stronger semantic path solely
        # because one video has denser keyframes.
        final_score = (
            0.45 * mean_quality
            + 0.20 * min_quality
            + 0.15 * support_coverage
            + 0.10 * score_coherence
            + 0.10 * modality_support
        )
        return float(final_score), {
            "coverage": float(coverage),
            "support_coverage": float(support_coverage),
            "score_coherence": float(score_coherence),
            "mean_event_quality": float(mean_quality),
            "min_event_quality": float(min_quality),
            "modality_support": float(modality_support),
            "selected_event_quality": [float(value) for value in selected_quality],
            "selected_modality_support": [float(value) for value in selected_modalities],
            "dante_score": float(dante_score),
        }

    def _candidate_base_score(self, candidate: _FusedCandidate) -> float:
        """Make a fused candidate comparable across active modality counts."""

        active_weight = sum(
            self.modality_weights.get(modality, 1.0)
            for modality in candidate.modality_scores
        )
        return float(candidate.fused_score / max(active_weight, 1e-12))

    @staticmethod
    def _safe_dante_align(
        scores: np.ndarray,
        valid_mask: np.ndarray,
        *,
        lam: float,
    ) -> tuple[float, list[int] | None]:
        """Fail-closed DANTE fallback with explicit valid-candidate masking.

        This local implementation is intentionally used only when the shared
        DANTE helper returns no valid backtrace.  It keeps this component
        robust to a broken/legacy helper without changing that helper or
        allowing the ``missing_score`` sentinel to enter a submission path.
        """

        values = np.asarray(scores, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        if values.ndim != 2 or mask.shape != values.shape:
            raise TrakeContractError("DANTE scores and valid mask must have the same 2-D shape")
        event_count, timeline_count = values.shape
        if event_count < 1 or timeline_count < event_count:
            return -1e9, None
        mask &= np.isfinite(values)
        if not np.all(np.any(mask, axis=1)):
            return -1e9, None

        dp = np.full((event_count, timeline_count), -np.inf, dtype=np.float64)
        backpointer = np.full((event_count, timeline_count), -1, dtype=np.int32)
        dp[0] = np.where(mask[0], values[0], -np.inf)
        for event_index in range(1, event_count):
            best_value = -np.inf
            best_previous = -1
            for timeline_index in range(timeline_count):
                previous = timeline_index - 1
                if (
                    previous >= 0
                    and np.isfinite(dp[event_index - 1, previous])
                    and dp[event_index - 1, previous] + lam * previous > best_value
                ):
                    best_value = dp[event_index - 1, previous] + lam * previous
                    best_previous = previous
                if best_previous >= 0 and mask[event_index, timeline_index]:
                    dp[event_index, timeline_index] = (
                        values[event_index, timeline_index]
                        + best_value
                        - lam * timeline_index
                    )
                    backpointer[event_index, timeline_index] = best_previous

        if not np.any(np.isfinite(dp[-1])):
            return -1e9, None
        end = int(np.argmax(dp[-1]))
        path = [0] * event_count
        path[-1] = end
        for event_index in range(event_count - 1, 0, -1):
            previous = int(backpointer[event_index, path[event_index]])
            if previous < 0:
                return -1e9, None
            path[event_index - 1] = previous
        return float(dp[-1, end]), path

    def align(
        self,
        events: Sequence[Any],
        *,
        top_k_videos: int = 10,
        top_k_per_event: int = 100,
        candidate_videos: Sequence[str] | None = None,
        lam: float = 0.0,
    ) -> dict[str, Any]:
        """Return complete top-ranked same-video event sequences."""

        raw_events = list(events)
        normalized = normalize_events(raw_events)
        if top_k_videos < 1 or top_k_videos > 100:
            raise ValueError("top_k_videos must be between 1 and 100")
        if not np.isfinite(float(lam)) or float(lam) < 0:
            raise ValueError("lam must be finite and non-negative")

        event_hits = self.retrieve_events(
            raw_events,
            top_k_per_event=top_k_per_event,
            candidate_videos=candidate_videos,
        )
        fused = self._fuse_event_hits(normalized, event_hits)
        if candidate_videos is not None:
            allowed = set(map(str, candidate_videos))
            fused = {video_id: value for video_id, value in fused.items() if video_id in allowed}

        policies = self._policies_for_events(raw_events, normalized)
        coverage_by_video: dict[str, dict[str, Any]] = {}
        for video_id, frame_map in fused.items():
            event_coverage: dict[str, bool] = {}
            required_modality_coverage: dict[str, bool] = {}
            for event in normalized:
                candidates = [
                    candidate
                    for candidate in frame_map.values()
                    if event.index in candidate
                ]
                event_coverage[str(event.index)] = bool(candidates)
                required = set(policies[event.index].required)
                required_modality_coverage[str(event.index)] = all(
                    any(modality in candidate[event.index].modality_scores for candidate in candidates)
                    for modality in required
                )
            coverage_by_video[video_id] = {
                "event_coverage": event_coverage,
                "required_modality_coverage": required_modality_coverage,
                "coverage": sum(event_coverage.values()) / max(1, len(normalized)),
            }

        complete_videos = {
            video_id: frame_map
            for video_id, frame_map in fused.items()
            if coverage_by_video[video_id]["coverage"] >= 1.0
            and (
                self.alignment_policy == LEGACY_ALIGNMENT_POLICY
                or all(coverage_by_video[video_id]["required_modality_coverage"].values())
            )
        }
        event_score_pools: dict[int, list[float]] = {
            event.index: [
                self._candidate_base_score(candidate)
                for frame_map in fused.values()
                for candidate_map in frame_map.values()
                for candidate in [candidate_map.get(event.index)]
                if candidate is not None
            ]
            for event in normalized
        }
        scored: list[dict[str, Any]] = []
        incomplete_coverage: dict[str, float] = {}
        for video_id, frame_map in sorted(complete_videos.items()):
            def timeline_key(key: tuple[str, int]) -> tuple[Any, ...]:
                candidates = list(frame_map[key].values())
                first = next(item for item in candidates if item.evidence)
                evidence = first.evidence[0]
                kf_n = next(
                    (item.evidence[0].kf_n for item in candidates if item.evidence and item.evidence[0].kf_n is not None),
                    -1,
                )
                return (int(key[1]), float(evidence.pts_time), int(kf_n))

            timeline = sorted(frame_map, key=timeline_key)
            if len(timeline) < len(normalized):
                continue
            index_by_key = {key: index for index, key in enumerate(timeline)}
            raw_scores = np.full(
                (len(normalized), len(timeline)), self.missing_score, dtype=np.float32
            )
            for event in normalized:
                for key in timeline:
                    candidate = frame_map[key].get(event.index)
                    if candidate is not None:
                        raw_scores[event.index, index_by_key[key]] = candidate.fused_score

            if self.alignment_policy == COVERAGE_COHERENT_ALIGNMENT_POLICY:
                scores = np.full_like(raw_scores, self.missing_score)
                for event in normalized:
                    for key in timeline:
                        candidate = frame_map[key].get(event.index)
                        if candidate is not None:
                            scores[event.index, index_by_key[key]] = self._event_score_quality(
                                candidate,
                                event_score_pools[event.index],
                                candidate_value=self._candidate_base_score(candidate),
                            )
            else:
                scores = raw_scores

            dante_score, path = dante_align(scores, lam=float(lam))
            # Zero is a valid normalized score (the lowest candidate in a
            # pool); only the explicit sentinel denotes missing evidence.
            valid_mask = scores != self.missing_score
            if path is None or len(path) != len(normalized) or any(
                index < 0
                or index >= len(timeline)
                or not valid_mask[event.index, int(index)]
                for event, index in zip(normalized, path or ())
            ):
                dante_score, path = self._safe_dante_align(
                    scores,
                    valid_mask,
                    lam=float(lam),
                )
                self.last_diagnostics["dante_fallback_used"] = True
            if path is None or len(path) != len(normalized):
                continue
            sequence: list[dict[str, Any]] = []
            for event in normalized:
                key = timeline[int(path[event.index])]
                candidate = frame_map[key].get(event.index)
                if candidate is None or candidate.fused_score <= 0:
                    sequence = []
                    break
                best = max(
                    candidate.evidence,
                    key=lambda item: (
                        candidate.modality_scores.get(item.modality, 0.0),
                        float(item.score),
                        item.modality,
                        item.source_id,
                    ),
                )
                sequence.append(
                    {
                        "event_index": event.index,
                        "event_id": event.event_id,
                        "event_desc": event.description,
                        "video_id": video_id,
                        "modality": best.modality,
                        "sources": sorted(candidate.modality_scores),
                        "frame_idx": best.frame_idx,
                        "kf_n": best.kf_n,
                        "pts_time": best.pts_time,
                        "score": candidate.fused_score,
                        "modality_scores": dict(sorted(candidate.modality_scores.items())),
                        "evidence": [
                            {
                                "modality": item.modality,
                                "score": item.score,
                                "frame_idx": item.frame_idx,
                                "kf_n": item.kf_n,
                                "pts_time": item.pts_time,
                                "source_id": item.source_id,
                                "text": item.text,
                            }
                            for item in candidate.evidence
                        ],
                    }
                )
            if not sequence:
                continue
            try:
                validated = validate_sequence_path(sequence, normalized, video_id=video_id)
            except TrakeContractError:
                continue
            if self.alignment_policy == COVERAGE_COHERENT_ALIGNMENT_POLICY:
                final_score, policy_diagnostics = self._coverage_coherent_path_score(
                    events=normalized,
                    policies=policies,
                    event_score_pools=event_score_pools,
                    frame_map=frame_map,
                    timeline=timeline,
                    path=path,
                    dante_score=dante_score,
                )
                if policy_diagnostics["coverage"] < 1.0:
                    # Do not let an incomplete path reach the ranked output.
                    incomplete_coverage[video_id] = policy_diagnostics["coverage"]
                    continue
            else:
                final_score = float(dante_score)
                policy_diagnostics = {
                    "coverage": 1.0,
                    "support_coverage": None,
                    "score_coherence": None,
                    "dante_score": float(dante_score),
                }
            result_record = {
                "video_id": video_id,
                "score": float(final_score),
                "path": sequence,
                "frame_ids": [item.frame_idx for item in validated],
                "modalities": [item.modality for item in validated],
            }
            if self.alignment_policy == COVERAGE_COHERENT_ALIGNMENT_POLICY:
                result_record.update(
                    {
                        "alignment_score": float(dante_score),
                        "coverage": policy_diagnostics["coverage"],
                        "support_coverage": policy_diagnostics["support_coverage"],
                        "score_coherence": policy_diagnostics["score_coherence"],
                        "policy_diagnostics": policy_diagnostics,
                    }
                )
            scored.append(result_record)

        scored.sort(key=lambda item: (-item["score"], item["video_id"]))
        top_limit = min(int(top_k_videos), 100)
        diagnostics = self._alignment_diagnostics(fused, event_hits, normalized)
        diagnostics.update(
            {
                "candidate_video_count": len(complete_videos),
                "scored_video_count": len(scored),
                "lambda": float(lam),
                "top_k_limit": top_limit,
                "complete_paths_only": True,
                "canonical_order": "frame_idx",
                "alignment_policy": self.alignment_policy,
                "dante_fallback_used": bool(
                    self.last_diagnostics.get("dante_fallback_used", False)
                ),
                "incomplete_coverage": incomplete_coverage,
                "coverage_by_video": coverage_by_video,
            }
        )
        return {
            "results": scored[:top_limit],
            "event_hits": {str(index): len(hits) for index, hits in sorted(event_hits.items())},
            "diagnostics": diagnostics,
        }
