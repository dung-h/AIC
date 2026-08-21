"""Deterministic video-level rank fusion.

This module is the single owner for the last step of modality retrieval:
frame/evidence hits are collapsed to one rank per video and then fused with
weighted reciprocal rank fusion (RRF).  The public functions intentionally
remain small and compatible with the original ``dict``/``tuple`` API.

Important semantics:

* a video's channel rank is the best (lowest) rank observed in that channel;
* duplicate frame/evidence hits do not create extra RRF votes;
* a missing or empty ASR/OCR channel contributes no term and no penalty;
* specialist-only videos can enter the visual top-k only through the explicit
  evidence/rank rescue gate, including a bounded lexical/quote-evidence path;
* all ties have a deterministic video-id tie-break.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from math import isfinite
from typing import TypeAlias


Candidate: TypeAlias = Mapping[str, object] | Sequence[object] | object


def _object_to_mapping(candidate: Candidate, *, video_key: str) -> dict[str, object]:
    """Convert a mapping, legacy tuple, or EvidenceHit-like object to a dict."""
    if isinstance(candidate, Mapping):
        return {str(key): value for key, value in candidate.items()}

    # Named tuples are both tuple-compatible and provide a lossless mapping.
    as_dict = getattr(candidate, "_asdict", None)
    if callable(as_dict):
        return {str(key): value for key, value in as_dict().items()}

    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        if len(candidate) < 2:
            raise ValueError("candidate tuple must contain video_id and frame data")
        payload: dict[str, object] = {
            "video_id": candidate[0],
            "frame_idx": candidate[1],
        }
        if len(candidate) >= 3:
            payload["kf_n"] = candidate[2]
        if len(candidate) >= 4:
            payload["score"] = candidate[3]
        return payload

    if is_dataclass(candidate) and not isinstance(candidate, type):
        return {field.name: getattr(candidate, field.name) for field in fields(candidate)}

    # A light structural fallback also supports slot-based EvidenceHit-like
    # objects without requiring an import from another agent's module.
    payload = {}
    for name in (
        video_key, "video_id", "frame_idx", "kf_n", "pts_time", "score", "rank",
        "evidence", "text", "asr_text", "ocr_text", "transcript", "metadata",
        "channel", "modality", "source", "view_provenance", "lexical_provenance",
        "quote_provenance", "retrieval_provenance", "match_type", "evidence_type",
    ):
        if hasattr(candidate, name):
            payload[name] = getattr(candidate, name)
    if not payload:
        raise TypeError(f"unsupported candidate type: {type(candidate).__name__}")
    return payload


def _as_video_id(payload: Mapping[str, object], video_key: str) -> str:
    value = payload.get(video_key)
    if value is None and video_key != "video_id":
        value = payload.get("video_id")
    if value is None:
        raise KeyError(f"candidate is missing {video_key!r}")
    video_id = str(value).strip()
    if not video_id:
        raise ValueError("candidate video_id must be non-empty")
    return video_id


def _positive_rank(value: object) -> int | None:
    try:
        rank = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(rank) or rank < 1 or not rank.is_integer():
        return None
    return int(rank)


def _normalise_candidate(
    candidate: Candidate,
    stream_position: int,
    *,
    video_key: str,
) -> tuple[str, dict[str, object], int]:
    payload = _object_to_mapping(candidate, video_key=video_key)
    video_id = _as_video_id(payload, video_key)
    # ``rank`` is honored when a channel already provides one.  Raw frame
    # streams normally omit it, in which case their 1-based stream position
    # is the channel rank.  Invalid explicit ranks safely fall back to order.
    rank = _positive_rank(payload.get("rank")) or stream_position
    normalised = dict(payload)
    normalised["video_id"] = video_id
    normalised["rank"] = rank
    return video_id, normalised, stream_position


def collapse_video_ranks(
    candidates: Iterable[Candidate],
    *,
    video_key: str = "video_id",
) -> dict[str, dict]:
    """Collapse ranked frame/evidence hits to the best hit for each video.

    The returned mapping preserves the first-seen video order for compatibility
    with callers that inspect it directly.  For duplicate hits, the lowest
    explicit/source rank wins; equal-rank duplicates retain the first input
    hit, making the operation deterministic without inventing a second vote.
    """
    result: dict[str, dict] = {}
    best_keys: dict[str, tuple[int, int]] = {}
    for stream_position, candidate in enumerate(candidates, 1):
        video_id, payload, source_position = _normalise_candidate(
            candidate, stream_position, video_key=video_key,
        )
        key = (int(payload["rank"]), source_position)
        if video_id not in result or key < best_keys[video_id]:
            result[video_id] = payload
            best_keys[video_id] = key
    return result


def weighted_video_rrf(
    channels: Mapping[str, Iterable[Candidate]],
    weights: Mapping[str, float],
    *,
    rrf_k: int = 60,
    topk: int = 20,
    visual_channel: str = "visual",
    specialist_strong_rank: int = 5,
    specialist_support_rank: int = 20,
    min_specialist_channels: int = 2,
    specialist_min_scores: Mapping[str, float] | None = None,
    require_specialist_evidence: bool = False,
    evidence_keys: Sequence[str] = (
        "evidence", "text", "asr_text", "ocr_text", "transcript",
    ),
    specialist_rescue_enabled: bool = True,
    allow_single_strong_rescue: bool = True,
    evidence_aware_rescue_enabled: bool = True,
    evidence_aware_max_rank: int = 20,
    provenance_max_rank: int = 3,
    provenance_min_score: float = 0.8,
    provenance_modes: Sequence[str] = (
        "bm25_coverage", "lexical_exact", "exact", "exact_match",
        "quote", "quoted_fact",
    ),
) -> list[dict]:
    """Fuse ranked channel hits with weighted reciprocal rank fusion.

    ``topk`` is both the output budget and the visual-anchor boundary used by
    the rescue policy.  All channels are unioned before scoring, but a
    specialist-only video is eligible to displace the visual boundary only if
    its configured rank/evidence gate passes. This keeps weak ASR/OCR noise
    from evicting visual candidates while still allowing a lower-ranked,
    explicitly grounded lexical/quote hit to rescue.

    A channel that is absent, empty, or has a non-positive weight contributes
    nothing.  It is never represented as a zero score or a penalty, which is
    the required coverage-aware behavior for partial ASR/OCR indexes.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if topk <= 0:
        return []
    if specialist_strong_rank < 1 or specialist_support_rank < 1:
        raise ValueError("specialist rank thresholds must be positive")
    if evidence_aware_max_rank < 1 or provenance_max_rank < 1:
        raise ValueError("evidence-aware provenance ranks must be positive")
    if min_specialist_channels < 1:
        raise ValueError("min_specialist_channels must be positive")
    if require_specialist_evidence and not evidence_keys:
        raise ValueError("evidence_keys cannot be empty when evidence is required")
    try:
        provenance_min_score = float(provenance_min_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance_min_score must be numeric") from exc
    if not isfinite(provenance_min_score):
        raise ValueError("provenance_min_score must be finite")
    normalized_provenance_modes = tuple(
        dict.fromkeys(
            str(mode).strip().lower() for mode in provenance_modes if str(mode).strip()
        )
    )
    if evidence_aware_rescue_enabled and not normalized_provenance_modes:
        raise ValueError("provenance_modes cannot be empty when evidence-aware rescue is enabled")

    # Normalize channel names once.  This makes mappings with non-string keys
    # behave the same as the public string-key API and makes channel iteration
    # deterministic for custom Mapping implementations.
    materialized_channels: dict[str, list[Candidate]] = {
        str(channel): list(candidates)
        for channel, candidates in channels.items()
    }
    positive_channels = {
        str(channel): float(weight)
        for channel, weight in weights.items()
        if _valid_positive_weight(weight)
    }
    min_scores = {
        str(channel): float(score)
        for channel, score in (specialist_min_scores or {}).items()
    }
    visual_name = str(visual_channel)
    visual_candidates = materialized_channels.get(visual_name, [])
    visual_ranks = collapse_video_ranks(visual_candidates)
    visual_order = _rank_order(visual_ranks)

    specialist_names = [
        channel
        for channel in materialized_channels
        if channel != visual_name and positive_channels.get(channel, 0.0) > 0
    ]
    # Keep the legacy visual-only path byte-for-byte in ranking terms and make
    # absent/empty specialist channels a no-op rather than negative evidence.
    if not specialist_names or not specialist_rescue_enabled:
        return _fuse_unrestricted(
            materialized_channels,
            positive_channels,
            rrf_k=rrf_k,
            topk=topk,
        )

    specialist_ranks: dict[str, dict[str, int]] = {}
    evidence_aware_hits: dict[str, dict[str, str]] = {}
    for channel in specialist_names:
        collapsed = collapse_video_ranks(materialized_channels[channel])
        for video_id, item in collapsed.items():
            if _specialist_evidence_eligible(
                item,
                channel,
                min_scores=min_scores,
                require_evidence=require_specialist_evidence,
                evidence_keys=evidence_keys,
            ):
                specialist_ranks.setdefault(video_id, {})[channel] = int(item["rank"])
                if (
                    evidence_aware_rescue_enabled
                    and int(item["rank"]) <= int(evidence_aware_max_rank)
                ):
                    reason = _evidence_aware_rescue_reason(
                        item,
                        provenance_max_rank=int(provenance_max_rank),
                        provenance_min_score=provenance_min_score,
                        provenance_modes=normalized_provenance_modes,
                    )
                    if reason is not None:
                        evidence_aware_hits.setdefault(video_id, {})[channel] = reason

    visual_top_ids = set(visual_order[:topk])
    rescue_reasons: dict[str, str] = {}
    # A specialist vote may reorder a video that the visual channel already
    # admitted, but only after it clears the same bounded evidence gate used
    # for a rescue.  Previously the visual guard returned the raw visual order
    # whenever there was no *new* specialist-only video; ASR/OCR therefore had
    # no ranking effect at all for visual top-k candidates.
    specialist_scoring_ids: set[str] = set()
    for video_id, per_channel in specialist_ranks.items():
        has_top_hit = any(rank <= specialist_strong_rank for rank in per_channel.values())
        supporting_channels = sum(
            rank <= specialist_support_rank for rank in per_channel.values()
        )
        evidence_reasons = evidence_aware_hits.get(video_id, {})
        specialist_gate_passed = (
            (allow_single_strong_rescue and has_top_hit)
            or supporting_channels >= min_specialist_channels
            or bool(evidence_reasons)
        )
        if specialist_gate_passed:
            specialist_scoring_ids.add(video_id)
        if video_id in visual_top_ids:
            continue
        if (
            (allow_single_strong_rescue and has_top_hit)
            or supporting_channels >= min_specialist_channels
        ):
            rescue_reasons[video_id] = "strong_rank"
            continue
        if evidence_reasons:
            # One explicit, locally-grounded rare fact/quote is enough to
            # rescue. The provenance gate above makes this stricter than a raw
            # non-empty transcript/OCR string.
            channel, reason = sorted(
                evidence_reasons.items(), key=lambda item: _channel_sort_key(item[0])
            )[0]
            rescue_reasons[video_id] = f"evidence_aware:{channel}:{reason}"

    allowed_ids = visual_top_ids | set(rescue_reasons)
    # Keep weak/non-provenanced specialist rows out of the RRF scorer even
    # when their video is already visual-admitted. This separates inclusion
    # (visual top-k plus gated rescues) from ranking (weighted RRF on trusted
    # votes) and prevents incidental transcript/OCR text from perturbing the
    # visual baseline.
    scoring_channels = {visual_name: visual_candidates}
    for channel in specialist_names:
        trusted_rows = []
        # Preserve the channel-global rank before filtering.  Re-enumerating
        # only trusted rows would turn a source rank of 16 into rank 1 and
        # falsely amplify a late rescue in RRF.
        for stream_position, row in enumerate(materialized_channels[channel], 1):
            payload = _object_to_mapping(row, video_key="video_id")
            video_id = str(payload.get("video_id", "")).strip()
            if video_id not in specialist_scoring_ids:
                continue
            payload["rank"] = _positive_rank(payload.get("rank")) or stream_position
            trusted_rows.append(payload)
        scoring_channels[channel] = trusted_rows
    guard_mode = (
        "strong_specialist_rescue"
        if rescue_reasons and all(reason == "strong_rank" for reason in rescue_reasons.values())
        else "evidence_aware_specialist_rescue"
    )
    if not rescue_reasons:
        guard_mode = "visual_boundary_rrf"
    return _fuse_unrestricted(
        scoring_channels,
        positive_channels,
        rrf_k=rrf_k,
        topk=topk,
        allowed_video_ids=allowed_ids,
        guard_mode=guard_mode,
        rescue_reasons=rescue_reasons,
    )


def _valid_positive_weight(value: object) -> bool:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(weight) and weight > 0


def _specialist_evidence_eligible(
    candidate: Mapping[str, object],
    channel: str,
    *,
    min_scores: Mapping[str, float],
    require_evidence: bool,
    evidence_keys: Sequence[str],
) -> bool:
    """Apply explicit score/evidence gates before a specialist can rescue."""
    if channel in min_scores:
        raw_score = candidate.get("score")
        try:
            if raw_score is None or float(raw_score) < float(min_scores[channel]):
                return False
        except (TypeError, ValueError):
            return False
    if not require_evidence:
        return True
    metadata = candidate.get("metadata")
    for key in evidence_keys:
        value = candidate.get(str(key))
        if value is None and isinstance(metadata, Mapping):
            value = metadata.get(str(key))
        if _has_nonempty_evidence(value):
            return True
    return False


def _has_nonempty_evidence(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_has_nonempty_evidence(item) for item in value)
    return True


def _evidence_aware_rescue_reason(
    candidate: Mapping[str, object],
    *,
    provenance_max_rank: int,
    provenance_min_score: float,
    provenance_modes: Sequence[str],
) -> str | None:
    """Return a traceable reason for a high-confidence lexical/quote hit.

    ``QNAModalityRouter.global_candidates_multi`` emits ``view_provenance``
    records such as ``bm25_coverage``. We require a recognized mode, a
    top-ranked lexical/quote view, and a bounded quality score. Explicit
    exact/quote modes may omit a numeric score because their match type is the
    provenance assertion.
    """
    for provenance in _iter_provenance_records(candidate):
        mode = str(
            provenance.get("score_mode", provenance.get("mode", provenance.get("match_type", "")))
        ).strip().lower()
        if mode not in provenance_modes:
            continue
        rank = _positive_rank(provenance.get("rank"))
        if rank is None or rank > provenance_max_rank:
            continue
        if _mode_is_explicit_fact(mode):
            return f"{mode}:rank_{rank}"
        raw_score = provenance.get("score", provenance.get("coverage"))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if isfinite(score) and score >= provenance_min_score:
            return f"{mode}:rank_{rank}:score_{score:.3f}"
    return None


def _iter_provenance_records(candidate: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    """Yield deterministic provenance mappings from a hit and its metadata."""
    sources: list[object] = [
        candidate.get("view_provenance"),
        candidate.get("lexical_provenance"),
        candidate.get("quote_provenance"),
        candidate.get("retrieval_provenance"),
    ]
    metadata = candidate.get("metadata")
    if isinstance(metadata, Mapping):
        sources.extend(
            metadata.get(key) for key in (
                "view_provenance", "lexical_provenance", "quote_provenance",
                "retrieval_provenance",
            )
        )
    for source in sources:
        if isinstance(source, Mapping):
            yield source
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            for item in source:
                if isinstance(item, Mapping):
                    yield item


def _mode_is_explicit_fact(mode: str) -> bool:
    return mode in {"lexical_exact", "exact", "exact_match", "quote", "quoted_fact"}


def _visual_baseline_rows(
    visual_ranks: dict[str, dict],
    visual_order: list[str],
    topk: int,
    *,
    rrf_k: int,
) -> list[dict]:
    """Materialize the visual-only ranking without specialist contributions."""
    output = []
    for rank, video_id in enumerate(visual_order[:topk], 1):
        item = dict(visual_ranks[video_id])
        item["video_rank"] = rank
        item["rrf_score"] = 1.0 / (int(item["rank"]) + rrf_k)
        item["rrf_guard"] = "visual_baseline"
        item["rrf_guard_reason"] = "no_eligible_specialist_rescue"
        output.append(item)
    return output


def _rank_order(ranks: Mapping[str, Mapping[str, object]]) -> list[str]:
    """Return one deterministic order for collapsed channel ranks."""
    return sorted(
        ranks,
        key=lambda video_id: (
            int(ranks[video_id]["rank"]),
            str(video_id),
        ),
    )


def _channel_sort_key(channel: str) -> tuple[int, str]:
    # Stable channel precedence is useful when callers pass an unordered
    # Mapping and two candidates otherwise have identical scores.  The final
    # video-id tie-break remains the public ordering rule.
    preferred = {"visual": 0, "asr": 1, "ocr": 2}
    return preferred.get(channel, 3), channel


def _fuse_unrestricted(
    channels: Mapping[str, Iterable[Candidate]],
    weights: Mapping[str, float],
    *,
    rrf_k: int,
    topk: int,
    allowed_video_ids: set[str] | None = None,
    guard_mode: str = "none",
    rescue_reasons: Mapping[str, str] | None = None,
) -> list[dict]:
    """Compute RRF over the channel union after any rescue filter."""
    fused: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    channel_names = sorted((str(channel) for channel in channels), key=_channel_sort_key)
    for channel in channel_names:
        weight = float(weights.get(channel, 0.0))
        if weight <= 0 or not isfinite(weight):
            continue
        ranks = collapse_video_ranks(channels[channel])
        for video_id, item in ranks.items():
            if allowed_video_ids is not None and video_id not in allowed_video_ids:
                continue
            fused[video_id] = fused.get(video_id, 0.0) + weight / (int(item["rank"]) + rrf_k)
            metadata.setdefault(video_id, {"video_id": video_id})[
                f"{channel}_rank"
            ] = int(item["rank"])
            metadata[video_id][f"{channel}_candidate"] = dict(item)

    # ``video_id`` is the final total-order tie-break, so identical RRF scores
    # never depend on source dict order or Python hash iteration order.
    order = sorted(
        fused,
        key=lambda video_id: (-fused[video_id], str(video_id)),
    )[:topk]
    output = []
    for rank, video_id in enumerate(order, 1):
        row = {
            **metadata[video_id],
            "rrf_score": float(fused[video_id]),
            "video_rank": rank,
            "rrf_guard": guard_mode,
        }
        if rescue_reasons is not None:
            row["rrf_guard_reason"] = rescue_reasons.get(video_id, "visual_anchor")
        output.append(row)
    return output


__all__ = ["collapse_video_ranks", "weighted_video_rrf"]
