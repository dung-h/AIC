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
  evidence/rank rescue gate;
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
        "channel", "modality", "source",
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
) -> list[dict]:
    """Fuse ranked channel hits with weighted reciprocal rank fusion.

    ``topk`` is both the output budget and the visual-anchor boundary used by
    the rescue policy.  All channels are unioned before scoring, but a
    specialist-only video is eligible to displace the visual boundary only if
    its configured rank/evidence gate passes.  This keeps weak ASR/OCR noise
    from evicting visual candidates while still allowing genuine rescue.

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
    if min_specialist_channels < 1:
        raise ValueError("min_specialist_channels must be positive")
    if require_specialist_evidence and not evidence_keys:
        raise ValueError("evidence_keys cannot be empty when evidence is required")

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

    visual_top_ids = set(visual_order[:topk])
    strong_rescue_ids: set[str] = set()
    for video_id, per_channel in specialist_ranks.items():
        if video_id in visual_top_ids:
            continue
        has_top_hit = any(rank <= specialist_strong_rank for rank in per_channel.values())
        supporting_channels = sum(
            rank <= specialist_support_rank for rank in per_channel.values()
        )
        if (
            (allow_single_strong_rescue and has_top_hit)
            or supporting_channels >= min_specialist_channels
        ):
            strong_rescue_ids.add(video_id)

    # Without an eligible rescue, retain the visual top-k exactly.  Specialist
    # evidence can still be present for a video already in that boundary, but
    # it must not reorder/demote the visual anchor on weak evidence.
    if not strong_rescue_ids:
        return _visual_baseline_rows(visual_ranks, visual_order, topk, rrf_k=rrf_k)

    allowed_ids = visual_top_ids | strong_rescue_ids
    return _fuse_unrestricted(
        materialized_channels,
        positive_channels,
        rrf_k=rrf_k,
        topk=topk,
        allowed_video_ids=allowed_ids,
        guard_mode="strong_specialist_rescue",
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
    return [
        {
            **metadata[video_id],
            "rrf_score": float(fused[video_id]),
            "video_rank": rank,
            "rrf_guard": guard_mode,
        }
        for rank, video_id in enumerate(order, 1)
    ]


__all__ = ["collapse_video_ranks", "weighted_video_rrf"]
