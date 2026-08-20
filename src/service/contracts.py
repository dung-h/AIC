"""Stable contracts between retrieval components and the service boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class RetrievalResult:
    video_id: str
    frame_idx: int
    kf_n: int = 0
    pts_time: float = 0.0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRequest:
    topk: int = 20
    mode: str = "default"


class TextRetriever(Protocol):
    def search(self, query: str, topk: int = 20) -> Sequence[Any]: ...


class ClipRetriever(Protocol):
    def search_clip(self, path: str, topk: int = 10, agg: str = "hybrid0.5") -> Sequence[Any]: ...


def normalize_result(raw: Any, *, metadata_lookup=None) -> RetrievalResult:
    """Convert current tuple/dict results into the stable boundary contract."""
    if isinstance(raw, RetrievalResult):
        if metadata_lookup is not None:
            kf_n, pts_time = metadata_lookup(raw.video_id, raw.frame_idx)
            if raw.kf_n not in (0, int(kf_n)):
                raise ValueError(
                    "retrieval result kf_n disagrees with the canonical map: "
                    f"video_id={raw.video_id!r}, frame_idx={raw.frame_idx!r}"
                )
            if raw.pts_time and abs(float(raw.pts_time) - float(pts_time)) > 1e-6:
                raise ValueError(
                    "retrieval result pts_time disagrees with the canonical map: "
                    f"video_id={raw.video_id!r}, frame_idx={raw.frame_idx!r}"
                )
            return RetrievalResult(video_id=raw.video_id, frame_idx=raw.frame_idx,
                                   kf_n=int(kf_n), pts_time=float(pts_time),
                                   score=raw.score, metadata=raw.metadata)
        return raw
    if isinstance(raw, dict):
        video_id, frame_idx = str(raw["video_id"]), int(raw["frame_idx"])
        if metadata_lookup is not None:
            kf_n, pts_time = metadata_lookup(video_id, frame_idx)
            if "kf_n" in raw and int(raw["kf_n"]) != int(kf_n):
                raise ValueError(
                    "retrieval result kf_n disagrees with the canonical map: "
                    f"video_id={video_id!r}, frame_idx={frame_idx!r}"
                )
            if "pts_time" in raw and abs(float(raw["pts_time"]) - float(pts_time)) > 1e-6:
                raise ValueError(
                    "retrieval result pts_time disagrees with the canonical map: "
                    f"video_id={video_id!r}, frame_idx={frame_idx!r}"
                )
        else:
            # Internal rerankers may only carry video/frame/score metadata.
            # Canonical production boundaries still pass ``metadata_lookup``;
            # use a neutral timestamp for provider-neutral diagnostics rather
            # than rejecting an otherwise valid candidate.
            kf_n, pts_time = int(raw.get("kf_n", 0)), float(raw.get("pts_time", 0.0))
        return RetrievalResult(video_id=video_id, frame_idx=frame_idx,
                               kf_n=int(kf_n), pts_time=float(pts_time),
                               score=float(raw.get("score", 0.0)),
                               metadata=dict(raw.get("metadata", {})))
    if not isinstance(raw, (tuple, list)) or len(raw) < 3:
        raise TypeError(f"unsupported retrieval result: {type(raw)!r}")
    video_id, frame_idx = str(raw[0]), int(raw[1])
    if len(raw) == 3:
        pts_time, score, kf_n = float(raw[2]), 0.0, 0
    elif len(raw) >= 4:
        third, fourth = raw[2], raw[3]
        # KIS tuple: (video, frame_idx, kf_n, score); VKIS tuple:
        # (video, frame_idx, pts_time, score). The adapter is explicit at call
        # sites through metadata_lookup, avoiding tuple knowledge in the API.
        if metadata_lookup is not None:
            kf_n, pts_time = metadata_lookup(video_id, frame_idx)
        else:
            kf_n, pts_time = int(third), 0.0
        score = float(fourth)
        if metadata_lookup is None and isinstance(third, float):
            pts_time, kf_n = float(third), 0
    else:
        kf_n, pts_time, score = 0, 0.0, 0.0
    return RetrievalResult(video_id=video_id, frame_idx=frame_idx, kf_n=kf_n,
                           pts_time=pts_time, score=score)
