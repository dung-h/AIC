"""Shared candidate-lattice records and branch-union utilities.

This module is deliberately model- and index-free. Retrieval branches can
emit candidates with provenance, while task pipelines decide how to ground
or align them later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CandidateRecord:
    video_id: str
    kf_n: int
    frame_idx: int
    pts_time: float
    score: float
    source: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    @property
    def frame_key(self) -> tuple[str, int]:
        return self.video_id, self.kf_n

    @property
    def video_key(self) -> str:
        return self.video_id


def union_candidates(
    branches: Mapping[str, Iterable[CandidateRecord]],
    *,
    topk_per_branch: int | None = None,
    topk_union: int | None = None,
) -> list[CandidateRecord]:
    """Return a deterministic frame union without mixing raw branch scores.

    Candidates are deduplicated by ``(video_id, kf_n)``. The first retained
    record keeps its branch score; all contributing branches are preserved in
    ``evidence['sources']``. Branch ranking is local to each input branch.
    """
    merged: dict[tuple[str, int], CandidateRecord] = {}
    for source, records in branches.items():
        ranked = sorted(records, key=lambda item: (-item.score, item.video_id, item.kf_n))
        if topk_per_branch is not None:
            ranked = ranked[:topk_per_branch]
        for record in ranked:
            key = record.frame_key
            prior = merged.get(key)
            if prior is None:
                evidence = dict(record.evidence)
                evidence["sources"] = [source]
                merged[key] = CandidateRecord(
                    record.video_id, record.kf_n, record.frame_idx,
                    record.pts_time, record.score, source, evidence
                )
            else:
                sources = list(prior.evidence.get("sources", [prior.source]))
                if source not in sources:
                    sources.append(source)
                evidence = dict(prior.evidence)
                evidence["sources"] = sources
                merged[key] = CandidateRecord(
                    prior.video_id, prior.kf_n, prior.frame_idx,
                    prior.pts_time, prior.score, prior.source, evidence
                )
    result = sorted(merged.values(), key=lambda item: (-item.score, item.video_id, item.kf_n))
    return result if topk_union is None else result[:topk_union]
