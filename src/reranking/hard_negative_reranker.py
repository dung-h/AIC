"""Deterministic hard-negative reranker.

This module is intentionally independent from model/index loading. It reranks a
candidate list using normalized numeric metadata features already attached by an
upstream retriever or diagnostic dump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.service.contracts import RetrievalResult, normalize_result


@dataclass(frozen=True)
class HardNegativeRerankerConfig:
    """Weights for second-stage candidate reranking.

    `base_score_weight` preserves the original retriever order/score. Feature
    weights read numeric values from `RetrievalResult.metadata` and add them to
    the rerank score after optional min-max normalization per query.
    """

    base_score_weight: float = 1.0
    feature_weights: Mapping[str, float] = field(default_factory=dict)
    normalize_features: bool = True


@dataclass(frozen=True)
class RerankedResult:
    result: RetrievalResult
    original_rank: int
    rerank_score: float
    feature_scores: dict[str, float]


class HardNegativeReranker:
    """Rerank top-k hard negatives using precomputed candidate features."""

    def __init__(self, config: HardNegativeRerankerConfig | None = None):
        self.config = config or HardNegativeRerankerConfig()

    def rerank(self, query: str, candidates: Sequence[Any], topk: int | None = None) -> list[RerankedResult]:
        """Return candidates sorted by rerank score descending.

        `query` is accepted for a stable future interface; this deterministic
        baseline only uses candidate metadata features.
        """
        del query
        normalized = [normalize_result(candidate) for candidate in candidates]
        if not normalized:
            return []

        feature_values = self._feature_values(normalized)
        reranked: list[RerankedResult] = []
        for idx, result in enumerate(normalized):
            original_rank = idx + 1
            # Stable tie-breaker favors the original rank without dominating the
            # explicit base score or feature weights.
            score = self.config.base_score_weight * float(result.score) - original_rank * 1e-9
            features: dict[str, float] = {}
            for name, weight in self.config.feature_weights.items():
                value = feature_values[name][idx]
                features[name] = value
                score += float(weight) * value
            reranked.append(RerankedResult(result=result, original_rank=original_rank,
                                          rerank_score=score, feature_scores=features))

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked[:topk] if topk is not None else reranked

    def _feature_values(self, candidates: Sequence[RetrievalResult]) -> dict[str, list[float]]:
        values: dict[str, list[float]] = {}
        for name in self.config.feature_weights:
            raw = [self._numeric_feature(candidate.metadata, name) for candidate in candidates]
            values[name] = self._minmax(raw) if self.config.normalize_features else raw
        return values

    @staticmethod
    def _numeric_feature(metadata: Mapping[str, Any], name: str) -> float:
        value = metadata.get(name, 0.0)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _minmax(values: Sequence[float]) -> list[float]:
        lo = min(values)
        hi = max(values)
        if hi - lo <= 1e-12:
            return [0.0 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]
