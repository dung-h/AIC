"""Stable, model-agnostic contracts for retrieval and multimodal reasoning.

The contracts deliberately contain no imports from torch, faiss, HTTP clients,
or vendor SDKs.  Concrete providers live under :mod:`src.providers` and are
responsible for adapting their native output into these records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence


class ContractError(ValueError):
    """Base error for malformed records or incompatible components."""


class CompatibilityError(ContractError):
    """Raised when an encoder and an index cannot be used together."""


class ProviderError(RuntimeError):
    """Base error raised by a provider adapter."""


class ProviderConfigurationError(ProviderError):
    """Raised for invalid provider configuration or missing credentials."""


class ProviderUnavailableError(ProviderError):
    """Raised when an optional local dependency is not installed."""


class ProviderRequestError(ProviderError):
    """Raised when a remote provider cannot complete a request."""


_METRICS = frozenset({"inner_product", "cosine", "l2", "none"})
_NORMALIZATIONS = frozenset({"l2", "none", "custom"})
_PLACEHOLDER_ANSWERS = frozenset(
    {
        "",
        "null",
        "none",
        "n/a",
        "na",
        "unknown",
        "cannot determine",
        "cannot be determined",
        "i don't know",
        "unavailable",
        "evidence-only",
        "evidence only",
        "không tìm thấy",
        "khong tim thay",
    }
)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must contain numeric values") from exc
    if not math.isfinite(result):
        raise ContractError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class ModelMetadata:
    """Identity and vector-space contract for every provider.

    Non-vector providers use ``dim=0``, ``metric='none'`` and
    ``normalization='none'`` explicitly.  This avoids hidden ``None`` values
    and makes compatibility checks deterministic.
    """

    model_id: str
    modality: str
    dim: int
    metric: str
    normalization: str
    version: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.model_id, "model_id")
        _required_text(self.modality, "modality")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim < 0:
            raise ContractError("dim must be a non-negative integer")
        if self.metric not in _METRICS:
            raise ContractError(f"unsupported metric: {self.metric!r}")
        if self.normalization not in _NORMALIZATIONS:
            raise ContractError(f"unsupported normalization: {self.normalization!r}")
        if self.dim == 0 and (self.metric != "none" or self.normalization != "none"):
            raise ContractError("non-vector providers must use metric='none' and normalization='none'")
        if self.dim > 0 and self.metric == "none":
            raise ContractError("vector providers must declare a non-'none' metric")
        if self.version is not None:
            _required_text(self.version, "version")

    @property
    def is_vector(self) -> bool:
        return self.dim > 0

    def assert_compatible(self, other: "ModelMetadata", *, context: str = "") -> None:
        """Require an exact vector-space match; no implicit projection/fallback."""

        if not isinstance(other, ModelMetadata):
            raise CompatibilityError(f"{context}metadata must be ModelMetadata")
        fields = ("model_id", "modality", "dim", "metric", "normalization", "version")
        mismatches = [
            f"{name}: expected {getattr(self, name)!r}, got {getattr(other, name)!r}"
            for name in fields
            if getattr(self, name) != getattr(other, name)
        ]
        if mismatches:
            prefix = f"{context}: " if context else ""
            raise CompatibilityError(prefix + "incompatible model/index metadata (" + "; ".join(mismatches) + ")")


@dataclass(frozen=True)
class IndexMetadata:
    """Persisted index identity plus the encoder space that created it."""

    index_id: str
    encoder: ModelMetadata
    size: int
    row_key: str = "row_id"
    corpus_hash: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.index_id, "index_id")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ContractError("index size must be a non-negative integer")
        _required_text(self.row_key, "row_key")
        if self.corpus_hash is not None:
            _required_text(self.corpus_hash, "corpus_hash")

    @property
    def model_id(self) -> str:
        return self.encoder.model_id

    @property
    def modality(self) -> str:
        return self.encoder.modality

    @property
    def dim(self) -> int:
        return self.encoder.dim

    @property
    def metric(self) -> str:
        return self.encoder.metric

    @property
    def normalization(self) -> str:
        return self.encoder.normalization


@dataclass(frozen=True)
class EmbeddingBatch:
    """Validated output of an :class:`Encoder`."""

    vectors: Any
    metadata: ModelMetadata

    def __post_init__(self) -> None:
        if not self.metadata.is_vector:
            raise ContractError("EmbeddingBatch requires vector metadata with dim > 0")
        try:
            row_count = len(self.vectors)
        except TypeError as exc:
            raise ContractError("vectors must be a batch-like sequence") from exc
        for row_index in range(row_count):
            try:
                row = self.vectors[row_index]
                row_dim = len(row)
            except (TypeError, IndexError) as exc:
                raise ContractError(f"vectors[{row_index}] must be a sequence") from exc
            if row_dim != self.metadata.dim:
                raise ContractError(
                    f"vectors[{row_index}] has dim {row_dim}, expected {self.metadata.dim}"
                )
            for col_index, value in enumerate(row):
                _validate_float(value, f"vectors[{row_index}][{col_index}]")

    @property
    def size(self) -> int:
        return len(self.vectors)


@dataclass(frozen=True)
class VectorQuery:
    vector: Sequence[float]
    metadata: ModelMetadata

    def __post_init__(self) -> None:
        if not self.metadata.is_vector:
            raise ContractError("VectorQuery requires vector metadata with dim > 0")
        try:
            vector_dim = len(self.vector)
        except TypeError as exc:
            raise ContractError("query vector must be a sequence") from exc
        if vector_dim != self.metadata.dim:
            raise ContractError(f"query vector has dim {vector_dim}, expected {self.metadata.dim}")
        for index, value in enumerate(self.vector):
            _validate_float(value, f"query vector[{index}]")


@dataclass(frozen=True)
class SearchHit:
    row_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        _required_text(self.row_id, "row_id")
        _validate_float(self.score, "score")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ContractError("rank must be a positive integer")


@dataclass(frozen=True)
class Candidate:
    """Model-independent candidate passed to a reranker/answerer."""

    candidate_id: str
    video_id: str
    frame_id: int
    payload: Any = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _required_text(self.video_id, "video_id")
        if not isinstance(self.frame_id, int) or isinstance(self.frame_id, bool) or self.frame_id < 0:
            raise ContractError("frame_id must be a non-negative canonical frame index")
        if not isinstance(self.evidence, Mapping):
            raise ContractError("evidence must be a mapping")


@dataclass(frozen=True)
class RerankRequest:
    query: str
    candidates: Sequence[Candidate]

    def __post_init__(self) -> None:
        _required_text(self.query, "query")
        if not self.candidates:
            raise ContractError("rerank request requires at least one candidate")


@dataclass(frozen=True)
class GroundingRecord:
    candidate_id: str
    grounding_score: float
    abstain: bool = False
    provider: str = ""
    model_id: str = ""

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _validate_float(self.grounding_score, "grounding_score")
        if not 0.0 <= self.grounding_score <= 1.0:
            raise ContractError("grounding_score must be in [0, 1]")
        if not isinstance(self.abstain, bool):
            raise ContractError("abstain must be bool")


@dataclass(frozen=True)
class AnswerRequest:
    query: str
    question: str
    candidates: Sequence[Candidate]

    def __post_init__(self) -> None:
        _required_text(self.query, "query")
        _required_text(self.question, "question")
        if not self.candidates:
            raise ContractError("answer request requires at least one candidate")


@dataclass(frozen=True)
class AnswerRecord:
    candidate_id: str
    answer: str | None
    grounding_score: float
    answer_confidence: float
    abstain: bool = False
    provider: str = ""
    model_id: str = ""

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        for name, value in (
            ("grounding_score", self.grounding_score),
            ("answer_confidence", self.answer_confidence),
        ):
            _validate_float(value, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ContractError(f"{name} must be in [0, 1]")
        if not isinstance(self.abstain, bool):
            raise ContractError("abstain must be bool")
        if self.answer is not None and not isinstance(self.answer, str):
            raise ContractError("answer must be a string or None")
        normalized = self.answer.strip().casefold() if isinstance(self.answer, str) else ""
        if self.abstain:
            if self.answer is not None and normalized in _PLACEHOLDER_ANSWERS:
                object.__setattr__(self, "answer", None)
        elif normalized in _PLACEHOLDER_ANSWERS:
            raise ContractError("non-abstaining answer must not be empty or a placeholder")


class Encoder(ABC):
    """Batch encoder contract shared by local and remote implementations."""

    metadata: ModelMetadata

    @abstractmethod
    def encode(self, inputs: Sequence[Any]) -> EmbeddingBatch:
        raise NotImplementedError

    def encode_one(self, value: Any) -> EmbeddingBatch:
        batch = self.encode([value])
        if batch.size != 1:
            raise ContractError(f"encode_one expected exactly one embedding, got {batch.size}")
        return batch


class VectorSearcher(ABC):
    """ANN search contract; implementations must validate encoder compatibility."""

    index_metadata: IndexMetadata

    @property
    def metadata(self) -> ModelMetadata:
        return self.index_metadata.encoder

    def validate_query_encoder(self, encoder_metadata: ModelMetadata) -> None:
        self.metadata.assert_compatible(encoder_metadata, context="vector search")

    @abstractmethod
    def search(self, query: VectorQuery, top_k: int = 100) -> tuple[SearchHit, ...]:
        raise NotImplementedError


class Reranker(ABC):
    metadata: ModelMetadata

    @abstractmethod
    def rerank(self, request: RerankRequest) -> tuple[GroundingRecord, ...]:
        raise NotImplementedError


class Answerer(ABC):
    metadata: ModelMetadata

    @abstractmethod
    def answer(self, request: AnswerRequest) -> AnswerRecord:
        raise NotImplementedError


def normalize_grounding_records(
    raw: Any,
    *,
    provider: str,
    model_id: str,
    candidate_ids: set[str],
) -> tuple[GroundingRecord, ...]:
    """Convert provider-native reranker output and enforce candidate integrity."""

    if isinstance(raw, Mapping):
        raw = raw.get("results", raw.get("records", raw))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ContractError("reranker output must be a sequence of records")
    records: list[GroundingRecord] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, GroundingRecord):
            record = item
        elif isinstance(item, Mapping):
            record = GroundingRecord(
                candidate_id=str(item.get("candidate_id", "")),
                grounding_score=item.get("grounding_score", item.get("score", 0.0)),
                abstain=bool(item.get("abstain", False)),
                provider=provider,
                model_id=model_id,
            )
        else:
            raise ContractError("reranker records must be mappings or GroundingRecord")
        if record.candidate_id not in candidate_ids:
            raise ContractError(f"reranker returned unknown candidate_id: {record.candidate_id!r}")
        if record.candidate_id in seen:
            raise ContractError(f"reranker returned duplicate candidate_id: {record.candidate_id!r}")
        seen.add(record.candidate_id)
        records.append(
            GroundingRecord(
                candidate_id=record.candidate_id,
                grounding_score=record.grounding_score,
                abstain=record.abstain,
                provider=provider,
                model_id=model_id,
            )
        )
    return tuple(records)


def normalize_answer_record(raw: Any, *, provider: str, model_id: str) -> AnswerRecord:
    """Convert one provider-native answer into the strict internal record."""

    if isinstance(raw, AnswerRecord):
        return AnswerRecord(
            candidate_id=raw.candidate_id,
            answer=raw.answer,
            grounding_score=raw.grounding_score,
            answer_confidence=raw.answer_confidence,
            abstain=raw.abstain,
            provider=provider,
            model_id=model_id,
        )
    if not isinstance(raw, Mapping):
        raise ContractError("answerer output must be a mapping or AnswerRecord")
    return AnswerRecord(
        candidate_id=str(raw.get("candidate_id", "")),
        answer=raw.get("answer"),
        grounding_score=raw.get("grounding_score", 0.0),
        answer_confidence=raw.get("answer_confidence", raw.get("confidence", 0.0)),
        abstain=bool(raw.get("abstain", False)),
        provider=provider,
        model_id=model_id,
    )


__all__ = [
    "AnswerRecord",
    "AnswerRequest",
    "Answerer",
    "Candidate",
    "CompatibilityError",
    "ContractError",
    "EmbeddingBatch",
    "Encoder",
    "GroundingRecord",
    "IndexMetadata",
    "ModelMetadata",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderRequestError",
    "ProviderUnavailableError",
    "RerankRequest",
    "Reranker",
    "SearchHit",
    "VectorQuery",
    "VectorSearcher",
    "normalize_answer_record",
    "normalize_grounding_records",
]
