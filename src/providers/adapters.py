"""Thin local and remote adapters for the stable contracts.

No provider SDK is imported at module import time.  Adapters never substitute
another provider when a dependency, credential, or request is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.contracts import (
    AnswerRecord,
    AnswerRequest,
    Answerer,
    CompatibilityError,
    ContractError,
    EmbeddingBatch,
    Encoder,
    GroundingRecord,
    IndexMetadata,
    ModelMetadata,
    ProviderConfigurationError,
    ProviderError,
    ProviderRequestError,
    ProviderUnavailableError,
    RerankRequest,
    Reranker,
    SearchHit,
    VectorQuery,
    VectorSearcher,
    normalize_answer_record,
    normalize_grounding_records,
)


def _require_callable(value: Any, name: str) -> Callable[..., Any]:
    if not callable(value):
        raise ProviderConfigurationError(f"{name} must be callable")
    return value


def _normalize_embeddings(raw: Any, metadata: ModelMetadata) -> EmbeddingBatch:
    if isinstance(raw, EmbeddingBatch):
        metadata.assert_compatible(raw.metadata, context="encoder output")
        return raw
    return EmbeddingBatch(raw, metadata)


def _candidate_payload(candidate: Any) -> Mapping[str, Any]:
    """Make the shared candidate record safe for JSON transports."""

    return {
        "candidate_id": candidate.candidate_id,
        "video_id": candidate.video_id,
        "frame_id": candidate.frame_id,
        "payload": candidate.payload,
        "evidence": dict(candidate.evidence),
    }


class LocalCallableEncoder(Encoder):
    """Wrap a local model object's ``encode`` method or a callable."""

    provider = "local"

    def __init__(self, metadata: ModelMetadata, encode_fn: Callable[[Sequence[Any]], Any]) -> None:
        self.metadata = metadata
        self._encode_fn = _require_callable(encode_fn, "encode_fn")

    def encode(self, inputs: Sequence[Any]) -> EmbeddingBatch:
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise ProviderConfigurationError("encoder inputs must be a sequence")
        try:
            return _normalize_embeddings(self._encode_fn(inputs), self.metadata)
        except ContractError:
            raise
        except Exception as exc:
            raise ProviderRequestError(f"local encoder {self.metadata.model_id!r} failed: {exc}") from exc


class LocalSentenceTransformerEncoder(LocalCallableEncoder):
    """Optional sentence-transformers adapter with a lazy import."""

    def __init__(
        self,
        metadata: ModelMetadata,
        model_name_or_path: str,
        *,
        model: Any = None,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_name_or_path or not isinstance(model_name_or_path, str):
            raise ProviderConfigurationError("model_name_or_path must be a non-empty string")
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "sentence-transformers is required for LocalSentenceTransformerEncoder; "
                    "install the local text-embedding dependency explicitly"
                ) from exc
            try:
                model = SentenceTransformer(model_name_or_path)
            except Exception as exc:
                raise ProviderConfigurationError(
                    f"could not load local sentence-transformers model {model_name_or_path!r}: {exc}"
                ) from exc
        encode_options = dict(encode_kwargs or {})
        encode_fn = lambda inputs: model.encode(inputs, **encode_options)
        super().__init__(metadata, encode_fn)


class RemoteCallableEncoder(Encoder):
    """Wrap an injected remote transport without binding to a vendor SDK."""

    provider = "remote"

    def __init__(
        self,
        metadata: ModelMetadata,
        request_fn: Callable[[Mapping[str, Any]], Any],
    ) -> None:
        self.metadata = metadata
        self._request_fn = _require_callable(request_fn, "request_fn")

    def encode(self, inputs: Sequence[Any]) -> EmbeddingBatch:
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise ProviderConfigurationError("encoder inputs must be a sequence")
        try:
            raw = self._request_fn({"model": self.metadata.model_id, "inputs": list(inputs)})
            if isinstance(raw, Mapping) and "vectors" in raw:
                raw = raw["vectors"]
            return _normalize_embeddings(raw, self.metadata)
        except (ContractError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderRequestError(f"remote encoder {self.metadata.model_id!r} failed: {exc}") from exc


class LocalCallableReranker(Reranker):
    provider = "local"

    def __init__(self, metadata: ModelMetadata, rerank_fn: Callable[[RerankRequest], Any]) -> None:
        self.metadata = metadata
        self._rerank_fn = _require_callable(rerank_fn, "rerank_fn")

    def rerank(self, request: RerankRequest) -> tuple[GroundingRecord, ...]:
        try:
            raw = self._rerank_fn(request)
            return normalize_grounding_records(
                raw,
                provider=self.provider,
                model_id=self.metadata.model_id,
                candidate_ids={candidate.candidate_id for candidate in request.candidates},
            )
        except (ContractError, ProviderConfigurationError, CompatibilityError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderRequestError(f"local reranker {self.metadata.model_id!r} failed: {exc}") from exc


class RemoteCallableReranker(Reranker):
    provider = "remote"

    def __init__(self, metadata: ModelMetadata, request_fn: Callable[[Mapping[str, Any]], Any]) -> None:
        self.metadata = metadata
        self._request_fn = _require_callable(request_fn, "request_fn")

    def rerank(self, request: RerankRequest) -> tuple[GroundingRecord, ...]:
        payload = {
            "model": self.metadata.model_id,
            "query": request.query,
            "candidates": [_candidate_payload(candidate) for candidate in request.candidates],
        }
        try:
            raw = self._request_fn(payload)
            return normalize_grounding_records(
                raw,
                provider=self.provider,
                model_id=self.metadata.model_id,
                candidate_ids={candidate.candidate_id for candidate in request.candidates},
            )
        except (ContractError, ProviderConfigurationError, CompatibilityError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderRequestError(f"remote reranker {self.metadata.model_id!r} failed: {exc}") from exc


class LocalCallableAnswerer(Answerer):
    provider = "local"

    def __init__(self, metadata: ModelMetadata, answer_fn: Callable[[AnswerRequest], Any]) -> None:
        self.metadata = metadata
        self._answer_fn = _require_callable(answer_fn, "answer_fn")

    def answer(self, request: AnswerRequest) -> AnswerRecord:
        try:
            return normalize_answer_record(
                self._answer_fn(request), provider=self.provider, model_id=self.metadata.model_id
            )
        except (ContractError, ProviderConfigurationError, CompatibilityError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderRequestError(f"local answerer {self.metadata.model_id!r} failed: {exc}") from exc


class RemoteCallableAnswerer(Answerer):
    provider = "remote"

    def __init__(self, metadata: ModelMetadata, request_fn: Callable[[Mapping[str, Any]], Any]) -> None:
        self.metadata = metadata
        self._request_fn = _require_callable(request_fn, "request_fn")

    def answer(self, request: AnswerRequest) -> AnswerRecord:
        payload = {
            "model": self.metadata.model_id,
            "query": request.query,
            "question": request.question,
            "candidates": [_candidate_payload(candidate) for candidate in request.candidates],
        }
        try:
            return normalize_answer_record(
                self._request_fn(payload), provider=self.provider, model_id=self.metadata.model_id
            )
        except (ContractError, ProviderConfigurationError, CompatibilityError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderRequestError(f"remote answerer {self.metadata.model_id!r} failed: {exc}") from exc


class LocalFaissVectorSearcher(VectorSearcher):
    """FAISS adapter; the import is deferred until ``from_file``/search time."""

    provider = "local"

    def __init__(self, index: Any, index_metadata: IndexMetadata, row_ids: Sequence[str]) -> None:
        self.index_metadata = index_metadata
        if not hasattr(index, "search"):
            raise ProviderConfigurationError("index must expose a search(query, top_k) method")
        if len(row_ids) != index_metadata.size:
            raise ProviderConfigurationError("row_ids length must equal index_metadata.size")
        self._index = index
        self._row_ids = tuple(str(row_id) for row_id in row_ids)
        if any(not row_id for row_id in self._row_ids):
            raise ProviderConfigurationError("row_ids must be non-empty")
        if len(set(self._row_ids)) != len(self._row_ids):
            raise ProviderConfigurationError("row_ids must be unique")
        index_dim = getattr(index, "d", None)
        if index_dim is not None and int(index_dim) != index_metadata.dim:
            raise ProviderConfigurationError(
                f"FAISS index dim {index_dim} does not match metadata dim {index_metadata.dim}"
            )

    @classmethod
    def from_file(
        cls,
        index_path: str | Path,
        index_metadata: IndexMetadata,
        row_ids: Sequence[str],
    ) -> "LocalFaissVectorSearcher":
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderUnavailableError(
                "faiss is required to load a FAISS index; install faiss-cpu/faiss-gpu explicitly"
            ) from exc
        path = Path(index_path)
        if not path.is_file():
            raise ProviderConfigurationError(f"FAISS index does not exist: {path}")
        try:
            index = faiss.read_index(str(path))
        except Exception as exc:
            raise ProviderConfigurationError(f"could not load FAISS index {path}: {exc}") from exc
        return cls(index, index_metadata, row_ids)

    def search(self, query: VectorQuery, top_k: int = 100) -> tuple[SearchHit, ...]:
        if not isinstance(query, VectorQuery):
            raise ProviderConfigurationError("query must be VectorQuery")
        self.validate_query_encoder(query.metadata)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ProviderConfigurationError("top_k must be a positive integer")
        top_k = min(top_k, self.index_metadata.size)
        if top_k == 0:
            return ()
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderUnavailableError("numpy is required for FAISS query adaptation") from exc
        try:
            scores, indices = self._index.search(np.asarray([query.vector], dtype="float32"), top_k)
        except Exception as exc:
            raise ProviderRequestError(f"FAISS search failed for {self.index_metadata.index_id!r}: {exc}") from exc
        hits: list[SearchHit] = []
        seen_positions: set[int] = set()
        for rank, (score, index_position) in enumerate(zip(scores[0], indices[0]), start=1):
            position = int(index_position)
            if position < 0:
                continue
            if position >= len(self._row_ids):
                raise ProviderRequestError(f"FAISS returned out-of-range row position {position}")
            if position in seen_positions:
                raise ProviderRequestError(f"FAISS returned duplicate row position {position}")
            seen_positions.add(position)
            hits.append(SearchHit(row_id=self._row_ids[position], score=float(score), rank=rank))
        return tuple(hits)


class RemoteHTTPTransport:
    """Vendor-neutral JSON transport using optional httpx, imported lazily."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ProviderConfigurationError("endpoint must be a non-empty URL")
        if timeout <= 0:
            raise ProviderConfigurationError("timeout must be positive")
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.headers = {str(key): str(value) for key, value in (headers or {}).items()}
        if api_key:
            self.headers.setdefault("Authorization", f"Bearer {api_key}")

    def __call__(self, payload: Mapping[str, Any]) -> Any:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ProviderUnavailableError(
                "httpx is required for RemoteHTTPTransport; install the remote-provider dependency explicitly"
            ) from exc
        try:
            response = httpx.post(self.endpoint, json=dict(payload), headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderRequestError(f"remote HTTP request failed for {self.endpoint!r}: {exc}") from exc


class RemoteJSONEncoder(RemoteCallableEncoder):
    def __init__(self, metadata: ModelMetadata, transport: Callable[[Mapping[str, Any]], Any]) -> None:
        super().__init__(metadata, transport)


class RemoteJSONReranker(RemoteCallableReranker):
    def __init__(self, metadata: ModelMetadata, transport: Callable[[Mapping[str, Any]], Any]) -> None:
        super().__init__(metadata, transport)


class RemoteJSONAnswerer(RemoteCallableAnswerer):
    def __init__(self, metadata: ModelMetadata, transport: Callable[[Mapping[str, Any]], Any]) -> None:
        super().__init__(metadata, transport)


__all__ = [
    "LocalCallableAnswerer",
    "LocalCallableEncoder",
    "LocalCallableReranker",
    "LocalFaissVectorSearcher",
    "LocalSentenceTransformerEncoder",
    "RemoteCallableAnswerer",
    "RemoteCallableEncoder",
    "RemoteCallableReranker",
    "RemoteHTTPTransport",
    "RemoteJSONAnswerer",
    "RemoteJSONEncoder",
    "RemoteJSONReranker",
]
