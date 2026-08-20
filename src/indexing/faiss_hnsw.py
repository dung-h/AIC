"""Model-agnostic normalized vector indexes with a FAISS HNSW backend.

The public contract is intentionally small: build/load an index, search it,
and receive stable external ``row_id`` values.  FAISS is optional at import
time but mandatory when a FAISS backend is requested; there is no silent
fallback that could hide a production configuration error.  ``exact_numpy``
is an explicit diagnostic backend and is useful for contract tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.indexing.index_manifest import (
    IndexManifest,
    ManifestValidationError,
    read_row_id_map,
    sha256_json,
    validate_row_ids,
    write_row_id_map,
)

try:  # Optional import: Windows tooling can inspect manifests without FAISS.
    import faiss as _faiss
except ImportError:  # pragma: no cover - exercised by the contract test via monkeypatch
    _faiss = None


class FaissUnavailableError(RuntimeError):
    """A FAISS backend was requested but the optional dependency is unavailable."""


class VectorIndexError(ValueError):
    """Invalid vectors, row ids, or persisted index artifacts."""


@dataclass(frozen=True)
class SearchResult:
    row_id: str | int
    score: float
    rank: int


class _Backend(Protocol):
    dimension: int
    size: int

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def save(self, path: Path) -> None: ...


def require_faiss() -> Any:
    if _faiss is None:
        raise FaissUnavailableError(
            "FAISS backend requested but faiss is unavailable; install a compatible faiss build "
            "or explicitly select backend='exact_numpy' for diagnostics"
        )
    return _faiss


def normalize_vectors(vectors: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 1:
        raise VectorIndexError("vectors must have shape (n, dimension)")
    if not np.isfinite(values).all():
        raise VectorIndexError("vectors must contain only finite values")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise VectorIndexError("zero vectors cannot be normalized")
    return np.ascontiguousarray(values / norms[:, None], dtype=np.float32)


class _NumpyExactBackend:
    def __init__(self, vectors: np.ndarray):
        self._vectors = normalize_vectors(vectors)
        self.dimension = self._vectors.shape[1]
        self.size = self._vectors.shape[0]

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = normalize_vectors(queries) @ self._vectors.T
        k = min(top_k, self.size)
        all_ids = np.arange(self.size, dtype=np.int64)
        ids = np.empty((len(scores), k), dtype=np.int64)
        values = np.empty((len(scores), k), dtype=np.float32)
        for i, row in enumerate(scores):
            order = np.lexsort((all_ids, -row))[:k]
            ids[i] = order
            values[i] = row[order]
        return values, ids

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            np.save(handle, self._vectors, allow_pickle=False)

    @classmethod
    def load(cls, path: Path) -> "_NumpyExactBackend":
        with path.open("rb") as handle:
            return cls(np.load(handle, allow_pickle=False))


class _FaissBackend:
    def __init__(self, index: Any):
        self._index = index
        self.dimension = int(index.d)
        self.size = int(index.ntotal)

    @classmethod
    def build(cls, vectors: np.ndarray, backend: str, hnsw_m: int, ef_construction: int, ef_search: int) -> "_FaissBackend":
        faiss = require_faiss()
        dimension = vectors.shape[1]
        if backend == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        elif backend == "faiss_exact":
            index = faiss.IndexFlatIP(dimension)
        else:
            raise VectorIndexError(f"unsupported FAISS backend: {backend}")
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(vectors, np.arange(len(vectors), dtype=np.int64))
        return cls(index)

    @classmethod
    def load(cls, path: Path) -> "_FaissBackend":
        return cls(require_faiss().read_index(str(path)))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._index.search(queries, min(top_k, self.size))

    def save(self, path: Path) -> None:
        require_faiss().write_index(self._index, str(path))


class VectorIndex:
    """Search facade that maps internal integer ids to stable external row ids."""

    def __init__(self, backend: _Backend, row_ids: Sequence[str | int], manifest: IndexManifest):
        ids = list(row_ids)
        try:
            validate_row_ids(ids)
        except ManifestValidationError as exc:
            raise VectorIndexError(str(exc)) from exc
        if len(ids) != backend.size:
            raise VectorIndexError(f"row-id/backend size mismatch: {len(ids)} vs {backend.size}")
        if manifest.dimension != backend.dimension or manifest.row_count != len(ids):
            raise ManifestValidationError("manifest does not match backend dimensions or row count")
        expected_checksum = sha256_json({"schema_version": "1.0", "row_ids": ids})
        if manifest.row_ids_sha256 != expected_checksum:
            raise ManifestValidationError("manifest row-id checksum does not match supplied row ids")
        self._backend = backend
        self._row_ids = ids
        self.manifest = manifest

    @classmethod
    def from_backend(cls, backend: _Backend, row_ids: Sequence[str | int], *, manifest: IndexManifest | None = None,
                     index_id: str = "in_memory", modality: str = "unknown", encoder_id: str = "unknown") -> "VectorIndex":
        ids = list(row_ids)
        if manifest is None:
            manifest = IndexManifest(index_id=index_id, modality=modality, encoder_id=encoder_id,
                                     dimension=backend.dimension, row_count=len(ids),
                                     row_ids_sha256=sha256_json({"schema_version": "1.0", "row_ids": ids}),
                                     backend="fake")
        return cls(backend, ids, manifest)

    def search(self, query: np.ndarray | Sequence[float], top_k: int = 10) -> list[SearchResult]:
        return self.search_batch(query, top_k=top_k)[0]

    def search_batch(self, queries: np.ndarray | Sequence[Sequence[float]], top_k: int = 10) -> list[list[SearchResult]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized = normalize_vectors(np.asarray(queries, dtype=np.float32))
        if normalized.shape[1] != self.manifest.dimension:
            raise VectorIndexError(f"query dimension mismatch: expected {self.manifest.dimension}, got {normalized.shape[1]}")
        scores, internal_ids = self._backend.search(normalized, top_k)
        output: list[list[SearchResult]] = []
        for score_row, id_row in zip(scores, internal_ids):
            pairs = [(int(idx), float(score)) for score, idx in zip(score_row, id_row) if 0 <= int(idx) < len(self._row_ids)]
            pairs.sort(key=lambda item: (-item[1], item[0]))
            output.append([SearchResult(self._row_ids[idx], score, rank) for rank, (idx, score) in enumerate(pairs, 1)])
        return output

    @property
    def row_ids(self) -> tuple[str | int, ...]:
        return tuple(self._row_ids)


def _artifact_paths(index_path: Path, manifest_path: Path | None, idmap_path: Path | None) -> tuple[Path, Path, Path]:
    return index_path, manifest_path or Path(str(index_path) + ".manifest.json"), idmap_path or Path(str(index_path) + ".idmap.json")


def build_vector_index(features: np.ndarray | Sequence[Sequence[float]], row_ids: Sequence[str | int], index_path: str | Path,
                       *, backend: str = "faiss_hnsw", manifest_path: str | Path | None = None,
                       idmap_path: str | Path | None = None, index_id: str = "vector_index",
                       modality: str = "unknown", encoder_id: str = "unknown", encoder_version: str | None = None,
                       corpus_hash: str | None = None,
                       hnsw_m: int = 32, ef_construction: int = 200, ef_search: int = 128,
                       overwrite: bool = False) -> VectorIndex:
    values = normalize_vectors(np.asarray(features, dtype=np.float32))
    ids = list(row_ids)
    if len(ids) != len(values):
        raise VectorIndexError(f"feature/row-id mismatch: {len(values)} vs {len(ids)}")
    index_file, manifest_file, idmap_file = _artifact_paths(Path(index_path), Path(manifest_path) if manifest_path else None,
                                                             Path(idmap_path) if idmap_path else None)
    if not overwrite and any(path.exists() for path in (index_file, manifest_file, idmap_file)):
        raise FileExistsError(f"refusing to overwrite index artifacts: {index_file}")
    if backend == "exact_numpy":
        engine: _Backend = _NumpyExactBackend(values)
    elif backend in {"faiss_hnsw", "faiss_exact"}:
        engine = _FaissBackend.build(values, backend, hnsw_m, ef_construction, ef_search)
    else:
        raise VectorIndexError(f"unsupported backend: {backend}")
    index_file.parent.mkdir(parents=True, exist_ok=True)
    idmap_file, ids_checksum = write_row_id_map(idmap_file, ids)
    engine.save(index_file)
    index_checksum = hashlib.sha256(index_file.read_bytes()).hexdigest()
    manifest = IndexManifest(index_id=index_id, modality=modality, encoder_id=encoder_id,
                             encoder_version=encoder_version,
                             dimension=values.shape[1], backend=backend, row_count=len(ids),
                             row_ids_sha256=ids_checksum, corpus_hash=corpus_hash,
                             normalization="l2", normalized=True,
                             index_sha256=index_checksum, hnsw_m=hnsw_m if backend == "faiss_hnsw" else None,
                             ef_construction=ef_construction if backend == "faiss_hnsw" else None)
    manifest.save(manifest_file)
    return VectorIndex(engine, ids, manifest)


def load_vector_index(index_path: str | Path, *, manifest_path: str | Path | None = None, idmap_path: str | Path | None = None,
                      expected: dict[str, Any] | None = None) -> VectorIndex:
    index_file, manifest_file, idmap_file = _artifact_paths(Path(index_path), Path(manifest_path) if manifest_path else None,
                                                             Path(idmap_path) if idmap_path else None)
    manifest = IndexManifest.load(manifest_file)
    row_ids, ids_checksum = read_row_id_map(idmap_file)
    manifest.assert_compatible(expected, row_count=len(row_ids), row_ids_sha256=ids_checksum)
    if not index_file.exists():
        raise ManifestValidationError(f"index artifact missing: {index_file}")
    if manifest.index_sha256 and hashlib.sha256(index_file.read_bytes()).hexdigest() != manifest.index_sha256:
        raise ManifestValidationError("index checksum mismatch")
    if manifest.backend == "exact_numpy":
        engine: _Backend = _NumpyExactBackend.load(index_file)
    elif manifest.backend in {"faiss_hnsw", "faiss_exact"}:
        engine = _FaissBackend.load(index_file)
    else:
        raise VectorIndexError(f"unsupported persisted backend: {manifest.backend}")
    return VectorIndex(engine, row_ids, manifest)
