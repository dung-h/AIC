"""Portable manifests for model-agnostic vector indexes.

The manifest is deliberately independent of FAISS.  It describes the encoder,
metric, normalization contract, and stable row-id map required to interpret an
index.  Loading code must validate it before serving search results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ManifestValidationError(ValueError):
    """The persisted manifest is incomplete or incompatible with a request."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    """Identity and compatibility contract for one vector index artifact."""

    index_id: str
    modality: str
    encoder_id: str
    dimension: int
    encoder_version: str | None = None
    metric: str = "inner_product"
    normalized: bool = True
    normalization: str = "l2"
    backend: str = "faiss_hnsw"
    row_count: int = 0
    row_ids_sha256: str = ""
    catalog_schema: str = "aic2026.catalog.v1"
    corpus_hash: str | None = None
    schema_version: str = "1.0"
    index_sha256: str | None = None
    hnsw_m: int | None = None
    ef_construction: int | None = None

    def __post_init__(self) -> None:
        if not self.index_id or not self.modality or not self.encoder_id:
            raise ManifestValidationError("index_id, modality, and encoder_id are required")
        if self.dimension < 1 or self.row_count < 0:
            raise ManifestValidationError("dimension must be positive and row_count non-negative")
        if self.metric != "inner_product":
            raise ManifestValidationError("only inner_product is supported by this contract")
        if self.normalization not in {"l2", "none"}:
            raise ManifestValidationError("normalization must be 'l2' or 'none'")
        if self.normalized != (self.normalization == "l2"):
            raise ManifestValidationError("normalized and normalization fields disagree")
        if not self.normalized:
            raise ManifestValidationError("vector indexes must use normalized vectors")
        if not self.row_ids_sha256:
            raise ManifestValidationError("row_ids_sha256 is required")
        if self.encoder_version is not None and not str(self.encoder_version).strip():
            raise ManifestValidationError("encoder_version must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json(self.to_dict()))
        return output

    @classmethod
    def load(cls, path: str | Path) -> "IndexManifest":
        source = Path(path)
        if not source.exists():
            raise ManifestValidationError(f"manifest missing: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return cls(**payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if isinstance(exc, ManifestValidationError):
                raise
            raise ManifestValidationError(f"invalid manifest {source}: {exc}") from exc

    def assert_compatible(self, expected: Mapping[str, Any] | None = None, *, row_count: int | None = None,
                          row_ids_sha256: str | None = None) -> None:
        """Fail closed when runtime identity differs from persisted identity."""
        if row_count is not None and self.row_count != row_count:
            raise ManifestValidationError(f"row_count mismatch: manifest={self.row_count}, runtime={row_count}")
        if row_ids_sha256 is not None and self.row_ids_sha256 != row_ids_sha256:
            raise ManifestValidationError("row-id map checksum mismatch")
        for key, actual in (expected or {}).items():
            if actual is None:
                continue
            if not hasattr(self, key):
                raise ManifestValidationError(f"unknown compatibility field: {key}")
            persisted = getattr(self, key)
            if persisted != actual:
                raise ManifestValidationError(f"{key} mismatch: manifest={persisted!r}, runtime={actual!r}")


def write_row_id_map(path: str | Path, row_ids: list[str | int]) -> tuple[Path, str]:
    """Write a deterministic external id map and return its content checksum."""
    validate_row_ids(row_ids)
    output = Path(path)
    payload = {"schema_version": "1.0", "row_ids": row_ids}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload))
    return output, sha256_json(payload)


def read_row_id_map(path: str | Path) -> tuple[list[str | int], str]:
    source = Path(path)
    if not source.exists():
        raise ManifestValidationError(f"row-id map missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        row_ids = payload["row_ids"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ManifestValidationError(f"invalid row-id map {source}: {exc}") from exc
    if not isinstance(row_ids, list):
        raise ManifestValidationError("row_ids must be a list")
    validate_row_ids(row_ids)
    return row_ids, sha256_json({"schema_version": "1.0", "row_ids": row_ids})


def validate_row_ids(row_ids: list[str | int]) -> None:
    """Validate the external identity sequence without touching the filesystem."""
    if len(set(row_ids)) != len(row_ids):
        raise ManifestValidationError("row_ids must be unique")
    for row_id in row_ids:
        if row_id is None or isinstance(row_id, bool) or not isinstance(row_id, (str, int)):
            raise ManifestValidationError("row_ids must be non-null strings or integers")
