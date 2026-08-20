"""Explicit construction boundary for the model/index architecture runtime.

The factory is deliberately stricter than the lower-level adapters.  A caller
must provide every channel's encoder and persisted index artifact explicitly;
there is no backend substitution, lexical fallback, or implicit partial
runtime.  A valid diagnostic fixture may be constructed without a full-corpus
manifest, but it is never reported as live.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

from src.catalog import CatalogAdapter
from src.contracts import Encoder, ModelMetadata

from .runtime import (
    ArchitectureDependencyError,
    ArchitectureError,
    ArchitectureRuntime,
    ChannelSpec,
    VectorIndexAdapter,
)


class RuntimeBlockedError(ArchitectureDependencyError):
    """Raised when a caller requires a runtime that cannot be served safely."""


@dataclass(frozen=True)
class EncoderSpec:
    """Validated encoder identity plus an explicit instance or loader."""

    channel_id: str
    modality: str
    metadata: ModelMetadata
    encoder: Encoder | None = None
    encoder_factory: Callable[[], Encoder] | None = None

    def __post_init__(self) -> None:
        if not str(self.channel_id).strip():
            raise ArchitectureError("encoder channel_id must be non-empty")
        if not str(self.modality).strip():
            raise ArchitectureError("encoder modality must be non-empty")
        if self.metadata.modality != self.modality:
            raise ArchitectureError(
                f"encoder modality mismatch: {self.modality!r} vs metadata {self.metadata.modality!r}"
            )
        if self.encoder is not None and self.encoder_factory is not None:
            raise ArchitectureError("encoder spec must provide encoder or encoder_factory, not both")
        if self.encoder is None and self.encoder_factory is None:
            raise ArchitectureError(
                f"encoder {self.channel_id!r} has no instance or explicit encoder_factory"
            )
        if self.encoder_factory is not None and not callable(self.encoder_factory):
            raise ArchitectureError("encoder_factory must be callable")

    def materialize(self) -> Encoder:
        """Create the encoder and verify its metadata before runtime wiring."""

        try:
            encoder = self.encoder if self.encoder is not None else self.encoder_factory()  # type: ignore[misc]
        except Exception as exc:
            raise RuntimeBlockedError(
                f"encoder {self.channel_id!r} is unavailable: {exc}"
            ) from exc
        if not isinstance(encoder, Encoder):
            raise RuntimeBlockedError(
                f"encoder {self.channel_id!r} factory returned {type(encoder).__name__}, expected Encoder"
            )
        try:
            self.metadata.assert_compatible(
                encoder.metadata,
                context=f"encoder {self.channel_id}",
            )
        except Exception as exc:
            raise RuntimeBlockedError(str(exc)) from exc
        return encoder


@dataclass(frozen=True)
class IndexSpec:
    """Persisted vector-index artifacts for one channel.

    ``manifest_path`` and ``idmap_path`` are optional only to support the
    deterministic sidecar naming contract of ``load_vector_index``.  No other
    path or backend is guessed by this factory.
    """

    index_path: Path | str
    manifest_path: Path | str | None = None
    idmap_path: Path | str | None = None
    shard_name: str | None = None
    expected_backend: str | None = None

    def __post_init__(self) -> None:
        if not str(self.index_path).strip():
            raise ArchitectureError("index_path must be non-empty")
        if self.expected_backend is not None and self.expected_backend not in {
            "exact_numpy",
            "faiss_exact",
            "faiss_hnsw",
        }:
            raise ArchitectureError(f"unsupported expected index backend: {self.expected_backend!r}")
        if self.shard_name is not None and not str(self.shard_name).strip():
            raise ArchitectureError("shard_name must be non-empty when provided")

    def paths(self) -> tuple[Path, Path | None, Path | None]:
        return (
            Path(self.index_path),
            Path(self.manifest_path) if self.manifest_path is not None else None,
            Path(self.idmap_path) if self.idmap_path is not None else None,
        )


@dataclass(frozen=True)
class ChannelBuildSpec:
    """One explicitly wired encoder/index channel."""

    encoder: EncoderSpec
    index: IndexSpec
    enabled: bool = True


@dataclass(frozen=True)
class CatalogSpec:
    """Canonical catalog and optional proof of full-corpus scope."""

    db_path: Path | str
    full_corpus_manifest_path: Path | str | None = None
    expected_video_count: int = 1478

    def __post_init__(self) -> None:
        if not str(self.db_path).strip():
            raise ArchitectureError("catalog db_path must be non-empty")
        if self.expected_video_count < 1:
            raise ArchitectureError("expected_video_count must be positive")


@dataclass(frozen=True)
class ArchitectureRuntimeSpec:
    """Complete input specification for :class:`ArchitectureRuntimeFactory`."""

    catalog: CatalogSpec
    channels: tuple[ChannelBuildSpec, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ArchitectureError("runtime spec requires at least one channel")
        channel_ids = [channel.encoder.channel_id for channel in self.channels]
        if len(set(channel_ids)) != len(channel_ids):
            raise ArchitectureError("runtime spec channel_ids must be unique")


BuildStatus = Literal["blocked", "diagnostic_ready", "live"]


@dataclass(frozen=True)
class RuntimeBuildResult:
    """Explicit factory outcome; ``live`` is never inferred from construction alone."""

    status: BuildStatus
    runtime: ArchitectureRuntime | None
    reasons: tuple[str, ...] = ()
    channel_ids: tuple[str, ...] = ()
    full_corpus_manifest_path: Path | None = None

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"

    @property
    def diagnostic_ready(self) -> bool:
        return self.status == "diagnostic_ready"

    @property
    def live(self) -> bool:
        return self.status == "live"

    def require_runtime(self, *, require_live: bool = False) -> ArchitectureRuntime:
        if self.runtime is None:
            detail = "; ".join(self.reasons) or "runtime construction was blocked"
            raise RuntimeBlockedError(detail)
        if require_live and not self.live:
            detail = "; ".join(self.reasons) or "full-corpus manifest proof is missing"
            raise RuntimeBlockedError(f"runtime is not live: {detail}")
        return self.runtime


class ArchitectureRuntimeFactory:
    """Build one complete runtime without fallback or partial channel wiring."""

    def __init__(self, spec: ArchitectureRuntimeSpec) -> None:
        self.spec = spec

    def build(self, *, require_live: bool = False) -> RuntimeBuildResult:
        """Validate all dependencies and build an atomic runtime result.

        Missing catalog/index/model dependencies produce ``blocked`` with no
        runtime.  A valid fixture without a full-corpus manifest produces
        ``diagnostic_ready``.  ``require_live=True`` additionally raises a
        ``RuntimeBlockedError`` unless a valid full-corpus proof exists.
        """

        catalog: CatalogAdapter | None = None
        channels: list[ChannelSpec] = []
        reasons: list[str] = []
        try:
            catalog = CatalogAdapter(self.spec.catalog.db_path)
        except Exception as exc:
            reasons.append(f"catalog unavailable: {exc}")

        if catalog is not None:
            for channel_spec in self.spec.channels:
                channel_id = channel_spec.encoder.channel_id
                try:
                    encoder = channel_spec.encoder.materialize()
                    index_path, manifest_path, idmap_path = channel_spec.index.paths()
                    searcher = VectorIndexAdapter.load(
                        index_path,
                        channel_spec.encoder.metadata,
                        manifest_path=manifest_path,
                        idmap_path=idmap_path,
                    )
                    backend = searcher._index.manifest.backend  # type: ignore[attr-defined]
                    if (
                        channel_spec.index.expected_backend is not None
                        and backend != channel_spec.index.expected_backend
                    ):
                        raise RuntimeBlockedError(
                            f"channel {channel_id!r} backend mismatch: "
                            f"expected {channel_spec.index.expected_backend!r}, got {backend!r}"
                        )
                    channels.append(
                        ChannelSpec(
                            modality=channel_spec.encoder.modality,
                            encoder=encoder,
                            searcher=searcher,
                            shard_name=channel_spec.index.shard_name,
                            enabled=channel_spec.enabled,
                            channel_id=channel_id,
                        )
                    )
                except Exception as exc:
                    reasons.append(f"channel {channel_id!r} blocked: {exc}")

        if reasons or catalog is None or not channels:
            if catalog is not None:
                catalog.close()
            result = RuntimeBuildResult(
                status="blocked",
                runtime=None,
                reasons=tuple(reasons) or ("no channel could be constructed",),
                channel_ids=tuple(channel.encoder.channel_id for channel in self.spec.channels),
                full_corpus_manifest_path=self._manifest_path(),
            )
            if require_live:
                result.require_runtime(require_live=True)
            return result

        try:
            runtime = ArchitectureRuntime(catalog, channels)
            runtime.validate_dependencies()
        except Exception as exc:
            catalog.close()
            result = RuntimeBuildResult(
                status="blocked",
                runtime=None,
                reasons=(f"runtime validation blocked: {exc}",),
                channel_ids=tuple(channel.encoder.channel_id for channel in self.spec.channels),
                full_corpus_manifest_path=self._manifest_path(),
            )
            if require_live:
                result.require_runtime(require_live=True)
            return result

        live, proof_reasons = self._full_corpus_proof(runtime, catalog)
        result = RuntimeBuildResult(
            status="live" if live else "diagnostic_ready",
            runtime=runtime,
            reasons=tuple(proof_reasons),
            channel_ids=runtime.channel_ids,
            full_corpus_manifest_path=self._manifest_path(),
        )
        if require_live and not result.live:
            result.require_runtime(require_live=True)
        return result

    def _manifest_path(self) -> Path | None:
        value = self.spec.catalog.full_corpus_manifest_path
        return Path(value) if value is not None else None

    def _full_corpus_proof(
        self,
        runtime: ArchitectureRuntime,
        catalog: CatalogAdapter,
    ) -> tuple[bool, list[str]]:
        """Return live proof only from an explicit, matching corpus manifest."""

        manifest_path = self._manifest_path()
        if manifest_path is None:
            return False, ["full-corpus manifest is not configured; runtime remains diagnostic_only"]
        if not manifest_path.is_file():
            return False, [f"full-corpus manifest is missing: {manifest_path}"]
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return False, [f"full-corpus manifest is unreadable: {exc}"]
        if not isinstance(payload, dict):
            return False, ["full-corpus manifest must contain a JSON object"]
        scope = payload.get("scope", payload.get("corpus_scope"))
        if payload.get("full_corpus") is not True and scope != "full_corpus":
            return False, ["full-corpus manifest does not declare scope='full_corpus'"]
        try:
            video_count = int(payload.get("video_count", payload.get("canonical_video_count")))
        except (TypeError, ValueError):
            return False, ["full-corpus manifest has no valid video_count"]
        if video_count != self.spec.catalog.expected_video_count:
            return False, [
                "full-corpus manifest video_count mismatch: "
                f"expected {self.spec.catalog.expected_video_count}, got {video_count}"
            ]
        corpus_hash = payload.get("corpus_hash")
        if not isinstance(corpus_hash, str) or not corpus_hash.strip():
            return False, ["full-corpus manifest must declare a non-empty corpus_hash"]
        index_hashes = {
            channel.searcher.index_metadata.corpus_hash
            for channel in runtime.channels
            if channel.enabled
        }
        if index_hashes != {corpus_hash}:
            return False, [
                "enabled index corpus_hash values do not match the full-corpus manifest: "
                f"{sorted(str(value) for value in index_hashes)}"
            ]
        return True, []


__all__ = [
    "ArchitectureRuntimeFactory",
    "ArchitectureRuntimeSpec",
    "BuildStatus",
    "CatalogSpec",
    "ChannelBuildSpec",
    "EncoderSpec",
    "IndexSpec",
    "RuntimeBlockedError",
    "RuntimeBuildResult",
]
