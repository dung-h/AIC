"""Provider- and model-agnostic retrieval runtime.

This is the integration boundary between the new contracts and the existing
artifacts. It intentionally does not import torch, a vendor SDK, or a remote
API. A channel supplies an ``Encoder`` and a vector-search implementation;
canonical frame resolution is delegated to the SQLite catalog. Fusion happens
only after each channel has emitted stable ranked hits, so unrelated embedding
dimensions remain compatible.
"""
from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from src.catalog import CatalogAdapter, CanonicalFrame, CatalogError
from src.contracts import CompatibilityError, Encoder, ModelMetadata, SearchHit, VectorQuery, VectorSearcher
from src.indexing.faiss_hnsw import (
    FaissUnavailableError,
    VectorIndex,
    load_vector_index,
)
from src.indexing.index_manifest import ManifestValidationError
from src.retrieval import ModalityHit, ModalityResult, FusedCandidate, fuse_channels


class ArchitectureError(RuntimeError):
    """Raised when a configured runtime cannot produce canonical results."""


class ArchitectureDependencyError(ArchitectureError):
    """Raised when a channel's model/index/canonical dependency is unavailable."""


@dataclass(frozen=True)
class ChannelSpec:
    """One retrieval channel with an independently replaceable model/index."""

    modality: str
    encoder: Encoder
    searcher: VectorSearcher
    shard_name: str | None = None
    enabled: bool = True
    channel_id: str | None = None
    # Keep the historical strict behavior by default.  New multimodal callers
    # can explicitly mark ASR/OCR (or another best-effort lane) optional.
    required: bool = True
    # The runtime never guesses whether an injected provider is remote.  The
    # composition root declares it, which lets strict offline requests skip it
    # before encoder/searcher code can make a network call.
    requires_network: bool = False
    lane: str = "default"

    def __post_init__(self) -> None:
        if not str(self.modality).strip():
            raise ArchitectureError("channel modality must be non-empty")
        if self.channel_id is not None and not str(self.channel_id).strip():
            raise ArchitectureError("channel_id must be non-empty when provided")
        if not isinstance(self.required, bool):
            raise ArchitectureError("channel required flag must be bool")
        if not isinstance(self.requires_network, bool):
            raise ArchitectureError("channel requires_network flag must be bool")
        if not str(self.lane).strip():
            raise ArchitectureError("channel lane must be non-empty")
        if self.encoder.metadata.modality != self.modality:
            raise ArchitectureError(
                f"channel modality mismatch: {self.modality!r} vs encoder {self.encoder.metadata.modality!r}"
            )
        if not callable(getattr(self.encoder, "encode_one", None)):
            raise ArchitectureError("channel encoder must expose callable encode_one")
        if not callable(getattr(self.searcher, "search", None)):
            raise ArchitectureError("channel searcher must expose callable search")
        index_metadata = getattr(self.searcher, "index_metadata", None)
        if index_metadata is None:
            raise ArchitectureError("channel searcher must expose index_metadata")
        try:
            self.encoder.metadata.assert_compatible(
                index_metadata.encoder,
                context=f"channel {self.resolved_channel_id}",
            )
        except AttributeError as exc:
            raise ArchitectureError(
                f"channel {self.resolved_channel_id} index_metadata must expose encoder metadata"
            ) from exc
        except CompatibilityError as exc:
            raise ArchitectureError(str(exc)) from exc

    @property
    def resolved_channel_id(self) -> str:
        return self.channel_id or f"{self.modality}:{self.encoder.metadata.model_id}"


@dataclass(frozen=True)
class ChannelExecutionStatus:
    """Observable result of one channel in a fanout request.

    A status is retained even when another channel fails.  This is important
    for diagnosing partial modality coverage without changing the old
    ``search() -> tuple[FusedCandidate, ...]`` API.
    """

    channel_id: str
    modality: str
    lane: str
    required: bool
    state: str
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.state == "success"


@dataclass(frozen=True)
class FanoutReport:
    """Deterministic channel-order report produced before fusion."""

    results: tuple[ModalityResult, ...]
    statuses: tuple[ChannelExecutionStatus, ...]

    @property
    def failed(self) -> tuple[ChannelExecutionStatus, ...]:
        return tuple(status for status in self.statuses if status.state in {
            "failed", "required_failed", "required_missing", "offline_blocked"
        })

    @property
    def successful_channel_ids(self) -> tuple[str, ...]:
        return tuple(status.channel_id for status in self.statuses if status.succeeded)


class VectorIndexAdapter(VectorSearcher):
    """Adapt the portable ``VectorIndex`` facade to the shared contract."""

    def __init__(self, index: VectorIndex, metadata: ModelMetadata) -> None:
        self._index = index
        self.index_metadata = _index_metadata(index, metadata)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata: ModelMetadata,
        *,
        manifest_path: str | Path | None = None,
        idmap_path: str | Path | None = None,
    ) -> "VectorIndexAdapter":
        """Load a persisted vector index with an architecture-level error boundary.

        The underlying index loader remains reusable by existing callers.  This
        adapter adds the context needed by a live multi-channel runtime and
        never falls back to an alternate backend when an artifact is missing or
        incompatible.
        """
        try:
            index = load_vector_index(
                index_path,
                manifest_path=manifest_path,
                idmap_path=idmap_path,
                expected={
                    "modality": metadata.modality,
                    "encoder_id": metadata.model_id,
                    "dimension": metadata.dim,
                    "metric": metadata.metric,
                    "normalization": metadata.normalization,
                    "encoder_version": metadata.version,
                },
            )
        except (FileNotFoundError, OSError, ManifestValidationError, FaissUnavailableError, ValueError) as exc:
            raise ArchitectureDependencyError(
                f"cannot load vector index {Path(index_path)!s} for "
                f"{metadata.modality}/{metadata.model_id}: {exc}"
            ) from exc
        return cls(index, metadata)

    def search(self, query: VectorQuery, top_k: int = 100) -> tuple[SearchHit, ...]:
        self.validate_query_encoder(query.metadata)
        results = self._index.search(query.vector, top_k=top_k)
        return tuple(SearchHit(row_id=str(item.row_id), score=item.score, rank=item.rank) for item in results)


def _index_metadata(index: VectorIndex, metadata: ModelMetadata):
    """Create the minimal ``IndexMetadata`` without importing provider code."""
    from src.contracts import IndexMetadata

    if index.manifest.dimension != metadata.dim:
        raise ArchitectureError(
            f"index/encoder dimension mismatch: {index.manifest.dimension} vs {metadata.dim}"
        )
    if index.manifest.encoder_id != metadata.model_id:
        raise ArchitectureError(
            f"index/encoder mismatch: {index.manifest.encoder_id!r} vs {metadata.model_id!r}"
        )
    if index.manifest.modality != metadata.modality:
        raise ArchitectureError(
            f"index/modality mismatch: {index.manifest.modality!r} vs {metadata.modality!r}"
        )
    if index.manifest.metric != metadata.metric:
        raise ArchitectureError(
            f"index/metric mismatch: {index.manifest.metric!r} vs {metadata.metric!r}"
        )
    if index.manifest.normalization != metadata.normalization:
        raise ArchitectureError(
            f"index/normalization mismatch: {index.manifest.normalization!r} vs {metadata.normalization!r}"
        )
    if index.manifest.encoder_version != metadata.version:
        raise ArchitectureError(
            f"index/encoder version mismatch: {index.manifest.encoder_version!r} vs {metadata.version!r}"
        )
    return IndexMetadata(
        index_id=index.manifest.index_id,
        encoder=metadata,
        size=index.manifest.row_count,
        row_key="row_id",
        corpus_hash=index.manifest.corpus_hash,
    )


class ArchitectureRuntime:
    """Run multi-channel retrieval and resolve every hit canonically."""

    def __init__(
        self,
        catalog: CatalogAdapter,
        channels: Sequence[ChannelSpec],
        *,
        resolver: Callable[[str, str | None], CanonicalFrame] | None = None,
        executor: Executor | None = None,
        executor_factory: Callable[[int], Executor] | None = None,
        max_workers: int | None = None,
        offline: bool = False,
    ) -> None:
        if not channels:
            raise ArchitectureError("at least one retrieval channel is required")
        channel_ids = [channel.resolved_channel_id for channel in channels]
        if len(set(channel_ids)) != len(channel_ids):
            raise ArchitectureError("channel ids must be unique")
        if executor is not None and executor_factory is not None:
            raise ArchitectureError("provide executor or executor_factory, not both")
        enabled_count = max(1, sum(1 for channel in channels if channel.enabled))
        if max_workers is not None and (
            not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1
        ):
            raise ArchitectureError("max_workers must be a positive integer")
        # There are currently three first-class lanes (visual/ASR/OCR).
        # Bound the pool even if a caller configures many experimental
        # channels; one request must not create one thread per channel.
        max_workers = min(max_workers or 3, enabled_count, 3)
        self.catalog = catalog
        self.channels = tuple(channels)
        self._channel_ids = tuple(channel_ids)
        self._resolver = resolver or self._resolve_row_id
        self._executor = executor
        self._executor_factory = executor_factory
        self._max_workers = max_workers
        self._owns_executor = executor is None
        self._offline = bool(offline)
        self._executor_lock = Lock()
        self._status_lock = Lock()
        self._last_fanout: FanoutReport | None = None

    @property
    def channel_ids(self) -> tuple[str, ...]:
        """Stable channel identities exposed for routing and observability."""
        return self._channel_ids

    @property
    def is_ready(self) -> bool:
        """Whether at least one channel is enabled and required lanes exist."""
        return bool(
            any(channel.enabled for channel in self.channels)
            and not any(channel.required and not channel.enabled for channel in self.channels)
        )

    @property
    def last_fanout(self) -> FanoutReport | None:
        """Most recent fanout report, including a report from a failed request."""
        with self._status_lock:
            return self._last_fanout

    @property
    def max_workers(self) -> int:
        """Configured upper bound for channel tasks in one request."""
        return self._max_workers

    def close(self) -> None:
        """Release the runtime-owned executor; injected executors remain caller-owned."""
        with self._executor_lock:
            executor = self._executor
            if executor is not None and self._owns_executor:
                executor.shutdown(wait=True)
                self._executor = None

    def __enter__(self) -> "ArchitectureRuntime":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def validate_dependencies(self) -> None:
        """Re-check live channel dependencies before serving traffic.

        Indexes loaded through :meth:`VectorIndexAdapter.load` are validated at
        load time.  This method is intentionally cheap and catches a channel
        disabled accidentally or a malformed injected searcher before a query
        is served.  It does not invent or silently substitute missing models.
        """
        enabled = [channel for channel in self.channels if channel.enabled]
        if not enabled:
            raise ArchitectureDependencyError("no enabled retrieval channels are configured")
        missing_required = [
            channel.resolved_channel_id
            for channel in self.channels
            if channel.required and not channel.enabled
        ]
        if missing_required:
            raise ArchitectureDependencyError(
                "required channel(s) are disabled: " + ", ".join(sorted(missing_required))
            )
        for channel in enabled:
            channel_id = channel.resolved_channel_id
            if getattr(channel.searcher, "index_metadata", None) is None:
                raise ArchitectureDependencyError(f"channel {channel_id} has no index metadata")
            if not callable(getattr(channel.encoder, "encode_one", None)):
                raise ArchitectureDependencyError(f"channel {channel_id} encoder is unavailable")
            if not callable(getattr(channel.searcher, "search", None)):
                raise ArchitectureDependencyError(f"channel {channel_id} searcher is unavailable")

    def _resolve_row_id(self, row_id: str, shard_name: str | None) -> CanonicalFrame:
        """Resolve either a catalog ordinal or explicit ``video_id#kf_n`` key.

        New indexes should prefer catalog ordinal ids for compact idmaps. The
        explicit key form is supported for independently built modality
        shards and makes the contract unambiguous without reading a parquet
        mapping at query time.
        """
        raw = str(row_id)
        if "#" in raw:
            video_id, kf_text = raw.rsplit("#", 1)
            try:
                return self.catalog.resolve_frame(video_id, int(kf_text))
            except (ValueError, TypeError, CatalogError) as exc:
                raise ArchitectureDependencyError(
                    f"invalid canonical row_id {raw!r}: {exc}"
                ) from exc
        try:
            if shard_name is None:
                raise ArchitectureError(
                    f"numeric row_id {raw!r} requires an explicit shard_name/idmap; "
                    "refusing to interpret it as a SQLite ordinal"
                )
            return self.catalog.resolve_row_id(int(raw), shard_name=shard_name)
        except (ValueError, TypeError):
            pass
        raise ArchitectureError(
            f"row_id {raw!r} is not a catalog ordinal or explicit video_id#kf_n key"
        )

    def _get_executor(self) -> Executor:
        """Create one bounded executor lazily, or return the injected one."""
        with self._executor_lock:
            if self._executor is None:
                if self._executor_factory is not None:
                    executor = self._executor_factory(self._max_workers)
                else:
                    executor = ThreadPoolExecutor(
                        max_workers=self._max_workers,
                        thread_name_prefix="architecture-retrieval",
                    )
                if executor is None or not callable(getattr(executor, "submit", None)):
                    raise ArchitectureError("executor must expose callable submit")
                self._executor = executor
            return self._executor

    @staticmethod
    def _status(
        channel: ChannelSpec,
        state: str,
        *,
        error: BaseException | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ChannelExecutionStatus:
        return ChannelExecutionStatus(
            channel_id=channel.resolved_channel_id,
            modality=channel.modality,
            lane=channel.lane,
            required=channel.required,
            state=state,
            error_type=error_type or (type(error).__name__ if error is not None else None),
            error_message=error_message if error_message is not None else (
                str(error) if error is not None else None
            ),
        )

    def _query_by_channel(self, query: Mapping[str, str] | str) -> dict[str, str]:
        if isinstance(query, str):
            return {channel_id: query for channel_id in self._channel_ids}
        query_by_channel = {str(key): str(value) for key, value in query.items()}
        unknown = sorted(
            set(query_by_channel)
            - set(self._channel_ids)
            - {channel.modality for channel in self.channels}
        )
        if unknown:
            raise ArchitectureError(f"query contains unknown channel(s): {', '.join(unknown)}")
        return query_by_channel

    def _retrieve_channel(
        self,
        channel: ChannelSpec,
        channel_id: str,
        text: str,
        *,
        top_k_per_channel: int,
    ) -> tuple[Any, tuple[SearchHit, ...]]:
        """Run one independent encoder/index lane.

        Canonical resolution is intentionally not done here.  The bundled
        SQLite catalog connection is thread-affine, so only model/index work
        belongs in worker threads; the caller thread materializes evidence
        after the fanout has completed.
        """
        try:
            batch = channel.encoder.encode_one(text)
            channel.encoder.metadata.assert_compatible(batch.metadata, context=f"{channel_id} encoder")
            vector = batch.vectors[0]
            hits = channel.searcher.search(
                VectorQuery(vector=vector, metadata=batch.metadata),
                top_k=top_k_per_channel,
            )
        except ArchitectureError:
            raise
        except (FileNotFoundError, OSError, ImportError, ModuleNotFoundError) as exc:
            raise ArchitectureDependencyError(
                f"channel {channel_id} dependency unavailable during search: {exc}"
            ) from exc
        return batch, tuple(hits)

    def _materialize_channel(
        self,
        channel: ChannelSpec,
        channel_id: str,
        batch: Any,
        hits: Sequence[SearchHit],
    ) -> ModalityResult:
        """Resolve raw worker hits on the caller thread into canonical evidence."""
        modality_hits = []
        for hit in hits:
            try:
                frame = self._resolver(hit.row_id, channel.shard_name)
            except ArchitectureError:
                raise
            except Exception as exc:
                raise ArchitectureDependencyError(
                    f"channel {channel_id} cannot resolve canonical evidence "
                    f"for row_id={hit.row_id!r}: {exc}"
                ) from exc
            if not isinstance(frame, CanonicalFrame):
                raise ArchitectureDependencyError(
                    f"channel {channel_id} resolver returned non-canonical evidence "
                    f"for row_id={hit.row_id!r}"
                )
            modality_hits.append(
                ModalityHit(
                    item_id=str(hit.row_id),
                    video_id=frame.video_id,
                    score=hit.score,
                    rank=hit.rank,
                    kf_n=frame.kf_n,
                    frame_idx=frame.frame_idx,
                    pts_time=frame.pts_time,
                    metadata={
                        "channel_id": channel_id,
                        "modality": channel.modality,
                        "encoder_id": channel.encoder.metadata.model_id,
                        "index_id": channel.searcher.index_metadata.index_id,
                        "row_id": str(hit.row_id),
                        "image_path": str(frame.image_path) if frame.image_path else None,
                    },
                )
            )
        return ModalityResult(
            modality=channel.modality,
            hits=tuple(modality_hits),
            encoder_id=channel.encoder.metadata.model_id,
            index_id=channel.searcher.index_metadata.index_id,
            embedding_dim=channel.encoder.metadata.dim,
            metric=channel.encoder.metadata.metric,
            channel_id=channel_id,
        )

    def fanout(
        self,
        query: Mapping[str, str] | str,
        *,
        top_k_per_channel: int = 100,
        offline: bool | None = None,
    ) -> FanoutReport:
        """Dispatch available channels through one bounded executor.

        Futures are collected in configured channel order rather than
        completion order.  Optional channel failures are recorded and omitted
        from fusion; required failures are recorded first and then fail closed
        after all submitted tasks have been observed.
        """
        self.validate_dependencies()
        if top_k_per_channel < 1 or top_k_per_channel > 100:
            raise ArchitectureError("top_k_per_channel must be in [1, 100]")
        query_by_channel = self._query_by_channel(query)
        offline_mode = self._offline if offline is None else bool(offline)

        statuses: dict[str, ChannelExecutionStatus] = {}
        futures: dict[str, Any] = {}
        for channel in self.channels:
            channel_id = channel.resolved_channel_id
            if not channel.enabled:
                statuses[channel_id] = self._status(channel, "disabled")
                continue
            text = query_by_channel.get(channel_id, query_by_channel.get(channel.modality))
            if not text or not text.strip():
                statuses[channel_id] = self._status(
                    channel,
                    "required_missing" if channel.required else "skipped",
                    error_type="MissingQuery" if channel.required else None,
                    error_message=(
                        f"no query text supplied for required channel {channel_id}"
                        if channel.required else None
                    ),
                )
                continue
            if offline_mode and channel.requires_network:
                statuses[channel_id] = self._status(
                    channel,
                    "offline_blocked",
                    error_type="OfflineNetworkBlocked",
                    error_message=(
                        f"channel {channel_id} requires network and is blocked in offline mode"
                    ),
                )
                continue
            try:
                futures[channel_id] = self._get_executor().submit(
                    self._retrieve_channel,
                    channel,
                    channel_id,
                    text,
                    top_k_per_channel=top_k_per_channel,
                )
            except Exception as exc:
                statuses[channel_id] = self._status(
                    channel,
                    "required_failed" if channel.required else "failed",
                    error=exc,
                )

        results_by_id: dict[str, ModalityResult] = {}
        # Iterating over ``self.channels`` is intentional: executor completion
        # order must never become result or provenance order.
        for channel in self.channels:
            channel_id = channel.resolved_channel_id
            future = futures.get(channel_id)
            if future is None:
                continue
            try:
                batch, hits = future.result()
                result = self._materialize_channel(channel, channel_id, batch, hits)
            except Exception as exc:
                statuses[channel_id] = self._status(
                    channel,
                    "required_failed" if channel.required else "failed",
                    error=exc,
                )
                continue
            statuses[channel_id] = self._status(channel, "success")
            results_by_id[channel_id] = result

        ordered_statuses = tuple(statuses[channel_id] for channel_id in self._channel_ids)
        report = FanoutReport(
            results=tuple(
                results_by_id[channel_id]
                for channel_id in self._channel_ids
                if channel_id in results_by_id
            ),
            statuses=ordered_statuses,
        )
        with self._status_lock:
            self._last_fanout = report

        required_failures = tuple(
            status for status in ordered_statuses
            if status.required and status.state != "success"
        )
        if required_failures:
            details = "; ".join(
                f"{status.channel_id} [{status.state}]: "
                f"{status.error_message or 'no successful result'}"
                for status in required_failures
            )
            raise ArchitectureDependencyError(
                f"required channel fanout failed closed: {details}"
            )
        return report

    def search(
        self,
        query: Mapping[str, str] | str,
        *,
        weights: Mapping[str, float] | None = None,
        top_k_per_channel: int = 100,
        top_k_videos: int = 20,
        top_k_frames: int | None = None,
        offline: bool | None = None,
    ) -> tuple[FusedCandidate, ...]:
        if top_k_videos < 1 or top_k_videos > 100:
            raise ArchitectureError("top_k_videos must be in [1, 100]")
        if top_k_frames is not None and (top_k_frames < 1 or top_k_frames > 100):
            raise ArchitectureError("top_k_frames must be in [1, 100]")
        report = self.fanout(
            query,
            top_k_per_channel=top_k_per_channel,
            offline=offline,
        )
        results = report.results
        if not results:
            return ()
        if top_k_frames is not None:
            return fuse_channels(results, weights=weights, top_k=top_k_frames, collapse_videos=False)
        return fuse_channels(
            results,
            weights=weights,
            top_k=top_k_videos,
            collapse_videos=True,
        )


__all__ = [
    "ArchitectureDependencyError",
    "ArchitectureError",
    "ArchitectureRuntime",
    "ChannelExecutionStatus",
    "ChannelSpec",
    "FanoutReport",
    "VectorIndexAdapter",
]
