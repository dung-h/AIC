from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import pytest

from src.architecture.runtime import (
    ArchitectureDependencyError,
    ArchitectureRuntime,
    ChannelSpec,
)
from src.catalog import CanonicalFrame
from src.contracts import EmbeddingBatch, Encoder, IndexMetadata, ModelMetadata, SearchHit, VectorQuery, VectorSearcher


class FakeEncoder(Encoder):
    def __init__(self, metadata: ModelMetadata, behavior=None):
        self.metadata = metadata
        self.behavior = behavior
        self.calls = 0

    def encode(self, inputs):
        self.calls += 1
        if self.behavior is not None:
            self.behavior()
        return EmbeddingBatch(((1.0, 0.0),), self.metadata)


class FakeSearcher(VectorSearcher):
    def __init__(self, metadata: ModelMetadata, row_id: str):
        self.index_metadata = IndexMetadata(
            index_id=f"{metadata.model_id}-index",
            encoder=metadata,
            size=1,
        )
        self.row_id = row_id
        self.calls = 0

    def search(self, query: VectorQuery, top_k: int = 100):
        self.calls += 1
        return (SearchHit(row_id=self.row_id, score=1.0, rank=1),)


class RecordingExecutor:
    """Injectable bounded executor that records independent submissions."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.submissions: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args, **kwargs):
        self.submissions.append((fn, args, kwargs))
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        self._executor.shutdown(wait=wait)


def _frame_resolver(row_id: str, _shard_name: str | None) -> CanonicalFrame:
    return CanonicalFrame(
        row_id=None,
        video_id=f"video-{row_id}",
        kf_n=1,
        frame_idx=10,
        pts_time=1.0,
        fps=25.0,
        image_path=None,
        image_exists=False,
        map_path=Path("map.csv"),
    )


def _channel(
    name: str,
    *,
    row_id: str | None = None,
    behavior=None,
    required: bool = True,
    requires_network: bool = False,
) -> ChannelSpec:
    metadata = ModelMetadata(
        model_id=f"{name}-model",
        modality=name,
        dim=2,
        metric="inner_product",
        normalization="l2",
        version="test",
    )
    return ChannelSpec(
        modality=name,
        encoder=FakeEncoder(metadata, behavior=behavior),
        searcher=FakeSearcher(metadata, row_id or f"{name}-row"),
        channel_id=name,
        required=required,
        requires_network=requires_network,
        lane=name,
    )


def test_fanout_dispatches_channels_independently_with_bounded_executor():
    executors: list[RecordingExecutor] = []

    def factory(max_workers: int):
        executor = RecordingExecutor(max_workers)
        executors.append(executor)
        return executor

    runtime = ArchitectureRuntime(
        object(),
        [_channel("visual"), _channel("asr"), _channel("ocr")],
        resolver=_frame_resolver,
        executor_factory=factory,
        max_workers=8,
    )
    try:
        report = runtime.fanout("weather")
    finally:
        runtime.close()
        for executor in executors:
            executor.shutdown()

    assert executors[0].max_workers == 3
    assert [args[1] for _fn, args, _kwargs in executors[0].submissions] == [
        "visual",
        "asr",
        "ocr",
    ]
    assert report.successful_channel_ids == ("visual", "asr", "ocr")


def test_optional_channel_failure_preserves_visual_result_and_status():
    def fail_asr():
        raise RuntimeError("local ASR index unavailable")

    runtime = ArchitectureRuntime(
        object(),
        [_channel("visual"), _channel("asr", behavior=fail_asr, required=False)],
        resolver=_frame_resolver,
    )
    try:
        result = runtime.search("weather", top_k_videos=5)
        report = runtime.last_fanout
    finally:
        runtime.close()

    assert result
    assert report is not None
    assert report.results[0].channel_id == "visual"
    status = {item.channel_id: item for item in report.statuses}["asr"]
    assert status.state == "failed"
    assert status.error_type == "RuntimeError"
    assert "ASR index unavailable" in (status.error_message or "")


def test_required_channel_failure_fails_closed_with_observable_state():
    def fail_asr():
        raise RuntimeError("required ASR unavailable")

    runtime = ArchitectureRuntime(
        object(),
        [_channel("visual"), _channel("asr", behavior=fail_asr)],
        resolver=_frame_resolver,
    )
    with pytest.raises(ArchitectureDependencyError, match="asr.*required_failed"):
        runtime.fanout("spoken query")

    report = runtime.last_fanout
    assert report is not None
    status = {item.channel_id: item for item in report.statuses}["asr"]
    assert status.required is True
    assert status.state == "required_failed"
    assert "required ASR unavailable" in (status.error_message or "")
    runtime.close()


def test_result_order_is_configuration_deterministic_not_completion_order():
    finished: list[str] = []
    lock = threading.Lock()

    def behavior(name: str, delay: float):
        def run():
            time.sleep(delay)
            with lock:
                finished.append(name)

        return run

    channels = [
        _channel("visual", behavior=behavior("visual", 0.04)),
        _channel("asr", behavior=behavior("asr", 0.01)),
        _channel("ocr", behavior=behavior("ocr", 0.0)),
    ]
    runtime = ArchitectureRuntime(object(), channels, resolver=_frame_resolver, max_workers=3)
    try:
        first = runtime.fanout("query")
        second = runtime.fanout("query")
    finally:
        runtime.close()

    assert finished[:3] == ["ocr", "asr", "visual"]
    assert [item.channel_id for item in first.results] == ["visual", "asr", "ocr"]
    assert [item.channel_id for item in second.results] == ["visual", "asr", "ocr"]
    assert [item.channel_id for item in first.results] == [item.channel_id for item in second.results]


def test_offline_mode_blocks_network_channel_before_provider_call():
    network_calls = 0

    def network_behavior():
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network channel must not be invoked in offline mode")

    executor = RecordingExecutor(max_workers=2)
    runtime = ArchitectureRuntime(
        object(),
        [
            _channel("visual"),
            _channel(
                "asr",
                behavior=network_behavior,
                required=False,
                requires_network=True,
            ),
        ],
        resolver=_frame_resolver,
        executor=executor,
    )
    try:
        report = runtime.fanout("query", offline=True)
    finally:
        runtime.close()
        executor.shutdown()

    assert network_calls == 0
    assert [args[1] for _fn, args, _kwargs in executor.submissions] == ["visual"]
    status = {item.channel_id: item for item in report.statuses}["asr"]
    assert status.state == "offline_blocked"
    assert status.error_type == "OfflineNetworkBlocked"


def test_disabled_required_channel_fails_closed_before_dispatch():
    runtime = ArchitectureRuntime(
        object(),
        [_channel("visual"), ChannelSpec(
            modality="asr",
            encoder=_channel("asr").encoder,
            searcher=_channel("asr").searcher,
            channel_id="asr",
            enabled=False,
            required=True,
        )],
        resolver=_frame_resolver,
    )
    with pytest.raises(ArchitectureDependencyError, match="required channel.*disabled.*asr"):
        runtime.fanout("query")
