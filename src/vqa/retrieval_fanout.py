"""Deterministic concurrent fan-out for independent VQA retrieval channels.

Visual retrieval owns the GPU while the ASR/OCR router owns CPU-side vector or
lexical search.  Running those independent operations serially turns a
multi-modal request into a latency tax without improving evidence quality.
This module provides a small, model-agnostic concurrency boundary: callers
submit named callables, receive results in declared order, and fail closed if
any required channel errors.  It never merges scores or changes ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any


class RetrievalFanoutError(RuntimeError):
    """A named retrieval channel failed during a bounded fan-out."""


@dataclass(frozen=True, slots=True)
class RetrievalFanoutResult:
    """Results and timing trace, preserving the caller-declared channel order."""

    results: Mapping[str, Any]
    timings_ms: Mapping[str, float]
    wall_time_ms: float
    parallel: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "channels": list(self.results),
            "timings_ms": {key: round(float(value), 3) for key, value in self.timings_ms.items()},
            "wall_time_ms": round(float(self.wall_time_ms), 3),
            "parallel": bool(self.parallel),
        }


def run_retrieval_fanout(
    tasks: Mapping[str, Callable[[], Any]], *, max_workers: int | None = None,
) -> RetrievalFanoutResult:
    """Execute independent retrieval callables concurrently and fail closed.

    Results are materialized in input order even if tasks complete in a
    different order.  This makes concurrent execution reproducible at the
    pipeline boundary; downstream RRF sees the exact same channel ordering as
    with a serial implementation.
    """

    ordered = tuple((str(name).strip(), task) for name, task in tasks.items())
    if not ordered:
        return RetrievalFanoutResult({}, {}, 0.0, False)
    if any(not name or not callable(task) for name, task in ordered):
        raise ValueError("retrieval fan-out tasks require non-empty names and callables")
    names = [name for name, _ in ordered]
    if len(set(names)) != len(names):
        raise ValueError("retrieval fan-out task names must be unique")
    workers = len(ordered) if max_workers is None else int(max_workers)
    if workers < 1:
        raise ValueError("max_workers must be >= 1")
    workers = min(workers, len(ordered))

    def timed(task: Callable[[], Any]) -> tuple[Any, float]:
        started = perf_counter()
        return task(), (perf_counter() - started) * 1000.0

    started = perf_counter()
    futures: dict[str, Future[tuple[Any, float]]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vqa-retrieval") as pool:
        for name, task in ordered:
            futures[name] = pool.submit(timed, task)
        results: dict[str, Any] = {}
        timings: dict[str, float] = {}
        first_error: tuple[str, BaseException] | None = None
        # Resolve in declared order.  Do not stop at the first error: leaving
        # worker calls running in the background would make a failed request
        # mutate caches unpredictably after its caller has returned.
        for name, _task in ordered:
            try:
                value, elapsed = futures[name].result()
                results[name] = value
                timings[name] = elapsed
            except Exception as exc:
                if first_error is None:
                    first_error = (name, exc)
        if first_error is not None:
            name, exc = first_error
            raise RetrievalFanoutError(
                f"retrieval channel {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
    wall_time = (perf_counter() - started) * 1000.0
    return RetrievalFanoutResult(results, timings, wall_time, workers > 1)


__all__ = ["RetrievalFanoutError", "RetrievalFanoutResult", "run_retrieval_fanout"]
