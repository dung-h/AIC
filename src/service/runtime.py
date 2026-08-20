"""Lazy resident runtime shared by API requests."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock
from time import perf_counter
from src.service.contracts import RetrievalResult, normalize_result
# Kept as module symbols for older test/integration monkeypatches.  Service
# request orchestration intentionally does not call either symbol anymore;
# HCMAIPipeline owns tracing and flow decisions.
from src.flow import FlowTrace, decide_specialist_flow  # noqa: F401
from src.runtime_policy import RuntimePolicy


@dataclass
class RuntimeStats:
    requests: int = 0
    errors: int = 0
    cache_hits: int = 0
    latency_ms: list[float] = field(default_factory=list)


class RetrievalRuntime:
    def __init__(self, policy: RuntimePolicy | None = None):
        self._policy = RuntimePolicy.from_env() if policy is None else policy
        if not isinstance(self._policy, RuntimePolicy):
            raise TypeError("policy must be a RuntimePolicy instance")
        self.policy = self._policy
        # HCMAIPipeline is the model/artifact composition owner.  The service
        # remains an HTTP adapter, but no longer constructs a second set of
        # task providers or a second preflight snapshot.
        from src.pipelines.hcmai_pipeline import HCMAIPipeline
        self._pipeline = HCMAIPipeline(policy=self._policy)
        self.context = self._pipeline.context
        self.last_trace = None
        self._lock = Lock()
        self._kis = None
        self._vkis = None
        self._vqa = None
        self._trake_visual = None
        self._trake_error = None
        self._trake_model_ready = False
        self._vqa_modality_router = None
        self._kis_frame_times = None
        self._kis_result_rows = None
        self._query_cache: OrderedDict[tuple[str, str, int], list] = OrderedDict()
        self._cache_limit = 1024
        self.last_timings_ms = {}
        self.stats = RuntimeStats()

    def kis(self):
        with self._lock:
            if self._kis is None:
                self._kis = self._pipeline._ensure_kis()
            return self._kis

    def kis_frame_time(self, video_id: str, kf_n: int, frame_idx: int) -> float:
        kis = self.kis()
        if self._kis_frame_times is None:
            self._kis_frame_times = {
                (str(v), int(k), int(f)): float(t)
                for v, k, f, t in zip(kis.km.video_id, kis.km.kf_n,
                                       kis.km.frame_idx, kis.km.pts_time)
            }
        key = (str(video_id), int(kf_n), int(frame_idx))
        if key not in self._kis_frame_times:
            raise RuntimeError(
                "KIS result is not present in the canonical keyframe map: "
                f"video_id={video_id!r}, kf_n={kf_n!r}, frame_idx={frame_idx!r}"
            )
        return self._kis_frame_times[key]

    def normalize_kis_result(self, raw) -> RetrievalResult:
        return normalize_result(raw, metadata_lookup=lambda v, f: self._kis_result_metadata(v, f))

    def normalize_vkis_result(self, raw) -> RetrievalResult:
        return normalize_result(raw, metadata_lookup=lambda v, f: self.vkis_frame_metadata(v, f))

    def _kis_result_metadata(self, video_id: str, frame_idx: int) -> tuple[int, float]:
        if self._kis_result_rows is None:
            kis = self.kis()
            self._kis_result_rows = {
                (str(v), int(f)): (int(k), float(t))
                for v, f, k, t in zip(kis.km.video_id, kis.km.frame_idx,
                                       kis.km.kf_n, kis.km.pts_time)
            }
        key = (str(video_id), int(frame_idx))
        if key not in self._kis_result_rows:
            raise RuntimeError(
                "KIS result frame_idx is not present in the canonical keyframe map: "
                f"video_id={video_id!r}, frame_idx={frame_idx!r}"
            )
        return self._kis_result_rows[key]

    def vkis_frame_metadata(self, video_id: str, frame_idx: int) -> tuple[int, float]:
        vkis = self.vkis()
        rows = vkis.vmap[(vkis.vmap.video_id.astype(str) == str(video_id)) &
                         (vkis.vmap.frame_idx == int(frame_idx))]
        if rows.empty:
            raise RuntimeError(
                "VKIS result frame_idx is not present in the canonical keyframe map: "
                f"video_id={video_id!r}, frame_idx={frame_idx!r}"
            )
        row = rows.iloc[0]
        return int(row.kf_n), float(row.pts_time)

    def vkis(self):
        with self._lock:
            if self._vkis is None:
                self._vkis = self._pipeline._ensure_vkis()
            return self._vkis

    def vqa(self):
        with self._lock:
            if self._vqa is None:
                self._vqa = self._pipeline._ensure_vqa_ranked(
                    local_vlm_path=self._policy.local_vlm_path,
                )
            return self._vqa

    def search_vqa(self, query: str, question: str, max_answers: int,
                   top_videos: int, frames_per_video: int,
                   max_vlm_candidates: int, question_type: str | None = None,
                   required_modalities: str | None = None):
        started = perf_counter()
        try:
            # HCMAIPipeline owns Q&A orchestration.  The service deliberately
            # does not construct a second router, trace, or ranked-answer
            # backend; this keeps service/API behavior identical to CLI/UI.
            public_method = getattr(self._pipeline, "vqa_ranked", None)
            if not callable(public_method):
                raise RuntimeError("HCMAIPipeline.vqa_ranked is unavailable")
            result = public_method(
                query, question, max_answers=max_answers,
                top_videos=top_videos, frames_per_video=frames_per_video,
                max_vlm_candidates=max_vlm_candidates, offline=True,
                question_type=question_type,
                required_modalities=required_modalities,
                visual_selector_policy=self._policy.vqa_visual_selector_policy,
            )
            if not isinstance(result, dict):
                raise RuntimeError("HCMAIPipeline.vqa_ranked returned a non-object result")
            self.last_trace = result.get("trace")
            self.record(started)
            return result
        except Exception as exc:
            self.record(started, error=True)
            raise

    def trake_visual(self):
        with self._lock:
            if self._trake_visual is None:
                try:
                    self._trake_visual = self._pipeline._ensure_trake_visual()
                except Exception as exc:
                    self._trake_error = str(exc)
                    raise
            return self._trake_visual

    def preload_trake(self):
        """Load the index/model before serving traffic, if explicitly requested."""
        pipeline = self.trake_visual()
        pipeline.warmup()
        self._trake_model_ready = True
        return pipeline

    def search_trake(self, events, top_k_videos: int, include_per_event_scores: bool):
        started = perf_counter()
        try:
            # The public HCMAIPipeline method owns mode selection, DANTE,
            # canonical validation, and tracing.  Do not bypass it through
            # the visual child pipeline from the service layer.
            public_method = getattr(self._pipeline, "trake", None)
            if not callable(public_method):
                raise RuntimeError("HCMAIPipeline.trake is unavailable")
            result = public_method(events, topk=top_k_videos)
            if not isinstance(result, dict) or not result.get("results"):
                raise RuntimeError("TRAKE backend returned no ranked answers")
            self._trake_model_ready = True
            self.last_trace = result.get("trace")
            diagnostics = result.get("diagnostics")
            timings = diagnostics.get("timings_ms") if isinstance(diagnostics, dict) else None
            if isinstance(timings, dict):
                self.last_timings_ms = dict(timings)
            self.record(started)
            return result
        except Exception as exc:
            self.record(started, error=True)
            raise

    def search_kis(self, query: str, topk: int, mode: str):
        started = perf_counter()
        key = (query.strip(), mode, topk)
        if key in self._query_cache:
            self.stats.cache_hits += 1
            result = self._query_cache[key]
            self._query_cache.move_to_end(key)
            self.last_timings_ms = {"cache": (perf_counter() - started) * 1000,
                                    "total": (perf_counter() - started) * 1000}
            self.record(started)
            return result
        try:
            if mode != "default":
                raise ValueError(f"unsupported KIS mode: {mode}")
            result = self.kis().search(query, topk=topk)
            self._query_cache[key] = result
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self._cache_limit:
                self._query_cache.popitem(last=False)
            self.last_timings_ms = dict(self.kis().last_timings_ms)
            self.record(started)
            return result
        except Exception:
            self.record(started, error=True)
            raise

    def record(self, started: float, error: bool = False):
        self.stats.requests += 1
        self.stats.errors += int(error)
        self.stats.latency_ms.append((perf_counter() - started) * 1000)
        if len(self.stats.latency_ms) > 1000:
            del self.stats.latency_ms[:-1000]

    def health(self):
        return {"status": "ok", "kis_loaded": self._kis is not None,
                "vkis_loaded": self._vkis is not None,
                "vqa_loaded": self._vqa is not None,
                "trake_visual_loaded": self._trake_visual is not None}

    def readiness(self):
        state = "ready" if self._trake_visual is not None else "not_loaded"
        if self._trake_error:
            state = "error"
        return {
            "status": "ok" if state == "ready" and self._trake_model_ready else "not_ready",
            "trake_visual": state,
            "trake_model": "ready" if self._trake_model_ready else "not_loaded",
            "trake_error": self._trake_error,
        }

    def snapshot(self):
        values = sorted(self.stats.latency_ms)
        p95 = values[int(.95 * (len(values) - 1))] if values else 0.0
        return {"requests": self.stats.requests, "errors": self.stats.errors,
                "cache_hits": self.stats.cache_hits, "cache_size": len(self._query_cache),
                "latency_p95_ms": round(p95, 3)}


_RUNTIME: RetrievalRuntime | None = None


def get_runtime() -> RetrievalRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = RetrievalRuntime()
    return _RUNTIME
