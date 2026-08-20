"""
HCMAIPipeline — production entry point cho 4 task của AIC HCMC 2026.

Single class với methods:
- kis(query, topk=10) → KIS retrieval (text → frame)
- vkis(image_or_clip_path, topk=10) → VKIS (image/clip → frame)
- vqa(query, question, topk=3) → VQA (text + question → frame + answer)
- trake(events, video_id=None, topk=10) → TRAKE (event list → ordered frames)

Pipelines:
- KIS = KISFusionRetriever (ViT-L + SO400M SigLIP2 fusion; remote translation opt-in)
- VKIS = VKISPipeline (SigLIP2 image encoder, measured hybrid0.5 selector)
- VQA = VQAPipelineV3 (visual retrieval + optional global ASR/OCR + local VLM)
- TRAKE = VisualTrakePipeline/DANTE by default; ASR is an explicit fallback

Lazy-load: chỉ load pipeline cần thiết.
"""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "router"))
sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
from paths import KEYFRAMES_DIR
from src.artifacts import build_catalog_preflight
from src.flow import FlowTrace, decide_specialist_flow
from src.runtime_context import RuntimeContext
from src.runtime_policy import RuntimePolicy


class HCMAIPipeline:
    """Single entry point for all 4 AIC tasks."""

    def __init__(self, kis_nllb_routing=None, kis_nllb_threshold=None,
                 kis_remote_translation=None, default_trake_mode=None,
                 default_vkis_selector=None, policy=None,
                 trake_multimodal_retrievers=None):
        self._kis = None
        self._vkis = None
        self._vqa = None
        # Interactive/API VQA and the official ranked offline path have
        # different dependencies and contracts.  Keep their model state
        # separate so calling one cannot replace the provider used by the
        # other halfway through a process.
        self._vqa_ranked = None
        self._vqa_ranked_spec = None
        self._vqa_answer_providers = {}
        self._trake = None
        self._trake_visual = None
        self._trake_multimodal = None
        self._trake_multimodal_retrievers = trake_multimodal_retrievers
        self._quest = None
        base_policy = RuntimePolicy.from_env() if policy is None else policy
        if not isinstance(base_policy, RuntimePolicy):
            raise TypeError("policy must be a RuntimePolicy instance")
        overrides = {}
        if kis_nllb_routing is not None:
            overrides["kis_nllb_routing"] = bool(kis_nllb_routing)
        if kis_nllb_threshold is not None:
            overrides["kis_nllb_threshold"] = float(kis_nllb_threshold)
        if kis_remote_translation is not None:
            overrides["kis_remote_translation"] = bool(kis_remote_translation)
        requested_trake_mode = None
        if default_trake_mode is not None:
            requested_trake_mode = str(default_trake_mode).strip().lower()
            if requested_trake_mode not in {"visual", "asr", "multimodal"}:
                raise ValueError(
                    "unsupported TRAKE mode: "
                    f"{requested_trake_mode!r}; choose 'visual', 'asr', or 'multimodal'"
                )
            # RuntimePolicy deliberately owns the legacy production modes.
            # Multimodal is an explicit injected research/competition route,
            # so keep it local to this composition root rather than widening
            # the process-wide policy contract.
            if requested_trake_mode != "multimodal":
                overrides["trake_mode"] = requested_trake_mode
        if default_vkis_selector is not None:
            overrides["vkis_selector"] = str(default_vkis_selector).strip().lower()
        self._policy = base_policy.override(**overrides)
        self.policy = self._policy
        try:
            artifact_snapshot = build_catalog_preflight().snapshot()
        except Exception as exc:
            artifact_snapshot = {"preflight": {"ready": False, "reason": str(exc)}}
        self.context = RuntimeContext.from_policy(
            self._policy, artifact_snapshot=artifact_snapshot
        )
        self.last_trace = None
        self._kis_nllb_routing = self._policy.kis_nllb_routing
        self._kis_nllb_threshold = self._policy.kis_nllb_threshold
        self._kis_remote_translation = self._policy.kis_remote_translation
        self._default_trake_mode = (
            requested_trake_mode
            if requested_trake_mode is not None
            else self._policy.trake_mode
        )
        self._default_vkis_selector = self._policy.vkis_selector
        self._vqa_modality_routing = self._policy.vqa_modality_routing
        self._vqa_modality_router = None
        self._vqa_modality_routers = {}

    @property
    def _offline_locked(self) -> bool:
        """Whether this boundary must reject remote/legacy execution paths.

        ``benchmark_strict`` and an explicit offline network policy are the
        fail-closed modes.  Production remains local by default, but an
        explicit ``network_mode=online`` plus provider choice may use the
        configured API; this is needed for the competition API profile and is
        never an implicit fallback.
        """
        return (
            self._policy.execution_mode == "benchmark_strict"
            or self._policy.network_mode == "offline"
        )

    def _new_request_context(self) -> RuntimeContext:
        """Create an immutable context for one public request.

        The startup context remains available for backwards-compatible
        artifact inspection, but no request trace reuses its request ID.
        """
        return RuntimeContext.from_policy(
            self._policy,
            mode=self.context.mode,
            split=self.context.split,
            artifact_snapshot=self.context.artifact_snapshot,
        )

    def _begin_trace(self, task: str, owner: str) -> FlowTrace:
        trace = FlowTrace(task=task, context=self._new_request_context(), owner=owner)
        self.last_trace = trace
        return trace

    @staticmethod
    def _finish_trace(trace: FlowTrace, decision=None) -> dict:
        if decision is not None:
            trace.decision(decision)
        trace.finish()
        return trace.to_dict()

    def _ensure_quest(self):
        if self._quest is None:
            sys.path.insert(0, os.path.join(ROOT, "..", "core"))
            from external_image_search import ExternalImageSearch
            # Reuse VKIS pipeline (shares SigLIP2 + global index)
            self._quest = ExternalImageSearch(vkis_pipeline=self._ensure_vkis(),
                                              backend="ddg")
        return self._quest

    def _ensure_kis(self):
        if self._kis is None:
            from kis_fusion_retriever import KISFusionRetriever
            self._kis = KISFusionRetriever(
                translate=self._kis_remote_translation, alpha=0.4,
                nllb_routing=self._kis_nllb_routing,
                nllb_threshold=self._kis_nllb_threshold,
            )
        return self._kis

    def kis_kmap(self):
        """Keyframe map used by KIS results."""
        return self._ensure_kis().km

    def _ensure_vkis(self):
        if self._vkis is None:
            from vkis_pipeline import VKISPipeline
            self._vkis = VKISPipeline()
        return self._vkis

    def _ensure_vqa(self):
        if self._vqa is None:
            from vqa_pipeline_v3 import VQAPipelineV3
            try:
                self._vqa = VQAPipelineV3(
                    translate=False if self._offline_locked else self._kis_remote_translation
                )
            except TypeError as exc:
                # Preserve lightweight dependency injection used by local
                # smoke tests/providers whose constructor has no translate
                # option; never hide a different constructor failure.
                if "unexpected keyword argument 'translate'" not in str(exc):
                    raise
                self._vqa = VQAPipelineV3()
        return self._vqa

    def _ensure_vqa_ranked(self, local_vlm_path=None, load_in_4bit=None):
        resolved_4bit = (
            self._policy.local_vlm_load_in_4bit
            if load_in_4bit is None else bool(load_in_4bit)
        )
        requested_spec = (
            None if local_vlm_path is None else os.path.abspath(str(local_vlm_path)),
            resolved_4bit,
        )
        if self._vqa_ranked is not None:
            if self._vqa_ranked_spec not in (None, requested_spec):
                raise RuntimeError(
                    "ranked VQA provider is already initialized with a different "
                    "local model configuration; create a new HCMAIPipeline for a "
                    "different provider"
                )
            return self._vqa_ranked
        if self._vqa_ranked is None:
            # Preserve test/integration injection of a ranked-capable provider
            # through the historical ``_vqa`` seam, but never alias an
            # interactive provider that has no local VLM.
            if self._vqa is not None and getattr(self._vqa, "_local_vlm", None) is not None:
                self._vqa_ranked = self._vqa
                self._vqa_ranked_spec = requested_spec
            else:
                from vqa_pipeline_v3 import VQAPipelineV3
                provider = self._ensure_vqa_answer_provider(
                    local_vlm_path=local_vlm_path,
                    load_in_4bit=resolved_4bit,
                )
                shared_kis = None
                if hasattr(self, "_kis"):
                    shared_kis = self._ensure_kis()
                vqa_kwargs = {
                    "translate": False,
                    "answer_provider": provider,
                }
                if shared_kis is not None:
                    vqa_kwargs["kis_retriever"] = shared_kis
                try:
                    self._vqa_ranked = VQAPipelineV3(**vqa_kwargs)
                except TypeError as exc:
                    # A ranked request must not silently switch to the legacy
                    # ``offline_vlm`` seam.  That bypasses the configured
                    # AnswerProvider and makes provider/trace behavior depend
                    # on constructor compatibility.
                    if "unexpected keyword argument 'kis_retriever'" in str(exc):
                        try:
                            self._vqa_ranked = VQAPipelineV3(
                                translate=False,
                                answer_provider=provider,
                            )
                        except TypeError as retry_exc:
                            if "unexpected keyword argument 'answer_provider'" in str(retry_exc):
                                raise RuntimeError(
                                    "ranked Q&A requires VQAPipelineV3 answer_provider support; "
                                    "legacy provider fallback is disabled"
                                ) from retry_exc
                            raise
                    elif "unexpected keyword argument 'answer_provider'" in str(exc):
                        raise RuntimeError(
                            "ranked Q&A requires VQAPipelineV3 answer_provider support; "
                            "legacy provider fallback is disabled"
                        ) from exc
                    raise
                self._vqa_ranked_spec = requested_spec
        return self._vqa_ranked

    def _ensure_vqa_answer_provider(self, *, local_vlm_path=None, load_in_4bit=None):
        """Build the selected answer provider without loading the model early."""
        provider_name = self._policy.vqa_answer_provider
        if self._offline_locked and provider_name != "local":
            raise RuntimeError(
                "offline execution requires the local Q&A answer provider; "
                f"got {provider_name!r}"
            )
        model_path = local_vlm_path or self._policy.local_vlm_path
        resolved_4bit = (
            self._policy.local_vlm_load_in_4bit
            if load_in_4bit is None else bool(load_in_4bit)
        )
        key = (provider_name, None if model_path is None else os.path.abspath(str(model_path)),
               resolved_4bit)
        cached = self._vqa_answer_providers.get(key)
        if cached is not None:
            return cached
        from src.vqa.answer_provider import (
            OpenAICompatibleAnswerProvider,
            QwenLocalAnswerProvider,
        )
        if provider_name == "local":
            path = model_path or os.path.join(
                os.path.dirname(ROOT), "..", "models", "Qwen2.5-VL-7B-Instruct"
            )
            provider = QwenLocalAnswerProvider(
                model_path=os.path.abspath(path),
                load_in_4bit=resolved_4bit,
            )
        elif provider_name == "openai":
            from src.core.providers import provider_for
            from paths import load_env
            config = provider_for("vision", load_env())
            if not config.base_url or not config.api_key:
                raise RuntimeError(
                    "VQA_ANSWER_PROVIDER=openai requires a configured vision API base URL and key"
                )
            provider = OpenAICompatibleAnswerProvider(
                base_url=config.base_url,
                model=config.model,
                api_key=config.api_key,
            )
        else:  # RuntimePolicy validates this; keep the boundary defensive.
            raise RuntimeError(f"unsupported VQA answer provider: {provider_name}")
        self._vqa_answer_providers[key] = provider
        return provider

    def _ensure_vqa_modality_router(self, active_modalities=None):
        requested = tuple(dict.fromkeys(
            str(value).strip().lower() for value in (active_modalities or ("asr", "ocr"))
            if str(value).strip().lower() in {"asr", "ocr"}
        ))
        if not requested:
            raise ValueError("active_modalities must contain asr and/or ocr")
        router = self._vqa_modality_routers.get(requested)
        if router is None:
            from src.reranking.qna_modality_router import QNAModalityRouter
            model_dir = self._policy.vqa_modality_model_dir
            router = QNAModalityRouter(
                model_dir=model_dir,
                strict=True,
                active_modalities=requested,
            )
            self._vqa_modality_routers[requested] = router
        self._vqa_modality_router = router
        return router

    def _ensure_trake(self):
        if self._trake is None:
            from trake_pipeline import TrakePipeline
            if self._offline_locked and self._policy.trake_remote_embeddings:
                raise RuntimeError(
                    "offline execution cannot enable remote TRAKE embeddings"
                )
            self._trake = TrakePipeline(
                online=(
                    False
                    if self._offline_locked
                    else self._policy.trake_remote_embeddings
                )
            )
        return self._trake

    def _ensure_trake_visual(self):
        if self._trake_visual is None:
            from trake_pipeline import VisualTrakePipeline
            self._trake_visual = VisualTrakePipeline()
        return self._trake_visual

    def _ensure_trake_multimodal(self):
        """Construct the explicitly injected multimodal TRAKE owner.

        This path is intentionally separate from the legacy ASR and visual
        caches.  A caller asking for multimodal retrieval must provide the
        event-level retrievers; missing dependencies are an error and never
        trigger a silent fallback to either legacy path.
        """
        retrievers = getattr(self, "_trake_multimodal_retrievers", None)
        if not retrievers:
            raise RuntimeError(
                "multimodal TRAKE requires injected visual/ASR/OCR retrievers; "
                "refusing to fall back to visual or ASR"
            )
        if self._trake_multimodal is None:
            from trake_pipeline import TrakePipeline

            self._trake_multimodal = TrakePipeline(
                mode="multimodal",
                retrievers=retrievers,
            )
        return self._trake_multimodal

    @staticmethod
    def _validate_trake_results(results, event_count, *, require_frame_ids=False):
        """Reject malformed alignment paths before they reach any adapter."""
        if not isinstance(results, list):
            raise RuntimeError("TRAKE backend returned a non-list result set")
        if not results:
            raise RuntimeError("TRAKE backend returned no ranked answers")
        for rank, item in enumerate(results, 1):
            if not isinstance(item, dict) or not str(item.get("video_id", "")).strip():
                raise RuntimeError(f"TRAKE rank {rank} has no video_id")
            path = item.get("path")
            if not isinstance(path, list) or len(path) != event_count:
                raise RuntimeError(
                    f"TRAKE rank {rank} has an incomplete path: "
                    f"expected {event_count} events"
                )
            frame_ids = []
            for step in path:
                if not isinstance(step, dict) or "frame_idx" not in step:
                    raise RuntimeError(f"TRAKE rank {rank} has a step without frame_idx")
                try:
                    frame_ids.append(int(step["frame_idx"]))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"TRAKE rank {rank} has a non-integer frame_idx") from exc
            if any(left >= right for left, right in zip(frame_ids, frame_ids[1:])):
                raise RuntimeError(f"TRAKE rank {rank} is not strictly time-ordered")
            raw_frame_ids = item.get("frame_ids")
            if require_frame_ids and not isinstance(raw_frame_ids, (list, tuple)):
                raise RuntimeError(f"TRAKE rank {rank} has no frame_ids contract field")
            if raw_frame_ids is not None:
                if len(raw_frame_ids) != event_count:
                    raise RuntimeError(
                        f"TRAKE rank {rank} has an incomplete frame_ids list: "
                        f"expected {event_count} events"
                    )
                contract_frame_ids = []
                for frame_id in raw_frame_ids:
                    if isinstance(frame_id, bool):
                        raise RuntimeError(f"TRAKE rank {rank} has a boolean frame_id")
                    try:
                        contract_frame_ids.append(int(frame_id))
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"TRAKE rank {rank} has a non-integer frame_id"
                        ) from exc
                if any(
                    left >= right
                    for left, right in zip(contract_frame_ids, contract_frame_ids[1:])
                ):
                    raise RuntimeError(
                        f"TRAKE rank {rank} frame_ids are not strictly increasing"
                    )
                if contract_frame_ids != frame_ids:
                    raise RuntimeError(
                        f"TRAKE rank {rank} frame_ids disagree with its canonical path"
                    )
        return results

    def kis(self, query, topk=10):
        """KIS: text query → top-K (video_id, frame_idx, pts_time, score)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("KIS query must be a non-empty string")
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 100:
            raise ValueError("KIS topk must be between 1 and 100")
        trace = self._begin_trace("KIS", "kis")
        trace.event("request", topk=topk, query_length=len(query.strip()))
        p = self._ensure_kis()
        raw = p.search(query, topk=topk)
        results = []
        for vid, fidx, kf_n, score in raw:
            m = p.km[(p.km.video_id == vid) & (p.km.kf_n == kf_n)]
            if not len(m):
                raise RuntimeError(
                    f"KIS returned a non-canonical frame: video_id={vid!r}, kf_n={kf_n!r}"
                )
            canonical_fidx = int(m.iloc[0].frame_idx)
            if int(fidx) != canonical_fidx:
                raise RuntimeError(
                    "KIS retriever frame_idx disagrees with the canonical keyframe map: "
                    f"video_id={vid!r}, kf_n={kf_n!r}, returned={fidx!r}, "
                    f"canonical={canonical_fidx!r}"
                )
            pts = float(m.iloc[0].pts_time)
            results.append((vid, canonical_fidx, pts, score))
        decision = decide_specialist_flow(
            trace.context, owner="kis", required_modalities=(),
            available_modalities=("visual",), specialist_hit=False,
        )
        return {"task": "KIS", "winner": "visual_fusion_vitl_so400m384", "results": results,
                "trace": self._finish_trace(trace, decision)}

    def vkis(self, path, topk=10):
        """VKIS: image/clip → top-K frames."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("VKIS input path must be non-empty")
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 100:
            raise ValueError("VKIS topk must be between 1 and 100")
        trace = self._begin_trace("VKIS", "vkis")
        trace.event("request", topk=topk, input_type="image" if path.lower().endswith((".jpg", ".jpeg", ".png")) else "clip")
        p = self._ensure_vkis()
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            results = p.search_image(path, topk=topk)
        else:
            results = p.search_clip(path, topk=topk, agg=self._default_vkis_selector)
        decision = decide_specialist_flow(
            trace.context, owner="vkis", required_modalities=(),
            available_modalities=("visual",), specialist_hit=False,
        )
        return {"task": "VKIS", "results": results,
                "trace": self._finish_trace(trace, decision)}

    def vqa(self, query, question, topk=3):
        """VQA compatibility facade backed by the ranked public owner.

        The historical interactive implementation had its own provider and
        fallback path.  Keep this signature/response shape for callers, but
        route the request through ``vqa_ranked`` so Q&A has one orchestration,
        provider and trace boundary.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("VQA query must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("VQA question must be a non-empty string")
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 100:
            raise ValueError("VQA topk must be between 1 and 100")
        out = self.vqa_ranked(
            query,
            question,
            max_answers=topk,
            top_videos=max(5, min(20, topk + 2)),
            frames_per_video=5,
            max_vlm_candidates=max(1, min(12, topk + 2)),
        )
        answers = out.get("answers") or []
        best = None
        if answers:
            first = answers[0]
            best = {
                "video": first.get("video_id"),
                "frame_idx": first.get("frame_id"),
                "answer": first.get("answer"),
            }
        return {
            "task": "VQA",
            "best": best,
            "answers": answers,
            "status": out.get("status"),
            "trace": out.get("trace"),
        }

    def vqa_ranked(self, query, question, max_answers=20, top_videos=20,
                   frames_per_video=5, max_vlm_candidates=12,
                   local_vlm_path=None, load_in_4bit=None, max_new_tokens=128,
                   use_context=True, question_type=None, required_modalities=None,
                   modality_routing=None, rrf_weights=None, answer_rerank_weights=None,
        visual_selector_policy=None, offline=None):
        """Offline-first ranked Q&A path for the official answer contract."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("VQA query must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("VQA question must be a non-empty string")
        for name, value, upper in (
            ("max_answers", max_answers, 20),
            ("top_videos", top_videos, 100),
            ("frames_per_video", frames_per_video, 20),
            ("max_vlm_candidates", max_vlm_candidates, 100),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be between 1 and {upper}")
        trace = self._begin_trace("Q&A", "qna")
        trace.event("request", top_videos=top_videos, frames_per_video=frames_per_video,
                    max_vlm_candidates=max_vlm_candidates)
        if offline is not None and not isinstance(offline, bool):
            raise ValueError("offline must be a boolean when provided")
        if load_in_4bit is not None and not isinstance(load_in_4bit, bool):
            raise ValueError("load_in_4bit must be a boolean when provided")
        execution_offline = (
            self._policy.vqa_answer_provider == "local"
            if offline is None else offline
        )
        if self._offline_locked and not execution_offline:
            raise RuntimeError("strict HCMAI execution requires offline Q&A")
        from vqa_pipeline_v3 import infer_question_type, normalize_question_type
        question_type_source = "explicit" if question_type is not None else "inferred"
        question_type = normalize_question_type(question_type)
        if question_type is None:
            question_type = infer_question_type(query, question)
        trace.event(
            "question_type",
            value=question_type,
            source=question_type_source,
        )
        # Validate routing configuration before constructing/loading the VLM.
        # A typo must fail at the boundary, not after a heavyweight model load.
        from vqa_pipeline_v3 import VQAPipelineV3
        VQAPipelineV3._parse_modalities(required_modalities)
        enabled = self._vqa_modality_routing if modality_routing is None else bool(modality_routing)
        modalities = required_modalities
        if enabled and not modalities and question_type in {"screen_text", "spoken_fact"}:
            modalities = f"visual,{'ocr' if question_type == 'screen_text' else 'asr'}"
        # Live query files do not carry the internal benchmark taxonomy.  The
        # conservative classifier above activates one specialist only for a
        # strong OCR/ASR cue; ambiguous questions stay visual rather than
        # requiring unrelated evidence from both text channels.
        visual_only_types = {
            "visual", "action", "color", "count", "person", "place",
            "temporal_relation",
        }
        if not modalities and enabled and question_type not in visual_only_types:
            raise RuntimeError(f"no routing policy for inferred question type {question_type!r}")
        requested_specialists = [
            part.strip().lower() for part in str(modalities or "").replace(";", ",").split(",")
            if part.strip().lower() in {"asr", "ocr"}
        ]
        global_router = self._ensure_vqa_modality_router(requested_specialists) if enabled and requested_specialists else None
        active_modalities = ["visual"] + (requested_specialists if global_router is not None else [])
        decision = decide_specialist_flow(
            trace.context,
            owner="qna",
            required_modalities=requested_specialists,
            available_modalities=active_modalities,
            specialist_hit=global_router is not None and bool(requested_specialists),
        )
        trace.event("routing", requested=modalities, enabled=enabled,
                    router_ready=global_router is not None)
        if decision.state == "failed":
            self._finish_trace(trace, decision)
            raise RuntimeError(decision.error or "Q&A modality route is unavailable")
        try:
            ranked_vqa = self._ensure_vqa_ranked(
                local_vlm_path=(self._policy.local_vlm_path if local_vlm_path is None else local_vlm_path),
                load_in_4bit=load_in_4bit,
            )
            out = ranked_vqa.ranked_answers(
                query, question, top_videos=top_videos,
                frames_per_video=frames_per_video,
                max_vlm_candidates=max_vlm_candidates,
                max_answers=max_answers, max_new_tokens=max_new_tokens,
                use_context=use_context, offline=execution_offline,
                question_type=question_type,
                required_modalities=modalities,
                global_modality_router=global_router,
                rrf_weights=rrf_weights,
                answer_rerank_weights=answer_rerank_weights,
                visual_selector_policy=(
                    self._policy.vqa_visual_selector_policy
                    if visual_selector_policy is None else visual_selector_policy
                ),
            )
            route_state = out.get("route_state")
            if requested_specialists and route_state is not None and route_state != "specialist_success":
                decision = decide_specialist_flow(
                    trace.context,
                    owner="qna",
                    required_modalities=requested_specialists,
                    available_modalities=active_modalities,
                    specialist_hit=False,
                )
                trace.event("routing_result", route_state=route_state)
                if decision.state == "failed":
                    self._finish_trace(trace, decision)
                    raise RuntimeError(decision.error or "Q&A specialist route returned no hit")
            out = {"task": "Q&A", **out}
            out["trace"] = self._finish_trace(trace, decision)
            return out
        except Exception as exc:
            trace.event("error", error_type=type(exc).__name__, message=str(exc))
            trace.finish()
            raise

    def trake(self, events, video_id=None, topk=10, mode=None):
        """TRAKE: event list → ordered frames in video.

        Visual DANTE is the default production path because the fixed
        independent benchmark is materially stronger than ASR DANTE. ASR is
        retained as an explicit fallback/diagnostic mode.
        """
        if isinstance(events, (str, bytes)) or not events:
            raise ValueError("TRAKE events must be a non-empty sequence")
        normalized_events = []
        for event in events:
            if isinstance(event, dict):
                event = event.get("desc", "")
            if not isinstance(event, str) or not event.strip():
                raise ValueError("each TRAKE event must be a non-empty string")
            normalized_events.append(event.strip())
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 100:
            raise ValueError("TRAKE topk must be between 1 and 100")
        video_id = None if video_id is None or not str(video_id).strip() else str(video_id).strip()
        mode = self._default_trake_mode if mode is None else str(mode).strip().lower()
        trace = self._begin_trace("TRAKE", "trake")
        trace.event("request", event_count=len(normalized_events), mode=mode or self._default_trake_mode)
        if mode == "asr":
            try:
                p = self._ensure_trake()
                results = p.align(normalized_events, video_id=video_id, top_k_videos=topk)
                self._validate_trake_results(results, len(normalized_events))
            except Exception as exc:
                trace.event("error", error_type=type(exc).__name__, message=str(exc))
                trace.finish()
                raise
            decision = decide_specialist_flow(
                trace.context, owner="trake_asr", required_modalities=("asr",),
                available_modalities=("asr",), specialist_hit=bool(results),
            )
            return {"task": "TRAKE", "winner": "trake_asr", "mode": mode, "results": results,
                    "trace": self._finish_trace(trace, decision)}
        if mode == "visual":
            try:
                p = self._ensure_trake_visual()
                out = p.search(normalized_events, video_id=video_id, top_k_videos=topk)
                self._validate_trake_results(out.get("results", []), len(normalized_events))
            except Exception as exc:
                trace.event("error", error_type=type(exc).__name__, message=str(exc))
                trace.finish()
                raise
            decision = decide_specialist_flow(
                trace.context, owner="trake_visual", required_modalities=(),
                available_modalities=("visual",), specialist_hit=False,
            )
            return {"task": "TRAKE", "winner": "trake_visual", "mode": mode,
                    "results": out["results"], "diagnostics": out.get("diagnostics", {}),
                    "trace": self._finish_trace(trace, decision)}
        if mode == "multimodal":
            try:
                p = self._ensure_trake_multimodal()
                results = p.align(
                    normalized_events,
                    video_id=video_id,
                    top_k_videos=topk,
                )
                self._validate_trake_results(
                    results,
                    len(normalized_events),
                    require_frame_ids=True,
                )
            except Exception as exc:
                trace.event("error", error_type=type(exc).__name__, message=str(exc))
                trace.finish()
                raise
            configured_modalities = tuple(
                sorted(
                    str(modality).strip().lower()
                    for modality in getattr(self, "_trake_multimodal_retrievers", {})
                    if str(modality).strip()
                )
            )
            decision = decide_specialist_flow(
                trace.context,
                owner="trake_multimodal",
                required_modalities=(),
                available_modalities=configured_modalities,
                specialist_hit=True,
            )
            return {
                "task": "TRAKE",
                "winner": "trake_multimodal",
                "mode": mode,
                "results": results,
                "trace": self._finish_trace(trace, decision),
            }
        error = ValueError(
            f"TRAKE mode '{mode}' is unsupported; "
            "choose 'asr', 'visual', or 'multimodal'"
        )
        trace.event("error", error_type=type(error).__name__, message=str(error))
        trace.finish()
        raise error

    def kis_ook(self, query, topk=10, max_ref_images=5):
        """KIS for Out-Of-Knowledge entities (QUEST): external image search →
        image-to-image retrieval. Dùng khi entity lạ (tên riêng/sản phẩm/sự kiện mới)
        mà text/visual embed không biết.
        """
        q = self._ensure_quest()
        results, meta = q.search_by_entity(query, topk=topk, max_ref_images=max_ref_images)
        return {"task": "KIS_OOK", "results": results, "meta": meta}

    def kis_auto(self, query, topk=10, ook_threshold=None):
        """KIS với fallback OOK: chạy KIS thường, nếu confidence thấp → QUEST.
        ook_threshold: optional calibrated score-margin threshold. Disabled by
        default because fused z-score margins are not calibrated confidence.
        """
        p = self._ensure_kis()
        results = p.search(query, topk=topk)
        # KISFusionRetriever returns ranked tuples, not the legacy router's
        # (winner, confidence, results) triple. Use score margin only as a
        # conservative OOK trigger; do not claim calibrated confidence.
        scores = [float(row[3]) for row in results]
        margin = scores[0] - scores[1] if len(scores) > 1 else 0.0
        if ook_threshold is not None and results and margin < ook_threshold:
            ook = self.kis_ook(query, topk=topk)
            if ook["results"]:
                return {"task": "KIS_AUTO", "winner": "ook_fallback",
                        "results": ook["results"], "base_conf": margin,
                        "ook_meta": ook["meta"]}
        return {"task": "KIS_AUTO", "winner": "visual_fusion_vitl_so400m384",
                "results": results, "base_conf": margin}


if __name__ == "__main__":
    import json
    p = HCMAIPipeline()

    print("=== HCMAI Pipeline smoke test ===\n")

    # KIS
    print("[KIS]")
    out = p.kis("siêu bão Biển Đông cấp 16", topk=3)
    print(f"  winner: {out['winner']}")
    for r in out["results"][:3]:
        print(f"  {r[0]} fidx={r[1]} t={r[2]:.1f}s sc={r[3]:.3f}")

    # VKIS (skip if no test image)
    print("\n[VKIS]")
    test_img = os.path.join(str(KEYFRAMES_DIR), "K01_V001", "010.jpg")
    if os.path.exists(test_img):
        out = p.vkis(test_img, topk=3)
        for r in out["results"][:3]:
            print(f"  {r[0]} fidx={r[1]} t={r[2]:.1f}s sc={r[3]:.4f}")

    # VQA
    print("\n[VQA]")
    out = p.vqa("siêu bão Biển Đông", "Cấp gió bão được nhắc đến là bao nhiêu?")
    if out.get("best"):
        print(f"  Best: {out['best']['video']} fidx={out['best']['frame_idx']} t={out['best']['pts_time']:.1f}s")
        print(f"  Answer: {out['best']['answer']}")

    # TRAKE
    print("\n[TRAKE]")
    events = ["Mở đầu bản tin với tin chính",
              "Thời tiết và dự báo bão"]
    out = p.trake(events, video_id="K01_V001", topk=1)
    if out["results"]:
        r = out["results"][0]
        print(f"  Video: {r['video_id']} score {r['score']:.3f}")
        for step in r["path"]:
            print(f"    [{step['start']:.0f}s] frame_idx={step['frame_idx']}: {step['event_desc']}")
