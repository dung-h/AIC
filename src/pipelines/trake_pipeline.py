"""
TRAKE ASR fallback/diagnostic pipeline: query (chuỗi sub-events) → align trên
ASR chunks → frames.  The shared production entrypoint defaults to visual
TRAKE; this module is selected only by an explicit ASR mode.

Input: List of N sub-event descriptions (from user/LLM split).
Process:
  1. bge-m3 embed N events (cached)
  2. For each candidate video, get its ASR chunks → embed similarity matrix
  3. DANTE align with λ=0.001 (news) → N chunks (with start time + frame_idx)
  4. Output: (video_id, [(frame_idx_1, t_1), (frame_idx_2, t_2), ...])

Use case AIC TRAKE: 1 query mô tả nhiều sub-event trong 1 video.
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
sys.path.insert(0, os.path.join(ROOT, "..", "core"))
from dante import dante_align
from cache import get_cache
from offline_fallback import get_text_embedder
from paths import INDEX_DIR
from src.core.providers import provider_for
from src.utils.open_clip_local import get_tokenizer as get_local_tokenizer
try:
    from src.trake.contracts import (
        normalize_events,
        validate_ranked_sequences,
        validate_sequence_path,
    )
    from src.trake.multimodal import EventLevelMultimodalDante
except ImportError:
    from trake.contracts import normalize_events, validate_ranked_sequences, validate_sequence_path
    from trake.multimodal import EventLevelMultimodalDante
try:
    from src.pipelines.trake_visual import VisualTrakeDante
except ImportError:
    from trake_visual import VisualTrakeDante
try:
    from src.pipelines.trake_asr_index import (
        ASRGlobalIndexError,
        SharedASRGlobalIndex,
    )
except ImportError:
    from trake_asr_index import ASRGlobalIndexError, SharedASRGlobalIndex

IDX = str(INDEX_DIR)

ASR_EMB_CACHE = get_cache("trake_asr_emb", version="v1")


class TrakePipeline:
    """TRAKE entrypoint for explicit ASR or injected multimodal retrieval.

    The historical ``HCMAIPipeline`` boundary constructs this class for
    ``mode="asr"`` and expects :meth:`align` to return a ranked list.  The
    explicit multimodal mode keeps that boundary while delegating retrieval
    and alignment to ``MultimodalTrakePipeline``.  It is constructor-gated so
    missing multimodal dependencies never trigger a silent visual/ASR
    fallback.
    """

    def __init__(self, online=False, *, mode="asr", retrievers=None,
                 modality_weights=None, asr_index_dir=None,
                 alignment_policy=None):
        resolved_mode = str(mode or "asr").strip().lower()
        if resolved_mode not in {"asr", "multimodal"}:
            raise ValueError(
                f"TRAKE mode {resolved_mode!r} is unsupported by TrakePipeline; "
                "choose 'asr' or 'multimodal'"
            )
        self.mode = resolved_mode
        self._multimodal = None
        if resolved_mode == "multimodal":
            if not retrievers:
                raise RuntimeError(
                    "multimodal TRAKE requires injected visual/ASR/OCR retrievers; "
                    "refusing to fall back to visual or ASR"
                )
            self._multimodal = MultimodalTrakePipeline(
                retrievers,
                modality_weights=modality_weights,
                alignment_policy=alignment_policy,
            )
            return

        if online:
            provider = provider_for("embedding")
            if not provider.configured:
                raise RuntimeError(
                    "Remote TRAKE text embedding is explicitly enabled but "
                    "EMBEDDING_BASE_URL, EMBEDDING_API_KEY and/or EMBEDDING_MODEL is missing; "
                    "use online=False for offline mode"
                )
        # The explicit ASR path has one source of truth.  Do not glob legacy
        # per-pack shards here: that creates partial-corpus behaviour and can
        # disagree with the canonical map used by the submission adapter.
        try:
            self.asr_index = SharedASRGlobalIndex(asr_index_dir)
        except ASRGlobalIndexError as exc:
            raise RuntimeError(
                "ASR TRAKE requires the ready shared merged ASR global index; "
                f"refusing legacy-shard fallback: {exc}"
            ) from exc
        self.ac = self.asr_index.chunks
        self.ce = self.asr_index.embeddings
        self.no_speech_videos = set(self.asr_index.no_speech_videos)
        self.index_diagnostics = self.asr_index.diagnostics()
        print(
            f"[TRAKE] ASR global pool: {len(self.ac)} chunks / "
            f"{self.ac.vid.nunique()} videos / "
            f"{len(self.no_speech_videos)} no-speech videos"
        )

        self.embedder = get_text_embedder("online" if online else "offline")

    def _embed_event(self, desc):
        """Cache event embedding."""
        v = ASR_EMB_CACHE.get("event_emb", desc)
        if v is not None: return v
        v = self.embedder.embed([desc])[0]
        ASR_EMB_CACHE.set(v, "event_emb", desc)
        return v

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        """Return timestamped ASR evidence for one event.

        This is the event-level hook consumed by the multimodal TRAKE
        aligner. It deliberately does not fill missing events; sequence
        alignment owns completeness and ordering.
        """
        event = normalize_events([event])[0]
        if top_k < 1 or top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")
        query = np.asarray(self._embed_event(event.description), dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        scores = self.ce @ query
        if candidate_videos is None:
            indices = np.arange(len(self.ac), dtype=np.int64)
        else:
            allowed = {str(value).strip().upper() for value in candidate_videos}
            indices = np.flatnonzero(self.ac["vid"].astype(str).isin(allowed).to_numpy())
        if len(indices) == 0:
            return []
        order = indices[np.argsort(-scores[indices], kind="mergesort")[:top_k]]
        hits = []
        for index in order:
            row = self.ac.iloc[int(index)]
            if pd.isna(row.get("frame_idx")) or pd.isna(row.get("kf_n")):
                continue
            start = row.get("start")
            if pd.isna(start):
                continue
            hits.append({
                "event_index": event.index,
                "video_id": str(row["vid"]),
                "modality": "asr",
                "score": float(scores[int(index)]),
                "frame_idx": int(row["frame_idx"]),
                "kf_n": int(row["kf_n"]),
                "pts_time": float(start),
                "source_id": "asr_chunks",
                "text": str(row.get("chunk", "")),
            })
        return hits

    def align(self, events, video_id=None, lam=0.001, top_k_videos=10,
              *, candidate_videos=None, top_k_per_event=100):
        """
        events: list of dict {"desc": str} or list of strings
        video_id: optional, restrict to single video
        Returns: list of {"video_id", "score", "path": [{frame_idx, pts_time, desc}]}

        lam: temporal penalty. ASR-channel default 0.001 (chunks gần nhau).
        LƯU Ý (Exp 122): với DANTE-on-VISUAL keyframes mà sub-events trải đều
        toàn video, λ=0 cho localization tốt nhất (λ≥0.005 phạt nhầm cấu trúc
        đúng → event±2s rớt 0.567→0.108). Tune λ theo độ thưa của sub-events.
        """
        if self.mode == "multimodal":
            if video_id is not None and candidate_videos is not None:
                raise ValueError("video_id and candidate_videos are mutually exclusive")
            allowed_videos = (
                [str(video_id)] if video_id is not None else candidate_videos
            )
            result = self._multimodal.align(
                events,
                top_k_videos=top_k_videos,
                top_k_per_event=top_k_per_event,
                candidate_videos=allowed_videos,
                lam=lam,
            )
            # Keep the historical TrakePipeline/HCMAIPipeline boundary:
            # callers receive a ranked list. The direct multimodal facade
            # retains diagnostics for callers that need them.
            return result["results"]

        events = normalize_events(events)
        N = len(events)
        ev_vecs = np.array([self._embed_event(event.description) for event in events])

        results = []

        if video_id is not None and candidate_videos is not None:
            raise ValueError("video_id and candidate_videos are mutually exclusive")
        if video_id is not None:
            target_vids = [str(video_id).strip().upper()]
        elif candidate_videos is not None:
            target_vids = list(
                dict.fromkeys(str(value).strip().upper() for value in candidate_videos)
            )
        else:
            target_vids = list(self.asr_index.videos)
        for vid in target_vids:
            # A no-speech video is valid build coverage but has no evidence;
            # skip it explicitly instead of inventing a frame or treating it
            # as a malformed index.
            if vid in self.no_speech_videos:
                continue
            ac_v = self.ac[self.ac.vid == vid].copy()
            if len(ac_v) < N: continue
            sort_columns = ["start", "end"]
            if "chunk_index" in ac_v.columns:
                sort_columns.append("chunk_index")
            if "embedding_row" in ac_v.columns:
                sort_columns.append("embedding_row")
            ac_v = ac_v.sort_values(sort_columns).reset_index(drop=False)
            chunk_emb = self.ce[ac_v["index"].values]
            S = ev_vecs @ chunk_emb.T  # [N events × T chunks]
            score, path = dante_align(S, lam=lam)
            if path is None: continue
            path_rows = []
            valid_path = True
            for i, t in enumerate(path):
                row = ac_v.iloc[t]
                if pd.isna(row.frame_idx) or pd.isna(row.kf_n):
                    valid_path = False
                    break
                path_rows.append({
                    "event_index": events[i].index,
                    "event_desc": events[i].description,
                    "video_id": str(vid),
                    "modality": "asr",
                    "chunk": str(row.chunk)[:100],
                    "start": float(row.start),
                    "end": float(row.end),
                    "frame_idx": int(row.frame_idx),
                    "kf_n": int(row.kf_n),
                    "pts_time": float(row.pts_time),
                })
            if not valid_path or any(
                a["frame_idx"] >= b["frame_idx"]
                for a, b in zip(path_rows, path_rows[1:])
            ):
                # Never manufacture frame 0 or emit an unordered path. The
                # output adapter can only submit a complete canonical path.
                continue
            if any(
                a["pts_time"] >= b["pts_time"]
                for a, b in zip(path_rows, path_rows[1:])
            ):
                continue
            validate_sequence_path(path_rows, events, video_id=str(vid))
            results.append({
                "video_id": vid,
                "score": float(score),
                "path": path_rows,
            })
        # Keep ranked output reproducible when two videos receive the same
        # score.  A tie must not be resolved by input/chunk traversal order.
        results.sort(key=lambda x: (-x["score"], str(x["video_id"])))
        return results[:top_k_videos]


class VisualTrakePipeline:
    """Visual TRAKE path backed by the frozen ViT-L/SigLIP2 index.

    This is intentionally separate from the legacy ASR pipeline. The visual
    benchmark established ``lambda=0`` as the safe default for this index.
    """

    def __init__(self, embed_provider=None, *, lattice_enabled=None,
                 lattice_top_k=10, temporal_neighbor_radius=1,
                 alignment_policy=None, candidate_video_limit=None,
                 video_relevance_weight=0.50,
                 alignment_evidence_weight=0.50):
        km_path = os.path.join(IDX, "global_keyframes_vitl.parquet")
        feat_path = os.path.join(IDX, "global_siglip_vitl.npy")
        if not os.path.exists(km_path) or not os.path.exists(feat_path):
            raise FileNotFoundError("visual TRAKE index is incomplete")
        self.km = pd.read_parquet(km_path).reset_index(drop=True)
        self.features = np.load(feat_path, mmap_mode="r")
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._embed_provider = embed_provider
        self.last_timings_ms = {}
        if alignment_policy is None:
            alignment_policy = os.getenv("TRAKE_VISUAL_ALIGNMENT_POLICY", "legacy")
        visual_alignment_policy = str(alignment_policy).strip().lower()
        if visual_alignment_policy not in {"legacy", "lattice_v1", "multi_video_v1"}:
                raise ValueError(
                    "TRAKE_VISUAL_ALIGNMENT_POLICY must be 'legacy', 'lattice_v1', "
                    "or 'multi_video_v1'"
                )
        if lattice_enabled is None:
            lattice_enabled = visual_alignment_policy == "lattice_v1"
        elif visual_alignment_policy != "multi_video_v1":
            # The pre-existing constructor flag takes precedence over the
            # environment selection.  Keep its diagnostic meaning unchanged
            # for callers that enabled/disabled the lattice programmatically.
            visual_alignment_policy = "lattice_v1" if lattice_enabled else "legacy"
        if not isinstance(lattice_top_k, int) or isinstance(lattice_top_k, bool) or lattice_top_k < 1:
            raise ValueError("lattice_top_k must be a positive integer")
        if not isinstance(temporal_neighbor_radius, int) or isinstance(temporal_neighbor_radius, bool) or temporal_neighbor_radius < 0:
            raise ValueError("temporal_neighbor_radius must be a non-negative integer")
        self.lattice_enabled = bool(lattice_enabled)
        self.lattice_top_k = int(lattice_top_k)
        self.temporal_neighbor_radius = int(temporal_neighbor_radius)
        self.visual_alignment_policy = visual_alignment_policy
        self.alignment_policy = (
            "multi_video_v1"
            if visual_alignment_policy == "multi_video_v1"
            else "legacy"
        )
        if candidate_video_limit is None and self.alignment_policy == "multi_video_v1":
            raw_limit = os.getenv("TRAKE_VISUAL_CANDIDATE_VIDEO_LIMIT")
            if raw_limit:
                try:
                    candidate_video_limit = int(raw_limit)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "TRAKE_VISUAL_CANDIDATE_VIDEO_LIMIT must be a positive integer"
                    ) from exc
        if candidate_video_limit is not None and (
            not isinstance(candidate_video_limit, int)
            or isinstance(candidate_video_limit, bool)
            or candidate_video_limit < 1
        ):
            raise ValueError("candidate_video_limit must be a positive integer")
        self.candidate_video_limit = candidate_video_limit
        self.video_relevance_weight = float(video_relevance_weight)
        self.alignment_evidence_weight = float(alignment_evidence_weight)
        self._visual = VisualTrakeDante(
            self.km,
            self.features,
            self._encode,
            lattice_enabled=self.lattice_enabled,
            alignment_policy=self.alignment_policy,
        )

    def warmup(self):
        """Load and validate the configured text encoder without searching."""
        self._encode(["TRAKE readiness probe"])
        return {"model_ready": self._model is not None, "feature_dim": int(self.features.shape[1])}

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        """Return global visual evidence for one event.

        The method is intentionally retrieval-only. Event ordering and
        completeness are enforced by ``EventLevelMultimodalDante``.
        """
        event = normalize_events([event])[0]
        if top_k < 1 or top_k > 1000:
            raise ValueError("top_k must be between 1 and 1000")
        vector = np.asarray(self._encode([event.description])[0], dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-8)
        scores = np.asarray(self.features @ vector, dtype=np.float32)
        if candidate_videos is None:
            indices = np.arange(len(self.km), dtype=np.int64)
        else:
            allowed = {str(value) for value in candidate_videos}
            indices = np.flatnonzero(self.km["video_id"].astype(str).isin(allowed).to_numpy())
        if len(indices) == 0:
            return []
        order = indices[np.argsort(-scores[indices], kind="mergesort")[:top_k]]
        hits = []
        for index in order:
            row = self.km.iloc[int(index)]
            if pd.isna(row.get("frame_idx")) or pd.isna(row.get("kf_n")):
                continue
            pts_time = row.get("pts_time")
            if pd.isna(pts_time):
                continue
            hits.append({
                "event_index": event.index,
                "video_id": str(row["video_id"]),
                "modality": "visual",
                "score": float(scores[int(index)]),
                "frame_idx": int(row["frame_idx"]),
                "kf_n": int(row["kf_n"]),
                "pts_time": float(pts_time),
                "source_id": "global_siglip_vitl",
            })
        return hits

    def _encode(self, texts):
        if self._embed_provider is not None:
            return np.asarray(self._embed_provider(texts), dtype=np.float32)
        if self._model is None:
            import torch
            import open_clip
            self._model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-16-SigLIP2-256", pretrained="webli")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.eval().to(self._device)
            self._tokenizer = get_local_tokenizer(open_clip, "ViT-L-16-SigLIP2-256")
        import torch
        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self._device)
            vectors = self._model.encode_text(tokens)
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return vectors.cpu().numpy().astype(np.float32)

    def search(self, events, *, top_k_videos=5, candidate_videos=None,
               video_id=None,
               include_per_event_scores=False,
               candidate_video_limit=None):
        if not events or any(not isinstance(event, str) or not event.strip() for event in events):
            raise ValueError("events must be a non-empty list of non-empty strings")
        if top_k_videos < 1 or top_k_videos > 100:
            raise ValueError("top_k_videos must be between 1 and 100")
        texts = [event.strip() for event in events]
        started = __import__("time").perf_counter()
        vectors = self._encode(texts)
        self._visual.embed_provider = lambda _: vectors
        if video_id is not None and candidate_videos is not None:
            raise ValueError("video_id and candidate_videos are mutually exclusive")
        result = self._visual.align(
            texts,
            video_id=video_id,
            top_k_videos=top_k_videos,
            candidate_videos=candidate_videos,
            lam=0.0,
            use_lattice=self.lattice_enabled,
            lattice_top_k=self.lattice_top_k,
            temporal_neighbor_radius=self.temporal_neighbor_radius,
            alignment_policy=self.alignment_policy,
            candidate_video_limit=(
                self.candidate_video_limit
                if candidate_video_limit is None else candidate_video_limit
            ),
            video_relevance_weight=self.video_relevance_weight,
            alignment_evidence_weight=self.alignment_evidence_weight,
        )
        result["diagnostics"]["visual_alignment_policy"] = self.visual_alignment_policy
        result["diagnostics"]["lattice_top_k"] = self.lattice_top_k
        result["diagnostics"]["temporal_neighbor_radius"] = self.temporal_neighbor_radius
        self.last_timings_ms = result.get("diagnostics", {})
        self.last_timings_ms["total"] = (__import__("time").perf_counter() - started) * 1000
        if not include_per_event_scores:
            for row in result["results"]:
                row.pop("per_event_scores", None)
        return result


class MultimodalTrakePipeline:
    """TRAKE facade for event-level visual/ASR/OCR retriever hooks.

    Model and index lifecycle remains owned by the supplied retrievers. This
    facade only owns event routing, DANTE alignment, and strict output
    validation, so it can be attached to an adapter without changing the
    existing visual or explicit-ASR fallback paths.
    """

    # Multimodal production composition must include both visual and ASR.
    # OCR remains optional until a full OCR global index exists.  Missing
    # required channels fail closed and never fall back to a legacy pipeline.
    REQUIRED_RETRIEVERS = frozenset({"visual", "asr"})

    def __init__(self, retrievers, *, modality_weights=None,
                 alignment_policy=None):
        if not isinstance(retrievers, dict) or not retrievers:
            raise RuntimeError(
                "multimodal TRAKE requires a retriever mapping; "
                "refusing to fall back to visual or ASR"
            )
        normalized = {
            str(modality).strip().lower(): retriever
            for modality, retriever in retrievers.items()
        }
        missing = sorted(self.REQUIRED_RETRIEVERS - set(normalized))
        if missing:
            raise RuntimeError(
                "multimodal TRAKE requires visual and ASR retrievers; "
                f"missing {missing}; refusing to fall back to visual or ASR"
            )
        self.retrievers = normalized
        if alignment_policy is None:
            alignment_policy = os.getenv("TRAKE_ALIGNMENT_POLICY", "legacy")
        alignment_policy = str(alignment_policy).strip().lower()
        self.aligner = EventLevelMultimodalDante(
            normalized,
            modality_weights=modality_weights,
            alignment_policy=alignment_policy,
        )

    @property
    def available_modalities(self):
        """Stable integration seam for a composition root/facade."""
        return tuple(sorted(self.retrievers))

    def align(self, events, *, top_k_videos=10, top_k_per_event=100,
              candidate_videos=None, lam=0.0):
        normalized = normalize_events(events)
        result = self.aligner.align(
            normalized,
            top_k_videos=top_k_videos,
            top_k_per_event=top_k_per_event,
            candidate_videos=candidate_videos,
            lam=lam,
        )
        if not result["results"]:
            raise RuntimeError("multimodal TRAKE produced no complete ranked sequence")
        # Use the normalized contract output, rather than validating and then
        # discarding it.  This guarantees ``frame_ids`` is present and agrees
        # with the canonical frame_idx values in every path before the result
        # crosses an entrypoint/adapter boundary.
        result["results"] = validate_ranked_sequences(result["results"], normalized)
        return result


if __name__ == "__main__":
    import json
    p = TrakePipeline(online=True)
    events = [
        "Mở đầu bản tin với tin chính",
        "Báo cáo về thiên tai và thiệt hại",
        "Hoạt động ngoại giao của lãnh đạo",
        "Kết thúc bản tin với thời tiết",
    ]
    print(f"\nQuery {len(events)} events:")
    for e in events: print(f"  - {e}")
    res = p.align(events, video_id="K01_V001")
    print(f"\nTop video result:")
    if res:
        r = res[0]
        print(f"  {r['video_id']} (score {r['score']:.3f})")
        for step in r["path"]:
            print(f"    [{step['start']:.0f}s] frame_idx={step['frame_idx']}")
            print(f"      Event: {step['event_desc']}")
            print(f"      ASR: {step['chunk']}")
