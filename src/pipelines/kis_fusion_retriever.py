"""
Production KIS fusion retriever.

Validated on full VBS-325 (Exp125):
- ViT-L-16-SigLIP2-256: R@1 0.406, R@20 0.720
- SO400M-16-SigLIP2-384: R@1 0.437, R@20 0.772
- zscore fusion alpha=0.4 (ViT-L weight): R@1 0.483, R@20 0.800

This is candidate-generation fusion, not a reranker. It keeps top-20 recall high
while improving R@1 substantially.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from paths import INDEX_DIR  # noqa: E402
from src.core.providers import provider_for  # noqa: E402
from src.utils.open_clip_local import (  # noqa: E402
    create_model_and_transforms_local,
    get_tokenizer as get_local_tokenizer,
)


IDX = str(INDEX_DIR)


def _normalise_active_video_prefixes(value: object | None) -> tuple[str, ...]:
    """Return an explicit, deterministic subset selector for installed packs.

    A global index can contain K and L videos while a preselection deployment
    intentionally installs only L keyframes.  Ranking unavailable videos and
    dropping their frames later wastes the finite top-k budget.  The selector
    is opt-in for backward compatibility and is applied before video ranking.
    """
    if value is None:
        value = os.getenv("HCMAI_ACTIVE_VIDEO_PREFIXES", "")
    if isinstance(value, str):
        values = value.split(",")
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("active video prefixes must be a comma-separated string or sequence") from exc
    prefixes = tuple(dict.fromkeys(str(item).strip().upper() for item in values if str(item).strip()))
    if any(not prefix.replace("_", "").isalnum() for prefix in prefixes):
        raise ValueError("active video prefixes must contain only alphanumeric characters or underscores")
    return prefixes


class KISFusionRetriever:
    def __init__(self, translate=False, alpha=0.4, translate_cache=None,
                 nllb_routing=False, nllb_threshold=0.05,
                 active_video_prefixes=None):
        import torch
        import open_clip

        self.torch = torch
        self.open_clip = open_clip
        self.translate_on = translate
        self.alpha = alpha  # ViT-L weight; SO400M weight is 1-alpha
        self.text_provider = provider_for("text")
        self._translate_cache: dict = translate_cache or {}
        if translate and not self.text_provider.configured:
            raise RuntimeError(
                "Remote KIS translation is explicitly enabled but TEXT_BASE_URL, "
                "TEXT_API_KEY and/or TEXT_MODEL is missing; use translate=False for offline mode"
            )
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.nllb_routing = bool(nllb_routing)
        self.nllb_threshold = float(nllb_threshold)
        self.m_nllb = None
        self.tk_nllb = None
        self.F_nllb = None
        self.nllb_load_ms = 0.0
        self.route_used = False
        self.route_reason = "disabled"
        self.baseline_margin = None
        self.baseline_top1 = None
        self._last_frame_scores = None
        self._last_video_order = None

        # Keep large feature matrices memory-mapped. On WSL /mnt, eager
        # np.load(...).astype(float32) can duplicate several GB in RAM before
        # models move to VRAM.
        self.F_vitl = np.load(os.path.join(IDX, "global_siglip_vitl.npy"), mmap_mode="r")
        self.km = pd.read_parquet(os.path.join(IDX, "global_keyframes_vitl.parquet")).reset_index(drop=True)
        if len(self.F_vitl) != len(self.km):
            raise ValueError(f"ViT-L feature/map length mismatch: {len(self.F_vitl)} vs {len(self.km)}")
        self.vid_arr = self.km.video_id.astype(str).values
        self.active_video_prefixes = _normalise_active_video_prefixes(active_video_prefixes)
        active_mask = np.ones(len(self.vid_arr), dtype=bool)
        if self.active_video_prefixes:
            active_mask = np.fromiter(
                (str(video_id).upper().startswith(self.active_video_prefixes) for video_id in self.vid_arr),
                dtype=bool,
                count=len(self.vid_arr),
            )
            if not bool(active_mask.any()):
                raise ValueError(
                    "active video prefixes matched no indexed video: "
                    + ", ".join(self.active_video_prefixes)
                )
        self._active_row_idx = np.flatnonzero(active_mask)
        self.all_vids = np.array(sorted(set(self.vid_arr[self._active_row_idx])))
        self._vidx = {v: i for i, v in enumerate(self.all_vids)}
        self._active_vid_idx_arr = np.array(
            [self._vidx[self.vid_arr[row]] for row in self._active_row_idx], dtype=np.int32
        )
        self._frame_idx_arr = self.km.frame_idx.to_numpy(dtype=np.int64, copy=False)
        self._kf_n_arr = self.km.kf_n.to_numpy(dtype=np.int64, copy=False)
        self._pts_time_arr = self.km.pts_time.to_numpy(dtype=np.float32, copy=False)
        self._video_rows = {
            str(video): np.flatnonzero(self.vid_arr == video)
            for video in self.all_vids
        }
        self.last_timings_ms = {}

        so_f = os.path.join(IDX, "global_so400m384.npy")
        so_km = os.path.join(IDX, "global_keyframes_so400m384.parquet")
        self.has_so400m = os.path.exists(so_f) and os.path.exists(so_km)
        if self.has_so400m:
            self.F_so = np.load(so_f, mmap_mode="r")
            self.km_so = pd.read_parquet(so_km).reset_index(drop=True)
            if len(self.F_so) != len(self.km_so):
                raise ValueError(f"SO400M feature/map length mismatch: {len(self.F_so)} vs {len(self.km_so)}")
            if len(self.km_so) != len(self.km):
                raise ValueError(f"Fusion map length mismatch: ViT-L={len(self.km)} vs SO400M={len(self.km_so)}")
            identity_cols = ["video_id", "kf_n", "frame_idx"]
            missing = [c for c in identity_cols if c not in self.km or c not in self.km_so]
            if missing:
                raise ValueError(f"Fusion maps missing alignment columns: {missing}")
            vitl_ids = self.km[identity_cols].astype(str).to_numpy()
            so_ids = self.km_so[identity_cols].astype(str).to_numpy()
            if not np.array_equal(vitl_ids, so_ids):
                mismatch = int(np.flatnonzero(np.any(vitl_ids != so_ids, axis=1))[0])
                raise ValueError(f"Fusion keyframe maps are not row-aligned at row {mismatch}")
        else:
            self.F_so = None
            self.km_so = None

        self.m_vitl, _, _ = create_model_and_transforms_local(
            open_clip, "ViT-L-16-SigLIP2-256"
        )
        self.m_vitl = self.m_vitl.to(self.dev).eval()
        if self.dev == "cuda":
            self.m_vitl = self.m_vitl.half()
        self.tk_vitl = get_local_tokenizer(open_clip, "ViT-L-16-SigLIP2-256")

        if self.has_so400m:
            self.m_so, _, _ = create_model_and_transforms_local(
                open_clip, "ViT-SO400M-16-SigLIP2-384"
            )
            self.m_so = self.m_so.to(self.dev).eval()
            if self.dev == "cuda":
                self.m_so = self.m_so.half()
            self.tk_so = get_local_tokenizer(open_clip, "ViT-SO400M-16-SigLIP2-384")
        else:
            self.m_so = None
            self.tk_so = None

        if self.nllb_routing:
            nllb_f = os.path.join(os.path.dirname(IDX), "..", "results",
                                  "nllb_clip_large_siglip_full.npy")
            nllb_f = os.path.normpath(nllb_f)
            if not os.path.exists(nllb_f):
                raise FileNotFoundError(f"NLLB routing feature index missing: {nllb_f}")
            self.F_nllb = np.load(nllb_f, mmap_mode="r")
            if len(self.F_nllb) != len(self.km):
                raise ValueError("NLLB feature index is not aligned with ViT-L keyframes")

        print(
            f"[KISFusion] vitl={self.F_vitl.shape}, so400m={None if self.F_so is None else self.F_so.shape}, "
            f"videos={len(self.all_vids)}, prefixes={self.active_video_prefixes or ('all',)}, "
            f"alpha={alpha}, translate={translate}, device={self.dev}"
        )

    def _ensure_nllb_model(self):
        if self.m_nllb is not None:
            return 0.0
        started = perf_counter()
        self.m_nllb, _, _ = create_model_and_transforms_local(
            self.open_clip, "nllb-clip-large-siglip"
        )
        self.m_nllb = self.m_nllb.to(self.dev).eval()
        if self.dev == "cuda":
            self.m_nllb = self.m_nllb.half()
        self.tk_nllb = get_local_tokenizer(self.open_clip, "nllb-clip-large-siglip")
        self.nllb_load_ms = (perf_counter() - started) * 1000
        return self.nllb_load_ms

    def translate(self, text: str) -> str:
        if not self.translate_on:
            return text
        if text in self._translate_cache:
            return self._translate_cache[text]
        provider = self.text_provider
        if not provider.configured:
            raise RuntimeError(
                "Remote KIS translation is enabled but credentials are unavailable"
            )
        payload = {
            "model": provider.model,
            "messages": [{"role": "user", "content":
                "Translate this Vietnamese visual scene description to a concise English image caption "
                "(keep all visual details). Output ONLY the caption:\n\n" + text}],
            "max_tokens": 150,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            provider.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.load(r)["choices"][0]["message"]["content"].strip()
                self._translate_cache[text] = out
                return out
        except Exception as exc:
            raise RuntimeError("Remote KIS translation failed; no offline fallback is allowed") from exc

    def _encode_text(self, model, tokenizer, q: str) -> np.ndarray:
        with self.torch.no_grad():
            toks = tokenizer([q]).to(self.dev)
            f = model.encode_text(toks)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()[0].astype(np.float32)

    def _maxvec(self, frame_sc: np.ndarray) -> np.ndarray:
        out = np.full(len(self.all_vids), -9.0, np.float32)
        np.maximum.at(out, self._active_vid_idx_arr, frame_sc[self._active_row_idx])
        return out

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - float(x.mean())) / (float(x.std()) + 1e-6)

    def _best_frame(self, video_id: str, frame_sc: np.ndarray) -> pd.Series:
        gidx = self._video_rows[str(video_id)]
        return self.km.iloc[int(gidx[np.argmax(frame_sc[gidx])])]

    def _best_fused_frame(self, video_id: str, frame_vitl: np.ndarray,
                          frame_so: np.ndarray) -> pd.Series:
        """Pick the best frame using the same normalized fusion as evaluation.

        Video-level fusion and frame selection use different granularities. The
        old implementation selected a frame from whichever encoder won for the
        video, which could discard complementary frame evidence.
        """
        gidx = self._video_rows[str(video_id)]
        vitl = frame_vitl[gidx].astype(np.float32, copy=False)
        so = frame_so[gidx].astype(np.float32, copy=False)
        vitl = (vitl - float(vitl.mean())) / (float(vitl.std()) + 1e-6)
        so = (so - float(so.mean())) / (float(so.std()) + 1e-6)
        return self.km.iloc[int(gidx[np.argmax(self.alpha * vitl + (1.0 - self.alpha) * so)])]

    def search(self, query_vn: str, topk=20):
        started = perf_counter()
        nllb_load_this_query_ms = 0.0
        t = perf_counter()
        q = self.translate(query_vn)
        translate_ms = (perf_counter() - t) * 1000

        t = perf_counter()
        qv_vitl = self._encode_text(self.m_vitl, self.tk_vitl, q)
        encode_vitl_ms = (perf_counter() - t) * 1000
        t = perf_counter()
        frame_vitl = self.F_vitl @ qv_vitl
        vid_vitl = self._maxvec(frame_vitl)
        search_vitl_ms = (perf_counter() - t) * 1000

        if not self.has_so400m:
            order = np.argsort(-vid_vitl)[:topk]
            t = perf_counter(); result = self._format_results(order, vid_vitl, frame_vitl, None, None)
            self.last_timings_ms = {"translate": translate_ms, "encode_vitl": encode_vitl_ms,
                                    "search_vitl": search_vitl_ms, "encode_so400m": 0.0,
                                    "search_so400m": 0.0, "format": (perf_counter()-t)*1000,
                                    "total": (perf_counter()-started)*1000}
            return result

        t = perf_counter()
        qv_so = self._encode_text(self.m_so, self.tk_so, q)
        encode_so_ms = (perf_counter() - t) * 1000
        if getattr(self.F_so, "dtype", None) == np.float16:
            qv_so = qv_so.astype(np.float16)
        t = perf_counter()
        frame_so = self.F_so @ qv_so
        vid_so = self._maxvec(frame_so)
        search_so_ms = (perf_counter() - t) * 1000

        zv = self._zscore(vid_vitl)
        zs = self._zscore(vid_so)
        fused = self.alpha * zv + (1.0 - self.alpha) * zs
        self.route_used = False
        self.route_reason = "confident"
        self.baseline_top1 = str(self.all_vids[np.argmax(fused)])
        selected_frame_vitl = frame_vitl
        selected_frame_so = frame_so
        selected_norm_pair = (zv, zs)
        nllb_ms = 0.0
        if self.nllb_routing and len(self.all_vids) > 1:
            base_order = np.argsort(-fused)
            margin = float(fused[base_order[0]] - fused[base_order[1]])
            self.baseline_margin = margin
            if margin < self.nllb_threshold:
                nllb_load_this_query_ms = self._ensure_nllb_model()
                nllb_started = perf_counter()
                # The full-corpus NLLB benchmark uses the same English caption
                # protocol as the production ViT-L/SO400M path. Keep routing
                # comparable by using the translated query, not raw Vietnamese.
                qn = self._encode_text(self.m_nllb, self.tk_nllb, q)
                if getattr(self.F_nllb, "dtype", None) == np.float16:
                    qn = qn.astype(np.float16)
                frame_nllb = self.F_nllb @ qn
                nllb = self._zscore(self._maxvec(frame_nllb))
                fused = nllb
                selected_frame_vitl = frame_nllb
                selected_frame_so = None
                selected_norm_pair = None
                nllb_ms = (perf_counter() - nllb_started) * 1000
                self.route_used = True
                self.route_reason = "low_margin"
        elif not self.nllb_routing:
            self.baseline_margin = None
        order = np.argsort(-fused)[:topk]
        self._last_frame_scores = (selected_frame_vitl, selected_frame_so, selected_norm_pair)
        self._last_video_order = order
        t = perf_counter(); result = self._format_results(
            order, fused, selected_frame_vitl, selected_frame_so, selected_norm_pair)
        self.last_timings_ms = {"translate": translate_ms, "encode_vitl": encode_vitl_ms,
                                "search_vitl": search_vitl_ms, "encode_so400m": encode_so_ms,
                                "search_so400m": search_so_ms, "nllb": nllb_ms,
                                "nllb_load": nllb_load_this_query_ms,
                                "nllb_load_total": self.nllb_load_ms,
                                "route_used": self.route_used,
                                "route_reason": self.route_reason,
                                "baseline_margin": self.baseline_margin,
                                "baseline_top1": self.baseline_top1,
                                "format": (perf_counter()-t)*1000,
                                "total": (perf_counter()-started)*1000}
        return result

    def search_peaks(self, query_vn: str, topk=20, peaks_per_video=5,
                     min_gap_kf=3, min_gap_s=2.0):
        """Search and attach diverse temporal peaks without changing ranking."""
        try:
            from .frame_peaks import select_temporal_peaks
        except ImportError:
            from frame_peaks import select_temporal_peaks

        results = self.search(query_vn, topk=topk)
        peaks = []
        frame_vitl, frame_so, norm_pair = self._last_frame_scores
        for result in results:
            video_id = str(result[0])
            indices = self._video_rows[video_id]
            if frame_so is not None and norm_pair is not None:
                zv, zs = norm_pair
                scores = self.alpha * self._zscore(frame_vitl[indices]) + (1.0 - self.alpha) * self._zscore(frame_so[indices])
            else:
                scores = frame_vitl[indices]
            rows = self.km.iloc[indices].sort_values("pts_time").reset_index(drop=True)
            # Re-align scores after the temporal sort.
            scores = scores[np.argsort(self.km.iloc[indices].pts_time.to_numpy())]
            peaks.append({"video_id": video_id, "peaks": select_temporal_peaks(
                rows, scores, count=peaks_per_video, min_gap_kf=min_gap_kf, min_gap_s=min_gap_s)})
        return {"results": results, "peaks": peaks}

    def _format_results(self, order, fused_scores, frame_vitl, frame_so, norm_pair):
        results = []
        for j in order:
            vid = str(self.all_vids[j])
            if frame_so is not None and norm_pair is not None:
                best = self._best_fused_frame(vid, frame_vitl, frame_so)
            else:
                best = self._best_frame(vid, frame_vitl)
            results.append((vid, int(best.frame_idx), int(best.kf_n), float(fused_scores[j])))
        return results


if __name__ == "__main__":
    r = KISFusionRetriever(translate=False)
    for row in r.search("Một nữ MC mặc áo blazer nâu đứng trong studio", topk=5):
        print(row)
