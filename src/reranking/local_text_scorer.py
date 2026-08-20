"""Local ASR/OCR scorers for candidate reranking.

These scorers operate on per-candidate temporal windows rather than global
video-level pooling. They are intentionally independent from production KIS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "utils"))
sys.path.insert(0, str(REPO / "src" / "core"))

from paths import INDEX_DIR  # noqa: E402
from offline_fallback import TextEmbedderOffline  # noqa: E402


class LocalASRScorer:
    """Score a video candidate by ASR chunk similarity in a temporal window.

    Loads all K-series ASR embeddings once and provides per-candidate scoring
    without re-encoding the index.
    """

    def __init__(self, window_s: float = 60.0, embedder: TextEmbedderOffline | None = None):
        self.window_s = window_s
        self._embedder = embedder
        self._own_embedder = embedder is None
        self._chunks_meta: pd.DataFrame | None = None
        self._chunks_emb: np.ndarray | None = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        idx = Path(INDEX_DIR)
        emb_files = sorted(idx.glob("emb_cache_asr_k*_chunks.npy"))
        feats, metas = [], []
        for emb_fp in emb_files:
            pack_l = emb_fp.name.replace("emb_cache_asr_", "").replace("_chunks.npy", "")
            chunks_fp = idx / f"asr_chunks_{pack_l}_ts.parquet"
            if not emb_fp.exists() or not chunks_fp.exists():
                continue
            emb = np.load(emb_fp).astype(np.float32)
            df = pd.read_parquet(chunks_fp).reset_index(drop=True)
            if len(df) != len(emb):
                continue
            df = df.rename(columns={"vid": "video_id"})
            df["global_asr_idx"] = np.arange(len(emb)) + sum(len(x) for x in feats)
            feats.append(emb)
            metas.append(
                df[["video_id", "start", "end", "frame_idx", "kf_n", "global_asr_idx"]]
            )
        if not feats:
            raise FileNotFoundError("No ASR embeddings found")
        self._chunks_emb = np.vstack(feats).astype(np.float32)
        norms = np.linalg.norm(self._chunks_emb, axis=1, keepdims=True).clip(min=1e-8)
        self._chunks_emb /= norms
        self._chunks_meta = pd.concat(metas, ignore_index=True)
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def chunks_meta(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self._chunks_meta  # type: ignore[return-value]

    @property
    def chunks_emb(self) -> np.ndarray:
        self._ensure_loaded()
        return self._chunks_emb  # type: ignore[return-value]

    @property
    def embedder(self) -> TextEmbedderOffline:
        if self._embedder is None:
            self._embedder = TextEmbedderOffline()
        return self._embedder

    def score(self, query_text: str, video_id: str, pts_time: float) -> float:
        """Cosine similarity between query embedding and best ASR chunk in window."""
        self._ensure_loaded()
        mask = (
            (self._chunks_meta["video_id"] == video_id)
            & (self._chunks_meta["start"] <= pts_time + self.window_s)
            & (self._chunks_meta["end"] >= pts_time - self.window_s)
        )
        indices = self._chunks_meta.loc[mask, "global_asr_idx"].values
        if len(indices) == 0:
            return 0.0
        q_emb = self.embedder.embed(query_text, batch_size=1, normalize=True)[0]
        chunk_vecs = self._chunks_emb[indices]
        return float(np.max(chunk_vecs @ q_emb))

    def score_batch(self, queries: list[tuple[str, str, float]]) -> np.ndarray:
        """Batch score: queries is list of (query_text, video_id, pts_time)."""
        self._ensure_loaded()
        texts = [q[0] for q in queries]
        q_embs = self.embedder.embed(texts, batch_size=min(32, len(texts)), normalize=True)
        scores = np.zeros(len(queries), dtype=np.float32)
        for i, (_, video_id, pts_time) in enumerate(queries):
            mask = (
                (self._chunks_meta["video_id"] == video_id)
                & (self._chunks_meta["start"] <= pts_time + self.window_s)
                & (self._chunks_meta["end"] >= pts_time - self.window_s)
            )
            indices = self._chunks_meta.loc[mask, "global_asr_idx"].values
            if len(indices) == 0:
                continue
            scores[i] = float(np.max(self._chunks_emb[indices] @ q_embs[i]))
        return scores

    def __del__(self):
        if self._own_embedder and self._embedder is not None:
            del self._embedder
            self._embedder = None
