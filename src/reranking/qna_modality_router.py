"""Offline modality candidates for Q&A diagnostics.

The visual KIS retriever remains the production default. This module only
adds bounded ASR/OCR frame hypotheses for queries whose annotation explicitly
requires that modality. Scores are not compared with visual scores; callers
keep visual video order and treat these as same-video alternatives.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))
from offline_fallback import TextEmbedderOffline  # noqa: E402
from src.utils.paths import INDEX_DIR  # noqa: E402
from src.reranking.modality_index_preflight import (  # noqa: E402
    ModalityIndexPreflightError,
    normalise_expected_packs,
    require_modality_index,
    run_modality_index_preflight,
)
from src.reranking.modality_index_registry import (  # noqa: E402
    ModalityIndexRegistry,
    ModalityIndexRegistryError,
)
from src.reranking.local_lexical_index import BM25Index
from src.vqa.evidence_fusion import build_evidence_packet


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


class QNAModalityRouter:
    """Lazy, local ASR/OCR candidate generator with scoped global retrieval."""

    def __init__(self, index_dir: str | Path | None = None,
                 embedder: TextEmbedderOffline | None = None,
                 model_dir: str | Path | None = None, strict: bool = False,
                 expected_packs: Iterable[str] | None = None,
                 text_mode: str = "dense",
                 active_modalities: Iterable[str] | None = None):
        self.index_dir = Path(index_dir or INDEX_DIR)
        self.strict = bool(strict)
        self.expected_packs = normalise_expected_packs(expected_packs)
        self._expected_pack_set = set(self.expected_packs)
        self.active_modalities = tuple(dict.fromkeys(
            str(value).strip().lower() for value in (active_modalities or ("asr", "ocr"))
            if str(value).strip().lower() in {"asr", "ocr"}
        ))
        if not self.active_modalities:
            raise ValueError("active_modalities must contain asr and/or ocr")
        self.model_dir = Path(model_dir) if model_dir else None
        self.text_mode = str(text_mode).strip().lower()
        if self.text_mode not in {"dense", "bm25"}:
            raise ValueError("text_mode must be 'dense' or 'bm25'")
        if self.strict and self.text_mode == "dense" and self.model_dir is not None and not self.model_dir.exists():
            raise FileNotFoundError(f"local modality model is missing: {self.model_dir}")
        if self.strict:
            try:
                self.preflight_report = run_modality_index_preflight(
                    self.index_dir, expected_packs=self.expected_packs,
                    active_modalities=self.active_modalities,
                    require_embeddings=self.text_mode != "bm25",
                )
                require_modality_index(self.preflight_report)
            except ModalityIndexPreflightError:
                raise
            except Exception as exc:
                raise RuntimeError(f"local modality index preflight could not run: {exc}") from exc
        else:
            self.preflight_report = None
        # Keep one source resolver for all ASR reads.  In particular, a ready
        # merged ASR manifest must win over historical per-pack shards; mixing
        # the two sources would make candidate provenance and coverage opaque.
        self.registry = ModalityIndexRegistry(self.index_dir)
        self.embedder = embedder
        if self.text_mode == "dense" and self.embedder is None:
            self.embedder = TextEmbedderOffline(
                device="cpu", model_name=str(self.model_dir) if self.model_dir else "BAAI/bge-m3")
        self._indexes: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
        self._lexical_indexes: dict[str, BM25Index] = {}
        self._embedder_error: str | None = None

    def _load(self, modality: str) -> tuple[np.ndarray, pd.DataFrame]:
        modality = str(modality).lower()
        if modality in self._indexes:
            return self._indexes[modality]
        registry = getattr(self, "registry", None)
        if registry is None:
            registry = ModalityIndexRegistry(getattr(self, "index_dir", INDEX_DIR))
            self.registry = registry
        if modality == "asr":
            # A present global source is authoritative.  Never silently
            # downgrade to root shards when it is malformed or incomplete.
            # Strict (production) routing also requires the global manifest:
            # per-pack shards are diagnostic artifacts, never a substitute
            # for a corpus-level ASR retrieval contract.
            global_source = registry.discover_global_asr()
            if global_source is not None:
                try:
                    global_emb, global_meta, global_info = registry.load_asr(
                        expected_packs=getattr(self, "expected_packs", None),
                        require_embeddings=self.text_mode != "bm25",
                        strict=self.strict,
                    )
                except ModalityIndexRegistryError as exc:
                    raise RuntimeError(
                        f"merged ASR global index is not usable: {exc}"
                    ) from exc
                global_meta = global_meta.copy()
                global_meta["global_idx"] = np.arange(len(global_meta), dtype=np.int64)
                # Keep provenance available to answer/evidence consumers while
                # exposing the stable router column name (chunk).
                global_meta.attrs["modality_source"] = global_info
                if self.text_mode == "bm25":
                    global_emb = np.zeros((len(global_meta), 1), dtype=np.float32)
                else:
                    global_emb = _normalise_rows(global_emb)
                self._indexes[modality] = (global_emb, global_meta)
                return self._indexes[modality]
            if self.strict:
                raise RuntimeError(
                    "strict ASR routing requires a ready global ASR manifest; "
                    "legacy per-pack shards are diagnostic-only"
                )
        if modality == "ocr" and registry.discover_global_ocr() is not None:
            try:
                global_emb, global_meta, global_info = registry.load_ocr(
                    expected_packs=getattr(self, "expected_packs", None),
                    require_embeddings=self.text_mode != "bm25",
                    strict=self.strict,
                )
            except ModalityIndexRegistryError as exc:
                raise RuntimeError(
                    f"global OCR index is not usable: {exc}"
                ) from exc
            global_meta = global_meta.copy()
            global_meta["global_idx"] = np.arange(len(global_meta), dtype=np.int64)
            global_meta.attrs["modality_source"] = global_info
            if self.text_mode == "bm25":
                global_emb = np.zeros((len(global_meta), 1), dtype=np.float32)
            else:
                global_emb = _normalise_rows(global_emb)
            self._indexes[modality] = (global_emb, global_meta)
            return self._indexes[modality]
        if modality == "ocr" and self.strict:
            raise RuntimeError(
                "strict OCR routing requires a ready global OCR manifest; "
                "legacy per-pack shards are diagnostic-only"
            )
        features: list[np.ndarray] = []
        metadata: list[pd.DataFrame] = []
        offset = 0
        if modality == "asr":
            files = sorted(
                self.index_dir.glob("asr_chunks_*_ts.parquet")
                if self.text_mode == "bm25"
                else self.index_dir.glob("emb_cache_asr_*_chunks.npy")
            )
            if self.strict and not files:
                raise FileNotFoundError(f"no local ASR embedding packs in {self.index_dir}")
            for emb_path in files:
                pack = (emb_path.stem.replace("asr_chunks_", "").replace("_ts", "")
                        if self.text_mode == "bm25" else
                        emb_path.name.replace("emb_cache_asr_", "").replace("_chunks.npy", ""))
                if pack.lower() not in self._expected_pack_set:
                    continue
                meta_path = (emb_path if self.text_mode == "bm25" else
                             self.index_dir / f"asr_chunks_{pack}_ts.parquet")
                if not meta_path.exists():
                    continue
                meta = pd.read_parquet(meta_path).reset_index(drop=True)
                if self.text_mode == "bm25":
                    emb = np.zeros((len(meta), 1), dtype=np.float32)
                else:
                    emb = np.load(emb_path).astype(np.float32)
                    if len(emb) != len(meta):
                        continue
                meta = meta.rename(columns={"vid": "video_id"})
                required = {"video_id", "kf_n", "frame_idx", "start", "end"}
                if not required.issubset(meta.columns):
                    continue
                keep = meta.kf_n.notna().to_numpy()
                meta = meta.loc[keep].copy().reset_index(drop=True)
                emb = emb[keep]
                meta["pts_time"] = (meta.start.astype(float) + meta.end.astype(float)) / 2.0
                meta["global_idx"] = np.arange(len(meta), dtype=np.int64) + offset
                offset += len(meta)
                features.append(emb)
                metadata.append(meta[["video_id", "kf_n", "frame_idx", "pts_time", "start", "end", "global_idx", "chunk"]])
        elif modality == "ocr":
            files = sorted(self.index_dir.glob("ocr_*.parquet"))
            for meta_path in files:
                if any(token in meta_path.name for token in ("_partial", "_compare", "_gt", "ocr_query", "ocr_chunks")):
                    continue
                pack = meta_path.stem.replace("ocr_", "").lower()
                if pack not in self._expected_pack_set:
                    continue
                emb_path = self.index_dir / f"emb_cache_ocr_{meta_path.stem.replace('ocr_', '')}.npy"
                if self.text_mode != "bm25" and not emb_path.exists():
                    continue
                meta = pd.read_parquet(meta_path).reset_index(drop=True)
                if self.text_mode == "bm25":
                    emb = np.zeros((len(meta), 1), dtype=np.float32)
                else:
                    emb = np.load(emb_path).astype(np.float32)
                    if len(emb) != len(meta):
                        continue
                required = {"video_id", "kf_n", "pts_time"}
                if not required.issubset(meta.columns):
                    continue
                keep = meta.kf_n.notna().to_numpy()
                meta = meta.loc[keep].copy().reset_index(drop=True)
                emb = emb[keep]
                meta["global_idx"] = np.arange(len(meta), dtype=np.int64) + offset
                offset += len(meta)
                features.append(emb)
                metadata.append(meta[["video_id", "kf_n", "pts_time", "global_idx", "ocr_text"]])
            if self.strict and not features:
                raise FileNotFoundError(f"no local OCR embedding packs in {self.index_dir}")
        else:
            raise ValueError(f"unsupported modality: {modality}")

        result = ((np.empty((0, 0), dtype=np.float32), pd.DataFrame()) if not features else
                  (_normalise_rows(np.vstack(features)), pd.concat(metadata, ignore_index=True)))
        if self.strict and len(result[1]) == 0:
            raise RuntimeError(f"local {modality} index is empty after validation")
        self._indexes[modality] = result
        return result

    def candidate_frames(self, text: str, modality: str, video_ids: list[str],
                         per_video: int = 2) -> list[dict]:
        """Return bounded same-video candidates ranked within each video."""
        if not text or not video_ids or per_video < 1:
            return []
        emb, meta = self._load(modality)
        if len(meta) == 0:
            return []
        allowed = {str(value) for value in video_ids}
        indices = np.flatnonzero(meta.video_id.astype(str).isin(allowed).to_numpy())
        if len(indices) == 0:
            return []
        work = meta.iloc[indices].copy()
        try:
            if self.text_mode == "bm25":
                work["_score"] = self._get_lexical_index(modality, meta).scores(text)[indices]
                score_mode = "bm25"
            else:
                query = self.embedder.embed([str(text)], batch_size=1, normalize=True)[0]
                work["_score"] = emb[indices] @ query
                score_mode = "embedding"
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"strict modality embedding failed: {exc}") from exc
            # The production path must remain usable on a minimal offline
            # install. A lexical rescue is weaker than bge-m3 but deterministic
            # and still grounded in the local transcript/OCR index.
            self._embedder_error = f"{type(exc).__name__}: {exc}"
            query_tokens = set(re.findall(r"[\wÀ-ỹ]+", str(text).casefold()))
            text_column = "chunk" if modality == "asr" else "ocr_text"
            def lexical(value):
                tokens = set(re.findall(r"[\wÀ-ỹ]+", str(value or "").casefold()))
                return len(query_tokens & tokens) / max(len(query_tokens), 1)
            work["_score"] = work[text_column].map(lexical).astype(np.float32)
            score_mode = "bm25_fallback"
        out = []
        for video_id, group in work.groupby(work.video_id.astype(str), sort=False):
            group = group.sort_values(["_score", "kf_n"], ascending=[False, True]).head(per_video)
            for _, row in group.iterrows():
                item = {"video_id": str(video_id), "kf_n": int(row.kf_n),
                        "pts_time": float(row.pts_time), "modality": str(modality),
                        "modality_score": float(row["_score"]), "score_mode": score_mode}
                text_column = "chunk" if modality == "asr" else "ocr_text"
                evidence_text = str(row.get(text_column, "") or "").strip()
                if evidence_text:
                    # The video-level rescue policy is evidence-gated. Keep
                    # the matched text attached to the ranked hit so the
                    # policy can distinguish a real specialist rescue from a
                    # score-only candidate without re-opening the index.
                    item["text"] = evidence_text
                    item["evidence"] = {
                        "modality": str(modality),
                        "text": evidence_text,
                    }
                if "frame_idx" in row.index and not pd.isna(row["frame_idx"]):
                    item["frame_idx"] = int(row["frame_idx"])
                out.append(item)
        return out

    def evidence_packet_for_candidate(
        self,
        candidate: dict,
        query: str,
        question: str,
        *,
        modalities: Iterable[str] | None = None,
    ) -> dict:
        """Build answer evidence from the same global index used for retrieval.

        The routed Q&A path must not retrieve from the merged global index and
        then answer from the historical per-pack shards.  That split made a
        specialist rescue visible in ranking but invisible to the answer
        provider.  This method deliberately reuses the router's cached,
        preflighted metadata and hands only same-video rows to the shared
        evidence packager.
        """
        video_id = str(candidate.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("candidate requires video_id for modality evidence")
        requested = tuple(dict.fromkeys(
            str(value).strip().lower()
            for value in (modalities if modalities is not None else self.active_modalities)
            if str(value).strip().lower() in {"asr", "ocr"}
        ))
        if not requested:
            raise ValueError("modality evidence requires asr and/or ocr")

        asr_rows = pd.DataFrame()
        ocr_rows = pd.DataFrame()
        modality_status: dict[str, dict] = {}
        for modality in requested:
            _, metadata = self._load(modality)
            source_info = dict(getattr(metadata, "attrs", {}).get("modality_source", {}) or {})
            if "video_id" not in metadata.columns:
                same_video = metadata.iloc[0:0].copy()
            else:
                same_video = metadata[
                    metadata["video_id"].astype(str) == video_id
                ].copy()
            no_speech_videos = {
                str(value).strip().upper()
                for value in source_info.get("no_speech_videos", ()) or ()
                if str(value).strip()
            }
            no_text_videos: set[str] = set()
            if modality == "ocr" and source_info.get("source") == "global_merged_v2":
                try:
                    source = self.registry.discover_global_ocr()
                    coverage = source.manifest.get("coverage", {}) if source else {}
                    raw_no_text = coverage.get("no_text_videos", ()) if isinstance(coverage, dict) else ()
                    no_text_videos = {
                        str(value).strip().upper() for value in (raw_no_text or ())
                        if str(value).strip()
                    }
                except Exception:
                    # The registry load already performed the authoritative
                    # artifact checks.  Missing optional coverage metadata is
                    # reported as coverage_missing rather than guessed as
                    # no-text.
                    no_text_videos = set()
            if video_id.upper() in no_speech_videos:
                status = "no_speech"
            elif video_id.upper() in no_text_videos:
                status = "no_text"
            elif len(metadata) == 0 and not source_info:
                status = "index_missing"
            elif len(same_video) == 0:
                # A materialized global source exists, but this video has no
                # rows in it.  This is coverage/missing-row state, not a
                # transcript or OCR mismatch and must remain observable.
                status = "coverage_missing"
            else:
                text_column = "chunk" if modality == "asr" else "ocr_text"
                usable = same_video[text_column].fillna("").astype(str).str.strip()
                status = "no_speech" if modality == "asr" and not usable.astype(bool).any() else (
                    "no_text" if modality == "ocr" and not usable.astype(bool).any() else "available"
                )
            modality_status[modality] = {
                "status": status,
                "requested": True,
                "index_source": source_info.get("source", "legacy_or_local"),
                "row_count": int(len(same_video)),
                "available": status == "available",
                "coverage": status not in {"index_missing", "coverage_missing"},
            }
            if modality == "asr":
                asr_rows = same_video
            else:
                ocr_rows = same_video

        for modality in {"asr", "ocr"} - set(requested):
            modality_status[modality] = {
                "status": "inactive", "requested": False, "available": False,
                "coverage": False, "row_count": 0,
            }

        packet = build_evidence_packet(
            candidate,
            asr_rows=asr_rows,
            ocr_rows=ocr_rows,
            query=query,
            question=question,
            modality_status=modality_status,
        )
        packet["index_source"] = "global_modality_registry"
        packet["index_modalities"] = list(requested)
        packet["canonical_mapping"]["status"] = "registry_validated" if any(
            modality_status[modality].get("index_source") == "global_merged_v2"
            for modality in requested
        ) else "candidate_supplied"
        return packet

    def global_candidates(self, text: str, modality: str, topk: int = 100) -> list[dict]:
        """Rank global modality frames with dense or local BM25 text search."""
        if not text or topk < 1:
            return []
        emb, meta = self._load(modality)
        if len(meta) == 0:
            return []
        try:
            if self.text_mode == "bm25":
                scores = self._get_lexical_index(modality, meta).scores(text)
                score_mode = "bm25"
            else:
                query = self.embedder.embed([str(text)], batch_size=1, normalize=True)[0]
                scores = emb @ query
                score_mode = "embedding"
        except Exception as exc:
            raise RuntimeError(f"global modality retrieval unavailable: {exc}") from exc
        topk = min(int(topk), len(scores))
        order = np.argpartition(-scores, topk - 1)[:topk]
        order = order[np.argsort(-scores[order], kind="stable")]
        out = []
        for index in order:
            row = meta.iloc[int(index)]
            item = {"video_id": str(row.video_id), "kf_n": int(row.kf_n),
                    "pts_time": float(row.pts_time), "modality": str(modality),
                    "modality_score": float(scores[index]), "rank": len(out) + 1,
                    "score_mode": score_mode}
            text_column = "chunk" if modality == "asr" else "ocr_text"
            evidence_text = str(row.get(text_column, "") or "").strip()
            if evidence_text:
                item["text"] = evidence_text
                item["evidence"] = {
                    "modality": str(modality),
                    "text": evidence_text,
                }
            if "frame_idx" in row.index and not pd.isna(row["frame_idx"]):
                item["frame_idx"] = int(row["frame_idx"])
            out.append(item)
        return out

    def _get_lexical_index(self, modality: str, meta: pd.DataFrame) -> BM25Index:
        """Build and cache a BM25 index over the loaded modality rows."""
        modality = str(modality).lower()
        cached = self._lexical_indexes.get(modality)
        if cached is not None:
            return cached
        text_column = "chunk" if modality == "asr" else "ocr_text"
        if text_column not in meta.columns:
            raise RuntimeError(f"local {modality} metadata has no {text_column!r} column")
        index = BM25Index(meta[text_column].fillna("").astype(str).tolist())
        self._lexical_indexes[modality] = index
        return index
