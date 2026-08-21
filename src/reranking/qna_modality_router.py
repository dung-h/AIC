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
from src.reranking.local_lexical_index import BM25Index, normalise_text, quote_match, tokenize
from src.vqa.evidence_fusion import build_evidence_packet
from src.vqa.claim_verifier import ClaimPolicy, score_claim


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
                 active_modalities: Iterable[str] | None = None,
                 asr_global_dir: str | Path | None = None):
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
        self.registry = ModalityIndexRegistry(
            self.index_dir, global_asr_dir=asr_global_dir
        )
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
                # The portable raw-Deepgram materializer deliberately emits
                # the global schema (``text``/``video_id``).  Historical K
                # shards use ``chunk``/``vid``.  Normalise before the common
                # candidate path so a rebuilt pack can be tested locally
                # before it is promoted into the merged global artifact.
                if {"text", "video_id"}.issubset(meta.columns):
                    meta = meta.rename(columns={"text": "chunk"})
                elif "vid" in meta.columns:
                    meta = meta.rename(columns={"vid": "video_id"})
                required = {"video_id", "kf_n", "frame_idx", "start", "end"}
                if not required.issubset(meta.columns):
                    continue
                keep = meta.kf_n.notna().to_numpy()
                meta = meta.loc[keep].copy().reset_index(drop=True)
                emb = emb[keep]
                if "pts_time" not in meta.columns:
                    meta["pts_time"] = (meta.start.astype(float) + meta.end.astype(float)) / 2.0
                else:
                    meta["pts_time"] = pd.to_numeric(meta["pts_time"], errors="coerce")
                    valid_pts = meta["pts_time"].notna().to_numpy()
                    meta = meta.loc[valid_pts].copy().reset_index(drop=True)
                    emb = emb[valid_pts]
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
        claim_policy: ClaimPolicy | None = None,
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
        self._attach_claim_evidence(
            packet,
            asr_rows=asr_rows,
            ocr_rows=ocr_rows,
            claim_policy=claim_policy,
        )
        self._attach_video_role_join(
            packet,
            candidate,
            query=query,
            question=question,
            asr_rows=asr_rows,
            ocr_rows=ocr_rows,
        )
        packet["index_source"] = "global_modality_registry"
        packet["index_modalities"] = list(requested)
        packet["canonical_mapping"]["status"] = "registry_validated" if any(
            modality_status[modality].get("index_source") == "global_merged_v2"
            for modality in requested
        ) else "candidate_supplied"
        return packet

    @staticmethod
    def _attach_claim_evidence(
        packet: dict,
        *,
        asr_rows: pd.DataFrame,
        ocr_rows: pd.DataFrame,
        claim_policy: ClaimPolicy | None,
    ) -> None:
        """Attach best same-video proof for query-side claims.

        This is not a second global retrieval and cannot rescue a video.  It
        only scans metadata already scoped to the candidate's one video, keeps
        canonical coordinates, and emits a row only if its deterministic claim
        threshold is met.
        """
        packet["claim_evidence"] = []
        if claim_policy is None or not claim_policy.active:
            return
        rows_by_source = (("asr", asr_rows, "chunk"), ("ocr", ocr_rows, "ocr_text"))
        attached: list[dict] = []
        for claim in claim_policy.claims:
            for source, rows, text_column in rows_by_source:
                if rows is None or len(rows) == 0 or text_column not in rows.columns:
                    continue
                best_score = 0.0
                best_row = None
                for _, row in rows.iterrows():
                    text = str(row.get(text_column, "") or "").strip()
                    score = score_claim(claim, text)
                    if score > best_score:
                        best_score = score
                        best_row = row
                if best_row is None or best_score < claim.min_score:
                    continue
                try:
                    kf_n = int(best_row["kf_n"])
                    frame_idx = int(best_row["frame_idx"])
                    pts_time = float(best_row["pts_time"])
                except (KeyError, TypeError, ValueError):
                    continue
                text = str(best_row.get(text_column, "") or "").strip()
                if not text:
                    continue
                row = {
                    "source": source,
                    "text": text,
                    "kf_n": kf_n,
                    "frame_idx": frame_idx,
                    "timestamp": pts_time,
                    "role": "claim_support",
                    "claim": claim.to_dict(),
                    "claim_score": round(float(best_score), 6),
                }
                if source == "asr":
                    row["start_time"] = float(best_row.get("start", pts_time) or pts_time)
                    row["end_time"] = float(best_row.get("end", pts_time) or pts_time)
                attached.append(row)
        attached.sort(key=lambda row: (
            str(row["claim"]["kind"]), str(row["claim"]["text"]).casefold(),
            str(row["source"]), int(row["kf_n"]),
        ))
        packet["claim_evidence"] = attached

    @staticmethod
    def _specialist_role_join_anchor(candidate: dict) -> dict | None:
        """Recover a real ASR/OCR row after visual/specialist deduplication.

        The selector intentionally makes visual the top-level source for a
        same-frame merge.  A role join, however, needs the actual specialist
        text and provenance that justified the join.  Do not infer either
        from the visual anchor: use only preserved specialist provenance.
        """
        records = [candidate, *(candidate.get("provenance", ()) or ())]
        for raw in records:
            if not isinstance(raw, dict):
                continue
            source = str(raw.get("source", "")).strip().lower()
            if source not in {"asr", "ocr"}:
                continue
            evidence = raw.get("evidence")
            evidence_text = evidence.get("text", "") if isinstance(evidence, dict) else ""
            text = str(raw.get("text") or evidence_text or "").strip()
            if not text:
                continue
            try:
                int(raw.get("kf_n", candidate.get("kf_n")))
                int(raw.get("frame_idx", candidate.get("frame_idx")))
                float(raw.get("pts_time", candidate.get("pts_time")))
            except (TypeError, ValueError):
                continue
            anchor = {
                **candidate,
                **raw,
                "source": source,
                "sources": [source],
                "text": text,
                "evidence": {"modality": source, "text": text},
                "kf_n": int(raw.get("kf_n", candidate.get("kf_n"))),
                "frame_idx": int(raw.get("frame_idx", candidate.get("frame_idx"))),
                "pts_time": float(raw.get("pts_time", candidate.get("pts_time"))),
            }
            return anchor
        return None

    @classmethod
    def _strong_role_join_anchor(cls, candidate: dict, query: str, question: str) -> bool:
        """Require auditable local specialist evidence before a video join.

        Role joining is intentionally not a generic second retrieval pass.
        A candidate must have actual ASR/OCR text plus either a high lexical
        coverage record or a numeric fact with at least two matching
        descriptors.  This lets ``200g thịt nạc dăm xay`` open a same-video
        title lookup while rejecting an arbitrary visual frame.
        """
        candidate = cls._specialist_role_join_anchor(candidate)
        if candidate is None:
            return False
        evidence = candidate.get("evidence")
        evidence_text = evidence.get("text", "") if isinstance(evidence, dict) else ""
        text = str(candidate.get("text") or evidence_text or "").strip()
        if not text:
            return False
        for entry in candidate.get("view_provenance", ()) or ():
            if not isinstance(entry, dict) or entry.get("score_mode") != "bm25_coverage":
                continue
            try:
                if float(entry.get("score", 0.0)) >= 0.90:
                    return True
            except (TypeError, ValueError):
                continue
        request_tokens = set(tokenize(f"{query}\n{question}"))
        evidence_tokens = set(tokenize(text))
        numeric_tokens = {
            token for token in request_tokens
            if any(char.isdigit() for char in token)
        }
        descriptor_overlap = {
            token for token in request_tokens.intersection(evidence_tokens)
            if not any(char.isdigit() for char in token) and len(token) >= 3
        }
        return bool(numeric_tokens.intersection(evidence_tokens)) and len(descriptor_overlap) >= 2

    @staticmethod
    def _role_join_rows(rows: pd.DataFrame, modality: str) -> list[dict]:
        """Materialize canonical same-video metadata with an explicit source."""
        if rows is None or len(rows) == 0:
            return []
        text_column = "chunk" if modality == "asr" else "ocr_text"
        required = {"video_id", "kf_n", "frame_idx", "pts_time", text_column}
        if not required.issubset(rows.columns):
            return []
        result: list[dict] = []
        for raw in rows.to_dict("records"):
            item = dict(raw)
            item["modality"] = modality
            result.append(item)
        return result

    def _attach_video_role_join(
        self,
        packet: dict,
        candidate: dict,
        *,
        query: str,
        question: str,
        asr_rows: pd.DataFrame,
        ocr_rows: pd.DataFrame,
    ) -> None:
        """Append bounded same-video title evidence without altering the anchor.

        The submission frame remains owned by the selector.  This only gives
        the answer provider an explicitly traced ASR/OCR row when a strong
        ingredient/name anchor identifies the video but the title is stated
        elsewhere in that same short programme.
        """
        from src.vqa.video_evidence_join import VideoEvidenceRoleJoiner

        joiner = VideoEvidenceRoleJoiner()
        joined: list[dict] = []
        specialist_anchor = self._specialist_role_join_anchor(candidate)
        if (specialist_anchor is not None
                and self._strong_role_join_anchor(candidate, query, question)):
            rows = [
                *self._role_join_rows(asr_rows, "asr"),
                *self._role_join_rows(ocr_rows, "ocr"),
            ]
            joined = joiner.join(
                rows,
                query,
                question,
                anchor_candidate={**specialist_anchor, "is_strong": True},
            )
        else:
            joiner.last_diagnostic = {"status": "inactive_weak_or_non_specialist_anchor"}

        packet["video_evidence_join"] = {
            "diagnostic": dict(joiner.last_diagnostic),
            "support_rows": [
                {
                    key: item.get(key)
                    for key in ("video_id", "kf_n", "frame_idx", "pts_time", "modality", "role", "score")
                }
                for item in joined
            ],
        }
        if not joined:
            return

        # A cooking programme's recurring teaser ("Món ngon mỗi ngày …")
        # can mention a *different* recipe inside the ordinary time window.
        # Once the joiner has a same-video, explicit dish-introduction row,
        # do not feed that boilerplate alongside the answer support; it is
        # demonstrably conflicting context, not corroboration.  This narrow
        # filter applies only to dish-name joins and retains every other ASR
        # context item unchanged.
        suppress_generic_recipe_teasers = (
            joiner.last_diagnostic.get("intent") == "dish_name"
            and any(str(item.get("modality")) == "asr" for item in joined)
        )
        suppressed_context_rows = 0

        for support in joined:
            modality = str(support["modality"])
            target_key = "asr_chunks" if modality == "asr" else "ocr_text"
            evidence_row = {
                "source": modality,
                "text": str(support["source_text"]),
                "start_time": float(support["pts_time"]),
                "end_time": float(support["pts_time"]),
                "timestamp": float(support["pts_time"]),
                "rank": int(support.get("rank", 0) or 0),
                "distance_s": abs(float(support["pts_time"]) - float(candidate.get("pts_time", 0.0))),
                "role": "answer_support",
                "provenance": dict(support.get("provenance", {}) or {}),
            }
            existing = list(packet.get(target_key, ()) or ())
            if suppress_generic_recipe_teasers and target_key == "asr_chunks":
                kept = [
                    item for item in existing
                    if "mon ngon moi ngay" not in normalise_text(item.get("text", ""))
                ]
                suppressed_context_rows += len(existing) - len(kept)
                existing = kept
            deduped = [evidence_row]
            # ASR chunks may carry an utterance timestamp while their
            # canonical keyframe timestamp differs. Textual identity is the
            # relevant duplicate boundary here; keep the joined canonical
            # record and avoid showing the same title twice to the VLM.
            seen = {(modality, normalise_text(evidence_row["text"]))}
            for item in existing:
                key = (
                    str(item.get("source", modality)),
                    normalise_text(item.get("text", "")),
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            # The role support is deliberately first. A later context window
            # cannot evict the named answer fact due to an arbitrary time sort.
            packet[target_key] = deduped[:5]
            if modality not in packet.setdefault("sources", []):
                packet["sources"].append(modality)
            timestamps = packet.setdefault("timestamps", [])
            timestamp_key = (modality, evidence_row["start_time"], evidence_row["text"])
            existing_timestamps = {
                (str(item.get("source", "")), item.get("start_time"), str(item.get("text", "")))
                for item in timestamps if isinstance(item, dict)
            }
            if timestamp_key not in existing_timestamps:
                timestamps.append({
                    "source": modality,
                    "start_time": evidence_row["start_time"],
                    "end_time": evidence_row["end_time"],
                    "text": evidence_row["text"],
                    "role": "answer_support",
                })
        packet["video_evidence_join"]["suppressed_context_rows"] = suppressed_context_rows

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

    def global_candidates_multi(
        self,
        texts: Iterable[str],
        modality: str,
        topk: int = 100,
        *,
        lexical: bool = True,
        rrf_k: int = 60,
        lexical_weight: float = 1.25,
    ) -> list[dict]:
        """Fuse dense and local-lexical results from deterministic query views.

        Dense retrieval remains the semantic channel. Local BM25 is an
        independent, offline view that preserves rare names, quotations and
        numeric tokens that one combined sentence embedding can dilute.
        Returned rows retain per-view provenance for forensic traces.
        """
        if isinstance(texts, (str, bytes)):
            texts = (str(texts),)
        views: list[str] = []
        seen: set[str] = set()
        for text in texts:
            value = str(text or "").strip()
            key = normalise_text(value)
            if value and key and key not in seen:
                views.append(value)
                seen.add(key)
        if not views or topk < 1:
            return []
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        try:
            lexical_weight = float(lexical_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("lexical_weight must be numeric") from exc
        if not np.isfinite(lexical_weight) or lexical_weight <= 0:
            raise ValueError("lexical_weight must be finite and > 0")
        emb, meta = self._load(modality)
        if len(meta) == 0:
            return []

        score_sets: list[tuple[str, str, np.ndarray]] = []
        try:
            if self.text_mode != "bm25":
                query_embeddings = self.embedder.embed(
                    views, batch_size=len(views), normalize=True)
                for view, query_embedding in zip(views, query_embeddings):
                    score_sets.append((view, "embedding", emb @ query_embedding))
            if lexical or self.text_mode == "bm25":
                lexical_index = self._get_lexical_index(modality, meta)
                for view in views:
                    bm25_scores = lexical_index.scores(view)
                    coverage_scores = lexical_index.coverage_scores(view)
                    # Coverage is intentionally the dominant lexical ranking
                    # signal: an exact/near quotation inside a long utterance
                    # should beat a short unrelated row sharing one common
                    # token. The tiny normalized BM25 term still resolves
                    # ties among rows with similar coverage.
                    bm25_peak = float(np.max(bm25_scores)) if len(bm25_scores) else 0.0
                    normalized_bm25 = (
                        bm25_scores / bm25_peak if bm25_peak > 0.0 else bm25_scores
                    )
                    combined = coverage_scores + 0.05 * normalized_bm25
                    score_sets.append((view, "bm25_coverage", combined))
        except Exception as exc:
            raise RuntimeError(f"global modality multi-retrieval unavailable: {exc}") from exc

        merged: dict[int, dict] = {}
        per_view_topk = min(max(int(topk), 1), len(meta))
        for view, mode, scores in score_sets:
            order = np.argpartition(-scores, per_view_topk - 1)[:per_view_topk]
            order = order[np.argsort(-scores[order], kind="stable")]
            for rank, index in enumerate(order, 1):
                index = int(index)
                # A zero BM25 score is not lexical evidence. Letting the
                # arbitrary tie order of non-matches contribute RRF mass can
                # suppress the one transcript/OCR row that actually matches
                # a rare entity or quotation.
                if mode == "bm25_coverage" and float(scores[index]) <= 0.0:
                    continue
                aggregate = merged.setdefault(index, {"rrf_score": 0.0, "view_provenance": []})
                # Exact local text is especially valuable for uncommon names,
                # numeric recipe fields and quotations. Give it a bounded
                # boost instead of letting a generic dense-only hit tie it.
                mode_weight = lexical_weight if mode == "bm25_coverage" else 1.0
                aggregate["rrf_score"] += mode_weight / (float(rrf_k) + rank)
                aggregate["view_provenance"].append({
                    "query_view": view,
                    "score_mode": mode,
                    "rank": rank,
                    "score": float(scores[index]),
                    "weight": mode_weight,
                })

        ordered = sorted(
            merged.items(), key=lambda item: (-float(item[1]["rrf_score"]), item[0])
        )[:min(int(topk), len(merged))]
        out: list[dict] = []
        text_column = "chunk" if str(modality).lower() == "asr" else "ocr_text"
        for rank, (index, aggregate) in enumerate(ordered, 1):
            row = meta.iloc[index]
            evidence_text = str(row.get(text_column, "") or "").strip()
            item = {
                "video_id": str(row.video_id),
                "kf_n": int(row.kf_n),
                "pts_time": float(row.pts_time),
                "modality": str(modality),
                "modality_score": float(aggregate["rrf_score"]),
                "rank": rank,
                "score_mode": "multi_view_rrf",
                "view_provenance": list(aggregate["view_provenance"]),
            }
            if evidence_text:
                item["text"] = evidence_text
                item["evidence"] = {"modality": str(modality), "text": evidence_text}
            if "frame_idx" in row.index and not pd.isna(row["frame_idx"]):
                item["frame_idx"] = int(row["frame_idx"])
            out.append(item)
        return out

    def global_claim_candidates(
        self,
        claim_policy: ClaimPolicy,
        modalities: Iterable[str],
        *,
        topk_per_claim: int = 100,
        max_seed_videos: int = 12,
    ) -> list[dict]:
        """Return videos that cover every high-precision query claim.

        This is a bounded global *rescue channel*, not an answer path.  A
        video is emitted only when each independently extracted query claim is
        supported by canonical ASR/OCR evidence in that same video.  This
        prevents generic single-term matches (for example, just ``Khánh Hòa``)
        from replacing a visual shortlist while allowing a rare entity plus a
        local condition to recover a missed video.
        """
        if not isinstance(claim_policy, ClaimPolicy) or not claim_policy.active:
            return []
        active = tuple(dict.fromkeys(
            str(modality).strip().lower() for modality in modalities
            if str(modality).strip().lower() in {"asr", "ocr"}
        ))
        if not active or topk_per_claim < 1:
            return []
        if max_seed_videos < 1:
            raise ValueError("max_seed_videos must be >= 1")

        # A rare claim can identify a candidate video even when a common
        # companion claim (for example a province) falls far below a global
        # top-100 cutoff.  First collect bounded seed videos, then verify all
        # remaining claims by scanning only those same-video rows.
        seed_ranks: dict[str, int] = {}
        # A claim channel needs one high-precision seed, then verifies every
        # remaining claim inside that video's bounded metadata. Searching all
        # claims globally is both slower and counterproductive for common
        # location words. Prefer an explicit quote/acronym/numeric phrase,
        # then the longest named entity.
        seed_priority = {"quote": 4, "acronym": 3, "numeric_phrase": 2, "entity": 1}
        seed_claim = max(
            claim_policy.claims,
            key=lambda claim: (seed_priority[claim.kind], len(claim.text), claim.text.casefold()),
        )
        for claim in (seed_claim,):
            for modality in active:
                for hit in self.global_candidates_multi(
                    (claim.text,), modality, topk=int(topk_per_claim)
                ):
                    text = str(hit.get("text", "") or "").strip()
                    if score_claim(claim, text) < claim.min_score:
                        continue
                    video_id = str(hit.get("video_id", "")).strip()
                    if not video_id:
                        continue
                    seed_ranks[video_id] = min(
                        seed_ranks.get(video_id, 10**9), int(hit.get("rank", 10**9))
                    )

        if not seed_ranks:
            return []
        # A global rescue may only promote a discriminative anchor.  Channel
        # labels such as ``TV`` or ``HTV7`` occur in many unrelated videos;
        # treating them as a video-level fact turns the claim lane into a
        # harmful popularity prior.  Keep it fail-closed rather than giving
        # those broad matches any RRF weight.  Multiple claims are still
        # checked below, but this cap also protects a one-claim policy.
        if len(seed_ranks) > int(max_seed_videos):
            return []
        seed_ids = set(seed_ranks)
        rows_by_modality_video: dict[str, dict[str, pd.DataFrame]] = {}
        for modality in active:
            metadata = self._load(modality)[1]
            if "video_id" not in metadata.columns:
                rows_by_modality_video[modality] = {}
                continue
            scoped = metadata[metadata["video_id"].astype(str).isin(seed_ids)]
            rows_by_modality_video[modality] = {
                str(video_id): group
                for video_id, group in scoped.groupby(scoped["video_id"].astype(str), sort=False)
            }
        coverage: dict[str, dict[tuple[str, str], dict]] = {}
        for video_id in sorted(seed_ranks):
            for claim in claim_policy.claims:
                claim_key = (claim.kind, claim.text.casefold())
                best: dict | None = None
                best_key = (0.0, 10**9, "")
                for modality, by_video in rows_by_modality_video.items():
                    text_column = "chunk" if modality == "asr" else "ocr_text"
                    rows = by_video.get(video_id)
                    if rows is None:
                        continue
                    if text_column not in rows.columns:
                        continue
                    for _, row in rows.iterrows():
                        text = str(row.get(text_column, "") or "").strip()
                        score = score_claim(claim, text)
                        if score < claim.min_score:
                            continue
                        try:
                            candidate = {
                                "video_id": video_id,
                                "kf_n": int(row["kf_n"]),
                                "frame_idx": int(row["frame_idx"]),
                                "pts_time": float(row["pts_time"]),
                                "rank": seed_ranks[video_id],
                                "modality_score": float(score),
                                "text": text,
                                "claim": claim.to_dict(),
                                "claim_modality": modality,
                            }
                        except (KeyError, TypeError, ValueError):
                            continue
                        candidate_key = (-float(score), int(candidate["kf_n"]), modality)
                        if best is None or candidate_key < best_key:
                            best = candidate
                            best_key = candidate_key
                if best is not None:
                    coverage.setdefault(video_id, {})[claim_key] = best

        verified: list[tuple[tuple[float, float, str], dict]] = []
        for video_id, by_claim in coverage.items():
            expected = {(claim.kind, claim.text.casefold()) for claim in claim_policy.claims}
            if set(by_claim) != expected:
                continue
            hits = [by_claim[(claim.kind, claim.text.casefold())] for claim in claim_policy.claims]
            representative = min(hits, key=lambda item: (
                int(item.get("rank", 10**9)), str(item.get("claim_modality", "")),
                int(item.get("kf_n", 10**9)),
            ))
            score_key = (
                float(max(int(item.get("rank", 10**9)) for item in hits)),
                float(sum(int(item.get("rank", 10**9)) for item in hits)),
                video_id,
            )
            verified.append((score_key, {
                **representative,
                "video_id": video_id,
                "modality": "claim",
                "source": "claim",
                "claim_coverage": [
                    {
                        "claim": item["claim"],
                        "modality": item["claim_modality"],
                        "kf_n": int(item["kf_n"]),
                        "frame_idx": int(item["frame_idx"]),
                        "pts_time": float(item["pts_time"]),
                        "text": str(item.get("text", "")),
                        "retrieval_rank": int(item.get("rank", 10**9)),
                    }
                    for item in hits
                ],
                "evidence": {
                    "kind": "query_claim_coverage",
                    "claims": [item["claim"] for item in hits],
                },
                "view_provenance": [{
                    "score_mode": "claim_coverage",
                    "covered_claim_count": len(hits),
                    "required_claim_count": len(claim_policy.claims),
                }],
            }))
        verified.sort(key=lambda item: item[0])
        output = []
        for rank, (_, item) in enumerate(verified, 1):
            item["rank"] = rank
            output.append(item)
        return output

    @staticmethod
    def _source_row_dict(row: pd.Series) -> dict:
        """Make a JSON-safe, lossless-enough copy of an index metadata row."""
        source: dict[str, object] = {}
        for key, value in row.items():
            missing = value is None
            if not missing and not isinstance(value, (list, tuple, dict, np.ndarray)):
                missing_value = pd.isna(value)
                missing = bool(missing_value) if isinstance(missing_value, (bool, np.bool_)) else False
            if missing:
                source[str(key)] = None
            elif isinstance(value, np.generic):
                source[str(key)] = value.item()
            elif hasattr(value, "isoformat"):
                source[str(key)] = value.isoformat()
            else:
                source[str(key)] = value
        return source

    @staticmethod
    def _localization_views(values: Iterable[str]) -> list[str]:
        """Keep only distinct non-empty query views in caller-supplied order."""
        if isinstance(values, (str, bytes)):
            values = (str(values),)
        views: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            normalized = normalise_text(text)
            if text and normalized and normalized not in seen:
                views.append(text)
                seen.add(normalized)
        return views

    @staticmethod
    def _quote_anchors(values: Iterable[str] | None) -> list[str]:
        """Accept explicit anchors only; arbitrary query text is not a quote."""
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            values = (str(values),)
        anchors: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            normalized = normalise_text(text)
            # A one-word 'anchor' is normally just an entity.  Treat it as a
            # normal query view, rather than falsely claiming quote evidence.
            if len(normalized.split()) < 2 or normalized in seen:
                continue
            anchors.append(text)
            seen.add(normalized)
        return anchors

    @staticmethod
    def _canonical_localization_row(row: pd.Series) -> dict | None:
        """Return canonical coordinates, or None without fabricating a frame."""
        required = ("video_id", "kf_n", "frame_idx", "pts_time")
        if any(column not in row.index or pd.isna(row[column]) for column in required):
            return None
        video_id = str(row["video_id"] or "").strip()
        if not video_id:
            return None
        try:
            kf_n = int(row["kf_n"])
            frame_idx = int(row["frame_idx"])
            pts_time = float(row["pts_time"])
        except (TypeError, ValueError):
            return None
        if kf_n < 0 or frame_idx < 0 or not np.isfinite(pts_time):
            return None
        return {
            "video_id": video_id,
            "kf_n": kf_n,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
        }

    def localize_evidence(
        self,
        modality: str,
        video_ids: Iterable[str],
        query_views: Iterable[str],
        *,
        per_video: int = 3,
        quote_anchors: Iterable[str] | None = None,
        lexical_weight: float = 1.25,
        quote_weight: float = 0.25,
        quote_min_score: float = 0.80,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Localize local ASR/OCR evidence after a video shortlist.

        The API deliberately operates *after* video selection.  It never
        changes video RRF and never maps a keyframe itself: a returned row is
        emitted only when its source metadata already contains canonical
        ``video_id``, ``kf_n``, ``frame_idx`` and ``pts_time``.  For every
        output row, ``source_row`` preserves the original metadata and
        ``view_provenance`` exposes dense, BM25/coverage and quote-anchor
        signals for each requested query view.

        ``quote_anchors`` are explicit hypotheses (for example, a quotation
        recovered from a cited source).  They are matched locally with
        Vietnamese-normalised exact/near matching.  When an explicit anchor
        is supplied, a row must clear ``quote_min_score`` to be emitted.  An
        external hypothesis can therefore *find* corroborating local text,
        but cannot turn a merely related entity/context chunk into purported
        quote evidence.  Without an anchor the method remains a normal
        lexical/dense shortlist localizer.
        """
        modality = str(modality or "").strip().lower()
        if modality not in {"asr", "ocr"}:
            raise ValueError("modality must be 'asr' or 'ocr'")
        if per_video < 1:
            return []
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        try:
            lexical_weight = float(lexical_weight)
            quote_weight = float(quote_weight)
            quote_min_score = float(quote_min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("localization weights must be numeric") from exc
        if (not np.isfinite(lexical_weight) or lexical_weight <= 0
                or not np.isfinite(quote_weight) or quote_weight < 0
                or not np.isfinite(quote_min_score) or not 0.0 < quote_min_score <= 1.0):
            raise ValueError(
                "localization weights must be finite; lexical > 0, quote >= 0, "
                "and quote_min_score must be in (0, 1]"
            )

        ordered_video_ids = list(dict.fromkeys(
            str(video_id).strip() for video_id in video_ids if str(video_id).strip()
        ))
        views = self._localization_views(query_views)
        anchors = self._quote_anchors(quote_anchors)
        if not ordered_video_ids or not views:
            return []

        embeddings, metadata = self._load(modality)
        if len(metadata) == 0:
            return []
        if "video_id" not in metadata.columns:
            raise RuntimeError(f"local {modality} metadata has no 'video_id'")
        text_column = "chunk" if modality == "asr" else "ocr_text"
        if text_column not in metadata.columns:
            raise RuntimeError(f"local {modality} metadata has no {text_column!r}")

        allowed = set(ordered_video_ids)
        scoped = metadata.loc[
            metadata["video_id"].astype(str).isin(allowed)
        ].copy()
        if scoped.empty:
            return []
        # Do not expose non-canonical rows as possible submission evidence.
        scoped["_source_index"] = scoped.index.astype(np.int64)
        canonical = scoped.apply(self._canonical_localization_row, axis=1)
        scoped = scoped.loc[canonical.notna()].copy()
        scoped["_canonical"] = canonical.loc[canonical.notna()]
        if scoped.empty:
            return []

        lexical_index = self._get_lexical_index(modality, metadata)
        source_indices = scoped["_source_index"].to_numpy(dtype=np.int64)
        documents = scoped[text_column].fillna("").astype(str).tolist()
        per_view: dict[str, dict[str, np.ndarray]] = {}
        for view in views:
            bm25 = lexical_index.scores(view)[source_indices]
            coverage = lexical_index.coverage_scores(view)[source_indices]
            dense = None
            if self.text_mode != "bm25":
                try:
                    query = self.embedder.embed([view], batch_size=1, normalize=True)[0]
                    dense = embeddings[source_indices] @ query
                except Exception as exc:
                    raise RuntimeError(
                        f"local {modality} dense localization unavailable for a query view: {exc}"
                    ) from exc
            per_view[view] = {"bm25": bm25, "coverage": coverage, "dense": dense}

        # Explicit quote hypotheses may be recovered by a planner from an
        # external citation, but all matching is still against local indexed
        # evidence.  No web result is returned or used as a frame source.
        anchor_scores = np.zeros(len(scoped), dtype=np.float32)
        anchor_provenance: list[list[dict]] = [[] for _ in range(len(scoped))]
        for row_pos, document in enumerate(documents):
            for anchor in anchors:
                match = quote_match(anchor, document)
                if float(match["score"]) <= 0.0:
                    continue
                detail = {"anchor": anchor, **match}
                anchor_provenance[row_pos].append(detail)
                anchor_scores[row_pos] = max(anchor_scores[row_pos], float(match["score"]))

        # Quote localization is a verification operation, not a broad search
        # operation.  In particular, a web page about the same person can
        # contain a different well-known verse; returning the strongest
        # *context* row in that case would make the downstream selector look
        # grounded when no local transcript actually corroborated the quote.
        # Filter before scoring so a failed hypothesis produces no candidates
        # and the caller falls back to the normal global route.
        if anchors:
            verified_mask = anchor_scores >= quote_min_score
            if not bool(np.any(verified_mask)):
                return []
            scoped = scoped.loc[verified_mask].copy()
            documents = [document for document, keep in zip(documents, verified_mask) if keep]
            anchor_scores = anchor_scores[verified_mask]
            anchor_provenance = [
                details for details, keep in zip(anchor_provenance, verified_mask) if keep
            ]
            # Query-view arrays and source positions have the original scoped
            # alignment. Keep that relation explicit rather than relying on
            # DataFrame positional indices after the filter below.
            retained_positions = np.flatnonzero(verified_mask)
        else:
            retained_positions = np.arange(len(scoped), dtype=np.int64)

        # Score/rank only within each shortlisted video.  RRF keeps the dense
        # and lexical scales comparable; quote evidence is a bounded local
        # tie-breaker rather than a global-video rescue mechanism.
        scoped["_score"] = 0.0
        provenance_by_position: list[list[dict]] = [[] for _ in range(len(scoped))]
        scoped_positions = {index: position for position, index in enumerate(scoped.index)}
        for video_id, group in scoped.groupby(scoped["video_id"].astype(str), sort=False):
            group_positions = np.asarray(
                [scoped_positions[index] for index in group.index], dtype=np.int64
            )
            for view in views:
                signals = per_view[view]
                dense = signals["dense"][retained_positions] if signals["dense"] is not None else None
                if dense is not None:
                    order = group_positions[np.argsort(-dense[group_positions], kind="stable")]
                    for rank, position in enumerate(order, 1):
                        contribution = 1.0 / (float(rrf_k) + rank)
                        scoped.loc[scoped.index[position], "_score"] += contribution
                        provenance_by_position[position].append({
                            "query_view": view,
                            "score_mode": "embedding",
                            "score": float(dense[position]),
                            "rank_within_video_view": rank,
                            "weight": 1.0,
                            "rrf_contribution": contribution,
                        })
                lexical = signals["coverage"][retained_positions]
                order = group_positions[np.argsort(-lexical[group_positions], kind="stable")]
                for rank, position in enumerate(order, 1):
                    # Zero coverage is deliberately auditable but cannot add
                    # arbitrary tie-order mass to unrelated evidence.
                    contribution = (lexical_weight / (float(rrf_k) + rank)
                                    if float(lexical[position]) > 0.0 else 0.0)
                    scoped.loc[scoped.index[position], "_score"] += contribution
                    provenance_by_position[position].append({
                        "query_view": view,
                        "score_mode": "bm25_coverage",
                        "score": float(lexical[position]),
                        "bm25_score": float(signals["bm25"][retained_positions[position]]),
                        "rank_within_video_view": rank,
                        "weight": lexical_weight,
                        "rrf_contribution": contribution,
                    })
            for position in group_positions:
                if anchor_scores[position] > 0.0:
                    scoped.loc[scoped.index[position], "_score"] += (
                        quote_weight * float(anchor_scores[position])
                    )

        output: list[dict] = []
        for video_rank, video_id in enumerate(ordered_video_ids, 1):
            group = scoped.loc[scoped["video_id"].astype(str) == video_id].copy()
            if group.empty:
                continue
            group = group.sort_values(
                ["_score", "pts_time", "kf_n"], ascending=[False, True, True], kind="stable"
            ).head(int(per_video))
            for rank_within_video, (index, row) in enumerate(group.iterrows(), 1):
                position = scoped_positions[index]
                canonical_row = dict(row["_canonical"])
                source_row = self._source_row_dict(row.drop(labels=[
                    "_source_index", "_canonical", "_score"
                ], errors="ignore"))
                evidence_text = str(row.get(text_column, "") or "").strip()
                item = {
                    **canonical_row,
                    "modality": modality,
                    "modality_score": float(row["_score"]),
                    "score_mode": "shortlist_localization_rrf",
                    "video_shortlist_rank": video_rank,
                    "rank_within_video": rank_within_video,
                    "text": evidence_text,
                    "evidence": {"modality": modality, "text": evidence_text},
                    "view_provenance": provenance_by_position[position],
                    "quote_anchor_provenance": anchor_provenance[position],
                    "source_row": source_row,
                }
                output.append(item)
        return output

    def localize_poetry_event(
        self,
        video_ids: Iterable[str],
        query: str,
        question: str,
        *,
        per_video: int = 4,
    ) -> list[dict]:
        """Find a locally evidenced verse-recital event after video fusion.

        This is intentionally narrower than :meth:`localize_evidence`: the
        latter verifies an explicit quotation hypothesis, whereas this method
        finds a compact local ASR sequence only when the question itself asks
        for poetry.  It uses no web text, does not alter video ranking, and
        returns only rows already canonical in the global ASR registry.
        """
        ordered_video_ids = list(dict.fromkeys(
            str(video_id).strip() for video_id in video_ids if str(video_id).strip()
        ))
        if not ordered_video_ids:
            self.last_poetry_event_diagnostic = {"status": "empty_shortlist"}
            return []
        try:
            return self._poetry_event_rows(
                query, question, shortlisted_video_ids=ordered_video_ids,
                per_video=int(per_video),
            )
        except Exception as exc:
            # The caller records this diagnostic and keeps the normal global
            # route.  A local event hint must never weaken an otherwise valid
            # retrieval run.
            self.last_poetry_event_diagnostic = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}"[:240],
            }
            return []

    def global_poetry_event_candidates(
        self,
        query: str,
        question: str,
        *,
        topk: int = 20,
    ) -> list[dict]:
        """Retrieve evidenced verse events globally, then collapse by video.

        This is a distinct precision lane for an explicit poetry question. It
        joins ASR recitation with same-video ASR/OCR entity context, so neither
        modality has to contain the entire fact. It never emits an answer and
        each result keeps an existing canonical ASR coordinate.
        """
        from src.vqa.asr_event_localizer import is_explicit_poetry_question

        if topk < 1 or not is_explicit_poetry_question(query, question):
            return []
        rows = self._poetry_event_rows(query, question, per_video=4)
        best_by_video: dict[str, dict] = {}
        for row in rows:
            video_id = str(row.get("video_id", "")).strip()
            if not video_id:
                continue
            key = (
                -float(row.get("modality_score", 0.0)),
                int(row.get("kf_n", 10**9)),
                float(row.get("pts_time", float("inf"))),
            )
            existing = best_by_video.get(video_id)
            existing_key = (
                -float(existing.get("modality_score", 0.0)),
                int(existing.get("kf_n", 10**9)),
                float(existing.get("pts_time", float("inf"))),
            ) if existing is not None else None
            if existing is None or key < existing_key:
                best_by_video[video_id] = row

        ordered = sorted(
            best_by_video.values(),
            key=lambda row: (
                -float(row.get("modality_score", 0.0)),
                str(row.get("video_id", "")),
                int(row.get("kf_n", 10**9)),
            ),
        )[:int(topk)]
        output: list[dict] = []
        for rank, row in enumerate(ordered, 1):
            score = float(row.get("modality_score", 0.0))
            output.append({
                **row,
                "rank": rank,
                "score": score,
                "source": "asr_poetry_event",
                "score_mode": "global_asr_poetry_event",
                "evidence": {"modality": "asr", "text": str(row.get("text", ""))},
                "view_provenance": [{
                    "score_mode": "local_asr_poetry_event",
                    "score": score,
                    "rank": rank,
                    "weight": 1.0,
                }],
            })
        return output

    def _poetry_event_rows(
        self,
        query: str,
        question: str,
        *,
        shortlisted_video_ids: Iterable[str] | None = None,
        per_video: int = 4,
    ) -> list[dict]:
        """Run the local ASR event detector with OCR available as context."""
        from src.vqa.asr_event_localizer import ASRPoetryEventLocalizer

        _, asr = self._load("asr")
        asr_rows = asr.copy()
        asr_rows["evidence_modality"] = "asr"
        context_frames = [asr_rows]
        try:
            _, ocr = self._load("ocr")
            required = {"video_id", "kf_n", "frame_idx", "pts_time", "ocr_text"}
            if required.issubset(ocr.columns):
                ocr_rows = ocr.loc[:, ["video_id", "kf_n", "frame_idx", "pts_time", "ocr_text"]].copy()
                ocr_rows = ocr_rows.rename(columns={"ocr_text": "chunk"})
                ocr_rows["evidence_modality"] = "ocr"
                context_frames.append(ocr_rows)
        except Exception:
            # OCR is supplemental context. A complete ASR index remains a
            # valid offline event detector when OCR is unavailable.
            pass
        combined = pd.concat(context_frames, ignore_index=True, sort=False)
        localizer = ASRPoetryEventLocalizer()
        rows = localizer.localize(
            combined,
            query,
            question,
            shortlisted_video_ids=(
                list(shortlisted_video_ids) if shortlisted_video_ids is not None else None
            ),
            max_candidates_per_video=int(per_video),
        )
        self.last_poetry_event_diagnostic = dict(localizer.last_diagnostic)
        return rows

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
