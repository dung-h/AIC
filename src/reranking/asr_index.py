"""Offline ASR index for spoken-fact retrieval.

This module is deliberately independent from the Q&A router and answerer.  It
owns the ASR-side invariants only:

* validate transcript/timestamp metadata without mutating persisted files;
* repair only unambiguous UTF-8 mojibake at read time;
* retrieve by local dense embeddings or BM25;
* map an ASR timestamp to the canonical ``kf_n``/``frame_idx`` pair.

No code path in this module downloads models, calls an API, or invents an ASR
transcript.  ``strict=True`` is fail-closed when the requested packs or local
dense model are unavailable.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.reranking.local_lexical_index import BM25Index
from src.reranking.modality_index_preflight import (
    DEFAULT_EXPECTED_PACKS,
    normalise_expected_packs,
    run_modality_index_preflight,
)


ASR_METADATA_RE = re.compile(r"^asr_chunks_(k\d{2}|l\d{2})_ts\.parquet$", re.IGNORECASE)
ASR_EMBED_RE = re.compile(r"^emb_cache_asr_(k\d{2}|l\d{2})_chunks\.npy$", re.IGNORECASE)
CANONICAL_COLUMNS = {"video_id", "kf_n", "frame_idx", "pts_time"}

# These are sequences produced by decoding UTF-8 bytes as cp1252/latin-1.
# Do not flag ordinary Vietnamese ``â``/``ă``/``đ`` characters by themselves.
_MOJIBAKE_RE = re.compile(
    # ``Âu`` and ``Âm`` are valid Vietnamese words, so a bare ``Â.`` rule
    # would reject clean ASR rows.  Keep only common cp1252 artefacts after
    # Â (degree sign, NBSP/typographic punctuation), while retaining the
    # unambiguous UTF-8 byte-leak prefixes.
    r"(?:Ã.|Â(?:°|\xa0|©|®|“|”|™|\s|$)|Ä.|Å.|Æ.|Ç.|á».|áº.|ï¿½|\ufffd)",
    re.UNICODE,
)


class ASRIndexPreflightError(RuntimeError):
    """Raised when a strict ASR index cannot be trusted."""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(math.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def mojibake_score(value: Any) -> int:
    """Return a conservative count of suspicious UTF-8 decoding artefacts."""
    return len(_MOJIBAKE_RE.findall(_as_text(value)))


def normalize_transcript(value: Any) -> str:
    """Normalize transcript text without silently changing valid Vietnamese.

    The repair is accepted only when a cp1252/latin-1 -> UTF-8 round trip
    reduces the mojibake score and does not introduce replacement characters.
    At most two passes handle double-encoded text.  Unrepairable text is kept
    intact so preflight can reject it rather than fabricate evidence.
    """
    # Repair before NFKC: NFKC expands characters such as ``™`` into ``TM``;
    # that would destroy the cp1252 byte sequence in strings like ``á»™``.
    text = _as_text(value).strip()
    for _ in range(2):
        before_score = mojibake_score(text)
        if before_score == 0:
            break
        candidates: list[str] = []
        for encoding in ("cp1252", "latin-1"):
            try:
                candidates.append(text.encode(encoding).decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if candidates:
            candidate = min(candidates, key=mojibake_score)
        else:
            # Real ASR packs can contain valid Vietnamese and mojibake in the
            # same sentence.  Repair suspicious whitespace-delimited tokens
            # independently so a valid character elsewhere does not block a
            # safe local repair.
            pieces = re.split(r"(\s+)", text)
            repaired_pieces: list[str] = []
            changed = False
            for piece in pieces:
                piece_candidates: list[str] = []
                for encoding in ("cp1252", "latin-1"):
                    try:
                        piece_candidates.append(piece.encode(encoding).decode("utf-8"))
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        continue
                if piece_candidates:
                    best_piece = min(piece_candidates, key=mojibake_score)
                    if mojibake_score(best_piece) < mojibake_score(piece):
                        piece = best_piece
                        changed = True
                repaired_pieces.append(piece)
            if not changed:
                break
            candidate = "".join(repaired_pieces)
        if (
            mojibake_score(candidate) >= before_score
            or candidate.count("\ufffd") > text.count("\ufffd")
        ):
            break
        text = unicodedata.normalize("NFKC", candidate).strip()
    return unicodedata.normalize("NFKC", text).strip()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_rows(canonical: pd.DataFrame) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    missing = sorted(CANONICAL_COLUMNS - set(canonical.columns))
    if missing:
        return {}, [{"code": "canonical_missing_columns", "columns": missing}]

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[tuple[str, int], tuple[int, float]] = {}
    for row_number, row in canonical.reset_index(drop=True).iterrows():
        video = _as_text(row["video_id"]).strip()
        kf = _number(row["kf_n"])
        frame = _number(row["frame_idx"])
        pts = _number(row["pts_time"])
        if not video or kf is None or frame is None or pts is None or kf < 0 or frame < 0 or pts < 0:
            errors.append({"code": "canonical_invalid_row", "row": int(row_number)})
            continue
        if kf != int(kf) or frame != int(frame):
            errors.append({"code": "canonical_non_integer_identity", "row": int(row_number)})
            continue
        identity = (video, int(kf))
        value = (int(frame), float(pts))
        if identity in seen and seen[identity] != value:
            errors.append({"code": "canonical_duplicate_identity", "row": int(row_number), "key": identity})
            continue
        seen[identity] = value
        by_video[video].append({"video_id": video, "kf_n": int(kf), "frame_idx": int(frame), "pts_time": float(pts)})
    for frames in by_video.values():
        frames.sort(key=lambda item: (item["pts_time"], item["kf_n"]))
    return dict(by_video), errors


def map_timestamp_to_canonical(
    video_id: str,
    timestamp: float,
    canonical: pd.DataFrame | Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Map one ASR timestamp to the nearest canonical keyframe.

    The returned ``distance_seconds`` is retained as evidence quality; callers
    can apply their own tolerance without losing the canonical mapping.
    """
    value = _number(timestamp)
    if value is None or value < 0:
        raise ASRIndexPreflightError("ASR timestamp must be finite and non-negative")
    if isinstance(canonical, pd.DataFrame):
        by_video, errors = _canonical_rows(canonical)
        if errors:
            raise ASRIndexPreflightError(f"invalid canonical map: {errors[0]}")
    else:
        # Callers performing a pack-wide preflight already hold the grouped
        # canonical map.  Do not copy millions of frame rows for every ASR
        # chunk; a read-only mapping is sufficient here.
        by_video = canonical  # type: ignore[assignment]
    frames = by_video.get(str(video_id))
    if not frames:
        raise ASRIndexPreflightError(f"video is absent from canonical map: {video_id}")
    nearest = min(frames, key=lambda item: (abs(float(item["pts_time"]) - value), item["kf_n"]))
    return {
        "video_id": str(video_id),
        "timestamp": float(value),
        "kf_n": int(nearest["kf_n"]),
        "frame_idx": int(nearest["frame_idx"]),
        "pts_time": float(nearest["pts_time"]),
        "distance_seconds": abs(float(nearest["pts_time"]) - value),
    }


def preflight_asr_metadata(
    metadata: pd.DataFrame,
    canonical: pd.DataFrame | Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    timestamp_warning_seconds: float = 10.0,
) -> dict[str, Any]:
    """Validate one ASR parquet frame against the canonical map."""
    report: dict[str, Any] = {
        "rows": int(len(metadata)),
        "videos": 0,
        "errors": [],
        "warnings": [],
        "text_quality": {
            "rows": int(len(metadata)),
            "empty_rows": 0,
            "replacement_chars": 0,
            "mojibake_rows": 0,
            "repairable_mojibake_rows": 0,
            "unrepairable_mojibake_rows": 0,
            "nfkc_changes": 0,
        },
        "timestamp_mapping": {
            "mapped_rows": 0,
            "recomputed_mapping_rows": 0,
            "missing_video_rows": 0,
            "max_distance_seconds": None,
            "median_distance_seconds": None,
        },
        "passed": False,
    }
    columns = set(metadata.columns)
    video_column = "video_id" if "video_id" in columns else "vid" if "vid" in columns else None
    text_column = next((name for name in ("chunk", "text", "transcript") if name in columns), None)
    missing = sorted({"kf_n", "frame_idx", "start", "end"} - columns)
    if video_column is None:
        missing.append("video_id/vid")
    if text_column is None:
        missing.append("chunk/text/transcript")
    if missing:
        report["errors"].append({"code": "missing_asr_columns", "columns": missing})
        return report

    if isinstance(canonical, pd.DataFrame):
        by_video, canonical_errors = _canonical_rows(canonical)
    else:
        by_video, canonical_errors = canonical, []
    report["errors"].extend(canonical_errors)
    report["videos"] = len({
        _as_text(value).strip() for value in metadata[video_column].tolist() if _as_text(value).strip()
    })
    report["video_ids"] = sorted({
        _as_text(value).strip()
        for value in metadata[video_column].tolist()
        if _as_text(value).strip()
    })
    distances: list[float] = []
    for row_number, row in metadata.reset_index(drop=True).iterrows():
        text = _as_text(row[text_column])
        normalized = normalize_transcript(text)
        quality = report["text_quality"]
        if not text.strip():
            quality["empty_rows"] += 1
        quality["replacement_chars"] += text.count("\ufffd")
        if unicodedata.normalize("NFKC", text) != text:
            quality["nfkc_changes"] += 1
        before = mojibake_score(text)
        after = mojibake_score(normalized)
        if before:
            quality["mojibake_rows"] += 1
            if after < before:
                quality["repairable_mojibake_rows"] += 1
            else:
                quality["unrepairable_mojibake_rows"] += 1
        if not normalized:
            report["errors"].append({"code": "empty_transcript", "row": int(row_number)})

        video = _as_text(row[video_column]).strip()
        start = _number(row["start"])
        end = _number(row["end"])
        kf = _number(row["kf_n"])
        frame = _number(row["frame_idx"])
        if not video or start is None or end is None:
            report["errors"].append({"code": "invalid_asr_identity_or_timestamp", "row": int(row_number)})
            continue
        if start < 0 or end < 0 or end < start:
            report["errors"].append({"code": "invalid_asr_interval", "row": int(row_number)})
            continue
        # Older ASR packs have the columns but leave the mapping null.  That
        # is recoverable from the timestamp and canonical map; non-null,
        # malformed values remain hard errors.
        raw_kf_missing = _as_text(row["kf_n"]).strip() == ""
        raw_frame_missing = _as_text(row["frame_idx"]).strip() == ""
        if (not raw_kf_missing and kf is None) or (not raw_frame_missing and frame is None):
            report["errors"].append({"code": "invalid_canonical_identity", "row": int(row_number)})
            continue
        try:
            mapped = map_timestamp_to_canonical(video, (start + end) / 2.0, by_video)
        except ASRIndexPreflightError:
            report["timestamp_mapping"]["missing_video_rows"] += 1
            report["errors"].append({"code": "canonical_mapping_missing", "row": int(row_number), "video_id": video})
            continue
        report["timestamp_mapping"]["mapped_rows"] += 1
        distances.append(float(mapped["distance_seconds"]))
        if raw_kf_missing or raw_frame_missing:
            report["timestamp_mapping"]["recomputed_mapping_rows"] += 1
        if (kf is not None and kf != int(kf)) or (frame is not None and frame != int(frame)):
            report["errors"].append({"code": "invalid_canonical_identity", "row": int(row_number)})
        elif (kf is not None and int(kf) != mapped["kf_n"]) or (frame is not None and int(frame) != mapped["frame_idx"]):
            report["errors"].append({
                "code": "canonical_mapping_mismatch",
                "row": int(row_number),
                "provided": {
                    "kf_n": int(kf) if kf is not None else None,
                    "frame_idx": int(frame) if frame is not None else None,
                },
                "nearest": {"kf_n": mapped["kf_n"], "frame_idx": mapped["frame_idx"]},
            })

    quality = report["text_quality"]
    if quality["empty_rows"]:
        report["errors"].append({"code": "empty_transcript", "count": quality["empty_rows"]})
    if quality["replacement_chars"]:
        report["errors"].append({"code": "text_replacement_char", "count": quality["replacement_chars"]})
    if quality["unrepairable_mojibake_rows"]:
        report["errors"].append({"code": "text_mojibake_unrepairable", "count": quality["unrepairable_mojibake_rows"]})
    if quality["repairable_mojibake_rows"]:
        report["warnings"].append({
            "code": "text_mojibake_repaired_at_read_time",
            "count": quality["repairable_mojibake_rows"],
        })
    if distances:
        report["timestamp_mapping"]["max_distance_seconds"] = max(distances)
        report["timestamp_mapping"]["median_distance_seconds"] = float(np.median(distances))
        if max(distances) > float(timestamp_warning_seconds):
            report["warnings"].append({
                "code": "coarse_keyframe_timestamp_mapping",
                "max_distance_seconds": max(distances),
                "threshold_seconds": float(timestamp_warning_seconds),
            })
    report["passed"] = not report["errors"]
    return report


def _materialize_asr_mapping(
    frame: pd.DataFrame,
    by_video: Mapping[str, Sequence[Mapping[str, Any]]],
) -> pd.DataFrame:
    """Fill nullable source mappings from ASR midpoint timestamps in memory."""
    output = frame.copy()
    mapped = [
        map_timestamp_to_canonical(
            str(video),
            (float(start) + float(end)) / 2.0,
            by_video,
        )
        for video, start, end in zip(output["video_id"], output["start"], output["end"])
    ]
    output["kf_n"] = [
        int(_number(value)) if _number(value) is not None else int(item["kf_n"])
        for value, item in zip(output["kf_n"], mapped)
    ]
    output["frame_idx"] = [
        int(_number(value)) if _number(value) is not None else int(item["frame_idx"])
        for value, item in zip(output["frame_idx"], mapped)
    ]
    return output


class ASRIndex:
    """Local ASR retrieval with a canonical timestamp/frame evidence contract."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        canonical_name: str = "global_keyframes.parquet",
        expected_packs: Iterable[str] | None = None,
        mode: str = "bm25",
        model_dir: str | Path | None = None,
        embedder: Any | None = None,
        strict: bool = True,
    ):
        self.index_dir = Path(index_dir)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"bm25", "dense"}:
            raise ValueError("mode must be 'bm25' or 'dense'")
        self.strict = bool(strict)
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.embedder = embedder
        canonical_path = self.index_dir / canonical_name
        if not canonical_path.exists() and canonical_name == "global_keyframes.parquet":
            fallback = self.index_dir / "global_keyframes_vitl.parquet"
            if fallback.exists():
                canonical_path = fallback
        if not canonical_path.exists():
            raise ASRIndexPreflightError(f"canonical map is missing: {canonical_path}")
        self.canonical_path = canonical_path
        self.canonical = pd.read_parquet(canonical_path).reset_index(drop=True)
        self.by_video, canonical_errors = _canonical_rows(self.canonical)
        if canonical_errors:
            raise ASRIndexPreflightError(f"invalid canonical map: {canonical_errors[0]}")

        metadata_paths = {
            match.group(1).lower(): path
            for path in self.index_dir.iterdir()
            if (match := ASR_METADATA_RE.match(path.name))
        }
        if expected_packs is None:
            # Strict filesystem indexes must prove full-corpus coverage.  A
            # caller that intentionally mounts a subset must declare it
            # explicitly via expected_packs (or use strict=False for legacy
            # diagnostic behavior).
            packs = DEFAULT_EXPECTED_PACKS if self.strict else tuple(sorted(metadata_paths))
        else:
            packs = normalise_expected_packs(expected_packs)
        if not packs:
            raise ASRIndexPreflightError("no ASR metadata packs found")
        missing = sorted(set(packs) - set(metadata_paths))
        if missing:
            raise ASRIndexPreflightError(f"ASR metadata packs missing: {missing}")

        frames: list[pd.DataFrame] = []
        embeddings: list[np.ndarray] = []
        self.pack_reports: list[dict[str, Any]] = []
        for pack in packs:
            metadata_path = metadata_paths[pack]
            metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
            pack_report = preflight_asr_metadata(metadata, self.by_video)
            pack_report.update({"pack": pack, "metadata": str(metadata_path)})
            embedding_path = self.index_dir / f"emb_cache_asr_{pack}_chunks.npy"
            if self.mode == "dense":
                if not embedding_path.exists():
                    pack_report["errors"].append({"code": "missing_embedding", "path": str(embedding_path)})
                else:
                    array = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
                    if array.ndim != 2 or len(array) != len(metadata):
                        pack_report["errors"].append({
                            "code": "embedding_metadata_length_mismatch",
                            "embedding_shape": list(array.shape),
                            "metadata_rows": int(len(metadata)),
                        })
                    else:
                        embeddings.append(np.asarray(array, dtype=np.float32))
            self.pack_reports.append(pack_report)
            if not pack_report["passed"] or (self.mode == "dense" and pack_report["errors"]):
                continue
            video_column = "video_id" if "video_id" in metadata.columns else "vid"
            text_column = next(name for name in ("chunk", "text", "transcript") if name in metadata.columns)
            normalized = metadata.copy()
            normalized["video_id"] = normalized[video_column].map(lambda value: _as_text(value).strip())
            normalized["chunk"] = normalized[text_column].map(normalize_transcript)
            normalized["start"] = normalized["start"].astype(float)
            normalized["end"] = normalized["end"].astype(float)
            normalized["pts_time"] = (normalized["start"] + normalized["end"]) / 2.0
            normalized = _materialize_asr_mapping(normalized, self.by_video)
            normalized = normalized[["video_id", "kf_n", "frame_idx", "start", "end", "pts_time", "chunk"]]
            normalized["pack"] = pack
            frames.append(normalized)

        errors = [item for report in self.pack_reports for item in report["errors"]]
        if strict:
            expected_video_ids = {
                video_id for video_id in self.by_video
                if re.match(r"^([kl]\d{2})_", video_id, re.IGNORECASE)
                and re.match(r"^([kl]\d{2})_", video_id, re.IGNORECASE).group(1).lower() in set(packs)
            }
            observed_video_ids = {
                video_id for report in self.pack_reports for video_id in report.get("video_ids", [])
            }
            missing_video_ids = sorted(expected_video_ids - observed_video_ids)
            if missing_video_ids:
                errors.append({
                    "code": "asr_video_coverage_incomplete",
                    "count": len(missing_video_ids),
                    "examples": missing_video_ids[:10],
                })
        if strict and errors:
            first = errors[0]
            raise ASRIndexPreflightError(f"ASR preflight failed: {first}")
        self.metadata = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if self.metadata.empty:
            raise ASRIndexPreflightError("ASR index has no valid metadata rows")
        self._bm25 = BM25Index(self.metadata["chunk"].tolist())
        self._embeddings: np.ndarray | None = None
        if self.mode == "dense":
            if not embeddings:
                raise ASRIndexPreflightError("dense ASR index has no valid local embeddings")
            self._embeddings = np.vstack(embeddings).astype(np.float32)
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True).clip(min=1e-8)
            self._embeddings /= norms
            if self.embedder is None and (self.model_dir is None or not self.model_dir.exists()):
                raise ASRIndexPreflightError("dense ASR retrieval requires an existing local model or injected embedder")

    @classmethod
    def from_frames(
        cls,
        metadata: pd.DataFrame,
        canonical: pd.DataFrame,
        *,
        mode: str = "bm25",
        embeddings: np.ndarray | None = None,
        embedder: Any | None = None,
    ) -> "ASRIndex":
        """Build an in-memory index for deterministic unit/smoke tests."""
        report = preflight_asr_metadata(metadata, canonical)
        if not report["passed"]:
            raise ASRIndexPreflightError(f"ASR preflight failed: {report['errors'][0]}")
        obj = cls.__new__(cls)
        obj.index_dir = None
        obj.mode = mode
        obj.strict = True
        obj.model_dir = None
        obj.embedder = embedder
        obj.canonical_path = None
        obj.canonical = canonical.reset_index(drop=True)
        obj.by_video, _ = _canonical_rows(obj.canonical)
        video_column = "video_id" if "video_id" in metadata.columns else "vid"
        text_column = next(name for name in ("chunk", "text", "transcript") if name in metadata.columns)
        frame = metadata.copy()
        frame["video_id"] = frame[video_column].map(lambda value: _as_text(value).strip())
        frame["chunk"] = frame[text_column].map(normalize_transcript)
        frame["pts_time"] = (frame["start"].astype(float) + frame["end"].astype(float)) / 2.0
        frame = _materialize_asr_mapping(frame, obj.by_video)
        obj.metadata = frame[["video_id", "kf_n", "frame_idx", "start", "end", "pts_time", "chunk"]].reset_index(drop=True)
        obj._bm25 = BM25Index(obj.metadata["chunk"].tolist())
        obj._embeddings = None
        if mode == "dense":
            if embeddings is None or np.asarray(embeddings).shape[0] != len(obj.metadata):
                raise ASRIndexPreflightError("in-memory dense embeddings do not align with ASR rows")
            obj._embeddings = np.asarray(embeddings, dtype=np.float32)
            obj._embeddings /= np.linalg.norm(obj._embeddings, axis=1, keepdims=True).clip(min=1e-8)
        obj.pack_reports = [report]
        return obj

    def _query_embedding(self, query: str) -> np.ndarray:
        if self.embedder is None:
            if self.model_dir is None or not self.model_dir.exists():
                raise ASRIndexPreflightError("local dense ASR model is unavailable")
            try:
                from sentence_transformers import SentenceTransformer
                # local_files_only prevents an accidental network dependency.
                model = SentenceTransformer(str(self.model_dir), device="cpu", local_files_only=True)
            except Exception as exc:
                raise ASRIndexPreflightError(f"local dense ASR model could not load offline: {exc}") from exc
            self.embedder = model
        if hasattr(self.embedder, "embed"):
            vector = self.embedder.embed([query], batch_size=1, normalize=True)[0]
        else:
            vector = self.embedder.encode([query], normalize_embeddings=True)[0]
        vector = np.asarray(vector, dtype=np.float32)
        if self._embeddings is None or vector.shape[-1] != self._embeddings.shape[1]:
            raise ASRIndexPreflightError("query embedding dimension does not match ASR index")
        return vector / np.linalg.norm(vector).clip(min=1e-8)

    def search(self, query: str, *, topk: int = 100, video_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Return ranked, non-empty ASR evidence with canonical frame mapping."""
        query = normalize_transcript(query)
        if not query:
            return []
        if topk < 1:
            return []
        if self.mode == "bm25":
            scores = self._bm25.scores(query)
            positive = scores > 0
        else:
            assert self._embeddings is not None
            scores = self._embeddings @ self._query_embedding(query)
            positive = np.isfinite(scores)
        positive &= self.metadata["chunk"].astype(str).str.len().to_numpy() > 0
        if video_ids is not None:
            allowed = {str(value) for value in video_ids}
            positive &= self.metadata["video_id"].astype(str).isin(allowed).to_numpy()
        candidates = np.flatnonzero(positive)
        if len(candidates) == 0:
            return []
        limit = min(int(topk), len(candidates))
        order = candidates[np.argsort(-scores[candidates], kind="stable")[:limit]]
        output: list[dict[str, Any]] = []
        for rank, index in enumerate(order, start=1):
            row = self.metadata.iloc[int(index)]
            mapped = map_timestamp_to_canonical(str(row.video_id), float(row.pts_time), self.by_video)
            output.append({
                "rank": rank,
                "video_id": str(row.video_id),
                "chunk": str(row.chunk),
                "start": float(row.start),
                "end": float(row.end),
                "timestamp": float(row.pts_time),
                "kf_n": int(row.kf_n),
                "frame_idx": int(row.frame_idx),
                "canonical_kf_n": int(mapped["kf_n"]),
                "canonical_frame_idx": int(mapped["frame_idx"]),
                "timestamp_distance_seconds": float(mapped["distance_seconds"]),
                "score": float(scores[index]),
                "score_mode": self.mode,
            })
        return output


def preflight_asr_index(
    index_dir: str | Path,
    *,
    canonical_name: str = "global_keyframes.parquet",
    expected_packs: Iterable[str] | None = None,
    require_embeddings: bool = False,
) -> dict[str, Any]:
    """Run the shared fail-closed ASR preflight.

    ``expected_packs=None`` intentionally means the full K01--K20/L21--L30
    corpus.  A mounted subset must be explicit via ``expected_packs`` so a
    diagnostic directory cannot look production-ready merely because it has
    a few discoverable files.
    """
    report = run_modality_index_preflight(
        index_dir,
        canonical_name=canonical_name,
        expected_packs=expected_packs,
        active_modalities=("asr",),
        require_embeddings=require_embeddings,
    )
    report["protocol"] = "offline ASR spoken-fact index preflight v2"
    # Preserve the v1 caller-facing alias while exposing the richer shared
    # report under ``asr``.
    report["packs"] = report.get("asr", [])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only offline ASR index preflight")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--canonical-name", default="global_keyframes.parquet")
    parser.add_argument("--pack", action="append", dest="packs")
    parser.add_argument("--require-embeddings", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight_asr_index(
        args.index_dir,
        canonical_name=args.canonical_name,
        expected_packs=args.packs,
        require_embeddings=args.require_embeddings,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": len(report["errors"]), "packs": len(report.get("packs", []))}, ensure_ascii=False))
    return 0 if report["passed"] else 1


__all__ = [
    "ASRIndex",
    "ASRIndexPreflightError",
    "map_timestamp_to_canonical",
    "mojibake_score",
    "normalize_transcript",
    "preflight_asr_index",
    "preflight_asr_metadata",
]


if __name__ == "__main__":
    raise SystemExit(main())
