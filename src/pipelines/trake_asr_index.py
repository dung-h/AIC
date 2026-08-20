"""Shared, read-only ASR index adapter for the TRAKE runtime.

The ASR global merge is the data-plane boundary for TRAKE.  This adapter
normalizes the persisted ``retrieval.parquet`` schema once, validates its
embedding alignment and canonical frame mapping, and exposes the historical
``vid``/``chunk`` columns used by the alignment code.  It intentionally does
not discover or concatenate legacy shards: a production TRAKE request must
point at one merged index with one manifest.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MERGED_RELATIVE = Path("modality_global_v2/asr_global_merged_v2")
MANIFEST_NAME = "asr_global_merge_v2_manifest.json"
REQUIRED_METADATA_COLUMNS = {
    "video_id",
    "text",
    "start",
    "end",
    "kf_n",
    "frame_idx",
    "pts_time",
}


class ASRGlobalIndexError(RuntimeError):
    """Raised when the shared ASR index is absent or not canonical-safe."""


class ASRIntervalError(ValueError):
    """Raised when ASR evidence cannot be represented as a valid interval."""


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict-like record without accepting missing values."""

    if isinstance(record, Mapping) or hasattr(record, "get"):
        return record.get(name, default)
    return default


def asr_interval(
    record_or_start: Any,
    end: Any = None,
) -> tuple[float, float]:
    """Return the explicit ``[start, end]`` interval for ASR evidence.

    A missing endpoint is an error by design.  Callers that historically only
    had a scalar timestamp should keep using the scalar-compatible matching
    APIs; they must not silently turn incomplete sentence/chunk evidence into
    a zero-width interval here.
    """

    if end is None:
        if isinstance(record_or_start, (tuple, list, np.ndarray)) and len(record_or_start) == 2:
            start_value, end_value = record_or_start
        else:
            start_value = _record_value(record_or_start, "start")
            end_value = _record_value(record_or_start, "end")
    else:
        start_value, end_value = record_or_start, end

    if start_value is None or end_value is None:
        raise ASRIntervalError("ASR evidence requires both start and end timestamps")
    try:
        start_value = float(start_value)
        end_value = float(end_value)
    except (TypeError, ValueError) as exc:
        raise ASRIntervalError("ASR evidence start/end must be numeric") from exc
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        raise ASRIntervalError("ASR evidence start/end must be finite")
    if start_value < 0 or end_value < 0 or end_value < start_value:
        raise ASRIntervalError(
            f"invalid ASR interval [{start_value}, {end_value}]"
        )
    return start_value, end_value


def representative_timestamp(
    record_or_start: Any,
    end: Any = None,
    *,
    pts_time: Any = None,
    strategy: str = "start",
) -> float:
    """Choose a timestamp while preserving the source interval semantics.

    ``start`` is the backward-compatible default used by the legacy evaluator.
    ``midpoint`` is useful when a sentence/chunk's center is the best temporal
    representative.  ``pts_time`` means the canonical keyframe timestamp and
    is intentionally not required to lie inside the ASR word interval: the
    merged index may map a chunk to a nearby canonical frame.
    """

    start_value, end_value = asr_interval(record_or_start, end)
    if pts_time is None:
        pts_time = _record_value(record_or_start, "pts_time")
    mode = str(strategy).strip().lower()
    if mode == "start":
        return start_value
    if mode == "end":
        return end_value
    if mode == "midpoint":
        return (start_value + end_value) / 2.0
    if mode == "pts_time":
        if pts_time is None:
            raise ASRIntervalError(
                "representative strategy 'pts_time' requires pts_time"
            )
        try:
            value = float(pts_time)
        except (TypeError, ValueError) as exc:
            raise ASRIntervalError("pts_time must be numeric") from exc
        if not np.isfinite(value) or value < 0:
            raise ASRIntervalError("pts_time must be a finite non-negative value")
        return value
    raise ValueError(
        "unknown representative strategy; choose start, midpoint, end, or pts_time"
    )


def interval_overlap_seconds(left: Any, right: Any) -> float:
    """Return temporal overlap in seconds for two explicit ASR intervals."""

    left_start, left_end = asr_interval(left)
    right_start, right_end = asr_interval(right)
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def interval_distance_seconds(left: Any, right: Any) -> float:
    """Return zero for overlap, otherwise the gap between two intervals."""

    left_start, left_end = asr_interval(left)
    right_start, right_end = asr_interval(right)
    if left_end >= right_start and right_end >= left_start:
        return 0.0
    if left_end < right_start:
        return right_start - left_end
    return left_start - right_end


def is_monotonic_alignment(
    evidence: list[Any],
    *,
    strategy: str = "start",
    strict: bool = True,
) -> bool:
    """Validate ordered evidence without collapsing intervals to one point.

    Missing ``start``/``end`` raises ``ASRIntervalError`` instead of being
    interpreted as zero.  The default strict ``start`` ordering matches the
    old DANTE path contract; callers can choose ``midpoint`` or ``pts_time``
    when that is the intended canonical ordering.
    """

    values = [
        representative_timestamp(item, strategy=strategy)
        for item in evidence
    ]
    if strict:
        return all(left < right for left, right in zip(values, values[1:]))
    return all(left <= right for left, right in zip(values, values[1:]))


def resolve_asr_global_dir(index_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the one ASR index directory used by explicit TRAKE ASR mode."""

    if index_dir is not None:
        return Path(index_dir).expanduser().resolve()
    configured = os.getenv("HCMAI_TRAKE_ASR_INDEX_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # ``INDEX_DIR`` is intentionally not imported here: tests and callers can
    # provide an isolated root without importing the global paths module.
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "data" / "index" / DEFAULT_MERGED_RELATIVE).resolve()


def _artifact_path(root: Path, value: Any, default_name: str) -> Path:
    """Resolve manifest artifacts without allowing a missing relative base."""

    if value:
        candidate = Path(str(value))
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        rooted = root / candidate
        if rooted.is_file():
            return rooted
        # Merge manifests written from the repository contain repo-relative
        # paths.  Keep this compatibility path, but still require the file to
        # exist under the current project root.
        project_candidate = Path(__file__).resolve().parents[2] / candidate
        if project_candidate.is_file():
            return project_candidate
    candidate = root / default_name
    if candidate.is_file():
        return candidate
    raise ASRGlobalIndexError(
        f"ASR global artifact is missing: {candidate}"
    )


def _numeric_column(table: pd.DataFrame, name: str, *, integer: bool = False) -> None:
    table[name] = pd.to_numeric(table[name], errors="raise")
    if integer:
        table[name] = table[name].astype(int)
    else:
        table[name] = table[name].astype(float)
    if not np.isfinite(table[name].to_numpy()).all():
        raise ASRGlobalIndexError(
            f"ASR global metadata column {name!r} contains non-finite values"
        )


class SharedASRGlobalIndex:
    """Canonical ASR metadata + embedding matrix loaded from one merged pack.

    ``no_speech_videos`` deliberately remains separate from ``metadata``:
    those videos are covered by the build but have no retrievable transcript
    rows.  A search therefore returns no evidence for them rather than
    manufacturing a frame or silently treating missing ASR as a failed build.
    """

    def __init__(
        self,
        index_dir: str | os.PathLike[str] | None = None,
        *,
        canonical_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = resolve_asr_global_dir(index_dir)
        self.manifest_path = self.root / MANIFEST_NAME
        if not self.manifest_path.is_file():
            raise ASRGlobalIndexError(
                "shared ASR global index manifest is missing: "
                f"{self.manifest_path}; build asr_global_merged_v2 first"
            )
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ASRGlobalIndexError(
                f"cannot read ASR global manifest {self.manifest_path}: {exc}"
            ) from exc
        if self.manifest.get("status") not in {None, "ready"}:
            raise ASRGlobalIndexError(
                f"ASR global index is not ready: {self.manifest.get('status')!r}"
            )

        retrieval_path = _artifact_path(
            self.root,
            self.manifest.get("artifacts", {}).get("retrieval"),
            "retrieval.parquet",
        )
        embedding_path = _artifact_path(
            self.root,
            self.manifest.get("artifacts", {}).get("embeddings"),
            "embeddings.npy",
        )
        try:
            metadata = pd.read_parquet(retrieval_path)
            embeddings = np.asarray(np.load(embedding_path, allow_pickle=False), dtype=np.float32)
        except Exception as exc:
            raise ASRGlobalIndexError(
                f"cannot load ASR global artifacts from {self.root}: {exc}"
            ) from exc

        missing = sorted(REQUIRED_METADATA_COLUMNS - set(metadata.columns))
        if missing:
            raise ASRGlobalIndexError(
                f"ASR global metadata missing columns: {missing}"
            )
        if embeddings.ndim != 2 or len(metadata) != embeddings.shape[0]:
            raise ASRGlobalIndexError(
                "ASR global metadata/embedding row mismatch: "
                f"metadata={len(metadata)}, embeddings={embeddings.shape}"
            )
        declared_rows = (self.manifest.get("rows", {}) or {}).get("metadata")
        if declared_rows is not None and int(declared_rows) != len(metadata):
            raise ASRGlobalIndexError(
                "ASR global manifest row count disagrees with retrieval metadata: "
                f"manifest={declared_rows}, actual={len(metadata)}"
            )
        declared_shape = (self.manifest.get("embedding", {}) or {}).get("shape")
        if declared_shape is not None and list(declared_shape) != list(embeddings.shape):
            raise ASRGlobalIndexError(
                "ASR global manifest embedding shape disagrees with embeddings.npy: "
                f"manifest={declared_shape}, actual={list(embeddings.shape)}"
            )
        if not np.isfinite(embeddings).all():
            raise ASRGlobalIndexError("ASR global embeddings contain non-finite values")

        table = metadata.copy()
        if "embedding_row" in table.columns:
            rows = pd.to_numeric(table["embedding_row"], errors="raise").astype(int)
            if sorted(rows.tolist()) != list(range(len(table))):
                raise ASRGlobalIndexError("ASR global embedding_row is not a permutation")
            order = np.argsort(rows.to_numpy(), kind="mergesort")
            table = table.iloc[order].reset_index(drop=True)
            embeddings = embeddings[order]
        table["video_id"] = table["video_id"].astype(str).str.strip().str.upper()
        table["text"] = table["text"].fillna("").astype(str).str.replace(
            r"\s+", " ", regex=True
        ).str.strip()
        if table["video_id"].eq("").any() or table["text"].eq("").any():
            raise ASRGlobalIndexError("ASR global metadata contains empty video_id/text")
        for column in ("kf_n", "frame_idx"):
            _numeric_column(table, column, integer=True)
        for column in ("start", "end", "pts_time"):
            _numeric_column(table, column)
        if (table["start"] < 0).any() or (table["end"] < table["start"]).any():
            raise ASRGlobalIndexError("ASR global metadata contains invalid chunk timestamps")
        if table[["kf_n", "frame_idx"]].lt(0).any().any():
            raise ASRGlobalIndexError("ASR global metadata contains negative canonical indices")
        if table.duplicated(["video_id", "chunk_index"] if "chunk_index" in table else ["video_id", "start", "end"]).any():
            raise ASRGlobalIndexError("ASR global metadata contains duplicate chunk identities")

        self.metadata = table
        self.embeddings = embeddings
        self.no_speech_videos = self._load_no_speech_videos()
        if set(self.metadata["video_id"]) & self.no_speech_videos:
            raise ASRGlobalIndexError("a video is both transcribed and marked no-speech")
        self._validate_canonical_mapping(canonical_path)

        # Compatibility view for the alignment implementation.  The source
        # provenance remains available in ``metadata``; these aliases avoid
        # reintroducing the legacy shard loader.
        self.chunks = self.metadata.copy()
        self.chunks["vid"] = self.chunks["video_id"]
        self.chunks["chunk"] = self.chunks["text"]
        self.embeddings = np.ascontiguousarray(self.embeddings, dtype=np.float32)

    @property
    def videos(self) -> tuple[str, ...]:
        return tuple(sorted(self.metadata["video_id"].unique().tolist()))

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def _load_no_speech_videos(self) -> set[str]:
        values: set[str] = set()
        for report in (self.manifest.get("packs", {}) or {}).values():
            for video_id in report.get("no_speech_videos", []) or []:
                text = str(video_id).strip().upper()
                if text:
                    values.add(text)
        scope_ids = set(
            str(value).strip().upper()
            for value in (self.manifest.get("scope", {}) or {}).get("video_ids", [])
            if str(value).strip()
        )
        observed = set(self.metadata["video_id"])
        covered = observed | values
        if scope_ids and covered != scope_ids:
            missing = sorted(scope_ids - covered)
            extra = sorted(covered - scope_ids)
            raise ASRGlobalIndexError(
                "ASR global video coverage disagrees with manifest scope; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        return values

    def _validate_canonical_mapping(
        self, canonical_path: str | os.PathLike[str] | None
    ) -> None:
        configured = canonical_path
        if configured is None:
            manifest_path = (self.manifest.get("canonical", {}) or {}).get("path")
            configured = manifest_path
        if configured is None:
            default = self.root.parent.parent / "global_keyframes.parquet"
            configured = default if default.is_file() else None
        if configured is None:
            raise ASRGlobalIndexError(
                "ASR global index has no canonical frame map; refusing to load"
            )
        path = Path(configured)
        if not path.is_absolute():
            candidates = [self.root / path, Path(__file__).resolve().parents[2] / path]
            path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if not path.is_file():
            raise ASRGlobalIndexError(f"canonical frame map is missing: {path}")
        try:
            canonical = pd.read_parquet(path, columns=["video_id", "kf_n", "frame_idx", "pts_time"])
        except Exception as exc:
            raise ASRGlobalIndexError(f"cannot load canonical frame map {path}: {exc}") from exc
        canonical["video_id"] = canonical["video_id"].astype(str).str.strip().str.upper()
        for column in ("kf_n", "frame_idx"):
            _numeric_column(canonical, column, integer=True)
        _numeric_column(canonical, "pts_time")
        if canonical.duplicated(["video_id", "kf_n"]).any():
            raise ASRGlobalIndexError("canonical frame map has duplicate video_id/kf_n")
        keys = canonical.set_index(["video_id", "kf_n"])
        probe = self.metadata.set_index(["video_id", "kf_n"])
        missing = ~probe.index.isin(keys.index)
        if missing.any():
            raise ASRGlobalIndexError(
                "ASR global metadata contains frame keys outside canonical map"
            )
        expected = keys.loc[probe.index, ["frame_idx", "pts_time"]].reset_index(drop=True)
        actual = self.metadata[["frame_idx", "pts_time"]].reset_index(drop=True)
        if not np.array_equal(actual["frame_idx"].to_numpy(), expected["frame_idx"].to_numpy()):
            raise ASRGlobalIndexError("ASR global frame_idx disagrees with canonical map")
        if not np.allclose(actual["pts_time"].to_numpy(), expected["pts_time"].to_numpy(), atol=1e-4):
            raise ASRGlobalIndexError("ASR global pts_time disagrees with canonical map")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "index_dir": str(self.root),
            "manifest": str(self.manifest_path),
            "index_id": self.manifest.get("index_id"),
            "rows": int(len(self.metadata)),
            "videos_with_transcript": int(self.metadata["video_id"].nunique()),
            "no_speech_videos": sorted(self.no_speech_videos),
            "dimension": self.dimension,
            "canonical_validated": True,
        }
