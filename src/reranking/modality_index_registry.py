"""Shared local modality-index registry.

The ASR global merge is the production boundary for transcript retrieval.  It
is intentionally kept separate from the Q&A router so other consumers can use
the same artifact without re-discovering legacy per-pack shards.  This module
only reads local files; it never downloads, repairs, or mutates an index.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


GLOBAL_ASR_RELATIVE_DIR = Path("modality_global_v2") / "asr_global_merged_v2"
GLOBAL_ASR_MANIFEST_NAME = "asr_global_merge_v2_manifest.json"
GLOBAL_ASR_RETRIEVAL_NAME = "retrieval.parquet"
GLOBAL_ASR_EMBEDDINGS_NAME = "embeddings.npy"
GLOBAL_OCR_RELATIVE_DIR = Path("modality_global_v2") / "ocr_global_merged_v2"
GLOBAL_OCR_MANIFEST_NAME = "manifest.json"
GLOBAL_OCR_RETRIEVAL_NAME = "retrieval.parquet"
GLOBAL_OCR_EMBEDDINGS_NAME = "embeddings.npy"
GLOBAL_ASR_REQUIRED_COLUMNS = {
    "video_id", "chunk_index", "text", "start", "end", "kf_n",
    "frame_idx", "pts_time", "embedding_row", "source_pack",
    "source_provenance",
}
GLOBAL_OCR_REQUIRED_COLUMNS = {
    "video_id", "kf_n", "frame_idx", "pts_time", "ocr_text",
    "embedding_row", "source_pack",
}
PACK_RE = re.compile(r"^[kl]\d{2}$", re.IGNORECASE)
DEFAULT_EXPECTED_PACKS = tuple(
    [f"k{i:02d}" for i in range(1, 21)]
    + [f"l{i:02d}" for i in range(21, 31)]
)


class ModalityIndexRegistryError(RuntimeError):
    """Raised when a local shared modality index cannot be trusted."""


@dataclass(frozen=True)
class GlobalASRSource:
    """Resolved paths and manifest for the merged ASR source."""

    directory: Path
    manifest_path: Path
    retrieval_path: Path
    embeddings_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class GlobalOCRSource:
    """Resolved paths and manifest for the sampled global OCR source."""

    directory: Path
    manifest_path: Path
    retrieval_path: Path
    embeddings_path: Path
    manifest: dict[str, Any]


def _normalise_packs(values: Iterable[str] | None) -> tuple[str, ...]:
    raw = DEFAULT_EXPECTED_PACKS if values is None else values
    output: list[str] = []
    seen: set[str] = set()
    for value in raw:
        pack = str(value).strip().lower()
        if not PACK_RE.fullmatch(pack):
            raise ValueError(f"invalid modality pack name: {value!r}")
        if pack not in seen:
            output.append(pack)
            seen.add(pack)
    if not output:
        raise ValueError("expected_packs must contain at least one pack")
    return tuple(output)


def _resolve_manifest_path(value: Any, *, root: Path, manifest_path: Path, fallback: Path) -> Path:
    """Resolve paths emitted by both repo-relative and local manifests."""
    if value:
        candidate = Path(str(value))
        if candidate.is_absolute():
            return candidate
        candidates = (
            manifest_path.parent / candidate,
            root / candidate,
            root.parent.parent / candidate,
        )
        for item in candidates:
            if item.exists():
                return item
    return fallback


class ModalityIndexRegistry:
    """Resolve and load shared local modality artifacts.

    The registry currently has one production source, the merged ASR v2
    artifact.  Legacy per-pack sources remain available to the preflight's
    compatibility fixtures and diagnostic mode, but are never preferred when
    a global manifest is present.
    """

    def __init__(
        self,
        index_dir: str | Path,
        canonical_name: str = "global_keyframes.parquet",
        global_asr_dir: str | Path | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.canonical_path = self.index_dir / canonical_name
        self.global_asr_dir = (
            Path(global_asr_dir)
            if global_asr_dir is not None and str(global_asr_dir).strip()
            else self.index_dir / GLOBAL_ASR_RELATIVE_DIR
        )
        self.global_ocr_dir = self.index_dir / GLOBAL_OCR_RELATIVE_DIR
        self._source: GlobalASRSource | None = None
        self._ocr_source: GlobalOCRSource | None = None

    def discover_global_asr(self) -> GlobalASRSource | None:
        """Return the merged ASR source, or ``None`` if it is not present.

        A partially materialized global directory is returned as a source as
        well; callers then fail closed with a precise missing-artifact error
        instead of silently falling back to a different ASR corpus.
        """
        if self._source is not None:
            return self._source
        manifest_path = self.global_asr_dir / GLOBAL_ASR_MANIFEST_NAME
        retrieval_fallback = self.global_asr_dir / GLOBAL_ASR_RETRIEVAL_NAME
        embeddings_fallback = self.global_asr_dir / GLOBAL_ASR_EMBEDDINGS_NAME
        if not (manifest_path.exists() or retrieval_fallback.exists() or embeddings_fallback.exists()):
            return None
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ModalityIndexRegistryError(
                    f"invalid merged ASR manifest {manifest_path}: {exc}"
                ) from exc
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
        self._source = GlobalASRSource(
            directory=self.global_asr_dir,
            manifest_path=manifest_path,
            retrieval_path=_resolve_manifest_path(
                artifacts.get("retrieval"), root=self.index_dir,
                manifest_path=manifest_path, fallback=retrieval_fallback,
            ),
            embeddings_path=_resolve_manifest_path(
                artifacts.get("embeddings"), root=self.index_dir,
                manifest_path=manifest_path, fallback=embeddings_fallback,
            ),
            manifest=manifest,
        )
        return self._source

    @staticmethod
    def _canonical_map(frame: pd.DataFrame) -> dict[tuple[str, int], tuple[int, float]]:
        required = {"video_id", "kf_n", "frame_idx", "pts_time"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ModalityIndexRegistryError(
                f"canonical map is missing columns: {missing}"
            )
        result: dict[tuple[str, int], tuple[int, float]] = {}
        for row_number, row in frame.iterrows():
            try:
                video_id = str(row["video_id"]).strip().upper()
                kf_n = int(row["kf_n"])
                frame_idx = int(row["frame_idx"])
                pts_time = float(row["pts_time"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ModalityIndexRegistryError(
                    f"canonical map has invalid row {row_number}: {exc}"
                ) from exc
            if not video_id or kf_n < 0 or frame_idx < 0 or not math.isfinite(pts_time) or pts_time < 0:
                raise ModalityIndexRegistryError(
                    f"canonical map has invalid row {row_number}"
                )
            key = (video_id, kf_n)
            previous = result.get(key)
            current = (frame_idx, pts_time)
            if previous is not None and previous != current:
                raise ModalityIndexRegistryError(
                    f"canonical map has conflicting mapping for {key}"
                )
            result[key] = current
        return result

    def load_asr(
        self,
        *,
        expected_packs: Iterable[str] | None = None,
        require_embeddings: bool = True,
        strict: bool = True,
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
        """Load merged ASR rows and embeddings after canonical validation.

        Returned metadata uses the router's stable ``chunk`` text column while
        retaining global provenance and ``asr_status``.  No-speech videos are
        represented in the returned info, not as fake transcript rows.
        """
        source = self.discover_global_asr()
        if source is None:
            raise ModalityIndexRegistryError(
                f"merged ASR source is not present under {self.global_asr_dir}"
            )
        manifest = source.manifest
        if manifest.get("status") != "ready":
            raise ModalityIndexRegistryError(
                f"merged ASR manifest is not ready: {manifest.get('status')!r}"
            )
        packs = _normalise_packs(expected_packs)
        manifest_packs = {
            str(value).strip().lower()
            for value in manifest.get("scope", {}).get("packs", [])
        }
        missing_manifest_packs = sorted(set(packs) - manifest_packs)
        if missing_manifest_packs:
            raise ModalityIndexRegistryError(
                f"merged ASR manifest does not cover packs: {missing_manifest_packs}"
            )
        if not source.retrieval_path.is_file():
            raise ModalityIndexRegistryError(
                f"merged ASR retrieval metadata is missing: {source.retrieval_path}"
            )
        if require_embeddings and not source.embeddings_path.is_file():
            raise ModalityIndexRegistryError(
                f"merged ASR embeddings are missing: {source.embeddings_path}"
            )
        try:
            metadata = pd.read_parquet(source.retrieval_path).reset_index(drop=True)
        except Exception as exc:
            raise ModalityIndexRegistryError(
                f"cannot read merged ASR metadata: {exc}"
            ) from exc
        missing_columns = sorted(GLOBAL_ASR_REQUIRED_COLUMNS - set(metadata.columns))
        if missing_columns:
            raise ModalityIndexRegistryError(
                f"merged ASR metadata is missing columns: {missing_columns}"
            )
        embeddings: np.ndarray
        if require_embeddings:
            try:
                embeddings = np.asarray(
                    np.load(source.embeddings_path, mmap_mode="r", allow_pickle=False),
                    dtype=np.float32,
                )
            except Exception as exc:
                raise ModalityIndexRegistryError(
                    f"cannot read merged ASR embeddings: {exc}"
                ) from exc
            if embeddings.ndim != 2 or len(embeddings) != len(metadata):
                raise ModalityIndexRegistryError(
                    "merged ASR embedding/metadata row alignment failed: "
                    f"{embeddings.shape} vs {len(metadata)}"
                )
            manifest_shape = tuple(manifest.get("embedding", {}).get("shape", ()))
            if manifest_shape and tuple(embeddings.shape) != manifest_shape:
                raise ModalityIndexRegistryError(
                    f"merged ASR shape disagrees with manifest: {embeddings.shape} vs {manifest_shape}"
                )
            if not np.isfinite(embeddings).all():
                raise ModalityIndexRegistryError("merged ASR embeddings contain non-finite values")
        else:
            embeddings = np.empty((len(metadata), 0), dtype=np.float32)

        metadata["video_id"] = metadata["video_id"].astype(str).str.strip().str.upper()
        metadata["source_pack"] = metadata["source_pack"].astype(str).str.strip().str.lower()
        metadata["text"] = metadata["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        if strict and metadata["text"].eq("").any():
            raise ModalityIndexRegistryError("merged ASR metadata contains empty transcript text")
        if strict and metadata["source_provenance"].astype(str).str.strip().eq("").any():
            raise ModalityIndexRegistryError("merged ASR metadata contains empty provenance")

        row_ids = pd.to_numeric(metadata["embedding_row"], errors="coerce")
        expected_row_ids = np.arange(len(metadata), dtype=np.int64)
        if row_ids.isna().any() or not np.array_equal(row_ids.to_numpy(dtype=np.int64), expected_row_ids):
            raise ModalityIndexRegistryError("merged ASR embedding_row is not a contiguous aligned range")

        canonical = self._canonical_map(pd.read_parquet(self.canonical_path))
        manifest_pack_info = manifest.get("packs", {})
        all_no_speech: set[str] = set()
        all_observed: set[str] = set(metadata["video_id"])
        for pack_name, pack_info in manifest_pack_info.items():
            pack = str(pack_name).strip().lower()
            values = pack_info.get("no_speech_videos", []) if isinstance(pack_info, dict) else []
            no_speech = {str(value).strip().upper() for value in values if str(value).strip()}
            all_no_speech.update(no_speech)
            if no_speech & all_observed:
                raise ModalityIndexRegistryError(
                    f"merged ASR marks transcript-bearing videos as no-speech in {pack}: "
                    f"{sorted(no_speech & all_observed)[:5]}"
                )

        for row_number, row in metadata.iterrows():
            try:
                key = (str(row["video_id"]), int(row["kf_n"]))
                frame_idx = int(row["frame_idx"])
                pts_time = float(row["pts_time"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ModalityIndexRegistryError(
                    f"merged ASR row {row_number} has invalid canonical fields: {exc}"
                ) from exc
            expected = canonical.get(key)
            if expected is None:
                raise ModalityIndexRegistryError(
                    f"merged ASR row {row_number} is outside canonical map: {key}"
                )
            if frame_idx != expected[0] or abs(pts_time - expected[1]) > 1e-2:
                raise ModalityIndexRegistryError(
                    f"merged ASR row {row_number} disagrees with canonical map: {key}"
                )
            if str(row["source_pack"]).lower() != key[0][:3].lower():
                raise ModalityIndexRegistryError(
                    f"merged ASR row {row_number} has incorrect source_pack for {key[0]}"
                )

        canonical_by_pack: dict[str, set[str]] = {}
        for video_id in {key[0] for key in canonical}:
            canonical_by_pack.setdefault(video_id[:3].lower(), set()).add(video_id)
        for pack in packs:
            expected_videos = canonical_by_pack.get(pack, set())
            observed_videos = set(metadata.loc[metadata["source_pack"] == pack, "video_id"])
            no_speech = {
                value for value in all_no_speech if value[:3].lower() == pack
            }
            if strict and observed_videos & no_speech:
                raise ModalityIndexRegistryError(f"ASR coverage conflict in {pack}")
            covered = observed_videos | no_speech
            if strict and covered != expected_videos:
                missing = sorted(expected_videos - covered)
                extra = sorted(covered - expected_videos)
                raise ModalityIndexRegistryError(
                    f"merged ASR video coverage mismatch in {pack}: missing={missing[:5]}, extra={extra[:5]}"
                )

        keep = metadata["source_pack"].isin(set(packs)).to_numpy()
        metadata = metadata.loc[keep].reset_index(drop=True)
        if require_embeddings:
            embeddings = embeddings[keep]
        metadata = metadata.rename(columns={"text": "chunk"})
        metadata["asr_status"] = "transcript"
        info = {
            "source": "global_merged_v2",
            "index_id": manifest.get("index_id"),
            "manifest_path": str(source.manifest_path),
            "retrieval_path": str(source.retrieval_path),
            "embeddings_path": str(source.embeddings_path),
            "no_speech_videos": sorted(value for value in all_no_speech if value[:3].lower() in set(packs)),
            "transcript_video_count": int(metadata["video_id"].nunique()),
            "covered_video_count": int(
                metadata["video_id"].nunique()
                + len([value for value in all_no_speech if value[:3].lower() in set(packs)])
            ),
        }
        return embeddings, metadata, info

    def discover_global_ocr(self) -> GlobalOCRSource | None:
        """Return the sampled global OCR source, if materialized."""
        if self._ocr_source is not None:
            return self._ocr_source
        manifest_path = self.global_ocr_dir / GLOBAL_OCR_MANIFEST_NAME
        retrieval_fallback = self.global_ocr_dir / GLOBAL_OCR_RETRIEVAL_NAME
        embeddings_fallback = self.global_ocr_dir / GLOBAL_OCR_EMBEDDINGS_NAME
        if not (manifest_path.exists() or retrieval_fallback.exists() or embeddings_fallback.exists()):
            return None
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ModalityIndexRegistryError(
                    f"invalid global OCR manifest {manifest_path}: {exc}"
                ) from exc
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
        self._ocr_source = GlobalOCRSource(
            directory=self.global_ocr_dir,
            manifest_path=manifest_path,
            retrieval_path=_resolve_manifest_path(
                artifacts.get("retrieval"), root=self.index_dir,
                manifest_path=manifest_path, fallback=retrieval_fallback,
            ),
            embeddings_path=_resolve_manifest_path(
                artifacts.get("embeddings"), root=self.index_dir,
                manifest_path=manifest_path, fallback=embeddings_fallback,
            ),
            manifest=manifest,
        )
        return self._ocr_source

    def load_ocr(
        self,
        *,
        expected_packs: Iterable[str] | None = None,
        require_embeddings: bool = True,
        strict: bool = True,
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
        """Load sampled global OCR rows after canonical validation.

        OCR is sampled by timestamp, so coverage is video-level rather than
        every-keyframe. The persisted manifest is the authority for that
        coverage claim; each returned row must still map to a canonical
        ``(video_id, kf_n)`` identity.
        """
        source = self.discover_global_ocr()
        if source is None:
            raise ModalityIndexRegistryError(
                f"global OCR source is not present under {self.global_ocr_dir}"
            )
        manifest = source.manifest
        if manifest.get("status") != "ready":
            raise ModalityIndexRegistryError(
                f"global OCR manifest is not ready: {manifest.get('status')!r}"
            )
        packs = _normalise_packs(expected_packs)
        scope = manifest.get("scope", {}) if isinstance(manifest, dict) else {}
        manifest_packs = {str(value).strip().lower() for value in scope.get("packs", [])}
        missing_packs = sorted(set(packs) - manifest_packs)
        if missing_packs:
            raise ModalityIndexRegistryError(
                f"global OCR manifest does not cover packs: {missing_packs}"
            )
        if not source.retrieval_path.is_file():
            raise ModalityIndexRegistryError(
                f"global OCR retrieval metadata is missing: {source.retrieval_path}"
            )
        if require_embeddings and not source.embeddings_path.is_file():
            raise ModalityIndexRegistryError(
                f"global OCR embeddings are missing: {source.embeddings_path}"
            )
        try:
            metadata = pd.read_parquet(source.retrieval_path).reset_index(drop=True)
        except Exception as exc:
            raise ModalityIndexRegistryError(f"cannot read global OCR metadata: {exc}") from exc
        missing_columns = sorted(GLOBAL_OCR_REQUIRED_COLUMNS - set(metadata.columns))
        if missing_columns:
            raise ModalityIndexRegistryError(
                f"global OCR metadata is missing columns: {missing_columns}"
            )
        if require_embeddings:
            try:
                embeddings = np.asarray(
                    np.load(source.embeddings_path, mmap_mode="r", allow_pickle=False),
                    dtype=np.float32,
                )
            except Exception as exc:
                raise ModalityIndexRegistryError(f"cannot read global OCR embeddings: {exc}") from exc
            if embeddings.ndim != 2 or len(embeddings) != len(metadata):
                raise ModalityIndexRegistryError(
                    f"global OCR embedding/metadata mismatch: {embeddings.shape} vs {len(metadata)}"
                )
        else:
            embeddings = np.empty((len(metadata), 0), dtype=np.float32)

        metadata["video_id"] = metadata["video_id"].astype(str).str.strip().str.upper()
        metadata["source_pack"] = metadata["source_pack"].astype(str).str.strip().str.lower()
        metadata["ocr_text"] = metadata["ocr_text"].fillna("").astype(str).str.strip()
        if strict and metadata["ocr_text"].eq("").any():
            raise ModalityIndexRegistryError("global OCR metadata contains empty OCR text")
        row_ids = pd.to_numeric(metadata["embedding_row"], errors="coerce")
        if row_ids.isna().any() or not np.array_equal(
            row_ids.to_numpy(dtype=np.int64), np.arange(len(metadata), dtype=np.int64)
        ):
            raise ModalityIndexRegistryError("global OCR embedding_row is not contiguous")

        canonical = self._canonical_map(pd.read_parquet(self.canonical_path))
        for row_number, row in metadata.iterrows():
            try:
                key = (str(row["video_id"]), int(row["kf_n"]))
                frame_idx = int(row["frame_idx"])
                pts_time = float(row["pts_time"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ModalityIndexRegistryError(
                    f"global OCR row {row_number} has invalid canonical fields: {exc}"
                ) from exc
            expected = canonical.get(key)
            if expected is None:
                raise ModalityIndexRegistryError(
                    f"global OCR row {row_number} is outside canonical map: {key}"
                )
            if frame_idx != expected[0] or abs(pts_time - expected[1]) > 1e-2:
                raise ModalityIndexRegistryError(
                    f"global OCR row {row_number} disagrees with canonical map: {key}"
                )
            if str(row["source_pack"]).lower() != key[0][:3].lower():
                raise ModalityIndexRegistryError(
                    f"global OCR row {row_number} has incorrect source_pack for {key[0]}"
                )
        keep = metadata["source_pack"].isin(set(packs)).to_numpy()
        metadata = metadata.loc[keep].reset_index(drop=True)
        if require_embeddings:
            embeddings = embeddings[keep]
        info = {
            "source": "global_merged_v2",
            "index_id": manifest.get("index_id"),
            "manifest_path": str(source.manifest_path),
            "retrieval_path": str(source.retrieval_path),
            "embeddings_path": str(source.embeddings_path),
            "sample_interval_seconds": manifest.get("sampling", {}).get("sample_interval_seconds"),
            "covered_video_count": manifest.get("coverage", {}).get("covered_videos"),
            "sampled_ocr_rows": len(metadata),
        }
        return embeddings, metadata, info


__all__ = [
    "GLOBAL_ASR_RELATIVE_DIR",
    "GLOBAL_ASR_MANIFEST_NAME",
    "GLOBAL_ASR_RETRIEVAL_NAME",
    "GLOBAL_ASR_EMBEDDINGS_NAME",
    "GLOBAL_OCR_RELATIVE_DIR",
    "GLOBAL_OCR_MANIFEST_NAME",
    "GLOBAL_OCR_RETRIEVAL_NAME",
    "GLOBAL_OCR_EMBEDDINGS_NAME",
    "GlobalASRSource",
    "GlobalOCRSource",
    "ModalityIndexRegistry",
    "ModalityIndexRegistryError",
]
