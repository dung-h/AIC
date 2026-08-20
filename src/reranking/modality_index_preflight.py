"""Fail-closed validation for the local ASR/OCR retrieval indexes.

The checked-in modality manifest is only an informational snapshot.  This
module validates the files that are actually present before a specialist
route is allowed to run.  It deliberately does not download, repair, or
rewrite any index data.
"""
from __future__ import annotations

from pathlib import Path
import math
import json
import re
import unicodedata
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.reranking.modality_index_registry import (  # noqa: E402
    GLOBAL_ASR_EMBEDDINGS_NAME,
    GLOBAL_ASR_MANIFEST_NAME,
    GLOBAL_ASR_RELATIVE_DIR,
    GLOBAL_ASR_RETRIEVAL_NAME,
    GLOBAL_OCR_EMBEDDINGS_NAME,
    GLOBAL_OCR_MANIFEST_NAME,
    GLOBAL_OCR_RELATIVE_DIR,
    GLOBAL_OCR_RETRIEVAL_NAME,
    ModalityIndexRegistry,
    ModalityIndexRegistryError,
)


DEFAULT_EXPECTED_PACKS = tuple(
    [f"k{i:02d}" for i in range(1, 21)]
    + [f"l{i:02d}" for i in range(21, 31)]
)
BENCHMARK_K_PACKS = tuple(f"k{i:02d}" for i in range(1, 21))
PACK_NAME_RE = re.compile(r"^[kl]\d{2}$", re.IGNORECASE)
EMBEDDING_DIM = 1024
CANONICAL_COLUMNS = {"video_id", "kf_n", "frame_idx", "pts_time"}
ASR_PACK_RE = re.compile(r"^asr_chunks_(k\d{2}|l\d{2})_ts\.parquet$", re.IGNORECASE)
ASR_EMBED_RE = re.compile(r"^emb_cache_asr_(k\d{2}|l\d{2})_chunks\.npy$", re.IGNORECASE)
OCR_PACK_RE = re.compile(r"^ocr_(k\d{2}|l\d{2})\.parquet$", re.IGNORECASE)
OCR_EMBED_RE = re.compile(r"^emb_cache_ocr_(k\d{2}|l\d{2})\.npy$", re.IGNORECASE)

# These are only candidate byte-decoding artefacts.  Several characters which
# look suspicious to a byte-oriented detector are valid Vietnamese letters
# (notably ``â`` and ``Â``), so marker presence alone must never become a
# production warning.  A warning is emitted only when a strict cp1252/latin-1
# -> UTF-8 round-trip produces a different string with fewer artefact tokens.
MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "Ð", "Ñ", "Ä", "Å", "Æ", "�")
_MOJIBAKE_RE = re.compile(
    r"(?:Ã.|Â(?:°|\xa0|©|®|“|”|™|\s|$)|Ä.|Å.|Æ.|Ç.|á».|áº.|ï¿½)",
    re.UNICODE,
)


class ModalityIndexPreflightError(RuntimeError):
    """Raised when a strict modality route cannot be trusted."""


def normalise_expected_packs(expected_packs: Iterable[str] | None) -> tuple[str, ...]:
    """Canonicalise an explicit validation scope without broadening it.

    ``None`` is deliberately the full-corpus scope.  Callers that benchmark a
    mounted subset must pass the active packs explicitly; missing packs inside
    that scope still fail closed.
    """
    values = DEFAULT_EXPECTED_PACKS if expected_packs is None else expected_packs
    packs: list[str] = []
    seen: set[str] = set()
    for value in values:
        pack = str(value).strip().lower()
        if not PACK_NAME_RE.fullmatch(pack):
            raise ValueError(f"invalid modality pack name: {value!r}")
        if pack not in seen:
            packs.append(pack)
            seen.add(pack)
    if not packs:
        raise ValueError("expected_packs must contain at least one pack")
    return tuple(packs)


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {"code": code, "message": message}
    item.update(extra)
    return item


def _empty_pack(pack: str, modality: str) -> dict[str, Any]:
    return {
        "pack": pack,
        "modality": modality,
        "embedding": None,
        "metadata": None,
        "errors": [],
        "warnings": [],
        "embedding_rows": 0,
        "embedding_dim": None,
        "metadata_rows": 0,
        "video_count": 0,
        "video_ids": [],
        "text_quality": {
            "rows": 0,
            "empty_rows": 0,
            "replacement_chars": 0,
            "mojibake_token_count": 0,
            "nfkc_changes": 0,
        },
        "canonical_missing_rows": 0,
        "mapping_mismatch_rows": 0,
        "source": "legacy_pack",
        "no_speech_videos": [],
        "covered_video_count": 0,
    }


def _default_read_parquet(path: Path):
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - pandas is a project dependency
        raise RuntimeError(f"pandas unavailable: {exc}") from exc
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            f"cannot read parquet {path.name}; install a local parquet engine "
            f"(pyarrow/fastparquet): {exc}"
        ) from exc


def _read_npy(path: Path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(math.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _mojibake_score(value: Any) -> int:
    """Count only byte-leak patterns, not ordinary Vietnamese characters."""
    return len(_MOJIBAKE_RE.findall(_as_text(value)))


def _repairable_mojibake(value: Any) -> bool:
    """Return whether a deterministic byte round-trip proves corruption.

    This intentionally does not classify a marker as bad by itself.  For
    example, ``NÃO`` and ordinary Vietnamese ``â`` contain characters that a
    broad marker list catches, but no valid round-trip improves them.  Mixed
    clean/corrupt text is checked token-wise when the whole sentence cannot
    be decoded in one pass.
    """
    current = _as_text(value)
    before = _mojibake_score(current)
    if before == 0 or "\ufffd" in current:
        return False

    candidates: list[str] = []
    for encoding in ("cp1252", "latin-1"):
        try:
            candidates.append(current.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    if candidates:
        best = min(candidates, key=_mojibake_score)
        return (
            best != current
            and _mojibake_score(best) < before
            and best.count("\ufffd") <= current.count("\ufffd")
        )

    # The same safe check for a sentence containing both valid Unicode and a
    # damaged token.  Do not mutate anything here; this is validation only.
    for piece in re.split(r"(\s+)", current):
        piece_score = _mojibake_score(piece)
        if piece_score == 0 or "\ufffd" in piece:
            continue
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = piece.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if (
                candidate != piece
                and _mojibake_score(candidate) < piece_score
                and candidate.count("\ufffd") <= piece.count("\ufffd")
            ):
                return True
    return False


def _text_quality(values: Iterable[Any]) -> dict[str, int]:
    empty = 0
    replacement = 0
    marker_rows = 0
    mojibake = 0
    nfkc_changes = 0
    rows = 0
    for value in values:
        rows += 1
        text = _as_text(value)
        if not text.strip():
            empty += 1
        replacement += text.count("\ufffd")
        if any(marker in text for marker in MOJIBAKE_MARKERS):
            marker_rows += 1
        if _repairable_mojibake(text):
            mojibake += 1
        if unicodedata.normalize("NFKC", text) != text:
            nfkc_changes += 1
    return {
        "rows": rows,
        "empty_rows": empty,
        "replacement_chars": replacement,
        "marker_rows": marker_rows,
        "mojibake_token_count": mojibake,
        "nfkc_changes": nfkc_changes,
    }


def _series_values(frame: Any, column: str) -> list[Any]:
    values = frame[column]
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _normalise_columns(columns: Iterable[Any]) -> set[str]:
    return {str(column).strip() for column in columns}


def _pick_column(columns: set[str], *names: str) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def _pack_from_video_id(video_id: Any) -> str | None:
    """Return the canonical K/L pack prefix for one video identity."""
    match = re.match(r"^([kl]\d{2})_", _as_text(video_id).strip(), re.IGNORECASE)
    return match.group(1).lower() if match else None


def _numeric_values(frame: Any, column: str) -> np.ndarray:
    values = _series_values(frame, column)
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float64)


def _key_set(frame: Any, video_column: str, kf_column: str) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for video, kf in zip(_series_values(frame, video_column), _series_values(frame, kf_column)):
        try:
            keys.add((str(video), int(kf)))
        except (TypeError, ValueError):
            continue
    return keys


def _canonical_index(frame: Any) -> tuple[dict[tuple[str, int], tuple[int, float]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    columns = _normalise_columns(frame.columns)
    missing = sorted(CANONICAL_COLUMNS - columns)
    if missing:
        return {}, [_error("canonical_missing_columns", "canonical map is missing columns", columns=missing)]

    result: dict[tuple[str, int], tuple[int, float]] = {}
    for row_number, (video, kf, frame_idx, pts_time) in enumerate(
        zip(
            _series_values(frame, "video_id"),
            _series_values(frame, "kf_n"),
            _series_values(frame, "frame_idx"),
            _series_values(frame, "pts_time"),
        )
    ):
        try:
            video_id = str(video).strip()
            kf_value = int(kf)
            frame_value = int(frame_idx)
            pts_value = float(pts_time)
        except (TypeError, ValueError):
            errors.append(_error("canonical_invalid_row", "canonical row has invalid identity/time", row=row_number))
            continue
        if not video_id or kf_value < 0 or frame_value < 0 or not math.isfinite(pts_value) or pts_value < 0:
            errors.append(_error("canonical_invalid_value", "canonical row has empty/negative/non-finite value", row=row_number))
            continue
        key = (video_id, kf_value)
        previous = result.get(key)
        current = (frame_value, pts_value)
        if previous is not None and previous != current:
            errors.append(_error("canonical_duplicate_identity", "canonical identity maps to multiple frames", row=row_number, key=key))
            continue
        result[key] = current
    return result, errors


def _validate_pack(
    *,
    modality: str,
    pack: str,
    embedding_path: Path | None,
    metadata_path: Path | None,
    canonical: dict[tuple[str, int], tuple[int, float]],
    read_parquet: Callable[[Path], Any],
    load_npy: Callable[[Path], Any],
    expected_dim: int | None,
    time_tolerance: float,
    require_embedding: bool = True,
) -> dict[str, Any]:
    result = _empty_pack(pack, modality)
    result["embedding"] = str(embedding_path) if embedding_path else None
    result["metadata"] = str(metadata_path) if metadata_path else None

    if require_embedding and embedding_path is None:
        result["errors"].append(_error("missing_embedding", f"missing {modality} embedding pack for {pack}"))
    if metadata_path is None:
        result["errors"].append(_error("missing_metadata", f"missing {modality} metadata pack for {pack}"))
    if result["errors"]:
        return result

    embeddings = None
    if require_embedding:
        try:
            embeddings = load_npy(embedding_path)
        except Exception as exc:
            result["errors"].append(_error("embedding_read_error", str(exc)))
            return result
    try:
        metadata = read_parquet(metadata_path)
    except Exception as exc:
        result["errors"].append(_error("metadata_read_error", str(exc)))
        return result

    shape = tuple(getattr(embeddings, "shape", ())) if embeddings is not None else ()
    result["embedding_rows"] = int(shape[0]) if shape else 0
    result["embedding_dim"] = int(shape[1]) if len(shape) >= 2 else None
    result["metadata_rows"] = int(len(metadata))
    if require_embedding and len(shape) != 2:
        result["errors"].append(_error("embedding_not_2d", "embedding array must be two-dimensional", shape=shape))
    if require_embedding and result["embedding_dim"] is None:
        result["errors"].append(_error("embedding_missing_dim", "embedding array has no feature dimension"))
    elif require_embedding and expected_dim is not None and result["embedding_dim"] != expected_dim:
        result["errors"].append(_error("embedding_dim_mismatch", "unexpected embedding dimension", expected=expected_dim, actual=result["embedding_dim"]))
    if require_embedding and result["embedding_rows"] != result["metadata_rows"]:
        result["errors"].append(_error("row_count_mismatch", "embedding rows do not match metadata rows", embeddings=result["embedding_rows"], metadata=result["metadata_rows"]))

    columns = _normalise_columns(metadata.columns)
    video_column = _pick_column(columns, "video_id", "vid")
    if video_column is None:
        result["errors"].append(_error("missing_video_id", "metadata requires video_id or vid"))
    if "kf_n" not in columns:
        result["errors"].append(_error("missing_kf_n", "metadata requires kf_n"))
    if modality == "asr":
        missing = sorted({"frame_idx", "start", "end"} - columns)
        if not ({"chunk", "text", "transcript"} & columns):
            missing.append("chunk/text/transcript")
        if missing:
            result["errors"].append(_error("missing_asr_columns", "ASR metadata schema is incomplete", columns=missing))
        time_columns = ("start", "end")
        text_column = _pick_column(columns, "chunk", "text", "transcript")
    else:
        missing = sorted({"pts_time", "ocr_text"} - columns)
        if missing:
            result["errors"].append(_error("missing_ocr_columns", "OCR metadata schema is incomplete", columns=missing))
        time_columns = ("pts_time",)
        text_column = "ocr_text" if "ocr_text" in columns else None
    schema_error_codes = {
        "missing_video_id",
        "missing_kf_n",
        "missing_asr_columns",
        "missing_ocr_columns",
    }
    if (
        video_column is None
        or text_column is None
        or any(item.get("code") in schema_error_codes for item in result["errors"])
    ):
        return result

    result["video_ids"] = sorted({
        _as_text(value).strip()
        for value in _series_values(metadata, video_column)
        if _as_text(value).strip()
    })
    result["video_count"] = len(result["video_ids"])
    result["text_quality"] = _text_quality(_series_values(metadata, text_column))
    quality = result["text_quality"]
    if quality["empty_rows"]:
        result["errors"].append(_error("empty_text", "metadata contains empty evidence text", count=quality["empty_rows"]))
    if quality["replacement_chars"]:
        result["errors"].append(_error("text_replacement_char", "text contains Unicode replacement characters", count=quality["replacement_chars"]))
    if quality["nfkc_changes"]:
        result["warnings"].append(_error("text_not_nfkc", "text contains values that change under NFKC", count=quality["nfkc_changes"]))
    if quality["mojibake_token_count"]:
        result["warnings"].append(_error("text_mojibake_suspected", "text contains suspected mojibake markers", count=quality["mojibake_token_count"]))

    keys = _key_set(metadata, video_column, "kf_n")
    missing_keys = [key for key in keys if key not in canonical]
    result["canonical_missing_rows"] = len(missing_keys)
    if missing_keys:
        result["errors"].append(_error("canonical_mapping_missing", "metadata identities are absent from canonical keyframe map", count=len(missing_keys), examples=sorted(missing_keys)[:5]))

    kf_values = _numeric_values(metadata, "kf_n")
    if np.any(~np.isfinite(kf_values)) or np.any(kf_values < 0) or np.any(kf_values != np.floor(kf_values)):
        result["errors"].append(_error("invalid_kf_n", "kf_n must be finite non-negative integers"))

    for column in time_columns:
        values = _numeric_values(metadata, column)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            result["errors"].append(_error("invalid_time", f"{column} must be finite and non-negative"))
    if modality == "asr":
        starts = _numeric_values(metadata, "start")
        ends = _numeric_values(metadata, "end")
        if np.any(ends < starts):
            result["errors"].append(_error("invalid_asr_interval", "ASR end time precedes start time"))

    frame_values = _numeric_values(metadata, "frame_idx") if "frame_idx" in columns else None
    mismatch = 0
    for video, kf, frame_idx, pts_time in zip(
        _series_values(metadata, video_column),
        _series_values(metadata, "kf_n"),
        _series_values(metadata, "frame_idx") if "frame_idx" in columns else [None] * len(metadata),
        _series_values(metadata, "pts_time") if "pts_time" in columns else [None] * len(metadata),
    ):
        try:
            key = (str(video), int(kf))
            canonical_frame, canonical_pts = canonical[key]
        except (KeyError, TypeError, ValueError):
            continue
        if frame_idx is not None:
            try:
                if int(frame_idx) != canonical_frame:
                    mismatch += 1
                    continue
            except (TypeError, ValueError):
                mismatch += 1
                continue
        # OCR timestamps are keyframe timestamps and must agree. ASR chunks
        # use their own start/end interval, so their midpoint is not compared
        # to the keyframe timestamp.
        if modality == "ocr" and pts_time is not None:
            try:
                if abs(float(pts_time) - canonical_pts) > time_tolerance:
                    mismatch += 1
            except (TypeError, ValueError):
                mismatch += 1
    result["mapping_mismatch_rows"] = mismatch
    if mismatch:
        result["errors"].append(_error("canonical_mapping_mismatch", "metadata frame/time disagrees with canonical map", count=mismatch))
    return result


def _resolve_global_artifact(value: Any, *, root: Path, manifest_path: Path, fallback: Path) -> Path:
    """Resolve repo-relative paths emitted by the ASR merge manifest."""
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


def _discover_global_asr(root: Path) -> dict[str, Any] | None:
    """Discover merged ASR without treating a partial directory as absent."""
    directory = root / GLOBAL_ASR_RELATIVE_DIR
    manifest_path = directory / GLOBAL_ASR_MANIFEST_NAME
    retrieval_fallback = directory / GLOBAL_ASR_RETRIEVAL_NAME
    embeddings_fallback = directory / GLOBAL_ASR_EMBEDDINGS_NAME
    if not (manifest_path.exists() or retrieval_fallback.exists() or embeddings_fallback.exists()):
        return None
    manifest: dict[str, Any] = {}
    manifest_error: str | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            manifest_error = str(exc)
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "retrieval_path": _resolve_global_artifact(
            artifacts.get("retrieval"), root=root,
            manifest_path=manifest_path, fallback=retrieval_fallback,
        ),
        "embeddings_path": _resolve_global_artifact(
            artifacts.get("embeddings"), root=root,
            manifest_path=manifest_path, fallback=embeddings_fallback,
        ),
        "manifest": manifest,
        "manifest_error": manifest_error,
    }


def _discover_global_ocr(root: Path) -> dict[str, Any] | None:
    """Discover sampled global OCR without treating a partial source absent."""
    directory = root / GLOBAL_OCR_RELATIVE_DIR
    manifest_path = directory / GLOBAL_OCR_MANIFEST_NAME
    retrieval_fallback = directory / GLOBAL_OCR_RETRIEVAL_NAME
    embeddings_fallback = directory / GLOBAL_OCR_EMBEDDINGS_NAME
    if not (manifest_path.exists() or retrieval_fallback.exists() or embeddings_fallback.exists()):
        return None
    manifest: dict[str, Any] = {}
    manifest_error: str | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            manifest_error = str(exc)
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "retrieval_path": directory / GLOBAL_OCR_RETRIEVAL_NAME,
        "embeddings_path": directory / GLOBAL_OCR_EMBEDDINGS_NAME,
        "manifest": manifest,
        "manifest_error": manifest_error,
    }


def _validate_global_asr(
    source: dict[str, Any],
    *,
    expected_packs: tuple[str, ...],
    canonical: dict[tuple[str, int], tuple[int, float]],
    canonical_video_ids: set[str],
    read_parquet: Callable[[Path], Any],
    load_npy: Callable[[Path], Any],
    expected_dim: int | None,
    time_tolerance: float,
    require_embedding: bool,
) -> dict[str, Any]:
    """Validate the merged ASR artifact as one shared source.

    This intentionally does not route through ``_validate_pack``: merged ASR
    has one metadata/embedding pair and represents no-speech videos in its
    manifest rather than as empty transcript rows.
    """
    manifest_path = Path(source["manifest_path"])
    retrieval_path = Path(source["retrieval_path"])
    embeddings_path = Path(source["embeddings_path"])
    result: dict[str, Any] = {
        "source": "global_merged_v2",
        "manifest": str(manifest_path),
        "metadata": str(retrieval_path),
        "embedding": str(embeddings_path) if require_embedding else None,
        "packs": [],
        "observed_packs": [],
        "missing_packs": [],
        "observed_video_ids": [],
        "covered_video_ids": [],
        "missing_video_ids": [],
        "no_speech_videos": [],
        "errors": [],
        "warnings": [],
    }
    if source.get("manifest_error"):
        result["errors"].append(_error("global_manifest_read_error", source["manifest_error"]))
        return result
    manifest = source.get("manifest", {})
    if not manifest_path.is_file():
        result["errors"].append(_error("global_manifest_missing", f"missing merged ASR manifest: {manifest_path}"))
        return result
    if manifest.get("status") != "ready":
        result["errors"].append(_error("global_manifest_not_ready", "merged ASR manifest is not ready", status=manifest.get("status")))
    manifest_packs = {
        str(value).strip().lower()
        for value in manifest.get("scope", {}).get("packs", [])
    }
    result["observed_packs"] = sorted(set(expected_packs) & manifest_packs)
    result["missing_packs"] = sorted(set(expected_packs) - manifest_packs)
    if result["missing_packs"]:
        result["errors"].append(_error(
            "global_asr_pack_coverage_incomplete",
            "merged ASR manifest does not cover all expected packs",
            missing=result["missing_packs"],
        ))
    if not retrieval_path.is_file():
        result["errors"].append(_error("global_metadata_missing", f"missing merged ASR metadata: {retrieval_path}"))
        return result
    if require_embedding and not embeddings_path.is_file():
        result["errors"].append(_error("global_embedding_missing", f"missing merged ASR embeddings: {embeddings_path}"))
        return result
    try:
        metadata = read_parquet(retrieval_path).reset_index(drop=True)
    except Exception as exc:
        result["errors"].append(_error("global_metadata_read_error", str(exc)))
        return result
    missing_columns = sorted({
        "video_id", "text", "start", "end", "kf_n", "frame_idx",
        "pts_time", "embedding_row", "source_pack", "source_provenance",
    } - set(metadata.columns))
    if missing_columns:
        result["errors"].append(_error("global_metadata_schema_error", "merged ASR schema is incomplete", columns=missing_columns))
        return result

    embeddings = None
    if require_embedding:
        try:
            embeddings = load_npy(embeddings_path)
        except Exception as exc:
            result["errors"].append(_error("global_embedding_read_error", str(exc)))
            return result
        shape = tuple(getattr(embeddings, "shape", ()))
        result["embedding_rows"] = int(shape[0]) if shape else 0
        result["embedding_dim"] = int(shape[1]) if len(shape) >= 2 else None
        if len(shape) != 2:
            result["errors"].append(_error("global_embedding_not_2d", "merged ASR embeddings must be two-dimensional", shape=shape))
        if expected_dim is not None and len(shape) >= 2 and shape[1] != expected_dim:
            result["errors"].append(_error("global_embedding_dim_mismatch", "unexpected merged ASR embedding dimension", expected=expected_dim, actual=shape[1]))
        if len(shape) >= 1 and shape[0] != len(metadata):
            result["errors"].append(_error("global_row_count_mismatch", "merged ASR embeddings do not match metadata rows", embeddings=shape[0], metadata=len(metadata)))

    metadata = metadata.copy()
    metadata["video_id"] = metadata["video_id"].astype(str).str.strip().str.upper()
    metadata["source_pack"] = metadata["source_pack"].astype(str).str.strip().str.lower()
    metadata["text"] = metadata["text"].fillna("").astype(str).str.strip()
    quality = _text_quality(_series_values(metadata, "text"))
    result["text_quality"] = quality
    if quality["empty_rows"]:
        result["errors"].append(_error("global_empty_text", "merged ASR contains empty transcript text", count=quality["empty_rows"]))
    if quality["replacement_chars"]:
        result["errors"].append(_error("global_text_replacement_char", "merged ASR contains Unicode replacement characters", count=quality["replacement_chars"]))
    if quality["mojibake_token_count"]:
        result["warnings"].append(_error("global_text_mojibake_suspected", "merged ASR contains suspected mojibake markers", count=quality["mojibake_token_count"]))

    row_ids = _numeric_values(metadata, "embedding_row")
    if np.any(~np.isfinite(row_ids)) or np.any(row_ids != np.floor(row_ids)) or not np.array_equal(row_ids.astype(np.int64), np.arange(len(metadata), dtype=np.int64)):
        result["errors"].append(_error("global_embedding_row_mismatch", "embedding_row must be a contiguous aligned range"))
    packs_in_rows = set(metadata["source_pack"])
    unexpected_packs = sorted(packs_in_rows - set(expected_packs))
    if unexpected_packs:
        result["errors"].append(_error("global_unexpected_pack", "merged ASR contains packs outside the requested scope", packs=unexpected_packs))

    missing_keys: set[tuple[str, int]] = set()
    mapping_mismatch = 0
    frame_values = _numeric_values(metadata, "frame_idx")
    kf_values = _numeric_values(metadata, "kf_n")
    starts = _numeric_values(metadata, "start")
    ends = _numeric_values(metadata, "end")
    pts_values = _numeric_values(metadata, "pts_time")
    if np.any(~np.isfinite(kf_values)) or np.any(kf_values < 0) or np.any(kf_values != np.floor(kf_values)):
        result["errors"].append(_error("global_invalid_kf_n", "merged ASR kf_n must be finite non-negative integers"))
    if np.any(~np.isfinite(frame_values)) or np.any(frame_values < 0) or np.any(frame_values != np.floor(frame_values)):
        result["errors"].append(_error("global_invalid_frame_idx", "merged ASR frame_idx must be finite non-negative integers"))
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)) or np.any(starts < 0) or np.any(ends < starts):
        result["errors"].append(_error("global_invalid_asr_interval", "merged ASR start/end timestamps are invalid"))

    for video, kf, frame_idx, pts_time in zip(
        _series_values(metadata, "video_id"),
        _series_values(metadata, "kf_n"),
        _series_values(metadata, "frame_idx"),
        _series_values(metadata, "pts_time"),
    ):
        try:
            key = (str(video), int(kf))
            frame_value = int(frame_idx)
            pts_value = float(pts_time)
        except (TypeError, ValueError, OverflowError):
            continue
        expected = canonical.get(key)
        if expected is None:
            missing_keys.add(key)
            continue
        if frame_value != expected[0] or not math.isfinite(pts_value) or abs(pts_value - expected[1]) > time_tolerance:
            mapping_mismatch += 1
    result["canonical_missing_rows"] = len(missing_keys)
    result["mapping_mismatch_rows"] = mapping_mismatch
    if missing_keys:
        result["errors"].append(_error("global_canonical_mapping_missing", "merged ASR identities are absent from canonical map", count=len(missing_keys), examples=sorted(missing_keys)[:5]))
    if mapping_mismatch:
        result["errors"].append(_error("global_canonical_mapping_mismatch", "merged ASR frame/time disagrees with canonical map", count=mapping_mismatch))

    manifest_pack_info = {
        str(key).strip().lower(): value
        for key, value in manifest.get("packs", {}).items()
    }
    all_no_speech: set[str] = set()
    for pack in expected_packs:
        item = _empty_pack(pack, "asr")
        item.update({
            "source": "global_merged_v2",
            "embedding": str(embeddings_path) if require_embedding else None,
            "metadata": str(retrieval_path),
        })
        subset = metadata.loc[metadata["source_pack"] == pack].copy()
        observed_videos = set(subset["video_id"])
        info = manifest_pack_info.get(pack, {})
        no_speech = {
            str(value).strip().upper()
            for value in info.get("no_speech_videos", [])
            if str(value).strip()
        } if isinstance(info, dict) else set()
        all_no_speech.update(no_speech)
        item["no_speech_videos"] = sorted(no_speech)
        item["video_ids"] = sorted(observed_videos)
        item["video_count"] = len(observed_videos)
        item["metadata_rows"] = int(len(subset))
        item["embedding_rows"] = int(len(subset)) if require_embedding else 0
        item["covered_video_count"] = len(observed_videos | no_speech)
        expected_videos = {
            video_id for video_id in canonical_video_ids
            if _pack_from_video_id(video_id) == pack
        }
        if observed_videos & no_speech:
            item["errors"].append(_error("no_speech_overlap", "a video has both transcript rows and no-speech status", examples=sorted(observed_videos & no_speech)[:5]))
        covered = observed_videos | no_speech
        if covered != expected_videos:
            item["errors"].append(_error(
                "video_coverage_incomplete",
                "global ASR observed/no-speech coverage does not match canonical videos",
                missing=sorted(expected_videos - covered)[:10],
                extra=sorted(covered - expected_videos)[:10],
            ))
        if isinstance(info, dict) and info.get("rows") is not None and int(info["rows"]) != len(subset):
            item["errors"].append(_error("manifest_row_count_mismatch", "manifest pack row count disagrees with retrieval metadata", manifest=int(info["rows"]), actual=len(subset)))
        result["packs"].append(item)

    result["no_speech_videos"] = sorted(all_no_speech)
    result["observed_video_ids"] = sorted(set(metadata["video_id"]) & canonical_video_ids)
    result["covered_video_ids"] = sorted(set(result["observed_video_ids"]) | (all_no_speech & canonical_video_ids))
    result["missing_video_ids"] = sorted(canonical_video_ids - set(result["covered_video_ids"]))
    if result["missing_video_ids"]:
        result["errors"].append(_error("global_video_coverage_incomplete", "merged ASR does not cover all canonical videos", count=len(result["missing_video_ids"]), examples=result["missing_video_ids"][:10]))
    for item in result["packs"]:
        result["errors"].extend({**entry, "pack": item["pack"], "modality": "asr"} for entry in item["errors"])
    return result


def run_modality_index_preflight(
    index_dir: str | Path,
    *,
    expected_packs: Iterable[str] | None = None,
    active_modalities: Iterable[str] | None = None,
    canonical_name: str = "global_keyframes.parquet",
    read_parquet: Callable[[Path], Any] | None = None,
    load_npy: Callable[[Path], Any] | None = None,
    expected_dim: int | None = EMBEDDING_DIM,
    time_tolerance: float = 1e-2,
    require_embeddings: bool = True,
) -> dict[str, Any]:
    """Return a deterministic preflight report; never mutates index files."""
    root = Path(index_dir)
    read_parquet = read_parquet or _default_read_parquet
    load_npy = load_npy or _read_npy
    packs = normalise_expected_packs(expected_packs)
    requested_modalities = ("asr", "ocr") if active_modalities is None else active_modalities
    modalities = tuple(sorted({
        str(item).strip().lower()
        for item in requested_modalities
        if str(item).strip().lower() in {"asr", "ocr"}
    }))
    if not modalities:
        raise ValueError("active_modalities must contain at least one of: asr, ocr")
    full_corpus = packs == DEFAULT_EXPECTED_PACKS
    report: dict[str, Any] = {
        "protocol": "offline modality index preflight v2",
        "index_dir": str(root),
        "expected_embedding_dim": expected_dim,
        "embeddings_required": bool(require_embeddings),
        "expected_packs": list(packs),
        "active_modalities": list(modalities),
        "scope": {
            "name": "full_corpus" if full_corpus else "custom",
            "is_full_corpus": full_corpus,
            "active_packs": list(packs),
        },
        "errors": [],
        "warnings": [],
        "coverage": {
            "expected": list(packs),
            "asr_observed": [],
            "asr_missing": [],
            "ocr_observed": [],
            "ocr_missing": [],
            "canonical_expected_video_count": 0,
            "asr_observed_video_count": 0,
            "asr_missing_videos": [],
            "asr_video_coverage_ratio": 0.0,
            "ocr_observed_video_count": 0,
            "ocr_missing_videos": [],
            "ocr_video_coverage_ratio": 0.0,
        },
        "canonical": {"path": str(root / canonical_name), "rows": 0, "videos": 0},
        "sources": {"asr": "legacy_pack", "ocr": "legacy_pack"},
        "asr": [],
        "ocr": [],
        "passed": False,
    }
    if not root.exists():
        report["errors"].append(_error("missing_index_dir", f"index directory does not exist: {root}"))
        return report

    canonical_path = root / canonical_name
    if not canonical_path.exists():
        report["errors"].append(_error("missing_canonical_map", f"canonical map does not exist: {canonical_path}"))
        canonical: dict[tuple[str, int], tuple[int, float]] = {}
    else:
        try:
            canonical_frame = read_parquet(canonical_path)
            report["canonical"]["rows"] = int(len(canonical_frame))
            canonical = {}
            canonical, canonical_errors = _canonical_index(canonical_frame)
            report["errors"].extend(canonical_errors)
            report["canonical"]["videos"] = len({key[0] for key in canonical})
        except Exception as exc:
            report["errors"].append(_error("canonical_read_error", str(exc)))
            canonical = {}

    canonical_video_ids = sorted({key[0] for key in canonical})
    scoped_video_ids = sorted({
        video_id
        for video_id in canonical_video_ids
        if _pack_from_video_id(video_id) in set(packs)
    })
    unresolved_video_ids = sorted(
        video_id for video_id in canonical_video_ids if _pack_from_video_id(video_id) is None
    )
    report["coverage"]["canonical_expected_video_count"] = len(scoped_video_ids)
    if full_corpus and unresolved_video_ids:
        report["errors"].append(_error(
            "canonical_video_pack_unresolved",
            "canonical video identities do not have a K/L pack prefix",
            count=len(unresolved_video_ids),
            examples=unresolved_video_ids[:5],
        ))

    # A ready merged ASR source is authoritative for ASR.  If its directory
    # exists but is incomplete, keep it as the selected source and report the
    # concrete global failure; do not fall back to a mixture of old shards.
    global_asr = _discover_global_asr(root) if "asr" in modalities else None
    global_asr_selected = global_asr is not None
    if global_asr_selected:
        global_report = _validate_global_asr(
            global_asr,
            expected_packs=packs,
            canonical=canonical,
            canonical_video_ids=set(scoped_video_ids),
            read_parquet=read_parquet,
            load_npy=load_npy,
            expected_dim=expected_dim,
            time_tolerance=time_tolerance,
            require_embedding=require_embeddings,
        )
        report["sources"]["asr"] = global_report["source"]
        report["asr"] = global_report["packs"]
        report["errors"].extend(global_report["errors"])
        report["warnings"].extend(global_report["warnings"])
        report["coverage"]["asr_observed"] = global_report["observed_packs"]
        report["coverage"]["asr_missing"] = global_report["missing_packs"]
        report["coverage"]["asr_observed_video_count"] = len(global_report["covered_video_ids"])
        report["coverage"]["asr_missing_videos"] = global_report["missing_video_ids"]
        report["coverage"]["asr_video_coverage_ratio"] = (
            len(global_report["covered_video_ids"]) / len(scoped_video_ids)
            if scoped_video_ids else 0.0
        )

    # A ready sampled global OCR source is authoritative for OCR.  Its
    # coverage contract is video-level: every OCR row is canonical-mapped,
    # while videos with no readable text are listed explicitly in provenance.
    global_ocr = _discover_global_ocr(root) if "ocr" in modalities else None
    global_ocr_selected = global_ocr is not None
    if global_ocr_selected:
        report["sources"]["ocr"] = "global_merged_v2"
        try:
            registry = ModalityIndexRegistry(root, canonical_name=canonical_name)
            _, global_meta, global_info = registry.load_ocr(
                expected_packs=packs,
                require_embeddings=require_embeddings,
                strict=True,
            )
            observed_videos = set(global_meta["video_id"].astype(str))
            manifest = global_ocr.get("manifest", {})
            no_text = {
                str(value).strip().upper()
                for value in manifest.get("provenance", {}).get("no_text_videos", [])
                if str(value).strip()
            }
            no_text &= set(scoped_video_ids)
            covered_videos = observed_videos | no_text
            observed_packs = sorted({str(value).strip().lower() for value in global_meta["source_pack"]})
            report["coverage"]["ocr_observed"] = sorted(set(packs) & set(observed_packs))
            report["coverage"]["ocr_missing"] = sorted(set(packs) - set(report["coverage"]["ocr_observed"]))
            report["coverage"]["ocr_observed_video_count"] = len(covered_videos)
            report["coverage"]["ocr_missing_videos"] = sorted(set(scoped_video_ids) - covered_videos)
            report["coverage"]["ocr_video_coverage_ratio"] = (
                len(covered_videos) / len(scoped_video_ids) if scoped_video_ids else 0.0
            )
            report["ocr"] = [
                {
                    "pack": pack,
                    "modality": "ocr",
                    "source": "global_merged_v2",
                    "metadata_rows": int((global_meta["source_pack"] == pack).sum()),
                    "video_count": int(global_meta.loc[global_meta["source_pack"] == pack, "video_id"].nunique()),
                    "no_text_videos": sorted(value for value in no_text if value[:3].lower() == pack),
                    "errors": [],
                    "warnings": [],
                }
                for pack in packs
            ]
            if report["coverage"]["ocr_missing_videos"]:
                report["errors"].append(_error(
                    "global_ocr_video_coverage_incomplete",
                    "global OCR does not cover all canonical videos",
                    count=len(report["coverage"]["ocr_missing_videos"]),
                    examples=report["coverage"]["ocr_missing_videos"][:10],
                ))
            if report["coverage"]["ocr_missing"]:
                report["errors"].append(_error(
                    "global_ocr_pack_coverage_incomplete",
                    "global OCR does not cover all expected packs",
                    missing=report["coverage"]["ocr_missing"],
                ))
        except Exception as exc:
            report["errors"].append(_error(
                "global_ocr_validation_failed",
                f"global OCR source is not usable: {exc}",
            ))

    asr_meta = {m.group(1).lower(): path for path in root.iterdir() if (m := ASR_PACK_RE.match(path.name))}
    asr_emb = {m.group(1).lower(): path for path in root.iterdir() if (m := ASR_EMBED_RE.match(path.name))}
    ocr_meta = {m.group(1).lower(): path for path in root.iterdir() if (m := OCR_PACK_RE.match(path.name))}
    ocr_emb = {m.group(1).lower(): path for path in root.iterdir() if (m := OCR_EMBED_RE.match(path.name))}

    # Coverage is reported for the active scope only. Files outside the scope
    # are intentionally ignored; they must not make a subset benchmark look
    # complete or incomplete. Missing files inside the scope remain errors.
    asr_available = set(asr_meta) & set(asr_emb) if require_embeddings else set(asr_meta)
    ocr_available = set(ocr_meta) & set(ocr_emb) if require_embeddings else set(ocr_meta)
    if "asr" in modalities and not global_asr_selected:
        report["coverage"]["asr_observed"] = sorted(set(packs) & asr_available)
        report["coverage"]["asr_missing"] = sorted(set(packs) - set(report["coverage"]["asr_observed"]))
    elif "asr" not in modalities:
        report["coverage"]["asr_observed"] = []
        report["coverage"]["asr_missing"] = []
    if "ocr" in modalities and not global_ocr_selected:
        report["coverage"]["ocr_observed"] = sorted(set(packs) & ocr_available)
        report["coverage"]["ocr_missing"] = sorted(set(packs) - set(report["coverage"]["ocr_observed"]))
    elif "ocr" not in modalities:
        report["coverage"]["ocr_observed"] = []
        report["coverage"]["ocr_missing"] = []
    if report["coverage"]["asr_missing"] and not global_asr_selected:
        report["errors"].append(_error("asr_pack_coverage_incomplete", "ASR does not cover all expected packs", missing=report["coverage"]["asr_missing"]))
    if report["coverage"]["ocr_missing"] and not global_ocr_selected:
        report["errors"].append(_error("ocr_pack_coverage_incomplete", "OCR does not cover all expected packs", missing=report["coverage"]["ocr_missing"]))

    for pack in packs:
        if "asr" in modalities and not global_asr_selected:
            item = _validate_pack(
                modality="asr", pack=pack, embedding_path=asr_emb.get(pack), metadata_path=asr_meta.get(pack),
                canonical=canonical, read_parquet=read_parquet, load_npy=load_npy,
                expected_dim=expected_dim, time_tolerance=time_tolerance,
                require_embedding=require_embeddings,
            )
            report["asr"].append(item)
            report["errors"].extend({**entry, "pack": pack, "modality": "asr"} for entry in item["errors"])
            report["warnings"].extend({**entry, "pack": pack, "modality": "asr"} for entry in item["warnings"])
        if "ocr" in modalities and not global_ocr_selected:
            item = _validate_pack(
                modality="ocr", pack=pack, embedding_path=ocr_emb.get(pack), metadata_path=ocr_meta.get(pack),
                canonical=canonical, read_parquet=read_parquet, load_npy=load_npy,
                expected_dim=expected_dim, time_tolerance=time_tolerance,
                require_embedding=require_embeddings,
            )
            report["ocr"].append(item)
            report["errors"].extend({**entry, "pack": pack, "modality": "ocr"} for entry in item["errors"])
            report["warnings"].extend({**entry, "pack": pack, "modality": "ocr"} for entry in item["warnings"])

    # Pack files are not enough to establish readiness: a pack can contain a
    # small diagnostic subset of videos while still having the expected file
    # name.  Compare the union of metadata video identities with the
    # canonical catalog scoped to the requested packs.
    for modality in modalities:
        if modality == "asr" and global_asr_selected:
            continue
        if modality == "ocr" and global_ocr_selected:
            continue
        observed_video_ids = {
            video_id
            for item in report[modality]
            for video_id in item.get("video_ids", [])
        }
        missing_video_ids = sorted(set(scoped_video_ids) - observed_video_ids)
        coverage_key = f"{modality}_observed_video_count"
        missing_key = f"{modality}_missing_videos"
        ratio_key = f"{modality}_video_coverage_ratio"
        report["coverage"][coverage_key] = len(observed_video_ids & set(scoped_video_ids))
        report["coverage"][missing_key] = missing_video_ids
        report["coverage"][ratio_key] = (
            len(observed_video_ids & set(scoped_video_ids)) / len(scoped_video_ids)
            if scoped_video_ids else 0.0
        )
        if missing_video_ids:
            report["errors"].append(_error(
                f"{modality}_video_coverage_incomplete",
                f"{modality.upper()} metadata does not cover all canonical videos in scope",
                count=len(missing_video_ids),
                examples=missing_video_ids[:10],
            ))

    report["passed"] = not report["errors"]
    return report


def require_modality_index(report: dict[str, Any]) -> dict[str, Any]:
    """Raise a concise error when a preflight report is not safe to use."""
    if not report.get("passed", False):
        errors = report.get("errors", [])
        summary = "; ".join(
            f"{item.get('code', 'error')}: {item.get('message', '')}" for item in errors[:6]
        ) or "unknown preflight failure"
        if len(errors) > 6:
            summary += f"; ... ({len(errors)} errors total)"
        raise ModalityIndexPreflightError(summary)
    return report


# Short aliases make the gate easy to discover for callers and tests.
preflight_modality_index = run_modality_index_preflight
assert_modality_index_ready = require_modality_index


__all__ = [
    "DEFAULT_EXPECTED_PACKS",
    "BENCHMARK_K_PACKS",
    "EMBEDDING_DIM",
    "ModalityIndexPreflightError",
    "normalise_expected_packs",
    "run_modality_index_preflight",
    "preflight_modality_index",
    "require_modality_index",
    "assert_modality_index_ready",
]
