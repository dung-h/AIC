"""Build the sampled, global OCR v2 retrieval artifact.

This module is an offline adapter only.  It consumes the three artifacts
written by ``src.eval.materialize_local_ocr_corpus_v1`` and never runs OCR,
loads a model, calls an API, or extracts frames.

The input parquet may contain a sampled subset of canonical keyframes.  Full
corpus coverage is a video-level requirement: every canonical video must have
at least one OCR row or an explicit no-text video record.  The output is a
versioned directory containing exactly ``retrieval.parquet``,
``embeddings.npy``, and ``manifest.json``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "hcmai.ocr_global_v2"
EMBEDDING_DIM = 1024
EXPECTED_VIDEO_COUNT = 1478
EXPECTED_PACKS = tuple(
    [f"K{i:02d}" for i in range(1, 21)]
    + [f"L{i:02d}" for i in range(21, 31)]
)
INPUT_COLUMNS = ("video_id", "kf_n", "frame_idx", "pts_time", "ocr_text")
OUTPUT_COLUMNS = (
    "embedding_row",
    "video_id",
    "kf_n",
    "frame_idx",
    "pts_time",
    "ocr_text",
    "source_pack",
)
_VIDEO_RE = re.compile(r"^([KL]\d{2})_V\d+$", re.IGNORECASE)
_NO_TEXT_STATUSES = {"no_text", "no-text", "no text", "no_ocr", "no-ocr"}


class OCRGlobalV2Error(RuntimeError):
    """Raised when the sampled OCR source cannot be promoted safely."""


@dataclass(frozen=True)
class ValidatedArtifacts:
    """In-memory, canonicalized source artifacts after all gates pass."""

    metadata: pd.DataFrame
    embeddings: np.ndarray
    manifest: dict[str, Any]
    no_text_videos: tuple[str, ...]
    sample_interval_seconds: float
    coverage: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_packs(packs: Sequence[str] | None) -> tuple[str, ...]:
    values = tuple(str(pack).strip().upper() for pack in (packs or EXPECTED_PACKS))
    if not values or len(set(values)) != len(values):
        raise OCRGlobalV2Error("expected_packs must be non-empty and unique")
    unknown = sorted(set(values) - set(EXPECTED_PACKS))
    if unknown:
        raise OCRGlobalV2Error(f"unknown expected pack(s): {unknown}")
    return values


def pack_for_video(video_id: str, *, expected_packs: Sequence[str] = EXPECTED_PACKS) -> str:
    match = _VIDEO_RE.fullmatch(str(video_id).strip())
    pack = match.group(1).upper() if match else ""
    if pack not in set(expected_packs):
        raise OCRGlobalV2Error(f"video_id has unknown pack prefix: {video_id!r}")
    return pack


def _read_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.is_file():
        raise OCRGlobalV2Error(f"metadata parquet does not exist: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise OCRGlobalV2Error(f"cannot read metadata parquet {path}: {exc}") from exc


def _integer_series(table: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | (values != np.floor(values))
    if invalid.any():
        raise OCRGlobalV2Error(f"{context} {column} must contain finite integers")
    return values.astype(np.int64)


def _float_series(table: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    values = pd.to_numeric(table[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise OCRGlobalV2Error(f"{context} {column} must contain finite numbers")
    return values.astype(float)


def load_canonical(
    source: str | Path | pd.DataFrame,
    *,
    expected_packs: Sequence[str] = EXPECTED_PACKS,
    expected_video_count: int | None = EXPECTED_VIDEO_COUNT,
) -> pd.DataFrame:
    """Load and prove the full canonical K/L video scope.

    ``expected_packs`` and ``expected_video_count`` are overridable for small
    deterministic unit fixtures.  Production defaults are the required 30
    packs and 1,478 videos.
    """
    packs = _normalize_packs(expected_packs)
    table = _read_table(source)
    required = {"video_id", "kf_n", "frame_idx", "pts_time"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise OCRGlobalV2Error(f"canonical index missing columns: {missing}")
    table = table.copy()
    table["video_id"] = table["video_id"].astype(str).str.strip().str.upper()
    if table["video_id"].eq("").any() or table["video_id"].eq("NAN").any():
        raise OCRGlobalV2Error("canonical video_id contains empty values")
    table["kf_n"] = _integer_series(table, "kf_n", context="canonical index")
    table["frame_idx"] = _integer_series(table, "frame_idx", context="canonical index")
    table["pts_time"] = _float_series(table, "pts_time", context="canonical index")
    if table[["video_id", "kf_n", "frame_idx", "pts_time"]].isna().any().any():
        raise OCRGlobalV2Error("canonical index contains null identity or timestamp")
    if (table["kf_n"] < 0).any() or (table["frame_idx"] < 0).any() or (table["pts_time"] < 0).any():
        raise OCRGlobalV2Error("canonical kf_n, frame_idx, and pts_time must be non-negative")
    if table.duplicated(["video_id", "kf_n"]).any():
        raise OCRGlobalV2Error("canonical index contains duplicate (video_id, kf_n)")
    table["source_pack"] = [pack_for_video(value, expected_packs=packs) for value in table["video_id"]]
    actual_packs = set(table["source_pack"])
    missing_packs = sorted(set(packs) - actual_packs)
    extra_packs = sorted(actual_packs - set(packs))
    if missing_packs or extra_packs:
        raise OCRGlobalV2Error(
            f"canonical pack coverage mismatch: missing={missing_packs}, extra={extra_packs}"
        )
    video_count = int(table["video_id"].nunique())
    if expected_video_count is not None and video_count != int(expected_video_count):
        raise OCRGlobalV2Error(
            f"canonical video count mismatch: expected {expected_video_count}, got {video_count}"
        )
    if any(int(table.loc[table["source_pack"] == pack, "video_id"].nunique()) == 0 for pack in packs):
        raise OCRGlobalV2Error("canonical scope has a pack with no videos")
    return table[["video_id", "source_pack", "kf_n", "frame_idx", "pts_time"]].sort_values(
        ["source_pack", "video_id", "pts_time", "kf_n"]
    ).reset_index(drop=True)


def _read_manifest(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        manifest = dict(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise OCRGlobalV2Error(f"source manifest does not exist: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OCRGlobalV2Error(f"invalid source manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise OCRGlobalV2Error("source manifest must be a JSON object")
    return manifest


def _manifest_value(manifest: Mapping[str, Any], *path: str) -> Any:
    value: Any = manifest
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _explicit_true(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_packs: Sequence[str],
    expected_video_count: int | None,
) -> float:
    if manifest.get("status") != "completed":
        raise OCRGlobalV2Error(
            f"source manifest status must be 'completed', got {manifest.get('status')!r}"
        )
    schema = manifest.get("schema_version")
    if schema is not None and schema not in {"ocr_local_corpus_v1", SCHEMA_VERSION}:
        raise OCRGlobalV2Error(f"unsupported source manifest schema_version: {schema!r}")

    api_values = []
    network_values = []
    local_only_values = []
    for container in (
        manifest,
        _manifest_value(manifest, "engine"),
        _manifest_value(manifest, "provenance"),
        _manifest_value(manifest, "policy"),
    ):
        if isinstance(container, Mapping):
            if "api_used" in container:
                api_values.append(container["api_used"])
            if "network_allowed" in container:
                network_values.append(container["network_allowed"])
            if "local_only" in container:
                local_only_values.append(container["local_only"])
    if any(value is True for value in api_values):
        raise OCRGlobalV2Error("source manifest reports api_used=true")
    if not any(value is False for value in api_values):
        raise OCRGlobalV2Error("source manifest does not prove api_used=false")
    if any(value is not False for value in network_values if value is not None):
        raise OCRGlobalV2Error("source manifest is not local-only: network_allowed is not false")
    if any(value is False for value in local_only_values):
        raise OCRGlobalV2Error("source manifest reports local_only=false")

    mode_values = [manifest.get("mode"), _manifest_value(manifest, "scope", "mode")]
    if any(value is not None and value != "full" for value in mode_values):
        raise OCRGlobalV2Error("source manifest is not a full-corpus run")
    if _manifest_value(manifest, "canonical", "full_corpus") is False:
        raise OCRGlobalV2Error("source manifest canonical scope is not full_corpus")
    if _manifest_value(manifest, "coverage", "full_canonical_video_coverage") is False:
        raise OCRGlobalV2Error("source manifest does not cover all canonical videos")

    declared_video_counts = [
        _manifest_value(manifest, "canonical", "videos"),
        _manifest_value(manifest, "scope", "video_count"),
        _manifest_value(manifest, "coverage", "canonical_video_count"),
    ]
    if expected_video_count is not None:
        try:
            count_mismatch = any(
                value is not None and int(value) != int(expected_video_count)
                for value in declared_video_counts
            )
        except (TypeError, ValueError) as exc:
            raise OCRGlobalV2Error("source manifest declared video count is invalid") from exc
        if count_mismatch:
            raise OCRGlobalV2Error("source manifest declared video count does not cover the full scope")
    declared_packs = [
        _manifest_value(manifest, "canonical", "packs"),
        _manifest_value(manifest, "scope", "selected_packs"),
    ]
    expected_pack_set = set(expected_packs)
    for value in declared_packs:
        if value is not None:
            values = [value] if isinstance(value, str) else value
            try:
                actual = {str(item).upper() for item in values}
            except TypeError as exc:
                raise OCRGlobalV2Error("source manifest declared pack scope is invalid") from exc
            if actual != expected_pack_set:
                raise OCRGlobalV2Error("source manifest declared pack scope is incomplete")

    interval_candidates = [
        _manifest_value(manifest, "sampling", "interval_seconds"),
        _manifest_value(manifest, "sampling", "sample_interval_seconds"),
        manifest.get("sample_interval_seconds"),
        manifest.get("sampling_interval_seconds"),
    ]
    interval = next((value for value in interval_candidates if value is not None), None)
    if interval is None:
        raise OCRGlobalV2Error("source manifest is missing sampling interval_seconds")
    try:
        interval = float(interval)
    except (TypeError, ValueError) as exc:
        raise OCRGlobalV2Error("source manifest sampling interval_seconds is invalid") from exc
    if not np.isfinite(interval) or interval <= 0:
        raise OCRGlobalV2Error("source manifest sampling interval_seconds must be positive")
    return interval


def _add_no_text_values(target: set[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        if "video_id" in value:
            status = str(value.get("status", value.get("ocr_status", "no_text"))).strip().casefold()
            explicit = value.get("no_text", value.get("is_no_text", True))
            if _explicit_true(explicit) or status in _NO_TEXT_STATUSES:
                target.add(str(value["video_id"]).strip().upper())
            return
        for key, item in value.items():
            if isinstance(item, Mapping):
                status = str(item.get("status", item.get("ocr_status", ""))).strip().casefold()
                if _explicit_true(item.get("no_text", item.get("is_no_text", False))) or status in _NO_TEXT_STATUSES:
                    target.add(str(key).strip().upper())
            elif isinstance(item, bool) and item:
                target.add(str(key).strip().upper())
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, Mapping):
                _add_no_text_values(target, item)
            elif str(item).strip():
                target.add(str(item).strip().upper())


def _manifest_no_text_videos(manifest: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    containers = [manifest, _manifest_value(manifest, "coverage")]
    for parent in containers[:]:
        if not isinstance(parent, Mapping):
            continue
        for key in ("packs", "by_pack"):
            pack_values = parent.get(key)
            if isinstance(pack_values, Mapping):
                for item in pack_values.values():
                    if isinstance(item, Mapping):
                        _add_no_text_values(values, item.get("no_text_videos"))
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in ("no_text_videos", "no_text_video_ids", "no_text_records"):
            _add_no_text_values(values, container.get(key))
        for key in ("by_video", "video_coverage", "videos"):
            candidate = container.get(key)
            if isinstance(candidate, Mapping):
                _add_no_text_values(values, candidate)
            elif isinstance(candidate, list):
                _add_no_text_values(values, candidate)
    return {value for value in values if value}


def _row_is_no_text(row: Mapping[str, Any]) -> bool:
    for key in ("no_text", "is_no_text"):
        if key in row and _explicit_true(row[key]):
            return True
    for key in ("status", "ocr_status"):
        if key in row and str(row[key]).strip().casefold() in _NO_TEXT_STATUSES:
            return True
    return False


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value)


def _validate_metadata(
    source: str | Path | pd.DataFrame,
    canonical: pd.DataFrame,
    embeddings_source: str | Path | np.ndarray,
    manifest: Mapping[str, Any],
    *,
    expected_packs: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, set[str]]:
    table = _read_table(source).reset_index(drop=True)
    missing = sorted(set(INPUT_COLUMNS) - set(table.columns))
    if missing:
        raise OCRGlobalV2Error(f"OCR metadata missing columns: {missing}")
    if isinstance(embeddings_source, np.ndarray):
        matrix = np.asarray(embeddings_source)
    else:
        path = Path(embeddings_source)
        if not path.is_file():
            raise OCRGlobalV2Error(f"embedding matrix does not exist: {path}")
        try:
            matrix = np.asarray(np.load(path, allow_pickle=False))
        except Exception as exc:
            raise OCRGlobalV2Error(f"cannot read embedding matrix {path}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape != (len(table), EMBEDDING_DIM):
        raise OCRGlobalV2Error(
            f"embedding shape {list(matrix.shape)} does not align with metadata rows={len(table)} and dim={EMBEDDING_DIM}"
        )
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise OCRGlobalV2Error("embedding matrix contains non-finite values")

    canonical_map = {
        (str(row.video_id), int(row.kf_n)): row
        for row in canonical.itertuples(index=False)
    }
    manifest_no_text = _manifest_no_text_videos(manifest)
    metadata_no_text: set[str] = set()
    text_rows: list[dict[str, Any]] = []
    text_source_rows: list[int] = []
    seen_identity: set[tuple[str, int]] = set()
    seen_no_text_records: set[str] = set()

    for source_row, raw in enumerate(table.to_dict("records")):
        video_id = "" if _is_null(raw.get("video_id")) else str(raw["video_id"]).strip().upper()
        if not video_id:
            raise OCRGlobalV2Error(f"OCR metadata row {source_row} has empty video_id")
        pack_for_video(video_id, expected_packs=expected_packs)
        raw_text = "" if _is_null(raw.get("ocr_text")) else str(raw["ocr_text"]).strip()
        no_text = _row_is_no_text(raw)
        identity_values = [raw.get("kf_n"), raw.get("frame_idx"), raw.get("pts_time")]
        has_identity = not all(_is_null(value) for value in identity_values)
        if no_text:
            if raw_text:
                raise OCRGlobalV2Error(f"no-text metadata row {source_row} contains OCR text")
            if video_id in seen_no_text_records:
                raise OCRGlobalV2Error(f"duplicate explicit no-text video record: {video_id}")
            seen_no_text_records.add(video_id)
            metadata_no_text.add(video_id)
        elif not raw_text:
            raise OCRGlobalV2Error(
                f"empty OCR row {source_row} is not an explicit no-text video record"
            )

        if has_identity:
            if not all(not _is_null(value) for value in identity_values):
                raise OCRGlobalV2Error(f"OCR metadata row {source_row} has a partial canonical identity")
            try:
                kf_value = float(pd.to_numeric(raw["kf_n"], errors="raise"))
                frame_value = float(pd.to_numeric(raw["frame_idx"], errors="raise"))
                pts_value = float(pd.to_numeric(raw["pts_time"], errors="raise"))
                if not np.isfinite(kf_value) or kf_value != np.floor(kf_value):
                    raise ValueError("kf_n is not an integer")
                if not np.isfinite(frame_value) or frame_value != np.floor(frame_value):
                    raise ValueError("frame_idx is not an integer")
                if not np.isfinite(pts_value):
                    raise ValueError("pts_time is not finite")
                kf_n = int(kf_value)
                frame_idx = int(frame_value)
                pts_time = pts_value
            except (TypeError, ValueError, OverflowError) as exc:
                raise OCRGlobalV2Error(f"OCR metadata row {source_row} has invalid canonical identity") from exc
            key = (video_id, kf_n)
            if key in seen_identity:
                raise OCRGlobalV2Error(f"duplicate OCR identity: {key}")
            seen_identity.add(key)
            expected = canonical_map.get(key)
            if expected is None:
                raise OCRGlobalV2Error(f"OCR row {key} is outside canonical map")
            if frame_idx != int(expected.frame_idx) or not np.isclose(
                pts_time, float(expected.pts_time), atol=1e-3, rtol=0.0
            ):
                raise OCRGlobalV2Error(f"OCR row {key} disagrees with canonical frame/timestamp map")
            if no_text:
                continue
            text_rows.append({
                "video_id": video_id,
                "kf_n": kf_n,
                "frame_idx": int(expected.frame_idx),
                "pts_time": float(expected.pts_time),
                "ocr_text": raw_text,
                "source_pack": str(expected.source_pack),
            })
            text_source_rows.append(source_row)
        elif not no_text:
            raise OCRGlobalV2Error(f"OCR text row {source_row} has no canonical identity")

    if metadata_no_text & set(row["video_id"] for row in text_rows):
        overlap = sorted(metadata_no_text & set(row["video_id"] for row in text_rows))
        raise OCRGlobalV2Error(f"video has both text and explicit no-text records: {overlap[:8]}")
    if manifest_no_text & set(row["video_id"] for row in text_rows):
        overlap = sorted(manifest_no_text & set(row["video_id"] for row in text_rows))
        raise OCRGlobalV2Error(f"manifest marks text video as no-text: {overlap[:8]}")

    no_text_videos = metadata_no_text | manifest_no_text
    canonical_videos = set(canonical["video_id"].astype(str))
    unknown_no_text = sorted(no_text_videos - canonical_videos)
    if unknown_no_text:
        raise OCRGlobalV2Error(f"explicit no-text records reference unknown videos: {unknown_no_text[:8]}")
    text_videos = {row["video_id"] for row in text_rows}
    uncovered = sorted(canonical_videos - text_videos - no_text_videos)
    if uncovered:
        raise OCRGlobalV2Error(
            f"video coverage incomplete: {len(uncovered)} canonical videos lack OCR or explicit no-text records; "
            f"first={uncovered[:8]}"
        )

    output = pd.DataFrame(text_rows, columns=["video_id", "kf_n", "frame_idx", "pts_time", "ocr_text", "source_pack"])
    if output.empty:
        output = pd.DataFrame({
            "video_id": pd.Series(dtype="string"),
            "kf_n": pd.Series(dtype="int64"),
            "frame_idx": pd.Series(dtype="int64"),
            "pts_time": pd.Series(dtype="float64"),
            "ocr_text": pd.Series(dtype="string"),
            "source_pack": pd.Series(dtype="string"),
        })
        selected = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    else:
        # The source-row order is only used to select aligned vectors.  Apply
        # the same deterministic sort as the metadata table afterward.
        source_order = pd.DataFrame({"source_row": text_source_rows, **{
            key: [row[key] for row in text_rows] for key in ("video_id", "kf_n", "frame_idx", "pts_time", "ocr_text", "source_pack")
        }})
        source_order = source_order.sort_values(["source_pack", "video_id", "pts_time", "kf_n"]).reset_index(drop=True)
        selected = matrix[source_order["source_row"].to_numpy(dtype=np.int64)]
        output = source_order[["video_id", "kf_n", "frame_idx", "pts_time", "ocr_text", "source_pack"]]
    output.insert(0, "embedding_row", np.arange(len(output), dtype=np.int64))
    output = output[list(OUTPUT_COLUMNS)]
    if selected.shape != (len(output), EMBEDDING_DIM):
        raise OCRGlobalV2Error("filtered metadata and embeddings lost row alignment")
    return output, selected.astype(np.float32, copy=False), no_text_videos


def _coverage(
    canonical: pd.DataFrame,
    metadata: pd.DataFrame,
    no_text_videos: Iterable[str],
    *,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    no_text = set(no_text_videos)
    text_counts = metadata.groupby("video_id").size().to_dict() if not metadata.empty else {}
    by_video: dict[str, Any] = {}
    for video_id, group in canonical.groupby("video_id", sort=True):
        video = str(video_id)
        rows = int(text_counts.get(video, 0))
        status = "text" if rows else "no_text"
        by_video[video] = {
            "pack": str(group["source_pack"].iloc[0]),
            "status": status,
            "covered": bool(rows or video in no_text),
            "canonical_keyframes": int(len(group)),
            "sampled_ocr_rows": rows,
            "sampled_fraction": (rows / len(group)) if len(group) else 0.0,
            "explicit_no_text": bool(video in no_text),
        }
    by_pack: dict[str, Any] = {}
    for pack, group in canonical.groupby("source_pack", sort=True):
        pack_videos = sorted(str(value) for value in group["video_id"].unique())
        text_videos = [video for video in pack_videos if text_counts.get(video, 0)]
        no_text_pack = [video for video in pack_videos if video in no_text]
        covered = sorted(set(text_videos) | set(no_text_pack))
        by_pack[str(pack)] = {
            "canonical_videos": len(pack_videos),
            "covered_videos": len(covered),
            "text_videos": len(text_videos),
            "no_text_videos": len(no_text_pack),
            "missing_videos": sorted(set(pack_videos) - set(covered)),
            "canonical_keyframes": int(len(group)),
            "sampled_ocr_rows": int(sum(int(text_counts.get(video, 0)) for video in pack_videos)),
            "sampled_videos": len(text_videos),
            "sampled_fraction": (
                sum(int(text_counts.get(video, 0)) for video in pack_videos) / len(group)
                if len(group) else 0.0
            ),
        }
    return {
        "canonical_packs": int(canonical["source_pack"].nunique()),
        "covered_packs": int(sum(item["covered_videos"] == item["canonical_videos"] for item in by_pack.values())),
        "canonical_videos": int(canonical["video_id"].nunique()),
        "covered_videos": int(sum(item["covered"] for item in by_video.values())),
        "text_videos": int(sum(item["status"] == "text" for item in by_video.values())),
        "no_text_videos": int(sum(item["status"] == "no_text" for item in by_video.values())),
        "canonical_keyframes": int(len(canonical)),
        "sampled_ocr_rows": int(len(metadata)),
        "sample_interval_seconds": float(sample_interval_seconds),
        "all_packs_covered": all(item["covered_videos"] == item["canonical_videos"] for item in by_pack.values()),
        "all_videos_covered": all(item["covered"] for item in by_video.values()),
        "by_pack": by_pack,
        "by_video": by_video,
    }


def validate_artifacts(
    metadata_path: str | Path | pd.DataFrame,
    embeddings_path: str | Path | np.ndarray,
    manifest_path: str | Path | Mapping[str, Any],
    canonical_path: str | Path | pd.DataFrame,
    *,
    expected_packs: Sequence[str] = EXPECTED_PACKS,
    expected_video_count: int | None = EXPECTED_VIDEO_COUNT,
) -> ValidatedArtifacts:
    """Validate and canonicalize one sampled local OCR source.

    This function raises ``OCRGlobalV2Error`` on every failed gate.  In
    particular, sampled frame coverage is allowed, but video coverage is not.
    """
    packs = _normalize_packs(expected_packs)
    manifest = _read_manifest(manifest_path)
    interval = _validate_manifest(
        manifest, expected_packs=packs, expected_video_count=expected_video_count
    )
    canonical = load_canonical(
        canonical_path, expected_packs=packs, expected_video_count=expected_video_count
    )
    metadata, embeddings, no_text_videos = _validate_metadata(
        metadata_path, canonical, embeddings_path, manifest, expected_packs=packs
    )
    coverage = _coverage(
        canonical, metadata, no_text_videos, sample_interval_seconds=interval
    )
    if not coverage["all_packs_covered"] or not coverage["all_videos_covered"]:
        raise OCRGlobalV2Error("validated coverage is incomplete")
    return ValidatedArtifacts(
        metadata=metadata,
        embeddings=embeddings,
        manifest=manifest,
        no_text_videos=tuple(sorted(no_text_videos)),
        sample_interval_seconds=interval,
        coverage=coverage,
    )


validate_input_artifacts = validate_artifacts


def _canonical_digest(canonical: pd.DataFrame) -> str:
    rows = canonical[["video_id", "source_pack", "kf_n", "frame_idx", "pts_time"]].to_dict("records")
    return _sha256_json(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    temporary.replace(path)


def _blocked_manifest(
    output: Path,
    *,
    error: Exception,
    metadata_path: Any,
    embeddings_path: Any,
    source_manifest_path: Any,
    canonical_path: Any,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "error": f"{type(error).__name__}: {error}",
        "network_allowed": False,
        "api_used": False,
        "artifacts": {
            "retrieval": str(output / "retrieval.parquet"),
            "embeddings": str(output / "embeddings.npy"),
            "manifest": str(output / "manifest.json"),
        },
        "source": {
            "metadata": str(metadata_path),
            "embeddings": str(embeddings_path),
            "manifest": str(source_manifest_path),
            "canonical": str(canonical_path),
        },
    }
    _write_json(output / "manifest.json", payload)
    return payload


def build_global_index(
    metadata_path: str | Path,
    embeddings_path: str | Path,
    manifest_path: str | Path,
    canonical_path: str | Path,
    output_dir: str | Path,
    *,
    expected_packs: Sequence[str] = EXPECTED_PACKS,
    expected_video_count: int | None = EXPECTED_VIDEO_COUNT,
) -> dict[str, Any]:
    """Validate source artifacts and materialize the versioned global index.

    Invalid inputs produce only a ``status=blocked`` output manifest; the
    retrieval parquet and embedding matrix are not written.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        validated = validate_artifacts(
            metadata_path,
            embeddings_path,
            manifest_path,
            canonical_path,
            expected_packs=expected_packs,
            expected_video_count=expected_video_count,
        )
    except Exception as exc:
        if not isinstance(exc, OCRGlobalV2Error):
            exc = OCRGlobalV2Error(str(exc))
        return _blocked_manifest(
            output,
            error=exc,
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            source_manifest_path=manifest_path,
            canonical_path=canonical_path,
        )

    canonical = load_canonical(
        canonical_path,
        expected_packs=expected_packs,
        expected_video_count=expected_video_count,
    )
    _write_parquet(output / "retrieval.parquet", validated.metadata)
    _write_npy(output / "embeddings.npy", validated.embeddings)
    source_manifest = validated.manifest
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "index_id": f"ocr-global-v2-{_canonical_digest(canonical)[:16]}",
        "scope": {
            "name": "full_corpus",
            "packs": list(_normalize_packs(expected_packs)),
            "video_count": int(canonical["video_id"].nunique()),
            "canonical_keyframes": int(len(canonical)),
            "canonical_digest": _canonical_digest(canonical),
        },
        "sampling": {
            "sample_interval_seconds": validated.sample_interval_seconds,
            "sampled_canonical_frames": True,
            "canonical_keyframes": int(len(canonical)),
            "sampled_ocr_rows": int(len(validated.metadata)),
        },
        "coverage": validated.coverage,
        "rows": {
            "retrieval": int(len(validated.metadata)),
            "embeddings": int(len(validated.embeddings)),
        },
        "embedding": {
            "dim": EMBEDDING_DIM,
            "shape": list(validated.embeddings.shape),
            "dtype": str(validated.embeddings.dtype),
            "rows_aligned": True,
        },
        "provenance": {
            "source_schema_version": source_manifest.get("schema_version"),
            "source_status": source_manifest.get("status"),
            "metadata": str(metadata_path),
            "embeddings": str(embeddings_path),
            "manifest": str(manifest_path),
            "canonical": str(canonical_path),
            "api_used": False,
            "network_allowed": False,
            "local_only": True,
            "no_text_videos": list(validated.no_text_videos),
        },
        "artifacts": {
            "retrieval": str(output / "retrieval.parquet"),
            "embeddings": str(output / "embeddings.npy"),
            "manifest": str(output / "manifest.json"),
        },
    }
    _write_json(output / "manifest.json", report)
    return report


build_global_ocr_index = build_global_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/index/ocr_local_corpus_v1.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/index/emb_cache_ocr_local_corpus_v1.npy"))
    parser.add_argument("--manifest", type=Path, default=Path("results/ocr_local_corpus_v1_manifest.json"))
    parser.add_argument("--canonical", type=Path, default=Path("data/index/global_keyframes.parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/index/modality_global_v2/ocr_global_merged_v2"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_global_index(
        args.metadata,
        args.embeddings,
        args.manifest,
        args.canonical,
        args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if report.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
