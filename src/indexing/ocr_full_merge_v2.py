"""Merge validated legacy OCR and local supplemental OCR into one global index.

The corpus already contains a large, useful OCR legacy set, but it is split by
pack and does not have one authoritative manifest.  This adapter makes that
provenance explicit, validates every row against the canonical keyframe map,
and adds a fresh local-Qwen sample for videos absent from the legacy set.

The resulting contract is video-global, not frame-complete: every canonical
video must have at least one validated OCR row or an explicit no-text record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_PACKS = tuple(
    [f"K{i:02d}" for i in range(1, 21)]
    + [f"L{i:02d}" for i in range(21, 31)]
)
EXPECTED_VIDEO_COUNT = 1478
EMBEDDING_DIM = 1024
VIDEO_RE = re.compile(r"^([KL]\d{2})_V\d+$", re.IGNORECASE)
REQUIRED_METADATA = {"video_id", "kf_n", "pts_time", "ocr_text"}


class OCRFullMergeError(RuntimeError):
    """Raised when a global OCR merge cannot be proven complete."""


def _canonical(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path).copy()
    required = {"video_id", "kf_n", "frame_idx", "pts_time"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise OCRFullMergeError(f"canonical index missing columns: {missing}")
    table["video_id"] = table["video_id"].astype(str).str.strip().str.upper()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="raise").astype(np.int64)
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="raise").astype(np.int64)
    table["pts_time"] = pd.to_numeric(table["pts_time"], errors="raise").astype(float)
    if table.duplicated(["video_id", "kf_n"]).any():
        raise OCRFullMergeError("canonical index has duplicate (video_id, kf_n)")
    table["source_pack"] = table["video_id"].str.extract(r"^([KL]\d{2})_", expand=False)
    if table["source_pack"].isna().any():
        raise OCRFullMergeError("canonical index contains invalid video_id")
    if set(table["source_pack"]) != set(EXPECTED_PACKS):
        raise OCRFullMergeError("canonical index does not cover all K/L packs")
    if int(table["video_id"].nunique()) != EXPECTED_VIDEO_COUNT:
        raise OCRFullMergeError(
            f"canonical video count mismatch: expected {EXPECTED_VIDEO_COUNT}, "
            f"got {table['video_id'].nunique()}"
        )
    return table[["video_id", "kf_n", "frame_idx", "pts_time", "source_pack"]]


def _embedding(path: Path, rows: int, source: str) -> np.ndarray:
    if not path.is_file():
        raise OCRFullMergeError(f"missing embedding file for {source}: {path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if matrix.shape != (rows, EMBEDDING_DIM):
        raise OCRFullMergeError(
            f"{source} embedding shape {matrix.shape} does not match "
            f"({rows}, {EMBEDDING_DIM})"
        )
    if not np.isfinite(matrix).all():
        raise OCRFullMergeError(f"{source} embedding contains non-finite values")
    return matrix


def _align(
    table: pd.DataFrame,
    matrix: np.ndarray,
    canonical: pd.DataFrame,
    *,
    source: str,
    priority: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = sorted(REQUIRED_METADATA - set(table.columns))
    if missing:
        raise OCRFullMergeError(f"{source} metadata missing columns: {missing}")
    table = table.copy().reset_index(drop=True)
    if len(matrix) != len(table):
        raise OCRFullMergeError(f"{source} metadata/embedding row count mismatch")
    table["video_id"] = table["video_id"].astype(str).str.strip().str.upper()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="coerce")
    if table["kf_n"].isna().any() or (table["kf_n"] % 1 != 0).any():
        raise OCRFullMergeError(f"{source} has invalid kf_n")
    table["kf_n"] = table["kf_n"].astype(np.int64)
    table["ocr_text"] = table["ocr_text"].fillna("").astype(str).str.strip()
    if table["ocr_text"].eq("").any():
        raise OCRFullMergeError(f"{source} contains empty OCR text")
    if table.duplicated(["video_id", "kf_n"]).any():
        raise OCRFullMergeError(f"{source} contains duplicate (video_id, kf_n)")
    aligned = table.merge(
        canonical,
        on=["video_id", "kf_n"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_canonical"),
    )
    if aligned["frame_idx"].isna().any():
        examples = aligned.loc[aligned["frame_idx"].isna(), ["video_id", "kf_n"]].head(5)
        raise OCRFullMergeError(f"{source} contains non-canonical frames: {examples.to_dict('records')}")
    if "pts_time_canonical" in aligned and not np.allclose(
        aligned["pts_time"].astype(float), aligned["pts_time_canonical"].astype(float), atol=1e-2
    ):
        raise OCRFullMergeError(f"{source} pts_time disagrees with canonical map")
    aligned["pts_time"] = aligned["pts_time_canonical"].astype(float) if "pts_time_canonical" in aligned else aligned["pts_time"].astype(float)
    aligned["source_pack"] = aligned["source_pack"].astype(str).str.upper()
    aligned["source_provenance"] = source
    aligned["source_priority"] = int(priority)
    keep = [
        "video_id", "kf_n", "frame_idx", "pts_time", "ocr_text",
        "source_pack", "source_provenance", "source_priority",
    ]
    return aligned[keep], matrix


def _legacy_sources(legacy_dir: Path) -> Iterable[tuple[str, Path, Path]]:
    for metadata_path in sorted(legacy_dir.glob("ocr_*.parquet")):
        match = re.fullmatch(r"ocr_([kl]\d{2})\.parquet", metadata_path.name, re.IGNORECASE)
        if not match:
            continue
        embedding_path = legacy_dir / f"emb_cache_{metadata_path.stem}.npy"
        if embedding_path.is_file():
            yield match.group(1).upper(), metadata_path, embedding_path


def _no_text_from_manifest(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: set[str] = set()
    for container in (payload, payload.get("coverage", {}), payload.get("provenance", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("no_text_videos", "no_text"):
            value = container.get(key, [])
            if isinstance(value, Mapping):
                value = value.keys()
            if isinstance(value, (list, tuple, set)):
                values.update(str(item).strip().upper() for item in value if str(item).strip())
    return values


def merge_full(
    *,
    canonical_path: Path,
    legacy_dir: Path,
    supplemental_metadata: Path,
    supplemental_embeddings: Path,
    supplemental_manifest: Path | None,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    canonical = _canonical(canonical_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise OCRFullMergeError(f"output directory is not empty: {output_dir}; pass --overwrite")

    frames: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    source_reports: dict[str, Any] = {}
    for pack, metadata_path, embedding_path in _legacy_sources(legacy_dir):
        raw = pd.read_parquet(metadata_path)
        matrix = _embedding(embedding_path, len(raw), metadata_path.name)
        aligned, matrix = _align(
            raw, matrix, canonical, source=f"legacy_local_ocr:{metadata_path.name}", priority=20
        )
        frames.append(aligned)
        matrices.append(matrix)
        source_reports[pack] = {"source": str(metadata_path), "rows": len(aligned), "videos": int(aligned["video_id"].nunique())}

    supplemental = pd.read_parquet(supplemental_metadata)
    supplemental_matrix = _embedding(supplemental_embeddings, len(supplemental), "supplemental_local_ocr")
    aligned, supplemental_matrix = _align(
        supplemental, supplemental_matrix, canonical,
        source="qwen_local_supplemental", priority=10,
    )
    frames.append(aligned)
    matrices.append(supplemental_matrix)

    metadata = pd.concat(frames, ignore_index=True)
    matrix = np.concatenate(matrices, axis=0).astype(np.float32, copy=False)
    metadata = metadata.sort_values(
        ["video_id", "kf_n", "source_priority", "source_provenance"],
        kind="stable",
    ).drop_duplicates(["video_id", "kf_n"], keep="first").reset_index(drop=True)
    # Reorder embeddings with the same stable de-duplication decision.
    all_rows = pd.concat(
        [frame.assign(_matrix_row=np.arange(len(frame), dtype=np.int64)) for frame in frames],
        ignore_index=True,
    )
    all_rows = all_rows.sort_values(
        ["video_id", "kf_n", "source_priority", "source_provenance"], kind="stable"
    ).drop_duplicates(["video_id", "kf_n"], keep="first").reset_index(drop=True)
    matrix = matrix[all_rows["_matrix_row"].to_numpy(dtype=np.int64)]
    metadata = all_rows.drop(columns=["_matrix_row"])

    observed = set(metadata["video_id"].astype(str))
    no_text = _no_text_from_manifest(supplemental_manifest) - observed
    expected = set(canonical["video_id"].astype(str))
    invalid_no_text = sorted(no_text - expected)
    if invalid_no_text:
        raise OCRFullMergeError(f"supplemental no-text records are outside canonical scope: {invalid_no_text[:5]}")
    uncovered = sorted(expected - observed - no_text)
    if uncovered:
        raise OCRFullMergeError(
            f"global OCR video coverage incomplete: {len(uncovered)} videos; examples={uncovered[:10]}"
        )

    metadata.insert(0, "embedding_row", np.arange(len(metadata), dtype=np.int64))
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = output_dir / "retrieval.parquet"
    embeddings_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "manifest.json"
    metadata.to_parquet(retrieval_path, index=False)
    np.save(embeddings_path, matrix)
    pack_reports = {}
    for pack in EXPECTED_PACKS:
        subset = metadata[metadata["source_pack"] == pack]
        expected_pack_videos = set(canonical.loc[canonical["source_pack"] == pack, "video_id"])
        observed_pack = set(subset["video_id"])
        no_text_pack = sorted(no_text & expected_pack_videos)
        pack_reports[pack] = {
            "canonical_videos": len(expected_pack_videos),
            "text_videos": len(observed_pack),
            "no_text_videos": no_text_pack,
            "covered_videos": len(observed_pack | set(no_text_pack)),
            "rows": len(subset),
        }
    manifest = {
        "schema_version": "hcmai.ocr_global_full_merge_v2",
        "status": "ready",
        "index_id": "ocr-global-full-" + hashlib.sha256(canonical_path.read_bytes()).hexdigest()[:16],
        "scope": {"name": "full_corpus_video_coverage", "video_count": len(expected), "packs": list(EXPECTED_PACKS)},
        "canonical": {"path": str(canonical_path), "rows": len(canonical), "videos": len(expected), "validated": True},
        "sampling": {"sample_interval_seconds": 30.0, "frame_complete": False},
        "coverage": {
            "canonical_videos": len(expected),
            "covered_videos": len(observed | no_text),
            "text_videos": len(observed),
            "no_text_videos": sorted(no_text),
            "video_coverage": len(observed | no_text) / len(expected),
            "ocr_row_count": len(metadata),
            "canonical_frame_coverage": len(metadata) / len(canonical),
            "frame_complete": False,
        },
        "embedding": {"dim": EMBEDDING_DIM, "shape": list(matrix.shape)},
        "provenance": {
            "api_used": False,
            "network_allowed": False,
            "no_text_videos": sorted(no_text),
            "legacy_sources": source_reports,
            "supplemental_metadata": str(supplemental_metadata),
            "supplemental_manifest": str(supplemental_manifest) if supplemental_manifest else None,
        },
        "packs": pack_reports,
        "artifacts": {"retrieval": str(retrieval_path), "embeddings": str(embeddings_path)},
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical", dest="canonical_path", type=Path,
        default=Path("data/index/global_keyframes.parquet"),
    )
    parser.add_argument("--legacy-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--supplemental-metadata", type=Path, required=True)
    parser.add_argument("--supplemental-embeddings", type=Path, required=True)
    parser.add_argument("--supplemental-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/index/modality_global_v2/ocr_global_merged_v2"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(merge_full(**vars(args)), ensure_ascii=False, indent=2))
    except OCRFullMergeError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
