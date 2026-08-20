"""Merge validated K and L ASR shards into one canonical global index.

The merger accepts both historical K shards and the uniform output of
``asr_global_v2`` for *either* series.  This makes a full re-materialization
on a new server possible without changing the query-time ASR contract.  It is
the single boundary where every shard is checked against
``video_id/kf_n -> frame_idx/pts_time`` and it refuses incomplete coverage.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.reranking.asr_index import normalize_transcript


SCHEMA_VERSION = "hcmai.asr_global_merge_v2"
EMBEDDING_DIM = 1024
EXPECTED_PACKS = tuple([f"K{i:02d}" for i in range(1, 21)] + [f"L{i:02d}" for i in range(21, 31)])
REQUIRED_COLUMNS = (
    "video_id", "chunk_index", "text", "start", "end", "kf_n", "frame_idx",
    "pts_time", "distance_seconds", "source_pack", "source_provenance",
)


class ASRGlobalMergeError(RuntimeError):
    """Raised when a global ASR merge cannot be proven complete and aligned."""


@dataclass(frozen=True)
class MergeConfig:
    canonical: Path
    legacy_dir: Path
    l_dir: Path
    output_dir: Path
    packs: tuple[str, ...] = EXPECTED_PACKS


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    required = {"video_id", "kf_n", "frame_idx", "pts_time", "pack"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ASRGlobalMergeError(f"canonical index missing columns: {missing}")
    table = table.copy()
    table["video_id"] = table["video_id"].astype(str).str.strip().str.upper()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="raise").astype(int)
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="raise").astype(int)
    table["pts_time"] = pd.to_numeric(table["pts_time"], errors="raise").astype(float)
    if table.duplicated(["video_id", "kf_n"]).any():
        raise ASRGlobalMergeError("canonical index has duplicate (video_id, kf_n)")
    return table[["video_id", "kf_n", "frame_idx", "pts_time", "pack"]]


def _canonical_align(table: pd.DataFrame, canonical: pd.DataFrame, pack: str, provenance: str) -> pd.DataFrame:
    required = {"video_id", "chunk_index", "text", "start", "end", "kf_n", "frame_idx"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ASRGlobalMergeError(f"{pack} shard missing columns after normalization: {missing}")
    table = table.copy()
    table["video_id"] = table["video_id"].astype(str).str.strip().str.upper()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="raise").astype(int)
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="raise").astype(int)
    table["start"] = pd.to_numeric(table["start"], errors="raise").astype(float)
    table["end"] = pd.to_numeric(table["end"], errors="raise").astype(float)
    raw_text = table["text"].fillna("").astype(str)
    normalized_text = raw_text.map(normalize_transcript).str.replace(r"\s+", " ", regex=True).str.strip()
    changed_rows = normalized_text != raw_text.str.replace(r"\s+", " ", regex=True).str.strip()
    if changed_rows.any():
        raise ASRGlobalMergeError(
            f"{pack} shard requires transcript normalization in {int(changed_rows.sum())} rows; "
            "existing embeddings would be stale, so re-embed the affected shard before merging"
        )
    table["text"] = normalized_text
    if table["text"].eq("").any():
        raise ASRGlobalMergeError(f"{pack} shard contains empty transcript text")
    if (table["start"] < 0).any() or (table["end"] < table["start"]).any():
        raise ASRGlobalMergeError(f"{pack} shard contains invalid timestamps")
    canonical_subset = canonical[canonical["pack"].str.upper() == pack]
    aligned = table.merge(
        canonical_subset[["video_id", "kf_n", "frame_idx", "pts_time"]].rename(
            columns={"frame_idx": "canonical_frame_idx", "pts_time": "canonical_pts_time"}
        ),
        on=["video_id", "kf_n"], how="left", validate="many_to_one",
    )
    if aligned["canonical_frame_idx"].isna().any():
        raise ASRGlobalMergeError(f"{pack} shard contains keyframes outside canonical map")
    if (aligned["frame_idx"] != aligned["canonical_frame_idx"].astype(int)).any():
        raise ASRGlobalMergeError(f"{pack} shard frame_idx disagrees with canonical map")
    aligned["pts_time"] = aligned["canonical_pts_time"].astype(float)
    midpoint = (aligned["start"] + aligned["end"]) / 2.0
    aligned["distance_seconds"] = (aligned["pts_time"] - midpoint).abs()
    aligned["source_pack"] = pack
    aligned["source_provenance"] = provenance
    return aligned[list(REQUIRED_COLUMNS)]


def _fill_missing_legacy_mapping(table: pd.DataFrame, canonical: pd.DataFrame, pack: str) -> tuple[pd.DataFrame, bool]:
    """Map legacy chunks without kf_n/frame_idx to the nearest canonical keyframe.

    K11-K15 were materialized from valid timestamped ASR, but their historical
    parquet shards lost the optional keyframe mapping columns.  The transcript
    timestamps are still authoritative, so use the chunk midpoint only for
    locating a canonical keyframe.  Existing mappings are left untouched and
    are still checked by ``_canonical_align``.
    """
    table = table.copy()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="coerce")
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="coerce")
    missing = table["kf_n"].isna() | table["frame_idx"].isna()
    if not missing.any():
        return table, False

    canonical_subset = canonical[canonical["pack"].str.upper() == pack]
    if canonical_subset.empty:
        raise ASRGlobalMergeError(f"{pack} has no canonical keyframes for timestamp mapping")
    canonical_by_video = {
        video_id: group.sort_values("pts_time").reset_index(drop=True)
        for video_id, group in canonical_subset.groupby("video_id", sort=False)
    }
    midpoint = (pd.to_numeric(table["start"], errors="raise") + pd.to_numeric(table["end"], errors="raise")) / 2.0
    for row_index in table.index[missing]:
        video_id = str(table.at[row_index, "video_id"]).strip().upper()
        candidates = canonical_by_video.get(video_id)
        if candidates is None or candidates.empty:
            raise ASRGlobalMergeError(f"{pack} timestamp mapping references unknown video: {video_id}")
        target = float(midpoint.at[row_index])
        pts = candidates["pts_time"].to_numpy(dtype=float)
        insertion = int(np.searchsorted(pts, target, side="left"))
        choices = [max(0, insertion - 1), min(len(candidates) - 1, insertion)]
        chosen = min(choices, key=lambda position: abs(float(pts[position]) - target))
        table.at[row_index, "kf_n"] = int(candidates.at[chosen, "kf_n"])
        table.at[row_index, "frame_idx"] = int(candidates.at[chosen, "frame_idx"])
    return table, True


def normalize_k_shard(path: Path, canonical: pd.DataFrame, pack: str) -> pd.DataFrame:
    if not path.is_file():
        raise ASRGlobalMergeError(f"missing {pack} metadata shard: {path}")
    table = pd.read_parquet(path)
    required = {"chunk", "vid", "start", "end", "kf_n", "frame_idx"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ASRGlobalMergeError(f"legacy {pack} shard missing columns: {missing}")
    table = table.rename(columns={"chunk": "text", "vid": "video_id"})
    table["chunk_index"] = table.groupby("video_id", sort=False).cumcount()
    table, time_mapped = _fill_missing_legacy_mapping(table, canonical, pack)
    provenance = "validated_legacy_k_time_mapped" if time_mapped else "validated_legacy_k"
    return _canonical_align(table, canonical, pack, provenance)


def normalize_l_shard(path: Path, canonical: pd.DataFrame, pack: str) -> pd.DataFrame:
    if not path.is_file():
        raise ASRGlobalMergeError(f"missing {pack} metadata shard: {path}")
    table = pd.read_parquet(path)
    if "chunk_index" not in table.columns:
        table["chunk_index"] = table.groupby("video_id", sort=False).cumcount()
    return _canonical_align(table, canonical, pack, "deepgram_local_bge_l")


def normalize_materialized_shard(path: Path, canonical: pd.DataFrame, pack: str) -> pd.DataFrame:
    """Normalize the schema emitted by ``asr_global_v2`` for K or L packs."""
    if not path.is_file():
        raise ASRGlobalMergeError(f"missing {pack} materialized shard: {path}")
    table = pd.read_parquet(path)
    if "chunk_index" not in table.columns:
        table["chunk_index"] = table.groupby("video_id", sort=False).cumcount()
    return _canonical_align(table, canonical, pack, f"deepgram_local_bge_{pack[0].lower()}")


def _select_pack_source(config: MergeConfig, canonical: pd.DataFrame, pack: str) -> tuple[pd.DataFrame, Path, str]:
    """Prefer validated historical K data, else use the portable materializer."""
    filename = f"asr_chunks_{pack.lower()}_ts.parquet"
    legacy_shard = config.legacy_dir / filename
    materialized_shard = config.l_dir / filename
    if pack.startswith("K") and legacy_shard.is_file():
        return normalize_k_shard(legacy_shard, canonical, pack), config.legacy_dir, "legacy"
    return normalize_materialized_shard(materialized_shard, canonical, pack), config.l_dir, "materialized"


def _load_embedding(path: Path, rows: int, pack: str) -> np.ndarray:
    if not path.is_file():
        raise ASRGlobalMergeError(f"missing {pack} embedding shard: {path}")
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.shape != (rows, EMBEDDING_DIM):
        raise ASRGlobalMergeError(
            f"{pack} embedding shape {array.shape} does not match ({rows}, {EMBEDDING_DIM})"
        )
    if not np.isfinite(array).all():
        raise ASRGlobalMergeError(f"{pack} embedding contains non-finite values")
    return array


def _load_no_speech_videos(l_dir: Path, pack: str) -> set[str]:
    manifest = l_dir / f"asr_global_v2_{pack.lower()}_manifest.json"
    if not manifest.is_file():
        return set()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        values = payload.get("no_speech_videos", [])
        return {str(value).strip().upper() for value in values if str(value).strip()}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ASRGlobalMergeError(f"invalid {pack} no-speech manifest: {manifest}: {exc}") from exc


def merge_global(config: MergeConfig) -> dict[str, Any]:
    expected = tuple(str(pack).upper() for pack in config.packs)
    if set(expected) != set(EXPECTED_PACKS):
        raise ASRGlobalMergeError("global merge requires exactly K01-K20 and L21-L30")
    canonical = load_canonical(config.canonical)
    frames: list[pd.DataFrame] = []
    embeddings: list[np.ndarray] = []
    pack_reports: dict[str, Any] = {}
    for pack in expected:
        frame, shard_dir, source_kind = _select_pack_source(config, canonical, pack)
        embedding = shard_dir / f"emb_cache_asr_{pack.lower()}_chunks.npy"
        no_speech = _load_no_speech_videos(config.l_dir, pack)
        array = _load_embedding(embedding, len(frame), pack)
        expected_videos = set(canonical.loc[canonical["pack"].str.upper() == pack, "video_id"])
        observed_videos = set(frame["video_id"])
        if observed_videos & no_speech:
            raise ASRGlobalMergeError(f"{pack} video is both transcribed and marked no-speech")
        covered_videos = observed_videos | no_speech
        if covered_videos != expected_videos:
            raise ASRGlobalMergeError(
                f"{pack} video coverage mismatch: expected {len(expected_videos)}, got {len(covered_videos)}"
            )
        frames.append(frame)
        embeddings.append(array)
        pack_reports[pack] = {
            "videos": len(observed_videos),
            "no_speech_videos": sorted(no_speech),
            "covered_videos": len(covered_videos),
            "rows": len(frame),
            "embedding_shape": list(array.shape),
            "provenance": sorted(frame["source_provenance"].unique().tolist()),
            "source_kind": source_kind,
        }
    metadata = pd.concat(frames, ignore_index=True)
    matrix = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    metadata.insert(0, "embedding_row", np.arange(len(metadata), dtype=np.int64))
    if matrix.shape != (len(metadata), EMBEDDING_DIM):
        raise ASRGlobalMergeError("global metadata/embedding row alignment failed")
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output / "retrieval.parquet", index=False)
    np.save(output / "embeddings.npy", matrix)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "index_id": f"asr-global-{_digest(config.canonical)[:16]}",
        "scope": {
            "name": "full_corpus",
            "video_count": int(canonical["video_id"].nunique()),
            "video_ids": sorted(canonical["video_id"].unique().tolist()),
            "packs": list(expected),
        },
        "canonical": {
            "path": str(config.canonical),
            "validated": True,
            "video_count": int(canonical["video_id"].nunique()),
            "mapping_errors": 0,
        },
        "rows": {"metadata": len(metadata), "embedding": len(metadata)},
        "embedding": {"dim": EMBEDDING_DIM, "shape": list(matrix.shape)},
        "artifacts": {
            "retrieval": str(output / "retrieval.parquet"),
            "embeddings": str(output / "embeddings.npy"),
        },
        "packs": pack_reports,
        "provenance": {
            "k_source": "validated_legacy_k_or_deepgram_local_bge_k",
            "l_source": "deepgram_local_bge_l",
            "network_used_by_merge": False,
        },
    }
    (output / "asr_global_merge_v2_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path("data/index/global_keyframes.parquet"))
    parser.add_argument("--legacy-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--l-dir", type=Path, default=Path("data/index/modality_global_v2/asr_global_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/index/modality_global_v2/asr_global_merged_v2"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = merge_global(MergeConfig(args.canonical, args.legacy_dir, args.l_dir, args.output_dir))
    except ASRGlobalMergeError as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
