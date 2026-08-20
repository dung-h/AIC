"""Build the portable canonical keyframe map from official map-keyframes CSVs.

This is the only supported boundary for creating the table that carries
``video_id, kf_n, frame_idx, pts_time``.  Encoders and submission adapters must
consume this table instead of reading map CSVs themselves.  The builder has no
machine-specific path, never touches images, and writes data atomically.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd


VIDEO_ID_RE = re.compile(r"^(?P<pack>[KL]\d{2})_V\d{3}$", re.IGNORECASE)
REQUIRED_COLUMNS = ("n", "frame_idx", "pts_time")
DEFAULT_MAP_ROOTS = (
    Path("data/raw/map-keyframes-aic25-b1"),
    Path("data/raw/map-keyframes-b2"),
)


class CanonicalMapError(RuntimeError):
    """Raised when official map CSVs cannot prove a canonical mapping."""


@dataclass(frozen=True)
class BuildConfig:
    map_roots: tuple[Path, ...]
    output: Path
    require_keyframes_root: Path | None = None
    overwrite: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_map_csvs(map_roots: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in map_roots:
        if not root.is_dir():
            raise CanonicalMapError(f"map root does not exist: {root}")
        paths.extend(path for path in root.rglob("*.csv") if VIDEO_ID_RE.fullmatch(path.stem))
    unique = sorted({path.resolve() for path in paths}, key=lambda path: (path.stem.upper(), str(path)))
    if not unique:
        raise CanonicalMapError("no official map-keyframes CSVs found")
    seen: dict[str, Path] = {}
    duplicated: list[str] = []
    for path in unique:
        video_id = path.stem.upper()
        if video_id in seen:
            duplicated.append(video_id)
        else:
            seen[video_id] = path
    if duplicated:
        names = sorted(set(duplicated))
        raise CanonicalMapError(f"duplicate video map CSVs across map roots: {names[:10]}")
    return unique


def _load_csv(path: Path, keyframes_root: Path | None) -> pd.DataFrame:
    video_id = path.stem.upper()
    match = VIDEO_ID_RE.fullmatch(video_id)
    if match is None:
        raise CanonicalMapError(f"invalid video id in map filename: {path.name}")
    if keyframes_root is not None and not (keyframes_root / video_id).is_dir():
        raise CanonicalMapError(f"map exists but keyframe directory is absent: {video_id}")
    try:
        table = pd.read_csv(path)
    except Exception as exc:  # pandas exposes several parser exceptions
        raise CanonicalMapError(f"cannot read map CSV: {path}: {exc}") from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise CanonicalMapError(f"map CSV missing required columns {missing}: {path}")
    result = table.loc[:, list(REQUIRED_COLUMNS)].copy()
    try:
        result["n"] = pd.to_numeric(result["n"], errors="raise").astype("int64")
        result["frame_idx"] = pd.to_numeric(result["frame_idx"], errors="raise").astype("int64")
        result["pts_time"] = pd.to_numeric(result["pts_time"], errors="raise").astype("float64")
    except Exception as exc:
        raise CanonicalMapError(f"non-numeric canonical coordinates in {path}") from exc
    if result.empty:
        raise CanonicalMapError(f"empty map CSV: {path}")
    if (result["n"] < 0).any() or (result["frame_idx"] < 0).any() or (result["pts_time"] < 0).any():
        raise CanonicalMapError(f"negative canonical coordinate in {path}")
    if result["n"].duplicated().any():
        raise CanonicalMapError(f"duplicate keyframe ordinal in {path}")
    if not result["pts_time"].is_monotonic_increasing:
        raise CanonicalMapError(f"non-monotonic keyframe timestamps in {path}")
    if not result["frame_idx"].is_monotonic_increasing:
        raise CanonicalMapError(f"non-monotonic frame indices in {path}")
    result.insert(0, "video_id", video_id)
    result.insert(1, "pack", match.group("pack").upper())
    return result.rename(columns={"n": "kf_n"})


def build_canonical_map(config: BuildConfig) -> dict[str, object]:
    if config.output.exists() and not config.overwrite:
        raise CanonicalMapError(f"refusing to overwrite existing canonical map: {config.output}")
    maps = discover_map_csvs(config.map_roots)
    tables = [_load_csv(path, config.require_keyframes_root) for path in maps]
    result = pd.concat(tables, ignore_index=True)
    result = result.sort_values(["pack", "video_id", "kf_n"], kind="stable").reset_index(drop=True)
    if result.duplicated(["video_id", "kf_n"]).any():
        raise CanonicalMapError("duplicate (video_id, kf_n) after merge")
    result.insert(0, "g", range(len(result)))
    config.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output.with_name(f".{config.output.name}.tmp")
    result.to_parquet(temporary, index=False)
    temporary.replace(config.output)
    report = {
        "schema_version": "hcmai.canonical_map.v1",
        "created_at": _utc_now(),
        "status": "ready",
        "output": str(config.output),
        "sha256": _sha256(config.output),
        "rows": int(len(result)),
        "videos": int(result["video_id"].nunique()),
        "packs": sorted(result["pack"].unique().tolist()),
        "map_roots": [str(path) for path in config.map_roots],
        "keyframe_directory_checked": str(config.require_keyframes_root) if config.require_keyframes_root else None,
    }
    manifest = config.output.with_suffix(config.output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-root",
        type=Path,
        action="append",
        dest="map_roots",
        help="Root containing official map-keyframes CSVs; repeat for b1 and b2.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/index/global_keyframes.parquet"))
    parser.add_argument(
        "--require-keyframes-root",
        type=Path,
        default=None,
        help="Fail if any official map video lacks this keyframe directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = tuple(args.map_roots) if args.map_roots else DEFAULT_MAP_ROOTS
    try:
        report = build_canonical_map(
            BuildConfig(roots, args.output, args.require_keyframes_root, args.overwrite)
        )
    except CanonicalMapError as exc:
        print(json.dumps({"schema_version": "hcmai.canonical_map.v1", "status": "blocked", "error": str(exc)}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
