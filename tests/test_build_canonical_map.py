from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.indexing.build_canonical_map import BuildConfig, CanonicalMapError, build_canonical_map


def _write_map(root: Path, video_id: str, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / f"{video_id}.csv", index=False)


def test_builds_canonical_map_portably_and_writes_manifest(tmp_path: Path) -> None:
    b1 = tmp_path / "b1"
    b2 = tmp_path / "b2"
    _write_map(b1, "L21_V001", [{"n": 1, "frame_idx": 0, "pts_time": 0.0}, {"n": 2, "frame_idx": 30, "pts_time": 1.0}])
    _write_map(b2, "K01_V001", [{"n": 1, "frame_idx": 0, "pts_time": 0.0}])
    output = tmp_path / "index" / "global_keyframes.parquet"

    report = build_canonical_map(BuildConfig((b1, b2), output))

    table = pd.read_parquet(output)
    assert table.columns.tolist() == ["g", "video_id", "pack", "kf_n", "frame_idx", "pts_time"]
    assert table["video_id"].tolist() == ["K01_V001", "L21_V001", "L21_V001"]
    assert report["videos"] == 2
    manifest = json.loads(output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["sha256"] == report["sha256"]


def test_canonical_builder_rejects_missing_keyframe_folder_when_requested(tmp_path: Path) -> None:
    maps = tmp_path / "maps"
    _write_map(maps, "K01_V001", [{"n": 1, "frame_idx": 0, "pts_time": 0.0}])
    with pytest.raises(CanonicalMapError, match="keyframe directory is absent"):
        build_canonical_map(BuildConfig((maps,), tmp_path / "out.parquet", tmp_path / "keyframes"))


def test_canonical_builder_rejects_duplicate_video_maps(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    rows = [{"n": 1, "frame_idx": 0, "pts_time": 0.0}]
    _write_map(first, "L21_V001", rows)
    _write_map(second, "L21_V001", rows)
    with pytest.raises(CanonicalMapError, match="duplicate video map"):
        build_canonical_map(BuildConfig((first, second), tmp_path / "out.parquet"))
