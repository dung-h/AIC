from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eval.materialize_asr_wave1a import (
    _canonical_map,
    _iter_raw_chunks,
    _raw_paths,
    _full_audit,
)


def test_raw_chunk_parser_is_timestamped_and_nonempty(tmp_path: Path) -> None:
    payload = {
        "results": {
            "utterances": [
                {"transcript": "Xin chào", "start": 1.0, "end": 2.0},
                {"transcript": "Xin chào", "start": 1.0, "end": 2.0},
                {"transcript": "Nha Trang", "start": 3.0, "end": 4.0},
            ]
        }
    }
    path = tmp_path / "K01_V001.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert list(_iter_raw_chunks(path)) == [
        {"text": "Xin chào", "start": 1.0, "end": 2.0},
        {"text": "Nha Trang", "start": 3.0, "end": 4.0},
    ]


def test_canonical_map_rejects_missing_columns() -> None:
    try:
        _canonical_map(pd.DataFrame({"video_id": ["K01_V001"]}))
    except RuntimeError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("invalid canonical map was accepted")


def test_full_audit_is_fail_closed_for_missing_raw_asr(tmp_path: Path) -> None:
    canonical = {
        "K01_V001": [{"kf_n": 1, "frame_idx": 0, "pts_time": 0.0}],
        "L21_V001": [{"kf_n": 1, "frame_idx": 0, "pts_time": 0.0}],
    }
    report = _full_audit(canonical, {"K01_V001": tmp_path / "K01_V001.json"}, tmp_path / "out", "catalog")
    assert report["blocked"] is True
    assert report["canonical_videos"] == 2
    assert report["raw_asr_videos"] == 1
    assert report["missing_raw_asr_videos"] == ["L21_V001"]
    assert (tmp_path / "out" / "asr_wave1a_full_audit.json").is_file()


def test_raw_paths_only_accepts_canonical_video_names(tmp_path: Path) -> None:
    folder = tmp_path / "asr_k01"
    folder.mkdir()
    (folder / "K01_V001.json").write_text("{}", encoding="utf-8")
    (folder / "not_a_video.json").write_text("{}", encoding="utf-8")
    assert list(_raw_paths(tmp_path)) == ["K01_V001"]
