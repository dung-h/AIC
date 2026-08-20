"""Offline contract tests for the bounded Deepgram benchmark crawler."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src.utils.deepgram_benchmark_crawl import (
    CrawlConfigError,
    build_targets,
    load_annotations,
    merge_resume,
    normalize_watch_url,
    require_deepgram_key,
    run,
)


def _write_media(directory: Path, video_id: str, url: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{video_id}.json").write_text(
        json.dumps({"watch_url": url}, ensure_ascii=False), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _args(tmp_path: Path, input_path: Path, media: Path, **overrides) -> Namespace:
    values = dict(
        input=input_path,
        media_info_dirs=[media],
        output=tmp_path / "manifest.json",
        cache=tmp_path / "cache.json",
        audio_dir=tmp_path / "audio",
        audio_backend="none",
        packs=None,
        pack=None,
        video_id=None,
        limit=None,
        video_limit=None,
        resume=False,
        overwrite=False,
        dry_run=False,
        strict=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_requires_key_without_logging_or_accepting_empty() -> None:
    with pytest.raises(CrawlConfigError, match="DEEPGRAM_API_KEY"):
        require_deepgram_key(env={})
    require_deepgram_key(env={"DEEPGRAM_API_KEY": "test-only-key"})


def test_filters_spoken_fact_dedupes_urls_and_counts_k_l(tmp_path: Path) -> None:
    media = tmp_path / "media"
    _write_media(media, "K01_V001", "https://youtube.com/watch?v=abc")
    _write_media(media, "K01_V002", "https://www.youtube.com/watch?v=abc")
    _write_media(media, "L25_V001", "https://youtu.be/l25")
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [
        {"annotation_id": "a1", "question_type": "spoken_fact", "video_id": "K01_V001", "pack": "K01", "answer_start_time": 1, "answer_end_time": 2},
        {"annotation_id": "a2", "question_type": "spoken_fact", "video_id": "K01_V002", "pack": "K01", "answer_start_time": 3, "answer_end_time": 4},
        {"annotation_id": "a3", "question_type": "color", "video_id": "L25_V001", "pack": "L25"},
        {"annotation_id": "a4", "question_type": "spoken_fact", "video_id": "L25_V001", "pack": "L25", "answer_start_time": 5, "answer_end_time": 6},
    ])
    rows = load_annotations(input_path)
    targets, summary = build_targets(rows, [media])
    assert len(targets) == 2
    assert summary["unique_target_urls"] == 2
    assert summary["target_k_count"] == 1
    assert summary["target_l_count"] == 1
    assert targets[0].annotation_ids == ["a1", "a2"]
    assert targets[0].url == "https://www.youtube.com/watch?v=abc"


def test_packs_limit_and_resume_are_deterministic(tmp_path: Path) -> None:
    media = tmp_path / "media"
    _write_media(media, "K01_V001", "https://youtube.com/watch?v=one")
    _write_media(media, "K02_V001", "https://youtube.com/watch?v=two")
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [
        {"annotation_id": "a1", "question_type": "spoken_fact", "video_id": "K02_V001", "pack": "K02"},
        {"annotation_id": "a2", "question_type": "spoken_fact", "video_id": "K01_V001", "pack": "K01"},
    ])
    args = _args(tmp_path, input_path, media, packs="K01", limit=1, dry_run=False)
    first = run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})
    assert first["summary"]["target_packs"] == {"K01": 1}
    args.packs = None
    args.resume = True
    second = run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})
    assert second["targets"][0]["resumed"] is True


def test_existing_manifest_requires_resume_or_overwrite(tmp_path: Path) -> None:
    media = tmp_path / "media"
    _write_media(media, "K01_V001", "https://youtube.com/watch?v=one")
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [{"annotation_id": "a1", "question_type": "spoken_fact", "video_id": "K01_V001", "pack": "K01"}])
    args = _args(tmp_path, input_path, media)
    run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})
    args.resume = False
    with pytest.raises(CrawlConfigError, match="refusing to overwrite"):
        run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})


def test_video_scope_and_pack_coverage_are_recorded(tmp_path: Path) -> None:
    media = tmp_path / "media"
    _write_media(media, "K01_V001", "https://youtube.com/watch?v=one")
    _write_media(media, "K02_V001", "https://youtube.com/watch?v=two")
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [
        {"annotation_id": "a1", "question_type": "spoken_fact", "video_id": "K01_V001", "pack": "K01"},
        {"annotation_id": "a2", "question_type": "spoken_fact", "video_id": "K02_V001", "pack": "K02"},
    ])
    args = _args(tmp_path, input_path, media, pack=["K02"], video_limit=1)
    manifest = run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})
    assert manifest["selection"]["selected_video_ids"] == ["K02_V001"]
    assert sorted(manifest["coverage_by_pack"]) == ["K02"]


def test_dry_run_writes_nothing_and_makes_no_api_call(tmp_path: Path) -> None:
    media = tmp_path / "media"
    _write_media(media, "K01_V001", "https://youtube.com/watch?v=one")
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [{"annotation_id": "a1", "question_type": "spoken_fact", "video_id": "K01_V001", "pack": "K01"}])
    args = _args(tmp_path, input_path, media, dry_run=True)
    manifest = run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})
    assert manifest["deepgram"]["api_calls_made"] == 0
    assert not args.output.exists()
    assert not args.cache.exists()


def test_missing_media_is_reported_and_strict_mode_fails(tmp_path: Path) -> None:
    media = tmp_path / "media"
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, [{"annotation_id": "a1", "question_type": "spoken_fact", "video_id": "L25_V999", "pack": "L25"}])
    args = _args(tmp_path, input_path, media, dry_run=True, strict=True)
    with pytest.raises(CrawlConfigError, match="Missing media-info"):
        run(args, env={"DEEPGRAM_API_KEY": "test-only-key"})
