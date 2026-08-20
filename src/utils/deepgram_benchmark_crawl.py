"""Build a bounded Deepgram ASR preparation queue for the VQA benchmark.

This module deliberately stops before any network operation.  It resolves the
15 (or otherwise bounded) ``spoken_fact`` benchmark targets to the local
media-info ``watch_url`` and writes a resumable manifest.  A later worker can
consume that manifest, extract audio, and submit it to Deepgram.

Examples (PowerShell, from C:\\HCMAI)::

    $env:DEEPGRAM_API_KEY = "..."
    python -m src.utils.deepgram_benchmark_crawl --dry-run
    python -m src.utils.deepgram_benchmark_crawl --limit 3 --audio-backend local_ffmpeg

The API key is required as an operational guard, but is never written to a
manifest or printed.  ``--dry-run`` and tests do not call Deepgram, YouTube,
yt-dlp, urllib, or any other network service.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "annotations" / "vqa_eval_v3_1.jsonl"
DEFAULT_MEDIA_INFO_DIRS = (
    ROOT / "data" / "raw" / "media-info-aic25-b2" / "media-info",
    ROOT / "data" / "raw" / "media-info-aic25-b1" / "media-info",
)
DEFAULT_OUTPUT = ROOT / "results" / "deepgram_benchmark_targets.json"
DEFAULT_CACHE = ROOT / "results" / "deepgram_benchmark_targets.cache.json"
DEFAULT_AUDIO_DIR = ROOT / "data" / "audio_benchmark_targets"
DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "vi"


class CrawlConfigError(ValueError):
    """Raised for an unsafe or incomplete crawler configuration."""


@dataclass
class Target:
    """One deduplicated source URL, retaining all benchmark annotations."""

    url: str
    video_id: str
    video_ids: list[str]
    packs: list[str]
    annotation_ids: list[str]
    source_splits: list[str]
    start_times: list[float]
    end_times: list[float]
    media_info_paths: list[str]
    audio_backend: str
    audio_path: str | None = None
    status: str = "pending"
    resumed: bool = False
    last_error: str | None = None


def _read_dotenv(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_file = root / ".env"
    if not env_file.exists():
        return values
    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def require_deepgram_key(env: Mapping[str, str] | None = None, root: Path = ROOT) -> None:
    """Fail closed when the key is not configured, without exposing it."""

    supplied = env if env is not None else os.environ
    key = str(supplied.get("DEEPGRAM_API_KEY", "")).strip()
    if not key and env is None:
        key = _read_dotenv(root).get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise CrawlConfigError(
            "DEEPGRAM_API_KEY is required; set it in the environment or C:\\HCMAI\\.env"
        )


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CrawlConfigError(f"Annotation file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CrawlConfigError(f"Invalid JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise CrawlConfigError(f"Annotation line {line_number} is not an object")
        if str(row.get("question_type", "")).strip() == "spoken_fact":
            rows.append(row)
    return rows


def normalize_watch_url(url: str) -> str:
    """Canonicalize common YouTube URL variants for safe deduplication."""

    raw = str(url or "").strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        return raw
    parts = urlsplit(raw)
    host = parts.netloc.lower().split(":", 1)[0]
    if host in {"youtu.be", "www.youtu.be"}:
        video_key = parts.path.strip("/")
        return f"https://www.youtube.com/watch?v={video_key}" if video_key else raw
    if host.endswith("youtube.com"):
        video_key = parse_qs(parts.query).get("v", [""])[0]
        if video_key:
            return f"https://www.youtube.com/watch?v={video_key}"
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/"), parts.query, ""))


def _media_info_index(media_info_dirs: Sequence[Path]) -> dict[str, tuple[str, Path]]:
    index: dict[str, tuple[str, Path]] = {}
    for directory in media_info_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.json")):
            video_id = path.stem
            if video_id in index:
                continue
            try:
                data = _json_load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            url = normalize_watch_url(str(data.get("watch_url", "")))
            if url:
                index[video_id] = (url, path)
    return index


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_packs(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    values: list[str] = []
    if isinstance(value, str):
        values = value.split(",")
    else:
        for item in value:
            values.extend(str(item).split(","))
    packs = {item.strip().upper() for item in values if item.strip()}
    return packs or None


def build_targets(
    annotations: Sequence[Mapping[str, Any]],
    media_info_dirs: Sequence[Path],
    packs: set[str] | None = None,
    limit: int | None = None,
    video_ids: set[str] | None = None,
    audio_backend: str = "none",
    audio_dir: Path = DEFAULT_AUDIO_DIR,
) -> tuple[list[Target], dict[str, Any]]:
    if limit is not None and limit < 0:
        raise CrawlConfigError("--limit must be >= 0")
    if audio_backend not in {"none", "local_ffmpeg", "yt_dlp"}:
        raise CrawlConfigError(f"Unsupported audio backend: {audio_backend}")

    media = _media_info_index(media_info_dirs)
    grouped: dict[str, dict[str, Any]] = {}
    missing_media: list[str] = []
    filtered_rows = 0
    for row in annotations:
        video_id = str(row.get("video_id", "")).strip()
        if video_ids and video_id not in video_ids:
            continue
        pack = str(row.get("pack", "") or video_id.split("_", 1)[0]).upper()
        if packs and pack not in packs:
            continue
        filtered_rows += 1
        media_item = media.get(video_id)
        if media_item is None:
            missing_media.append(video_id)
            continue
        url, media_path = media_item
        item = grouped.setdefault(
            url,
            {
                "url": url,
                "video_ids": set(),
                "packs": set(),
                "annotation_ids": [],
                "source_splits": set(),
                "start_times": [],
                "end_times": [],
                "media_info_paths": set(),
            },
        )
        item["video_ids"].add(video_id)
        item["packs"].add(pack)
        if row.get("annotation_id") is not None:
            item["annotation_ids"].append(str(row["annotation_id"]))
        if row.get("source_split") is not None or row.get("split") is not None:
            item["source_splits"].add(str(row.get("source_split", row.get("split", ""))))
        start = _as_float(row.get("answer_start_time"))
        end = _as_float(row.get("answer_end_time"))
        if start is not None:
            item["start_times"].append(start)
        if end is not None:
            item["end_times"].append(end)
        item["media_info_paths"].add(str(media_path))

    ordered = [grouped[key] for key in sorted(grouped)]
    if limit is not None:
        ordered = ordered[:limit]

    targets: list[Target] = []
    for item in ordered:
        video_ids = sorted(item["video_ids"])
        packs_for_target = sorted(item["packs"])
        audio_path = None
        if audio_backend == "local_ffmpeg" and video_ids:
            audio_path = str(audio_dir / f"{video_ids[0]}.wav")
        targets.append(
            Target(
                url=item["url"],
                video_id=video_ids[0],
                video_ids=video_ids,
                packs=packs_for_target,
                annotation_ids=sorted(set(item["annotation_ids"])),
                source_splits=sorted(item["source_splits"]),
                start_times=sorted(item["start_times"]),
                end_times=sorted(item["end_times"]),
                media_info_paths=sorted(item["media_info_paths"]),
                audio_backend=audio_backend,
                audio_path=audio_path,
            )
        )

    family_counts = Counter(pack[:1] for target in targets for pack in target.packs if pack)
    pack_counts = Counter(pack for target in targets for pack in target.packs)
    summary = {
        "spoken_fact_rows_after_pack_filter": filtered_rows,
        "unique_target_urls": len(targets),
        "unique_target_videos": len({video_id for target in targets for video_id in target.video_ids}),
        "deduplicated_url_count": max(0, filtered_rows - len(targets) - len(missing_media)),
        "missing_media_info_video_ids": sorted(set(missing_media)),
        "target_packs": dict(sorted(pack_counts.items())),
        "target_k_count": int(family_counts.get("K", 0)),
        "target_l_count": int(family_counts.get("L", 0)),
        "audio_backend": audio_backend,
        "audio_extraction": "not_run",
    }
    return targets, summary


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = _json_load(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise CrawlConfigError(f"Invalid cache file {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def merge_resume(targets: Sequence[Target], cache: Mapping[str, Any]) -> list[Target]:
    previous = {
        normalize_watch_url(str(item.get("url", ""))): item
        for item in cache.get("targets", [])
        if isinstance(item, dict) and item.get("url")
    }
    merged: list[Target] = []
    for target in targets:
        old = previous.get(target.url)
        if old:
            target.status = str(old.get("status", target.status))
            target.audio_path = old.get("audio_path") or target.audio_path
            target.last_error = old.get("last_error")
            target.resumed = True
        merged.append(target)
    return merged


def _manifest(
    targets: Sequence[Target],
    summary: Mapping[str, Any],
    input_path: Path,
    media_info_dirs: Sequence[Path],
    audio_backend: str,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pack_coverage: dict[str, dict[str, Any]] = {}
    for target in targets:
        for pack in target.packs:
            item = pack_coverage.setdefault(pack, {"target_videos": 0, "target_urls": 0})
            item["target_videos"] += len([v for v in target.video_ids if v.startswith(pack + "_")])
            item["target_urls"] += 1
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "media_info_dirs": [str(path) for path in media_info_dirs],
        "deepgram": {
            "endpoint": DEEPGRAM_ENDPOINT,
            "model": DEEPGRAM_MODEL,
            "language": DEEPGRAM_LANGUAGE,
            "key_configured": True,
            "api_calls_made": 0,
        },
        "audio": {
            "backend": audio_backend,
            "extraction": "not_run",
            "note": "Manifest preparation only; Deepgram and source downloads were not called.",
        },
        "selection": dict(selection or {}),
        "coverage_by_pack": pack_coverage,
        "summary": dict(summary),
        "targets": [asdict(target) for target in targets],
    }


def _atomic_json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def extract_audio_local_ffmpeg(
    video_path: Path,
    audio_path: Path,
    ffmpeg_binary: str = "ffmpeg",
) -> None:
    """Extract mono 16 kHz WAV from an already-local video; no network."""

    if not video_path.exists():
        raise CrawlConfigError(f"Local video not found: {video_path}")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise CrawlConfigError(f"Audio backend local_ffmpeg unavailable: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()[-500:]
        raise CrawlConfigError(f"ffmpeg audio extraction failed: {detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--media-info-dir", action="append", type=Path, dest="media_info_dirs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--audio-backend", choices=("none", "local_ffmpeg", "yt_dlp"), default="none")
    parser.add_argument("--packs", help="Comma-separated pack filter, e.g. K01,K07,L25")
    parser.add_argument("--pack", action="append", default=None,
                        help="Pack filter alias; repeat or comma-separate")
    parser.add_argument("--video-id", action="append", default=None,
                        help="Exact video filter; repeat for multiple videos")
    parser.add_argument("--limit", type=int, help="Maximum unique target URLs after deduplication")
    parser.add_argument("--video-limit", type=int, default=None,
                        help="Maximum selected target URLs after deterministic sorting")
    parser.add_argument("--resume", action="store_true",
                        help="Resume/update the existing manifest and cache")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing manifest/cache")
    parser.add_argument("--dry-run", action="store_true", help="Print summary; do not write cache/output or call network")
    parser.add_argument("--strict", action="store_true", help="Fail if a spoken_fact row has no media-info watch_url")
    return parser


def run(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    require_deepgram_key(env=env)
    media_dirs = tuple(args.media_info_dirs or DEFAULT_MEDIA_INFO_DIRS)
    annotations = load_annotations(args.input)
    requested_packs = getattr(args, "pack", None) or getattr(args, "packs", None)
    requested_limit = getattr(args, "video_limit", None)
    if requested_limit is None:
        requested_limit = getattr(args, "limit", None)
    requested_video_ids = {
        str(value).strip() for value in (getattr(args, "video_id", None) or []) if str(value).strip()
    }
    targets, summary = build_targets(
        annotations,
        media_dirs,
        packs=parse_packs(requested_packs),
        limit=requested_limit,
        video_ids=requested_video_ids or None,
        audio_backend=args.audio_backend,
        audio_dir=args.audio_dir,
    )
    if args.strict and summary["missing_media_info_video_ids"]:
        missing = ", ".join(summary["missing_media_info_video_ids"])
        raise CrawlConfigError(f"Missing media-info watch_url for: {missing}")
    cache = _load_cache(args.cache)
    targets = merge_resume(targets, cache)
    selected_video_ids = sorted({video_id for target in targets for video_id in target.video_ids})
    selection = {
        "requested_packs": sorted(parse_packs(requested_packs) or []),
        "requested_video_ids": sorted(requested_video_ids),
        "video_limit": requested_limit,
        "selected_video_ids": selected_video_ids,
        "selected_packs": sorted({pack for target in targets for pack in target.packs}),
        "resume": bool(getattr(args, "resume", False)),
        "overwrite": bool(getattr(args, "overwrite", False)),
    }
    if not args.dry_run:
        exists = args.output.exists() or args.cache.exists()
        if exists and not (selection["resume"] or selection["overwrite"]):
            raise CrawlConfigError(
                "refusing to overwrite ASR preparation artifacts; pass --resume or --overwrite"
            )
        if selection["resume"] and args.output.exists():
            previous = _json_load(args.output)
            previous_selection = previous.get("selection", {}) if isinstance(previous, dict) else {}
            previous_ids = sorted(previous_selection.get("selected_video_ids", []))
            if previous_ids and previous_ids != selected_video_ids:
                raise CrawlConfigError("--resume scope mismatch: selected video IDs differ")
    manifest = _manifest(
        targets, summary, args.input, media_dirs, args.audio_backend, selection=selection
    )
    if not args.dry_run:
        _atomic_json_dump(args.cache, manifest)
        _atomic_json_dump(args.output, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = run(args)
    except CrawlConfigError as exc:
        parser.error(str(exc))
    summary = manifest["summary"]
    print(json.dumps({
        "dry_run": bool(args.dry_run),
        "target_urls": summary["unique_target_urls"],
        "target_k": summary["target_k_count"],
        "target_l": summary["target_l_count"],
        "packs": summary["target_packs"],
        "missing_media_info": len(summary["missing_media_info_video_ids"]),
        "api_calls_made": manifest["deepgram"]["api_calls_made"],
        "output": None if args.dry_run else str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
