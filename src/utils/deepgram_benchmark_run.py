"""Execute a bounded Deepgram comparison run for prepared benchmark targets.

This is deliberately opt-in and resumable.  It downloads audio-only streams
with yt-dlp, uploads them to Deepgram, and stores timestamped utterances.  It
never writes the API key to disk or stdout.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.utils.deepgram_benchmark_crawl import (
    DEEPGRAM_ENDPOINT,
    DEEPGRAM_LANGUAGE,
    DEEPGRAM_MODEL,
)
from src.utils.paths import load_runtime_env


ROOT = Path(__file__).resolve().parents[2]


def _key() -> str:
    value = load_runtime_env(ROOT / ".env").get("DEEPGRAM_API_KEY", "").strip()
    if not value:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured")
    return value


def _download_audio(url: str, output_dir: Path, video_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{video_id}.%(ext)s"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--format",
        "bestaudio/best",
        "--js-runtimes",
        "node",
        "--extractor-args",
        "youtube:player_client=web_safari,android",
        "--output",
        str(prefix),
        url,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "yt-dlp failed").strip()[-800:]
        raise RuntimeError(f"audio download failed for {video_id}: {detail}")
    candidates = sorted(output_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [path for path in candidates if path.suffix.lower() not in {".part", ".ytdl"}]
    if not candidates:
        raise RuntimeError(f"yt-dlp produced no audio file for {video_id}")
    return candidates[0]


def _transcribe(audio_path: Path, key: str) -> dict[str, Any]:
    params = urlencode({
        "model": DEEPGRAM_MODEL,
        "language": DEEPGRAM_LANGUAGE,
        "smart_format": "true",
        "utterances": "true",
        "punctuate": "true",
    })
    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    request = Request(
        f"{DEEPGRAM_ENDPOINT}?{params}",
        data=audio_path.read_bytes(),
        headers={"Authorization": f"Token {key}", "Content-Type": content_type},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    utterances = payload.get("results", {}).get("utterances") or []
    chunks = [
        {
            "start": float(item.get("start", 0.0)),
            "end": float(item.get("end", item.get("start", 0.0))),
            "chunk": str(item.get("transcript", "")).strip(),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        }
        for item in utterances
        if str(item.get("transcript", "")).strip()
    ]
    if not chunks:
        alternative = (payload.get("results", {}).get("channels") or [{}])[0].get("alternatives", [{}])[0]
        transcript = str(alternative.get("transcript", "")).strip()
        if transcript:
            chunks = [{"start": 0.0, "end": 0.0, "chunk": transcript, "confidence": float(alternative.get("confidence", 0.0) or 0.0)}]
    return {"chunks": chunks, "model": DEEPGRAM_MODEL, "language": DEEPGRAM_LANGUAGE}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/deepgram_benchmark_targets.json")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "data/audio_benchmark_targets")
    parser.add_argument("--output", type=Path, default=ROOT / "results/deepgram_benchmark_results.json")
    parser.add_argument("--max-calls", type=int, default=15)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.max_calls < 1:
        raise ValueError("--max-calls must be positive")
    key = _key()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        results = previous.get("results", {}) if isinstance(previous, dict) else {}
    calls = 0
    for target in manifest.get("targets", []):
        video_id = str(target.get("video_id", "")).strip()
        url = str(target.get("url", "")).strip()
        if not video_id or not url:
            continue
        if video_id in results and (results[video_id].get("status") != "error" or not args.retry_errors):
            continue
        if calls >= args.max_calls:
            break
        try:
            audio_path = _download_audio(url, args.audio_dir, video_id)
            transcription = _transcribe(audio_path, key)
            results[video_id] = {
                "status": "completed",
                "video_id": video_id,
                "url": url,
                "audio_path": str(audio_path),
                "packs": target.get("packs", []),
                "annotation_ids": target.get("annotation_ids", []),
                "start_times": target.get("start_times", []),
                "end_times": target.get("end_times", []),
                **transcription,
            }
            calls += 1
        except Exception as exc:
            results[video_id] = {
                "status": "error",
                "video_id": video_id,
                "url": url,
                "packs": target.get("packs", []),
                "error": f"{type(exc).__name__}: {exc}",
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mktemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=str(args.output.parent)))
        temporary.write_text(json.dumps({"api_calls_made": len(results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps({"api_calls_made_this_run": calls, "total_results": len(results), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
