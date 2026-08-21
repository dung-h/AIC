"""Resumable ASR materializer for the AIC K01--K20 and L21--L30 corpus.

This module is deliberately isolated from the production router and from the
historical ASR artifacts.  It materializes a new, auditable index under an
explicit output directory:

* video ZIP archives are inspected without mutating them;
* one pack can be resumed from per-video raw Deepgram JSON;
* audio is derived with a repository-local ``.venv/bin/ffmpeg`` when present;
* Deepgram is reachable only with ``--execute --allow-network --confirm-api``;
* timestamped chunks are mapped to canonical ``frame_idx`` values;
* bge-m3 embeddings are written beside the per-pack parquet files;
* incomplete runs are marked partial and are never promoted implicitly.

The command defaults to a no-network dry-run.  The runner accepts injected
transcribers, embedders, and ffmpeg functions so tests can exercise the whole
flow without a real API call or a model download.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZipInfo

import numpy as np
import pandas as pd

from src.reranking.asr_index import normalize_transcript


SCHEMA_VERSION = "hcmai.asr_global_v2"
PROTOCOL_VERSION = "ASR global v2"
CHUNKING_POLICY_VERSION = "deepgram_word_window_v1"
SUPPORTED_PACKS = tuple([f"K{number:02d}" for number in range(1, 21)] + [f"L{number:02d}" for number in range(21, 31)])
VIDEO_RE = re.compile(r"^(?P<pack>[KL]\d{2})_V(?P<number>\d{3})$", re.IGNORECASE)
ZIP_RE = re.compile(r"^Videos_(?P<pack>[KL]\d{2})(?:_[A-Za-z0-9]+)?\.zip$", re.IGNORECASE)
REQUIRED_CANONICAL_COLUMNS = {"video_id", "kf_n", "frame_idx", "pts_time"}
REQUIRED_CHUNK_COLUMNS = (
    "video_id",
    "chunk_index",
    "text",
    "start",
    "end",
    "kf_n",
    "frame_idx",
    "pts_time",
    "distance_seconds",
)


class ASRGlobalV2Error(RuntimeError):
    """A fail-closed materialization or schema error."""


class ASRNetworkApprovalError(ASRGlobalV2Error):
    """The caller did not explicitly approve a provider call."""


@dataclass(frozen=True)
class CanonicalFrame:
    video_id: str
    kf_n: int
    frame_idx: int
    pts_time: float


@dataclass(frozen=True)
class PackArchive:
    pack: str
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class RunnerConfig:
    archive_root: Path
    canonical_path: Path
    output_dir: Path
    work_dir: Path
    raw_dir: Path
    model: str = "models/bge-m3"
    ffmpeg_path: Path | None = None
    batch_size: int = 32
    execute: bool = False
    allow_network: bool = False
    confirm_api: bool = False
    metadata_only: bool = False
    max_videos: int | None = None
    timeout_seconds: int = 600
    api_model: str = "nova-3"
    language: str = "vi"
    env_path: Path | None = None
    resume: bool = False
    raw_only: bool = False

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ASRGlobalV2Error("batch_size must be positive")
        if self.max_videos is not None and self.max_videos < 1:
            raise ASRGlobalV2Error("max_videos must be positive when provided")
        if self.execute and not self.raw_only and not (self.allow_network and self.confirm_api):
            raise ASRNetworkApprovalError(
                "real ASR execution requires --execute --allow-network --confirm-api"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_pack(value: str) -> str:
    pack = str(value).strip().upper()
    if pack not in SUPPORTED_PACKS:
        raise ASRGlobalV2Error(
            f"unsupported ASR global v2 pack: {value!r}; expected K01-K20 or L21-L30"
        )
    return pack


def _video_id_from_member(name: str) -> str | None:
    stem = Path(name.replace("\\", "/")).name.rsplit(".", 1)[0]
    match = VIDEO_RE.fullmatch(stem)
    return stem.upper() if match else None


def discover_pack_archives(archive_root: Path, packs: Sequence[str] = SUPPORTED_PACKS) -> dict[str, PackArchive]:
    """Discover L-series video ZIPs, including all five L26 parts."""
    requested = tuple(_normalise_pack(pack) for pack in packs)
    discovered: dict[str, list[Path]] = {pack: [] for pack in requested}
    if not archive_root.is_dir():
        raise ASRGlobalV2Error(f"archive root does not exist: {archive_root}")
    for path in sorted(archive_root.glob("Videos_*.zip")):
        match = ZIP_RE.fullmatch(path.name)
        if not match:
            continue
        pack = match.group("pack").upper()
        if pack in discovered:
            discovered[pack].append(path)
    missing = [pack for pack, paths in discovered.items() if not paths]
    if missing:
        raise ASRGlobalV2Error(f"missing video ZIP archive(s): {missing}")
    return {pack: PackArchive(pack, tuple(paths)) for pack, paths in discovered.items()}


def _canonical_map_from_frame(canonical: pd.DataFrame) -> dict[str, tuple[CanonicalFrame, ...]]:
    missing = sorted(REQUIRED_CANONICAL_COLUMNS - set(canonical.columns))
    if missing:
        raise ASRGlobalV2Error(f"canonical map missing columns: {missing}")
    result: dict[str, list[CanonicalFrame]] = {}
    seen: set[tuple[str, int]] = set()
    for row_number, row in canonical.reset_index(drop=True).iterrows():
        video_id = str(row["video_id"]).strip().upper()
        if VIDEO_RE.fullmatch(video_id) is None:
            raise ASRGlobalV2Error(f"invalid canonical video_id at row {row_number}: {video_id!r}")
        try:
            kf_float = float(row["kf_n"])
            frame_float = float(row["frame_idx"])
            pts = float(row["pts_time"])
        except (TypeError, ValueError) as exc:
            raise ASRGlobalV2Error(f"non-numeric canonical row {row_number}") from exc
        if not all(math.isfinite(value) for value in (kf_float, frame_float, pts)):
            raise ASRGlobalV2Error(f"non-finite canonical row {row_number}")
        if kf_float < 0 or frame_float < 0 or pts < 0 or not kf_float.is_integer() or not frame_float.is_integer():
            raise ASRGlobalV2Error(f"invalid canonical coordinates at row {row_number}")
        key = (video_id, int(kf_float))
        if key in seen:
            raise ASRGlobalV2Error(f"duplicate canonical (video_id,kf_n): {key}")
        seen.add(key)
        result.setdefault(video_id, []).append(
            CanonicalFrame(video_id, int(kf_float), int(frame_float), pts)
        )
    return {
        video_id: tuple(sorted(frames, key=lambda item: (item.pts_time, item.kf_n, item.frame_idx)))
        for video_id, frames in result.items()
    }


def load_canonical_map(path: Path) -> dict[str, tuple[CanonicalFrame, ...]]:
    if not path.is_file():
        raise ASRGlobalV2Error(f"canonical map does not exist: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - exact parquet backend varies
        raise ASRGlobalV2Error(f"cannot read canonical map {path}: {exc}") from exc
    return _canonical_map_from_frame(frame)


def inspect_archive_videos(archive: PackArchive) -> dict[str, str]:
    """Return video_id -> source ZIP member without extracting anything."""
    members: dict[str, str] = {}
    for path in archive.paths:
        try:
            with ZipFile(path) as zipped:
                for info in zipped.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                        continue
                    video_id = _video_id_from_member(info.filename)
                    if video_id is None or not video_id.startswith(archive.pack + "_"):
                        raise ASRGlobalV2Error(
                            f"invalid {archive.pack} video member in {path.name}: {info.filename!r}"
                        )
                    if video_id in members:
                        raise ASRGlobalV2Error(
                            f"duplicate video {video_id} across ZIPs for {archive.pack}"
                        )
                    members[video_id] = f"{path.name}:{info.filename}"
        except ASRGlobalV2Error:
            raise
        except Exception as exc:
            raise ASRGlobalV2Error(f"cannot inspect ZIP {path}: {exc}") from exc
    if not members:
        raise ASRGlobalV2Error(f"no MP4 members found for {archive.pack}")
    return dict(sorted(members.items()))


def load_deepgram_key(env_path: Path | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Read the key without logging it; environment wins over .env files."""
    env = os.environ if environ is None else environ
    direct = str(env.get("DEEPGRAM_API_KEY", "")).strip()
    if direct:
        return direct
    candidates = [env_path] if env_path is not None else []
    candidates.extend([Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"])
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None or candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "DEEPGRAM_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return ""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalise_text(value: Any) -> str:
    return normalize_transcript(str(value or "").replace("\x00", " "))


def _chunk_from_item(item: Mapping[str, Any]) -> dict[str, Any] | None:
    text = _normalise_text(
        item.get("transcript") or item.get("text") or item.get("punctuated_transcript")
    )
    start = _number(item.get("start"))
    end = _number(item.get("end"))
    if not text or start is None or end is None or start < 0 or end < start:
        return None
    return {"text": text, "start": start, "end": end}


def _word_timestamp_chunks(
    words: Sequence[Any],
    *,
    target_seconds: float = 8.0,
    min_seconds: float = 6.0,
    overlap_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Build short, overlapping text windows from Deepgram word timestamps.

    Paragraphs returned by Deepgram are useful for display but can span several
    minutes.  Mapping a paragraph midpoint to one canonical keyframe destroys
    temporal retrieval precision.  Word timestamps are already present in the
    raw provider response, so use them whenever they are valid and retain the
    paragraph/utterance paths only as a compatibility fallback.

    The window is bounded at ``target_seconds`` and advances with a small
    overlap.  This makes a spoken fact near either boundary retrievable without
    duplicating the full transcript in every chunk.
    """
    if target_seconds <= 0 or min_seconds < 0 or min_seconds > target_seconds:
        raise ASRGlobalV2Error("invalid word timestamp chunking configuration")
    if overlap_seconds < 0 or overlap_seconds >= target_seconds:
        raise ASRGlobalV2Error("word timestamp overlap must be in [0, target_seconds)")

    tokens: list[dict[str, Any]] = []
    for item in words:
        if not isinstance(item, Mapping):
            continue
        text = _normalise_text(item.get("punctuated_word") or item.get("word") or "")
        start = _number(item.get("start"))
        end = _number(item.get("end"))
        if not text or start is None or end is None or start < 0 or end < start:
            continue
        tokens.append({"text": text, "start": start, "end": end})
    tokens.sort(key=lambda item: (item["start"], item["end"], item["text"]))
    if not tokens:
        return []

    chunks: list[dict[str, Any]] = []
    first = 0
    while first < len(tokens):
        start = tokens[first]["start"]
        last = first
        while last + 1 < len(tokens):
            candidate = last + 1
            duration = tokens[candidate]["end"] - start
            gap = tokens[candidate]["start"] - tokens[last]["end"]
            # Do not bridge programme cuts/silence or let a single sparse
            # timestamp expand the anchor window beyond the declared budget.
            if duration > target_seconds or gap > overlap_seconds:
                break
            last = candidate
            # Prefer a natural sentence boundary once the chunk is long enough;
            # otherwise cut at the bounded target duration.
            if duration >= min_seconds and tokens[candidate]["text"].rstrip().endswith((".", "!", "?", "…")):
                break
            if duration >= target_seconds:
                break

        selected = tokens[first : last + 1]
        text = _normalise_text(" ".join(token["text"] for token in selected))
        if text:
            chunks.append({"text": text, "start": start, "end": selected[-1]["end"]})
        if last >= len(tokens) - 1:
            break

        next_start_time = selected[-1]["end"] - overlap_seconds
        next_first = first + 1
        while next_first <= last and tokens[next_first]["start"] < next_start_time:
            next_first += 1
        # A very short final window may otherwise make no progress.
        first = max(first + 1, next_first)
    return chunks


def iter_timestamped_chunks(payload: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Extract deterministic, timestamped Deepgram chunks at retrieval resolution."""
    results = payload.get("results") if isinstance(payload, Mapping) else None
    results = results if isinstance(results, Mapping) else {}
    candidates: list[dict[str, Any]] = []
    channels = results.get("channels") or []
    channel = channels[0] if channels and isinstance(channels[0], Mapping) else {}
    alternatives = channel.get("alternatives") or []
    alternative = alternatives[0] if alternatives and isinstance(alternatives[0], Mapping) else {}

    # Prefer the most granular trustworthy representation.  Deepgram's
    # paragraphs can be hundreds of seconds long even though ``words`` has
    # exact timings, as observed in the L27 spoken-fact regression case.
    candidates.extend(_word_timestamp_chunks(alternative.get("words") or []))
    if not candidates:
        utterances = results.get("utterances") or []
        for item in utterances:
            if isinstance(item, Mapping):
                chunk = _chunk_from_item(item)
                if chunk is not None:
                    candidates.append(chunk)
    if not candidates:
        paragraphs = alternative.get("paragraphs") or {}
        for paragraph in paragraphs.get("paragraphs") or []:
            if not isinstance(paragraph, Mapping):
                continue
            for item in paragraph.get("sentences") or []:
                if isinstance(item, Mapping):
                    chunk = _chunk_from_item(item)
                    if chunk is not None:
                        candidates.append(chunk)
    candidates.sort(key=lambda item: (item["start"], item["end"], item["text"]))
    previous: tuple[float, float, str] | None = None
    for chunk in candidates:
        identity = (chunk["start"], chunk["end"], chunk["text"])
        if identity == previous:
            continue
        previous = identity
        yield chunk


def map_chunk_to_canonical(
    video_id: str,
    chunk: Mapping[str, Any],
    canonical: Mapping[str, Sequence[CanonicalFrame]],
) -> dict[str, Any]:
    video = str(video_id).upper()
    frames = canonical.get(video)
    if not frames:
        raise ASRGlobalV2Error(f"ASR video absent from canonical map: {video}")
    start = _number(chunk.get("start"))
    end = _number(chunk.get("end"))
    text = _normalise_text(chunk.get("text"))
    if start is None or end is None or end < start or not text:
        raise ASRGlobalV2Error(f"invalid ASR chunk for {video}: {chunk!r}")
    midpoint = (start + end) / 2.0
    nearest = min(frames, key=lambda frame: (abs(frame.pts_time - midpoint), frame.kf_n, frame.frame_idx))
    return {
        "video_id": video,
        "chunk_index": 0,
        "text": text,
        "start": start,
        "end": end,
        "kf_n": nearest.kf_n,
        "frame_idx": nearest.frame_idx,
        "pts_time": nearest.pts_time,
        "distance_seconds": abs(nearest.pts_time - midpoint),
    }


def validate_chunk_frame(frame: pd.DataFrame, canonical: Mapping[str, Sequence[CanonicalFrame]]) -> None:
    missing = sorted(set(REQUIRED_CHUNK_COLUMNS) - set(frame.columns))
    if missing:
        raise ASRGlobalV2Error(f"materialized ASR chunks missing columns: {missing}")
    canonical_keys = {
        (item.video_id, item.kf_n, item.frame_idx)
        for frames in canonical.values()
        for item in frames
    }
    for row_number, row in frame.reset_index(drop=True).iterrows():
        key = (str(row["video_id"]).upper(), int(row["kf_n"]), int(row["frame_idx"]))
        if key not in canonical_keys:
            raise ASRGlobalV2Error(f"chunk row {row_number} is not canonical: {key}")
        if not _normalise_text(row["text"]):
            raise ASRGlobalV2Error(f"chunk row {row_number} has empty text")
        start = _number(row["start"])
        end = _number(row["end"])
        if start is None or end is None or start < 0 or end < start:
            raise ASRGlobalV2Error(f"chunk row {row_number} has invalid timestamps")


def _extract_member(zipped: ZipFile, info: ZipInfo, destination: Path) -> None:
    member_path = PurePosixPath(info.filename.replace("\\", "/"))
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ASRGlobalV2Error(f"unsafe ZIP member path: {info.filename!r}")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = destination.parent.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ASRGlobalV2Error(f"unsafe extraction destination: {destination}")
    temporary = destination.with_name(f".{destination.name}.part")
    with zipped.open(info, "r") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, destination)


def _extract_video(archive: PackArchive, member: str, video_id: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    member_path = member.split(":", 1)[1]
    archive_path = next(path for path in archive.paths if path.name == member.split(":", 1)[0])
    with ZipFile(archive_path) as zipped:
        info = zipped.getinfo(member_path)
        _extract_member(zipped, info, destination)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise ASRGlobalV2Error(f"ZIP extraction produced no MP4 for {video_id}")


def resolve_ffmpeg(path: Path | None = None) -> Path:
    candidates = [path] if path is not None else []
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / ".venv" / "bin" / "ffmpeg",
            Path(shutil.which("ffmpeg")) if shutil.which("ffmpeg") else None,
        ]
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise ASRGlobalV2Error("ffmpeg not found; pass --ffmpeg or install .venv/bin/ffmpeg")


def extract_audio_default(mp4: Path, wav: Path, ffmpeg_path: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    if wav.is_file() and wav.stat().st_size > 44:
        return
    # Keep the codec suffix on the temporary path.  ffmpeg infers the output
    # muxer from the filename unless ``-f`` is supplied; a name ending in
    # ``.wav.part`` therefore fails with "Unable to choose an output format".
    # The final destination remains the stable ``*.wav`` path used by the
    # resumable runner.
    temporary = wav.with_name(f".{wav.stem}.part.wav")
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(mp4),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()[-1000:]
        raise ASRGlobalV2Error(f"ffmpeg failed for {mp4.name}: {detail}")
    if not temporary.is_file() or temporary.stat().st_size <= 44:
        raise ASRGlobalV2Error(f"ffmpeg produced an empty WAV for {mp4.name}")
    os.replace(temporary, wav)


class DeepgramTranscriber:
    """Small stdlib client; construction is only allowed after approval."""

    def __init__(self, api_key: str, model: str = "nova-3", language: str = "vi", timeout: int = 600):
        if not api_key.strip():
            raise ASRNetworkApprovalError("Deepgram API key is not configured")
        self.api_key = api_key
        self.model = model
        self.language = language
        self.timeout = timeout
        self.calls = 0

    def transcribe(self, wav_path: Path) -> Mapping[str, Any]:
        query = urlencode(
            {
                "model": self.model,
                "language": self.language,
                "punctuate": "true",
                "smart_format": "true",
                "utterances": "true",
            }
        )
        request = Request(
            f"https://api.deepgram.com/v1/listen?{query}",
            data=wav_path.read_bytes(),
            method="POST",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
        )
        self.calls += 1
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, Mapping):
            raise ASRGlobalV2Error("Deepgram returned a non-object payload")
        return payload


def _write_raw_response(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(path, payload)


def _load_raw_response(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ASRGlobalV2Error(f"invalid existing Deepgram JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ASRGlobalV2Error(f"existing Deepgram JSON is not an object: {path}")
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise ASRGlobalV2Error(f"existing Deepgram JSON has no results object: {path}")
    # Deepgram can validly return a results/channels envelope with zero
    # utterances for a video that contains no speech.  That is a completed
    # no-speech observation, not a malformed provider response.
    if "channels" not in results and "utterances" not in results:
        raise ASRGlobalV2Error(f"existing Deepgram JSON has no channel/utterance result: {path}")
    return payload


def _quarantine_invalid_raw(path: Path) -> Path:
    """Preserve an invalid provider payload before a resumable retry."""
    candidate = path.with_name(path.name + ".invalid")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".invalid.{suffix}")
        suffix += 1
    shutil.copy2(path, candidate)
    return candidate


def _embed_frame(frame: pd.DataFrame, model: str, batch_size: int, embedder: Any | None) -> np.ndarray:
    provider = embedder
    if provider is None:
        from src.core.offline_fallback import TextEmbedderOffline

        provider = TextEmbedderOffline(model_name=model)
    values = provider.embed(frame["text"].tolist(), batch_size=batch_size, normalize=True)
    embeddings = np.asarray(values, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(frame) or embeddings.shape[1] < 1:
        raise ASRGlobalV2Error(
            f"embedding shape mismatch: got {embeddings.shape}, expected ({len(frame)}, D)"
        )
    if not np.isfinite(embeddings).all():
        raise ASRGlobalV2Error("embedding contains non-finite values")
    return embeddings


def _pack_manifest_path(output_dir: Path, pack: str) -> Path:
    return output_dir / f"asr_global_v2_{pack.lower()}_manifest.json"


class ASRGlobalV2Runner:
    def __init__(
        self,
        config: RunnerConfig,
        *,
        transcriber: Any | None = None,
        embedder: Any | None = None,
        ffmpeg_runner: Callable[[Path, Path], None] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.transcriber = transcriber
        self.embedder = embedder
        # Keep one local bge model alive for the whole run.  Constructing a
        # fresh SentenceTransformer for every pack repeatedly allocates CUDA
        # weights and can fragment VRAM during a long global materialization.
        self._embedder_instance = embedder
        self.ffmpeg_runner = ffmpeg_runner
        self._canonical: dict[str, tuple[CanonicalFrame, ...]] | None = None
        self._archives: dict[str, PackArchive] | None = None
        self._archive_members: dict[str, dict[str, str]] = {}
        self._network_calls = 0

    @property
    def canonical(self) -> dict[str, tuple[CanonicalFrame, ...]]:
        if self._canonical is None:
            self._canonical = load_canonical_map(self.config.canonical_path)
        return self._canonical

    def preflight(self, packs: Sequence[str]) -> dict[str, Any]:
        requested = tuple(_normalise_pack(pack) for pack in packs)
        if self.config.raw_only:
            pack_reports: dict[str, Any] = {}
            for pack in requested:
                canonical_videos = sorted(
                    video for video in self.canonical if video.startswith(pack + "_")
                )
                raw_files = {
                    video: self.config.raw_dir / pack.lower() / f"{video}.json"
                    for video in canonical_videos
                }
                missing_raw = [video for video, path in raw_files.items() if not path.is_file()]
                if missing_raw:
                    raise ASRGlobalV2Error(
                        f"raw-only ASR materialization missing raw JSON for {pack}: "
                        f"{missing_raw[:5]}"
                    )
                pack_reports[pack] = {
                    "archive_paths": [],
                    "archive_video_count": 0,
                    "canonical_video_count": len(canonical_videos),
                    "video_ids": canonical_videos,
                    "raw_video_count": len(raw_files),
                    "status": "ready",
                }
            return {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL_VERSION,
                "selected_packs": list(requested),
                "canonical_path": str(self.config.canonical_path),
                "canonical_sha256": _sha256(self.config.canonical_path),
                "archive_root": None,
                "raw_dir": str(self.config.raw_dir),
                "raw_only": True,
                "packs": pack_reports,
                "deepgram_api_key_configured": bool(load_deepgram_key(self.config.env_path)),
                "network_approved": False,
                "dry_run": not self.config.execute,
            }
        self._archives = discover_pack_archives(self.config.archive_root, requested)
        pack_reports: dict[str, Any] = {}
        for pack, archive in self._archives.items():
            members = inspect_archive_videos(archive)
            self._archive_members[pack] = members
            archive_videos = set(members)
            canonical_videos = {video for video in self.canonical if video.startswith(pack + "_")}
            missing_canonical = sorted(archive_videos - canonical_videos)
            missing_archive = sorted(canonical_videos - archive_videos)
            if missing_canonical or missing_archive:
                raise ASRGlobalV2Error(
                    f"canonical/archive mismatch for {pack}: "
                    f"missing_canonical={missing_canonical[:5]}, missing_archive={missing_archive[:5]}"
                )
            pack_reports[pack] = {
                "archive_paths": [str(path) for path in archive.paths],
                "archive_video_count": len(archive_videos),
                "canonical_video_count": len(canonical_videos),
                "video_ids": sorted(archive_videos),
                "status": "ready",
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_VERSION,
            "selected_packs": list(requested),
            "canonical_path": str(self.config.canonical_path),
            "canonical_sha256": _sha256(self.config.canonical_path),
            "archive_root": str(self.config.archive_root),
            "packs": pack_reports,
            "deepgram_api_key_configured": bool(load_deepgram_key(self.config.env_path)),
            "network_approved": bool(
                not self.config.raw_only
                and self.config.execute
                and self.config.allow_network
                and self.config.confirm_api
            ),
            "dry_run": not self.config.execute,
        }

    def _get_transcriber(self) -> Any:
        if self.transcriber is not None:
            return self.transcriber
        if not self.config.execute:
            raise ASRNetworkApprovalError("dry-run cannot create a Deepgram client")
        if not (self.config.allow_network and self.config.confirm_api):
            raise ASRNetworkApprovalError(
                "Deepgram client creation blocked; pass all explicit network approval flags"
            )
        return DeepgramTranscriber(
            load_deepgram_key(self.config.env_path),
            model=self.config.api_model,
            language=self.config.language,
            timeout=self.config.timeout_seconds,
        )

    def _get_embedder(self) -> Any:
        if self._embedder_instance is None:
            from src.core.offline_fallback import TextEmbedderOffline

            self._embedder_instance = TextEmbedderOffline(model_name=self.config.model)
        return self._embedder_instance

    def _manifest_base(self, selected_packs: Sequence[str], mode: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_VERSION,
            "created_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
            "mode": mode,
            "resume": bool(self.config.resume),
            "status": "running",
            "selected_packs": list(selected_packs),
            "expected_packs": list(SUPPORTED_PACKS),
            "canonical_path": str(self.config.canonical_path),
            "canonical_sha256": _sha256(self.config.canonical_path),
            "archive_root": str(self.config.archive_root),
            "output_dir": str(self.config.output_dir),
            "work_dir": str(self.config.work_dir),
            "raw_dir": str(self.config.raw_dir),
            "deepgram": {
                "provider": "deepgram",
                "model": self.config.api_model,
                "language": self.config.language,
                "api_key_configured": bool(load_deepgram_key(self.config.env_path)),
                "network_approved": bool(
                    not self.config.raw_only
                    and self.config.execute
                    and self.config.allow_network
                    and self.config.confirm_api
                ),
                "network_calls": 0,
            },
            "embedding": {
                "provider": "local_bge_m3",
                "model": self.config.model,
                "materialized": False,
                "dimension": None,
            },
            "chunking": {
                "policy": CHUNKING_POLICY_VERSION,
                "source_priority": ["word_timestamps", "utterances", "paragraph_sentences"],
                "target_seconds": 8.0,
                "overlap_seconds": 2.0,
            },
            "packs": {},
            "ready_for_global": False,
            "ready_for_production": False,
        }

    def _write_global_manifest(self, manifest: Mapping[str, Any]) -> None:
        _atomic_json(self.config.output_dir / "asr_global_v2_manifest.json", manifest)

    def _dry_run(self, packs: Sequence[str]) -> dict[str, Any]:
        preflight = self.preflight(packs)
        manifest = self._manifest_base(packs, "dry_run")
        manifest["status"] = "dry_run"
        manifest["preflight"] = preflight
        manifest["packs"] = preflight["packs"]
        manifest["notes"] = [
            "No ZIP was extracted.",
            "No MP4 was modified or deleted.",
            "No audio was generated.",
            "No Deepgram/API call was made.",
            "No bge-m3 model was loaded.",
        ]
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_global_manifest(manifest)
        return manifest

    def _run_one_video(
        self,
        pack: str,
        archive: PackArchive,
        member: str,
        video_id: str,
        transcriber: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        media_path = self.config.work_dir / pack.lower() / "media" / f"{video_id}.mp4"
        wav_path = self.config.work_dir / pack.lower() / "audio" / f"{video_id}.wav"
        raw_path = self.config.raw_dir / pack.lower() / f"{video_id}.json"
        payload: Mapping[str, Any] | None = None
        if raw_path.is_file():
            try:
                payload = _load_raw_response(raw_path)
            except ASRGlobalV2Error:
                # A provider error response can be valid JSON but contain no
                # timestamped chunks.  Keep it for audit and retry only this
                # video instead of poisoning the entire resumable pack.
                _quarantine_invalid_raw(raw_path)
        if payload is None:
            if self.config.raw_only:
                raise ASRGlobalV2Error(
                    f"raw-only ASR materialization cannot recover missing/invalid raw response: {raw_path}"
                )
            _extract_video(archive, member, video_id, media_path)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            if self.ffmpeg_runner is None:
                ffmpeg_path = resolve_ffmpeg(self.config.ffmpeg_path)
                extract_audio_default(media_path, wav_path, ffmpeg_path)
            else:
                self.ffmpeg_runner(media_path, wav_path)
            if not self.config.execute and self.transcriber is None:
                raise ASRNetworkApprovalError("dry-run must not transcribe a video")
            method = getattr(transcriber, "transcribe", None)
            if not callable(method) and callable(transcriber):
                method = transcriber
            if not callable(method):
                raise ASRGlobalV2Error("transcriber must expose transcribe(path) or be callable")
            payload = method(wav_path)
            if not isinstance(payload, Mapping):
                raise ASRGlobalV2Error(f"transcriber returned a non-object for {video_id}")
            _write_raw_response(raw_path, payload)
            self._network_calls += 1 if self.transcriber is None else 0
        chunks = list(iter_timestamped_chunks(payload))
        if not chunks:
            return [], str(raw_path)
        mapped = []
        for index, chunk in enumerate(chunks):
            row = map_chunk_to_canonical(video_id, chunk, self.canonical)
            row["chunk_index"] = index
            mapped.append(row)
        return mapped, str(raw_path)

    def _run_pack(self, pack: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
        if self.config.raw_only:
            archive = None
            members = {
                video: ""
                for video in self.canonical
                if video.startswith(pack + "_")
            }
        else:
            assert self._archives is not None
            archive = self._archives[pack]
            members = self._archive_members[pack]
        selected = list(members)
        if self.config.max_videos is not None:
            selected = selected[: self.config.max_videos]
        transcriber = None if self.config.raw_only else self._get_transcriber()
        rows: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        raw_files: dict[str, str] = {}
        no_speech_videos: list[str] = []
        for video_id in selected:
            try:
                mapped, raw_path = self._run_one_video(pack, archive, members[video_id], video_id, transcriber)
                rows.extend(mapped)
                raw_files[video_id] = raw_path
                if not mapped:
                    no_speech_videos.append(video_id)
            except Exception as exc:
                errors[video_id] = f"{type(exc).__name__}: {exc}"
                break
        pack_report: dict[str, Any] = {
            "pack": pack,
            "archive_paths": [] if archive is None else [str(path) for path in archive.paths],
            "expected_videos": len(selected),
            "completed_videos": len(raw_files),
            "failed_videos": sorted(errors),
            "errors": errors,
            "rows": len(rows),
            "raw_files": raw_files,
            "no_speech_videos": sorted(no_speech_videos),
            "no_speech_video_count": len(no_speech_videos),
            "status": "complete" if len(raw_files) == len(selected) and not errors else "partial",
            "scope_limited": self.config.max_videos is not None,
            "output_files": [],
        }
        if rows:
            frame = pd.DataFrame(rows)
            validate_chunk_frame(frame, self.canonical)
            pack_dir = self.config.output_dir
            pack_dir.mkdir(parents=True, exist_ok=True)
            chunks_path = pack_dir / f"asr_chunks_{pack.lower()}_ts.parquet"
            frame.to_parquet(chunks_path, index=False)
            pack_report["output_files"].append(str(chunks_path))
            if not self.config.metadata_only:
                embeddings = _embed_frame(
                    frame,
                    self.config.model,
                    self.config.batch_size,
                    self._get_embedder(),
                )
                embedding_path = pack_dir / f"emb_cache_asr_{pack.lower()}_chunks.npy"
                np.save(embedding_path, embeddings)
                pack_report["output_files"].append(str(embedding_path))
                pack_report["embedding_dimension"] = int(embeddings.shape[1])
                pack_report["embeddings_materialized"] = True
            else:
                pack_report["embeddings_materialized"] = False
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(_pack_manifest_path(self.config.output_dir, pack), pack_report)
        return pack_report

    def run(self, packs: Sequence[str] = SUPPORTED_PACKS) -> dict[str, Any]:
        selected = tuple(_normalise_pack(pack) for pack in packs)
        if not selected:
            raise ASRGlobalV2Error("at least one pack is required")
        if not self.config.execute:
            return self._dry_run(selected)
        if (
            not self.config.raw_only
            and not load_deepgram_key(self.config.env_path)
            and self.transcriber is None
        ):
            raise ASRNetworkApprovalError("DEEPGRAM_API_KEY is not configured")
        preflight = self.preflight(selected)
        manifest_path = self.config.output_dir / "asr_global_v2_manifest.json"
        existing: dict[str, Any] | None = None
        if self.config.resume and manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ASRGlobalV2Error(f"cannot resume invalid ASR manifest: {manifest_path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ASRGlobalV2Error("cannot resume ASR manifest that is not an object")
            if loaded.get("schema_version") != SCHEMA_VERSION:
                raise ASRGlobalV2Error("cannot resume ASR manifest with a different schema")
            if set(loaded.get("selected_packs", [])) != set(selected):
                raise ASRGlobalV2Error(
                    "resume scope mismatch: selected packs differ from existing ASR manifest"
                )
            if loaded.get("canonical_sha256") != preflight.get("canonical_sha256"):
                raise ASRGlobalV2Error("resume scope mismatch: canonical index changed")
            existing = loaded
        manifest = existing or self._manifest_base(selected, "execute")
        manifest["mode"] = "execute"
        manifest["resume"] = bool(self.config.resume)
        manifest["preflight"] = preflight
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_global_manifest(manifest)
        for pack in selected:
            previous = manifest.get("packs", {}).get(pack, {})
            if (
                isinstance(previous, Mapping)
                and previous.get("status") == "complete"
                and not previous.get("scope_limited", False)
                and not previous.get("errors")
            ):
                continue
            report = self._run_pack(pack, preflight["packs"][pack])
            manifest["packs"][pack] = report
            manifest["deepgram"]["network_calls"] = self._network_calls
            manifest["embedding"]["materialized"] = bool(
                report.get("embeddings_materialized")
            ) or bool(manifest["embedding"]["materialized"])
            if report.get("embedding_dimension") is not None:
                manifest["embedding"]["dimension"] = report["embedding_dimension"]
            manifest["updated_at_utc"] = _utc_now()
            self._write_global_manifest(manifest)
        complete = all(
            manifest["packs"].get(pack, {}).get("status") == "complete"
            and not manifest["packs"].get(pack, {}).get("scope_limited", False)
            for pack in SUPPORTED_PACKS
        )
        manifest["status"] = "complete" if complete else "partial"
        manifest["ready_for_global"] = complete and set(selected) == set(SUPPORTED_PACKS)
        manifest["ready_for_production"] = bool(
            manifest["ready_for_global"]
            and all(
                bool(manifest["packs"].get(pack, {}).get("embeddings_materialized"))
                and not manifest["packs"].get(pack, {}).get("errors")
                for pack in SUPPORTED_PACKS
            )
        )
        manifest["updated_at_utc"] = _utc_now()
        self._write_global_manifest(manifest)
        return manifest


def _parse_packs(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return SUPPORTED_PACKS
    return tuple(_normalise_pack(item) for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable ASR global v2 materializer")
    parser.add_argument("--packs", default="all", help="all or comma-separated K01-K20,L21-L30 packs")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("data/raw/video_archives"),
        help="Directory containing official Videos_Kxx/Lxx*.zip archives.",
    )
    parser.add_argument("--canonical", type=Path, default=Path("data/index/global_keyframes.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/index/asr_global_v2"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work/asr_global_v2"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/asr_global_v2_raw"))
    parser.add_argument("--model", default="models/bge-m3")
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--metadata-only", action="store_true", help="skip bge embedding; not production-ready")
    parser.add_argument("--execute", action="store_true", help="perform extraction/transcription")
    parser.add_argument("--allow-network", action="store_true", help="explicitly allow Deepgram network calls")
    parser.add_argument("--confirm-api", action="store_true", help="confirm Deepgram usage/cost")
    parser.add_argument("--resume", action="store_true", help="resume the exact existing selected-pack manifest")
    parser.add_argument(
        "--raw-only", action="store_true",
        help="rebuild chunks/embeddings from existing raw Deepgram JSON only; never extract media or call Deepgram",
    )
    parser.add_argument("--env", dest="env_path", type=Path, default=None)
    parser.add_argument("--api-model", default="nova-3")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--timeout", dest="timeout_seconds", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = RunnerConfig(
            archive_root=args.archive_root,
            canonical_path=args.canonical,
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            raw_dir=args.raw_dir,
            model=args.model,
            ffmpeg_path=args.ffmpeg,
            batch_size=args.batch_size,
            execute=args.execute,
            allow_network=args.allow_network,
            confirm_api=args.confirm_api,
            metadata_only=args.metadata_only,
            max_videos=args.max_videos,
            timeout_seconds=args.timeout_seconds,
            api_model=args.api_model,
            language=args.language,
            env_path=args.env_path,
            resume=args.resume,
            raw_only=args.raw_only,
        )
        report = ASRGlobalV2Runner(config).run(_parse_packs(args.packs))
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.get("status") in {"dry_run", "complete"} else 2
    except ASRGlobalV2Error as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
