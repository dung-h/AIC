"""Offline OCR extraction over sampled local keyframes.

This module deliberately has no API/network path.  The production backend is
the repository's lazy ``LocalVLM`` adapter (Qwen2.5-VL when the local model and
dependencies are present); tests and callers may inject another local backend.

The cache is JSONL and records every attempted frame, including frames where
the model returned no visible text.  The materialized output contains only
non-empty OCR text and has the stable schema::

    video_id, kf_n, pts_time, ocr_text

Example (small, explicit subset only)::

    python src/utils/ocr_local_pipeline.py \
      --metadata data/index/global_keyframes.parquet \
      --frame-root data/keyframes \
      --output data/index/ocr_k03_local.jsonl \
      --max-frames-per-video 20
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Protocol


OUTPUT_FIELDS = ("video_id", "kf_n", "pts_time", "ocr_text")
OCR_PROMPT = (
    "Read only text that is visibly present in this image. Preserve the exact "
    "spelling, case, numbers, and Vietnamese diacritics. Do not infer or "
    "complete text that is not visible. Return one text block only; return "
    "NONE when no readable text is visible."
)
_NO_TEXT = {"", "none", "n/a", "na", "no text", "no visible text"}
_MOJIBAKE_MARKERS = frozenset(
    "ÃÂÐÑâðï¿½ìëêíîïæåçƒ„€™š›œž"
)


def _mojibake_score(text: str) -> int:
    """Score byte-decoding artefacts; lower is better."""
    # Keep the marker set above for backwards compatibility, but score the
    # common UTF-8-as-Latin-1 lead characters explicitly.  The previous set
    # was accidentally materialized as one corrupted Unicode string, so text
    # such as ``giÃ¢y`` received a clean score and was never repaired.
    marker_count = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    marker_count += sum(text.count(marker) for marker in ("Ã", "Â", "â", "ð", "Ð", "Ñ", "ì", "í", "î", "ï"))
    control_count = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
    pair_count = sum(text.count(pair) for pair in ("Ã", "Â", "â€", "â€™", "ì", "í"))
    return marker_count + pair_count + (2 * control_count)


def _repair_mojibake(text: str) -> str:
    """Conservatively undo repeated UTF-8 decoded as Latin-1/CP1252.

    A candidate is accepted only when it can be encoded as a single-byte
    encoding, decoded as UTF-8, and strictly lowers the mojibake score.  This
    keeps ordinary Vietnamese and Korean Unicode untouched and avoids guessing
    text when no byte-level repair is justified.
    """
    current = text
    for _ in range(3):
        candidates = []
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "\ufffd" not in candidate:
                candidates.append(candidate)
        if not candidates:
            break
        best = min(candidates, key=_mojibake_score)
        if _mojibake_score(best) >= _mojibake_score(current):
            break
        current = best
    return current


class LocalOCRBackend(Protocol):
    """Minimal contract for an offline OCR backend."""

    def recognize(self, image_path: str, prompt: str) -> str:
        ...


class QwenLocalOCRBackend:
    """Lazy adapter around the repository-local Qwen2.5-VL loader."""

    def __init__(self, model_path: str | Path, *, load_in_4bit: bool = False):
        self.model_path = Path(model_path)
        self.load_in_4bit = load_in_4bit
        self._vlm = None

        if not self.model_path.exists():
            raise RuntimeError(
                "local OCR backend unavailable: Qwen model directory does not "
                f"exist: {self.model_path}"
            )
        if not self.model_path.is_dir():
            raise RuntimeError(
                "local OCR backend unavailable: Qwen model path is not a directory: "
                f"{self.model_path}"
            )

    def _ensure(self):
        if self._vlm is not None:
            return self._vlm
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from src.core.local_vlm import LocalVLM
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "local OCR backend unavailable: cannot import src.core.local_vlm; "
                "install the local transformers/Qwen dependencies"
            ) from exc
        self._vlm = LocalVLM(self.model_path, load_in_4bit=self.load_in_4bit)
        return self._vlm

    def recognize(self, image_path: str, prompt: str) -> str:
        try:
            return str(self._ensure().answer(image_path, prompt, max_new_tokens=160))
        except Exception as exc:
            raise RuntimeError(
                "local OCR inference failed; no API fallback is configured: "
                f"{exc}"
            ) from exc

    def recognize_batch(self, image_paths: list[str], prompt: str) -> list[str]:
        """Recognize several images through independent local batch items.

        The VLM adapter returns one raw string per image.  The strict length
        check remains here as the frame-to-text safety boundary.
        """
        paths = [str(image_path) for image_path in image_paths]
        if not paths:
            return []
        try:
            raw = self._ensure().answer_batch(
                paths, prompt, max_new_tokens=160
            )
        except Exception as exc:
            raise RuntimeError(
                "local OCR batch inference failed; no API fallback is configured: "
                f"{exc}"
            ) from exc
        return _clean_batch_ocr_response(raw, len(paths))


@dataclass(frozen=True)
class OCRRunReport:
    sampled_frames: int
    cache_hits: int
    backend_calls: int
    text_rows: int
    no_text_rows: int
    missing_frames: int
    output_path: str
    cache_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampled_frames": self.sampled_frames,
            "cache_hits": self.cache_hits,
            "backend_calls": self.backend_calls,
            "text_rows": self.text_rows,
            "no_text_rows": self.no_text_rows,
            "missing_frames": self.missing_frames,
            "output_path": self.output_path,
            "cache_path": self.cache_path,
        }


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata field {field!r} must be numeric") from exc
    if result < 0:
        raise ValueError(f"metadata field {field!r} must be non-negative")
    return result


def _as_frame(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        video_id = str(record["video_id"]).strip()
        kf_n = int(record["kf_n"])
        pts_time = _as_float(record["pts_time"], "pts_time")
    except KeyError as exc:
        raise ValueError(f"metadata is missing required field: {exc.args[0]}") from exc
    if not video_id:
        raise ValueError("metadata video_id must not be empty")
    if kf_n < 0:
        raise ValueError("metadata kf_n must be non-negative")
    item = {"video_id": video_id, "kf_n": kf_n, "pts_time": pts_time}
    if record.get("frame_path"):
        item["frame_path"] = str(record["frame_path"])
    return item


def load_metadata(metadata: Any) -> list[dict[str, Any]]:
    """Load canonical frame metadata from a path, dataframe, or row mappings."""
    if isinstance(metadata, (str, Path)):
        path = Path(metadata)
        if not path.exists():
            raise FileNotFoundError(f"metadata file does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                import pandas as pd
                rows = pd.read_parquet(path).to_dict("records")
            except Exception as exc:
                raise RuntimeError(
                    f"cannot read metadata {path}; install a local parquet engine or use JSONL"
                ) from exc
    elif hasattr(metadata, "to_dict"):
        rows = metadata.to_dict("records")
    else:
        rows = metadata
    try:
        return [_as_frame(row) for row in rows]
    except TypeError as exc:
        raise ValueError("metadata must be an iterable of row mappings") from exc


def sample_frames(
    metadata: Iterable[Mapping[str, Any]],
    *,
    interval_seconds: float = 10.0,
    max_frames_per_video: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically sample keyframes by video and timestamp."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    if max_frames_per_video is not None and max_frames_per_video <= 0:
        raise ValueError("max_frames_per_video must be positive when provided")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in metadata:
        item = _as_frame(raw)
        grouped.setdefault(item["video_id"], []).append(item)

    selected: list[dict[str, Any]] = []
    for video_id in sorted(grouped):
        rows = sorted(grouped[video_id], key=lambda row: (row["pts_time"], row["kf_n"]))
        chosen: list[dict[str, Any]] = []
        last_time = None
        for row in rows:
            if last_time is None or row["pts_time"] - last_time >= interval_seconds:
                chosen.append(row)
                last_time = row["pts_time"]
        if max_frames_per_video is not None and len(chosen) > max_frames_per_video:
            if max_frames_per_video == 1:
                chosen = [chosen[0]]
            else:
                indexes = [round(i * (len(chosen) - 1) / (max_frames_per_video - 1))
                           for i in range(max_frames_per_video)]
                chosen = [chosen[index] for index in indexes]
        selected.extend(chosen)
    return selected


def _clean_ocr_text(value: Any) -> str:
    """Normalize a backend response without inventing or completing text."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("local OCR backend must return a string")
    # Repair byte-decoding artefacts before NFKC.  NFKC turns the CP1252
    # non-breaking spaces inside strings such as ``HÃ n`` into ordinary spaces
    # and makes a valid byte-level repair impossible.
    text = _repair_mojibake(value.strip())
    text = unicodedata.normalize("NFKC", text)
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("text"):
            text = text[4:].lstrip(" :\n")
    compact = " ".join(text.casefold().strip(".!?").split())
    if compact in _NO_TEXT or compact.startswith((
        "no text is visible", "no readable text is visible", "there is no visible text",
    )):
        return ""
    return text


_BATCH_RESULT_KEYS = ("results", "texts", "ocr_texts", "ocr", "items")


def _decode_batch_payload(value: Any) -> Any:
    """Decode a model's list-like response without guessing its items."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        raise ValueError("local OCR batch response must be a JSON/list value")

    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text[3:-3].strip().splitlines()
        if lines and lines[0].strip().casefold() in {"json", "list"}:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    if not text:
        raise ValueError("local OCR batch response is empty")

    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            # Some local runtimes return Python list notation despite the JSON
            # instruction.  literal_eval is restricted to data literals and
            # does not execute model output.
            payload = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            # Permit a short preamble only when it contains one complete JSON
            # value.  Trailing content is rejected to avoid selecting one list
            # from an ambiguous multi-answer response.
            decoder = json.JSONDecoder()
            starts = [index for index, char in enumerate(text) if char in "["]
            starts += [index for index, char in enumerate(text) if char == "{"]
            candidates = []
            for start in sorted(starts):
                try:
                    candidate, end = decoder.raw_decode(text[start:])
                except json.JSONDecodeError:
                    continue
                if text[start + end:].strip():
                    continue
                candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(
                    "local OCR batch response is not one complete JSON/list value"
                ) from exc
            payload = candidates[0]

    if isinstance(payload, Mapping):
        candidates = [
            payload[key]
            for key in _BATCH_RESULT_KEYS
            if key in payload and isinstance(payload[key], (list, tuple))
        ]
        if len(candidates) != 1:
            raise ValueError(
                "local OCR batch JSON object must contain exactly one recognized list"
            )
        payload = candidates[0]
    if not isinstance(payload, (list, tuple)):
        raise ValueError("local OCR batch response must contain a list")
    return list(payload)


def _clean_batch_ocr_response(value: Any, expected_length: int) -> list[str]:
    """Parse and clean an ordered batch response, failing closed on mismatch."""
    if expected_length < 0:
        raise ValueError("expected batch length must be non-negative")
    values = _decode_batch_payload(value)
    if len(values) != expected_length:
        raise ValueError(
            "local OCR batch response length is ambiguous: expected "
            f"{expected_length}, received {len(values)}"
        )
    cleaned: list[str] = []
    for index, item in enumerate(values):
        try:
            cleaned.append(_clean_ocr_text(item))
        except TypeError as exc:
            raise ValueError(
                f"local OCR batch item {index} is not text or null"
            ) from exc
    return cleaned


def _load_cache(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            normalized = {
                "video_id": str(item["video_id"]),
                "kf_n": int(item["kf_n"]),
                "pts_time": float(item["pts_time"]),
                "ocr_text": _clean_ocr_text(item.get("ocr_text", "")),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid OCR cache row {line_number} in {path}") from exc
        cache[(normalized["video_id"], normalized["kf_n"])] = normalized
    return cache


def _append_cache(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND makes each completed JSONL row a single append operation, so a
    # batch can be resumed after any individual result is persisted.  fsync is
    # intentional here: the cache is the durable checkpoint for local OCR.
    payload = (json.dumps(dict(item), ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o666
    )
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("local OCR cache append was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_frame_path(item: Mapping[str, Any], frame_root: Path) -> Path:
    if item.get("frame_path"):
        return Path(str(item["frame_path"]))
    return frame_root / str(item["video_id"]) / f"{int(item['kf_n']):03d}.jpg"


def _materialize_output(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    rows = [dict(record) for record in records if _clean_ocr_text(record.get("ocr_text", ""))]
    rows.sort(key=lambda row: (row["video_id"], int(row["kf_n"]), float(row["pts_time"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".jsonl":
        raise ValueError("local OCR output currently supports .jsonl only")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({field: row[field] for field in OUTPUT_FIELDS}, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return len(rows)


def run_local_ocr(
    metadata: Any,
    frame_root: str | Path,
    output_path: str | Path,
    *,
    backend: LocalOCRBackend | Callable[[str, str], str] | None = None,
    model_path: str | Path | None = None,
    load_in_4bit: bool = False,
    cache_path: str | Path | None = None,
    interval_seconds: float = 10.0,
    max_frames_per_video: int | None = None,
    batch_size: int = 1,
    strict_frames: bool = True,
) -> OCRRunReport:
    """Run local OCR on a bounded sampled subset and resume from JSONL cache."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = load_metadata(metadata)
    sampled = sample_frames(rows, interval_seconds=interval_seconds,
                            max_frames_per_video=max_frames_per_video)
    output = Path(output_path)
    cache = Path(cache_path) if cache_path else output.with_name(output.stem + ".cache.jsonl")
    cached = _load_cache(cache)

    if backend is None:
        default_model = Path(model_path) if model_path else (
            Path(__file__).resolve().parents[2] / "models" / "Qwen2.5-VL-3B-Instruct"
        )
        backend = QwenLocalOCRBackend(default_model, load_in_4bit=load_in_4bit)

    cache_hits = 0
    backend_calls = 0
    missing_frames = 0
    pending: list[tuple[dict[str, Any], Path]] = []
    for item in sampled:
        key = (item["video_id"], item["kf_n"])
        if key in cached:
            cache_hits += 1
            continue
        frame_path = _resolve_frame_path(item, Path(frame_root))
        if not frame_path.exists():
            missing_frames += 1
            if strict_frames:
                raise FileNotFoundError(f"sampled keyframe does not exist: {frame_path}")
            result = {**item, "ocr_text": ""}
            cached[key] = result
            _append_cache(cache, result)
            continue
        pending.append((item, frame_path))

    supports_batch = batch_size > 1 and hasattr(backend, "recognize_batch")
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        image_paths = [str(frame_path) for _, frame_path in batch]
        if supports_batch:
            raw_batch = backend.recognize_batch(image_paths, OCR_PROMPT)  # type: ignore[attr-defined]
            results = _clean_batch_ocr_response(raw_batch, len(batch))
            backend_calls += 1
        else:
            results = []
            for image_path in image_paths:
                if hasattr(backend, "recognize"):
                    raw = backend.recognize(image_path, OCR_PROMPT)  # type: ignore[attr-defined]
                elif callable(backend):
                    raw = backend(image_path, OCR_PROMPT)
                else:
                    raise TypeError(
                        "backend must implement recognize(image_path, prompt) or be callable"
                    )
                backend_calls += 1
                results.append(_clean_ocr_text(raw))

        # Validate the complete batch before writing any of its rows.  Each
        # row is then appended immediately, preserving resume progress if a
        # later operation is interrupted.
        for (item, _), ocr_text in zip(batch, results):
            result = {**item, "ocr_text": ocr_text}
            cached[(item["video_id"], item["kf_n"])] = result
            _append_cache(cache, result)

    current_keys = {(item["video_id"], item["kf_n"]) for item in sampled}
    current_records = [cached[key] for key in current_keys if key in cached]
    text_rows = _materialize_output(output, current_records)
    no_text_rows = sum(not _clean_ocr_text(item.get("ocr_text", "")) for item in cached.values())
    return OCRRunReport(
        sampled_frames=len(sampled), cache_hits=cache_hits, backend_calls=backend_calls,
        text_rows=text_rows, no_text_rows=no_text_rows, missing_frames=missing_frames,
        output_path=str(output), cache_path=str(cache),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline local OCR over sampled keyframes")
    parser.add_argument("--metadata", required=True, help="canonical metadata parquet/jsonl")
    parser.add_argument("--frame-root", required=True)
    parser.add_argument("--output", required=True, help="non-empty OCR JSONL output")
    parser.add_argument("--cache", default=None, help="resume cache JSONL")
    parser.add_argument("--model", default=None, help="local Qwen2.5-VL model directory")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--allow-missing-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_local_ocr(
        args.metadata, args.frame_root, args.output, model_path=args.model,
        cache_path=args.cache, interval_seconds=args.interval_seconds,
        max_frames_per_video=args.max_frames_per_video, load_in_4bit=args.load_in_4bit,
        batch_size=args.batch_size,
        strict_frames=not args.allow_missing_frames,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
