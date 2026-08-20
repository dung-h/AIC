"""Resumable, local-only OCR materializer for the full canonical corpus.

This module is intentionally independent from the historical OCR artifacts and
the production router.  It materializes one versioned directory with:

* an append-only attempt manifest containing every attempted keyframe,
  including ``no_text`` rows;
* one checkpoint per pack, so an interrupted pack can resume without
  re-running completed frames;
* a non-empty retrieval parquet containing only readable text;
* a matching local bge-m3 embedding matrix.

The default backends are local Qwen2.5-VL-3B-Instruct and local bge-m3.  No
remote provider, API, or network fallback is implemented.  Tests inject fake
backends and never load either model.

Examples::

    # Plan only; does not load a model or inspect frame files.
    .venv/bin/python -m src.eval.ocr_global_v2 --mode full --dry-run

    # Small explicit pilot.
    .venv/bin/python -m src.eval.ocr_global_v2 --mode pilot --execute

    # Full corpus requires the explicit --execute guard.
    .venv/bin/python -m src.eval.ocr_global_v2 --mode full --execute \
        --load-in-4bit --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "hcmai.ocr_global_v2"
ENGINE_NAME = "Qwen2.5-VL-3B-Instruct-local"
EMBEDDER_NAME = "bge-m3-local"
DEFAULT_OUTPUT_DIR = "data/index/modality_global_v2/ocr"
EXPECTED_PACKS = tuple(
    [f"K{i:02d}" for i in range(1, 21)]
    + [f"L{i:02d}" for i in range(21, 31)]
)
OCR_PROMPT = (
    "Read only text visibly present in this image. Preserve exact spelling, "
    "case, numbers, and Vietnamese diacritics. Do not infer or complete text "
    "that is not visible. Return one text block only; return NONE when no "
    "readable text is visible."
)
_NO_TEXT = {
    "",
    "none",
    "n/a",
    "na",
    "no text",
    "no visible text",
    "no readable text",
}
_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "Ð", "Ñ", "ì", "í", "î", "ï")


class LocalOCRBackend(Protocol):
    def recognize(self, image_path: str, prompt: str) -> str:
        ...


class LocalEmbedder(Protocol):
    def embed(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        ...


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS) + sum(
        text.count(pair) for pair in ("Ã", "Â", "â€", "â€™", "ì", "í")
    ) + 2 * sum(0x80 <= ord(char) <= 0x9F for char in text)


def repair_mojibake(value: Any) -> str:
    """Repair repeated UTF-8/Latin-1 corruption without changing valid text."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("OCR backend must return a string")
    current = value.strip()
    for _ in range(3):
        candidates: list[str] = []
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
    return unicodedata.normalize("NFKC", current).strip()


def clean_ocr_text(value: Any) -> str:
    """Normalize model output and remove explicit no-text responses."""
    text = repair_mojibake(value)
    if "\ufffd" in text:
        raise ValueError("OCR text still contains a Unicode replacement character")
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("text"):
            text = text[4:].lstrip(" :\n")
    compact = " ".join(text.casefold().strip(".!?").split())
    if compact in _NO_TEXT or compact.startswith(
        ("no text is visible", "no readable text is visible", "there is no visible text")
    ):
        return ""
    return text


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_table(source: Any) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"canonical metadata does not exist: {path}")
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".jsonl":
            return pd.DataFrame(
                [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            )
        if path.suffix.lower() == ".json":
            return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        raise ValueError(f"unsupported canonical metadata format: {path.suffix}")
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.DataFrame(list(source))


def _pack_for_video(video_id: str) -> str:
    pack = str(video_id).strip().upper().split("_", 1)[0]
    if pack not in EXPECTED_PACKS:
        raise ValueError(f"video_id has unknown pack prefix: {video_id}")
    return pack


def load_canonical(source: Any) -> pd.DataFrame:
    """Load and validate canonical ``kf_n -> frame_idx`` metadata."""
    table = _read_table(source)
    required = {"video_id", "kf_n", "frame_idx", "pts_time"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"canonical keyframe index missing columns: {missing}")
    table = table.copy()
    table["video_id"] = table["video_id"].astype(str).str.strip()
    if table["video_id"].eq("").any():
        raise ValueError("canonical video_id must not be empty")
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="raise").astype(int)
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="raise").astype(int)
    table["pts_time"] = pd.to_numeric(table["pts_time"], errors="raise").astype(float)
    if (table["kf_n"] < 0).any() or (table["frame_idx"] < 0).any() or (table["pts_time"] < 0).any():
        raise ValueError("canonical kf_n, frame_idx, and pts_time must be non-negative")
    if table.duplicated(["video_id", "kf_n"]).any():
        raise ValueError("canonical keyframe index contains duplicate (video_id, kf_n)")
    table["pack"] = table["video_id"].map(_pack_for_video)
    columns = ["video_id", "pack", "kf_n", "frame_idx", "pts_time"]
    if "frame_path" in table.columns:
        table["frame_path"] = table["frame_path"].astype(str)
        columns.append("frame_path")
    return table[columns].sort_values(["pack", "video_id", "pts_time", "kf_n"]).reset_index(drop=True)


def _identity_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in table.to_dict("records"):
        item = {
            "video_id": str(row["video_id"]),
            "pack": str(row["pack"]),
            "kf_n": int(row["kf_n"]),
            "frame_idx": int(row["frame_idx"]),
            "pts_time": float(row["pts_time"]),
        }
        if row.get("frame_path") and str(row["frame_path"]).lower() != "nan":
            item["frame_path"] = str(row["frame_path"])
        rows.append(item)
    return rows


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_digest(canonical: pd.DataFrame) -> str:
    return _digest(_identity_rows(canonical))


def parse_packs(value: str | Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    values: list[str] = []
    if isinstance(value, str):
        values.extend(value.split(","))
    else:
        for item in value:
            values.extend(str(item).split(","))
    packs = {item.strip().upper() for item in values if item.strip()}
    invalid = sorted(packs - set(EXPECTED_PACKS))
    if invalid:
        raise ValueError(f"unknown pack(s): {invalid}")
    return packs or None


def select_scope(
    canonical: pd.DataFrame,
    *,
    mode: str,
    packs: str | Sequence[str] | None = None,
    video_ids: Iterable[str] | None = None,
    video_limit: int | None = None,
    max_frames: int | None = None,
    pilot_video_limit: int = 12,
) -> pd.DataFrame:
    """Select a deterministic pilot or full scope from canonical rows."""
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be 'pilot' or 'full'")
    if video_limit is not None and video_limit < 0:
        raise ValueError("video_limit must be >= 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if pilot_video_limit <= 0:
        raise ValueError("pilot_video_limit must be positive")
    selected_packs = parse_packs(packs)
    wanted = {str(value).strip() for value in (video_ids or []) if str(value).strip()}
    available = set(canonical["video_id"].astype(str))
    unknown = sorted(wanted - available)
    if unknown:
        raise ValueError(f"video_id not found in canonical index: {unknown}")
    pack_mask = canonical["pack"].isin(selected_packs) if selected_packs is not None else pd.Series(
        True, index=canonical.index
    )
    scoped = canonical[pack_mask]
    if wanted:
        scoped = scoped[scoped["video_id"].isin(wanted)]
    videos = sorted(scoped["video_id"].unique())
    if mode == "pilot" and not wanted:
        videos = videos[:pilot_video_limit]
    if video_limit is not None:
        videos = videos[:video_limit]
    scoped = scoped[scoped["video_id"].isin(videos)].copy()
    if mode == "pilot":
        # One middle keyframe per selected video exercises all pack/frame maps
        # while keeping the default pilot cheap and deterministic.
        chosen: list[pd.Series] = []
        for video_id in videos:
            subset = scoped[scoped["video_id"] == video_id].sort_values(["pts_time", "kf_n"])
            if not subset.empty:
                chosen.append(subset.iloc[len(subset) // 2])
        scoped = pd.DataFrame(chosen, columns=scoped.columns) if chosen else scoped.iloc[0:0]
    scoped = scoped.sort_values(["pack", "video_id", "pts_time", "kf_n"]).reset_index(drop=True)
    if max_frames is not None:
        scoped = scoped.iloc[:max_frames].copy()
    return scoped


class QwenLocalOCRBackend:
    """Lazy local Qwen backend; construction never installs or calls APIs."""

    def __init__(self, model_path: str | Path, *, load_in_4bit: bool = False):
        self.model_path = Path(model_path)
        self.load_in_4bit = bool(load_in_4bit)
        if not self.model_path.is_dir():
            raise RuntimeError(
                "local OCR backend unavailable: Qwen model directory does not exist: "
                f"{self.model_path}"
            )
        self._vlm: Any = None

    def _ensure(self) -> Any:
        if self._vlm is not None:
            return self._vlm
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        repo_root = str(_root())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from src.core.local_vlm import LocalVLM
            self._vlm = LocalVLM(self.model_path, load_in_4bit=self.load_in_4bit)
        except Exception as exc:
            raise RuntimeError(
                "local OCR backend unavailable; install local transformers/Qwen dependencies; "
                "no API fallback is configured"
            ) from exc
        return self._vlm

    def recognize(self, image_path: str, prompt: str) -> str:
        try:
            return str(self._ensure().answer(image_path, prompt, max_new_tokens=160))
        except Exception as exc:
            raise RuntimeError(f"local Qwen OCR inference failed; no API fallback: {exc}") from exc


class LocalBGEEmbedder:
    """Local-only bge-m3 adapter with an explicit output dimension check."""

    def __init__(self, model_path: str | Path, *, device: str = "cuda", expected_dim: int = 1024):
        self.model_path = Path(model_path)
        self.device = device
        self.expected_dim = int(expected_dim)
        if not self.model_path.is_dir():
            raise RuntimeError(f"local bge-m3 model directory does not exist: {self.model_path}")
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(
                    str(self.model_path), device=self.device, local_files_only=True
                )
            except TypeError:
                # Older sentence-transformers versions do not expose the
                # keyword; the offline env vars still prohibit network access.
                self._model = SentenceTransformer(str(self.model_path), device=self.device)
        except Exception as exc:
            raise RuntimeError(
                "local bge-m3 embedder unavailable; no API/network fallback is configured"
            ) from exc
        return self._model

    def embed(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        model = self._ensure()
        output = model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        array = np.asarray(output, dtype=np.float32)
        if array.ndim != 2 or array.shape != (len(texts), self.expected_dim):
            raise ValueError(
                f"unexpected bge-m3 embedding shape: {array.shape}; "
                f"expected ({len(texts)}, {self.expected_dim})"
            )
        return array


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (str(row["video_id"]), int(row["kf_n"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid OCR attempt row {line_number} in {path}") from exc
        # The manifest is append-only: a retry intentionally appends a newer
        # record for the same identity.  The last record is the durable
        # current state; the full history remains available in the JSONL.
        rows[key] = row
    return rows


def _write_parquet_atomic(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_npy_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def _frame_path(row: Mapping[str, Any], frame_root: Path) -> Path:
    explicit = row.get("frame_path")
    if explicit and str(explicit).lower() != "nan":
        return Path(str(explicit))
    return frame_root / str(row["video_id"]) / f"{int(row['kf_n']):03d}.jpg"


@dataclass(frozen=True)
class OCRGlobalV2Runner:
    output_dir: Path
    model_path: Path
    embed_model_path: Path
    device: str = "cuda"
    load_in_4bit: bool = False
    batch_size: int = 32
    embedding_dim: int = 1024
    backend: LocalOCRBackend | Callable[[str, str], str] | None = None
    embedder: LocalEmbedder | None = None

    def _backend(self) -> LocalOCRBackend | Callable[[str, str], str]:
        if self.backend is not None:
            return self.backend
        return QwenLocalOCRBackend(self.model_path, load_in_4bit=self.load_in_4bit)

    def _embedder(self) -> LocalEmbedder:
        if self.embedder is not None:
            return self.embedder
        return LocalBGEEmbedder(
            self.embed_model_path, device=self.device, expected_dim=self.embedding_dim
        )

    def _scope(self, canonical: pd.DataFrame, selected: pd.DataFrame, mode: str) -> dict[str, Any]:
        identities = _identity_rows(selected)
        scope = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "canonical_digest": canonical_digest(canonical),
            "selected_digest": _digest(identities),
            "selected_rows": len(identities),
            "selected_video_ids": sorted(selected["video_id"].astype(str).unique().tolist()),
            "selected_packs": sorted(selected["pack"].astype(str).unique().tolist()),
            "model_path": str(self.model_path),
            "embed_model_path": str(self.embed_model_path),
            "embedding_dim": self.embedding_dim,
        }
        scope["scope_digest"] = _digest(scope)
        return scope

    def _paths(self, pack: str) -> dict[str, Path]:
        pack_dir = self.output_dir / "packs" / pack
        return {
            "dir": pack_dir,
            "attempts": pack_dir / "attempts.jsonl",
            "checkpoint": pack_dir / "checkpoint.json",
            "retrieval": pack_dir / "retrieval.parquet",
            "embeddings": pack_dir / "embeddings.npy",
        }

    def _checkpoint(
        self, path: Path, *, pack: str, scope: Mapping[str, Any], attempts: Mapping[tuple[str, int], Mapping[str, Any]], status: str
    ) -> dict[str, Any]:
        counts = {"text": 0, "no_text": 0, "error": 0}
        for row in attempts.values():
            counts[str(row.get("status", "error"))] = counts.get(str(row.get("status", "error")), 0) + 1
        payload = {
            "schema_version": SCHEMA_VERSION,
            "pack": pack,
            "scope_digest": scope["selected_digest"],
            "status": status,
            "attempted_rows": len(attempts),
            "text_rows": counts.get("text", 0),
            "no_text_rows": counts.get("no_text", 0),
            "error_rows": counts.get("error", 0),
        }
        _atomic_json(path, payload)
        return payload

    def _materialize_pack(
        self, pack: str, attempts: Mapping[tuple[str, int], Mapping[str, Any]], paths: Mapping[str, Path]
    ) -> tuple[int, list[int] | None]:
        rows = [dict(row) for row in attempts.values() if row.get("status") == "text" and row.get("ocr_text")]
        rows.sort(key=lambda row: (str(row["video_id"]), float(row["pts_time"]), int(row["kf_n"])))
        if not rows:
            raise RuntimeError(f"OCR pack {pack} has no non-empty text rows; retrieval index would be empty")
        metadata = pd.DataFrame(rows)
        metadata["embedding_row"] = np.arange(len(metadata), dtype=np.int64)
        metadata = metadata[
            ["embedding_row", "video_id", "pack", "kf_n", "frame_idx", "pts_time", "ocr_text", "ocr_engine"]
        ]
        embeddings = self._embedder().embed(metadata["ocr_text"].tolist(), batch_size=self.batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(metadata), self.embedding_dim):
            raise ValueError(
                f"embedder returned {embeddings.shape}; expected ({len(metadata)}, {self.embedding_dim})"
            )
        _write_parquet_atomic(metadata, paths["retrieval"])
        _write_npy_atomic(embeddings, paths["embeddings"])
        return len(metadata), list(embeddings.shape)

    def _materialize_global_index(self, packs: Mapping[str, Mapping[str, Any]]) -> tuple[int, list[int]]:
        """Create one unified retrieval index only after every pack is valid."""
        tables: list[pd.DataFrame] = []
        arrays: list[np.ndarray] = []
        for pack in sorted(packs):
            pack_report = packs[pack]
            if pack_report.get("status") != "completed":
                raise RuntimeError(f"cannot build global OCR index while pack {pack} is not complete")
            paths = self._paths(pack)
            if not paths["retrieval"].exists() or not paths["embeddings"].exists():
                raise RuntimeError(f"pack {pack} is marked complete but retrieval artifacts are missing")
            table = pd.read_parquet(paths["retrieval"])
            array = np.asarray(np.load(paths["embeddings"], allow_pickle=False), dtype=np.float32)
            if table.empty or array.shape != (len(table), self.embedding_dim):
                raise ValueError(f"pack {pack} retrieval/embedding shape mismatch")
            tables.append(table)
            arrays.append(array)
        if not tables:
            raise RuntimeError("global OCR retrieval index would be empty")
        metadata = pd.concat(tables, ignore_index=True)
        metadata["embedding_row"] = np.arange(len(metadata), dtype=np.int64)
        embeddings = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
        if embeddings.shape != (len(metadata), self.embedding_dim):
            raise ValueError("global OCR embedding matrix does not align with retrieval metadata")
        _write_parquet_atomic(metadata, self.output_dir / "retrieval.parquet")
        _write_npy_atomic(embeddings, self.output_dir / "embeddings.npy")
        return len(metadata), list(embeddings.shape)

    def run(
        self,
        canonical: Any,
        frame_root: str | Path,
        *,
        mode: str = "pilot",
        packs: str | Sequence[str] | None = None,
        video_ids: Iterable[str] | None = None,
        video_limit: int | None = None,
        max_frames: int | None = None,
        pilot_video_limit: int = 12,
        execute: bool = True,
        dry_run: bool = False,
        resume: bool = False,
    ) -> dict[str, Any]:
        canonical_table = load_canonical(canonical)
        selected = select_scope(
            canonical_table, mode=mode, packs=packs, video_ids=video_ids,
            video_limit=video_limit, max_frames=max_frames,
            pilot_video_limit=pilot_video_limit,
        )
        scope = self._scope(canonical_table, selected, mode)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if existing and existing.get("scope", {}).get("scope_digest") != scope["scope_digest"]:
            if not resume:
                raise FileExistsError(
                    f"output directory already belongs to another OCR scope: {self.output_dir}; "
                    "use a new versioned directory or --resume for the exact same scope"
                )
            raise ValueError("--resume scope mismatch: selected canonical rows differ")
        if mode == "full" and not execute and not dry_run:
            raise ValueError("full OCR requires explicit execute=True; use dry_run=True to inspect the plan")

        full_corpus = len(selected) == len(canonical_table) and set(selected["video_id"]) == set(canonical_table["video_id"])
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run" if dry_run else "planned" if not execute else "in_progress",
            "scope": scope,
            "canonical": {
                "rows": len(canonical_table),
                "videos": int(canonical_table["video_id"].nunique()),
                "packs": sorted(canonical_table["pack"].unique().tolist()),
                "full_corpus": full_corpus,
            },
            "selection": {
                "rows": len(selected),
                "videos": int(selected["video_id"].nunique()),
                "packs": sorted(selected["pack"].unique().tolist()),
                "requested_packs": sorted(parse_packs(packs) or []),
                "requested_video_ids": sorted(str(value) for value in (video_ids or [])),
                "video_limit": video_limit,
                "max_frames": max_frames,
                "pilot_video_limit": pilot_video_limit,
                "resume": bool(resume),
            },
            "engine": {
                "ocr": ENGINE_NAME,
                "ocr_model_path": str(self.model_path),
                "embedding": EMBEDDER_NAME,
                "embedding_model_path": str(self.embed_model_path),
                "api_used": False,
                "network_allowed": False,
                "prompt": OCR_PROMPT,
            },
            "coverage": {
                "selected_rows": len(selected),
                "attempted_rows": 0,
                "text_rows": 0,
                "no_text_rows": 0,
                "error_rows": 0,
                "retrieval_rows": 0,
                "video_text_coverage": 0.0,
            },
            "packs": {},
            "artifacts": {
                "manifest": str(manifest_path),
                "attempt_manifest": str(self.output_dir / "attempt_manifest.jsonl"),
                "pack_root": str(self.output_dir / "packs"),
            },
            "requirements": {
                "full_execution_requires_execute": True,
                "remaining": [] if execute and not dry_run else ["explicit execution"],
            },
        }
        _atomic_json(manifest_path, report)
        if dry_run or not execute:
            return report

        backend = self._backend()
        frame_root_path = Path(frame_root)
        global_attempt_path = self.output_dir / "attempt_manifest.jsonl"
        for pack in sorted(selected["pack"].unique().tolist()):
            pack_rows = selected[selected["pack"] == pack]
            paths = self._paths(pack)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            attempts = _load_jsonl(paths["attempts"])
            pack_scope_digest = _digest({
                "selected_digest": _digest(_identity_rows(pack_rows)),
                "scope_digest": scope["scope_digest"],
            })
            checkpoint_data = (
                json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
                if paths["checkpoint"].exists() else None
            )
            if checkpoint_data and checkpoint_data.get("scope_digest") != pack_scope_digest:
                raise ValueError(f"pack {pack} checkpoint scope mismatch; use a new output version")
            errors = 0
            for row in _identity_rows(pack_rows):
                key = (row["video_id"], row["kf_n"])
                previous = attempts.get(key)
                if previous and previous.get("status") in {"text", "no_text"}:
                    if (
                        int(previous.get("frame_idx", -1)) != int(row["frame_idx"])
                        or float(previous.get("pts_time", -1.0)) != float(row["pts_time"])
                    ):
                        raise ValueError(f"attempt does not match canonical mapping: {key}")
                    continue
                image_path = _frame_path(row, frame_root_path)
                record = {
                    **row,
                    "image_path": str(image_path),
                    "ocr_engine": ENGINE_NAME,
                    "status": "error",
                    "ocr_text": "",
                    "error": None,
                }
                try:
                    if not image_path.exists():
                        raise FileNotFoundError(f"canonical keyframe does not exist: {image_path}")
                    if hasattr(backend, "recognize"):
                        raw = backend.recognize(str(image_path), OCR_PROMPT)  # type: ignore[attr-defined]
                    elif callable(backend):
                        raw = backend(str(image_path), OCR_PROMPT)
                    else:
                        raise TypeError("OCR backend must implement recognize or be callable")
                    text = clean_ocr_text(raw)
                    record["ocr_text"] = text
                    record["status"] = "text" if text else "no_text"
                except Exception as exc:
                    errors += 1
                    record["error"] = f"{type(exc).__name__}: {exc}"
                attempts[key] = record
                _append_jsonl(paths["attempts"], record)
                _append_jsonl(global_attempt_path, {"pack": pack, **record})
                self._checkpoint(
                    paths["checkpoint"], pack=pack, scope={"selected_digest": pack_scope_digest},
                    attempts=attempts, status="in_progress",
                )

            counts = {"text": 0, "no_text": 0, "error": 0}
            for attempt in attempts.values():
                counts[str(attempt.get("status", "error"))] = counts.get(str(attempt.get("status", "error")), 0) + 1
            pack_report: dict[str, Any] = {
                "selected_rows": len(pack_rows),
                "attempted_rows": len(attempts),
                "text_rows": counts.get("text", 0),
                "no_text_rows": counts.get("no_text", 0),
                "error_rows": counts.get("error", 0),
                "status": "blocked" if errors or len(attempts) != len(pack_rows) else "in_progress",
                "artifacts": {key: str(value) for key, value in paths.items() if key != "dir"},
            }
            if pack_report["status"] == "in_progress":
                try:
                    retrieval_rows, embedding_shape = self._materialize_pack(pack, attempts, paths)
                    pack_report["retrieval_rows"] = retrieval_rows
                    pack_report["embedding_shape"] = embedding_shape
                    pack_report["status"] = "completed"
                except Exception as exc:
                    pack_report["status"] = "blocked"
                    pack_report["materialize_error"] = f"{type(exc).__name__}: {exc}"
            self._checkpoint(
                paths["checkpoint"], pack=pack, scope={"selected_digest": pack_scope_digest},
                attempts=attempts, status=pack_report["status"],
            )
            report["packs"][pack] = pack_report
            report["coverage"]["attempted_rows"] += len(attempts)
            report["coverage"]["text_rows"] += counts.get("text", 0)
            report["coverage"]["no_text_rows"] += counts.get("no_text", 0)
            report["coverage"]["error_rows"] += counts.get("error", 0)
            report["coverage"]["retrieval_rows"] += int(pack_report.get("retrieval_rows", 0))
            report["status"] = "blocked" if pack_report["status"] == "blocked" else report["status"]
            _atomic_json(manifest_path, report)

        if report["status"] != "blocked":
            try:
                retrieval_rows, embedding_shape = self._materialize_global_index(report["packs"])
                report["coverage"]["global_retrieval_rows"] = retrieval_rows
                report["coverage"]["global_embedding_shape"] = embedding_shape
                report["artifacts"]["retrieval"] = str(self.output_dir / "retrieval.parquet")
                report["artifacts"]["embeddings"] = str(self.output_dir / "embeddings.npy")
            except Exception as exc:
                report["status"] = "blocked"
                report["global_materialize_error"] = f"{type(exc).__name__}: {exc}"

        selected_videos = set(selected["video_id"].astype(str))
        text_videos: set[str] = set()
        for pack in report["packs"]:
            retrieval_path = self._paths(pack)["retrieval"]
            if not retrieval_path.exists():
                continue
            try:
                text_videos.update(pd.read_parquet(retrieval_path)["video_id"].astype(str).tolist())
            except Exception:
                # The pack status already records materialization failure; do
                # not hide it behind a secondary coverage exception.
                continue
        report["coverage"]["video_text_coverage"] = (
            len(text_videos & selected_videos) / len(selected_videos) if selected_videos else 0.0
        )
        if report["status"] != "blocked":
            report["status"] = "completed"
        report["requirements"]["remaining"] = (
            [] if full_corpus and report["status"] == "completed" else
            ["complete every selected pack without errors", "run global preflight before promotion"]
        )
        _atomic_json(manifest_path, report)
        return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/index/global_keyframes.parquet")
    parser.add_argument("--frame-root", default="data/keyframes")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--embed-model", default="models/bge-m3")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--execute", action="store_true", help="required guard for model inference")
    parser.add_argument("--dry-run", action="store_true", help="write only a deterministic plan manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pack", action="append", default=None)
    parser.add_argument("--video-id", action="append", default=None)
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pilot-video-limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "full" and not args.execute and not args.dry_run:
        raise SystemExit("refusing full OCR without --execute; use --dry-run to inspect the plan")
    root = _root()
    runner = OCRGlobalV2Runner(
        output_dir=(root / args.output_dir).resolve(),
        model_path=(root / args.model).resolve(),
        embed_model_path=(root / args.embed_model).resolve(),
        device=args.device, load_in_4bit=args.load_in_4bit, batch_size=args.batch_size,
    )
    report = runner.run(
        (root / args.metadata).resolve(), (root / args.frame_root).resolve(),
        mode=args.mode, packs=args.pack, video_ids=args.video_id,
        video_limit=args.video_limit, max_frames=args.max_frames,
        pilot_video_limit=args.pilot_video_limit, execute=args.execute,
        dry_run=args.dry_run, resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] not in {"blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
