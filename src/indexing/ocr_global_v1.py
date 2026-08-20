"""Audit and merge a provenance-safe OCR global index.

This module is deliberately a *merge-only* builder.  It does not run OCR and
does not call an embedder.  A pack can enter the v1 index only when it comes
from a versioned local OCR manifest whose scope is ``full`` and whose selected
rows cover every canonical keyframe in that pack.  Historical ``ocr_*.parquet``
files are audited, but remain diagnostic because they have no trustworthy
engine/provenance manifest.

The resulting retrieval table has one stable schema::

    embedding_row, video_id, kf_n, frame_idx, pts_time, text, source_pack

``embedding_row`` is zero-based and always aligns with ``embeddings.npy``.
The builder is fail-closed: missing packs, provisional artifacts, malformed
text, canonical mismatches, duplicate keyframes, or embedding misalignment
produce a ``blocked`` manifest and no promoted global index.

Run from WSL::

    .venv/bin/python -m src.indexing.ocr_global_v1 \
      --index-dir data/index \
      --canonical data/index/global_keyframes.parquet \
      --output-dir data/index/modality_global_v1/ocr_global_v1 \
      --report results/ocr_global_v1_report.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import unicodedata

import numpy as np
import pandas as pd


SCHEMA_VERSION = "hcmai.ocr_global_v1"
EMBEDDING_DIM = 1024
EXPECTED_PACKS = tuple(
    [f"K{i:02d}" for i in range(1, 21)]
    + [f"L{i:02d}" for i in range(21, 31)]
)
OUTPUT_COLUMNS = (
    "embedding_row",
    "video_id",
    "kf_n",
    "frame_idx",
    "pts_time",
    "text",
    "source_pack",
)
_PACK_RE = re.compile(r"^([KL]\d{2})_V\d+$", re.IGNORECASE)
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
_PROVISIONAL_TOKENS = ("pilot", "plan", "synthetic", "provisional", "diagnostic", "legacy")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repair_mojibake(value: Any) -> str:
    """Repair common UTF-8-as-Latin-1 corruption, then apply NFKC."""
    if value is None:
        return ""
    text = str(value).strip()
    for _ in range(3):
        before = _mojibake_score(text)
        candidates: list[str] = []
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "\ufffd" not in candidate:
                candidates.append(candidate)
        if not candidates:
            break
        candidate = min(candidates, key=_mojibake_score)
        if _mojibake_score(candidate) >= before:
            break
        text = candidate
    return unicodedata.normalize("NFKC", text).strip()


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS) + 2 * sum(
        0x80 <= ord(char) <= 0x9F for char in text
    )


def normalize_text(value: Any) -> tuple[str, dict[str, bool]]:
    """Return normalized OCR text plus quality flags used by the audit."""
    raw = "" if value is None else str(value)
    repaired = repair_mojibake(raw)
    compact = " ".join(repaired.casefold().strip(".!?").split())
    is_no_text = compact in _NO_TEXT or compact.startswith(
        ("no text is visible", "no readable text is visible", "there is no visible text")
    )
    text = "" if is_no_text else repaired
    return text, {
        "nfkc_changed": text != raw.strip() and unicodedata.normalize("NFKC", raw.strip()) != raw.strip(),
        "mojibake_suspected": _mojibake_score(raw) > 0,
        "replacement_character": "\ufffd" in raw or "\ufffd" in text,
        "no_text": is_no_text,
    }


def load_canonical(path: str | Path) -> pd.DataFrame:
    """Load the canonical map and reject duplicate or malformed identities."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"canonical index does not exist: {source}")
    table = pd.read_parquet(source).copy()
    required = {"video_id", "kf_n", "frame_idx", "pts_time"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"canonical index missing columns: {missing}")
    table["video_id"] = table["video_id"].astype(str).str.strip()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="raise").astype(int)
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="raise").astype(int)
    table["pts_time"] = pd.to_numeric(table["pts_time"], errors="raise").astype(float)
    if table[["video_id", "kf_n", "frame_idx", "pts_time"]].isna().any().any():
        raise ValueError("canonical index contains null identity or timestamp")
    if table.duplicated(["video_id", "kf_n"]).any():
        raise ValueError("canonical index contains duplicate (video_id, kf_n)")
    packs = table["video_id"].map(pack_for_video)
    table["source_pack"] = packs
    return table.sort_values(["source_pack", "video_id", "pts_time", "kf_n"]).reset_index(drop=True)


def pack_for_video(video_id: str) -> str:
    match = _PACK_RE.match(str(video_id).strip())
    if not match or match.group(1).upper() not in EXPECTED_PACKS:
        raise ValueError(f"unknown canonical video_id: {video_id}")
    return match.group(1).upper()


def _resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _is_provisional_path(path: Path) -> bool:
    parts = {part.casefold() for part in path.parts}
    return any(token in part for part in parts for token in _PROVISIONAL_TOKENS)


def _manifest_candidates(index_dir: Path) -> list[Path]:
    root = index_dir / "modality_global_v2"
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("manifest.json")
        if "ocr" in str(path.parent).casefold()
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return value


def _legacy_sources(index_dir: Path) -> list[Path]:
    ignored = {"ocr_compare", "ocr_gt_set", "ocr_gt_texts"}
    return sorted(
        path for path in index_dir.glob("ocr_*.parquet")
        if path.stem not in ignored and "partial" not in path.stem
    )


def _audit_legacy(path: Path, canonical: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "artifact_class": "diagnostic_legacy",
        "eligible": False,
        "rejection_reason": "legacy artifact has no trusted OCR engine/provenance manifest",
    }
    try:
        table = pd.read_parquet(path)
    except Exception as exc:
        report["status"] = "unreadable"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    required = {"video_id", "kf_n", "pts_time", "ocr_text"}
    missing = sorted(required - set(table.columns))
    if missing:
        report.update(status="invalid", missing_columns=missing)
        return report
    text_col = table["ocr_text"].map(normalize_text)
    texts = text_col.map(lambda item: item[0])
    flags = [item[1] for item in text_col]
    pack_name = path.stem.removeprefix("ocr_").upper()
    embedding_path = index_dir_for(path) / f"emb_cache_ocr_{pack_name.casefold()}.npy"
    # The top-level legacy files use pack-specific embeddings with the same
    # row order.  Alignment is useful evidence, but not sufficient for
    # promotion because frame_idx/provenance are missing.
    embedding: dict[str, Any] = {"path": str(embedding_path), "exists": embedding_path.exists()}
    if embedding_path.exists():
        try:
            array = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            embedding.update({
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "rows_match": bool(array.ndim == 2 and array.shape[0] == len(table)),
                "dim_valid": bool(array.ndim == 2 and array.shape[1] == EMBEDDING_DIM),
            })
        except Exception as exc:
            embedding["load_error"] = f"{type(exc).__name__}: {exc}"
    videos = set(table["video_id"].astype(str))
    report.update({
        "status": "diagnostic_existing",
        "source_pack": pack_name,
        "rows": int(len(table)),
        "nonempty_rows": int(texts.ne("").sum()),
        "videos": int(len(videos)),
        "canonical_scope": bool(videos <= set(canonical["video_id"])),
        "embedding": embedding,
        "quality": {
            "nfkc_changed_rows": int(sum(flag["nfkc_changed"] for flag in flags)),
            "mojibake_rows": int(sum(flag["mojibake_suspected"] for flag in flags)),
            "replacement_rows": int(sum(flag["replacement_character"] for flag in flags)),
            "duplicate_identities": int(table.duplicated(["video_id", "kf_n"]).sum()),
        },
    })
    return report


def index_dir_for(path: Path) -> Path:
    return path.parent


@dataclass(frozen=True)
class PackCandidate:
    pack: str
    manifest_path: Path
    retrieval_path: Path
    embeddings_path: Path
    manifest: Mapping[str, Any]


def _candidate_from_manifest(manifest_path: Path, manifest: Mapping[str, Any], pack: str) -> PackCandidate | None:
    scope = manifest.get("scope") or {}
    engine = manifest.get("engine") or {}
    pack_report = (manifest.get("packs") or {}).get(pack) or {}
    artifacts = pack_report.get("artifacts") or {}
    if scope.get("mode") != "full":
        return None
    if manifest.get("status") not in {"completed", "ready"}:
        return None
    if pack_report.get("status") != "completed":
        return None
    if engine.get("api_used") is True or engine.get("network_allowed") is True:
        return None
    retrieval = _resolve_path(artifacts.get("retrieval"), manifest_path.parent)
    embeddings = _resolve_path(artifacts.get("embeddings"), manifest_path.parent)
    if retrieval is None:
        retrieval = manifest_path.parent / "packs" / pack / "retrieval.parquet"
    if embeddings is None:
        embeddings = manifest_path.parent / "packs" / pack / "embeddings.npy"
    if _is_provisional_path(manifest_path) or _is_provisional_path(retrieval):
        return None
    return PackCandidate(pack, manifest_path, retrieval, embeddings, manifest)


def discover_pack_candidates(index_dir: str | Path) -> dict[str, list[PackCandidate]]:
    """Discover only versioned, local, full-scope pack candidates."""
    candidates: dict[str, list[PackCandidate]] = {pack: [] for pack in EXPECTED_PACKS}
    for manifest_path in _manifest_candidates(Path(index_dir)):
        try:
            manifest = _read_json(manifest_path)
        except ValueError:
            continue
        for pack in EXPECTED_PACKS:
            candidate = _candidate_from_manifest(manifest_path, manifest, pack)
            if candidate is not None:
                candidates[pack].append(candidate)
    return candidates


def _audit_versioned_manifest(path: Path) -> dict[str, Any]:
    """Explain why a versioned OCR manifest is or is not promotable."""
    item: dict[str, Any] = {
        "path": str(path),
        "artifact_class": "versioned_ocr",
        "eligible_source": False,
        "reasons": [],
    }
    try:
        manifest = _read_json(path)
    except ValueError as exc:
        item["status"] = "invalid"
        item["reasons"].append(str(exc))
        return item
    scope = manifest.get("scope") or {}
    engine = manifest.get("engine") or {}
    item.update({
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status"),
        "mode": scope.get("mode"),
        "api_used": bool(engine.get("api_used", False)),
        "network_allowed": bool(engine.get("network_allowed", False)),
        "selected_packs": list(scope.get("selected_packs") or []),
        "provisional_path": _is_provisional_path(path),
    })
    if item["provisional_path"]:
        item["reasons"].append("path contains provisional/pilot/diagnostic marker")
    if scope.get("mode") != "full":
        item["reasons"].append(f"scope mode is {scope.get('mode')!r}, not 'full'")
    if manifest.get("status") not in {"completed", "ready"}:
        item["reasons"].append(f"manifest status is {manifest.get('status')!r}")
    if item["api_used"] or item["network_allowed"]:
        item["reasons"].append("source is not offline/local-only")
    pack_status = {
        str(pack): str((report or {}).get("status"))
        for pack, report in (manifest.get("packs") or {}).items()
    }
    item["pack_status"] = pack_status
    item["eligible_source"] = not item["reasons"]
    return item


def _canonical_digest(canonical: pd.DataFrame) -> str:
    rows = canonical[["video_id", "kf_n", "frame_idx", "pts_time", "source_pack"]].to_dict("records")
    return _sha256(rows)


def _validate_candidate(
    candidate: PackCandidate,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame | None, np.ndarray | None, dict[str, Any]]:
    """Validate one candidate and return normalized rows/embeddings if valid."""
    report: dict[str, Any] = {
        "pack": candidate.pack,
        "manifest": str(candidate.manifest_path),
        "retrieval": str(candidate.retrieval_path),
        "embeddings": str(candidate.embeddings_path),
        "eligible": False,
        "errors": [],
        "quality": {
            "input_rows": 0,
            "output_rows": 0,
            "nfkc_changed_rows": 0,
            "mojibake_rows": 0,
            "replacement_rows": 0,
            "empty_rows": 0,
            "duplicate_identities": 0,
            "canonical_mismatches": 0,
        },
    }
    manifest_digest = (candidate.manifest.get("scope") or {}).get("canonical_digest")
    expected_digest = _canonical_digest(canonical)
    if not manifest_digest:
        report["errors"].append("manifest is missing canonical_digest")
    elif manifest_digest != expected_digest:
        report["errors"].append("manifest canonical_digest does not match current canonical map")
    if not candidate.retrieval_path.exists():
        report["errors"].append("retrieval parquet is missing")
        return None, None, report
    if not candidate.embeddings_path.exists():
        report["errors"].append("embedding matrix is missing")
        return None, None, report
    try:
        table = pd.read_parquet(candidate.retrieval_path).copy()
        embeddings = np.asarray(np.load(candidate.embeddings_path, mmap_mode="r", allow_pickle=False))
    except Exception as exc:
        report["errors"].append(f"artifact load failed: {type(exc).__name__}: {exc}")
        return None, None, report
    report["quality"]["input_rows"] = int(len(table))
    if embeddings.ndim != 2 or embeddings.shape != (len(table), EMBEDDING_DIM):
        report["errors"].append(
            f"embedding shape {list(embeddings.shape)} does not match ({len(table)}, {EMBEDDING_DIM})"
        )
    required = {"video_id", "kf_n", "pts_time"}
    text_col = "text" if "text" in table.columns else "ocr_text" if "ocr_text" in table.columns else None
    if text_col is None:
        required.add("text")
    missing = sorted(required - set(table.columns))
    if missing:
        report["errors"].append(f"retrieval missing columns: {missing}")
    if report["errors"]:
        return None, None, report

    table["video_id"] = table["video_id"].astype(str).str.strip()
    table["kf_n"] = pd.to_numeric(table["kf_n"], errors="coerce")
    table["pts_time"] = pd.to_numeric(table["pts_time"], errors="coerce")
    if table[["video_id", "kf_n", "pts_time"]].isna().any().any():
        report["errors"].append("retrieval contains null identity or timestamp")
        return None, None, report
    table["kf_n"] = table["kf_n"].astype(int)
    if table.duplicated(["video_id", "kf_n"]).any():
        report["quality"]["duplicate_identities"] = int(table.duplicated(["video_id", "kf_n"]).sum())
        report["errors"].append("duplicate (video_id, kf_n) identities")
    if "frame_idx" in table.columns:
        table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="coerce")
    else:
        table["frame_idx"] = np.nan
    canonical_pack = canonical[canonical["source_pack"] == candidate.pack]
    canonical_map = canonical_pack.set_index(["video_id", "kf_n"])
    normalized: list[str] = []
    flags: list[dict[str, bool]] = []
    for value in table[text_col].tolist():
        text, flag = normalize_text(value)
        normalized.append(text)
        flags.append(flag)
    table["text"] = normalized
    report["quality"]["nfkc_changed_rows"] = int(sum(flag["nfkc_changed"] for flag in flags))
    report["quality"]["mojibake_rows"] = int(sum(flag["mojibake_suspected"] for flag in flags))
    report["quality"]["replacement_rows"] = int(sum(flag["replacement_character"] for flag in flags))
    report["quality"]["empty_rows"] = int(table["text"].eq("").sum())
    if report["quality"]["replacement_rows"]:
        report["errors"].append("OCR text contains Unicode replacement characters")
    # Retrieval rows are text-only by contract; no-text rows belong in the
    # attempt log, not in the embedding index.
    if report["quality"]["empty_rows"]:
        report["errors"].append("retrieval contains empty/no-text rows")
    canonical_frame: list[int] = []
    canonical_pts: list[float] = []
    for row in table.itertuples(index=False):
        key = (row.video_id, int(row.kf_n))
        if key not in canonical_map.index:
            report["quality"]["canonical_mismatches"] += 1
            continue
        expected = canonical_map.loc[key]
        frame_value = getattr(row, "frame_idx")
        if pd.isna(frame_value) or int(frame_value) != int(expected.frame_idx) or abs(float(row.pts_time) - float(expected.pts_time)) > 1e-3:
            report["quality"]["canonical_mismatches"] += 1
            continue
        canonical_frame.append(int(expected.frame_idx))
        canonical_pts.append(float(expected.pts_time))
    if len(canonical_frame) != len(table):
        report["errors"].append("retrieval row does not exactly match canonical frame/timestamp map")
    if report["quality"]["canonical_mismatches"]:
        report["errors"].append("canonical mapping mismatch")
    selected_rows = int((candidate.manifest.get("packs") or {}).get(candidate.pack, {}).get("selected_rows", -1))
    canonical_rows = int(len(canonical_pack))
    report["canonical_rows"] = canonical_rows
    report["manifest_selected_rows"] = selected_rows
    if selected_rows != canonical_rows:
        report["errors"].append(
            f"manifest selected_rows={selected_rows} does not cover canonical pack rows={canonical_rows}"
        )
    if len(table) != canonical_rows:
        report["errors"].append(f"retrieval rows={len(table)} does not cover canonical pack rows={canonical_rows}")
    if report["errors"]:
        return None, None, report
    output = pd.DataFrame({
        "video_id": table["video_id"].astype(str).to_numpy(),
        "kf_n": table["kf_n"].astype(int).to_numpy(),
        "frame_idx": np.asarray(canonical_frame, dtype=np.int64),
        "pts_time": np.asarray(canonical_pts, dtype=np.float64),
        "text": table["text"].astype(str).to_numpy(),
        "source_pack": candidate.pack,
    })
    report["quality"]["output_rows"] = int(len(output))
    report["eligible_videos"] = int(output["video_id"].nunique())
    report["eligible"] = True
    return output, np.asarray(embeddings, dtype=np.float32), report


def audit_ocr_artifacts(index_dir: str | Path, canonical_path: str | Path) -> dict[str, Any]:
    """Audit legacy and versioned OCR artifacts without promoting anything."""
    index = Path(index_dir)
    canonical = load_canonical(canonical_path)
    legacy = [_audit_legacy(path, canonical) for path in _legacy_sources(index)]
    versioned = [_audit_versioned_manifest(path) for path in _manifest_candidates(index)]
    candidates = discover_pack_candidates(index)
    candidate_report: list[dict[str, Any]] = []
    eligible_video_ids: set[str] = set()
    for pack in EXPECTED_PACKS:
        if not candidates[pack]:
            continue
        for candidate in candidates[pack]:
            table, _, report = _validate_candidate(candidate, canonical)
            if table is not None and report.get("eligible"):
                eligible_video_ids.update(table["video_id"].astype(str).tolist())
            candidate_report.append(report)
    eligible_counts = {
        pack: sum(1 for item in candidate_report if item.get("pack") == pack and item.get("eligible"))
        for pack in EXPECTED_PACKS
    }
    eligible_packs = sorted(pack for pack, count in eligible_counts.items() if count == 1)
    duplicate_source_packs = sorted(pack for pack, count in eligible_counts.items() if count > 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready" if len(eligible_packs) == len(EXPECTED_PACKS) else "blocked",
        "scope": {
            "canonical_path": str(Path(canonical_path)),
            "canonical_digest": _canonical_digest(canonical),
            "canonical_rows": int(len(canonical)),
            "canonical_videos": int(canonical["video_id"].nunique()),
            "expected_packs": list(EXPECTED_PACKS),
        },
        "policy": {
            "network_allowed": False,
            "api_allowed": False,
            "legacy_top_level_artifacts_promoted": False,
            "provisional_or_synthetic_artifacts_promoted": False,
            "required_pack_scope": "full canonical coverage per pack",
            "embedding_dim": EMBEDDING_DIM,
        },
        "legacy_artifacts": legacy,
        "versioned_artifacts": versioned,
        "versioned_candidates": candidate_report,
        "coverage": {
            "legacy_rows": int(sum(item.get("rows", 0) for item in legacy)),
            "legacy_videos": int(len(set().union(*[
                set(pd.read_parquet(item["path"])["video_id"].astype(str))
                for item in legacy if item.get("status") == "diagnostic_existing"
            ]))) if any(item.get("status") == "diagnostic_existing" for item in legacy) else 0,
            "eligible_packs": eligible_packs,
            "eligible_pack_count": len(eligible_packs),
            "missing_packs": [pack for pack in EXPECTED_PACKS if pack not in eligible_packs],
            "eligible_rows": int(sum(item["quality"]["output_rows"] for item in candidate_report if item.get("eligible"))),
            "eligible_videos": int(len(eligible_video_ids)),
            "duplicate_source_packs": duplicate_source_packs,
        },
        "blockers": [
            *(["one or more expected packs lack a complete local OCR manifest and aligned embeddings"]
              if len(eligible_packs) != len(EXPECTED_PACKS) else []),
            *([f"multiple eligible sources found for pack(s): {duplicate_source_packs}"]
              if duplicate_source_packs else []),
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.parquet")
    table.to_parquet(temp, index=False)
    temp.replace(path)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npy")
    with temp.open("wb") as handle:
        np.save(handle, array)
    temp.replace(path)


def build_global_index(
    index_dir: str | Path,
    canonical_path: str | Path,
    output_dir: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the v1 global index or write an explicit blocked manifest."""
    audit = audit_ocr_artifacts(index_dir, canonical_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": audit["status"],
        "network_allowed": False,
        "api_used": False,
        "scope": audit["scope"],
        "policy": audit["policy"],
        "coverage": audit["coverage"],
        "blockers": audit["blockers"],
        "artifacts": {
            "manifest": str(output / "manifest.json"),
            "retrieval": str(output / "retrieval.parquet"),
            "embeddings": str(output / "embeddings.npy"),
        },
        "source_candidates": audit["versioned_candidates"],
    }
    if audit["status"] != "ready":
        manifest["status"] = "blocked"
        manifest["promotion"] = "not_promoted"
        _atomic_json(output / "manifest.json", manifest)
        if report_path is not None:
            _atomic_json(Path(report_path), audit)
        return manifest

    canonical = load_canonical(canonical_path)
    candidates = discover_pack_candidates(index_dir)
    tables: list[pd.DataFrame] = []
    arrays: list[np.ndarray] = []
    used_sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for pack in EXPECTED_PACKS:
        valid: list[tuple[pd.DataFrame, np.ndarray, dict[str, Any], PackCandidate]] = []
        for candidate in candidates[pack]:
            table, array, report = _validate_candidate(candidate, canonical)
            if table is not None and array is not None:
                valid.append((table, array, report, candidate))
        if len(valid) != 1:
            raise RuntimeError(f"expected exactly one valid OCR source for pack {pack}, found {len(valid)}")
        table, array, report, candidate = valid[0]
        identities = set(zip(table["video_id"], table["kf_n"]))
        if seen & identities:
            raise RuntimeError(f"duplicate OCR identities across sources in pack {pack}")
        seen.update(identities)
        tables.append(table)
        arrays.append(array)
        used_sources.append(report)
    merged = pd.concat(tables, ignore_index=True)
    embedding_matrix = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    merged["embedding_row"] = np.arange(len(merged), dtype=np.int64)
    merged = merged[["embedding_row", "video_id", "kf_n", "frame_idx", "pts_time", "text", "source_pack"]]
    if embedding_matrix.shape != (len(merged), EMBEDDING_DIM):
        raise RuntimeError("merged embedding matrix is not aligned with retrieval rows")
    _atomic_parquet(merged, output / "retrieval.parquet")
    _atomic_npy(embedding_matrix, output / "embeddings.npy")
    manifest.update({
        "status": "ready",
        "promotion": "ready",
        "coverage": {
            **audit["coverage"],
            "global_rows": int(len(merged)),
            "global_videos": int(merged["video_id"].nunique()),
            "embedding_shape": list(embedding_matrix.shape),
        },
        "source_candidates": used_sources,
    })
    _atomic_json(output / "manifest.json", manifest)
    if report_path is not None:
        audit["status"] = "ready"
        audit["coverage"].update({
            "global_rows": int(len(merged)),
            "global_videos": int(merged["video_id"].nunique()),
            "embedding_shape": list(embedding_matrix.shape),
        })
        _atomic_json(Path(report_path), audit)
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--canonical", default="data/index/global_keyframes.parquet")
    parser.add_argument("--output-dir", default="data/index/modality_global_v1/ocr_global_v1")
    parser.add_argument("--report", default="results/ocr_global_v1_report.json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_global_index(args.index_dir, args.canonical, args.output_dir, report_path=args.report)
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(Path(args.output_dir)),
        "eligible_packs": manifest["coverage"].get("eligible_packs", []),
        "missing_packs": manifest["coverage"].get("missing_packs", []),
        "global_rows": manifest["coverage"].get("global_rows", 0),
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
