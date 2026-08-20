"""Cheap read-only preflight for the canonical SQLite catalog."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.artifacts.registry import ArtifactRegistry, ArtifactStatus


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def _distinct_videos(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(DISTINCT video_id) FROM {table}").fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def build_catalog_preflight(
    db_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> ArtifactRegistry:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    db = Path(db_path or root / "data" / "catalog" / "aic2026_catalog.sqlite").resolve()
    registry = ArtifactRegistry()
    if not db.is_file():
        registry.register(ArtifactStatus("catalog", False, reason=f"missing:{db}"))
        return registry
    uri = f"file:{db.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        registry.register(ArtifactStatus("catalog", False, reason=f"open_error:{exc}"))
        return registry
    try:
        videos = _count(conn, "videos")
        keyframes = _count(conn, "keyframes")
        asr = _count(conn, "asr_chunks")
        ocr = _count(conn, "ocr_records")
        asr_videos = _distinct_videos(conn, "asr_chunks")
        ocr_videos = _distinct_videos(conn, "ocr_records")
        shards = _count(conn, "embedding_shards")
        registry.register(ArtifactStatus(
            "catalog", True, "global", str(db), row_count=videos,
            coverage=f"videos={videos};keyframes={keyframes};asr={asr};ocr={ocr}",
            canonical_mapping=keyframes > 0,
            reason="read_only_sqlite_catalog",
        ))
        registry.register(ArtifactStatus(
            "canonical_frames", keyframes > 0, "global" if keyframes > 0 else "missing",
            str(db), row_count=keyframes, canonical_mapping=keyframes > 0,
            reason="keyframes table",
        ))
        asr_scope = "global" if asr_videos >= videos and asr > 0 else ("partial" if asr > 0 else "missing")
        ocr_scope = "global" if ocr_videos >= videos and ocr >= keyframes and ocr > 0 else ("partial" if ocr > 0 else "missing")
        registry.register(ArtifactStatus(
            "asr", asr > 0, asr_scope,
            str(db), row_count=asr,
            coverage=f"videos={asr_videos}/{videos};rows={asr}", reason="asr_chunks table",
        ))
        registry.register(ArtifactStatus(
            "ocr", ocr > 0, ocr_scope,
            str(db), row_count=ocr,
            coverage=f"videos={ocr_videos}/{videos};rows={ocr}/{keyframes}", reason="ocr_records table",
        ))
        registry.register(ArtifactStatus(
            "embedding_shards", shards > 0, "diagnostic" if shards else "missing",
            str(db), row_count=shards, reason="registered_shards_are_not_validated_vectors",
        ))
    finally:
        conn.close()
    return registry


__all__ = ["build_catalog_preflight"]
