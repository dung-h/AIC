"""Read-only adapter for the canonical AIC2026 SQLite catalog.

The adapter deliberately does not own indexing or preprocessing.  It provides
the stable boundary between an ANN/text retriever and canonical output data:

* ``resolve_frame`` validates ``(video_id, kf_n)`` and returns ``frame_idx``;
* ``resolve_row_id`` resolves a deterministic catalog ordinal, or a row from a
  registered metadata mapping when a shard is explicitly supplied;
* ``search_asr`` and ``search_ocr`` search the catalog's FTS5 evidence tables;
* ``list_embedding_shards`` and ``list_manifests`` expose registered artifacts.

The connection is opened with SQLite's ``mode=ro`` URI and ``query_only`` is
enabled.  A missing or malformed catalog therefore fails at construction,
rather than silently degrading to a different source of truth.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CatalogError(RuntimeError):
    """Base error for catalog access and contract violations."""


class UnknownVideoError(CatalogError):
    """The requested video does not exist in the canonical catalog."""


class UnknownFrameError(CatalogError):
    """The requested keyframe or row id cannot be mapped canonically."""


@dataclass(frozen=True)
class CanonicalFrame:
    """Canonical frame identity and filesystem mapping."""

    row_id: int | None
    video_id: str
    kf_n: int
    frame_idx: int
    pts_time: float
    fps: float | None
    image_path: Path | None
    image_exists: bool
    map_path: Path


@dataclass(frozen=True)
class TextEvidence:
    """A searchable ASR/OCR hit with canonical frame anchors when available."""

    modality: str
    evidence_id: int
    video_id: str
    text: str
    score: float
    kf_n: int | None = None
    pts_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    start_kf_n: int | None = None
    end_kf_n: int | None = None
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class EmbeddingShard:
    """Registered vector or text/ANN artifact."""

    shard_id: int
    modality: str
    name: str
    vector_path: Path | None
    metadata_path: Path | None
    rows: int | None
    dim: int | None
    dtype: str | None
    metric: str | None
    model: str | None
    row_key: str | None
    status: str


@dataclass(frozen=True)
class ManifestRecord:
    """A manifest or registered shard metadata artifact."""

    path: Path
    kind: str
    shard_name: str | None
    exists: bool


@dataclass(frozen=True)
class CatalogManifest:
    """Parsed catalog manifest, preserving unknown fields for forward compatibility."""

    path: Path
    data: Mapping[str, Any]


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_REQUIRED_TABLES = {
    "catalog_meta",
    "videos",
    "keyframes",
    "asr_chunks",
    "ocr_records",
    "embedding_shards",
}
_ROW_ID_FIELDS = ("row_id", "global_id", "g", "id")


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"{field} must be an integer, got {value!r}") from exc


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _tokens(query: str) -> list[str]:
    return _TOKEN_RE.findall(str(query or ""))


class CatalogAdapter:
    """Read-only facade over one canonical SQLite catalog."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        if not self.db_path.is_file():
            raise CatalogError(f"catalog database does not exist: {self.db_path}")
        # The generated catalog lives at ``<project>/data/catalog/*.sqlite``;
        # registered paths are written relative to ``<project>``.
        # ``parents[2]`` also keeps the same contract for test catalogs under
        # ``<tmp>/data/catalog``.
        self.project_root = self.db_path.parents[2]
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        try:
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only=ON")
            self._validate_schema()
        except sqlite3.Error as exc:
            self.close()
            raise CatalogError(f"cannot open catalog read-only: {self.db_path}: {exc}") from exc

    def __enter__(self) -> "CatalogAdapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise CatalogError("catalog adapter is closed")
        return self._connection

    def _validate_schema(self) -> None:
        tables = {
            str(row[0])
            for row in self._conn().execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            )
        }
        missing = sorted(_REQUIRED_TABLES - tables)
        if missing:
            raise CatalogError(f"catalog schema is missing tables: {', '.join(missing)}")

    def _resolve_image(self, value: Any) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def _resolve_path(self, value: Any) -> Path | None:
        return self._resolve_image(value)

    def _frame_from_row(self, row: sqlite3.Row, row_id: int | None = None) -> CanonicalFrame:
        return CanonicalFrame(
            row_id=row_id,
            video_id=str(row["video_id"]),
            kf_n=int(row["kf_n"]),
            frame_idx=int(row["frame_idx"]),
            pts_time=float(row["pts_time"]),
            fps=_optional_float(row["fps"]),
            image_path=self._resolve_image(row["image_path"]),
            image_exists=bool(row["image_exists"]),
            map_path=self._resolve_path(row["map_path"]) or Path("."),
        )

    def video_exists(self, video_id: str) -> bool:
        row = self._conn().execute("SELECT 1 FROM videos WHERE video_id=?", (str(video_id),)).fetchone()
        return row is not None

    def resolve_frame(self, video_id: str, kf_n: int) -> CanonicalFrame:
        """Resolve and validate a canonical ``(video_id, kf_n)`` identity."""
        video_id = str(video_id)
        kf_n = _as_int(kf_n, "kf_n")
        if not self.video_exists(video_id):
            raise UnknownVideoError(f"unknown canonical video_id: {video_id!r}")
        row = self._conn().execute(
            """SELECT video_id,kf_n,frame_idx,pts_time,fps,image_path,image_exists,map_path
               FROM keyframes WHERE video_id=? AND kf_n=?""",
            (video_id, kf_n),
        ).fetchone()
        if row is None:
            raise UnknownFrameError(f"unknown canonical keyframe: video_id={video_id!r}, kf_n={kf_n}")
        return self._frame_from_row(row)

    def resolve_row_id(self, row_id: int, *, shard_name: str | None = None) -> CanonicalFrame:
        """Resolve a row id to a canonical frame.

        Without ``shard_name``, ``row_id`` is the zero-based deterministic
        catalog ordinal ordered by ``(video_id, kf_n)``.  This is the only row
        id guaranteed by the SQLite catalog itself.  A shard may be supplied
        when its metadata mapping contains ``row_id/global_id/g/id`` plus
        ``video_id`` and ``kf_n``; unsupported or missing mappings fail loudly.
        """
        row_id = _as_int(row_id, "row_id")
        if row_id < 0:
            raise UnknownFrameError(f"unknown canonical row_id: {row_id}")
        if shard_name is None:
            row = self._conn().execute(
                """SELECT video_id,kf_n,frame_idx,pts_time,fps,image_path,image_exists,map_path
                   FROM keyframes ORDER BY video_id,kf_n LIMIT 1 OFFSET ?""",
                (row_id,),
            ).fetchone()
            if row is None:
                raise UnknownFrameError(f"unknown canonical row_id: {row_id}")
            return self._frame_from_row(row, row_id=row_id)

        shard = self._shard_by_name(shard_name)
        if shard.metadata_path is None:
            raise CatalogError(f"shard {shard_name!r} has no metadata mapping")
        mapping = self._read_mapping_row(shard.metadata_path, row_id)
        if mapping is None:
            raise UnknownFrameError(f"row_id {row_id} not found in shard {shard_name!r}")
        if "video_id" not in mapping or "kf_n" not in mapping:
            raise CatalogError(
                f"shard {shard_name!r} metadata must contain video_id and kf_n for canonical resolution"
            )
        frame = self.resolve_frame(str(mapping["video_id"]), _as_int(mapping["kf_n"], "kf_n"))
        return CanonicalFrame(**{**frame.__dict__, "row_id": row_id})

    def _shard_by_name(self, name: str) -> EmbeddingShard:
        rows = self.list_embedding_shards()
        matches = [row for row in rows if row.name == name]
        if not matches:
            raise CatalogError(f"unknown registered embedding shard: {name!r}")
        if len(matches) > 1:
            raise CatalogError(f"ambiguous embedding shard name: {name!r}; pass a unique name")
        return matches[0]

    def _read_mapping_row(self, path: Path, row_id: int) -> dict[str, Any] | None:
        if not path.is_file():
            raise CatalogError(f"registered shard metadata does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    candidate = record.get("row_id", record.get("global_id", record.get("g", record.get("id"))))
                    if candidate is not None and int(candidate) == row_id:
                        return record
            return None
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            records = data if isinstance(data, list) else data.get("rows", data.get("records", []))
            for record in records:
                candidate = record.get("row_id", record.get("global_id", record.get("g", record.get("id"))))
                if candidate is not None and int(candidate) == row_id:
                    return record
            return None
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for record in csv.DictReader(handle):
                    candidate = next((record.get(field) for field in _ROW_ID_FIELDS if record.get(field) not in (None, "")), None)
                    if candidate is not None and int(candidate) == row_id:
                        return dict(record)
            return None
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as parquet  # type: ignore[import-not-found]
            except ImportError as exc:
                raise CatalogError(
                    f"cannot resolve shard row_id from parquet without pyarrow: {path}"
                ) from exc
            table = parquet.read_table(path)
            records = table.to_pylist()
            for index, record in enumerate(records):
                candidate = next((record.get(field) for field in _ROW_ID_FIELDS if record.get(field) is not None), index)
                if int(candidate) == row_id:
                    return record
            return None
        raise CatalogError(f"unsupported shard metadata format for row_id mapping: {path.suffix}")

    def _evidence_from_asr(self, row: sqlite3.Row) -> TextEvidence:
        start_frame = self._frame_for_anchor(row["video_id"], row["start_kf_n"])
        end_frame = self._frame_for_anchor(row["video_id"], row["end_kf_n"])
        return TextEvidence(
            modality="asr",
            evidence_id=int(row["chunk_id"]),
            video_id=str(row["video_id"]),
            text=str(row["text"]),
            score=float(row["score"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            start_kf_n=_optional_int(row["start_kf_n"]),
            end_kf_n=_optional_int(row["end_kf_n"]),
            start_frame_idx=start_frame,
            end_frame_idx=end_frame,
            source_path=self._resolve_path(row["source_path"]),
        )

    def _evidence_from_ocr(self, row: sqlite3.Row) -> TextEvidence:
        frame_idx = self._frame_for_anchor(row["video_id"], row["kf_n"])
        return TextEvidence(
            modality="ocr",
            evidence_id=int(row["ocr_id"]),
            video_id=str(row["video_id"]),
            text=str(row["text"]),
            score=float(row["score"]),
            kf_n=_optional_int(row["kf_n"]),
            pts_time=_optional_float(row["pts_time"]),
            start_frame_idx=frame_idx,
            end_frame_idx=frame_idx,
            source_path=self._resolve_path(row["source_path"]),
        )

    def _frame_for_anchor(self, video_id: str, kf_n: Any) -> int | None:
        if kf_n is None:
            return None
        row = self._conn().execute(
            "SELECT frame_idx FROM keyframes WHERE video_id=? AND kf_n=?",
            (str(video_id), int(kf_n)),
        ).fetchone()
        return None if row is None else int(row[0])

    def _fts_query(self, query: str) -> str:
        tokens = _tokens(query)
        if not tokens:
            raise CatalogError("text search query must contain at least one alphanumeric token")
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    def search_asr(self, query: str, *, limit: int = 20, video_id: str | None = None) -> list[TextEvidence]:
        """Search ASR FTS evidence, returning canonical timestamp anchors."""
        return self._search_text("asr", query, limit=limit, video_id=video_id)

    def search_ocr(self, query: str, *, limit: int = 20, video_id: str | None = None) -> list[TextEvidence]:
        """Search OCR FTS evidence, returning canonical keyframe anchors."""
        return self._search_text("ocr", query, limit=limit, video_id=video_id)

    def _search_text(self, modality: str, query: str, *, limit: int, video_id: str | None) -> list[TextEvidence]:
        if limit <= 0:
            return []
        limit = min(int(limit), 1000)
        match = self._fts_query(query)
        if modality == "asr":
            sql = """SELECT a.*, bm25(asr_fts) AS score
                     FROM asr_fts JOIN asr_chunks a ON a.chunk_id=asr_fts.rowid
                     WHERE asr_fts MATCH ?"""
            factory = self._evidence_from_asr
        elif modality == "ocr":
            sql = """SELECT o.*, bm25(ocr_fts) AS score
                     FROM ocr_fts JOIN ocr_records o ON o.ocr_id=ocr_fts.rowid
                     WHERE ocr_fts MATCH ?"""
            factory = self._evidence_from_ocr
        else:  # pragma: no cover - private method guard
            raise CatalogError(f"unsupported text modality: {modality}")
        params: list[Any] = [match]
        if video_id is not None:
            sql += " AND a.video_id=?" if modality == "asr" else " AND o.video_id=?"
            params.append(str(video_id))
        id_column = "chunk_id" if modality == "asr" else "ocr_id"
        sql += f" ORDER BY score ASC, {id_column} ASC LIMIT ?"
        params.append(limit)
        try:
            return [factory(row) for row in self._conn().execute(sql, params)]
        except sqlite3.OperationalError as exc:
            raise CatalogError(f"{modality} text search failed: {exc}") from exc

    def list_embedding_shards(
        self, *, modality: str | None = None, status: str | None = None
    ) -> list[EmbeddingShard]:
        clauses: list[str] = []
        params: list[str] = []
        if modality is not None:
            clauses.append("modality=?")
            params.append(str(modality))
        if status is not None:
            clauses.append("status=?")
            params.append(str(status))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn().execute(
            "SELECT shard_id,modality,name,vector_path,metadata_path,rows,dim,dtype,metric,model,row_key,status "
            f"FROM embedding_shards{where} ORDER BY modality,name,shard_id",
            params,
        )
        result: list[EmbeddingShard] = []
        for row in rows:
            result.append(
                EmbeddingShard(
                    shard_id=int(row["shard_id"]),
                    modality=str(row["modality"]),
                    name=str(row["name"]),
                    vector_path=self._resolve_path(row["vector_path"]),
                    metadata_path=self._resolve_path(row["metadata_path"]),
                    rows=_optional_int(row["rows"]),
                    dim=_optional_int(row["dim"]),
                    dtype=row["dtype"],
                    metric=row["metric"],
                    model=row["model"],
                    row_key=row["row_key"],
                    status=str(row["status"]),
                )
            )
        return result

    def list_manifests(self) -> list[ManifestRecord]:
        """List catalog and registered shard metadata artifacts."""
        records: list[ManifestRecord] = []
        seen: set[Path] = set()
        meta = self._conn().execute("SELECT value FROM catalog_meta WHERE key='manifest'").fetchone()
        if meta is not None:
            path = self._resolve_path(meta[0])
            if path is not None:
                seen.add(path)
                records.append(ManifestRecord(path, "catalog", None, path.is_file()))
        for shard in self.list_embedding_shards():
            if shard.metadata_path is not None and shard.metadata_path not in seen:
                seen.add(shard.metadata_path)
                records.append(ManifestRecord(shard.metadata_path, "shard_metadata", shard.name, shard.metadata_path.is_file()))
        return records

    def load_manifest(self) -> CatalogManifest:
        records = [record for record in self.list_manifests() if record.kind == "catalog"]
        if not records:
            raise CatalogError("catalog_meta does not register a catalog manifest")
        record = records[0]
        if not record.exists:
            raise CatalogError(f"registered catalog manifest does not exist: {record.path}")
        try:
            data = json.loads(record.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"cannot read catalog manifest: {record.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise CatalogError(f"catalog manifest must be a JSON object: {record.path}")
        return CatalogManifest(record.path, data)


__all__ = [
    "CatalogAdapter",
    "CatalogError",
    "CatalogManifest",
    "CanonicalFrame",
    "EmbeddingShard",
    "ManifestRecord",
    "TextEvidence",
    "UnknownFrameError",
    "UnknownVideoError",
]
