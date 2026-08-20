"""Read-only exact FAISS index for normalized row-aligned feature matrices.

This module deliberately does not alter any production retriever.  The index
and manifest are separate artifacts and a manifest is checked on every load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.indexing.canonical import FRAME_COLUMNS, canonicalize_frame_map


class IndexValidationError(ValueError):
    """The persisted index no longer matches its source data."""


def _identity_checksum(map_path: Path) -> tuple[str, int]:
    raw = pd.read_parquet(map_path) if map_path.suffix == ".parquet" else pd.read_csv(map_path)
    frame_map, issues = canonicalize_frame_map(raw, str(map_path))
    errors = [i.message for i in issues if i.severity == "error"]
    if errors:
        raise IndexValidationError("invalid canonical map: " + "; ".join(errors))
    h = hashlib.sha256()
    for row in frame_map.loc[:, list(FRAME_COLUMNS)].itertuples(index=False, name=None):
        h.update(json.dumps(row, ensure_ascii=True, separators=(",", ":"), default=str).encode())
        h.update(b"\n")
    return h.hexdigest(), len(frame_map)


def _manifest_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".manifest.json")


class ReadOnlyVectorIndex:
    def __init__(self, index: faiss.Index, manifest: dict):
        self._index = index
        self.manifest = manifest

    @classmethod
    def load(cls, index_path: str | Path, map_path: str | Path | None = None) -> "ReadOnlyVectorIndex":
        path = Path(index_path)
        manifest_path = _manifest_path(path)
        if not manifest_path.exists():
            raise IndexValidationError(f"manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_map = Path(map_path) if map_path else Path(manifest["map_path"])
        checksum, rows = _identity_checksum(expected_map)
        if checksum != manifest.get("identity_checksum") or rows != manifest.get("row_count"):
            raise IndexValidationError("canonical map identity checksum/row count mismatch")
        index = faiss.read_index(str(path))
        if index.ntotal != rows or index.d != manifest["shape"][1] or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise IndexValidationError("FAISS index shape, metric, or row count mismatch")
        return cls(index, manifest)

    def search(self, queries: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self._index.d:
            raise ValueError(f"queries must have shape (n, {self._index.d})")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return self._index.search(q, min(top_k, self._index.ntotal))


def build_index(features_path: str | Path, map_path: str | Path, index_path: str | Path, *, overwrite: bool = False) -> Path:
    feature_path, mapping_path, output = map(Path, (features_path, map_path, index_path))
    manifest_path = _manifest_path(output)
    if not feature_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("feature or canonical map does not exist")
    if (output.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    features = np.load(feature_path, mmap_mode="r")
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.floating):
        raise ValueError("features must be a 2-D floating-point .npy")
    checksum, rows = _identity_checksum(mapping_path)
    if len(features) != rows:
        raise IndexValidationError(f"feature/map row mismatch: {len(features)} vs {rows}")
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(np.asarray(features, dtype=np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output))
    manifest = {"schema_version": "1.0", "feature_path": str(feature_path.resolve()),
                "dtype": str(features.dtype), "shape": list(features.shape),
                "metric": "inner_product", "normalization": "l2_normalized",
                "map_path": str(mapping_path.resolve()), "identity_checksum": checksum,
                "row_count": rows}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--features", required=True, type=Path)
    b.add_argument("--map", required=True, type=Path)
    b.add_argument("--output", required=True, type=Path)
    b.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.command == "build":
        print(build_index(args.features, args.map, args.output, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
