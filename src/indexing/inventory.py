"""Inventory and validate existing indexes without copying large features.

Run from the repository root with ``.venv/bin/python -m src.indexing.inventory``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.indexing.canonical import FRAME_KEY, canonicalize_frame_map, identity_set
from src.utils.paths import INDEX_DIR

VISUAL = {
    "vitl": ("global_siglip_vitl.npy", "global_keyframes_vitl.parquet"),
    "so400m384": ("global_so400m384.npy", "global_keyframes_so400m384.parquet"),
}


def _issue(report, severity, source, message):
    report["issues"].append({"severity": severity, "source": source, "message": message})


def build_inventory(index_dir: Path = INDEX_DIR) -> dict:
    report = {"schema_version": "1.0", "index_dir": str(index_dir), "visual": {}, "links": {}, "issues": []}
    maps = {}
    for name, (npy_name, map_name) in VISUAL.items():
        fp, mp = index_dir / npy_name, index_dir / map_name
        if not fp.exists() or not mp.exists():
            _issue(report, "warning", name, "feature or metadata file is absent")
            continue
        feats = np.load(fp, mmap_mode="r")
        raw = pd.read_parquet(mp)
        can, issues = canonicalize_frame_map(raw, name)
        for i in issues: _issue(report, i.severity, i.source, i.message)
        if len(feats) != len(raw): _issue(report, "error", name, f"feature/map length mismatch: {len(feats)} vs {len(raw)}")
        dup = int(can.duplicated(list(FRAME_KEY)).sum())
        if dup: _issue(report, "error", name, f"duplicate frame identities: {dup}")
        report["visual"][name] = {"features": npy_name, "metadata": map_name, "rows": len(raw), "shape": list(feats.shape), "dtype": str(feats.dtype), "videos": int(can.video_id.nunique()), "columns": list(raw.columns), "duplicate_identity_rows": dup}
        maps[name] = can
    if "vitl" in maps and "so400m384" in maps:
        a, b = maps["vitl"], maps["so400m384"]
        if len(a) == len(b):
            mismatches = int((a[list(FRAME_KEY)].astype(str).to_numpy() != b[list(FRAME_KEY)].astype(str).to_numpy()).any(axis=1).sum())
            if mismatches: _issue(report, "error", "visual-fusion", f"row identity mismatches: {mismatches}")
            report["links"]["visual_fusion"] = {"aligned": mismatches == 0, "rows": len(a), "identity_columns": list(FRAME_KEY)}
    base = maps.get("vitl")
    if base is None:
        base = maps.get("so400m384")
    if base is not None:
        base_keys = identity_set(base)
        for p in sorted(index_dir.glob("*.parquet")):
            if p.name in {x[1] for x in VISUAL.values()}: continue
            try: df = pd.read_parquet(p)
            except Exception: continue
            cols = set(df.columns)
            if {"video_id", "kf_n"}.issubset(cols) or {"vid", "kf_n"}.issubset(cols):
                if "vid" in df and "video_id" not in df: df = df.rename(columns={"vid": "video_id"})
                if "frame_idx" not in df:
                    keys = set(zip(df.video_id, df.kf_n))
                    known = set((v, k) for v, k, _ in base_keys)
                    orphan = len(keys - known)
                else: orphan = len(identity_set(df) - base_keys)
                report["links"][p.name] = {"rows": len(df), "orphan_rows_or_keys": int(orphan), "identity_columns": [c for c in FRAME_KEY if c in df]}
                if orphan: _issue(report, "warning", p.name, f"orphan mappings: {orphan} (source may be partial or legacy)")
    report["summary"] = {"errors": sum(i["severity"] == "error" for i in report["issues"]), "warnings": sum(i["severity"] == "warning" for i in report["issues"])}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    report = build_inventory(args.index_dir)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output: args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(1 if report["summary"]["errors"] else 0)


if __name__ == "__main__": main()
