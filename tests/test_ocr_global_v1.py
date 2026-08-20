from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.indexing.ocr_global_v1 import (
    EXPECTED_PACKS,
    OUTPUT_COLUMNS,
    audit_ocr_artifacts,
    build_global_index,
    load_canonical,
    normalize_text,
)


def _canonical() -> pd.DataFrame:
    rows = []
    for number, pack in enumerate(EXPECTED_PACKS, start=1):
        rows.append({
            "video_id": f"{pack}_V001",
            "kf_n": 1,
            "frame_idx": 1000 + number,
            "pts_time": float(number),
        })
    return pd.DataFrame(rows)


def _write_full_pack_source(root: Path, canonical_path: Path) -> None:
    source_root = root / "modality_global_v2" / "ocr_full_local_v1"
    (source_root / "packs").mkdir(parents=True)
    canonical_table = load_canonical(canonical_path)
    manifest = {
        "schema_version": "hcmai.ocr_global_v2",
        "status": "completed",
        "engine": {
            "ocr": "Qwen2.5-VL-3B-Instruct-local",
            "embedding": "bge-m3-local",
            "api_used": False,
            "network_allowed": False,
        },
        "scope": {
            "mode": "full",
            "canonical_digest": __import__("src.indexing.ocr_global_v1", fromlist=["_canonical_digest"])._canonical_digest(canonical_table),
            "selected_packs": list(EXPECTED_PACKS),
        },
        "packs": {},
    }
    for number, pack in enumerate(EXPECTED_PACKS, start=1):
        pack_dir = source_root / "packs" / pack
        pack_dir.mkdir()
        row = canonical_table[canonical_table["source_pack"] == pack].iloc[0]
        pd.DataFrame([{
            "embedding_row": 0,
            "video_id": row.video_id,
            "pack": pack,
            "kf_n": int(row.kf_n),
            "frame_idx": int(row.frame_idx),
            "pts_time": float(row.pts_time),
            "ocr_text": f"Nhãn {pack}",
            "ocr_engine": "Qwen2.5-VL-3B-Instruct-local",
        }]).to_parquet(pack_dir / "retrieval.parquet", index=False)
        np.save(pack_dir / "embeddings.npy", np.ones((1, 1024), dtype=np.float32) * number)
        manifest["packs"][pack] = {
            "status": "completed",
            "selected_rows": 1,
            "retrieval_rows": 1,
            "artifacts": {
                "retrieval": f"packs/{pack}/retrieval.parquet",
                "embeddings": f"packs/{pack}/embeddings.npy",
            },
        }
    (source_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_normalize_text_repairs_mojibake_and_nfkc():
    text, flags = normalize_text("HÃ  Ná»™i")
    assert text == "Hà Nội"
    assert flags["mojibake_suspected"]
    text, flags = normalize_text("Ｎａｍ　Ｂộ")
    assert text == "Nam Bộ"
    assert flags["nfkc_changed"]
    assert normalize_text("NONE")[0] == ""


def test_legacy_artifacts_are_audited_but_never_eligible(tmp_path: Path):
    canonical = _canonical().iloc[[0]].copy()
    canonical_path = tmp_path / "canonical.parquet"
    canonical.to_parquet(canonical_path, index=False)
    index = tmp_path / "index"
    index.mkdir()
    pd.DataFrame([{
        "video_id": "K01_V001",
        "kf_n": 1,
        "pts_time": 1.0,
        "ocr_text": "HÃ  Ná»™i",
    }]).to_parquet(index / "ocr_k01.parquet", index=False)
    np.save(index / "emb_cache_ocr_k01.npy", np.zeros((1, 1024), dtype=np.float32))

    report = audit_ocr_artifacts(index, canonical_path)
    artifact = report["legacy_artifacts"][0]
    assert artifact["artifact_class"] == "diagnostic_legacy"
    assert artifact["eligible"] is False
    assert artifact["embedding"]["rows_match"] is True
    assert artifact["quality"]["mojibake_rows"] == 1
    assert report["status"] == "blocked"


def test_full_local_sources_merge_with_stable_schema(tmp_path: Path):
    canonical = _canonical()
    canonical_path = tmp_path / "canonical.parquet"
    canonical.to_parquet(canonical_path, index=False)
    index = tmp_path / "index"
    _write_full_pack_source(index, canonical_path)
    output = tmp_path / "out"

    manifest = build_global_index(index, canonical_path, output)

    assert manifest["status"] == "ready"
    table = pd.read_parquet(output / "retrieval.parquet")
    assert tuple(table.columns) == OUTPUT_COLUMNS
    assert len(table) == len(EXPECTED_PACKS)
    assert table["embedding_row"].tolist() == list(range(len(EXPECTED_PACKS)))
    assert set(table["source_pack"]) == set(EXPECTED_PACKS)
    assert np.load(output / "embeddings.npy").shape == (len(EXPECTED_PACKS), 1024)


def test_provisional_pilot_is_not_promoted(tmp_path: Path):
    canonical = _canonical()
    canonical_path = tmp_path / "canonical.parquet"
    canonical.to_parquet(canonical_path, index=False)
    index = tmp_path / "index"
    pilot = index / "modality_global_v2" / "ocr_pilot_l21"
    (pilot / "packs" / "L21").mkdir(parents=True)
    (pilot / "manifest.json").write_text(json.dumps({
        "schema_version": "hcmai.ocr_global_v2",
        "status": "completed",
        "engine": {"api_used": False, "network_allowed": False},
        "scope": {"mode": "pilot"},
        "packs": {"L21": {"status": "completed", "selected_rows": 1}},
    }), encoding="utf-8")

    manifest = build_global_index(index, canonical_path, tmp_path / "out")
    assert manifest["status"] == "blocked"
    assert not (tmp_path / "out" / "retrieval.parquet").exists()
    assert "L21" in manifest["coverage"]["missing_packs"]
