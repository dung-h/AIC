from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.ocr_global_v2 import (
    OCRGlobalV2Runner,
    QwenLocalOCRBackend,
    clean_ocr_text,
    load_canonical,
    repair_mojibake,
    select_scope,
)
from src.indexing.ocr_global_v2 import (
    OUTPUT_COLUMNS as GLOBAL_OUTPUT_COLUMNS,
    OCRGlobalV2Error,
    build_global_index,
    validate_artifacts,
)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"video_id": "K01_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 0.0},
            {"video_id": "K01_V001", "kf_n": 2, "frame_idx": 20, "pts_time": 3.0},
            {"video_id": "L21_V001", "kf_n": 1, "frame_idx": 30, "pts_time": 1.0},
        ]
    )


class FakeOCR:
    def __init__(self, *, fail_kf: int | None = None):
        self.calls: list[tuple[str, str]] = []
        self.fail_kf = fail_kf

    def recognize(self, image_path: str, prompt: str) -> str:
        self.calls.append((image_path, prompt))
        kf_n = int(Path(image_path).stem)
        if self.fail_kf == kf_n:
            raise RuntimeError("synthetic backend interruption")
        return {1: "HÃ  Ná»™i", 2: "NONE"}.get(kf_n, "PRICE 25")


class FakeEmbedder:
    def __init__(self, dim: int = 3):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts, *, batch_size: int) -> np.ndarray:
        self.calls.append(list(texts))
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _frames(root: Path) -> None:
    for video_id, kfs in (("K01_V001", (1, 2)), ("L21_V001", (1,))):
        folder = root / video_id
        folder.mkdir(parents=True)
        for kf_n in kfs:
            (folder / f"{kf_n:03d}.jpg").write_bytes(b"fake-image")


def _runner(output: Path, backend, embedder) -> OCRGlobalV2Runner:
    return OCRGlobalV2Runner(
        output_dir=output,
        model_path=output / "missing-qwen-model",
        embed_model_path=output / "missing-bge-model",
        backend=backend,
        embedder=embedder,
        embedding_dim=3,
        batch_size=2,
    )


def test_mojibake_repair_and_no_text_normalization():
    assert repair_mojibake("HÃ  Ná»™i") == "Hà Nội"
    assert clean_ocr_text("```text\nPRICE 25\n```") == "PRICE 25"
    assert clean_ocr_text("no readable text is visible") == ""
    assert clean_ocr_text("Hàn Quốc — 안녕하세요") == "Hàn Quốc — 안녕하세요"


def test_canonical_contract_and_pilot_selection():
    canonical = load_canonical(_canonical())
    assert list(canonical.columns[:5]) == ["video_id", "pack", "kf_n", "frame_idx", "pts_time"]
    pilot = select_scope(canonical, mode="pilot", pilot_video_limit=2)
    assert list(pilot["video_id"]) == ["K01_V001", "L21_V001"]
    assert list(pilot["frame_idx"]) == [20, 30]

    duplicate = pd.concat([_canonical(), _canonical().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_canonical(duplicate)


def test_dry_run_does_not_load_model_or_call_backend(tmp_path: Path):
    report = _runner(tmp_path / "ocr", FakeOCR(), FakeEmbedder()).run(
        _canonical(), tmp_path / "frames", mode="full", dry_run=True, execute=False
    )
    assert report["status"] == "dry_run"
    assert report["selection"]["rows"] == 3
    assert not (tmp_path / "ocr" / "attempt_manifest.jsonl").exists()


def test_full_materializer_records_no_text_and_builds_nonempty_index(tmp_path: Path):
    frame_root = tmp_path / "frames"
    _frames(frame_root)
    output = tmp_path / "modality_global_v2" / "ocr"
    backend = FakeOCR()
    embedder = FakeEmbedder()
    report = _runner(output, backend, embedder).run(
        _canonical(), frame_root, mode="full", execute=True
    )

    assert report["status"] == "completed"
    assert report["coverage"]["attempted_rows"] == 3
    assert report["coverage"]["no_text_rows"] == 1
    assert report["coverage"]["retrieval_rows"] == 2
    assert report["coverage"]["global_retrieval_rows"] == 2
    assert report["engine"]["network_allowed"] is False
    attempts = [json.loads(line) for line in (output / "attempt_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["status"] for row in attempts} == {"text", "no_text"}
    assert any(row["status"] == "no_text" and row["ocr_text"] == "" for row in attempts)

    k01 = pd.read_parquet(output / "packs" / "K01" / "retrieval.parquet")
    l21 = pd.read_parquet(output / "packs" / "L21" / "retrieval.parquet")
    assert list(k01["frame_idx"]) == [10]
    assert list(l21["frame_idx"]) == [30]
    assert np.load(output / "packs" / "K01" / "embeddings.npy").shape == (1, 3)
    assert len(pd.read_parquet(output / "retrieval.parquet")) == 2
    assert np.load(output / "embeddings.npy").shape == (2, 3)
    assert json.loads((output / "packs" / "K01" / "checkpoint.json").read_text())["status"] == "completed"


def test_resume_retries_only_error_rows_and_keeps_pack_checkpoint(tmp_path: Path):
    frame_root = tmp_path / "frames"
    _frames(frame_root)
    output = tmp_path / "ocr"
    first = _runner(output, FakeOCR(fail_kf=2), FakeEmbedder()).run(
        _canonical().iloc[:2], frame_root, mode="full", execute=True
    )
    assert first["status"] == "blocked"
    assert first["coverage"]["error_rows"] == 1

    second_backend = FakeOCR()
    second = _runner(output, second_backend, FakeEmbedder()).run(
        _canonical().iloc[:2], frame_root, mode="full", execute=True, resume=True
    )
    assert second["status"] == "completed"
    assert len(second_backend.calls) == 1
    assert int(Path(second_backend.calls[0][0]).stem) == 2
    checkpoint = json.loads((output / "packs" / "K01" / "checkpoint.json").read_text())
    assert checkpoint["attempted_rows"] == 2
    assert checkpoint["error_rows"] == 0
    # A second resume is safe: successful rows are not appended again and the
    # append-only retry history remains readable.
    third_backend = FakeOCR()
    third = _runner(output, third_backend, FakeEmbedder()).run(
        _canonical().iloc[:2], frame_root, mode="full", execute=True, resume=True
    )
    assert third["status"] == "completed"
    assert third_backend.calls == []


def test_all_no_text_fails_closed_without_retrieval_index(tmp_path: Path):
    frame_root = tmp_path / "frames"
    _frames(frame_root)

    class NoText(FakeOCR):
        def recognize(self, image_path: str, prompt: str) -> str:
            self.calls.append((image_path, prompt))
            return "NONE"

    output = tmp_path / "ocr"
    report = _runner(output, NoText(), FakeEmbedder()).run(
        _canonical().iloc[:1], frame_root, mode="full", execute=True
    )
    assert report["status"] == "blocked"
    assert report["coverage"]["retrieval_rows"] == 0
    assert not (output / "packs" / "K01" / "retrieval.parquet").exists()


def test_missing_local_qwen_model_has_no_fallback(tmp_path: Path):
    with pytest.raises(RuntimeError, match="local OCR backend unavailable"):
        QwenLocalOCRBackend(tmp_path / "missing")


def _adapter_canonical() -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 0.0},
        {"video_id": "K01_V001", "kf_n": 2, "frame_idx": 20, "pts_time": 10.0},
        {"video_id": "L21_V001", "kf_n": 1, "frame_idx": 30, "pts_time": 0.0},
    ])


def _write_adapter_inputs(root: Path, *, metadata: pd.DataFrame | None = None,
                          embeddings: np.ndarray | None = None,
                          manifest: dict | None = None) -> tuple[Path, Path, Path, Path]:
    canonical_path = root / "canonical.parquet"
    metadata_path = root / "ocr.parquet"
    embeddings_path = root / "embeddings.npy"
    manifest_path = root / "source_manifest.json"
    _adapter_canonical().to_parquet(canonical_path, index=False)
    (metadata if metadata is not None else pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "ocr_text": "K01 text"},
    ])).to_parquet(metadata_path, index=False)
    np.save(embeddings_path, embeddings if embeddings is not None else np.ones((1, 1024), dtype=np.float32))
    payload = manifest or {
        "schema_version": "ocr_local_corpus_v1",
        "status": "completed",
        "mode": "full",
        "sampling": {"interval_seconds": 10.0},
        "engine": {"api_used": False, "network_allowed": False},
        "canonical": {"videos": 2, "packs": ["K01", "L21"]},
        "coverage": {"full_canonical_video_coverage": True, "no_text_videos": ["L21_V001"]},
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return metadata_path, embeddings_path, manifest_path, canonical_path


def test_sampled_global_adapter_materializes_exact_schema_and_coverage(tmp_path: Path):
    metadata, embeddings, manifest, canonical = _write_adapter_inputs(tmp_path)
    output = tmp_path / "ocr_global_v2"

    report = build_global_index(
        metadata, embeddings, manifest, canonical, output,
        expected_packs=("K01", "L21"), expected_video_count=2,
    )

    assert report["status"] == "ready"
    table = pd.read_parquet(output / "retrieval.parquet")
    assert tuple(table.columns) == GLOBAL_OUTPUT_COLUMNS
    assert table[["video_id", "kf_n", "frame_idx", "pts_time", "ocr_text"]].to_dict("records") == [{
        "video_id": "K01_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "ocr_text": "K01 text",
    }]
    assert np.load(output / "embeddings.npy").shape == (1, 1024)
    assert report["sampling"]["sample_interval_seconds"] == 10.0
    assert report["coverage"]["all_packs_covered"] is True
    assert report["coverage"]["by_pack"]["K01"]["sampled_ocr_rows"] == 1
    assert report["coverage"]["by_video"]["L21_V001"]["status"] == "no_text"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "ready"
    assert sorted(path.name for path in output.iterdir()) == [
        "embeddings.npy", "manifest.json", "retrieval.parquet",
    ]


def test_materialized_output_is_readable_by_existing_ocr_registry_contract(tmp_path: Path):
    metadata, embeddings, manifest, canonical = _write_adapter_inputs(tmp_path)
    index_root = tmp_path / "index"
    index_root.mkdir()
    canonical_in_index = index_root / "global_keyframes.parquet"
    _adapter_canonical().to_parquet(canonical_in_index, index=False)
    output = index_root / "modality_global_v2" / "ocr_global_merged_v2"

    report = build_global_index(
        metadata, embeddings, manifest, canonical_in_index, output,
        expected_packs=("K01", "L21"), expected_video_count=2,
    )
    assert report["status"] == "ready"

    from src.reranking.modality_index_registry import ModalityIndexRegistry

    loaded_embeddings, loaded_metadata, info = ModalityIndexRegistry(index_root).load_ocr(
        expected_packs=("k01", "l21"),
    )
    assert loaded_embeddings.shape == (1, 1024)
    assert list(loaded_metadata["ocr_text"]) == ["K01 text"]
    assert info["sample_interval_seconds"] == 10.0
    assert info["covered_video_count"] == 2


def test_explicit_no_text_metadata_record_is_allowed_and_not_retrievable(tmp_path: Path):
    metadata = pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 0.0, "ocr_text": "K01 text"},
        {"video_id": "L21_V001", "kf_n": None, "frame_idx": None, "pts_time": None,
         "ocr_text": "", "status": "no_text"},
    ])
    manifest = {
        "schema_version": "ocr_local_corpus_v1",
        "status": "completed",
        "mode": "full",
        "sampling": {"interval_seconds": 10.0},
        "engine": {"api_used": False, "network_allowed": False},
        "canonical": {"videos": 2, "packs": ["K01", "L21"]},
        "coverage": {"full_canonical_video_coverage": True},
    }
    metadata_path, embeddings_path, manifest_path, canonical = _write_adapter_inputs(
        tmp_path, metadata=metadata, embeddings=np.ones((2, 1024), dtype=np.float32), manifest=manifest,
    )

    validated = validate_artifacts(
        metadata_path, embeddings_path, manifest_path, canonical,
        expected_packs=("K01", "L21"), expected_video_count=2,
    )
    assert len(validated.metadata) == 1
    assert validated.no_text_videos == ("L21_V001",)
    assert validated.embeddings.shape == (1, 1024)


def test_global_adapter_fails_closed_for_missing_video_coverage(tmp_path: Path):
    metadata, embeddings, manifest, canonical = _write_adapter_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["coverage"].pop("no_text_videos")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OCRGlobalV2Error, match="coverage incomplete"):
        validate_artifacts(
            metadata, embeddings, manifest, canonical,
            expected_packs=("K01", "L21"), expected_video_count=2,
        )
    report = build_global_index(
        metadata, embeddings, manifest, canonical, tmp_path / "blocked",
        expected_packs=("K01", "L21"), expected_video_count=2,
    )
    assert report["status"] == "blocked"
    assert not (tmp_path / "blocked" / "retrieval.parquet").exists()


@pytest.mark.parametrize("mutation,match", [
    ("api", "api_used=true"),
    ("dim", "embedding shape"),
    ("duplicate", "duplicate OCR identity"),
    ("outside", "outside canonical map"),
])
def test_global_adapter_rejects_provenance_shape_duplicates_and_unknown_rows(
    tmp_path: Path, mutation: str, match: str,
):
    metadata, embeddings, manifest, canonical = _write_adapter_inputs(tmp_path)
    if mutation == "api":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["engine"]["api_used"] = True
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "dim":
        np.save(embeddings, np.ones((1, 3), dtype=np.float32))
    elif mutation == "duplicate":
        table = pd.read_parquet(metadata)
        pd.concat([table, table], ignore_index=True).to_parquet(metadata, index=False)
        np.save(embeddings, np.ones((2, 1024), dtype=np.float32))
    elif mutation == "outside":
        table = pd.read_parquet(metadata)
        table.loc[0, "kf_n"] = 999
        table.to_parquet(metadata, index=False)

    with pytest.raises(OCRGlobalV2Error, match=match):
        validate_artifacts(
            metadata, embeddings, manifest, canonical,
            expected_packs=("K01", "L21"), expected_video_count=2,
        )
