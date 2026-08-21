from __future__ import annotations

import json
from pathlib import Path
import socket
import zipfile

import numpy as np
import pandas as pd
import pytest

import src.cli.competition_ready as competition_ready
from src.cli.competition_ready import PreflightConfig, main, run_preflight


@pytest.fixture(autouse=True)
def _isolate_host_import_probe(monkeypatch):
    """Unit tests validate preflight logic, not workstation cold-I/O speed."""
    ok = {"ok": True, "returncode": 0, "error": ""}
    base = tuple(competition_ready.REQUIRED_MODULES)
    monkeypatch.setattr(competition_ready, "_IMPORT_PROBE_CACHE", {
        base: ok,
        base + ("bitsandbytes",): ok,
    })


def _write_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl", "hidden_size": 3584}), encoding="utf-8"
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}), encoding="utf-8"
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")


def _write_modality(path: Path, name: str, canonical: Path) -> None:
    path.mkdir(parents=True)
    if name == "asr":
        rows = pd.DataFrame({
            "video_id": ["K01_V001"], "text": ["spoken"], "frame_idx": [10],
            "kf_n": [1], "pts_time": [0.4], "embedding_row": [0],
            "chunk_index": [0], "start": [0.0], "end": [1.0],
            "source_pack": ["K01"], "source_provenance": ["test"],
        })
        manifest_name = "asr_global_merge_v2_manifest.json"
    else:
        rows = pd.DataFrame({
            "video_id": ["K01_V001"], "ocr_text": ["screen"], "frame_idx": [10],
            "kf_n": [1], "pts_time": [0.4], "embedding_row": [0],
            "source_pack": ["K01"],
        })
        manifest_name = "manifest.json"
    rows.to_parquet(path / "retrieval.parquet", index=False)
    np.save(path / "embeddings.npy", np.ones((1, 4), dtype=np.float32))
    manifest = {
        "status": "ready",
        "canonical": {"validated": True, "path": str(canonical)},
        "scope": {"packs": list(__import__("src.cli.competition_ready", fromlist=["EXPECTED_PACKS"]).EXPECTED_PACKS), "video_count": 1},
        "embedding": {"shape": [1, 4]},
        "artifacts": {"retrieval": "retrieval.parquet", "embeddings": "embeddings.npy"},
    }
    if name == "ocr":
        manifest["coverage"] = {"canonical_frame_coverage": 0.5, "frame_complete": False}
    else:
        manifest["packs"] = {}
    (path / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")


def _ready_config(tmp_path: Path, *, provider: str = "local") -> PreflightConfig:
    canonical = tmp_path / "canonical.parquet"
    frame = pd.DataFrame({"video_id": ["K01_V001"], "kf_n": [1], "frame_idx": [10], "pts_time": [0.4]})
    frame.to_parquet(canonical, index=False)
    visual_map = tmp_path / "visual.parquet"
    frame.to_parquet(visual_map, index=False)
    visual = tmp_path / "visual.npy"
    np.save(visual, np.ones((1, 4), dtype=np.float32))
    asr, ocr = tmp_path / "asr", tmp_path / "ocr"
    _write_modality(asr, "asr", canonical)
    _write_modality(ocr, "ocr", canonical)
    model = tmp_path / "model"
    _write_model(model)
    keyframes = tmp_path / "keyframes"
    (keyframes / "K01_V001").mkdir(parents=True)
    (keyframes / "K01_V001" / "001.jpg").write_bytes(b"jpeg-probe")
    backbone = tmp_path / "hf" / "models--timm--test" / "snapshots" / "revision"
    backbone.mkdir(parents=True)
    (backbone / "open_clip_model.safetensors").write_bytes(b"weights")
    (backbone / "tokenizer.json").write_text("{}", encoding="utf-8")
    (backbone / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return PreflightConfig(
        project_root=tmp_path,
        canonical_map=canonical,
        keyframes_dir=keyframes,
        visual_indexes=(visual,),
        visual_maps=(visual_map,),
        visual_backbone_dirs=(backbone.parents[1],),
        asr_dir=asr,
        ocr_dir=ocr,
        provider=provider,
        local_model=model,
        output_dir=tmp_path / "out",
        expected_video_count=1,
        min_free_gb=0,
    )


def test_ready_local_preflight_is_offline_and_warns_for_sampled_ocr(tmp_path, monkeypatch):
    config = _ready_config(tmp_path)

    def forbid_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    report = run_preflight(config)
    assert report["ready"] is True
    assert report["exit_code"] == 0
    assert report["network_calls"] == 0
    assert report["warnings"] == ["ocr_frame_coverage"]
    assert not report["blockers"]


def test_missing_artifacts_fail_closed_and_main_returns_nonzero(tmp_path, capsys):
    code = main([
        "--canonical-map", str(tmp_path / "missing.parquet"),
        "--visual-index", str(tmp_path / "missing.npy"),
        "--visual-map", str(tmp_path / "missing-map.parquet"),
        "--asr-dir", str(tmp_path / "missing-asr"),
        "--ocr-dir", str(tmp_path / "missing-ocr"),
        "--local-model", str(tmp_path / "missing-model"),
        "--output-dir", str(tmp_path / "out"),
        "--expected-video-count", "0", "--min-free-gb", "0",
    ])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["ready"] is False
    assert {"canonical_map", "visual_indexes", "asr_index", "ocr_index", "answer_provider"} <= set(report["blockers"])


def test_l_only_preflight_accepts_full_metadata_but_requires_only_l_assets(tmp_path):
    """The portable preselection deploy installs L images, not every K pack."""
    canonical = tmp_path / "canonical.parquet"
    pd.DataFrame({
        "video_id": ["L21_V001", "K01_V001"],
        "kf_n": [1, 1],
        "frame_idx": [10, 10],
        "pts_time": [0.4, 0.4],
    }).to_parquet(canonical, index=False)
    summary, pairs = competition_ready._canonical_summary(
        canonical, active_video_prefixes=("L",), collect_pairs=True,
    )
    assert summary["videos"] == 1
    assert pairs == {("L21_V001", 10)}

    keyframes = tmp_path / "keyframes"
    (keyframes / "L21_V001").mkdir(parents=True)
    (keyframes / "L21_V001" / "001.jpg").write_bytes(b"jpeg-probe")
    config = PreflightConfig(
        canonical_map=canonical,
        keyframes_dir=keyframes,
        active_video_prefixes=("L",),
        expected_video_count=1,
    )
    builder = competition_ready.ReportBuilder()
    competition_ready._check_keyframes(builder, config)
    assert builder.checks[0]["status"] == "pass"

    for name in ("asr", "ocr"):
        directory = tmp_path / name
        _write_modality(directory, name, canonical)
        manifest_path = directory / (
            "asr_global_merge_v2_manifest.json" if name == "asr" else "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        l_packs = [f"L{number:02d}" for number in range(21, 31)]
        manifest["scope"] = {"packs": l_packs, "video_count": 1}
        if name == "asr":
            manifest["scope"]["video_ids"] = ["L21_V001"]
        else:
            manifest["packs"] = {
                pack: {"canonical_videos": 1 if pack == "L21" else 0}
                for pack in l_packs
            }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        modality_builder = competition_ready.ReportBuilder()
        competition_ready._check_modality(
            modality_builder, name, directory, expected_videos=1, active_video_prefixes=("L",),
        )
        assert modality_builder.checks[0]["status"] == "pass"


def test_explicit_api_provider_never_prints_secret(tmp_path, monkeypatch):
    config = _ready_config(tmp_path, provider="openai")
    secret = "super-secret-value-must-not-leak"
    monkeypatch.setenv("VLM_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("VLM_API_KEY", secret)
    monkeypatch.setenv("VLM_MODEL", "strong-vlm")
    report = run_preflight(config)
    rendered = json.dumps(report)
    assert report["ready"] is True
    assert secret not in rendered
    assert "https://provider.invalid" not in rendered
    provider = next(check for check in report["checks"] if check["name"] == "answer_provider")
    assert provider["details"]["configured"] is True


def test_query_and_output_package_are_structurally_and_canonically_validated(tmp_path):
    base = _ready_config(tmp_path)
    queries = tmp_path / "queries.csv"
    queries.write_text("query_id,query,question\nq1,a rainy scene,what weather\n", encoding="utf-8")
    package = tmp_path / "submission.json"
    package.write_text(json.dumps({
        "task": "qa",
        "queries": {"q1": [{"video_id": "K01_V001", "frame_id": 10, "answer": "rain"}]},
    }), encoding="utf-8")
    config = PreflightConfig(**{**base.__dict__, "query_path": queries, "output_package": package})
    report = run_preflight(config)
    assert report["ready"] is True
    names = {item["name"]: item for item in report["checks"]}
    assert names["query_input"]["status"] == "pass"
    assert names["output_package"]["details"]["canonical_validated"] is True


def test_noncanonical_output_package_is_a_blocker(tmp_path):
    base = _ready_config(tmp_path)
    package = tmp_path / "submission.json"
    package.write_text(json.dumps({
        "task": "qa",
        "queries": {"q1": [{"video_id": "K01_V001", "frame_id": 999, "answer": "rain"}]},
    }), encoding="utf-8")
    config = PreflightConfig(**{**base.__dict__, "output_package": package})
    report = run_preflight(config)
    assert report["ready"] is False
    assert "output_package" in report["blockers"]


def test_duplicate_ranked_answer_is_a_blocker(tmp_path):
    base = _ready_config(tmp_path)
    answer = {"video_id": "K01_V001", "frame_id": 10, "answer": "rain"}
    package = tmp_path / "submission.json"
    package.write_text(json.dumps({"task": "qa", "queries": {"q1": [answer, answer]}}), encoding="utf-8")
    report = run_preflight(PreflightConfig(**{**base.__dict__, "output_package": package}))
    assert report["ready"] is False
    assert "output_package" in report["blockers"]


def test_official_text_queries_and_headerless_mixed_zip_are_validated(tmp_path):
    base = _ready_config(tmp_path)
    queries = tmp_path / "queries"
    queries.mkdir()
    (queries / "query-1-kis.txt").write_text("a rainy scene", encoding="utf-8")
    (queries / "query-2-qa.txt").write_text(
        "Description: a weather graphic\nQuestion: What is written on screen?\n",
        encoding="utf-8",
    )
    package = tmp_path / "submission.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("submission/", b"")
        archive.writestr("submission/query-1-kis.csv", "K01_V001,10\n")
        archive.writestr("submission/query-2-qa.csv", "K01_V001,10,rain\n")

    report = run_preflight(PreflightConfig(**{
        **base.__dict__, "query_path": queries, "output_package": package,
    }))

    assert report["ready"] is True
    names = {item["name"]: item for item in report["checks"]}
    assert names["query_input"]["details"]["queries"] == 2
    assert names["output_package"]["details"]["answers"] == 2


def test_official_zip_rejects_wrong_root_and_bad_trake_order(tmp_path):
    base = _ready_config(tmp_path)
    for member, content in (
        ("query-1-kis.csv", "K01_V001,10\n"),
        ("submission/query-1-trake.csv", "K01_V001,10,10\n"),
    ):
        package = tmp_path / ("bad-root.zip" if not member.startswith("submission/") else "bad-order.zip")
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(member, content)
        report = run_preflight(PreflightConfig(**{**base.__dict__, "output_package": package}))
        assert report["ready"] is False
        assert "output_package" in report["blockers"]
