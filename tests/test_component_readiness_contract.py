import json
import sqlite3
from pathlib import Path

import pytest

from src.eval.build_architecture_readiness import build_report
from src.runtime_context import RuntimeContext
from src.runtime_policy import RuntimePolicy


def _catalog(root: Path, *, invalid: bool = False, video_count: int = 1) -> Path:
    path = root / "data" / "catalog" / "aic2026_catalog.sqlite"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE videos(video_id TEXT PRIMARY KEY, keyframe_count INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE keyframes(
            video_id TEXT NOT NULL, kf_n INTEGER NOT NULL, frame_idx INTEGER NOT NULL,
            pts_time REAL NOT NULL, PRIMARY KEY(video_id, kf_n)
        );
        CREATE TABLE asr_chunks(chunk_id INTEGER PRIMARY KEY, video_id TEXT);
        CREATE TABLE ocr_records(ocr_id INTEGER PRIMARY KEY, video_id TEXT);
        CREATE TABLE embedding_shards(shard_id INTEGER PRIMARY KEY);
        """
    )
    for index in range(video_count):
        video_id = f"K01_V{index + 1:03d}"
        conn.execute("INSERT INTO videos(video_id, keyframe_count) VALUES (?, 1)", (video_id,))
        conn.execute(
            "INSERT INTO keyframes(video_id, kf_n, frame_idx, pts_time) VALUES (?, ?, ?, ?)",
            (video_id, 0, -1 if invalid else index, 0.0),
        )
    conn.commit()
    conn.close()
    return path


def test_component_summary_has_single_owner_and_operational_contract(tmp_path):
    report = build_report(tmp_path)
    components = report["component_readiness"]
    expected = {
        "runtime_policy",
        "runtime_context",
        "modality_routing",
        "provider_choice",
        "canonical_frame_mapping",
        "fallback_policy",
        "architecture_runtime",
        "promotion_gate",
    }
    assert set(components) == expected
    for component in components.values():
        assert component["owner"]
        assert component["output"]
        assert component["definition_of_done"]
        assert isinstance(component["metrics"], dict)
        assert component["status"]
    assert components["promotion_gate"]["owner"].startswith(
        "src/eval/build_architecture_readiness.py:"
    )


def test_textual_architecture_reference_never_promotes_runtime(tmp_path):
    source = tmp_path / "src" / "pipelines" / "hcmai_pipeline.py"
    source.parent.mkdir(parents=True)
    source.write_text("# ArchitectureRuntime appears in a legacy note\n", encoding="utf-8")
    report = build_report(tmp_path)
    assert report["architecture_runtime"]["referenced_by_live_entrypoints"]
    assert report["architecture_runtime"]["status"] == "shadow_only"
    assert report["release_gates"]["architecture_runtime_live"] is False


def test_canonical_gate_rejects_positive_but_invalid_catalog(tmp_path):
    _catalog(tmp_path, invalid=True, video_count=1478)
    report = build_report(tmp_path)
    preflight = report["canonical_mapping_preflight"]
    assert preflight["rows"] == 1478
    assert preflight["invalid_value_rows"] == 1478
    assert preflight["passed"] is False
    assert report["release_gates"]["canonical_frames_global"] is False
    assert report["release_gates"]["promotion_allowed"] is False


def test_canonical_gate_rejects_partial_catalog_even_with_keyframes(tmp_path):
    _catalog(tmp_path, video_count=1)
    report = build_report(tmp_path)
    assert report["canonical_mapping_preflight"]["passed"] is False
    assert "video_count_mismatch" in " ".join(report["canonical_mapping_preflight"]["errors"])
    assert report["release_gates"]["catalog_global"] is False


def test_context_deep_freezes_nested_artifact_snapshot():
    artifacts = {"catalog": {"version": "v1", "packs": ["K01"]}}
    context = RuntimeContext.from_policy(RuntimePolicy(), artifact_snapshot=artifacts)
    artifacts["catalog"]["packs"].append("K02")
    with pytest.raises(TypeError):
        context.artifact_snapshot["catalog"]["version"] = "v2"
    with pytest.raises(AttributeError):
        context.artifact_snapshot["catalog"]["packs"].append("K03")
    assert context.artifact_snapshot["catalog"]["packs"] == ("K01",)


def test_live_proof_is_explicit_and_does_not_bypass_other_gates(tmp_path):
    proof = tmp_path / "results" / "architecture_runtime_live.json"
    proof.parent.mkdir(parents=True)
    proof.write_text(json.dumps({
        "status": "live",
        "scope": "full_corpus",
        "canonical_mapping": "validated",
    }), encoding="utf-8")
    report = build_report(tmp_path)
    assert report["architecture_runtime"]["status"] == "live"
    assert report["release_gates"]["architecture_runtime_live"] is True
    assert report["release_gates"]["promotion_allowed"] is False
