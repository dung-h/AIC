import json
from pathlib import Path
from types import SimpleNamespace

from src.artifacts import ArtifactStatus
from src.eval.build_architecture_readiness import build_report
import src.eval.build_architecture_readiness as readiness


def test_readiness_report_is_fail_closed_for_partial_specialists(tmp_path):
    report = build_report(tmp_path)
    assert report["runtime"]["research_routes_enabled"] is False
    assert report["fallback_contract"]["automatic_online_to_offline_embedder_fallback"] == "disabled"
    assert report["release_gates"]["promotion_allowed"] is False


def _write_global_manifest(root: Path, modality: str, *, video_count: int = 1478, rows: int = 1234) -> Path:
    if modality == "asr":
        path = root / "data" / "index" / "modality_global_v2" / "asr_global_merged_v2" / "asr_global_merge_v2_manifest.json"
        scope_name = "full_corpus"
        manifest_rows = {"metadata": rows, "embedding": rows}
        coverage = {}
        schema_version = "hcmai.asr_global_merge_v2"
    else:
        path = root / "data" / "index" / "modality_global_v2" / "ocr_global_merged_v2" / "manifest.json"
        scope_name = "full_corpus_video_coverage"
        manifest_rows = {}
        coverage = {
            "canonical_videos": video_count,
            "covered_videos": video_count,
            "ocr_row_count": rows,
            "video_coverage": 1.0,
            "frame_complete": False,
        }
        schema_version = "hcmai.ocr_global_full_merge_v2"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "ready",
        "scope": {"name": scope_name, "video_count": video_count},
        "rows": manifest_rows,
        "coverage": coverage,
        "embedding": {"dim": 1024, "shape": [rows, 1024]},
        "schema_version": schema_version,
        "index_id": f"{modality}-global-test",
        "canonical": {"validated": True, "videos": video_count},
    }), encoding="utf-8")
    return path


def _patch_readiness_inputs(monkeypatch, tmp_path, *, source: str):
    video_ids = [f"K01_V{index + 1:04d}" for index in range(1478)]
    _write_global_manifest(tmp_path, "asr", rows=76593)
    _write_global_manifest(tmp_path, "ocr", rows=38636)

    class _Registry:
        def get(self, name):
            return SimpleNamespace(scope="global")

        def snapshot(self):
            # Deliberately stale specialist values: readiness must replace
            # these with the global-manifest contract.
            return {
                "catalog": {},
                "canonical_frames": {},
                "asr": ArtifactStatus(
                    "asr", True, "partial", "/legacy/catalog.sqlite", 51352,
                    coverage="videos=605/1478;rows=51352",
                ).to_dict(),
                "ocr": ArtifactStatus(
                    "ocr", True, "partial", "/legacy/catalog.sqlite", 74,
                    coverage="videos=15/1478;rows=74",
                ).to_dict(),
            }

    monkeypatch.setattr(readiness, "build_catalog_preflight", lambda **_: _Registry())
    monkeypatch.setattr(
        readiness,
        "_canonical_mapping_preflight",
        lambda _: {"passed": True, "rows": 100, "videos": 1478, "errors": []},
    )
    monkeypatch.setattr(
        readiness,
        "_canonical_video_inventory",
        lambda _: {
            "catalog_video_ids": video_ids,
            "keyframe_video_ids": video_ids,
            "errors": [],
        },
    )

    def _preflight(_, *, active_modalities, **__):
        modality = active_modalities[0]
        return {
            "passed": True,
            "sources": {modality: source},
            "scope": {"name": "full_corpus", "is_full_corpus": True},
            "expected_packs": ["k01"],
            "coverage": {
                f"{modality}_observed": ["k01"],
                f"{modality}_missing": [],
                f"{modality}_observed_video_count": 1478,
                f"{modality}_missing_videos": [],
                f"{modality}_video_coverage_ratio": 1.0,
            },
            modality: [{"pack": "k01", "video_ids": video_ids, "metadata_rows": 1}],
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr(readiness, "run_modality_index_preflight", _preflight)


def test_global_manifest_replaces_stale_sqlite_specialist_fields(tmp_path, monkeypatch):
    _patch_readiness_inputs(monkeypatch, tmp_path, source="global_merged_v2")

    report = build_report(tmp_path)

    for modality, rows in (("asr", 76593), ("ocr", 38636)):
        modality_coverage = report["modality_coverage"]["modalities"][modality]
        assert modality_coverage["video_coverage"] == {
            "expected": 1478,
            "observed": 1478,
            "ratio": 1.0,
        }
        artifact = report["artifacts"][modality]
        if modality == "ocr":
            # Global video coverage is not equivalent to frame-complete OCR;
            # the fixture intentionally declares frame_complete=False.
            assert artifact["ready"] is False
            assert artifact["scope"] == "partial"
            assert artifact["reason"] == "ocr_frame_coverage_incomplete"
            assert modality_coverage["production_ready"] is False
            assert modality_coverage["frame_coverage"]["complete"] is False
        else:
            assert artifact["ready"] is True
            assert artifact["scope"] == "global"
        assert artifact["path"].startswith(str(tmp_path / "data" / "index" / "modality_global_v2"))
        assert artifact["row_count"] == rows
        assert artifact["coverage"].startswith("videos=1478/1478;")
        assert artifact["metadata"]["authority"] == "global_manifest_v2"
        assert "605/1478" not in artifact["coverage"]
        assert "15/1478" not in artifact["coverage"]
        assert modality_coverage["pack_coverage"]["k01"]["state"] == (
            "partial" if modality == "ocr" else "usable"
        )


def test_legacy_preflight_cannot_override_global_manifest_requirement(tmp_path, monkeypatch):
    _patch_readiness_inputs(monkeypatch, tmp_path, source="legacy_pack")

    report = build_report(tmp_path)

    assert report["release_gates"]["asr_global"] is False
    assert report["release_gates"]["ocr_global"] is False
    for modality in ("asr", "ocr"):
        artifact = report["artifacts"][modality]
        assert artifact["ready"] is False
        assert artifact["scope"] == "missing"
        assert artifact["path"].startswith(str(tmp_path / "data" / "index" / "modality_global_v2"))
        assert artifact["metadata"]["preflight_source"] == "legacy_pack"
        assert "authoritative_global_source_required" in {
            item["code"] for item in artifact["index_preflight"]["errors"]
        }
