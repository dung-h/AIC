import json
from types import SimpleNamespace

import importlib


readiness = importlib.import_module("src.eval.build_architecture_readiness")


def _synthetic_manifests(tmp_path):
    payloads = {
        "asr": {
            "protocol": "synthetic modality preflight",
            "passed": False,
            "sources": {"asr": "global_merged_v2"},
            "coverage": {
                "asr_observed": ["k01"],
                "asr_missing": ["l21"],
                "asr_observed_video_count": 2,
                "asr_missing_videos": ["L21_V001", "L21_V002"],
            },
            "asr": [
                {"pack": "k01", "video_ids": ["K01_V001", "K01_V002"]},
                {"pack": "l21", "video_ids": []},
            ],
            "errors": [{"code": "asr_pack_coverage_incomplete"}],
        },
        "ocr": {},
    }
    paths = {}
    for modality, payload in payloads.items():
        path = tmp_path / f"{modality}_preflight.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[modality] = path
    return {
        modality: json.loads(path.read_text(encoding="utf-8"))
        for modality, path in paths.items()
    }


def test_partial_manifests_have_pack_video_states_without_negative_evidence(tmp_path):
    manifests = _synthetic_manifests(tmp_path)
    coverage = readiness.build_modality_coverage(
        ["K01_V001", "K01_V002", "L21_V001", "L21_V002"],
        manifests,
        visual_ready=True,
        visual_observed_video_ids=["K01_V001", "K01_V002", "L21_V001", "L21_V002"],
        expected_packs=["k01", "l21"],
        expected_video_count=4,
    )

    assert coverage["schema"] == {
        "name": "hcmai.modality_coverage",
        "version": 1,
    }
    assert coverage["evidence_policy"] == {
        "missing_modality_is_negative_evidence": False,
        "missing_modality_semantics": "unknown_not_negative",
        "negative_evidence_count": 0,
    }
    modalities = coverage["modalities"]
    assert modalities["visual"]["state"] == "usable"
    assert modalities["asr"]["state"] == "partial"
    assert modalities["asr"]["video_coverage"] == {
        "expected": 4,
        "observed": 2,
        "ratio": 0.5,
    }
    assert modalities["ocr"]["state"] == "blocked"
    assert "modality_index_missing" in modalities["ocr"]["reason_codes"]
    assert modalities["asr"]["pack_coverage"]["k01"]["state"] == "partial"
    assert modalities["asr"]["pack_coverage"]["l21"]["state"] == "blocked"
    for modality in modalities.values():
        assert modality["negative_evidence"] == []
        assert modality["missing_is_unknown"] is True
        for pack in modality["pack_coverage"].values():
            assert pack["negative_evidence"] == []


def test_incomplete_coverage_keeps_promotion_gate_false(tmp_path, monkeypatch):
    manifests = _synthetic_manifests(tmp_path)
    expected_ids = ["K01_V001", "K01_V002", "L21_V001", "L21_V002"]

    class _Registry:
        def get(self, name):
            return SimpleNamespace(scope="global")

        def snapshot(self):
            return {
                "catalog": {},
                "canonical_frames": {},
                "asr": {},
                "ocr": {},
            }

    monkeypatch.setattr(readiness, "build_catalog_preflight", lambda **_: _Registry())
    monkeypatch.setattr(
        readiness,
        "_canonical_mapping_preflight",
        lambda _: {"passed": True, "rows": 4, "videos": 4, "errors": []},
    )
    monkeypatch.setattr(
        readiness,
        "_canonical_video_inventory",
        lambda _: {
            "catalog_video_ids": expected_ids,
            "keyframe_video_ids": expected_ids,
            "errors": [],
        },
    )

    def _preflight(_, *, active_modalities, **__):
        return manifests[active_modalities[0]]

    monkeypatch.setattr(readiness, "run_modality_index_preflight", _preflight)
    report = readiness.build_report(tmp_path)

    assert report["schema"] == {
        "name": "hcmai.architecture_readiness",
        "version": 3,
    }
    coverage = report["modality_coverage"]
    assert coverage["modalities"]["visual"]["state"] == "usable"
    assert coverage["modalities"]["asr"]["state"] == "partial"
    assert coverage["modalities"]["ocr"]["state"] == "blocked"
    assert coverage["evidence_policy"]["negative_evidence_count"] == 0
    assert report["release_gates"]["visual_global"] is True
    assert report["release_gates"]["asr_global"] is False
    assert report["release_gates"]["ocr_global"] is False
    assert report["release_gates"]["promotion_allowed"] is False
    assert report["artifacts"]["asr"]["production_ready"] is False
    assert report["artifacts"]["ocr"]["production_ready"] is False
