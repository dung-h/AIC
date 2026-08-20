import json
from pathlib import Path

from src.eval.build_component_bottleneck_report import build_report


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_preserves_explicit_counts_and_provenance(tmp_path):
    readiness = tmp_path / "readiness.json"
    decomposition = tmp_path / "decomposition.json"
    trake = tmp_path / "trake.json"
    provider = tmp_path / "provider.json"
    _write(readiness, {
        "release_gates": {"catalog_global": True, "canonical_frames_global": True, "asr_global": False, "ocr_global": False},
        "artifacts": {
            "asr": {"index_preflight": {"coverage": {"asr_observed_video_count": 4, "canonical_expected_video_count": 10, "asr_video_coverage_ratio": 0.4}}},
            "ocr": {"index_preflight": {"coverage": {"ocr_observed_video_count": 7, "canonical_expected_video_count": 10, "ocr_video_coverage_ratio": 0.7}}},
        },
    })
    _write(decomposition, {"baseline": {"video_recall": {"r20": {"value": 0.5, "present": 5, "total": 10}}}})
    _write(trake, {"status": "blocked"})
    _write(provider, [{"provider": "local", "cases": 3, "answered": 2, "latency_ms": {"p95": 1.2}}])
    report = build_report(tmp_path, readiness, decomposition, trake, provider)
    asr = next(c for c in report["components"] if c["component_id"] == "modality_indexes")
    metric = next(m for m in asr["metrics"] if m["metric_id"] == "asr_video_coverage")
    assert metric["numerator"] == 4
    assert metric["denominator"] == 10
    assert metric["value"] == 0.4
    assert metric["provenance"] == "readiness.json"


def test_missing_metric_is_blocked_and_not_inferred(tmp_path):
    readiness = tmp_path / "readiness.json"
    _write(readiness, {"release_gates": {}, "artifacts": {}})
    report = build_report(tmp_path, readiness, tmp_path / "missing.json", tmp_path / "missing2.json", tmp_path / "missing3.json")
    qna = next(c for c in report["components"] if c["component_id"] == "qna_retrieval_selector_answer")
    metric = qna["metrics"][0]
    assert metric["status"] == "blocked"
    assert metric["value"] is None
    assert metric["numerator"] is None
    assert metric["denominator"] is None
    assert report["rules"]["no_metric_inference"] is True


def test_ranking_is_deterministic_and_explains_blockers(tmp_path):
    readiness = tmp_path / "readiness.json"
    _write(readiness, {"release_gates": {"catalog_global": True, "canonical_frames_global": True}, "artifacts": {}})
    report = build_report(tmp_path, readiness, tmp_path / "qna.json", tmp_path / "trake.json", tmp_path / "provider.json")
    ranking = report["bottleneck_ranking"]
    assert [row["rank"] for row in ranking] == list(range(1, len(ranking) + 1))
    assert all(row["next_action"] for row in ranking)
    assert any("blocked" in row["reason"] for row in ranking)
