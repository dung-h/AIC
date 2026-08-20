import json
import os

# The repo's optional parquet dependency cache was built for Windows.  WSL
# does not expose os.add_dll_directory; the import only needs this no-op hook
# for the focused pure-contract tests below.
if not hasattr(os, "add_dll_directory"):
    os.add_dll_directory = lambda _path: None

from src.queryset.materialize_vqa_eval_v3 import (
    _load_ocr_sidecar,
    _ocr_evidence_mode,
    _parquet_rows_v3,
    _production_exclusion_reason,
)
from src.queryset.validate_vqa_benchmark_v3 import _evidence_issues


def _row(ocr_evidence=None, **extra):
    row = {
        "question_type": "screen_text",
        "canonical_evidence": [{"file": "00_kf10.jpg", "kf_n": 10, "pts_time": 12.0}],
        "ocr_evidence": ocr_evidence,
    }
    row.update(extra)
    return row


def test_screen_text_missing_ocr_is_hard_error():
    errors, warnings = _evidence_issues(_row(), {})

    assert "screen_text_missing_ocr_evidence" in errors
    assert not warnings


def test_ocr_evidence_must_reference_canonical_keyframe_and_time():
    evidence = [{
        "kf_n": 11, "pts_time": 13.0, "start": 13.0, "end": 13.0,
        "text": "TITLE", "source": "local_ocr_index",
    }]
    errors, _ = _evidence_issues(
        _row(evidence, ocr_evidence_mode="local_ocr", ocr_evidence_diagnostic=False), {},
    )

    assert "ocr_evidence_not_canonical" in errors


def test_manual_only_evidence_is_explicit_diagnostic_and_not_production_valid():
    evidence = [{
        "kf_n": 10, "pts_time": 12.0, "start": 12.0, "end": 12.0,
        "text": "TITLE", "source": "manual_frame_read",
    }]
    errors, _ = _evidence_issues(
        _row(evidence, ocr_evidence_mode="manual_only", ocr_evidence_diagnostic=True), {},
    )

    assert _ocr_evidence_mode(evidence) == ("manual_only", True)
    assert "screen_text_ocr_evidence_manual_only" in errors


def test_manual_only_screen_text_is_excluded_from_production_pool():
    assert _production_exclusion_reason({
        "question_type": "screen_text",
        "ocr_evidence_mode": "manual_only",
    }) == "screen_text_manual_only_ocr_evidence_diagnostic"


def test_targeted_local_qwen_ocr_is_machine_evidence_with_provenance():
    evidence = [{
        "kf_n": 10, "pts_time": 12.0, "start": 12.0, "end": 12.0,
        "text": "TUYỂN SINH 2025", "source": "local_qwen_repair",
        "provenance": "hcmai.vqa_ocr_evidence_repair_v1",
        "model_path": "/local/Qwen2.5-VL-3B-Instruct",
    }]
    errors, warnings = _evidence_issues(
        _row(evidence, ocr_evidence_mode="local_targeted_ocr", ocr_evidence_diagnostic=False), {},
    )

    assert errors == []
    assert warnings == []


def test_materializer_sidecar_and_parquet_preserve_ocr_fields(tmp_path):
    sidecar = tmp_path / "ocr.jsonl"
    sidecar.write_text(json.dumps({
        "annotation_id": "a1",
        "ocr_evidence": [{
            "kf_n": 10, "pts_time": 12.0, "start": 12.0, "end": 12.0,
            "text": "TITLE", "source": "local_ocr_index",
        }],
    }) + "\n", encoding="utf-8")

    loaded = _load_ocr_sidecar(sidecar)
    rows = [{
        "annotation_id": "a1",
        "ocr_evidence": loaded["a1"],
        "ocr_evidence_mode": "local_ocr",
        "ocr_evidence_diagnostic": False,
    }]
    exported = _parquet_rows_v3(rows)[0]

    assert json.loads(exported["ocr_evidence"])[0]["kf_n"] == 10
    assert exported["ocr_evidence_mode"] == "local_ocr"
    assert exported["ocr_evidence_diagnostic"] is False
