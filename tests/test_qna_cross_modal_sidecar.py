from __future__ import annotations

import json

from src.eval.benchmark_qna_cross_modal_contract import apply_evidence_sidecar


def test_evidence_sidecar_is_applied_without_mutating_rows(tmp_path):
    rows = [{"annotation_id": "q1", "question_type": "screen_text"}]
    sidecar = tmp_path / "evidence.jsonl"
    sidecar.write_text(json.dumps({
        "annotation_id": "q1",
        "ocr_evidence": [{"start": 1.0, "end": 1.0, "text": "2025"}],
    }) + "\n", encoding="utf-8")

    merged = apply_evidence_sidecar(rows, sidecar)

    assert "ocr_evidence" not in rows[0]
    assert merged[0]["ocr_evidence"][0]["text"] == "2025"
