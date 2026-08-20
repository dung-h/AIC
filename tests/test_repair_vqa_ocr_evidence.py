from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.repair_vqa_ocr_evidence_v1 import repair_vqa_ocr_evidence


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    annotations = tmp_path / "vqa_eval_v3.jsonl"
    sidecar = tmp_path / "ocr_sidecar.jsonl"
    canonical = tmp_path / "canonical.parquet"
    frame_root = tmp_path / "frames"
    (frame_root / "L21_V001").mkdir(parents=True)
    (frame_root / "L21_V002").mkdir(parents=True)
    for video_id, kfs in {"L21_V001": (1, 2, 3), "L21_V002": (4,)}.items():
        for kf_n in kfs:
            (frame_root / video_id / f"{kf_n:03d}.jpg").write_bytes(b"fake-image")

    rows = [
        {
            "annotation_id": "screen_manual",
            "question_type": "screen_text",
            "video_id": "L21_V001",
            "acceptable_kf_n": "1,2",
        },
        {
            "annotation_id": "screen_missing",
            "question_type": "screen_text",
            "video_id": "L21_V002",
            "acceptable_kf_n": "4",
        },
        {
            "annotation_id": "visual_ignored",
            "question_type": "action",
            "video_id": "L21_V001",
            "acceptable_kf_n": "3",
        },
    ]
    _write_jsonl(annotations, rows)
    _write_jsonl(sidecar, [
        {
            "annotation_id": "screen_manual",
            "ocr_evidence": [{
                "kf_n": 1,
                "pts_time": 1.0,
                "text": "human read",
                "source": "manual_frame_read",
            }],
        },
        {"annotation_id": "screen_missing", "ocr_evidence": []},
    ])
    pd.DataFrame([
        {"video_id": "L21_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 1.0},
        {"video_id": "L21_V001", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
        {"video_id": "L21_V001", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
        {"video_id": "L21_V002", "kf_n": 4, "frame_idx": 40, "pts_time": 4.0},
    ]).to_parquet(canonical, index=False)
    return annotations, sidecar, canonical, frame_root


class _FakeOCR:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def recognize(self, image_path: str, prompt: str) -> str:
        self.calls.append((image_path, prompt))
        return self.responses[Path(image_path).stem]


def test_repair_is_bounded_canonical_and_preserves_classification(tmp_path: Path):
    annotations, sidecar, canonical, frame_root = _fixture(tmp_path)
    fake = _FakeOCR({"001": "NONE", "002": "  Nhiệt độ 25°C  ", "004": "NONE"})
    output_sidecar = tmp_path / "diagnostic.jsonl"
    output_report = tmp_path / "diagnostic.json"

    report = repair_vqa_ocr_evidence(
        annotations,
        sidecar,
        canonical,
        frame_root,
        output_sidecar,
        output_report,
        backend=fake,
        model_path=tmp_path / "fake-model",
    )

    assert len(fake.calls) == 3
    assert all("Read only text" in prompt for _, prompt in fake.calls)
    rows = [json.loads(line) for line in output_sidecar.read_text(encoding="utf-8").splitlines()]
    assert [row["annotation_id"] for row in rows] == ["screen_manual", "screen_missing"]
    manual = rows[0]
    assert manual["classification"] == "manual_only"
    assert manual["source_classification"] == "manual_only"
    assert manual["repair_status"] == "repaired_diagnostic"
    assert manual["source_ocr_evidence"][0]["source"] == "manual_frame_read"
    assert manual["ocr_evidence"] == [{
        "video_id": "L21_V001",
        "kf_n": 2,
        "frame_idx": 20,
        "pts_time": 2.0,
        "start": 2.0,
        "end": 2.0,
        "text": "Nhiệt độ 25°C",
        "source": "local_qwen_repair",
        "provenance": "hcmai.vqa_ocr_evidence_repair_v1",
        "model_path": str((tmp_path / "fake-model").resolve()),
    }]
    missing = rows[1]
    assert missing["classification"] == "missing"
    assert missing["repair_status"] == "missing_preserved_no_text"
    assert missing["ocr_evidence"] == []
    assert report["offline"] is True
    assert report["production_eligible"] is False
    assert report["counts"]["screen_text_rows"] == 2
    assert report["counts"]["selected_frame_candidates"] == 3
    assert json.loads(output_report.read_text(encoding="utf-8"))["schema_version"] == (
        "hcmai.vqa_ocr_evidence_repair_v1"
    )


def test_repair_fails_closed_before_backend_on_missing_frame(tmp_path: Path):
    annotations, sidecar, canonical, frame_root = _fixture(tmp_path)
    (frame_root / "L21_V001" / "002.jpg").unlink()
    fake = _FakeOCR({"001": "text"})
    output_sidecar = tmp_path / "diagnostic.jsonl"
    output_report = tmp_path / "diagnostic.json"

    with pytest.raises(FileNotFoundError, match="L21_V001/002.jpg"):
        repair_vqa_ocr_evidence(
            annotations,
            sidecar,
            canonical,
            frame_root,
            output_sidecar,
            output_report,
            backend=fake,
        )
    assert fake.calls == []
    assert not output_sidecar.exists()
    assert not output_report.exists()


def test_repair_fails_closed_on_missing_model_without_api_fallback(tmp_path: Path):
    annotations, sidecar, canonical, frame_root = _fixture(tmp_path)
    output_sidecar = tmp_path / "diagnostic.jsonl"
    output_report = tmp_path / "diagnostic.json"

    with pytest.raises(RuntimeError, match="Qwen model directory does not exist"):
        repair_vqa_ocr_evidence(
            annotations,
            sidecar,
            canonical,
            frame_root,
            output_sidecar,
            output_report,
            model_path=tmp_path / "does-not-exist",
        )
    assert not output_sidecar.exists()
    assert not output_report.exists()


def test_repair_refuses_source_overwrite(tmp_path: Path):
    annotations, sidecar, canonical, frame_root = _fixture(tmp_path)
    fake = _FakeOCR({"001": "text", "002": "text", "004": "text"})

    with pytest.raises(ValueError, match="must not overwrite source"):
        repair_vqa_ocr_evidence(
            annotations,
            sidecar,
            canonical,
            frame_root,
            sidecar,
            tmp_path / "diagnostic.json",
            backend=fake,
        )
