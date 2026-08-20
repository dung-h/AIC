"""Evidence-first local persistence for VQA candidate triage and review."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.queryset.validate_vqa_annotations import QUESTION_TYPES, validate_row

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "data/annotations/vqa_candidate_review"
SOURCE = ROOT / "data/annotations/vqa_evidence_candidates.parquet"
MANIFEST = PACK / "manifest.json"
REVIEWS = PACK / "triage_reviews.json"
MATERIALIZED = PACK / "reviewed_candidates.parquet"
VALID_OUTPUT = PACK / "vqa_eval_v2.parquet"
VALIDATION_REPORT = PACK / "validation.json"
TRIAGE = {"untriaged", "keep", "reject", "needs_context"}
STATES = {"draft", "reviewed", "valid", "rejected"}


def _load_reviews() -> dict[str, dict]:
    return json.loads(REVIEWS.read_text()) if REVIEWS.exists() else {}


def _base_rows() -> dict[str, dict]:
    frame = pd.read_parquet(SOURCE).fillna("")
    evidence = json.loads(MANIFEST.read_text())
    by_id = {str(item["annotation_id"]): item for item in evidence["rows"]}
    rows = {}
    for item in frame.itertuples(index=False):
        row = item._asdict()
        row["evidence"] = by_id[str(item.annotation_id)]["evidence"]
        rows[str(item.annotation_id)] = row
    return rows


def list_rows() -> list[dict]:
    base, reviews = _base_rows(), _load_reviews()
    rows = []
    for annotation_id, row in base.items():
        merged = {**row, **reviews.get(annotation_id, {})}
        merged["annotation_id"] = annotation_id
        merged.setdefault("triage", "untriaged")
        merged.setdefault("status", "draft")
        merged["evidence_images"] = [item["file"] for item in merged["evidence"]]
        merged["evidence_url"] = f"/annotation/evidence/{annotation_id}/"
        rows.append(merged)
    return sorted(rows, key=lambda row: row["annotation_id"])


def _nonempty(payload: dict, keys: tuple[str, ...]) -> list[str]:
    return [key for key in keys if not str(payload.get(key, "")).strip()]


def save(annotation_id: str, payload: dict) -> dict:
    if annotation_id not in _base_rows():
        raise KeyError(annotation_id)
    triage = str(payload.get("triage", "untriaged"))
    status = str(payload.get("status", "draft"))
    if triage not in TRIAGE or status not in STATES:
        raise ValueError("invalid triage or status")
    record = {key: payload.get(key, "") for key in (
        "triage", "question_type", "query", "question", "answer", "required_modalities",
        "acceptable_kf_n", "answer_start_time", "answer_end_time", "review_notes",
        "annotator_id", "reviewer_id", "status")}
    record["updated_at"] = datetime.now(UTC).isoformat()
    previous = _load_reviews().get(annotation_id, {})
    if triage != "keep" and status != "draft":
        raise ValueError("only kept candidates may enter annotation review")
    if triage == "keep" and record["question_type"] not in QUESTION_TYPES:
        raise ValueError("kept candidate requires evidence-backed question_type")
    if record["question_type"] == "spoken_fact" and "asr" not in record["required_modalities"].split(","):
        raise ValueError("spoken_fact requires asr evidence")
    if record["question_type"] == "temporal_relation" and not str(record["review_notes"]).strip():
        raise ValueError("temporal_relation requires temporal evidence notes")
    if status == "valid":
        if previous.get("status") not in {"reviewed", "valid"}:
            raise ValueError("valid requires a prior independent reviewed state")
        missing = _nonempty(record, ("query", "question", "answer", "required_modalities", "acceptable_kf_n",
                                     "answer_start_time", "answer_end_time", "review_notes", "annotator_id", "reviewer_id"))
        if missing or record["annotator_id"] == record["reviewer_id"]:
            raise ValueError("valid requires complete fields and distinct reviewer")
        errors = validate_row({**_base_rows()[annotation_id], **record})
        if errors:
            raise ValueError("valid annotation failed shared validator: " + ", ".join(errors))
    reviews = _load_reviews(); reviews[annotation_id] = record
    REVIEWS.write_text(json.dumps(reviews, ensure_ascii=False, indent=2))
    return record


def export_reviewed() -> dict:
    frame = pd.DataFrame(list_rows())
    frame.to_parquet(MATERIALIZED, index=False)
    valid = frame[frame.apply(lambda row: not validate_row(row.to_dict()), axis=1)].copy()
    if len(valid): valid.to_parquet(VALID_OUTPUT, index=False)
    elif VALID_OUTPUT.exists(): VALID_OUTPUT.unlink()
    report = {"source": str(MATERIALIZED), "rows": len(frame), "valid_rows": len(valid),
              "invalid_rows": len(frame) - len(valid), "triage_counts": frame["triage"].value_counts().to_dict(),
              "status_counts": frame["status"].value_counts().to_dict(), "exported_valid_eval": bool(len(valid))}
    VALIDATION_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
