from fastapi.testclient import TestClient

from src.service.app import app
from src.service import annotation_review


def test_annotation_rows_are_read_only_from_source(monkeypatch, tmp_path):
    monkeypatch.setattr(annotation_review, "REVIEWS", tmp_path / "reviews.json")
    client = TestClient(app)
    rows = client.get("/annotation/rows")
    assert rows.status_code == 200
    assert len(rows.json()) == 240
    assert {row["triage"] for row in rows.json()} == {"untriaged"}


def test_candidate_requires_keep_and_type_before_review(monkeypatch, tmp_path):
    monkeypatch.setattr(annotation_review, "REVIEWS", tmp_path / "reviews.json")
    monkeypatch.setattr(annotation_review, "MATERIALIZED", tmp_path / "reviewed.parquet")
    monkeypatch.setattr(annotation_review, "VALID_OUTPUT", tmp_path / "valid.parquet")
    monkeypatch.setattr(annotation_review, "VALIDATION_REPORT", tmp_path / "report.json")
    payload = {"triage": "untriaged", "question_type": "", "query": "A visible scene", "question": "What color is the item?", "answer": "red",
               "required_modalities": "visual", "acceptable_kf_n": "94", "answer_start_time": "1",
               "answer_end_time": "2", "review_notes": "checked", "annotator_id": "a",
               "reviewer_id": "a", "status": "valid"}
    response = TestClient(app).put("/annotation/rows/vqa_candidate_0000", json=payload)
    assert response.status_code == 422
    payload.update({"triage": "keep", "question_type": "color", "status": "reviewed", "reviewer_id": "b"})
    response = TestClient(app).put("/annotation/rows/vqa_candidate_0000", json=payload)
    assert response.status_code == 200
    exported = TestClient(app).post("/annotation/export")
    assert exported.status_code == 200
    assert exported.json()["valid_rows"] == 0
    assert not (tmp_path / "valid.parquet").exists()
