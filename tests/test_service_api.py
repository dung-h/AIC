from fastapi.testclient import TestClient

from src.service.app import app
from src.service.contracts import RetrievalResult
from src.service.runtime import get_runtime


def test_health_does_not_load_models():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_not_loaded_without_loading_models():
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["trake_model"] in {"not_loaded", "ready"}


def test_kis_contract_with_mock_runtime(monkeypatch):
    runtime = get_runtime()
    runtime._query_cache.clear()
    monkeypatch.setattr(runtime, "search_kis", lambda query, topk, mode: [("K01_V001", 12, 3, 0.9)])
    monkeypatch.setattr(runtime, "normalize_kis_result",
                        lambda raw: RetrievalResult("K01_V001", 12, 3, 0.0, 0.9))
    response = TestClient(app).post("/search/kis", json={"query": "test", "topk": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [{"rank": 1, "video_id": "K01_V001", "frame_idx": 12,
                                 "kf_n": 3, "pts_time": 0.0, "score": 0.9}]


def test_trake_contract_with_mock_runtime(monkeypatch):
    runtime = get_runtime()
    monkeypatch.setattr(runtime, "search_trake", lambda events, topk, alternatives: {
        "results": [{"video_id": "K01_V001", "score": 1.2, "path": [
            {"event_desc": events[0], "kf_n": 3, "frame_idx": 12, "pts_time": 4.5}
        ]}],
    })
    runtime.last_timings_ms = {"total": 1.0}
    response = TestClient(app).post("/search/trake", json={
        "events": ["first event"], "top_k_videos": 1
    })
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["path"][0]["frame_idx"] == 12
    assert body["results"][0]["path"][0]["pts_time"] == 4.5


def test_trake_rejects_blank_event():
    response = TestClient(app).post("/search/trake", json={"events": [" "]})
    assert response.status_code == 422
