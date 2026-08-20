from __future__ import annotations

import pytest

from src.vqa.reranker import (
    EvidenceContractError,
    ListwiseEvidenceReranker,
    MissingEvidenceError,
    ScorerContractError,
)


def _candidate(candidate_id: str, frame_idx: int, *, source: str = "visual") -> dict:
    return {
        "candidate_id": candidate_id,
        "video_id": f"V-{candidate_id}",
        "frames": [{"video_id": f"V-{candidate_id}", "frame_idx": frame_idx,
                    "kf_n": frame_idx // 10, "pts_time": float(frame_idx)}],
        "sources": [source],
        "provenance": [{"source": source, "rank": frame_idx}],
    }


class _RecordingScorer:
    def __init__(self, values_by_id: dict[str, float]):
        self.values_by_id = values_by_id
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, query, candidates):
        ids = tuple(item.candidate_id for item in candidates)
        self.calls.append((query, ids))
        return [self.values_by_id[item.candidate_id] for item in candidates]


def test_two_stage_reranker_shortlists_then_runs_listwise_stage_and_keeps_provenance():
    candidates = [_candidate("a", 10), _candidate("b", 20), _candidate("c", 30)]
    stage1 = _RecordingScorer({"a": 0.9, "b": 0.8, "c": 0.1})
    stage2 = _RecordingScorer({"a": 0.2, "b": 0.95, "c": 1.0})
    reranker = ListwiseEvidenceReranker(stage1, stage2, stage1_k=2)

    ranked = reranker.rerank("weather", candidates, top_k=2)

    assert stage1.calls == [("weather", ("a", "b", "c"))]
    assert stage2.calls == [("weather", ("a", "b"))]
    assert [item.candidate.candidate_id for item in ranked] == ["b", "a"]
    assert ranked[0].stage1_rank == 2
    assert ranked[0].stage2_rank == 1
    assert ranked[0].candidate.provenance == ({"source": "visual", "rank": 20},)
    assert ranked[0].to_dict()["canonical_provenance"][0]["frame_idx"] == 20


def test_reranker_is_deterministic_on_ties_and_accepts_mapping_scores():
    candidates = [_candidate("b", 20), _candidate("a", 10)]

    def scorer(query, items):
        return [{"relevance": 1.0} for _ in items]

    ranked = ListwiseEvidenceReranker(scorer, scorer, stage1_k=2).rerank("q", candidates)
    assert [item.candidate.candidate_id for item in ranked] == ["a", "b"]
    assert [item.final_rank for item in ranked] == [1, 2]


def test_reranker_fails_closed_before_scorers_for_missing_canonical_frame():
    called = False

    def scorer(query, items):
        nonlocal called
        called = True
        return [1.0 for _ in items]

    with pytest.raises(MissingEvidenceError):
        ListwiseEvidenceReranker(scorer, scorer).rerank(
            "q", [{"candidate_id": "x", "video_id": "V1", "frames": []}])
    assert called is False


def test_reranker_rejects_duplicate_ids_and_bad_scorer_shape():
    scorer = lambda query, items: [1.0 for _ in items]
    with pytest.raises(EvidenceContractError):
        ListwiseEvidenceReranker(scorer, scorer).rerank(
            "q", [_candidate("x", 1), _candidate("x", 2)])

    def bad_shape(query, items):
        return []

    with pytest.raises(ScorerContractError):
        ListwiseEvidenceReranker(bad_shape, scorer).rerank("q", [_candidate("x", 1)])


def test_reranker_preserves_asr_and_ocr_evidence_without_fabricating_it():
    candidate = _candidate("spoken", 42, source="visual")
    candidate.update({
        "asr_chunks": [{"start": 4.0, "end": 5.0, "chunk": "Nha Trang 25 độ"}],
        "ocr_text": [{"pts_time": 4.5, "ocr_text": "DỰ BÁO THỜI TIẾT"}],
    })
    scorer = lambda query, items: [1.0 for _ in items]
    ranked = ListwiseEvidenceReranker(scorer, scorer).rerank("temperature", [candidate])
    bundle = ranked[0].candidate
    assert [row.source for row in bundle.asr] == ["asr"]
    assert [row.source for row in bundle.ocr] == ["ocr"]
    assert ranked[0].to_dict()["provenance"] == [{"source": "visual", "rank": 42}]
