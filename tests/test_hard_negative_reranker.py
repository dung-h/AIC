from src.reranking.hard_negative_reranker import HardNegativeReranker, HardNegativeRerankerConfig
from src.service.contracts import RetrievalResult


def test_reranker_preserves_base_order_without_feature_weights():
    candidates = [
        RetrievalResult("a", 1, score=0.9),
        RetrievalResult("b", 2, score=0.8),
    ]

    out = HardNegativeReranker().rerank("query", candidates)

    assert [item.result.video_id for item in out] == ["a", "b"]
    assert [item.original_rank for item in out] == [1, 2]


def test_reranker_can_promote_candidate_by_numeric_metadata_feature():
    candidates = [
        RetrievalResult("visual_top", 1, score=1.0, metadata={"local_text_match": 0.0}),
        RetrievalResult("hard_positive", 2, score=0.8, metadata={"local_text_match": 3.0}),
    ]
    config = HardNegativeRerankerConfig(base_score_weight=0.1, feature_weights={"local_text_match": 1.0})

    out = HardNegativeReranker(config).rerank("query", candidates)

    assert out[0].result.video_id == "hard_positive"
    assert out[0].feature_scores["local_text_match"] == 1.0


def test_reranker_accepts_dict_candidates_and_topk():
    candidates = [
        {"video_id": "a", "frame_idx": 1, "score": 0.1, "metadata": {"flag": False}},
        {"video_id": "b", "frame_idx": 2, "score": 0.1, "metadata": {"flag": True}},
    ]
    config = HardNegativeRerankerConfig(base_score_weight=0.0, feature_weights={"flag": 1.0})

    out = HardNegativeReranker(config).rerank("query", candidates, topk=1)

    assert len(out) == 1
    assert out[0].result.video_id == "b"
