import sys

sys.path.insert(0, "src/eval")

from aic2026_output_adapter import normalize_submission
from aic2026_scoring import validate_ranked_answers
from promotion_gate import qa_promotion_gate
from trake_baseline_selection import select_trake_path


def test_qa_contract_rejects_empty_and_evidence_only_answers():
    for value in (None, "", "  ", "evidence-only"):
        try:
            normalize_submission("vqa", {"queries": {"q": [
                {"video_id": "K01_V001", "frame_id": 12, "answer": value}
            ]}})
        except ValueError:
            pass
        else:
            raise AssertionError("Q&A must reject empty/evidence-only answers")


def test_trake_contract_rejects_empty_or_non_monotonic_sequence():
    for frames in ([], [3, 3], [4, 2]):
        try:
            validate_ranked_answers("trake", [{"video_id": "K01_V001", "frame_ids": frames}])
        except ValueError:
            pass
        else:
            raise AssertionError("TRAKE must reject invalid ordered frame sequences")


def test_both_contracts_cap_ranked_answers_at_100():
    qa = [{"video_id": "v", "frame_id": 1, "answer": "yes"}] * 101
    trake = [{"video_id": "v", "frame_ids": [1, 2]}] * 101
    for task, answers in (("qa", qa), ("trake", trake)):
        try:
            validate_ranked_answers(task, answers)
        except ValueError as error:
            assert "100" in str(error)
        else:
            raise AssertionError("ranked answer count must be capped")


def test_adapter_rejects_duplicate_answers_but_preserves_rank_order():
    payload = {"queries": {"q": [
        {"video_id": "v", "frame_id": 9, "answer": "later"},
        {"video_id": "v", "frame_id": 3, "answer": "earlier"},
    ]}}
    normalized = normalize_submission("qa", payload)
    assert [item["frame_id"] for item in normalized["queries"]["q"]] == [9, 3]
    try:
        normalize_submission("qa", {"queries": {"q": [
            {"video_id": "v", "frame_id": 9, "answer": "one"},
            {"video_id": "v", "frame_id": 9, "answer": "two"},
        ]}})
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate ranked answers must fail")


def test_adapter_rejects_empty_query_ranked_list():
    try:
        normalize_submission("trake", {"queries": {"q": []}})
    except ValueError as error:
        assert "no ranked answers" in str(error)
    else:
        raise AssertionError("empty query result must fail closed")


def test_adapter_can_enforce_canonical_frames_at_final_boundary():
    payload = {"queries": {"q": [
        {"video_id": "v", "frame_id": 10, "answer": "yes"},
    ]}}
    assert normalize_submission("qa", payload, canonical_frames={("v", 10)}) == {
        "task": "qa", "queries": {"q": [
            {"video_id": "v", "frame_id": 10, "answer": "yes"}
        ]}
    }
    try:
        normalize_submission("qa", payload, canonical_frames={("v", 11)})
    except ValueError as error:
        assert "non-canonical" in str(error)
    else:
        raise AssertionError("final adapter must reject non-canonical frames")


def test_trake_selection_is_fail_closed_without_official_final_score():
    result = select_trake_path(
        {"metrics": {"holdout": {"video_r1": 0.9}}},
        {"metrics": {"holdout": {"video_r1": 0.8}}},
    )
    assert result["status"] == "blocked"
    assert result["selected_path"] is None


def test_trake_selection_prefers_visual_when_scores_are_close():
    result = select_trake_path(
        {"holdout_metrics": {"final_score": 0.70}},
        {"holdout_metrics": {"final_score": 0.68}},
    )
    assert result["status"] == "selected"
    assert result["selected_path"] == "visual"


def test_trake_selection_can_promote_asr_when_gap_is_material():
    result = select_trake_path(
        {"holdout_metrics": {"final_score": 0.50}},
        {"holdout_metrics": {"final_score": 0.57}},
    )
    assert result["selected_path"] == "asr"


def test_qa_promotion_gate_requires_five_point_gain_and_non_regression():
    baseline = {"answer_accuracy_retrieved": 0.70, "frame_recall": 0.72}
    candidate = {"answer_accuracy_retrieved": 0.76, "frame_recall": 0.71}
    assert qa_promotion_gate(baseline, candidate)["promote"] is True
    regressed = {"answer_accuracy_retrieved": 0.68, "frame_recall": 0.80}
    assert qa_promotion_gate(baseline, regressed)["promote"] is False
