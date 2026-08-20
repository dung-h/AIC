import json
import sys

sys.path.insert(0, "src/eval")

from aic2026_scoring import ranked_metrics, score_answer, validate_ranked_answers
from aic2026_output_adapter import normalize, normalize_submission
from aic2026_submission_audit import audit_submission
from export_kis_submission import build_submission
from export_qa_submission import build_submission as build_qa_submission
from aic2026_trake_audit import audit_trake
import pandas as pd
from eval_aic2026_ranked import evaluate


def test_official_ranked_metrics_use_max_at_each_cutoff():
    assert ranked_metrics([0.0, 0.4, 0.8]) == {
        "r@1": 0.0, "r@5": 0.8, "r@20": 0.8, "r@50": 0.8,
        "r@100": 0.8, "final_score": 0.64,
    }


def test_task_scores():
    assert score_answer("kis", {"video_id": "v", "frame_id": 12},
                        {"video_id": "v", "interval": [10, 20]}) == 1.0
    assert score_answer("trake", {"video_id": "v", "frame_ids": [12, 25]},
                        {"video_id": "v", "intervals": [[10, 20], [20, 30]]}) == 1.0


def test_adapter_normalizes_and_limits_answers():
    answers = normalize("qa", [{"video_id": "v", "frame_id": "2", "answer": "đỏ"}])
    assert answers == [{"video_id": "v", "frame_id": 2, "answer": "đỏ"}]
    try:
        validate_ranked_answers("kis", [{"video_id": "v", "frame_id": 1}] * 101)
    except ValueError as error:
        assert "100" in str(error)
    else:
        raise AssertionError("more than 100 answers must fail")


def test_ranked_evaluator_aggregates_query_metrics():
    submission = {"queries": {"q1": [
        {"video_id": "v", "frame_id": 99},
        {"video_id": "v", "frame_id": 12},
    ]}}
    truth = {"queries": {"q1": {
        "task": "kis", "video_id": "v", "interval": [10, 20]
    }}}
    result = evaluate(submission, truth)
    assert result["aggregate"]["r@1"] == 0.0
    assert result["aggregate"]["r@5"] == 1.0
    assert result["aggregate"]["final_score"] == 0.8


def test_query_keyed_adapter():
    result = normalize_submission("kis", {"queries": {"q": [{"video_id": "v", "frame_id": "1"}]}})
    assert result == {"task": "kis", "queries": {"q": [{"video_id": "v", "frame_id": 1}]}}


def test_query_keyed_adapter_rejects_empty_qa_answer_text():
    for answer in (None, "", "   "):
        try:
            normalize_submission("qa", {"queries": {"q": [
                {"video_id": "v", "frame_id": 1, "answer": answer}
            ]}})
        except ValueError as error:
            assert "Q&A answer text" in str(error)
        else:
            raise AssertionError("empty QA answers must fail shared normalization")


def test_review_readiness_poll_is_blocked_without_reviewed_rows(tmp_path):
    from poll_review_readiness import poll
    qa_path = tmp_path / "qa.parquet"
    qa = pd.DataFrame([{"status": "draft"}])
    qa.to_parquet(qa_path)
    worksheet_dir = tmp_path / "worksheets"
    worksheet_dir.mkdir()
    pd.DataFrame([{"status": "unreviewed"}]).to_csv(worksheet_dir / "v.csv", index=False)
    result = poll(qa_path, worksheet_dir)
    assert result["status"] == "blocked"
    assert result["evaluation_eligible"] is False
    assert result["qa"]["valid_state_rows"] == 0
    assert result["trake"]["reviewed_rows"] == 0


def test_review_readiness_poll_requires_both_qa_and_trake_gates(tmp_path):
    from poll_review_readiness import poll
    qa_path = tmp_path / "qa.parquet"
    pd.DataFrame([{"status": "draft"}]).to_parquet(qa_path)
    worksheet_dir = tmp_path / "worksheets"
    worksheet_dir.mkdir()
    pd.DataFrame([{"status": "reviewed"}]).to_csv(worksheet_dir / "v.csv", index=False)
    result = poll(qa_path, worksheet_dir)
    assert result["trake"]["ready"] is True
    assert result["qa"]["ready"] is False
    assert result["status"] == "blocked"


def test_ready_materializers_skip_when_poll_is_blocked(tmp_path):
    from run_ready_materializers import run
    poll_path = tmp_path / "poll.json"
    poll_path.write_text(json.dumps({
        "status": "blocked",
        "evaluation_eligible": False,
        "qa": {"ready": False},
        "trake": {"ready": False},
    }))
    result = run(poll_path)
    assert result["status"] == "blocked"
    assert result["materializers_attempted"] is False
    assert result["evaluation_eligible"] is False


def test_trake_wrong_video_gets_zero():
    assert score_answer("trake", {"video_id": "wrong", "frame_ids": [1]},
                        {"video_id": "right", "intervals": [[1, 2]]}) == 0.0


def test_submission_audit_checks_frame_mapping_and_trake_order():
    index = pd.DataFrame({"video_id": ["v", "v", "w"], "frame_idx": [10, 20, 30]})
    valid = {"queries": {"q": [{"video_id": "v", "frame_ids": [10, 20]}]}}
    assert audit_submission("trake", valid, index)["valid"]
    invalid = {"queries": {"q": [{"video_id": "v", "frame_ids": [20, 10]}]}}
    result = audit_submission("trake", invalid, index)
    assert not result["valid"]
    assert any("strictly increasing" in error for error in result["errors"])


def test_submission_audit_rejects_in_bounds_non_keyframe_for_kis_and_qa():
    index = pd.DataFrame({"video_id": ["v", "v"], "frame_idx": [10, 20]})
    kis = {"queries": {"q": [{"video_id": "v", "frame_id": 15}]}}
    qa = {"queries": {"q": [{"video_id": "v", "frame_id": 15, "answer": "red"}]}}
    for task, payload in [("kis", kis), ("qa", qa)]:
        result = audit_submission(task, payload, index)
        assert not result["valid"]
        assert any("not a mapped keyframe" in error for error in result["errors"])


def test_kis_export_uses_frame_idx_and_caps_at_100(tmp_path):
    candidates = pd.DataFrame([
        {"qid": 0, "candidate_rank": 1, "video_id": "v", "frame_idx": 20, "kf_n": 2},
        {"qid": 0, "candidate_rank": 2, "video_id": "v", "frame_idx": 10, "kf_n": 1},
    ])
    source = tmp_path / "candidates.parquet"
    candidates.to_parquet(source)
    payload = build_submission(source)
    assert payload["queries"]["0"] == [
        {"video_id": "v", "frame_id": 20},
        {"video_id": "v", "frame_id": 10},
    ]


def test_qa_export_rejects_empty_answers():
    try:
        build_qa_submission([{"query_id": "q", "answers": [
            {"video_id": "v", "frame_id": 10, "answer": ""}
        ]}])
    except ValueError as error:
        assert "answer cannot be empty" in str(error)
    else:
        raise AssertionError("empty Q&A answer must fail export")


def test_qa_export_rejects_null_and_evidence_only_answers():
    for item in [
        {"video_id": "v", "frame_id": 10, "answer": None},
        {"video_id": "v", "frame_id": 10, "answer": None, "status": "evidence_only"},
    ]:
        try:
            build_qa_submission([{"query_id": "q", "answers": [item]}])
        except ValueError as error:
            assert "Q&A answer" in str(error) or "evidence-only" in str(error)
        else:
            raise AssertionError("null/evidence-only Q&A output must fail export")


def test_qa_export_accepts_real_answer_and_preserves_frame_id():
    payload = build_qa_submission([{"query_id": "q", "answers": [
        {"video_id": "v", "frame_id": "10", "answer": "màu đỏ"}
    ]}])
    assert payload == {"task": "qa", "queries": {"q": [
        {"video_id": "v", "frame_id": 10, "answer": "màu đỏ"}
    ]}}


def test_trake_audit_checks_event_count_and_frame_bounds():
    index = pd.DataFrame({"video_id": ["v", "v"], "frame_idx": [10, 20]})
    payload = {"queries": {"q": [{"video_id": "v", "frame_ids": [10, 20]}]}}
    assert audit_trake(payload, index, {"q": 2})["valid"]
    bad = {"queries": {"q": [{"video_id": "v", "frame_ids": [20, 10]}]}}
    result = audit_trake(bad, index, {"q": 2})
    assert not result["valid"]
    assert any("strictly increasing" in e for e in result["errors"])


def test_trake_audit_rejects_wrong_event_count():
    index = pd.DataFrame({"video_id": ["v", "v"], "frame_idx": [10, 20]})
    payload = {"queries": {"q": [{"video_id": "v", "frame_ids": [10]}]}}
    result = audit_trake(payload, index, {"q": 2})
    assert not result["valid"]
    assert any("expected 2 events" in e for e in result["errors"])


def test_trake_audit_rejects_in_bounds_non_keyframe():
    index = pd.DataFrame({"video_id": ["v", "v"], "frame_idx": [10, 20]})
    payload = {"queries": {"q": [{"video_id": "v", "frame_ids": [15]}]}}
    result = audit_trake(payload, index, {"q": 1})
    assert not result["valid"]
    assert any("not a mapped keyframe" in e for e in result["errors"])


def test_trake_expected_events_load_from_queryset(tmp_path):
    from aic2026_trake_audit import load_expected_events
    path = tmp_path / "trake_queryset.parquet"
    pd.DataFrame([
        {"query_id": "q1", "step": 0},
        {"query_id": "q1", "step": 1},
        {"query_id": "q2", "step": 0},
    ]).to_parquet(path)
    assert load_expected_events(path) == {"q1": 2, "q2": 1}


def test_trake_candidate_alignment_analysis_imports():
    from analyze_trake_candidate_alignment import main
    assert callable(main)


def test_trake_lattice_alignment_returns_ordered_path():
    from eval_trake_lattice_alignment import align_lattice
    import numpy as np
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.9, 0.2]])
    score, path = align_lattice(scores, np.array([0.0, 1.0, 2.0]), 2)
    assert score is not None
    assert path == [0, 1]


def test_trake_holdout_manifest_sources_are_explicit():
    from audit_trake_holdout_manifest import SOURCES, EXCLUDED_SUBSET
    assert len(SOURCES) == 2
    assert EXCLUDED_SUBSET.name == "trake_action_pack_v1_user_confirmed.parquet"


def test_trake_independent_manifest_builder_is_importable():
    from build_trake_independent_manifest import SOURCES
    assert len(SOURCES) == 2


def test_trake_video_loss_analysis_is_importable():
    from analyze_trake_video_losses import classify
    assert "person_action" in classify("A person operates a medical instrument")


def test_trake_video_competitor_report_is_importable():
    import report_trake_video_competitors
    assert callable(report_trake_video_competitors.main)


def test_trake_loss_cohort_split_is_importable():
    from split_trake_video_loss_cohorts import main
    assert callable(main)


def test_trake_cohort_queue_builder_is_importable():
    from build_trake_cohort_annotation_queue import main
    assert callable(main)


def test_prospective_review_validator_rejects_queue_provenance(tmp_path):
    from validate_trake_prospective_review import validate_prospective
    row = {"queue_id":"q", "video_id":"v", "query_id":"q1", "step":0,
           "caption":"event", "kf_n":1, "frame_idx":10, "pts_time":1.0,
           "provenance":"prospective_annotation_queue", "split":"holdout",
           "annotator_id":"a", "reviewer_id":"b", "confidence":"high",
           "authoring_method":"human_timeline_review", "target_selection_method":"human_timestamp_review"}
    path = tmp_path / "review.parquet"
    import pandas as pd
    pd.DataFrame([row] * 3).assign(step=[0,1,2]).to_parquet(path)
    try:
        validate_prospective(path)
    except ValueError as error:
        assert "cannot be evaluated" in str(error)
    else:
        raise AssertionError("queue provenance must be rejected")


def test_prospective_materializer_fails_closed_on_unreviewed_queue():
    from materialize_trake_prospective_review import materialize
    from pathlib import Path
    try:
        materialize(Path("data/annotations/trake_cohort_annotation_queue.parquet"), Path("/tmp/should-not-exist.parquet"))
    except ValueError as error:
        assert "empty values" in str(error) or "prospective queue" in str(error)
    else:
        raise AssertionError("unreviewed queue must not materialize")


def test_provisional_packet_is_not_evaluation_eligible():
    import json
    packet = json.load(open("data/annotations/trake_provisional_review_pack/manifest.json"))
    assert len(packet) == 10
    assert all(item["evaluation_eligible"] is False for item in packet)


def test_provisional_navigation_is_not_ground_truth():
    import json
    data = json.load(open("data/annotations/trake_provisional_review_pack/navigation_candidates.json"))
    assert len(data) == 10
    assert all(not item["evaluation_eligible"] for item in data)
    assert all(not candidate["is_event_label"] for item in data for candidate in item["candidates"])


def test_review_worksheet_builder_is_importable():
    from build_trake_review_worksheets import main
    assert callable(main)


def test_review_dashboard_and_readiness_tools_import():
    from build_trake_review_dashboard import main as dashboard
    from audit_trake_review_readiness import main as readiness
    assert callable(dashboard)
    assert callable(readiness)


def test_worksheet_collector_rejects_blank_worksheets():
    from collect_trake_review_worksheets import collect
    from pathlib import Path
    try:
        collect(Path("data/annotations/trake_provisional_review_pack/worksheets"), Path("/tmp/should-not-be-created.parquet"))
    except ValueError as error:
        assert "status=reviewed" in str(error)
    else:
        raise AssertionError("blank worksheets must not be collected")


def test_review_session_builder_is_importable():
    from build_trake_review_session import main
    assert callable(main)


def test_review_pipeline_runner_is_importable():
    from run_trake_review_pipeline import main
    assert callable(main)


def test_provisional_packet_audit_is_importable():
    from audit_trake_provisional_packet import audit
    assert audit()["valid"]


def test_provisional_notes_audit_rejects_labels(tmp_path):
    import json
    from audit_trake_provisional_notes import audit
    path = tmp_path / "notes.json"
    path.write_text(json.dumps({"provisional": True, "evaluation_eligible": False,
                                "rows": [{"video_id": "v", "notes": "ok",
                                          "provenance": "provisional_browser_note",
                                          "evaluation_eligible": False}]}))
    assert audit(path)["valid"]


def test_review_checklist_is_pending_and_ineligible():
    import json
    data = json.load(open("data/annotations/trake_provisional_review_pack/review_checklist.json"))
    assert data["provisional"] is True
    assert data["evaluation_eligible"] is False
    assert len(data["videos"]) == 10
    assert all(v["review_status"] == "pending" for v in data["videos"])


def test_review_checklist_sync_is_importable():
    from sync_trake_review_checklist import main
    assert callable(main)


def test_review_preflight_is_importable():
    from audit_trake_review_preflight import main
    assert callable(main)
