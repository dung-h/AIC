from src.eval.validate_qna_bottlenecks_v1 import _check_selector


def _report(*, baseline_post: float, routed_post: float) -> dict:
    return {
        "selector_policy": "adaptive",
        "metrics": {
            "baseline": {"frame_recall_post_selector_budget_12": baseline_post},
            "routed": {
                "frame_recall_pre_selector": 0.4,
                "frame_recall_post_selector_budget_12": routed_post,
                "selector_loss_taxonomy": {
                    "preselector_hit_rows": 24,
                    "postselector_hit_rows": int(routed_post * 60),
                    "lost_after_selector_rows": 12,
                    "lost_due_to_gt_video_rank_gt_budget": 5,
                    "lost_within_video_budget_due_to_frame_allocation": 7,
                },
            },
        },
    }


def test_fixed_budget_selector_loss_is_nonblocking_on_baseline_parity():
    findings = []
    result = _check_selector(_report(baseline_post=0.18, routed_post=0.18), findings)

    assert result["route_non_regression"] is True
    finding = next(item for item in findings if item["id"] == "S5_selector_loss_present")
    assert finding["blocking"] is False
    assert finding["severity"] == "P1"


def test_selector_regression_remains_blocking():
    findings = []
    result = _check_selector(_report(baseline_post=0.18, routed_post=0.16), findings)

    assert result["route_non_regression"] is False
    finding = next(item for item in findings if item["id"] == "S5_selector_loss_present")
    assert finding["blocking"] is True
    assert finding["severity"] == "P0"
