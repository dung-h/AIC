import pytest

from src.reranking.query_routing_policy import (
    RescueGate,
    RoutingConfig,
    RoutingPolicyError,
    build_routing_plan,
    canonical_question_type,
    route_video_candidates,
)


def _ranked(*video_ids, evidence=True):
    rows = []
    for rank, video_id in enumerate(video_ids, 1):
        row = {"video_id": video_id, "kf_n": rank, "score": 1.0 / rank}
        if evidence:
            row["evidence"] = f"evidence for {video_id}"
        rows.append(row)
    return rows


def test_annotation_types_resolve_to_explicit_five_plan_names():
    assert canonical_question_type("color") == "visual"
    assert canonical_question_type("spoken_fact") == "spoken_fact"
    assert canonical_question_type("screen-text") == "screen_text"
    assert canonical_question_type("temporal relation") == "temporal_relation"
    assert canonical_question_type("new_label") == "unknown"


@pytest.mark.parametrize(
    ("question_type", "primary", "channels"),
    [
        ("visual", "visual", ("visual",)),
        ("spoken_fact", "asr", ("visual", "asr", "ocr", "poetry")),
        ("screen_text", "ocr", ("visual", "asr", "ocr")),
        ("temporal_relation", "visual", ("visual", "asr", "ocr")),
        ("unknown", "visual", ("visual", "asr", "ocr")),
    ],
)
def test_baseline_has_explicit_primary_and_channels(question_type, primary, channels):
    plan = build_routing_plan(question_type, RoutingConfig.baseline(enabled=True))
    assert plan.primary_channel == primary
    assert plan.channels == channels
    assert plan.weights
    assert plan.question_type == question_type


def test_weights_are_taken_from_config_and_visible_in_plan():
    types = ("visual", "spoken_fact", "screen_text", "temporal_relation", "unknown")
    config = RoutingConfig(
        enabled=True,
        weights_by_type={
            "visual": {"visual": 1.0},
            "spoken_fact": {"visual": 0.2, "asr": 1.7},
            "screen_text": {"visual": 0.3, "ocr": 1.4},
            "temporal_relation": {"visual": 1.0, "asr": 0.1, "ocr": 0.2},
            "unknown": {"visual": 1.0, "asr": 0.4, "ocr": 0.4},
        },
        rescue_by_type={
            "visual": RescueGate.disabled(),
            "spoken_fact": RescueGate(),
            "screen_text": RescueGate(),
            "temporal_relation": RescueGate(min_specialist_channels=2,
                                              allow_single_strong_rescue=False),
            "unknown": RescueGate(min_specialist_channels=2,
                                   allow_single_strong_rescue=False),
        },
    )
    plan = build_routing_plan("spoken_fact", config)
    assert dict(plan.weights) == {"visual": 0.2, "asr": 1.7}
    assert plan.required_channels == ("visual", "asr")


def test_routing_off_ignores_specialists_and_preserves_visual_order():
    plan = build_routing_plan("spoken_fact", RoutingConfig.baseline(enabled=False))
    rows = route_video_candidates(
        {"visual": _ranked("V1", "V2"), "asr": _ranked("ASR_TARGET")},
        plan,
    )
    assert [row["video_id"] for row in rows] == ["V1", "V2"]
    assert all(row["rrf_guard"] == "none" for row in rows)


def test_spoken_fact_uses_evidence_gated_asr_rescue():
    plan = build_routing_plan("spoken_fact", RoutingConfig.baseline(enabled=True))
    rows = route_video_candidates(
        {"visual": _ranked("V1", "V2", "V3"), "asr": _ranked("SPOKEN_TARGET", "V1")},
        plan,
    )
    assert "SPOKEN_TARGET" in [row["video_id"] for row in rows]
    assert any(row["rrf_guard"] == "strong_specialist_rescue" for row in rows)


def test_spoken_fact_without_primary_asr_is_fail_closed():
    plan = build_routing_plan("spoken_fact", RoutingConfig.baseline(enabled=True))
    with pytest.raises(RoutingPolicyError, match="missing required"):
        route_video_candidates({"visual": _ranked("V1")}, plan)


def test_temporal_relation_requires_two_modalities_for_specialist_rescue():
    plan = build_routing_plan("temporal_relation", RoutingConfig.baseline(enabled=True))
    rows = route_video_candidates(
        {
            "visual": _ranked("V1", "V2"),
            "asr": _ranked("ASR_ONLY"),
            "ocr": _ranked("V2"),
        },
        plan,
    )
    assert "ASR_ONLY" not in [row["video_id"] for row in rows]


def test_unknown_plan_is_multimodal_but_rescue_is_conservative():
    plan = build_routing_plan("unknown", RoutingConfig.baseline(enabled=True))
    assert plan.channels == ("visual", "asr", "ocr")
    assert plan.rescue_gate.min_specialist_channels == 2


def test_multimodal_config_cannot_disable_specialist_rescue_gate():
    config = RoutingConfig(
        enabled=True,
        weights_by_type={
            "visual": {"visual": 1.0},
            "spoken_fact": {"visual": 1.0, "asr": 1.0},
        },
        rescue_by_type={
            "visual": RescueGate.disabled(),
            "spoken_fact": RescueGate.disabled(),
        },
    )
    with pytest.raises(RoutingPolicyError, match="enabled evidence/rank"):
        build_routing_plan("spoken_fact", config)


def test_config_rejects_negative_weights_and_untyped_rescue_values():
    with pytest.raises(RoutingPolicyError, match="non-negative"):
        RoutingConfig(
            enabled=True,
            weights_by_type={"visual": {"visual": -1.0}},
            rescue_by_type={"visual": RescueGate.disabled()},
        )
    with pytest.raises(RoutingPolicyError, match="RescueGate"):
        RoutingConfig(
            enabled=True,
            weights_by_type={"visual": {"visual": 1.0}},
            rescue_by_type={"visual": {}},
        )
