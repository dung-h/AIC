"""Pure contract tests for Wave 2 query plans and evidence hits."""

from __future__ import annotations

import json
import math

import pytest

from src.retrieval.evidence import EvidenceHit, validate_evidence_hit
from src.retrieval.query_plan import QueryPlan


def _qna_plan(**overrides):
    values = {
        "task": "VQA",
        "request_id": "req-001",
        "query": "weather forecast in Nha Trang",
        "question": "What temperature is mentioned?",
        "required_modalities": ["visual", "speech"],
        "available_modalities": ["visual", "ASR", "ocr"],
        "channel_weights": {"visual": 1.0, "speech": 0.5},
        "top_k": 20,
        "budget": 12,
        "offline": True,
    }
    values.update(overrides)
    return QueryPlan(**values)


def _hit(**overrides):
    values = {
        "task": "qna",
        "video_id": "K01_V001",
        "frame_idx": 800,
        "kf_n": 8,
        "pts_time": 12.0,
        "modality": "speech",
        "channel": "bge-m3/asr",
        "rank": 2,
        "score": 0.75,
        "text": "Nha Trang hôm nay khoảng 25 độ.",
        "span": {"start": 11.0, "end": 13.0},
        "model_id": "BAAI/bge-m3",
        "index_id": "asr-b1-v1",
        "provenance": {"source": "local_asr", "chunk_id": "K01_V001:11"},
    }
    values.update(overrides)
    return EvidenceHit(**values)


def test_query_plan_normalizes_aliases_and_is_immutable():
    plan = _qna_plan()

    assert plan.task == "qna"
    assert plan.required_modalities == ("asr", "visual")
    assert plan.available_modalities == ("asr", "ocr", "visual")
    assert dict(plan.channel_weights) == {"asr": 0.5, "visual": 1.0}
    with pytest.raises(TypeError):
        plan.channel_weights["ocr"] = 1.0
    with pytest.raises((AttributeError, TypeError)):
        plan.top_k = 10


def test_query_plan_trake_requires_ordered_events_and_qna_requires_question():
    plan = QueryPlan(
        task="TRAKE", request_id="r2", query="cooking sequence",
        events=["oil touches pan", "meat enters pan"],
        required_modalities=["visual"], available_modalities=["visual"],
    )
    assert plan.task == "trake"
    assert plan.events == ("oil touches pan", "meat enters pan")

    with pytest.raises(ValueError, match="at least one event"):
        QueryPlan(task="trake", request_id="r3", query="sequence")
    with pytest.raises(ValueError, match="question"):
        QueryPlan(task="qna", request_id="r4", query="scene")


@pytest.mark.parametrize("field,value", [
    ("task", "kis"),
    ("request_id", ""),
    ("query", ""),
    ("required_modalities", ["visual", ""]),
    ("available_modalities", [" "]),
    ("channel_weights", {"visual": math.nan}),
    ("channel_weights", {"visual": -0.1}),
    ("top_k", 0),
    ("budget", 0),
])
def test_query_plan_rejects_invalid_routing_fields(field, value):
    with pytest.raises(ValueError):
        _qna_plan(**{field: value})


def test_query_plan_allows_required_modality_to_be_missing_for_preflight():
    plan = _qna_plan(required_modalities=["asr"], available_modalities=["visual"])
    assert plan.required_modalities == ("asr",)
    assert plan.available_modalities == ("visual",)


def test_query_plan_mapping_rejects_answer_leakage_and_unknown_fields():
    mapping = _qna_plan().to_dict()
    mapping["answer"] = "25 degrees"
    with pytest.raises(ValueError, match="leakage"):
        QueryPlan.from_mapping(mapping)

    mapping = _qna_plan().to_dict()
    mapping["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        QueryPlan.from_mapping(mapping)


def test_evidence_requires_canonical_frame_and_normalizes_modality():
    hit = _hit()
    assert hit.frame_idx == 800
    assert hit.modality == "asr"
    assert hit.provenance["source"] == "local_asr"
    assert hit.span["start"] == 11.0
    with pytest.raises(TypeError):
        hit.provenance["new"] = "value"


@pytest.mark.parametrize("field,value", [
    ("task", "kis"),
    ("video_id", "../other"),
    ("video_id", ""),
    ("frame_idx", -1),
    ("frame_idx", 1.5),
    ("modality", ""),
    ("channel", " "),
    ("rank", 0),
    ("score", math.inf),
    ("score", math.nan),
    ("model_id", ""),
    ("index_id", ""),
    ("kf_n", -1),
    ("pts_time", -0.1),
])
def test_evidence_rejects_invalid_identity_or_score(field, value):
    with pytest.raises(ValueError):
        _hit(**{field: value})


def test_evidence_rejects_answer_leakage_in_provenance_or_mapping():
    with pytest.raises(ValueError, match="leakage"):
        _hit(provenance={"answer": "25 degrees"})
    mapping = _hit().to_dict()
    mapping["ground_truth_answer"] = "25 degrees"
    with pytest.raises(ValueError, match="leakage"):
        EvidenceHit.from_mapping(mapping)


def test_evidence_rejects_wrong_task_and_function_adapter_checks_type():
    hit = _hit()
    trake = QueryPlan(task="trake", request_id="t1", query="sequence", events=["one"])
    with pytest.raises(ValueError, match="does not match"):
        hit.validate_for(trake)
    with pytest.raises(TypeError):
        validate_evidence_hit({"task": "qna"}, _qna_plan())
    assert validate_evidence_hit(hit, _qna_plan()) is hit


def test_serialization_is_json_compatible_deterministic_and_roundtrips():
    plan = _qna_plan()
    hit = _hit()
    assert plan.to_json() == plan.to_json()
    assert hit.to_json() == hit.to_json()
    assert list(json.loads(plan.to_json())) == sorted(json.loads(plan.to_json()))
    assert list(json.loads(hit.to_json())) == sorted(json.loads(hit.to_json()))
    assert QueryPlan.from_mapping(json.loads(plan.to_json())).to_json() == plan.to_json()
    assert EvidenceHit.from_mapping(json.loads(hit.to_json())).to_json() == hit.to_json()


def test_span_supports_text_or_numeric_interval_and_rejects_bad_interval():
    assert _hit(span="11s-13s").span == "11s-13s"
    assert _hit(span=[11, 13]).span == (11.0, 13.0)
    with pytest.raises(ValueError):
        _hit(span=[13, 11])
    with pytest.raises(ValueError):
        _hit(span={})
