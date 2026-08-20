from __future__ import annotations

import json

import pytest

from src.eval.evidence_registry import (
    DuplicateRecordError,
    EvidenceRecord,
    EvidenceRegistry,
    EvidenceRegistryError,
    PromotionRejectedError,
    STATUS_BLOCKED_PLACEHOLDER,
    STATUS_DIAGNOSTIC_PROXY,
    STATUS_PRODUCTION,
    STATUS_REJECTED,
    STATUS_STALE_LEGACY,
    STATUS_VALIDATED_FINDING,
    STATUSES,
    seed_known_facts,
)


TIMESTAMP = "2026-08-17T01:02:03Z"


def add_artifact(registry: EvidenceRegistry, **overrides):
    payload = dict(
        record_id="artifact-1",
        artifact_id="results/artifact-1.json",
        status=STATUS_VALIDATED_FINDING,
        provenance={"path": "results/artifact-1.json", "revision": "abc123"},
        metric_source="offline holdout evaluator",
        owner="qa",
        reason="Independent holdout check passed.",
        timestamp=TIMESTAMP,
        benchmark_id="qna-v3",
        holdout_id="qna-v3-holdout",
        metrics={"accuracy": 0.8},
    )
    payload.update(overrides)
    return registry.add(**payload)


def test_append_only_history_and_duplicate_ids_are_preserved():
    registry = EvidenceRegistry()
    first = add_artifact(registry)
    second = add_artifact(
        registry,
        record_id="artifact-2",
        artifact_id="results/artifact-2.json",
        status=STATUS_REJECTED,
        reason="Regression on the fixed holdout.",
    )

    assert [record.sequence for record in registry] == [1, 2]
    assert registry.get(first.record_id).status == STATUS_VALIDATED_FINDING
    assert registry.history(first.artifact_id) == (first,)
    assert second.status == STATUS_REJECTED
    with pytest.raises(DuplicateRecordError):
        add_artifact(registry)


@pytest.mark.parametrize(
    "field,value",
    [
        ("provenance", ""),
        ("metric_source", ""),
        ("owner", ""),
        ("reason", ""),
        ("timestamp", "2026-08-17"),
    ],
)
def test_required_audit_fields_are_validated(field, value):
    with pytest.raises(EvidenceRegistryError):
        add_artifact(EvidenceRegistry(), **{field: value})


def test_experiments_require_paired_benchmark_and_holdout_ids():
    with pytest.raises(EvidenceRegistryError, match="provided together"):
        add_artifact(
            EvidenceRegistry(),
            benchmark_id="only-benchmark",
            holdout_id=None,
        )
    with pytest.raises(EvidenceRegistryError, match="experiment records require"):
        add_artifact(
            EvidenceRegistry(),
            record_type="experiment",
            benchmark_id=None,
            holdout_id=None,
        )


def test_promotion_appends_and_does_not_mutate_source():
    registry = EvidenceRegistry()
    source = add_artifact(registry)
    promoted = registry.promote(
        source.record_id,
        new_record_id="artifact-1-production",
        target_status=STATUS_PRODUCTION,
        reason="Passed the locked holdout promotion gate.",
        timestamp="2026-08-17T02:00:00Z",
    )

    assert len(registry) == 2
    assert registry.get(source.record_id) == source
    assert promoted.status == STATUS_PRODUCTION
    assert promoted.supersedes == source.record_id
    assert promoted.sequence == 2


@pytest.mark.parametrize(
    "status,synthetic,tags",
    [
        (STATUS_DIAGNOSTIC_PROXY, False, ()),
        (STATUS_VALIDATED_FINDING, True, ()),
        (STATUS_VALIDATED_FINDING, False, ("proxy",)),
        (STATUS_BLOCKED_PLACEHOLDER, False, ()),
        (STATUS_STALE_LEGACY, False, ("legacy",)),
    ],
)
def test_promotion_rejects_non_promotable_evidence(status, synthetic, tags):
    registry = EvidenceRegistry()
    source = add_artifact(
        registry,
        record_id=f"blocked-{len(registry)}",
        status=status,
        synthetic=synthetic,
        tags=tags,
    )
    with pytest.raises(PromotionRejectedError):
        registry.promote(source.record_id, timestamp=TIMESTAMP)
    assert len(registry) == 1
    assert registry.get(source.record_id).status == status


def test_seed_known_facts_is_stable_idempotent_and_blocks_promotion():
    registry = seed_known_facts()
    exported_once = registry.to_json()
    assert len(registry) == 4
    assert registry.summary_by_status()[STATUS_DIAGNOSTIC_PROXY] == 3
    assert registry.summary_by_status()[STATUS_STALE_LEGACY] == 1

    seed_known_facts(registry)
    assert len(registry) == 4
    assert registry.to_json() == exported_once

    with pytest.raises(PromotionRejectedError):
        registry.promote("trake-synthetic-final-score-0933")

    synthetic = registry.get("trake-synthetic-final-score-0933")
    assert synthetic.metrics["final_score"] == 0.933
    assert registry.get("qna-local-vlm-078-068").metrics == {
        "metric_1": 0.78,
        "metric_2": 0.68,
    }


def test_json_export_is_deterministic_round_trippable_and_has_all_statuses(tmp_path):
    registry = EvidenceRegistry()
    add_artifact(registry, provenance={"z": 1, "a": [2, 1]})
    first = registry.to_json()
    second = registry.to_json()
    assert first == second

    payload = json.loads(first)
    assert payload["schema"] == "hcmai.evidence_registry.v1"
    assert list(payload["records"][0]) == sorted(payload["records"][0])
    assert set(payload["summary_by_status"]) == set(STATUSES)
    assert payload["summary_by_status"][STATUS_VALIDATED_FINDING] == 1

    path = tmp_path / "evidence_registry.json"
    registry.write_json(path)
    restored = EvidenceRegistry.read_json(path)
    assert restored.to_json() == first


def test_unknown_status_and_target_are_rejected():
    with pytest.raises(EvidenceRegistryError, match="status must be one of"):
        add_artifact(EvidenceRegistry(), status="maybe")
    registry = EvidenceRegistry()
    source = add_artifact(registry)
    with pytest.raises(PromotionRejectedError, match="target"):
        registry.promote(source.record_id, target_status=STATUS_STALE_LEGACY)
