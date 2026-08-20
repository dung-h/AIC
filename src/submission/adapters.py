"""Serialization adapters for the stable Q&A/TRAKE contracts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import QnAAnswerRecord, TrakeAnswerRecord
from .validators import validate_qna_answers, validate_trake_answers


_QA_EXTERNAL_FIELDS = ("video_id", "frame_id", "answer")
_TRAKE_EXTERNAL_FIELDS = ("video_id", "frame_ids")


def _task(task: str) -> str:
    value = str(task).strip().casefold()
    if value in {"qa", "q&a", "vqa"}:
        return "qa"
    if value == "trake":
        return "trake"
    raise ValueError(f"unsupported submission task: {task}")


def serialize_qna_submission(
    queries: Mapping[str, Sequence[QnAAnswerRecord | Mapping[str, Any]]],
    *,
    canonical_frames: Any,
) -> dict[str, Any]:
    """Serialize ranked Q&A answers to the existing query-keyed JSON shape."""
    normalized: dict[str, list[dict[str, Any]]] = {}
    for query_id, answers in queries.items():
        if not str(query_id).strip():
            raise ValueError("query_id must not be empty")
        normalized[str(query_id)] = [
            record.external()
            for record in validate_qna_answers(answers, canonical_frames=canonical_frames)
        ]
    if not normalized:
        raise ValueError("submission must contain at least one query")
    return {"task": "qa", "queries": normalized}


def serialize_trake_submission(
    queries: Mapping[str, Sequence[TrakeAnswerRecord | Mapping[str, Any]]],
    *,
    event_counts: Mapping[str, int],
    canonical_frames: Any,
) -> dict[str, Any]:
    """Serialize ranked TRAKE answers to the existing query-keyed JSON shape."""
    normalized: dict[str, list[dict[str, Any]]] = {}
    for query_id, answers in queries.items():
        query_key = str(query_id)
        if not query_key.strip():
            raise ValueError("query_id must not be empty")
        if query_key not in event_counts:
            raise ValueError(f"missing event_count for query {query_key}")
        normalized[query_key] = [
            record.external()
            for record in validate_trake_answers(
                answers,
                event_count=event_counts[query_key],
                canonical_frames=canonical_frames,
            )
        ]
    if not normalized:
        raise ValueError("submission must contain at least one query")
    return {"task": "trake", "queries": normalized}


def serialize_submission(
    task: str,
    queries: Mapping[str, Sequence[QnAAnswerRecord | TrakeAnswerRecord | Mapping[str, Any]]],
    *,
    canonical_frames: Any,
    event_counts: Mapping[str, int] | None = None,
    include_audit: bool = False,
    audit_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch to a task adapter while keeping provider details out of output."""
    canonical = _task(task)
    if canonical == "qa":
        payload = serialize_qna_submission(queries, canonical_frames=canonical_frames)
    else:
        if event_counts is None:
            raise ValueError("event_counts is required for TRAKE serialization")
        payload = serialize_trake_submission(
            queries, event_counts=event_counts, canonical_frames=canonical_frames
        )
    if include_audit:
        payload = dict(payload)
        payload["audit"] = _build_audit_metadata(
            payload, external_fields=(
                _QA_EXTERNAL_FIELDS if canonical == "qa" else _TRAKE_EXTERNAL_FIELDS
            ), extra=audit_metadata,
        )
    return payload


def _build_audit_metadata(
    payload: Mapping[str, Any],
    *,
    external_fields: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build diagnostics without changing the default external payload."""
    queries = payload.get("queries")
    if not isinstance(queries, Mapping):
        raise ValueError("serialized submission has no query mapping")
    counts = {
        str(query_id): len(answers) if isinstance(answers, Sequence) else 0
        for query_id, answers in queries.items()
    }
    audit: dict[str, Any] = {
        "schema": "hcmai.submission_audit.v1",
        "task": payload.get("task"),
        "query_count": len(counts),
        "answer_count": sum(counts.values()),
        "answers_per_query": counts,
        "max_answers_per_query": max(counts.values(), default=0),
        "ranked_order_preserved": True,
        "canonical_frames_validated": True,
        "external_fields": list(external_fields),
        "placeholder_rows": 0,
        "empty_rows": 0,
    }
    if extra:
        audit["diagnostics"] = dict(extra)
    return audit


def audit_submission(
    task: str,
    queries: Mapping[str, Sequence[QnAAnswerRecord | TrakeAnswerRecord | Mapping[str, Any]]],
    *,
    canonical_frames: Any,
    event_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a payload and return explicit audit metadata.

    The returned report is deliberately separate from the submission payload;
    callers can persist it as a sidecar without changing the transport schema.
    Validation is performed by the same serializer used for production output.
    """
    payload = serialize_submission(
        task,
        queries,
        canonical_frames=canonical_frames,
        event_counts=event_counts,
    )
    canonical = _task(task)
    return _build_audit_metadata(
        payload,
        external_fields=_QA_EXTERNAL_FIELDS if canonical == "qa" else _TRAKE_EXTERNAL_FIELDS,
        extra=metadata,
    )


def _public_qna_answers(result: Any) -> Sequence[Any]:
    """Extract answers from the public ranked-Q&A result shape.

    The service/UI result is currently ``{"answers": [...]}``, while the
    submission adapter accepts the query-keyed list.  Keep this conversion at
    the boundary so public callers use the same validator as Codabench.
    """
    if isinstance(result, Mapping):
        answers = result.get("answers")
    else:
        answers = result
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        raise TypeError("public Q&A result must contain an answers sequence")
    return answers


def _public_trake_answers(result: Any) -> list[Any]:
    """Convert the public TRAKE ``results/path`` shape to stable records."""
    if isinstance(result, Mapping):
        results = result.get("results")
    else:
        results = result
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("public TRAKE result must contain a results sequence")

    normalized: list[Any] = []
    for item in results:
        if isinstance(item, Mapping) and "frame_ids" in item:
            normalized.append(item)
            continue
        if not isinstance(item, Mapping) or "path" not in item:
            raise TypeError("public TRAKE result must contain frame_ids or path")
        path = item.get("path")
        if not isinstance(path, Sequence) or isinstance(path, (str, bytes)):
            raise TypeError("public TRAKE path must be a sequence")
        frame_ids: list[Any] = []
        for step in path:
            if not isinstance(step, Mapping):
                raise TypeError("public TRAKE path steps must be mappings")
            if "frame_idx" in step:
                frame_ids.append(step["frame_idx"])
            elif "frame_id" in step:
                frame_ids.append(step["frame_id"])
            else:
                raise ValueError("public TRAKE path step has no canonical frame index")
        normalized.append({
            "video_id": item.get("video_id"),
            "frame_ids": frame_ids,
            "score": item.get("score", 0.0),
            "provider": item.get("provider"),
            "model_id": item.get("model_id"),
            "metadata": item.get("metadata", {}),
        })
    return normalized


def serialize_public_result(
    task: str,
    result: Any,
    *,
    query_id: str,
    canonical_frames: Any,
    event_count: int | None = None,
    include_audit: bool = False,
    audit_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one public API result through the stable validator path.

    This preserves the existing external ``task/queries`` JSON shape and
    accepts the internal public result shapes without duplicating validation.
    """
    if not str(query_id).strip():
        raise ValueError("query_id must not be empty")
    canonical = _task(task)
    if canonical == "qa":
        return serialize_submission(
            "qa",
            {str(query_id): _public_qna_answers(result)},
            canonical_frames=canonical_frames,
            include_audit=include_audit,
            audit_metadata=audit_metadata,
        )
    if event_count is None:
        raise ValueError("event_count is required for public TRAKE serialization")
    queries = {str(query_id): _public_trake_answers(result)}
    if include_audit:
        return serialize_submission(
            "trake",
            queries,
            event_counts={str(query_id): event_count},
            canonical_frames=canonical_frames,
            include_audit=True,
            audit_metadata=audit_metadata,
        )
    return serialize_trake_submission(
        queries,
        event_counts={str(query_id): event_count},
        canonical_frames=canonical_frames,
    )
