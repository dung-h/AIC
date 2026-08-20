"""Fail-closed paired TRAKE benchmark contract.

Visual, ASR, and optional multimodal reports are compared only when they
declare the same immutable holdout.  Diagnostic or proxy scores are retained
for audit but are never eligible for production-path selection.
"""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Mapping, NoReturn, Sequence


class PairedBenchmarkError(ValueError):
    """Raised when TRAKE reports cannot be proven to share one holdout."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_paired_benchmark",
        report: str | None = None,
        field: str | None = None,
        missing_fields: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.report = report
        self.field = field
        self.missing_fields = list(missing_fields)
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        """Return stable, CLI-safe diagnostics without exposing report payloads."""
        result: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
        }
        if self.report is not None:
            result["report"] = self.report
        if self.field is not None:
            result["field"] = self.field
        if self.missing_fields:
            result["missing_fields"] = list(self.missing_fields)
        if self.details:
            result["details"] = self.details
        return result


def _raise(
    message: str,
    *,
    code: str,
    report: str | None = None,
    field: str | None = None,
    missing_fields: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> NoReturn:
    raise PairedBenchmarkError(
        message,
        code=code,
        report=report,
        field=field,
        missing_fields=missing_fields,
        details=details,
    )


METRIC_NAMES = (
    "video_r1", "video_r5", "video_r20", "video_r50", "video_r100",
    "event_hit_2s", "event_hit_5s", "event_hit_10s", "full_sequence",
    "temporal_order_validity", "final_score",
)
_RANK_METRICS = ("video_r1", "video_r5", "video_r20", "video_r50", "video_r100")
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "final_score": ("official_final_score", "final_score"),
    "video_r1": ("video_r1", "frozen_lambda0_video_r1", "video_r@1", "r@1"),
    "video_r5": ("video_r5", "frozen_lambda0_video_r5", "video_r@5", "r@5"),
    "video_r20": ("video_r20", "video_r@20", "r@20"),
    "video_r50": ("video_r50", "video_r@50", "r@50"),
    "video_r100": ("video_r100", "video_r@100", "r@100"),
    "event_hit_2s": ("event_hit_2s", "event_2s", "oracle_video_event_2s", "end_to_end_event_2s"),
    "event_hit_5s": ("event_hit_5s", "event_5s", "oracle_video_event_5s", "end_to_end_event_5s"),
    "event_hit_10s": ("event_hit_10s", "event_10s", "oracle_video_event_10s", "end_to_end_event_10s"),
    "full_sequence": (
        "full_sequence", "full_sequence_2s", "full_sequence_5s",
        "oracle_video_full_seq_2s", "oracle_video_full_seq_5s", "full_seq_2s", "full_seq_5s",
    ),
    "temporal_order_validity": (
        "temporal_order_validity", "temporal_order_valid", "order_validity",
        "temporal_order_accuracy", "order_valid",
    ),
}
_FAILURE_FIELDS = ("video_miss", "candidate_miss", "alignment_miss")
_RANK_BUCKETS = ("r1_5", "r6_20", "r21_100", "gt_not_in_r100", "rank_unavailable")
_FAILURE_TYPE_TO_FIELD = {
    "video_miss": "video_miss", "video_retrieval_loss": "video_miss",
    "video_retrieval_miss": "video_miss", "candidate_miss": "candidate_miss",
    "candidate_missing": "candidate_miss", "candidate_loss": "candidate_miss",
    "alignment_miss": "alignment_miss", "alignment_loss": "alignment_miss",
    "temporal_alignment_loss": "alignment_miss",
}
_PROXY_WORDS = ("proxy", "synthetic", "diagnostic", "surrogate", "placeholder", "provisional", "legacy")


def _rows(report: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    value = report.get("production_per_query", report.get("per_query"))
    if not isinstance(value, list) or not value:
        _raise(
            f"{label} report must expose non-empty per-query rows",
            code="missing_per_query_rows",
            report=label,
            field="production_per_query|per_query",
            missing_fields=["production_per_query or per_query"],
        )
    if any(not isinstance(row, Mapping) for row in value):
        _raise(
            f"{label} per-query rows must be mappings",
            code="invalid_per_query_rows",
            report=label,
            field="production_per_query|per_query",
        )
    return value


def _declared_ids(report: Mapping[str, Any], label: str) -> set[str]:
    value = report.get("holdout_query_ids")
    if not isinstance(value, list) or not value:
        _raise(
            f"{label} report must declare explicit holdout_query_ids; do not infer holdout from n_queries or report order",
            code="missing_holdout_query_ids",
            report=label,
            field="holdout_query_ids",
            missing_fields=["holdout_query_ids"],
        )
    ids = {str(item).strip() for item in value if str(item).strip()}
    if len(ids) != len(value):
        _raise(
            f"{label} holdout_query_ids contain empty or duplicate IDs",
            code="invalid_holdout_query_ids",
            report=label,
            field="holdout_query_ids",
        )
    return ids


def _metadata(
    report: Mapping[str, Any],
    label: str,
    trusted: tuple[str, str] | None = None,
) -> tuple[str, str]:
    benchmark_id = str(report.get("benchmark_id", "")).strip()
    holdout_id = str(report.get("holdout_id", "")).strip()
    if trusted is not None:
        trusted_benchmark_id, trusted_holdout_id = trusted
        if benchmark_id and benchmark_id != trusted_benchmark_id:
            _raise(
                f"{label} report benchmark_id does not match trusted paired metadata",
                code="benchmark_id_mismatch",
                report=label,
                field="benchmark_id",
                details={"expected": trusted_benchmark_id, "actual": benchmark_id},
            )
        if holdout_id and holdout_id != trusted_holdout_id:
            _raise(
                f"{label} report holdout_id does not match trusted paired metadata",
                code="holdout_id_mismatch",
                report=label,
                field="holdout_id",
                details={"expected": trusted_holdout_id, "actual": holdout_id},
            )
        return trusted
    if not benchmark_id or not holdout_id:
        missing = [field for field, value in (("benchmark_id", benchmark_id), ("holdout_id", holdout_id)) if not value]
        _raise(
            f"{label} report must include benchmark_id and holdout_id; missing: {', '.join(missing)}",
            code="missing_paired_metadata",
            report=label,
            missing_fields=missing,
        )
    return benchmark_id, holdout_id


def _normal_key(value: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())


def _metric_containers(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only explicitly holdout-scoped metric containers."""
    containers: list[Mapping[str, Any]] = []
    direct = report.get("holdout_metrics")
    if isinstance(direct, Mapping):
        nested_direct = direct.get("holdout")
        containers.append(nested_direct if isinstance(nested_direct, Mapping) else direct)
    metrics = report.get("metrics")
    if isinstance(metrics, Mapping):
        nested = metrics.get("holdout")
        if isinstance(nested, Mapping):
            containers.append(nested)
        else:
            aliases = {_normal_key(alias) for values in _METRIC_ALIASES.values() for alias in values}
            if any(_normal_key(key) in aliases for key in metrics):
                containers.append(metrics)
    top_holdout = report.get("holdout")
    if isinstance(top_holdout, Mapping):
        nested = top_holdout.get("metrics", top_holdout)
        if isinstance(nested, Mapping):
            containers.append(nested)
    return containers


def _number(value: Any) -> float | None:
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric_with_source(report: Mapping[str, Any], name: str) -> tuple[float | None, str | None]:
    aliases = {_normal_key(alias) for alias in _METRIC_ALIASES.get(name, (name,))}
    for container in _metric_containers(report):
        for key, raw_value in container.items():
            if _normal_key(key) in aliases:
                value = _number(raw_value)
                if value is not None:
                    return value, str(key)
    return None, None


def _metric(report: Mapping[str, Any], name: str) -> float | None:
    """Extract one explicit holdout metric, retaining the legacy helper API."""
    return _metric_with_source(report, name)[0]


def _proxy_reasons(report: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in ("provenance", "score_provenance", "metric_provenance", "evaluation_status",
                "status", "protocol", "experiment", "source"):
        value = report.get(key)
        if value is None:
            continue
        text = str(value).casefold()
        for word in _PROXY_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                reason = f"{key} is marked {word}; score is not official-holdout eligible"
                if reason not in reasons:
                    reasons.append(reason)
    return reasons


def _metric_bundle(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    values = {name: _metric(report, name) for name in METRIC_NAMES}
    proxy_reasons = _proxy_reasons(report)
    reported_final, final_source_key = _metric_with_source(report, "final_score")
    score_source: str | None = None
    blocked_reasons = list(proxy_reasons)
    if proxy_reasons:
        values["final_score"] = None
    elif reported_final is not None:
        values["final_score"] = reported_final
        score_source = "explicit_report"
    elif all(values[name] is not None for name in _RANK_METRICS):
        values["final_score"] = sum(values[name] for name in _RANK_METRICS) / len(_RANK_METRICS)
        score_source = "derived_from_holdout_video_ranks"
    else:
        blocked_reasons.append(
            f"{label} report has no explicit final_score and lacks one or more holdout video R@k metrics needed for the official-style mean"
        )
    missing_metrics = [name for name, value in values.items() if value is None]
    if proxy_reasons:
        status = "blocked"
    elif missing_metrics:
        status = "partial"
    else:
        status = "ready"
    return {
        "label": label,
        "values": values,
        "reported_final_score": reported_final,
        "reported_final_score_key": final_source_key,
        "official_style_final_score": values["final_score"],
        "final_score_source": score_source,
        "score_eligible": values["final_score"] is not None and not proxy_reasons,
        "status": status,
        "available_metrics": [name for name, value in values.items() if value is not None],
        "missing_metrics": missing_metrics,
        "blocked_reasons": blocked_reasons,
    }


def _holdout_rows(report: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    declared = _declared_ids(report, label)
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _rows(report, label):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            _raise(
                f"{label} report has a row without query_id",
                code="missing_query_id",
                report=label,
                field="query_id",
                missing_fields=["query_id"],
            )
        split = row.get("split")
        if split is not None and str(split).strip().lower() != "holdout":
            continue
        if query_id in indexed:
            _raise(
                f"{label} report duplicates query_id {query_id}",
                code="duplicate_query_id",
                report=label,
                field="query_id",
                details={"query_id": query_id},
            )
        indexed[query_id] = row
    missing = declared - set(indexed)
    extra = set(indexed) - declared
    if missing:
        _raise(
            f"{label} report declares holdout IDs without rows: {sorted(missing)}",
            code="holdout_query_rows_missing",
            report=label,
            field="per_query",
            details={"missing_query_ids": sorted(missing)},
        )
    if extra:
        _raise(
            f"{label} report contains holdout rows not declared in holdout_query_ids: {sorted(extra)}",
            code="holdout_query_rows_extra",
            report=label,
            field="holdout_query_ids",
            details={"extra_query_ids": sorted(extra)},
        )
    return indexed


def _event_count(row: Mapping[str, Any], label: str, query_id: str) -> int:
    for key in ("n_events", "event_count"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    for key in ("events", "ground_truth_events"):
        events = row.get(key)
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes)) and events:
            return len(events)
    _raise(
        f"{label} row {query_id} must expose positive n_events/event_count",
        code="missing_event_count",
        report=label,
        field="n_events|event_count|events|ground_truth_events",
        missing_fields=["positive event count"],
        details={"query_id": query_id},
    )


def _ground_truth_video(row: Mapping[str, Any]) -> str:
    for key in ("ground_truth_video_id", "gt_video_id", "ground_truth_video", "video_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, Mapping):
        value = ground_truth.get("video_id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _manifest_ids(manifest: Mapping[str, Any] | None) -> set[str] | None:
    if manifest is None:
        return None
    value = manifest.get("holdout_query_ids")
    if value is None and isinstance(manifest.get("holdout"), Mapping):
        value = manifest["holdout"].get("query_ids")
    if not isinstance(value, list) or not value:
        _raise(
            "holdout manifest must explicitly expose holdout_query_ids; source query_ids without split membership are insufficient",
            code="missing_manifest_holdout_query_ids",
            report="manifest",
            field="holdout_query_ids",
            missing_fields=["holdout_query_ids"],
        )
    ids = {str(item).strip() for item in value if str(item).strip()}
    if len(ids) != len(value):
        _raise(
            "holdout manifest has empty or duplicate query IDs",
            code="invalid_manifest_holdout_query_ids",
            report="manifest",
            field="holdout_query_ids",
        )
    return ids


def _validate_manifest_metadata(manifest: Mapping[str, Any] | None, benchmark_id: str, holdout_id: str) -> None:
    if manifest is None:
        return
    for key, expected in (("benchmark_id", benchmark_id), ("holdout_id", holdout_id)):
        declared = manifest.get(key)
        if declared is not None and str(declared).strip() != expected:
            _raise(
                f"holdout manifest {key} does not match paired reports",
                code="manifest_metadata_mismatch",
                report="manifest",
                field=key,
                details={"expected": expected, "actual": str(declared).strip()},
            )


def _trusted_metadata(
    manifest: Mapping[str, Any] | None,
    benchmark_id: str | None,
    holdout_id: str | None,
) -> tuple[str, str] | None:
    """Resolve an explicit metadata anchor without inventing benchmark identity.

    Legacy baseline reports may not carry top-level IDs.  They can be paired only
    when the caller supplies both IDs, or when the holdout manifest supplies both
    IDs.  A partial pair is rejected so a missing field cannot be silently filled
    from an unrelated report.
    """
    explicit = (str(benchmark_id or "").strip(), str(holdout_id or "").strip())
    if bool(explicit[0]) != bool(explicit[1]):
        missing = [
            field for field, value in (("benchmark_id", explicit[0]), ("holdout_id", explicit[1])) if not value
        ]
        _raise(
            "benchmark_id and holdout_id must be supplied together; missing: " + ", ".join(missing),
            code="incomplete_trusted_metadata",
            report="caller",
            missing_fields=missing,
        )

    manifest_pair = ("", "")
    if manifest is not None:
        manifest_pair = (
            str(manifest.get("benchmark_id", "")).strip(),
            str(manifest.get("holdout_id", "")).strip(),
        )
        if bool(manifest_pair[0]) != bool(manifest_pair[1]):
            missing = [
                field for field, value in (("benchmark_id", manifest_pair[0]), ("holdout_id", manifest_pair[1])) if not value
            ]
            _raise(
                "holdout manifest benchmark_id and holdout_id must be supplied together; missing: "
                + ", ".join(missing),
                code="incomplete_manifest_metadata",
                report="manifest",
                missing_fields=missing,
            )

    explicit_pair = explicit if explicit[0] else None
    manifest_pair_value = manifest_pair if manifest_pair[0] else None
    if explicit_pair is not None and manifest_pair_value is not None and explicit_pair != manifest_pair_value:
        _raise(
            "explicit paired metadata does not match holdout manifest",
            code="trusted_metadata_mismatch",
            report="caller/manifest",
            details={
                "explicit": {"benchmark_id": explicit_pair[0], "holdout_id": explicit_pair[1]},
                "manifest": {"benchmark_id": manifest_pair_value[0], "holdout_id": manifest_pair_value[1]},
            },
        )
    return explicit_pair or manifest_pair_value


def _failure_value(row: Mapping[str, Any], field: str) -> tuple[bool, bool]:
    direct = row.get(field)
    if isinstance(direct, bool):
        return True, direct
    taxonomy = row.get("failure_taxonomy")
    if isinstance(taxonomy, Mapping) and isinstance(taxonomy.get(field), bool):
        return True, bool(taxonomy[field])
    failure_type = str(row.get("failure_type", "")).strip().casefold()
    mapped = _FAILURE_TYPE_TO_FIELD.get(failure_type)
    if mapped is not None:
        return True, mapped == field
    if failure_type in {"stable_hit", "hit", "success", "no_failure", "none"}:
        return True, False
    return False, False


def _row_rank(row: Mapping[str, Any]) -> int | None:
    """Read an explicit rank without treating a GT video ID as a prediction."""
    for key in ("video_rank", "rank", "gt_video_rank", "frozen_lambda0_video_rank"):
        value = row.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
            # A fractional rank is malformed evidence, not rank 1 after
            # truncation.  Leave it unavailable so the paired report cannot
            # silently promote a corrupted metric.
            if float(value).is_integer() and int(value) > 0:
                return int(value)
            return None
    ranked = row.get("ranked_video_ids", row.get("ranked_videos"))
    gt_video = _ground_truth_video(row)
    if isinstance(ranked, Sequence) and not isinstance(ranked, (str, bytes)):
        for index, item in enumerate(ranked, start=1):
            item_video = item.get("video_id", item.get("vid")) if isinstance(item, Mapping) else item
            if str(item_video or "").strip() == gt_video:
                return index
    return None


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "rank_unavailable"
    if rank <= 5:
        return "r1_5"
    if rank <= 20:
        return "r6_20"
    if rank <= 100:
        return "r21_100"
    return "gt_not_in_r100"


def _predicted_video(row: Mapping[str, Any]) -> str | None:
    """Resolve only explicit prediction fields; ``video_id`` is GT in legacy rows."""
    for key in ("predicted_video_id", "selected_video_id", "output_video_id", "result_video_id", "top_video_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    ranked = row.get("ranked_video_ids", row.get("ranked_videos"))
    if isinstance(ranked, Sequence) and not isinstance(ranked, (str, bytes)) and ranked:
        first = ranked[0]
        if isinstance(first, Mapping):
            first = first.get("video_id", first.get("vid"))
        if first is not None and str(first).strip():
            return str(first).strip()
    return None


def _ground_truth_video(row: Mapping[str, Any]) -> str:
    for key in ("ground_truth_video_id", "gt_video_id", "ground_truth_video", "video_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, Mapping):
        value = ground_truth.get("video_id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _failure_taxonomy(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int | None] = {}
    rates: dict[str, float | None] = {}
    observed_rows: dict[str, int] = {}
    query_ids: dict[str, list[str]] = {}
    total = len(rows)
    for field in _FAILURE_FIELDS:
        observed = count = 0
        positives: list[str] = []
        for query_id in sorted(rows):
            is_observed, value = _failure_value(rows[query_id], field)
            if not is_observed:
                continue
            observed += 1
            if value:
                count += 1
                positives.append(query_id)
        observed_rows[field] = observed
        counts[field] = count if observed else None
        rates[field] = count / total if observed == total and total else None
        query_ids[field] = positives
    if not any(observed_rows.values()):
        status = "unavailable"
        reason = "per-query rows do not expose video_miss/candidate_miss/alignment_miss"
    elif all(value == total for value in observed_rows.values()):
        status = "available"
        reason = None
    else:
        status = "partial"
        reason = "failure taxonomy is present for only a subset of required fields/rows"
    rank_counts = {bucket: 0 for bucket in _RANK_BUCKETS}
    rank_available = 0
    rank_r5_miss = rank_r20_miss = rank_r100_miss = 0
    stage_counts = {
        "success_or_unclassified": 0,
        "rank_6_20": 0,
        "rank_21_100": 0,
        "gt_not_in_r100": 0,
        "alignment_after_r5": 0,
        "wrong_video": 0,
        "rank_unavailable": 0,
    }
    wrong_video_query_ids: list[str] = []
    for query_id in sorted(rows):
        row = rows[query_id]
        rank = _row_rank(row)
        bucket = _rank_bucket(rank)
        rank_counts[bucket] += 1
        if rank is not None:
            rank_available += 1
            rank_r5_miss += rank > 5
            rank_r20_miss += rank > 20
            rank_r100_miss += rank > 100
        if bucket == "r6_20":
            stage_counts["rank_6_20"] += 1
        elif bucket == "r21_100":
            stage_counts["rank_21_100"] += 1
        elif bucket == "gt_not_in_r100":
            stage_counts["gt_not_in_r100"] += 1
        elif bucket == "rank_unavailable":
            stage_counts["rank_unavailable"] += 1
        else:
            predicted_video = _predicted_video(row)
            gt_video = _ground_truth_video(row)
            wrong_video = bool(predicted_video and gt_video and predicted_video != gt_video)
            if wrong_video:
                stage_counts["wrong_video"] += 1
                wrong_video_query_ids.append(query_id)
            else:
                observed_alignment, alignment = _failure_value(row, "alignment_miss")
                if observed_alignment and alignment:
                    stage_counts["alignment_after_r5"] += 1
                else:
                    stage_counts["success_or_unclassified"] += 1
    # ``alignment_miss`` is the legacy strict-2-second, full-sequence flag.
    # Keep it for compatibility, but expose tolerance-specific sequence loss
    # so a 13/13 miss cannot be misread as all events being semantically wrong.
    alignment_tolerance: dict[str, dict[str, int | float | None]] = {}
    for suffix in ("2s", "5s", "10s"):
        field = f"event_hit_{suffix}"
        observed = misses = full = 0
        for row in rows.values():
            n_events = row.get("n_events")
            hits = row.get(field)
            if not isinstance(n_events, Real) or isinstance(n_events, bool):
                continue
            if not isinstance(hits, Real) or isinstance(hits, bool):
                continue
            n_events_int = int(n_events)
            hits_int = int(hits)
            if n_events_int <= 0 or hits_int < 0 or hits_int > n_events_int:
                continue
            observed += 1
            if hits_int == n_events_int:
                full += 1
            else:
                misses += 1
        alignment_tolerance[suffix] = {
            "observed_queries": observed,
            "miss_queries": misses if observed else None,
            "full_sequence_queries": full if observed else None,
            "miss_rate": misses / observed if observed else None,
        }
    source_failure_type_counts: dict[str, int] = {}
    for row in rows.values():
        source_type = str(row.get("failure_type", "")).strip()
        if source_type:
            source_failure_type_counts[source_type] = source_failure_type_counts.get(source_type, 0) + 1
    total = len(rows)
    rank_buckets = {
        bucket: {"count": rank_counts[bucket], "rate": rank_counts[bucket] / total if total else None}
        for bucket in _RANK_BUCKETS
    }
    return {
        "status": status, "reason": reason, "total_queries": total,
        "counts": counts, "rates": rates, "observed_rows": observed_rows,
        "query_ids": query_ids, "fields": list(_FAILURE_FIELDS),
        # New stage taxonomy.  The legacy fields above remain unchanged so old
        # consumers keep working, while ranks 6-20 and 21-100 are no longer
        # mislabeled as a single generic alignment failure.
        "rank_available_queries": rank_available,
        "rank_buckets": rank_buckets,
        "rank_bucket_query_ids": {
            bucket: sorted(query_id for query_id in rows if _rank_bucket(_row_rank(rows[query_id])) == bucket)
            for bucket in _RANK_BUCKETS
        },
        "rank_threshold_misses": {
            "r5": {"count": rank_r5_miss, "rate": rank_r5_miss / total if total else None},
            "r20": {"count": rank_r20_miss, "rate": rank_r20_miss / total if total else None},
            "r100": {"count": rank_r100_miss, "rate": rank_r100_miss / total if total else None},
        },
        "stage_counts": stage_counts,
        "alignment_tolerance": alignment_tolerance,
        "source_failure_type_counts": dict(sorted(source_failure_type_counts.items())),
        "wrong_video_query_ids": sorted(wrong_video_query_ids),
        "wrong_video_event_scores_counted_as_oracle": False,
    }


def _event_metric_scope(report: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]]) -> str:
    """Classify event metrics without upgrading legacy oracle scores."""
    texts: list[str] = []
    for key in ("metric_semantics", "metric_provenance", "protocol", "evaluation_status"):
        value = report.get(key)
        if isinstance(value, Mapping):
            value = " ".join(str(item) for item in value.values())
        if value is not None:
            texts.append(str(value).casefold())
    for row in rows.values():
        value = row.get("metric_semantics")
        if value is not None:
            texts.append(str(value).casefold())
    text = " ".join(texts)
    if "oracle_gt_video_alignment" in text or "oracle" in text and "gt-video" in text:
        return "oracle_gt_video_alignment"
    if "end_to_end" in text or "end-to-end" in text or "wrong-video" in text:
        return "end_to_end"
    return "unspecified"


def _event_alignment_contract(report: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], bundle: Mapping[str, Any]) -> dict[str, Any]:
    scope = _event_metric_scope(report, rows)
    event_names = ("event_hit_2s", "event_hit_5s", "event_hit_10s", "full_sequence", "temporal_order_validity")
    return {
        "scope": scope,
        "official_end_to_end_eligible": scope == "end_to_end",
        "wrong_video_event_scores_counted_as_zero": scope == "end_to_end",
        "oracle_gt_video_alignment_is_diagnostic_only": scope == "oracle_gt_video_alignment",
        "available_metrics": [name for name in event_names if bundle["values"].get(name) is not None],
        "blocked_reason": None if scope == "end_to_end" else "event metrics are not explicitly declared end-to-end; no production/official claim",
    }


def _selection(bundles: Mapping[str, Mapping[str, Any]], min_gap: float) -> dict[str, Any]:
    scores = {label: bundle["values"].get("final_score") for label, bundle in bundles.items()}
    blocked = {
        label: list(bundle["blocked_reasons"])
        for label, bundle in bundles.items()
        if not bundle["score_eligible"]
    }
    if blocked:
        reasons = [f"{label}: {reason}" for label in bundles for reason in blocked.get(label, [])]
        if not reasons:
            reasons = ["every production path needs an eligible holdout final score"]
        result: dict[str, Any] = {
            "status": "blocked", "selected_path": None,
            "reason": "paired holdout exists but all paths need eligible final scores",
            "blocked_reasons": reasons, "path_scores": scores,
            "score_sources": {label: bundle["final_score_source"] for label, bundle in bundles.items()},
        }
        if scores.get("visual") is not None and scores.get("asr") is not None:
            result["final_score_gap_visual_minus_asr"] = scores["visual"] - scores["asr"]
        else:
            result["final_score_gap_visual_minus_asr"] = None
        return result

    best_score = max(float(value) for value in scores.values())
    visual_score = scores.get("visual")
    if visual_score is not None and best_score - float(visual_score) < min_gap:
        selected = "visual"
        reason = "visual is within the configured Final Score margin of the best path"
    else:
        selected = next(
            label for label in ("visual", "asr", "multimodal")
            if label in scores and float(scores[label]) == best_score
        )
        reason = f"{selected} has the highest eligible holdout Final Score"
    result = {
        "status": "selected", "selected_path": selected, "reason": reason,
        "path_scores": scores,
        "score_sources": {label: bundle["final_score_source"] for label, bundle in bundles.items()},
        "final_score_gap_to_best": {
            label: best_score - float(value) for label, value in scores.items()
        },
    }
    if scores.get("visual") is not None and scores.get("asr") is not None:
        result["final_score_gap_visual_minus_asr"] = scores["visual"] - scores["asr"]
    return result


def build_paired_report(
    visual_report: Mapping[str, Any],
    asr_report: Mapping[str, Any],
    multimodal_report: Mapping[str, Any] | None = None,
    *,
    manifest: Mapping[str, Any] | None = None,
    benchmark_id: str | None = None,
    holdout_id: str | None = None,
    min_gap: float = 0.05,
) -> dict[str, Any]:
    """Build a visual/ASR/(optional)multimodal paired report."""
    if not 0 <= min_gap <= 1:
        raise ValueError("min_gap must be between 0 and 1")
    reports: dict[str, Mapping[str, Any]] = {"visual": visual_report, "asr": asr_report}
    if multimodal_report is not None:
        reports["multimodal"] = multimodal_report

    for label, report in reports.items():
        if not isinstance(report, Mapping):
            _raise(
                f"{label} report must be a JSON object",
                code="invalid_report_object",
                report=label,
            )

    metadata: dict[str, tuple[str, str]] = {}
    rows_by_path: dict[str, dict[str, Mapping[str, Any]]] = {}
    bundles: dict[str, dict[str, Any]] = {}
    trusted_metadata = _trusted_metadata(manifest, benchmark_id, holdout_id)
    for label, report in reports.items():
        metadata[label] = _metadata(report, label, trusted=trusted_metadata)
        rows_by_path[label] = _holdout_rows(report, label)
        bundles[label] = _metric_bundle(report, label)
    benchmark_id, holdout_id = metadata["visual"]
    _validate_manifest_metadata(manifest, benchmark_id, holdout_id)
    for label, pair in metadata.items():
        if pair != (benchmark_id, holdout_id):
            _raise(
                f"visual and {label} reports do not share the same benchmark_id/holdout_id",
                code="paired_metadata_mismatch",
                report=label,
                details={
                    "visual": {"benchmark_id": benchmark_id, "holdout_id": holdout_id},
                    label: {"benchmark_id": pair[0], "holdout_id": pair[1]},
                },
            )

    path_ids = {label: set(rows) for label, rows in rows_by_path.items()}
    visual_ids = path_ids["visual"]
    for label, ids in path_ids.items():
        if ids != visual_ids:
            _raise(
                f"visual and {label} reports do not contain exactly the same holdout query IDs",
                code="holdout_query_id_mismatch",
                report=label,
                field="holdout_query_ids",
                details={
                    "visual_query_ids": sorted(visual_ids),
                    f"{label}_query_ids": sorted(ids),
                    "missing_from_report": sorted(visual_ids - ids),
                    "extra_in_report": sorted(ids - visual_ids),
                },
            )
    manifest_ids = _manifest_ids(manifest)
    if manifest_ids is not None and visual_ids != manifest_ids:
        _raise(
            "reports do not match the explicit holdout manifest query IDs",
            code="manifest_query_id_mismatch",
            report="manifest",
            field="holdout_query_ids",
            details={
                "report_query_ids": sorted(visual_ids),
                "manifest_query_ids": sorted(manifest_ids),
                "missing_from_manifest": sorted(visual_ids - manifest_ids),
                "extra_in_manifest": sorted(manifest_ids - visual_ids),
            },
        )

    parity: dict[str, dict[str, Any]] = {}
    for query_id in sorted(visual_ids):
        event_counts = {label: _event_count(rows_by_path[label][query_id], label, query_id) for label in reports}
        if len(set(event_counts.values())) != 1:
            detail = ", ".join(f"{label}={count}" for label, count in event_counts.items())
            _raise(
                f"event count mismatch for {query_id}: {detail}",
                code="event_count_mismatch",
                field="n_events|event_count|events|ground_truth_events",
                details={"query_id": query_id, "event_counts": event_counts},
            )
        gt_videos = {label: _ground_truth_video(rows_by_path[label][query_id]) for label in reports}
        if not all(gt_videos.values()) or len(set(gt_videos.values())) != 1:
            detail = ", ".join(f"{label}={video or '<missing>'}" for label, video in gt_videos.items())
            _raise(
                f"ground-truth video mismatch for paired query {query_id}: {detail}",
                code="ground_truth_video_mismatch",
                field="ground_truth_video_id|gt_video_id|ground_truth_video|video_id",
                details={"query_id": query_id, "ground_truth_video_ids": gt_videos},
            )
        parity[query_id] = {
            "n_events": event_counts["visual"],
            "ground_truth_video_id": gt_videos["visual"],
            "event_counts": event_counts,
            "ground_truth_video_ids": gt_videos,
            "event_count_parity": True,
            "ground_truth_video_parity": True,
        }

    selection = _selection(bundles, min_gap)
    metric_values = {label: bundle["values"] for label, bundle in bundles.items()}
    metric_status = {
        label: {key: value for key, value in bundle.items() if key not in {"label", "values"}}
        for label, bundle in bundles.items()
    }
    failure_taxonomy = {label: _failure_taxonomy(rows_by_path[label]) for label in reports}
    event_alignment = {
        label: _event_alignment_contract(report, rows_by_path[label], bundles[label])
        for label, report in reports.items()
    }
    all_explicit = all(bundle["final_score_source"] == "explicit_report" for bundle in bundles.values())
    if selection["status"] != "selected":
        official_score_status = "blocked"
    elif all_explicit:
        official_score_status = "available"
    else:
        official_score_status = "derived_official_style"

    result: dict[str, Any] = {
        "protocol_version": "trake-paired-v1", "status": "paired",
        "benchmark_id": benchmark_id, "holdout_id": holdout_id,
        "paths": list(reports), "holdout_queries": len(parity),
        "holdout_events": sum(item["n_events"] for item in parity.values()),
        "holdout_query_ids": sorted(parity), "query_parity": parity,
        "visual_metrics": metric_values["visual"], "asr_metrics": metric_values["asr"],
        "metrics": metric_values, "path_metrics": metric_values, "metric_status": metric_status,
        "video_retrieval": {
            label: {
                key: bundle["values"].get(key)
                for key in _RANK_METRICS
            }
            for label, bundle in bundles.items()
        },
        "event_alignment": event_alignment,
        "event_alignment_status": (
            "available" if all(item["official_end_to_end_eligible"] for item in event_alignment.values())
            else "blocked"
        ),
        "contract": {
            "video_retrieval_separate_from_event_alignment": True,
            "wrong_video_event_scores_are_zero": True,
            "oracle_gt_video_alignment_is_diagnostic_only": True,
            "rank_buckets": list(_RANK_BUCKETS),
        },
        "failure_taxonomy": failure_taxonomy, "selection": selection,
        "official_score_status": official_score_status,
    }
    if "multimodal" in metric_values:
        result["multimodal_metrics"] = metric_values["multimodal"]
    return result
