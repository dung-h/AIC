from __future__ import annotations

import pandas as pd
import pytest

from src.eval.eval_trake_dante_asr import (
    ASRIntervalError,
    f1_match,
    materialize_aligned_evidence,
)
from src.pipelines.trake_asr_index import (
    asr_interval,
    interval_distance_seconds,
    interval_overlap_seconds,
    is_monotonic_alignment,
    representative_timestamp,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": "V1",
                "start": 1.0,
                "end": 5.0,
                "pts_time": 2.0,
                "frame_idx": 20,
                "kf_n": 2,
                "text": "event one",
            },
            {
                "video_id": "V1",
                "start": 7.0,
                "end": 11.0,
                "pts_time": 8.0,
                "frame_idx": 80,
                "kf_n": 8,
                "text": "event two",
            },
        ]
    )


def test_representative_timestamp_preserves_legacy_and_interval_modes():
    record = {"start": 10.0, "end": 20.0, "pts_time": 8.0}

    assert asr_interval(record) == (10.0, 20.0)
    assert representative_timestamp(record) == 10.0
    assert representative_timestamp(record, strategy="midpoint") == 15.0
    assert representative_timestamp(record, strategy="end") == 20.0
    # pts_time is a canonical frame time and may be outside the ASR word span.
    assert representative_timestamp(record, strategy="pts_time") == 8.0


def test_interval_overlap_and_distance_do_not_collapse_chunks_to_points():
    left = {"start": 1.0, "end": 5.0}
    right = {"start": 3.0, "end": 7.0}
    separated = {"start": 9.0, "end": 11.0}

    assert interval_overlap_seconds(left, right) == pytest.approx(2.0)
    assert interval_distance_seconds(left, right) == 0.0
    assert interval_overlap_seconds(left, separated) == 0.0
    assert interval_distance_seconds(left, separated) == pytest.approx(4.0)


def test_f1_match_accepts_overlapping_asr_evidence_and_legacy_points():
    predicted = [
        {"start": 1.0, "end": 5.0},
        {"start": 10.0, "end": 12.0},
    ]
    ground_truth = [
        {"start": 3.0, "end": 3.5},
        14.0,
    ]

    assert f1_match(predicted, ground_truth, threshold=2.0) == pytest.approx(
        (1.0, 1.0, 1.0 - 1e-9 / 2), abs=1e-8
    )
    assert f1_match([3.0, 14.0], [3.0, 14.0]) == pytest.approx(
        (1.0, 1.0, 1.0 - 1e-9 / 2), abs=1e-8
    )


def test_missing_start_or_end_fails_closed():
    with pytest.raises(ASRIntervalError):
        asr_interval({"start": 1.0})
    with pytest.raises(ASRIntervalError):
        asr_interval({"end": 2.0})
    with pytest.raises(ASRIntervalError):
        f1_match([{"start": 1.0}], [1.0])


def test_monotonic_alignment_and_materialized_path_keep_interval_fields():
    table = _metadata()
    evidence = materialize_aligned_evidence(table, [0, 1], representative="midpoint")

    assert [row["start"] for row in evidence] == [1.0, 7.0]
    assert [row["end"] for row in evidence] == [5.0, 11.0]
    assert [row["representative_time"] for row in evidence] == [3.0, 9.0]
    assert is_monotonic_alignment(evidence, strategy="midpoint")

    assert not is_monotonic_alignment(
        [
            {"start": 7.0, "end": 8.0},
            {"start": 3.0, "end": 4.0},
        ]
    )
    with pytest.raises(ASRIntervalError):
        materialize_aligned_evidence(
            pd.DataFrame([{"start": 1.0, "pts_time": 1.0}]), [0]
        )


def test_materialized_path_rejects_non_monotonic_canonical_time():
    table = _metadata().copy()
    table.loc[1, "pts_time"] = 1.5
    with pytest.raises(ASRIntervalError, match="canonical pts_time"):
        materialize_aligned_evidence(table, [0, 1])
