"""Deterministic tests for the opt-in visual TRAKE lattice seam."""

import numpy as np
import pandas as pd

from src.pipelines.trake_visual import (
    VisualTrakeDante,
    build_event_candidate_lattice,
)
from src.utils.dante import normalize_event_scores, sequence_quality


def _metadata(size=7):
    return pd.DataFrame({
        "video_id": ["V1"] * size,
        "pts_time": np.arange(size, dtype=float),
        "kf_n": np.arange(1, size + 1),
        "frame_idx": np.arange(100, 100 + size),
    })


def test_lattice_keeps_top_k_seeds_and_temporal_neighbors_deterministically():
    metadata = _metadata()
    scores = np.asarray([
        [0.0, 0.1, 0.3, 0.95, 0.2, 0.2, 0.1],
        [0.1, 0.9, 0.8, 0.2, 0.1, 0.0, 0.0],
    ], dtype=np.float32)

    lattice = build_event_candidate_lattice(
        metadata,
        scores,
        top_k=1,
        temporal_neighbor_radius=1,
    )

    assert [row["position"] for row in lattice[0]] == [3, 2, 4]
    assert [row["position"] for row in lattice[1]] == [1, 2, 0]
    assert lattice[0][0]["source"] == "seed"
    assert all(row["source"] == "temporal_neighbor" for row in lattice[0][1:])
    assert all("frame_idx" in row and "pts_time" in row for row in lattice[0])


def test_sequence_quality_penalizes_weak_event_and_irregular_temporal_gaps():
    normalized = normalize_event_scores(np.asarray([
        [0.1, 0.5, 0.9, 0.2],
        [0.1, 0.5, 0.9, 0.2],
        [0.1, 0.5, 0.9, 0.2],
    ]))

    strong = sequence_quality(normalized, [2, 2, 2], [1.0, 2.0, 3.0])
    weak = sequence_quality(normalized, [2, 0, 2], [1.0, 2.0, 3.0])
    irregular = sequence_quality(normalized, [2, 2, 2], [1.0, 2.0, 8.0])

    assert strong["coverage"] > weak["coverage"]
    assert strong["score"] > weak["score"]
    assert strong["coherence"] > irregular["coherence"]


def test_opt_in_lattice_returns_monotonic_canonical_path():
    metadata = _metadata(6)
    features = np.eye(6, dtype=np.float32)
    provider = lambda descriptions: features[[1, 3, 5]]
    aligner = VisualTrakeDante(
        metadata,
        features,
        provider,
        lattice_enabled=True,
    )

    result = aligner.align(
        ["event one", "event two", "event three"],
        video_id="V1",
        lam=0.0,
        lattice_top_k=1,
        temporal_neighbor_radius=1,
    )

    winner = result["results"][0]
    assert result["diagnostics"]["lattice_enabled"] is True
    assert winner["scoring_mode"] == "lattice_path_baseline_rank"
    assert winner["lattice_raw_score"] is not None
    assert [row["frame_idx"] for row in winner["path"]] == [101, 103, 105]
    assert [row["pts_time"] for row in winner["path"]] == [1.0, 3.0, 5.0]
    assert winner["coverage"] > 0.90
    assert winner["coherence"] > 0.99


def test_lattice_falls_back_to_full_dante_when_candidates_are_not_orderable():
    metadata = _metadata(4)
    features = np.eye(4, dtype=np.float32)
    provider = lambda descriptions: features[[2, 1]]
    aligner = VisualTrakeDante(metadata, features, provider, lattice_enabled=True)

    result = aligner.align(
        ["event one", "event two"],
        video_id="V1",
        lam=0.0,
        lattice_top_k=1,
        temporal_neighbor_radius=0,
    )

    winner = result["results"][0]
    assert winner["scoring_mode"] == "legacy_fallback"
    assert [row["frame_idx"] for row in winner["path"]] == [100, 101]
    assert winner["path"][0]["frame_idx"] < winner["path"][1]["frame_idx"]


def test_feature_flag_off_preserves_legacy_result_shape_and_path():
    metadata = _metadata(5)
    features = np.eye(5, dtype=np.float32)
    provider = lambda descriptions: features[[1, 3]]
    aligner = VisualTrakeDante(metadata, features, provider)

    result = aligner.align(["event one", "event two"], video_id="V1", lam=0.0)
    winner = result["results"][0]

    assert result["diagnostics"]["lattice_enabled"] is False
    assert "scoring_mode" not in winner
    assert "candidate_lattice" not in winner
    assert [row["frame_idx"] for row in winner["path"]] == [101, 103]
