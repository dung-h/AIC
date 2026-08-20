from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.ocr_adaptive_v2 import (
    OCRAdaptiveV2Runner,
    QwenLocalOCRBackend,
    align_legacy_ocr,
    load_canonical,
    select_adaptive_candidates,
)


def _canonical() -> pd.DataFrame:
    return pd.DataFrame([
        {"video_id": "K01_V001", "kf_n": 1, "frame_idx": 10, "pts_time": 0.0},
        {"video_id": "K01_V001", "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
        {"video_id": "K01_V001", "kf_n": 3, "frame_idx": 30, "pts_time": 4.0},
        {"video_id": "K01_V001", "kf_n": 4, "frame_idx": 40, "pts_time": 6.0},
        {"video_id": "K01_V001", "kf_n": 5, "frame_idx": 50, "pts_time": 8.0},
        {"video_id": "K01_V002", "kf_n": 1, "frame_idx": 110, "pts_time": 1.0},
        {"video_id": "K01_V002", "kf_n": 2, "frame_idx": 120, "pts_time": 3.0},
        {"video_id": "K01_V002", "kf_n": 3, "frame_idx": 130, "pts_time": 5.0},
        {"video_id": "L21_V001", "kf_n": 1, "frame_idx": 210, "pts_time": 1.0},
        {"video_id": "L21_V001", "kf_n": 2, "frame_idx": 220, "pts_time": 3.0},
    ])


def _legacy() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "video_id": "K01_V001", "kf_n": 2, "frame_idx": 20,
            "pts_time": 2.0, "ocr_text": "HÃ  Ná»™i",
        },
        {
            "video_id": "L21_V001", "kf_n": 1, "frame_idx": 999,
            "pts_time": 1.5, "ocr_text": "NONE",
        },
    ])


class FakeQwen:
    def __init__(self, fail_key: tuple[str, int] | None = None):
        self.fail_key = fail_key
        self.calls: list[tuple[str, str]] = []

    def recognize(self, image_path: str, prompt: str) -> str:
        self.calls.append((image_path, prompt))
        video_id = Path(image_path).parent.name
        kf_n = int(Path(image_path).stem)
        if self.fail_key == (video_id, kf_n):
            raise RuntimeError("synthetic interruption")
        return "PRICE 25" if kf_n % 2 else "NONE"


def _frames(root: Path) -> None:
    for video_id, count in (("K01_V001", 5), ("K01_V002", 3), ("L21_V001", 2)):
        folder = root / video_id
        folder.mkdir(parents=True)
        for kf_n in range(1, count + 1):
            (folder / f"{kf_n:03d}.jpg").write_bytes(b"fake-image")


def test_legacy_rows_are_canonical_aligned_and_diagnostic_only():
    canonical = load_canonical(_canonical())
    aligned = align_legacy_ocr(_legacy(), canonical)

    row = aligned[(aligned.video_id == "K01_V001") & (aligned.kf_n == 2)].iloc[0]
    assert row.frame_idx == 20
    assert row.pts_time == 2.0
    assert row.legacy_text == "Hà Nội"
    assert bool(row.legacy_has_text) is True

    mismatch = aligned[aligned.video_id == "L21_V001"].iloc[0]
    assert mismatch.frame_idx == 210
    assert mismatch.alignment_status == "canonical_aligned_with_metadata_mismatch"
    assert bool(mismatch.legacy_has_text) is False

    with pytest.raises(ValueError, match="outside canonical map"):
        align_legacy_ocr(
            pd.DataFrame([{"video_id": "K01_V001", "kf_n": 99, "ocr_text": "x"}]),
            canonical,
        )


def test_directory_seed_ignores_compare_and_gt_artifacts(tmp_path: Path):
    _legacy().to_parquet(tmp_path / "ocr_k01.parquet", index=False)
    pd.DataFrame([{"video_id": "K01_V001", "kf_n": 3, "ocr_text": "no metadata"}]).to_parquet(
        tmp_path / "ocr_k02.parquet", index=False
    )
    pd.DataFrame([{"video_id": "K01_V001", "kf_n": 1, "llm": "answer"}]).to_parquet(
        tmp_path / "ocr_compare.parquet", index=False
    )
    pd.DataFrame([{"video_id": "K01_V001", "kf_n": 1, "ocr": "ground truth"}]).to_parquet(
        tmp_path / "ocr_gt_texts.parquet", index=False
    )
    aligned = align_legacy_ocr(tmp_path, _canonical())
    assert len(aligned) == len(_legacy()) + 1
    assert set(aligned["legacy_source"]) == {
        str(tmp_path / "ocr_k01.parquet"), str(tmp_path / "ocr_k02.parquet")
    }
    assert bool(aligned.loc[aligned["kf_n"] == 3, "frame_idx_matches"].iloc[0]) is False


def test_candidate_selection_is_bounded_deterministic_and_unique():
    canonical = load_canonical(_canonical())
    legacy = align_legacy_ocr(_legacy(), canonical)
    first = select_adaptive_candidates(
        canonical, legacy, max_candidates_per_video=4, min_candidates_per_video=3,
        neighbor_radius=1, video_ids=["K01_V001", "K01_V002"],
    )
    second = select_adaptive_candidates(
        canonical, legacy, max_candidates_per_video=4, min_candidates_per_video=3,
        neighbor_radius=1, video_ids=["K01_V001", "K01_V002"],
    )
    assert first[["video_id", "kf_n", "candidate_source"]].to_dict("records") == second[[
        "video_id", "kf_n", "candidate_source"
    ]].to_dict("records")
    assert not first.duplicated(["video_id", "kf_n"]).any()
    assert first.groupby("video_id").size().to_dict() == {"K01_V001": 4, "K01_V002": 3}
    seed = first[(first.video_id == "K01_V001") & (first.kf_n == 2)].iloc[0]
    assert seed.candidate_source == "legacy_seed"
    assert bool(seed.legacy_seed) is True
    assert int(seed.frame_idx) == 20


def test_dry_run_writes_diagnostic_and_candidate_artifacts_without_backend(tmp_path: Path):
    backend = FakeQwen()
    output = tmp_path / "adaptive"
    report = OCRAdaptiveV2Runner(
        output_dir=output, model_path=tmp_path / "missing-model", backend=backend
    ).run(
        _canonical(), _legacy(), tmp_path / "frames", video_ids=["K01_V001"],
        max_candidates_per_video=3, min_candidates_per_video=3,
        execute=False, dry_run=True,
    )
    assert report["status"] == "dry_run"
    assert report["provenance"]["network_allowed"] is False
    assert report["legacy_diagnostics"]["promoted_as_qwen_output"] is False
    assert report["coverage"]["scope_type"] == "adaptive_candidate_subset"
    assert backend.calls == []
    assert (output / "legacy_diagnostics.parquet").exists()
    assert (output / "candidates.parquet").exists()
    assert not (output / "attempts.jsonl").exists()


def test_resume_retries_only_errors_and_preserves_provenance(tmp_path: Path):
    frame_root = tmp_path / "frames"
    _frames(frame_root)
    output = tmp_path / "adaptive"
    first_backend = FakeQwen(("K01_V001", 3))
    first = OCRAdaptiveV2Runner(output, tmp_path / "missing-model", backend=first_backend).run(
        _canonical(), _legacy(), frame_root, video_ids=["K01_V001"],
        max_candidates_per_video=4, min_candidates_per_video=3,
    )
    assert first["status"] == "blocked"
    assert first["coverage"]["error_candidate_rows"] == 1
    assert first["provenance"]["api_used"] is False

    second_backend = FakeQwen()
    second = OCRAdaptiveV2Runner(output, tmp_path / "missing-model", backend=second_backend).run(
        _canonical(), _legacy(), frame_root, video_ids=["K01_V001"],
        max_candidates_per_video=4, min_candidates_per_video=3, resume=True,
    )
    assert second["status"] == "completed"
    assert len(second_backend.calls) == 1
    assert Path(second_backend.calls[0][0]).stem == "003"
    assert second["coverage"]["candidate_coverage_ratio"] == 1.0
    assert second["coverage"]["by_video"]["K01_V001"]["canonical_rows"] == 5
    assert second["coverage"]["by_video"]["K01_V001"]["selected_candidates"] == 4

    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(attempts) == 5  # four initial attempts plus one retry
    retry = [row for row in attempts if row["kf_n"] == 3][-1]
    assert retry["status"] == "text"
    assert retry["frame_idx"] == 30
    assert retry["source"] == "adaptive_qwen_candidate"
    assert retry["network_allowed"] is False
    assert (output / "qwen_ocr.parquet").exists()

    third_backend = FakeQwen()
    third = OCRAdaptiveV2Runner(output, tmp_path / "missing-model", backend=third_backend).run(
        _canonical(), _legacy(), frame_root, video_ids=["K01_V001"],
        max_candidates_per_video=4, min_candidates_per_video=3, resume=True,
    )
    assert third["status"] == "completed"
    assert third_backend.calls == []


def test_missing_local_model_fails_without_fallback(tmp_path: Path):
    with pytest.raises(RuntimeError, match="local Qwen model directory"):
        QwenLocalOCRBackend(tmp_path / "missing-model")
