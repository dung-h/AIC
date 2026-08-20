from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.ocr_local_pipeline import (
    OCR_PROMPT,
    QwenLocalOCRBackend,
    _clean_ocr_text,
    _clean_batch_ocr_response,
    run_local_ocr,
    sample_frames,
)


def _metadata():
    return [
        {"video_id": "V2", "kf_n": 2, "pts_time": 11.0},
        {"video_id": "V2", "kf_n": 1, "pts_time": 0.0},
        {"video_id": "V2", "kf_n": 3, "pts_time": 20.0},
        {"video_id": "V1", "kf_n": 1, "pts_time": 0.0},
        {"video_id": "V1", "kf_n": 2, "pts_time": 4.0},
        {"video_id": "V1", "kf_n": 3, "pts_time": 12.0},
    ]


class FakeLocalOCR:
    def __init__(self):
        self.calls = []

    def recognize(self, image_path: str, prompt: str) -> str:
        self.calls.append((image_path, prompt))
        numeric_name = f"{int(Path(image_path).stem)}.jpg"
        return {"1.jpg": "TITLE", "2.jpg": "Two.", "3.jpg": "NONE"}[numeric_name]


class FakeBatchLocalOCR:
    def __init__(self, response=None):
        self.calls = []
        self.response = response

    def recognize_batch(self, image_paths: list[str], prompt: str):
        self.calls.append((list(image_paths), prompt))
        if self.response is not None:
            return self.response
        return [f"TEXT-{Path(image_path).parent.name}-{Path(image_path).stem}"
                for image_path in image_paths]


class FakeQwenVLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def answer_batch(self, image_paths, prompt, max_new_tokens):
        self.calls.append((list(image_paths), prompt, max_new_tokens))
        return self.response


def _make_frames(root: Path):
    for video_id in ("V1", "V2"):
        (root / video_id).mkdir(parents=True)
        for kf_n in (1, 2, 3):
            (root / video_id / f"{kf_n:03d}.jpg").write_bytes(b"fake-image")


def test_sampling_is_deterministic_and_bounded():
    rows = sample_frames(_metadata(), interval_seconds=5.0, max_frames_per_video=2)
    assert [(row["video_id"], row["kf_n"]) for row in rows] == [("V1", 1), ("V1", 3), ("V2", 1), ("V2", 3)]


def test_clean_ocr_repairs_vietnamese_and_korean_mojibake():
    assert _clean_ocr_text("HÃ n Quá»‘c...") == "Hàn Quốc..."
    assert _clean_ocr_text("ì°¬ì„±: 0") == "찬성: 0"


def test_clean_ocr_repairs_iteratively_and_preserves_unicode():
    original = "Hàn Quốc"
    twice_corrupted = original
    for _ in range(2):
        twice_corrupted = twice_corrupted.encode("utf-8").decode("cp1252")
    assert _clean_ocr_text(twice_corrupted) == original
    assert _clean_ocr_text("Hàn Quốc — 안녕하세요") == "Hàn Quốc — 안녕하세요"
    assert _clean_ocr_text("Cà phê") == "Cà phê"


def test_local_ocr_materializes_schema_and_does_not_bake_none(tmp_path):
    frame_root = tmp_path / "frames"
    _make_frames(frame_root)
    backend = FakeLocalOCR()
    output = tmp_path / "ocr.jsonl"
    report = run_local_ocr(_metadata(), frame_root, output, backend=backend, interval_seconds=5.0)

    assert report.backend_calls == 5
    assert report.text_rows == 3
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert set(records[0]) == {"video_id", "kf_n", "pts_time", "ocr_text"}
    assert {record["ocr_text"] for record in records} == {"TITLE", "Two."}
    assert all(record["ocr_text"].casefold() != "none" for record in records)
    assert OCR_PROMPT in {prompt for _, prompt in backend.calls}


def test_cache_resume_skips_processed_no_text_rows(tmp_path):
    frame_root = tmp_path / "frames"
    _make_frames(frame_root)
    output = tmp_path / "ocr.jsonl"
    cache = tmp_path / "cache.jsonl"
    first_backend = FakeLocalOCR()
    first = run_local_ocr(_metadata(), frame_root, output, backend=first_backend,
                          cache_path=cache, interval_seconds=5.0)
    second_backend = FakeLocalOCR()
    second = run_local_ocr(_metadata(), frame_root, output, backend=second_backend,
                           cache_path=cache, interval_seconds=5.0)
    assert first.backend_calls == 5
    assert second.backend_calls == 0
    assert second.cache_hits == 5
    assert second.no_text_rows == 2


def test_batch_backend_preserves_order_and_checkpoints_each_result(tmp_path):
    frame_root = tmp_path / "frames"
    _make_frames(frame_root)
    output = tmp_path / "ocr.jsonl"
    cache = tmp_path / "cache.jsonl"
    backend = FakeBatchLocalOCR()

    report = run_local_ocr(_metadata(), frame_root, output, backend=backend,
                           cache_path=cache, interval_seconds=5.0, batch_size=2)

    assert report.backend_calls == 3
    assert [len(paths) for paths, _ in backend.calls] == [2, 2, 1]
    cached_rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert [(row["video_id"], row["kf_n"]) for row in cached_rows] == [
        ("V1", 1), ("V1", 3), ("V2", 1), ("V2", 2), ("V2", 3)
    ]
    assert [row["ocr_text"] for row in cached_rows] == [
        "TEXT-V1-001", "TEXT-V1-003", "TEXT-V2-001", "TEXT-V2-002", "TEXT-V2-003"
    ]

    resumed = run_local_ocr(_metadata(), frame_root, output,
                            backend=FakeBatchLocalOCR(), cache_path=cache,
                            interval_seconds=5.0, batch_size=2)
    assert resumed.backend_calls == 0
    assert resumed.cache_hits == 5


def test_batch_size_falls_back_to_single_image_backend(tmp_path):
    frame_root = tmp_path / "frames"
    _make_frames(frame_root)
    backend = FakeLocalOCR()

    report = run_local_ocr(_metadata(), frame_root, tmp_path / "ocr.jsonl",
                           backend=backend, interval_seconds=5.0, batch_size=4)

    assert report.backend_calls == 5
    assert len(backend.calls) == 5


def test_batch_response_parsing_is_ordered_cleaned_and_strict():
    assert _clean_batch_ocr_response(
        '```json\n{"results": [" HÃ n ", "NONE"]}\n```', 2
    ) == ["Hàn", ""]
    assert _clean_batch_ocr_response("['TITLE', None]", 2) == ["TITLE", ""]
    with pytest.raises(ValueError, match="length is ambiguous"):
        _clean_batch_ocr_response('["only one"]', 2)
    with pytest.raises(ValueError, match="length is ambiguous"):
        _clean_batch_ocr_response('["one", "two", "three"]', 2)


def test_qwen_batch_adapter_uses_bounded_generation_and_cleans_response(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    backend = QwenLocalOCRBackend(model_path)
    fake_vlm = FakeQwenVLM('["TITLE", "NONE"]')
    backend._vlm = fake_vlm

    result = backend.recognize_batch(["one.jpg", "two.jpg"], OCR_PROMPT)

    assert result == ["TITLE", ""]
    assert fake_vlm.calls[0][0] == ["one.jpg", "two.jpg"]
    assert fake_vlm.calls[0][2] == 160
    assert fake_vlm.calls[0][1] == OCR_PROMPT


def test_batch_length_mismatch_fails_before_cache_write(tmp_path):
    frame_root = tmp_path / "frames"
    _make_frames(frame_root)
    output = tmp_path / "ocr.jsonl"
    cache = tmp_path / "cache.jsonl"

    with pytest.raises(ValueError, match="length is ambiguous"):
        run_local_ocr(_metadata(), frame_root, output,
                      backend=FakeBatchLocalOCR('["only one"]'),
                      cache_path=cache, interval_seconds=5.0, batch_size=2)

    assert not cache.exists()


def test_missing_local_qwen_model_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="local OCR backend unavailable"):
        QwenLocalOCRBackend(tmp_path / "missing-model")


def test_missing_frame_is_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="sampled keyframe does not exist"):
        run_local_ocr(_metadata()[:1], tmp_path / "frames", tmp_path / "ocr.jsonl",
                      backend=FakeLocalOCR(), interval_seconds=5.0)
