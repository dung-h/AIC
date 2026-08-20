import json
from pathlib import Path

from src.eval import benchmark_qna_gt_frame_oracle as oracle


class FakeVLM:
    def __init__(self, answer="A fork."):
        self.answer_text = answer
        self.calls = []

    def answer(self, image_path, prompt, max_new_tokens=32):
        self.calls.append((image_path, prompt, max_new_tokens))
        return self.answer_text


def _row():
    return {
        "annotation_id": "q1",
        "split": "dev",
        "question_type": "action",
        "video_id": "v1",
        "kf_n": 2,
        "frame_idx": 20,
        "query": "A plate is shown.",
        "question": "What utensil is visible?",
        "answer": "A fork.",
        "acceptable_kf_n": "2",
        "status": "valid",
    }


def _fixture(tmp_path):
    frame_root = tmp_path / "frames"
    frame_dir = frame_root / "v1"
    frame_dir.mkdir(parents=True)
    frame = frame_dir / "002.jpg"
    frame.write_bytes(b"fake-jpeg")
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    return input_path, frame_root


def test_gt_oracle_uses_primary_canonical_frame_and_records_raw_normalized_latency(tmp_path):
    input_path, frame_root = _fixture(tmp_path)
    vlm = FakeVLM(" A fork. ")
    report = oracle.evaluate(input_path, frame_root=frame_root, canonical_index=None,
                             model_path=tmp_path / "unused-model", vlm=vlm,
                             max_new_tokens=7)
    row = report["rows"][0]
    assert len(vlm.calls) == 1
    assert Path(vlm.calls[0][0]).parts[-2:] == ("v1", "002.jpg")
    assert vlm.calls[0][2] == 7
    assert "A fork." not in vlm.calls[0][1]
    assert row["raw_answer"] == " A fork. "
    assert row["normalized_answer"] == "a fork"
    assert row["latency_ms"] >= 0
    assert row["exact_match"] is True
    assert row["normalized_exact_match"] is True


def test_gt_oracle_cache_avoids_second_vlm_call(tmp_path):
    input_path, frame_root = _fixture(tmp_path)
    cache = tmp_path / "oracle-cache.json"
    first_vlm = FakeVLM("fork")
    first = oracle.evaluate(input_path, frame_root=frame_root, canonical_index=None,
                            model_path=tmp_path / "unused-model", vlm=first_vlm,
                            cache_path=cache)
    second_vlm = FakeVLM("different")
    second = oracle.evaluate(input_path, frame_root=frame_root, canonical_index=None,
                             model_path=tmp_path / "unused-model", vlm=second_vlm,
                             cache_path=cache)
    assert len(first_vlm.calls) == 1
    assert len(second_vlm.calls) == 0
    assert second["cache_hits"] == 1
    assert second["rows"][0]["raw_answer"] == "fork"


def test_gt_oracle_fails_clearly_for_missing_frame(tmp_path):
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    try:
        oracle.evaluate(input_path, frame_root=tmp_path / "missing", canonical_index=None,
                        model_path=tmp_path / "unused-model", vlm=FakeVLM())
    except FileNotFoundError as exc:
        assert "missing canonical GT frame" in str(exc)
    else:
        raise AssertionError("missing GT frame should fail closed")
