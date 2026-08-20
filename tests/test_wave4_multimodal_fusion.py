import threading
import time

import pytest

from src.trake.multimodal import EventLevelMultimodalDante, MissingModalityRetriever


class FakeRetriever:
    def __init__(self, modality, rows, *, delay=0.0, fail=False):
        self.modality = modality
        self.rows = rows
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        with self._lock:
            self.calls.append(event.index)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("synthetic retriever failure")
            allowed = set(map(str, candidate_videos)) if candidate_videos is not None else None
            return [
                {**row, "event_index": event.index, "modality": self.modality}
                for row in self.rows.get(event.index, [])
                if allowed is None or str(row["video_id"]) in allowed
            ][:top_k]
        finally:
            with self._lock:
                self.active -= 1


def _row(video_id, frame_idx, score, pts_time=None, source_id=""):
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "kf_n": frame_idx // 10,
        "pts_time": float(pts_time if pts_time is not None else frame_idx / 10),
        "score": score,
        "source_id": source_id,
    }


def test_true_fusion_prefers_shared_frame_and_retains_all_modalities():
    # Visual alone prefers frame 20. ASR alone prefers frame 10. A real fused
    # cell should prefer frame 10 because both modalities support it.
    visual = FakeRetriever(
        "visual",
        {0: [_row("v", 20, 0.99), _row("v", 10, 0.80)], 1: [_row("v", 30, 0.9)]},
    )
    asr = FakeRetriever(
        "asr",
        {0: [_row("v", 10, 0.99)], 1: [_row("v", 30, 0.9)]},
    )
    result = EventLevelMultimodalDante(
        {"visual": visual, "asr": asr}, modality_weights={"visual": 1.0, "asr": 1.0}
    ).align([
        {"description": "first", "modalities": ["visual", "asr"]},
        {"description": "second", "modalities": ["visual", "asr"]},
    ])

    assert result["results"][0]["frame_ids"] == [10, 30]
    first = result["results"][0]["path"][0]
    assert first["sources"] == ["asr", "visual"]
    assert {item["modality"] for item in first["evidence"]} == {"asr", "visual"}
    assert first["score"] > 1.0
    assert result["diagnostics"]["missing_is_unknown"] is True
    assert result["diagnostics"]["fused_candidate_count"] >= 3


def test_parallel_fanout_is_bounded_and_result_order_is_deterministic():
    rows = {0: [_row("v", 10, 0.9)], 1: [_row("v", 20, 0.9)]}
    visual = FakeRetriever("visual", rows, delay=0.02)
    asr = FakeRetriever("asr", rows, delay=0.02)
    ocr = FakeRetriever("ocr", rows, delay=0.02)
    aligner = EventLevelMultimodalDante(
        {"ocr": ocr, "visual": visual, "asr": asr}, max_workers=2
    )
    hits = aligner.retrieve_events(
        ["one", "two"], top_k_per_event=5
    )

    assert [item.event_index for item in hits[0]] == [0, 0, 0]
    assert [item.modality for item in hits[0]] == ["asr", "ocr", "visual"]
    assert aligner.last_diagnostics["parallel_workers"] == 2
    assert aligner.last_diagnostics["parallel_job_count"] == 6
    assert max(visual.max_active, asr.max_active, ocr.max_active) <= 2


def test_optional_missing_or_failing_modality_is_unknown_without_penalty():
    visual = FakeRetriever("visual", {0: [_row("v", 10, 0.8)]})
    failing_ocr = FakeRetriever("ocr", {}, fail=True)
    aligner = EventLevelMultimodalDante({"visual": visual, "ocr": failing_ocr})
    result = aligner.align(
        [{"description": "scene", "required_modalities": ["visual"], "optional_modalities": ["ocr"]}]
    )

    assert result["results"][0]["frame_ids"] == [10]
    assert result["results"][0]["path"][0]["sources"] == ["visual"]
    assert result["diagnostics"]["optional_skipped"]
    assert result["diagnostics"]["missing_is_unknown"] is True


def test_required_unavailable_modality_fails_closed_but_optional_is_skipped():
    aligner = EventLevelMultimodalDante({"visual": FakeRetriever("visual", {})})
    with pytest.raises(MissingModalityRetriever):
        aligner.align([{"description": "spoken", "required_modalities": ["asr"]}])

    result = aligner.align(
        [{"description": "scene", "optional_modalities": ["asr"]}], top_k_videos=1
    )
    assert result["results"] == []


def test_alignment_uses_canonical_frame_idx_and_limits_ranked_results_to_100():
    retrievers = {}
    for modality in ("visual", "asr"):
        retrievers[modality] = FakeRetriever(
            modality,
            {
                0: [_row("v", 30, 0.9, pts_time=3.0), _row("v", 10, 0.8, pts_time=1.0)],
                1: [_row("v", 20, 0.9, pts_time=2.0)],
            },
        )
    result = EventLevelMultimodalDante(retrievers).align(
        [{"description": "a", "modalities": ["visual", "asr"]}, {"description": "b", "modalities": ["visual", "asr"]}],
        top_k_videos=100,
    )
    assert result["results"][0]["frame_ids"] == [10, 20]
    assert all(
        left < right
        for left, right in zip(
            result["results"][0]["frame_ids"], result["results"][0]["frame_ids"][1:]
        )
    )
    assert result["diagnostics"]["canonical_order"] == "frame_idx"
