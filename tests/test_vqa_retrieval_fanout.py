from __future__ import annotations

import threading

import pytest

from src.vqa.retrieval_fanout import RetrievalFanoutError, run_retrieval_fanout


def test_fanout_runs_independent_channels_concurrently_and_preserves_declared_order():
    visual_started = threading.Event()
    release_visual = threading.Event()

    def visual():
        visual_started.set()
        return {"released_while_visual_running": release_visual.wait(timeout=0.5)}

    def asr():
        assert visual_started.wait(timeout=0.5)
        release_visual.set()
        return ["asr"]

    result = run_retrieval_fanout({"visual": visual, "asr": asr})

    assert list(result.results) == ["visual", "asr"]
    assert result.parallel is True
    assert result.results["visual"]["released_while_visual_running"] is True
    assert result.results["asr"] == ["asr"]
    assert set(result.timings_ms) == {"visual", "asr"}


def test_fanout_waits_for_all_channels_then_surfaces_named_error():
    completed = threading.Event()

    def failing():
        raise ValueError("broken index")

    def other():
        completed.set()
        return "done"

    with pytest.raises(RetrievalFanoutError, match="'asr'.*ValueError"):
        run_retrieval_fanout({"asr": failing, "ocr": other})
    assert completed.is_set()


def test_fanout_rejects_invalid_or_duplicate_task_names():
    with pytest.raises(ValueError, match="callables"):
        run_retrieval_fanout({"": lambda: None})
