from __future__ import annotations

import csv
import io
import json
import os
import zipfile

import pytest

from src.cli import competition_run
from src.runtime_policy import RuntimePolicy


def test_query_directory_parser_supports_official_names_and_safe_formats(tmp_path):
    (tmp_path / "query-1-kis.txt").write_text("người đang nấu ăn", encoding="utf-8")
    (tmp_path / "query-2-qa.txt").write_text(
        "Description: a weather presenter\nQuestion: According to the presenter, what city is named?\n",
        encoding="utf-8",
    )
    (tmp_path / "query-3-trake.txt").write_text(
        json.dumps({"events": ["oil enters the pan", "meat enters the pan"]}),
        encoding="utf-8",
    )

    specs = competition_run.load_query_specs(tmp_path)

    assert [(item.query_id, item.task) for item in specs] == [
        ("1", "kis"), ("2", "qa"), ("3", "trake"),
    ]
    assert specs[1].question.startswith("According to")
    assert specs[2].events == ("oil enters the pan", "meat enters the pan")


def test_ambiguous_qa_text_fails_closed(tmp_path):
    path = tmp_path / "query-2-qa.txt"
    path.write_text("one unlabelled line", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous Q&A"):
        competition_run.parse_query_file(path)


def test_mixed_runner_uses_api_online_and_writes_one_package(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    class FakePipeline:
        def __init__(self, policy):
            self.policy = policy

        def kis(self, query, topk):
            return {"results": [("V1", 10, 0.1, 0.9)]}

        def vqa_ranked(self, query, question, **kwargs):
            calls.append(kwargs)
            return {"answers": [{"video_id": "V1", "frame_id": 20, "answer": "25°C"}]}

        def trake(self, events, topk, mode):
            assert mode == "visual"
            return {"results": [{
                "video_id": "V1",
                "path": [{"frame_idx": 10}, {"frame_idx": 30}],
            }]}

    import src.pipelines.hcmai_pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", FakePipeline)
    monkeypatch.setattr(
        competition_run, "_canonical_pairs",
        lambda *_args: {("V1", 10), ("V1", 20), ("V1", 30)},
    )
    monkeypatch.setattr(
        competition_run.RuntimePolicy, "from_env",
        classmethod(lambda cls: RuntimePolicy()),
    )
    specs = [
        competition_run.QuerySpec("1", "kis", query="cooking"),
        competition_run.QuerySpec("2", "qa", query="weather", question="temperature?"),
        competition_run.QuerySpec("3", "trake", events=("first", "second")),
    ]
    output = tmp_path / "submission.zip"

    report = competition_run.run_competition(
        specs, output=output, answer_provider="openai", modality_routing=True,
        topk=20, max_vlm_candidates=None,
    )

    assert report["ready"] is True
    assert report["task_counts"] == {"kis": 1, "qa": 1, "trake": 1}
    assert calls[0]["offline"] is False
    assert calls[0]["max_vlm_candidates"] == 3
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    with zipfile.ZipFile(output) as bundle:
        assert set(bundle.namelist()) == {
            "submission/", "submission/query-1-kis.csv",
            "submission/query-2-qa.csv", "submission/query-3-trake.csv",
        }
        rows = list(csv.reader(io.StringIO(
            bundle.read("submission/query-2-qa.csv").decode("utf-8")
        )))
        assert rows == [["V1", "20", "25°C"]]
