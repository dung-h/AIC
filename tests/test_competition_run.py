from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from pathlib import Path

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


def test_official_round1_qa_one_paragraph_formats_are_parsed():
    root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "AIC-HCMC-2025-main", "queries", "query-p1-groupA",
    )
    specs = competition_run.load_query_specs(Path(root))
    qa = {item.query_id: item for item in specs if item.task == "qa"}
    assert set(qa) == {"p1-15", "p1-19", "p1-22"}
    assert qa["p1-15"].question.startswith("Hỏi")
    assert qa["p1-15"].question_type == "unknown"
    assert qa["p1-15"].required_modalities is None
    assert "Hai câu thơ đó là gì?" in qa["p1-19"].question
    assert qa["p1-19"].question_type == "unknown"
    assert qa["p1-19"].required_modalities is None
    assert qa["p1-22"].question.startswith("Hỏi")
    assert qa["p1-22"].question_type == "unknown"
    assert qa["p1-22"].required_modalities is None


@pytest.mark.parametrize(
    ("text", "expected_question"),
    [
        (
            "Một hoạt động của câu lạc bộ diễn ra tại Khánh Hòa. Hỏi xã nào được nhắc đến?",
            "Hỏi xã nào được nhắc đến?",
        ),
        (
            "Phóng sự kể về Nguyễn Trung Trực ở Kiên Giang. Hỏi hai câu thơ đó là gì?",
            "Hỏi hai câu thơ đó là gì?",
        ),
    ],
    ids=("visible_place_is_not_forced_to_asr", "quote_is_not_forced_to_asr"),
)
def test_unlabelled_plain_qa_uses_neutral_contract(tmp_path, text, expected_question):
    path = tmp_path / "query-legacy-qa.txt"
    path.write_text(text, encoding="utf-8")

    spec = competition_run.parse_query_file(path)

    assert spec.question == expected_question
    assert spec.question_type == "unknown"
    assert spec.required_modalities is None


def test_explicit_json_qa_contract_remains_authoritative(tmp_path):
    path = tmp_path / "query-json-qa.txt"
    path.write_text(json.dumps({
        "query": "a presenter beside a recipe card",
        "question": "What ingredient is written on the card?",
        "question_type": "screen_text",
        "required_modalities": ["visual", "ocr"],
    }), encoding="utf-8")

    spec = competition_run.parse_query_file(path)

    assert spec.question_type == "screen_text"
    assert spec.required_modalities == ["visual", "ocr"]


@pytest.mark.parametrize(
    ("query", "question"),
    [
        ("Một hoạt động của câu lạc bộ tại Khánh Hòa", "Xã nào được nhắc đến?"),
        ("Phóng sự về Nguyễn Trung Trực", "Hai câu thơ đó là gì?"),
        ("Một người đứng cạnh công thức", "Tên món ăn là gì?"),
    ],
    ids=("place_not_forced_to_asr", "quote_not_forced_to_asr", "recipe_not_forced_to_ocr"),
)
def test_unlabelled_json_qa_uses_neutral_contract(tmp_path, query, question):
    path = tmp_path / "query-json-qa.txt"
    path.write_text(json.dumps({"query": query, "question": question}), encoding="utf-8")

    spec = competition_run.parse_query_file(path)

    assert spec.question_type == "unknown"
    assert spec.required_modalities is None


def test_mixed_runner_openai_profile_allows_only_vlm_network_and_uses_policy_trake_mode(
        monkeypatch, tmp_path):
    calls = []
    policies = []
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    class FakePipeline:
        def __init__(self, policy):
            self.policy = policy
            policies.append(policy)

        def kis(self, query, topk):
            return {"results": [("V1", 10, 0.1, 0.9)]}

        def vqa_ranked(self, query, question, **kwargs):
            calls.append(kwargs)
            return {"answers": [{"video_id": "V1", "frame_id": 20, "answer": "25°C"}]}

        def trake(self, events, topk, mode):
            assert mode == "asr"
            return {"results": [{
                "video_id": "V1",
                "path": [{"frame_idx": 10}, {"frame_idx": 30}],
            }]}

    import src.pipelines.hcmai_pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", FakePipeline)
    monkeypatch.setattr(
        competition_run, "load_production_canonical_registry",
        lambda: ({("V1", 10), ("V1", 20), ("V1", 30)}, {"sha256": "test"}),
    )
    monkeypatch.setattr(
        competition_run.RuntimePolicy, "from_env",
        classmethod(lambda cls: RuntimePolicy(
            trake_mode="asr", kis_remote_translation=True,
            trake_remote_embeddings=True,
        )),
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
    assert calls[0]["max_vlm_candidates"] == 20
    assert policies == [RuntimePolicy(
        execution_mode="production", network_mode="online",
        vqa_answer_provider="openai", trake_mode="asr",
        kis_remote_translation=False, trake_remote_embeddings=False,
        vqa_modality_routing=True, vqa_visual_selector_policy="balanced",
    )]
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


def test_mixed_runner_local_profile_locks_all_network_features_off(monkeypatch, tmp_path):
    policies = []

    class FakePipeline:
        def __init__(self, policy):
            policies.append(policy)

        def kis(self, query, topk):
            return {"results": [("V1", 10, 0.1, 0.9)]}

    import src.pipelines.hcmai_pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", FakePipeline)
    monkeypatch.setattr(
        competition_run, "load_production_canonical_registry",
        lambda: ({("V1", 10)}, {"sha256": "test"}),
    )
    monkeypatch.setattr(
        competition_run.RuntimePolicy, "from_env",
        classmethod(lambda cls: RuntimePolicy(
            kis_remote_translation=True,
            trake_remote_embeddings=True,
        )),
    )

    report = competition_run.run_competition(
        [competition_run.QuerySpec("1", "kis", query="cooking")],
        output=tmp_path / "submission.zip", answer_provider="local",
        modality_routing=False, topk=1, max_vlm_candidates=None,
    )

    assert report["ready"] is True
    assert policies == [RuntimePolicy(
        execution_mode="production", network_mode="offline",
        vqa_answer_provider="local", kis_remote_translation=False,
        trake_remote_embeddings=False, vqa_modality_routing=False,
        vqa_visual_selector_policy="balanced",
    )]


def test_local_vlm_can_use_explicit_external_hypotheses_without_becoming_a_remote_vlm(
        monkeypatch, tmp_path):
    policies = []
    calls = []

    class FakePipeline:
        def __init__(self, policy):
            self.policy = policy
            policies.append(policy)

        def vqa_ranked(self, query, question, **kwargs):
            calls.append(kwargs)
            return {"answers": [{"video_id": "V1", "frame_id": 20, "answer": "Giang Ly"}]}

    import src.pipelines.hcmai_pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "HCMAIPipeline", FakePipeline)
    monkeypatch.setattr(
        competition_run, "load_production_canonical_registry",
        lambda: ({("V1", 20)}, {"sha256": "test"}),
    )
    monkeypatch.setattr(
        competition_run.RuntimePolicy, "from_env",
        classmethod(lambda cls: RuntimePolicy(
            vqa_external_search_url="https://search.example.org",
            vqa_external_allowed_domains=("example.org",),
        )),
    )

    competition_run.run_competition(
        [competition_run.QuerySpec("qa", "qa", query="FANA", question="Tên xã nào?")],
        output=tmp_path / "submission.zip", answer_provider="local",
        modality_routing=True, topk=1, max_vlm_candidates=None,
        external_grounding=True,
    )

    assert calls[0]["offline"] is False
    assert calls[0]["external_grounding"] is True
    assert policies[0].network_mode == "online"
    assert policies[0].vqa_answer_provider == "local"
    assert policies[0].vqa_external_grounding is True
