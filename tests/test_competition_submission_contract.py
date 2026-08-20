"""Official AIC26 submission transport contract tests."""
from __future__ import annotations

import csv
import io
import sys
import types
import zipfile
from types import SimpleNamespace

import pytest

from src.pipelines.codabench_submit import (
    format_aic26_query_csv,
    write_aic26_mixed_submission_zip,
    write_aic26_submission_zip,
)
from src.runtime_policy import RuntimePolicy


@pytest.fixture
def canonical():
    return {("V1", 10), ("V1", 20), ("V1", 30), ("V2", 42)}


def _read_csv_member(bundle: zipfile.ZipFile, name: str):
    raw = bundle.read(name)
    text = raw.decode("utf-8")
    return raw, list(csv.reader(io.StringIO(text)))


@pytest.mark.parametrize(
    ("task", "answers", "event_count", "expected"),
    [
        ("kis", [{"video_id": "V2", "frame_idx": 42}], None, [["V2", "42"]]),
        (
            "qa",
            [{"video_id": "V2", "frame_id": 42, "answer": "Nha Trang, 25°"}],
            None,
            [["V2", "42", "Nha Trang, 25°"]],
        ),
        (
            "trake",
            [{"video_id": "V1", "frame_ids": [10, 20, 30]}],
            3,
            [["V1", "10", "20", "30"]],
        ),
    ],
)
def test_each_task_is_headerless_utf8_csv(task, answers, event_count, expected, canonical):
    raw = format_aic26_query_csv(
        task,
        "7",
        answers,
        canonical_frames=canonical,
        event_count=event_count,
    )
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert list(csv.reader(io.StringIO(raw.decode("utf-8")))) == expected
    assert b"video_id" not in raw


def test_zip_has_submission_root_and_official_query_filenames(tmp_path, canonical):
    output = tmp_path / "answer.zip"
    report = write_aic26_submission_zip(
        "qa",
        {
            "1": [{"video_id": "V2", "frame_id": 42, "answer": "mưa"}],
            "query-2": [{"video_id": "V1", "frame_id": 10, "answer": "nắng"}],
        },
        output,
        canonical_frames=canonical,
    )

    assert report["query_count"] == 2
    assert report["row_count"] == 2
    with zipfile.ZipFile(output) as bundle:
        assert bundle.namelist() == [
            "submission/",
            "submission/query-1-qa.csv",
            "submission/query-2-qa.csv",
        ]
        _, rows = _read_csv_member(bundle, "submission/query-1-qa.csv")
        assert rows == [["V2", "42", "mưa"]]


def test_mixed_zip_contains_all_three_tasks_and_preserves_rank_order(tmp_path, canonical):
    output = tmp_path / "answer.zip"
    report = write_aic26_mixed_submission_zip(
        [
            {
                "task": "kis", "query_id": "1",
                "answers": [
                    {"video_id": "V2", "frame_idx": 42},
                    {"video_id": "V1", "frame_idx": 10},
                ],
            },
            {
                "task": "qa", "query_id": "2",
                "answers": [{"video_id": "V1", "frame_id": 20, "answer": "mưa"}],
            },
            {
                "task": "trake", "query_id": "3", "event_count": 2,
                "answers": [{"video_id": "V1", "frame_ids": [10, 30]}],
            },
        ],
        output,
        canonical_frames_by_task={
            "kis": canonical, "qa": canonical, "trake": canonical,
        },
    )

    assert report["task_counts"] == {"kis": 1, "qa": 1, "trake": 1}
    assert report["row_count"] == 4
    with zipfile.ZipFile(output) as bundle:
        assert bundle.namelist() == [
            "submission/",
            "submission/query-1-kis.csv",
            "submission/query-2-qa.csv",
            "submission/query-3-trake.csv",
        ]
        _, rows = _read_csv_member(bundle, "submission/query-1-kis.csv")
        assert rows == [["V2", "42"], ["V1", "10"]]


def test_mixed_zip_fails_closed_before_replacing_existing_file(tmp_path, canonical):
    output = tmp_path / "answer.zip"
    output.write_bytes(b"known-good")
    with pytest.raises(ValueError, match="strictly increasing"):
        write_aic26_mixed_submission_zip(
            [{
                "task": "trake", "query_id": "3", "event_count": 2,
                "answers": [{"video_id": "V1", "frame_ids": [30, 10]}],
            }],
            output,
            canonical_frames_by_task={"trake": canonical},
        )
    assert output.read_bytes() == b"known-good"


def test_qa_rejects_empty_overlong_and_more_than_100_rows(canonical):
    base = {"video_id": "V2", "frame_id": 42}
    for answer in (None, "", "   ", "two\nlines"):
        with pytest.raises(ValueError, match="answer"):
            format_aic26_query_csv(
                "qa", "1", [{**base, "answer": answer}], canonical_frames=canonical
            )
    with pytest.raises(ValueError, match="100 characters"):
        format_aic26_query_csv(
            "qa", "1", [{**base, "answer": "á" * 101}], canonical_frames=canonical
        )
    with pytest.raises(ValueError, match="at most 100 rows"):
        format_aic26_query_csv(
            "qa",
            "1",
            [{**base, "answer": str(index)} for index in range(101)],
            canonical_frames=canonical,
        )


@pytest.mark.parametrize("frame", [True, 10.0, "10.0", -1, "not-an-int"])
def test_frames_must_be_nonnegative_integers(frame, canonical):
    with pytest.raises(ValueError, match="frame_idx"):
        format_aic26_query_csv(
            "kis",
            "1",
            [{"video_id": "V1", "frame_idx": frame}],
            canonical_frames=canonical,
        )


def test_trake_requires_exact_n_strict_order_and_canonical_frames(canonical):
    cases = [
        ([10, 20], 3, "expected 3"),
        ([20, 10], 2, "strictly increasing"),
        ([10, 10], 2, "strictly increasing"),
        ([10, 999], 2, "non-canonical"),
        ([10, 20.0], 2, "frame_idx"),
    ]
    for frames, count, message in cases:
        with pytest.raises(ValueError, match=message):
            format_aic26_query_csv(
                "trake",
                "1",
                [{"video_id": "V1", "frame_ids": frames}],
                canonical_frames=canonical,
                event_count=count,
            )


def test_packaging_fails_before_writing_for_bad_contract(tmp_path, canonical):
    output = tmp_path / "answer.zip"
    with pytest.raises(ValueError, match="query_id"):
        write_aic26_submission_zip(
            "kis",
            {"../escape": [{"video_id": "V1", "frame_idx": 10}]},
            output,
            canonical_frames=canonical,
        )
    assert not output.exists()

    with pytest.raises(ValueError, match=".zip"):
        write_aic26_submission_zip(
            "kis",
            {"1": [{"video_id": "V1", "frame_idx": 10}]},
            tmp_path / "answer.csv",
            canonical_frames=canonical,
        )


def test_cli_official_qa_writes_one_csv_per_query(monkeypatch, tmp_path):
    from src.pipelines import codabench_submit

    class FakePipeline:
        def __init__(self, policy=None):
            self.policy = policy or RuntimePolicy()
            self._vqa_ranked = SimpleNamespace()

        def vqa_ranked(self, query, question, **kwargs):
            assert kwargs["max_answers"] == 20
            assert kwargs["max_vlm_candidates"] == 12
            return {"answers": [{"video_id": "V2", "frame_id": 42, "answer": "25°C"}]}

    fake_module = types.ModuleType("hcmai_pipeline")
    fake_module.HCMAIPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "hcmai_pipeline", fake_module)
    monkeypatch.setattr(codabench_submit, "_canonical_pairs", lambda *args: {("V2", 42)})

    input_path = tmp_path / "queries.csv"
    input_path.write_text(
        "query_id,query,question,task_type\n"
        "9,dự báo thời tiết,Nhiệt độ là bao nhiêu?,VQA\n",
        encoding="utf-8",
    )
    output = tmp_path / "submission.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codabench_submit.py",
            "--compatibility-only",
            "--input", str(input_path),
            "--output", str(output),
            "--task", "VQA",
            "--topk", "100",
            "--offline",
        ],
    )

    codabench_submit.main()

    with zipfile.ZipFile(output) as bundle:
        assert bundle.namelist() == ["submission/", "submission/query-9-qa.csv"]
        _, rows = _read_csv_member(bundle, "submission/query-9-qa.csv")
        assert rows == [["V2", "42", "25°C"]]


def test_cli_official_qa_allows_explicit_online_provider(monkeypatch, tmp_path):
    from src.pipelines import codabench_submit

    observed = {}

    class FakePipeline:
        def __init__(self, policy=None):
            observed["policy"] = policy
            self.policy = policy
            self._vqa_ranked = SimpleNamespace()

        def vqa_ranked(self, query, question, **kwargs):
            observed["kwargs"] = kwargs
            return {"answers": [{"video_id": "V2", "frame_id": 42, "answer": "BENVENUTI"}]}

    fake_module = types.ModuleType("hcmai_pipeline")
    fake_module.HCMAIPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "hcmai_pipeline", fake_module)
    monkeypatch.setattr(codabench_submit, "_canonical_pairs", lambda *args: {("V2", 42)})
    input_path = tmp_path / "queries.csv"
    input_path.write_text(
        "query_id,query,question,task_type\n"
        "9,a large sign,What is written?,VQA\n",
        encoding="utf-8",
    )
    output = tmp_path / "submission.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codabench_submit.py", "--compatibility-only", "--input", str(input_path),
            "--output", str(output), "--task", "VQA",
            "--answer-provider", "openai",
        ],
    )

    codabench_submit.main()

    assert observed["policy"].vqa_answer_provider == "openai"
    assert observed["policy"].network_mode == "online"
    assert observed["kwargs"]["max_vlm_candidates"] == 3
    with zipfile.ZipFile(output) as bundle:
        _, rows = _read_csv_member(bundle, "submission/query-9-qa.csv")
        assert rows == [["V2", "42", "BENVENUTI"]]


def test_cli_official_kis_branch_uses_two_columns(monkeypatch, tmp_path):
    from src.pipelines import codabench_submit

    class FakePipeline:
        def __init__(self, policy=None):
            self.policy = policy or RuntimePolicy()

        def kis(self, query, topk):
            return {"results": [("V1", 10, 0.4, 0.9)]}

    fake_module = types.ModuleType("hcmai_pipeline")
    fake_module.HCMAIPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "hcmai_pipeline", fake_module)
    monkeypatch.setattr(codabench_submit, "_canonical_pairs", lambda *args: {("V1", 10)})

    input_path = tmp_path / "kis.csv"
    input_path.write_text("query_id,query\n1,người đang nấu ăn\n", encoding="utf-8")
    output = tmp_path / "kis.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codabench_submit.py", "--compatibility-only", "--input", str(input_path),
            "--output", str(output),
        ],
    )

    codabench_submit.main()

    with zipfile.ZipFile(output) as bundle:
        _, rows = _read_csv_member(bundle, "submission/query-1-kis.csv")
        assert rows == [["V1", "10"]]


def test_cli_official_trake_branch_uses_video_plus_n_frames(monkeypatch, tmp_path):
    from src.pipelines import codabench_submit

    class FakePipeline:
        def __init__(self, policy=None):
            self.policy = policy or RuntimePolicy()

        def trake(self, events, topk, mode):
            assert events == ["event A", "event B"]
            return {
                "results": [{
                    "video_id": "V1",
                    "path": [{"frame_idx": 10}, {"frame_idx": 30}],
                    "score": 0.8,
                }]
            }

    fake_module = types.ModuleType("hcmai_pipeline")
    fake_module.HCMAIPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "hcmai_pipeline", fake_module)
    monkeypatch.setattr(
        codabench_submit,
        "_canonical_pairs",
        lambda *args: {("V1", 10), ("V1", 30)},
    )

    input_path = tmp_path / "trake.csv"
    input_path.write_text(
        "query_id,query,event_1,event_2\n1,sequence,event A,event B\n",
        encoding="utf-8",
    )
    output = tmp_path / "trake.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codabench_submit.py",
            "--compatibility-only",
            "--input", str(input_path),
            "--output", str(output),
            "--task", "TRAKE",
            "--offline",
        ],
    )

    codabench_submit.main()

    with zipfile.ZipFile(output) as bundle:
        _, rows = _read_csv_member(bundle, "submission/query-1-trake.csv")
        assert rows == [["V1", "10", "30"]]


def test_cli_requires_explicit_compatibility_opt_in(monkeypatch, tmp_path, capsys):
    """The standalone CLI must not accidentally become a production owner."""
    from src.pipelines import codabench_submit

    output = tmp_path / "submission.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "codabench_submit.py", "--input", str(tmp_path / "queries.csv"),
            "--output", str(output),
        ],
    )

    with pytest.raises(SystemExit) as error:
        codabench_submit.main()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "compatibility-only" in stderr
    assert "./scripts/competition.sh run" in stderr
    assert not output.exists()
