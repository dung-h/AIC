from __future__ import annotations

import json
from pathlib import Path

from src.eval.run_vqa_provider_benchmark import (
    SCHEMA_VERSION,
    build_offline_providers,
    default_cases,
    main,
    run_benchmark,
)


def test_offline_ab_runs_both_real_adapters_without_network():
    report = run_benchmark(mode="offline")

    assert report["status"] == "ok"
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["providers_requested"] == ["local", "openai"]
    assert len(report["records"]) == 6
    assert all(row["error"] is None for row in report["records"])
    assert {row["provider"] for row in report["records"]} == {"local", "openai"}
    assert sum(row["status"] == "abstained" for row in report["records"]) == 2
    assert all(row["answer_contract"]["valid"] for row in report["records"])
    assert all(row["latency_ms"] >= 0 for row in report["records"])
    assert all("api_key" not in json.dumps(row) for row in report["records"])


def test_injected_provider_path_is_network_free_and_records_contract():
    providers = build_offline_providers(default_cases())
    report = run_benchmark(mode="injected", providers=providers, provider_names=("openai",))

    assert report["status"] == "ok"
    assert len(report["records"]) == len(default_cases())
    assert report["provider_config"]["openai"]["mode"] == "injected"
    assert report["summary"][0]["answered"] == 2
    assert report["summary"][0]["abstained"] == 1


def test_real_openai_missing_credentials_is_fail_closed(tmp_path: Path):
    output = tmp_path / "blocked.json"
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("# intentionally empty\n", encoding="utf-8")
    exit_code = main(
        [
            "--mode",
            "real",
            "--provider",
            "openai",
            "--env-file",
            str(empty_env),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "requires VLM_BASE_URL" in report["blocked_reason"]
    assert "api_key" not in json.dumps(report)


def test_offline_cli_smoke_writes_report(tmp_path: Path):
    output = tmp_path / "smoke.json"
    assert main(["--offline-smoke", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["case_count"] == 3
    assert report["summary"]
