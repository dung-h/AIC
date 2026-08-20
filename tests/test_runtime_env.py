from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from src.runtime_policy import RuntimePolicy
from src.utils.paths import activate_runtime_env, load_env, load_runtime_env


ROOT = Path(__file__).resolve().parents[1]


def test_dotenv_parser_is_safe_and_preserves_standardized_values(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comments are ignored\n"
        "export VQA_MODALITY_ROUTING='1'\n"
        "INVALID-NAME=ignored\n"
        "VLM_BASE_URL=https://vision.example/v1\n"
        "VLM_API_KEY=vision-key\n"
        "VLM_MODEL=vision-model\n",
        encoding="utf-8",
    )

    values = load_env(dotenv)

    assert values["VQA_MODALITY_ROUTING"] == "1"
    assert values["VLM_BASE_URL"] == "https://vision.example/v1"
    assert values["VLM_API_KEY"] == "vision-key"
    assert values["VLM_MODEL"] == "vision-model"
    assert "INVALID-NAME" not in values


def test_explicit_environment_overrides_dotenv(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "VLM_BASE_URL=https://file.example/v1\n"
        "VQA_MODALITY_ROUTING=0\n",
        encoding="utf-8",
    )

    values = load_runtime_env(
        dotenv,
        {"VLM_BASE_URL": "https://export.example/v1", "VQA_MODALITY_ROUTING": "1"},
    )

    assert values["VLM_BASE_URL"] == "https://export.example/v1"
    assert values["VQA_MODALITY_ROUTING"] == "1"


def test_activation_never_overwrites_an_explicit_environment_value(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("HCMAI_TRAKE_MODE=asr\nVQA_MODALITY_ROUTING=1\n", encoding="utf-8")
    target = {"HCMAI_TRAKE_MODE": "visual"}

    values = activate_runtime_env(dotenv, target)

    assert target["HCMAI_TRAKE_MODE"] == "visual"
    assert target["VQA_MODALITY_ROUTING"] == "1"
    assert values["HCMAI_TRAKE_MODE"] == "visual"


def test_runtime_policy_reads_shared_dotenv_but_export_wins(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / "policy.env"
    dotenv.write_text(
        "HCMAI_TRAKE_MODE=asr\n"
        "VQA_MODALITY_ROUTING=1\n"
        "HCMAI_NETWORK_MODE=online\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HCMAI_DOTENV", str(dotenv))
    monkeypatch.setenv("HCMAI_TRAKE_MODE", "visual")
    monkeypatch.delenv("VQA_MODALITY_ROUTING", raising=False)

    policy = RuntimePolicy.from_env()

    assert policy.trake_mode == "visual"
    assert policy.vqa_modality_routing is True


def test_bash_wrapper_uses_safe_dotenv_parser_with_matching_precedence(tmp_path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is not installed on this host")
    dotenv = tmp_path / "runtime.env"
    marker = tmp_path / "must-not-exist"
    dotenv.write_text(
        "VLM_BASE_URL=https://file.example/v1\n"
        "VLM_MODEL=file-model\n"
        "VQA_MODALITY_ROUTING=1\n"
        f"UNSAFE=$(touch {marker})\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("VLM_BASE_URL", None)
    env.pop("VLM_MODEL", None)
    env.pop("VQA_MODALITY_ROUTING", None)
    env["VLM_BASE_URL"] = "https://export.example/v1"
    command = (
        f"source {shlex.quote(str(ROOT / 'scripts' / 'runtime_env.sh'))}; "
        f"hcmai_load_dotenv {shlex.quote(str(dotenv))}; "
        "printf '%s|%s|%s' \"$VLM_BASE_URL\" \"$VLM_MODEL\" \"$VQA_MODALITY_ROUTING\""
    )

    result = subprocess.run(
        [bash, "-lc", command], env=env, text=True, capture_output=True, check=True
    )

    assert result.stdout == "https://export.example/v1|file-model|1"
    assert not marker.exists()
