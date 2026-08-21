from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
import sys


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "export_remote_env.py"
SPEC = importlib.util.spec_from_file_location("export_remote_env", MODULE_PATH)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def test_export_openai_profile_copies_only_vlm_secret(tmp_path: Path):
    source = tmp_path / ".env"
    source.write_text(
        "VLM_BASE_URL=https://lightning.example/v1\n"
        "VLM_API_KEY=private-vlm-value\n"
        "VLM_MODEL=openai/gpt-4o\n"
        "DEEPGRAM_API_KEY=private-deepgram-value\n",
        encoding="utf-8",
    )
    output = tmp_path / "remote.env"

    path, names = exporter.export_remote_env(source, output, answer_provider="openai")

    rendered = path.read_text(encoding="utf-8")
    assert "VQA_ANSWER_PROVIDER=openai" in rendered
    assert "VLM_API_KEY=private-vlm-value" in rendered
    assert "DEEPGRAM_API_KEY" not in rendered
    assert "HCMAI_ACTIVE_VIDEO_PREFIXES=L" in rendered
    assert "VLM_API_KEY" in names
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_export_requires_complete_vlm_profile(tmp_path: Path):
    source = tmp_path / ".env"
    source.write_text("VLM_API_KEY=private-vlm-value\n", encoding="utf-8")

    try:
        exporter.export_remote_env(source, tmp_path / "remote.env", answer_provider="openai")
    except exporter.RemoteEnvError as exc:
        assert "VLM_BASE_URL" in str(exc)
    else:  # pragma: no cover - explicit negative assertion
        raise AssertionError("incomplete VLM profile was accepted")
