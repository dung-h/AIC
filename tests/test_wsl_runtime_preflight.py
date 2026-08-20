from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preflight_wsl_runtime as preflight  # noqa: E402


def test_forbidden_qna_dependency_path_is_detected() -> None:
    assert preflight._is_forbidden_path("/mnt/c/HCMAI/.codex/qna_deps")
    assert preflight._is_forbidden_path(r"C:\HCMAI\.codex\qna_deps\torch")
    assert not preflight._is_forbidden_path("/mnt/c/HCMAI/.venv/lib/python3.12/site-packages")


def test_clean_environment_removes_python_injection(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/mnt/c/HCMAI/.codex/qna_deps")
    monkeypatch.setenv("PYTHONHOME", r"C:\Python312")
    monkeypatch.setenv("PYTHONUSERBASE", "/mnt/c/users/admin/.local")
    clean = preflight._clean_env()
    assert "PYTHONPATH" not in clean
    assert "PYTHONHOME" not in clean
    assert "PYTHONUSERBASE" not in clean
    assert clean["PYTHONNOUSERSITE"] == "1"
    assert clean["PYTHONSAFEPATH"] == "1"


def test_import_probe_fails_closed_on_timeout(monkeypatch) -> None:
    def timeout_probe(*args, **kwargs):
        assert kwargs["timeout"] == preflight.IMPORT_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(preflight.subprocess, "run", timeout_probe)
    result = preflight._run_import_check("transformers")
    assert result["ok"] is False
    assert "timeout" in result["error"]
