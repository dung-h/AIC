from __future__ import annotations

from src.cli import competition_ready


def test_windows_mounted_venv_skips_expensive_dependency_probe(monkeypatch):
    """An invalid WSL mount must fail before importing heavyweight packages."""
    monkeypatch.setattr(competition_ready.sys, "prefix", "/mnt/c/hcmai/.venv")
    monkeypatch.setattr(competition_ready.sys, "exec_prefix", "/mnt/c/hcmai/.venv")

    def forbidden_probe(_name: str):
        raise AssertionError("dependency probe must not run for a /mnt virtualenv")

    monkeypatch.setattr(competition_ready.importlib.util, "find_spec", forbidden_probe)
    competition_ready._IMPORT_PROBE_CACHE.clear()
    builder = competition_ready.ReportBuilder()

    competition_ready._check_python(
        builder,
        competition_ready.PreflightConfig(),
    )

    check = builder.checks[0]
    assert check["status"] == "blocker"
    assert check["details"]["windows_mounted"] is True
    assert check["details"]["import_probe"]["error"] == "invalid_linux_virtualenv"
