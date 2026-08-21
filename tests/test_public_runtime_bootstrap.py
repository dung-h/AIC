from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
import tarfile


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "public_runtime_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("public_runtime_bootstrap", MODULE_PATH)
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


def _archive(path: Path, video_id: str) -> None:
    with tarfile.open(path, "w") as archive:
        content = b"frame"
        info = tarfile.TarInfo(f"{video_id}/001.jpg")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def _config(tmp_path: Path) -> bootstrap.PublicConfig:
    return bootstrap.PublicConfig(
        index_url="https://drive.google.com/drive/folders/index",
        keyframes_url="https://drive.google.com/drive/folders/keyframes",
        models_url="https://drive.google.com/drive/folders/models",
        project_root=tmp_path / "repo",
        model_root=tmp_path / "models",
        download_root=tmp_path / "downloads",
        gdown_bin="gdown",
    )


def test_plan_is_local_only(tmp_path: Path):
    report = bootstrap.plan(_config(tmp_path))

    assert report["network_used"] is False
    assert report["downloads"] == {"index": True, "keyframes": True, "models": True}


def test_fetch_installs_public_payloads_without_rclone(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)

    def fake_download(*, url: str, destination: Path, gdown_bin: str) -> None:
        if url == config.index_url:
            payload = destination / "index"
            payload.mkdir(parents=True)
            for name in bootstrap.INDEX_FILES:
                (payload / name).write_bytes(b"index")
        elif url == config.keyframes_url:
            payload = destination / "archives"
            payload.mkdir(parents=True)
            for name, prefixes in bootstrap.ARCHIVES.items():
                _archive(payload / name, prefixes[0] + "V001")
        elif url == config.models_url:
            payload = destination / "models"
            for name in bootstrap.MODEL_DIRS:
                (payload / name).mkdir(parents=True)
                (payload / name / "config.json").write_text("{}")
        else:  # pragma: no cover - guards the fixture itself
            raise AssertionError(url)

    monkeypatch.setattr(bootstrap, "_download_folder", fake_download)
    report = bootstrap.fetch(config)

    assert report["installed"] == ["index", "keyframes", "models"]
    assert (config.index_target / "global_keyframes.parquet").is_file()
    assert (config.keyframes_target / "K01_V001/001.jpg").read_bytes() == b"frame"
    assert (config.model_root / "bge-m3/config.json").is_file()
    assert Path(report["receipt"]).is_file()
    assert not (config.download_root / "public-runtime-v1").exists()


def test_public_folder_url_rejects_non_drive_file_links():
    try:
        bootstrap._public_folder_url("https://example.com/file.zip", field="index")
    except bootstrap.PublicBootstrapError as exc:
        assert "drive.google.com" in str(exc)
    else:  # pragma: no cover - explicit negative assertion
        raise AssertionError("non-Drive URL was accepted")
