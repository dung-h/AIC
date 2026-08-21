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
    assert report["keyframe_archives"] == list(bootstrap.DEFAULT_ARCHIVES)
    assert report["models"] == list(bootstrap.MODEL_DIRS)


def test_fetch_installs_public_payloads_without_rclone(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    downloaded_urls: list[str] = []

    def fake_download(*, url: str, destination: Path, gdown_bin: str) -> None:
        if url == config.index_url:
            payload = destination / "index"
            payload.mkdir(parents=True)
            for name in bootstrap.INDEX_FILES:
                (payload / name).write_bytes(b"index")
        else:  # pragma: no cover - guards the fixture itself
            raise AssertionError(url)

    def fake_list(*, url: str, gdown_bin: str):
        if url == config.keyframes_url:
            return [
                (bootstrap.PurePosixPath("keyframe_archives_v2") / name, f"https://files.example/{name}")
                for name in bootstrap.ARCHIVES
            ]
        if url == config.models_url:
            entries = []
            for name in (*bootstrap.MODEL_DIRS, "Qwen2.5-VL-3B-Instruct"):
                entries.extend((
                    (bootstrap.PurePosixPath("models") / name / "config.json", f"https://files.example/{name}/config"),
                    (bootstrap.PurePosixPath("models") / name / "weights.safetensors", f"https://files.example/{name}/weights"),
                ))
            return entries
        raise AssertionError(url)  # pragma: no cover - guards the fixture itself

    def fake_download_file(*, url: str, destination: Path, gdown_bin: str) -> None:
        downloaded_urls.append(url)
        if destination.suffix == ".tar":
            _archive(destination, bootstrap.ARCHIVES[destination.name][0] + "V001")
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}" if destination.name == "config.json" else "weights")

    monkeypatch.setattr(bootstrap, "_download_folder", fake_download)
    monkeypatch.setattr(bootstrap, "_list_public_folder", fake_list)
    monkeypatch.setattr(bootstrap, "_download_public_file", fake_download_file)
    report = bootstrap.fetch(config)

    assert report["installed"] == ["index", "keyframes", "models"]
    assert (config.index_target / "global_keyframes.parquet").is_file()
    assert (config.keyframes_target / "L21_V001/001.jpg").read_bytes() == b"frame"
    assert not (config.keyframes_target / "K01_V001").exists()
    assert (config.model_root / "bge-m3/config.json").is_file()
    assert not (config.model_root / "Qwen2.5-VL-3B-Instruct").exists()
    assert not any("Qwen2.5-VL-3B-Instruct" in url for url in downloaded_urls)
    assert not any("keyframes-K" in url for url in downloaded_urls)
    assert Path(report["receipt"]).is_file()
    assert not (config.download_root / "public-runtime-v1").exists()


def test_keyframes_only_can_add_one_pack_without_reinstalling_index_or_models(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    config = bootstrap.PublicConfig(
        **{**config.__dict__, "assets": ("keyframes",), "archives": ("keyframes-K01-K05.tar",)}
    )
    (config.keyframes_target / "L21_V001").mkdir(parents=True)
    called: list[str] = []

    def fake_list(*, url: str, gdown_bin: str):
        assert url == config.keyframes_url
        return [(bootstrap.PurePosixPath("folder/keyframes-K01-K05.tar"), "https://files.example/K01")]

    def fake_download_file(*, url: str, destination: Path, gdown_bin: str) -> None:
        called.append(url)
        _archive(destination, "K01_V001")

    monkeypatch.setattr(bootstrap, "_list_public_folder", fake_list)
    monkeypatch.setattr(bootstrap, "_download_public_file", fake_download_file)
    report = bootstrap.fetch(config)

    assert report["installed"] == ["keyframes"]
    assert called == ["https://files.example/K01"]
    assert (config.keyframes_target / "L21_V001").is_dir()
    assert (config.keyframes_target / "K01_V001/001.jpg").is_file()


def test_public_folder_url_rejects_non_drive_file_links():
    try:
        bootstrap._public_folder_url("https://example.com/file.zip", field="index")
    except bootstrap.PublicBootstrapError as exc:
        assert "drive.google.com" in str(exc)
    else:  # pragma: no cover - explicit negative assertion
        raise AssertionError("non-Drive URL was accepted")
