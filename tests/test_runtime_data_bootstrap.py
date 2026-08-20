from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "runtime_data_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("runtime_data_bootstrap", MODULE_PATH)
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


def _tar(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _manifest(path: Path, *, checksum: str | None = None) -> None:
    payload = {
        "schema": bootstrap.SCHEMA,
        "remote_root": "fake:runtime",
        "artifacts": [{
            "id": "keyframes-k01",
            "kind": "tar",
            "remote_path": "data/keyframes-K01.tar",
            "target": "data/keyframes",
            "expected_size_bytes": None,
            "checksum": {
                "algorithm": "md5",
                "value": checksum,
                "source": "rclone_remote_metadata",
                "required": True,
            },
            "owned_prefixes": ["K01_"],
            "extract": {"strip_components": 0},
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tree_manifest(path: Path) -> None:
    payload = {
        "schema": bootstrap.SCHEMA,
        "remote_root": "fake:runtime",
        "artifacts": [{
            "id": "runtime-index",
            "kind": "tree",
            "remote_path": "data/index",
            "target": "data/index",
            "expected_size_bytes": None,
            "checksum": {
                "algorithm": "md5",
                "value": None,
                "source": "rclone_remote_metadata",
                "required": True,
            },
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mock_rclone(archive: Path, *, include_md5: bool = True):
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    calls: list[list[str]] = []

    def runner(args, **kwargs):
        calls.append(list(args))
        if "lsjson" in args:
            hashes = {"MD5": digest} if include_md5 else {}
            return subprocess.CompletedProcess(args, 0, json.dumps([{
                "Name": "keyframes-K01.tar", "Path": "keyframes-K01.tar",
                "Size": archive.stat().st_size, "IsDir": False, "Hashes": hashes,
            }]), "")
        if "copyto" in args:
            Path(args[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[-1]).write_bytes(archive.read_bytes())
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"Unexpected rclone command: {args}")

    return runner, calls


def _mock_rclone_tree(
    remote_files: dict[str, bytes],
    *,
    copied_files: dict[str, bytes] | None = None,
    include_md5: bool = True,
):
    calls: list[list[str]] = []
    copied_files = remote_files if copied_files is None else copied_files

    def runner(args, **kwargs):
        calls.append(list(args))
        if "lsjson" in args:
            entries = []
            for relative, content in remote_files.items():
                entries.append({
                    "Path": relative,
                    "Name": Path(relative).name,
                    "Size": len(content),
                    "IsDir": False,
                    "Hashes": {"MD5": hashlib.md5(content).hexdigest()} if include_md5 else {},
                })
            return subprocess.CompletedProcess(args, 0, json.dumps(entries), "")
        if "copy" in args:
            target = Path(args[-1])
            for relative, content in copied_files.items():
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(content)
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"Unexpected rclone command: {args}")

    return runner, calls


def test_plan_is_local_only_and_never_calls_rclone(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)

    report = bootstrap.plan(manifest)

    assert report["network_used"] is False
    assert report["missing"] == ["keyframes-k01"]
    assert report["artifacts"][0]["target"] == str(tmp_path / "data" / "keyframes")


def test_fetch_uses_remote_md5_verifies_then_extracts_without_overwrite(tmp_path: Path):
    archive = tmp_path / "keyframes.tar"
    _tar(archive, {"K01_V001/000001.jpg": b"jpeg"})
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    runner, calls = _mock_rclone(archive)

    report = bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    assert report["installed"] == ["keyframes-k01"]
    assert (tmp_path / "data/keyframes/K01_V001/000001.jpg").read_bytes() == b"jpeg"
    receipt = tmp_path / "data/keyframes/.runtime-bootstrap/keyframes-k01.receipt.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["checksum"]["algorithm"] == "md5"
    assert any("lsjson" in command for command in calls)
    assert any("copyto" in command and "--partial" in command for command in calls)
    assert bootstrap.plan(manifest)["missing"] == []


def test_fetch_fails_closed_when_drive_metadata_has_no_required_checksum(tmp_path: Path):
    archive = tmp_path / "keyframes.tar"
    _tar(archive, {"K01_V001/000001.jpg": b"jpeg"})
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    runner, calls = _mock_rclone(archive, include_md5=False)

    with pytest.raises(bootstrap.BootstrapError, match="checksum md5 is unavailable"):
        bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    assert not any("copyto" in command for command in calls)
    assert not (tmp_path / "data/keyframes").exists()


def test_tar_path_traversal_is_rejected_before_any_target_write(tmp_path: Path):
    archive = tmp_path / "unsafe.tar"
    _tar(archive, {"../outside.txt": b"unsafe"})
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    runner, calls = _mock_rclone(archive)

    with pytest.raises(bootstrap.BootstrapError, match="Unsafe archive member path"):
        bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "data/keyframes").exists()


def test_fetch_refuses_to_overwrite_existing_target_file(tmp_path: Path):
    archive = tmp_path / "keyframes.tar"
    _tar(archive, {"K01_V001/000001.jpg": b"replacement"})
    destination = tmp_path / "data/keyframes/K01_V001/000001.jpg"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"keep")
    manifest_path = tmp_path / "manifest.json"
    _manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    runner, calls = _mock_rclone(archive)

    with pytest.raises(bootstrap.BootstrapError, match="Target contains unmanaged paths"):
        bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    assert destination.read_bytes() == b"keep"
    assert calls == []


def test_runtime_index_tree_enumerates_then_verifies_every_file_and_promotes(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    _tree_manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    remote_files = {
        "asr_global_v2/embeddings.npy": b"asr-index",
        "modality_global_v2/ocr/retrieval.parquet": b"ocr-index",
    }
    runner, calls = _mock_rclone_tree(remote_files)

    report = bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    target = tmp_path / "data/index"
    assert report["installed"] == ["runtime-index"]
    assert (target / "asr_global_v2/embeddings.npy").read_bytes() == b"asr-index"
    assert (target / "modality_global_v2/ocr/retrieval.parquet").read_bytes() == b"ocr-index"
    receipt = json.loads((tmp_path / "data/.runtime-bootstrap/runtime-index.receipt.json").read_text())
    assert receipt["files"] == [
        {"md5": hashlib.md5(b"asr-index").hexdigest(), "path": "asr_global_v2/embeddings.npy", "size": 9},
        {"md5": hashlib.md5(b"ocr-index").hexdigest(), "path": "modality_global_v2/ocr/retrieval.parquet", "size": 9},
    ]
    assert any("lsjson" in command and "--recursive" in command for command in calls)
    assert any("copy" in command and "--partial" in command for command in calls)
    assert bootstrap.plan(manifest)["missing"] == []


def test_runtime_index_tree_rejects_missing_or_extra_staged_files_without_promotion(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    _tree_manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    runner, _ = _mock_rclone_tree(
        {"asr/a.bin": b"a", "ocr/b.bin": b"b"},
        copied_files={"asr/a.bin": b"a", "unexpected.bin": b"x"},
    )

    with pytest.raises(bootstrap.BootstrapError, match="does not exactly match enumerated remote manifest"):
        bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    assert not (tmp_path / "data/index").exists()


def test_runtime_index_tree_fails_closed_before_copy_when_any_md5_is_missing(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    _tree_manifest(manifest_path)
    manifest = bootstrap.load_manifest(manifest_path, tmp_path)
    runner, calls = _mock_rclone_tree({"asr/a.bin": b"a"}, include_md5=False)

    with pytest.raises(bootstrap.BootstrapError, match="has no MD5 checksum"):
        bootstrap.fetch(manifest, tmp_path, selected=None, rclone_bin="rclone", runner=runner)

    assert not any("copy" in command for command in calls)
    assert not (tmp_path / "data/index").exists()


def test_production_manifest_declares_runtime_index_tree():
    manifest_path = Path(__file__).parents[1] / "configs/runtime_artifacts.v1.json"
    manifest = bootstrap.load_manifest(manifest_path, Path(__file__).parents[1])
    runtime_index = next(item for item in manifest.artifacts if item.artifact_id == "runtime-index")

    assert runtime_index.kind == "tree"
    assert runtime_index.remote_path == "data/index"
    assert runtime_index.target == Path(__file__).parents[1] / "data/index"


def test_tree_receipt_rejects_same_size_local_corruption(tmp_path: Path):
    target = tmp_path / "data/index"
    target.mkdir(parents=True)
    payload = target / "global.npy"
    payload.write_bytes(b"good")
    receipt = {
        "files": [{
            "path": "global.npy", "size": 4,
            "md5": hashlib.md5(b"good").hexdigest(),
        }],
    }

    assert bootstrap._tree_matches_receipt(target, receipt)
    payload.write_bytes(b"evil")
    assert not bootstrap._tree_matches_receipt(target, receipt)
