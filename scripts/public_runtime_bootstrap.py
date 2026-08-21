#!/usr/bin/env python3
"""Install public Google Drive runtime assets without OAuth or rclone.

This is a provisioning tool, never part of a competition query.  It downloads
three intentionally separate public folders through ``gdown``:

* ``data/index``;
* the six compressed keyframe archives, not the raw ``data/keyframes`` tree;
* the model directories.

Separating the folders prevents an accidental download of a legacy raw
keyframe tree containing hundreds of thousands of images.  The command never
reads an rclone config or any credential.  ``plan`` is local-only; ``fetch
--yes`` is the only mutating/networked operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from typing import Callable, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
INDEX_FILES = (
    "global_keyframes.parquet",
    "global_siglip_vitl.npy",
    "global_keyframes_vitl.parquet",
    "global_so400m384.npy",
    "global_keyframes_so400m384.parquet",
)
MODEL_DIRS = ("bge-m3", "Qwen2.5-VL-7B-Instruct")
ARCHIVES = {
    "keyframes-K01-K05.tar": ("K01_", "K02_", "K03_", "K04_", "K05_"),
    "keyframes-K06-K10.tar": ("K06_", "K07_", "K08_", "K09_", "K10_"),
    "keyframes-K11-K15.tar": ("K11_", "K12_", "K13_", "K14_", "K15_"),
    "keyframes-K16-K20.tar": ("K16_", "K17_", "K18_", "K19_", "K20_"),
    "keyframes-L21-L25.tar": ("L21_", "L22_", "L23_", "L24_", "L25_"),
    "keyframes-L26-L30.tar": ("L26_", "L27_", "L28_", "L29_", "L30_"),
}


class PublicBootstrapError(RuntimeError):
    """A public asset cannot be downloaded or installed safely."""


@dataclass(frozen=True)
class PublicConfig:
    index_url: str
    keyframes_url: str
    models_url: str
    project_root: Path
    model_root: Path
    download_root: Path
    gdown_bin: str

    @property
    def index_target(self) -> Path:
        return self.project_root / "data" / "index"

    @property
    def keyframes_target(self) -> Path:
        return self.project_root / "data" / "keyframes"


def _public_folder_url(value: str, *, field: str) -> str:
    value = str(value or "").strip()
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.netloc.casefold() != "drive.google.com":
        raise PublicBootstrapError(f"{field} must be an https://drive.google.com folder URL")
    if "/folders/" not in parsed.path:
        raise PublicBootstrapError(f"{field} must point to a Google Drive folder, not a file")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_payload(root: Path, required: Sequence[str], *, kind: str, files: bool) -> Path:
    candidates = [root, *sorted((path for path in root.rglob("*") if path.is_dir()), key=str)]
    matches = [
        candidate for candidate in candidates
        if all((candidate / name).is_file() if files else (candidate / name).is_dir() for name in required)
    ]
    if len(matches) != 1:
        raise PublicBootstrapError(
            f"public {kind} download has {len(matches)} matching payload roots; expected exactly one"
        )
    return matches[0]


def _assert_empty_destination(destination: Path, *, label: str) -> None:
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        else:
            raise PublicBootstrapError(f"refusing to overwrite existing {label}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)


def _safe_archive_members(archive: tarfile.TarFile, prefixes: tuple[str, ...]) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise PublicBootstrapError(f"keyframe archive is empty: {archive.name}")
    valid_top_level = False
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            raise PublicBootstrapError(f"unsafe keyframe archive member: {member.name!r}")
        if member.issym() or member.islnk() or member.isdev():
            raise PublicBootstrapError(f"unsupported link/device in keyframe archive: {member.name!r}")
        if not (member.isdir() or member.isfile()):
            raise PublicBootstrapError(f"unsupported member type in keyframe archive: {member.name!r}")
        if path.parts and any(path.parts[0].startswith(prefix) for prefix in prefixes):
            valid_top_level = True
    if not valid_top_level:
        raise PublicBootstrapError(f"keyframe archive has no expected video directories: {archive.name}")
    return members


def _install_keyframe_archives(payload: Path, destination: Path) -> dict[str, str]:
    for archive_name, prefixes in ARCHIVES.items():
        archive_path = payload / archive_name
        if not archive_path.is_file():
            raise PublicBootstrapError(f"public keyframe payload is missing {archive_name}")
        existing = [
            entry.name for prefix in prefixes for entry in destination.glob(f"{prefix}*")
        ] if destination.is_dir() else []
        if existing:
            raise PublicBootstrapError(
                f"refusing to merge keyframes over unmanaged videos for {archive_name}: {sorted(existing)[:5]}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for archive_name, prefixes in ARCHIVES.items():
        archive_path = payload / archive_name
        with tarfile.open(archive_path, "r:*") as archive:
            members = _safe_archive_members(archive, prefixes)
            archive.extractall(destination, members=members, filter="data")
        hashes[archive_name] = _sha256(archive_path)
    return hashes


def _download_folder(
    *,
    url: str,
    destination: Path,
    gdown_bin: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [gdown_bin, "--folder", "--continue", "--no-cookies", "-O", str(destination), url]
    try:
        completed = runner(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise PublicBootstrapError(
            f"public bootstrap requires gdown; install requirements.txt first ({gdown_bin!r} missing)"
        ) from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "gdown failed").strip()[-500:]
        raise PublicBootstrapError(f"public folder download failed: {detail}")


def plan(config: PublicConfig) -> dict[str, object]:
    """Return a local-only deployment plan without contacting Google Drive."""
    return {
        "schema": "hcmai.public_runtime_bootstrap.v1",
        "network_used": False,
        "downloads": {
            "index": bool(config.index_url),
            "keyframes": bool(config.keyframes_url),
            "models": bool(config.models_url),
        },
        "targets": {
            "index": str(config.index_target),
            "keyframes": str(config.keyframes_target),
            "models": str(config.model_root),
        },
        "blocked_targets": [str(path) for path in (
            config.index_target, config.keyframes_target, config.model_root,
        ) if path.exists() and (not path.is_dir() or any(path.iterdir()))],
    }


def fetch(config: PublicConfig) -> dict[str, object]:
    """Download public assets, validate their layout, then install without overwrite."""
    for url, field in (
        (config.index_url, "index URL"),
        (config.keyframes_url, "keyframes URL"),
        (config.models_url, "models URL"),
    ):
        _public_folder_url(url, field=field)
    _assert_empty_destination(config.index_target, label="index target")
    _assert_empty_destination(config.keyframes_target, label="keyframes target")
    _assert_empty_destination(config.model_root, label="model target")

    config.download_root.mkdir(parents=True, exist_ok=True)
    # Keep a deterministic staging directory across interrupted public Drive
    # transfers: gdown's --continue can then resume large archives. It is
    # removed only after all downloads validate and installation succeeds.
    staging = config.download_root / "public-runtime-v1"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        _download_folder(url=config.index_url, destination=staging / "index", gdown_bin=config.gdown_bin)
        _download_folder(url=config.keyframes_url, destination=staging / "keyframes", gdown_bin=config.gdown_bin)
        _download_folder(url=config.models_url, destination=staging / "models", gdown_bin=config.gdown_bin)

        index_payload = _find_payload(staging / "index", INDEX_FILES, kind="index", files=True)
        keyframes_payload = _find_payload(
            staging / "keyframes", tuple(ARCHIVES), kind="keyframe archives", files=True
        )
        models_payload = _find_payload(staging / "models", MODEL_DIRS, kind="models", files=False)
        # Moving the two directory trees avoids a second full copy on a large
        # server volume. Keyframes are safely extracted from the six archives.
        shutil.move(str(index_payload), str(config.index_target))
        shutil.move(str(models_payload), str(config.model_root))
        archive_hashes = _install_keyframe_archives(keyframes_payload, config.keyframes_target)
    except Exception:
        # Preserve only non-secret public artifacts for a future --continue.
        # Targets have not been touched until all three payload roots validate.
        raise
    else:
        shutil.rmtree(staging)

    receipt_dir = config.project_root / ".runtime_state"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "public_runtime_bootstrap.json"
    receipt_path.write_text(json.dumps({
        "schema": "hcmai.public_runtime_bootstrap.v1",
        "network_used": True,
        "targets": {
            "index": str(config.index_target),
            "keyframes": str(config.keyframes_target),
            "models": str(config.model_root),
        },
        "keyframe_archive_sha256": archive_hashes,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"installed": ["index", "keyframes", "models"], "receipt": str(receipt_path)}


def _config_from_args(args: argparse.Namespace) -> PublicConfig:
    return PublicConfig(
        index_url=args.index_url or os.getenv("HCMAI_PUBLIC_INDEX_URL", ""),
        keyframes_url=args.keyframes_url or os.getenv("HCMAI_PUBLIC_KEYFRAMES_URL", ""),
        models_url=args.models_url or os.getenv("HCMAI_PUBLIC_MODELS_URL", ""),
        project_root=Path(args.project_root).resolve(),
        model_root=Path(args.model_root).resolve(),
        download_root=Path(args.download_root).resolve(),
        gdown_bin=args.gdown_bin,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fetch"):
        command = subparsers.add_parser(name)
        command.add_argument("--index-url", default="")
        command.add_argument("--keyframes-url", default="")
        command.add_argument("--models-url", default="")
        command.add_argument("--project-root", default=str(ROOT))
        command.add_argument("--model-root", default=os.getenv("HCMAI_PUBLIC_MODEL_ROOT", "/opt/hcmai-models"))
        command.add_argument("--download-root", default=os.getenv("HCMAI_PUBLIC_DOWNLOAD_ROOT", str(ROOT / ".runtime-downloads")))
        command.add_argument("--gdown-bin", default=os.getenv("HCMAI_GDOWN_BIN", "gdown"))
    subparsers.choices["fetch"].add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    try:
        if args.command == "plan":
            report = plan(config)
        else:
            if not args.yes:
                raise PublicBootstrapError("fetch changes local storage; repeat with --yes")
            report = fetch(config)
    except PublicBootstrapError as exc:
        print(f"public runtime bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
