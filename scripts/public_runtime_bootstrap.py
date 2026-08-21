#!/usr/bin/env python3
"""Install public Google Drive runtime assets without OAuth or rclone.

This is a provisioning tool, never part of a competition query.  It downloads
three intentionally separate public folders through ``gdown``:

* ``data/index``;
* selected compressed keyframe archives, not the raw ``data/keyframes`` tree;
* selected model directories.

Separating the folders prevents an accidental download of a legacy raw
keyframe tree containing hundreds of thousands of images.  The preselection
default installs only the two L-series archives and the production bge-m3 plus
Qwen 7B models; K-series archives and Qwen 3B stay remote until explicitly
requested. The command never reads an rclone config or any credential.
``plan`` is local-only; ``fetch --yes`` is the only mutating/networked
operation.
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
ASSET_TYPES = ("index", "keyframes", "models")
ARCHIVES = {
    "keyframes-K01-K05.tar": ("K01_", "K02_", "K03_", "K04_", "K05_"),
    "keyframes-K06-K10.tar": ("K06_", "K07_", "K08_", "K09_", "K10_"),
    "keyframes-K11-K15.tar": ("K11_", "K12_", "K13_", "K14_", "K15_"),
    "keyframes-K16-K20.tar": ("K16_", "K17_", "K18_", "K19_", "K20_"),
    "keyframes-L21-L25.tar": ("L21_", "L22_", "L23_", "L24_", "L25_"),
    "keyframes-L26-L30.tar": ("L26_", "L27_", "L28_", "L29_", "L30_"),
}
DEFAULT_ARCHIVES = ("keyframes-L21-L25.tar", "keyframes-L26-L30.tar")


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
    archives: tuple[str, ...] = DEFAULT_ARCHIVES
    assets: tuple[str, ...] = ASSET_TYPES

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


def _existing_keyframe_groups(destination: Path, archive_names: Sequence[str]) -> dict[str, list[str]]:
    if not destination.is_dir():
        return {}
    existing: dict[str, list[str]] = {}
    for archive_name in archive_names:
        names = [entry.name for prefix in ARCHIVES[archive_name] for entry in destination.glob(f"{prefix}*")]
        if names:
            existing[archive_name] = sorted(names)
    return existing


def _assert_keyframe_destinations_available(destination: Path, archive_names: Sequence[str]) -> None:
    existing = _existing_keyframe_groups(destination, archive_names)
    if existing:
        archive_name, paths = next(iter(existing.items()))
        raise PublicBootstrapError(
            f"refusing to merge keyframes over unmanaged videos for {archive_name}: {paths[:5]}"
        )
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


def _install_keyframe_archives(
    payload: Path, destination: Path, archive_names: Sequence[str]
) -> dict[str, str]:
    for archive_name in archive_names:
        prefixes = ARCHIVES[archive_name]
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
    for archive_name in archive_names:
        prefixes = ARCHIVES[archive_name]
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


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    path = PurePosixPath(str(value or ""))
    if not str(value or "") or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise PublicBootstrapError(f"public {label} has an unsafe path: {value!r}")
    return path


def _list_public_folder(
    *,
    url: str,
    gdown_bin: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[tuple[PurePosixPath, str]]:
    """List public Drive files without downloading the whole folder."""
    command = [gdown_bin, "--folder", "--json", "--no-cookies", url]
    try:
        completed = runner(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise PublicBootstrapError(
            f"public bootstrap requires gdown; install requirements.txt first ({gdown_bin!r} missing)"
        ) from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "gdown failed").strip()[-500:]
        raise PublicBootstrapError(f"public folder listing failed: {detail}")
    try:
        raw_entries = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PublicBootstrapError("gdown did not return a JSON public-folder listing") from exc
    if not isinstance(raw_entries, list):
        raise PublicBootstrapError("gdown public-folder listing must be a JSON list")

    entries: list[tuple[PurePosixPath, str]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise PublicBootstrapError("gdown public-folder listing contains a non-object entry")
        path = _safe_relative_path(item.get("path"), label="folder listing")
        file_url = str(item.get("url") or "").strip()
        parsed = urlparse(file_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise PublicBootstrapError(f"gdown public-folder listing has an invalid file URL for {path}")
        entries.append((path, file_url))
    return entries


def _download_public_file(
    *,
    url: str,
    destination: Path,
    gdown_bin: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [gdown_bin, "--continue", "--no-cookies", "-O", str(destination), url]
    try:
        completed = runner(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise PublicBootstrapError(
            f"public bootstrap requires gdown; install requirements.txt first ({gdown_bin!r} missing)"
        ) from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "gdown failed").strip()[-500:]
        raise PublicBootstrapError(f"public file download failed for {destination.name}: {detail}")


def _selected_archive_entries(
    entries: Sequence[tuple[PurePosixPath, str]], archive_names: Sequence[str]
) -> list[tuple[PurePosixPath, str]]:
    selected: list[tuple[PurePosixPath, str]] = []
    for archive_name in archive_names:
        matches = [(path, url) for path, url in entries if path.name == archive_name]
        if len(matches) != 1:
            raise PublicBootstrapError(
                f"public keyframe folder has {len(matches)} matches for {archive_name}; expected exactly one"
            )
        # Drive's JSON path can include its parent folder; archive extraction
        # deliberately receives a flat staging directory with only the chosen
        # tar files.
        selected.append((PurePosixPath(archive_name), matches[0][1]))
    return selected


def _model_relative_path(path: PurePosixPath) -> PurePosixPath | None:
    for index, component in enumerate(path.parts):
        if component in MODEL_DIRS:
            return PurePosixPath(*path.parts[index:])
    return None


def _selected_model_entries(
    entries: Sequence[tuple[PurePosixPath, str]]
) -> list[tuple[PurePosixPath, str]]:
    selected: list[tuple[PurePosixPath, str]] = []
    seen: set[PurePosixPath] = set()
    for path, url in entries:
        relative = _model_relative_path(path)
        if relative is not None:
            if relative in seen:
                raise PublicBootstrapError(f"public model folder has duplicate file path: {relative}")
            seen.add(relative)
            selected.append((relative, url))
    missing = [name for name in MODEL_DIRS if not any(path.parts[0] == name for path, _ in selected)]
    if missing:
        raise PublicBootstrapError(f"public model folder is missing selected model files for: {', '.join(missing)}")
    return selected


def _download_selected_files(
    entries: Sequence[tuple[PurePosixPath, str]], *, destination: Path, gdown_bin: str
) -> None:
    for relative_path, url in entries:
        target = destination / relative_path
        # Keep the staging layout deterministic even when a downloader is
        # substituted in tests or by a deployment wrapper.
        target.parent.mkdir(parents=True, exist_ok=True)
        _download_public_file(url=url, destination=target, gdown_bin=gdown_bin)


def _validate_selected_models(payload: Path) -> None:
    for name in MODEL_DIRS:
        model_dir = payload / name
        if not (model_dir / "config.json").is_file():
            raise PublicBootstrapError(f"public model payload is missing {name}/config.json")
        if not any(path.is_file() and path.name != "config.json" for path in model_dir.rglob("*")):
            raise PublicBootstrapError(f"public model payload has no model files for {name}")


def _validate_selection(config: PublicConfig) -> None:
    unknown_archives = [name for name in config.archives if name not in ARCHIVES]
    if unknown_archives:
        raise PublicBootstrapError(f"unknown keyframe archive(s): {', '.join(unknown_archives)}")
    if not config.archives and "keyframes" in config.assets:
        raise PublicBootstrapError("at least one --archive is required when installing keyframes")
    if len(set(config.archives)) != len(config.archives):
        raise PublicBootstrapError("duplicate keyframe archive selected")
    unknown_assets = [name for name in config.assets if name not in ASSET_TYPES]
    if unknown_assets:
        raise PublicBootstrapError(f"unknown public asset type(s): {', '.join(unknown_assets)}")
    if not config.assets or len(set(config.assets)) != len(config.assets):
        raise PublicBootstrapError("select one or more distinct public asset types")


def plan(config: PublicConfig) -> dict[str, object]:
    """Return a local-only deployment plan without contacting Google Drive."""
    _validate_selection(config)
    blocked_targets: list[str] = []
    if "index" in config.assets and config.index_target.exists() and (
        not config.index_target.is_dir() or any(config.index_target.iterdir())
    ):
        blocked_targets.append(str(config.index_target))
    if "models" in config.assets and config.model_root.exists() and (
        not config.model_root.is_dir() or any(config.model_root.iterdir())
    ):
        blocked_targets.append(str(config.model_root))
    if "keyframes" in config.assets:
        blocked_targets.extend(
            str(config.keyframes_target / archive_name)
            for archive_name in _existing_keyframe_groups(config.keyframes_target, config.archives)
        )
    return {
        "schema": "hcmai.public_runtime_bootstrap.v2",
        "network_used": False,
        "downloads": {
            "index": "index" in config.assets,
            "keyframes": "keyframes" in config.assets,
            "models": "models" in config.assets,
        },
        "assets": list(config.assets),
        "keyframe_archives": list(config.archives) if "keyframes" in config.assets else [],
        "models": list(MODEL_DIRS) if "models" in config.assets else [],
        "targets": {
            "index": str(config.index_target),
            "keyframes": str(config.keyframes_target),
            "models": str(config.model_root),
        },
        "blocked_targets": blocked_targets,
    }


def fetch(config: PublicConfig) -> dict[str, object]:
    """Download public assets, validate their layout, then install without overwrite."""
    _validate_selection(config)
    url_fields = {
        "index": (config.index_url, "index URL"),
        "keyframes": (config.keyframes_url, "keyframes URL"),
        "models": (config.models_url, "models URL"),
    }
    for asset in config.assets:
        url, field = url_fields[asset]
        _public_folder_url(url, field=field)
    if "index" in config.assets:
        _assert_empty_destination(config.index_target, label="index target")
    if "keyframes" in config.assets:
        _assert_keyframe_destinations_available(config.keyframes_target, config.archives)
    if "models" in config.assets:
        _assert_empty_destination(config.model_root, label="model target")

    config.download_root.mkdir(parents=True, exist_ok=True)
    # Keep a deterministic staging directory across interrupted public Drive
    # transfers: gdown's --continue can then resume large archives. It is
    # removed only after all downloads validate and installation succeeds.
    staging = config.download_root / "public-runtime-v1"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        index_payload: Path | None = None
        if "index" in config.assets:
            _download_folder(url=config.index_url, destination=staging / "index", gdown_bin=config.gdown_bin)
            index_payload = _find_payload(staging / "index", INDEX_FILES, kind="index", files=True)

        keyframe_payload = staging / "keyframes"
        if "keyframes" in config.assets:
            archive_entries = _selected_archive_entries(
                _list_public_folder(url=config.keyframes_url, gdown_bin=config.gdown_bin), config.archives
            )
            _download_selected_files(archive_entries, destination=keyframe_payload, gdown_bin=config.gdown_bin)

        model_payload = staging / "models"
        if "models" in config.assets:
            model_entries = _selected_model_entries(
                _list_public_folder(url=config.models_url, gdown_bin=config.gdown_bin)
            )
            _download_selected_files(model_entries, destination=model_payload, gdown_bin=config.gdown_bin)
            _validate_selected_models(model_payload)

        # Moving complete directory trees avoids a second full copy on a large
        # server volume. Keyframes are only extracted from explicitly selected
        # archives, so a later K-series installation can be keyframes-only.
        if index_payload is not None:
            shutil.move(str(index_payload), str(config.index_target))
        if "models" in config.assets:
            shutil.move(str(model_payload), str(config.model_root))
        archive_hashes = (
            _install_keyframe_archives(keyframe_payload, config.keyframes_target, config.archives)
            if "keyframes" in config.assets else {}
        )
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
        "schema": "hcmai.public_runtime_bootstrap.v2",
        "network_used": True,
        "assets": list(config.assets),
        "models": list(MODEL_DIRS) if "models" in config.assets else [],
        "keyframe_archives": list(config.archives) if "keyframes" in config.assets else [],
        "targets": {
            "index": str(config.index_target),
            "keyframes": str(config.keyframes_target),
            "models": str(config.model_root),
        },
        "keyframe_archive_sha256": archive_hashes,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"installed": list(config.assets), "receipt": str(receipt_path)}


def _config_from_args(args: argparse.Namespace) -> PublicConfig:
    env_archives = tuple(filter(None, (item.strip() for item in os.getenv("HCMAI_PUBLIC_KEYFRAME_ARCHIVES", "").split(","))))
    archives = tuple(args.archive) if args.archive else (env_archives or DEFAULT_ARCHIVES)
    assets = tuple(args.asset) if args.asset else ASSET_TYPES
    return PublicConfig(
        index_url=args.index_url or os.getenv("HCMAI_PUBLIC_INDEX_URL", ""),
        keyframes_url=args.keyframes_url or os.getenv("HCMAI_PUBLIC_KEYFRAMES_URL", ""),
        models_url=args.models_url or os.getenv("HCMAI_PUBLIC_MODELS_URL", ""),
        project_root=Path(args.project_root).resolve(),
        model_root=Path(args.model_root).resolve(),
        download_root=Path(args.download_root).resolve(),
        gdown_bin=args.gdown_bin,
        archives=archives,
        assets=assets,
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
        command.add_argument("--asset", choices=ASSET_TYPES, action="append", default=None,
                             help="asset to install; repeat to select multiple (default: all)")
        command.add_argument("--archive", choices=tuple(ARCHIVES), action="append", default=None,
                             help="keyframe archive to install; repeat to add packs (default: L21-L30)")
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
