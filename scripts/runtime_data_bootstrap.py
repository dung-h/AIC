#!/usr/bin/env python3
"""Safely bootstrap published HCMAI runtime artifacts from Google Drive.

This tool is deliberately opt-in. ``plan`` is a local-only inspection and
never invokes rclone or the network. ``fetch --yes`` is the only command that
can download or extract an artifact. Archives are verified before extraction,
checked for unsafe tar members, and merged without overwriting existing files.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


SCHEMA = "hcmai.runtime_artifacts.v1"
DEFAULT_REMOTE_ROOT = "gdrive:HCMAI-2026/runtime"
DEFAULT_MANIFEST = Path("configs/runtime_artifacts.v1.json")
STATE_DIR_NAME = ".runtime-bootstrap"
DOWNLOAD_DIR_NAME = "downloads"


class BootstrapError(RuntimeError):
    """An intentionally fail-closed bootstrap error."""


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str
    remote_path: str
    target: Path
    expected_size_bytes: int | None
    checksum_algorithm: str | None
    checksum_value: str | None
    checksum_remote_required: bool
    strip_components: int
    owned_prefixes: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    path: Path
    remote_root: str
    artifacts: tuple[Artifact, ...]


Run = Callable[..., subprocess.CompletedProcess[str]]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_relative(value: str, *, field: str) -> Path:
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise BootstrapError(f"Unsafe {field} in runtime manifest: {value!r}")
    return candidate


def _project_path(project_root: Path, relative: str, *, field: str) -> Path:
    root = project_root.resolve()
    path = (root / _safe_relative(relative, field=field)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BootstrapError(f"{field} escapes project root: {relative!r}") from exc
    return path


def _remote_file(remote_root: str, remote_path: str) -> str:
    if not remote_root or remote_root.rstrip(":/") == "":
        raise BootstrapError("Remote root is empty. Set --remote-root or HCMAI_RUNTIME_REMOTE.")
    relative = _safe_relative(remote_path, field="remote_path").as_posix()
    return f"{remote_root.rstrip('/')}/{relative}"


def load_manifest(path: Path, project_root: Path, remote_override: str | None = None) -> Manifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BootstrapError(f"Runtime manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BootstrapError(f"Runtime manifest is invalid JSON: {path}: {exc}") from exc
    if raw.get("schema") != SCHEMA:
        raise BootstrapError(f"Unsupported runtime manifest schema: {raw.get('schema')!r}")
    remote_root = remote_override or os.environ.get("HCMAI_RUNTIME_REMOTE") or raw.get("remote_root") or DEFAULT_REMOTE_ROOT
    artifacts: list[Artifact] = []
    seen_ids: set[str] = set()
    for item in raw.get("artifacts", []):
        artifact_id = str(item.get("id", ""))
        kind = str(item.get("kind", ""))
        if not artifact_id or artifact_id in seen_ids or kind not in {"file", "tar", "tree"}:
            raise BootstrapError(f"Invalid or duplicate artifact id/kind in manifest: {artifact_id!r}/{kind!r}")
        seen_ids.add(artifact_id)
        checksum = item.get("checksum") or {}
        algorithm = checksum.get("algorithm")
        if algorithm is not None and algorithm not in hashlib.algorithms_available:
            raise BootstrapError(f"Unsupported checksum algorithm for {artifact_id}: {algorithm!r}")
        remote_required = bool(checksum.get("required", False))
        if kind in {"tar", "tree"} and not algorithm:
            raise BootstrapError(f"Artifact {artifact_id} lacks a checksum algorithm; refusing unsafe installation")
        expected_size = item.get("expected_size_bytes")
        if expected_size is not None and (not isinstance(expected_size, int) or expected_size <= 0):
            raise BootstrapError(f"Invalid expected_size_bytes for {artifact_id}")
        extract = item.get("extract") or {}
        strip_components = int(extract.get("strip_components", 0))
        if strip_components < 0:
            raise BootstrapError(f"Invalid strip_components for {artifact_id}")
        owned_prefixes = tuple(str(prefix) for prefix in item.get("owned_prefixes", []))
        for prefix in owned_prefixes:
            # Prefixes are only used to identify an unmanaged partial target
            # before any network I/O.  They are not filesystem paths.
            if not prefix or "/" in prefix or "\\" in prefix or ".." in prefix:
                raise BootstrapError(f"Unsafe owned_prefix for {artifact_id}: {prefix!r}")
        artifacts.append(Artifact(
            artifact_id=artifact_id,
            kind=kind,
            remote_path=str(item.get("remote_path", "")),
            target=_project_path(project_root, str(item.get("target", "")), field="target"),
            expected_size_bytes=expected_size,
            checksum_algorithm=algorithm,
            checksum_value=checksum.get("value"),
            checksum_remote_required=remote_required,
            strip_components=strip_components,
            owned_prefixes=owned_prefixes,
        ))
    if not artifacts:
        raise BootstrapError("Runtime manifest contains no artifacts")
    return Manifest(path=path, remote_root=str(remote_root), artifacts=tuple(artifacts))


def _state_path(artifact: Artifact) -> Path:
    base = artifact.target if artifact.kind == "tar" else artifact.target.parent
    return base / STATE_DIR_NAME / f"{artifact.artifact_id}.receipt.json"


def _read_receipt(artifact: Artifact) -> dict[str, Any] | None:
    path = _state_path(artifact)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if value.get("schema") != SCHEMA or value.get("artifact_id") != artifact.artifact_id:
        return None
    return value


def _checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_valid_file(artifact: Artifact) -> bool:
    if not artifact.target.is_file() or artifact.expected_size_bytes and artifact.target.stat().st_size != artifact.expected_size_bytes:
        return False
    return not artifact.checksum_value or _checksum(artifact.target, artifact.checksum_algorithm or "sha256") == artifact.checksum_value.lower()


def artifact_status(artifact: Artifact) -> dict[str, Any]:
    if artifact.kind == "file":
        return {
            "id": artifact.artifact_id,
            "kind": artifact.kind,
            "target": str(artifact.target),
            "status": "present" if _is_valid_file(artifact) else "missing",
            "network_used": False,
        }
    receipt = _read_receipt(artifact)
    if artifact.kind == "tree":
        valid = bool(receipt and _tree_matches_receipt(artifact.target, receipt))
        unmanaged = artifact.target.exists() and not valid
        return {
            "id": artifact.artifact_id,
            "kind": artifact.kind,
            "target": str(artifact.target),
            "status": "present" if valid else ("unmanaged_existing" if unmanaged else "missing"),
            "network_used": False,
            "receipt": str(_state_path(artifact)),
        }
    expected_prefixes_present = all(
        any(artifact.target.glob(f"{prefix}*")) for prefix in artifact.owned_prefixes
    )
    valid = bool(receipt and artifact.target.is_dir() and expected_prefixes_present)
    unmanaged = []
    if not valid and artifact.target.is_dir():
        for prefix in artifact.owned_prefixes:
            unmanaged.extend(path.name for path in artifact.target.glob(f"{prefix}*") if path.exists())
    return {
        "id": artifact.artifact_id,
        "kind": artifact.kind,
        "target": str(artifact.target),
        "status": "present" if valid else ("unmanaged_existing" if unmanaged else "missing"),
        "network_used": False,
        "receipt": str(_state_path(artifact)),
        "unmanaged_count": len(set(unmanaged)),
        "unmanaged_samples": sorted(set(unmanaged))[:5],
    }


def plan(manifest: Manifest, selected: set[str] | None = None) -> dict[str, Any]:
    artifacts = [a for a in manifest.artifacts if selected is None or a.artifact_id in selected]
    unknown = (selected or set()) - {a.artifact_id for a in manifest.artifacts}
    if unknown:
        raise BootstrapError(f"Unknown artifact id(s): {', '.join(sorted(unknown))}")
    states = [artifact_status(artifact) for artifact in artifacts]
    return {
        "schema": "hcmai.runtime_bootstrap_plan.v1",
        "manifest": str(manifest.path),
        "remote_root": manifest.remote_root,
        "network_used": False,
        "artifacts": states,
        "missing": [item["id"] for item in states if item["status"] == "missing"],
        "blocked": [item["id"] for item in states if item["status"] == "unmanaged_existing"],
        "next_command": "fetch --yes" if any(item["status"] == "missing" for item in states) else None,
    }


def _run_rclone(args: Sequence[str], *, runner: Run) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(list(args), check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise BootstrapError(f"rclone is not installed or not executable: {args[0]!r}") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "rclone failed").strip()
        raise BootstrapError(f"rclone command failed: {message}")
    return result


def _remote_metadata(rclone_bin: str, remote: str, *, runner: Run) -> dict[str, Any]:
    result = _run_rclone([rclone_bin, "lsjson", "--hash", "--files-only", remote], runner=runner)
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(f"rclone returned invalid metadata for {remote}") from exc
    if len(entries) != 1 or entries[0].get("IsDir"):
        raise BootstrapError(f"Remote artifact is missing or not a file: {remote}")
    entry = entries[0]
    if not isinstance(entry.get("Size"), int) or entry["Size"] <= 0:
        raise BootstrapError(f"Remote artifact has no usable size metadata: {remote}")
    return entry


def _remote_tree_metadata(rclone_bin: str, remote: str, *, runner: Run) -> list[dict[str, Any]]:
    """Return an immutable, complete file manifest before any tree download."""
    result = _run_rclone(
        [rclone_bin, "lsjson", "--recursive", "--files-only", "--hash", remote], runner=runner,
    )
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BootstrapError(f"rclone returned invalid tree metadata for {remote}") from exc
    if not isinstance(entries, list) or not entries:
        raise BootstrapError(f"Remote tree is missing or empty: {remote}")
    manifest: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("IsDir"):
            raise BootstrapError(f"Remote tree metadata contains an invalid directory entry: {remote}")
        relative = str(entry.get("Path") or "")
        normalized = _safe_relative(relative, field="remote tree path").as_posix()
        if normalized in seen_paths:
            raise BootstrapError(f"Remote tree metadata contains duplicate path: {normalized}")
        seen_paths.add(normalized)
        size = entry.get("Size")
        if not isinstance(size, int) or size < 0:
            raise BootstrapError(f"Remote tree entry has no usable size metadata: {normalized}")
        hashes = entry.get("Hashes") or {}
        md5 = next((value for key, value in hashes.items() if str(key).lower() == "md5"), None)
        if not isinstance(md5, str) or not md5:
            raise BootstrapError(
                f"Remote tree entry has no MD5 checksum: {normalized}; refusing fail-open tree download"
            )
        manifest.append({"path": normalized, "size": size, "md5": md5.lower()})
    return sorted(manifest, key=lambda item: item["path"])


def _local_tree_inventory(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    inventory: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise BootstrapError(f"Tree contains prohibited symbolic link: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(root).as_posix()
            inventory[relative] = candidate
    return inventory


def _tree_matches_receipt(target: Path, receipt: dict[str, Any]) -> bool:
    files = receipt.get("files")
    if not isinstance(files, list) or not target.is_dir():
        return False
    try:
        expected = {
            _safe_relative(str(item["path"]), field="receipt tree path").as_posix(): {
                "size": int(item["size"]),
                "md5": str(item["md5"]).lower(),
            }
            for item in files
        }
        local = _local_tree_inventory(target)
    except (BootstrapError, KeyError, TypeError, ValueError):
        return False
    if set(local) != set(expected):
        return False
    # A receipt only proves what was fetched at install time. Recheck both
    # size and MD5 locally before a later plan calls the tree healthy; a
    # same-size corruption must not be silently treated as a valid index.
    return all(
        local[path].stat().st_size == item["size"]
        and _checksum(local[path], "md5").lower() == item["md5"]
        for path, item in expected.items()
    )


def _verify_downloaded_tree(staging_target: Path, remote_manifest: list[dict[str, Any]]) -> None:
    expected = {item["path"]: item for item in remote_manifest}
    local = _local_tree_inventory(staging_target)
    missing = sorted(set(expected) - set(local))
    extra = sorted(set(local) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        raise BootstrapError("Downloaded tree does not exactly match enumerated remote manifest: " + "; ".join(details))
    for relative, expected_item in expected.items():
        local_path = local[relative]
        if local_path.stat().st_size != expected_item["size"]:
            raise BootstrapError(f"Downloaded tree size mismatch: {relative}")
        actual = _checksum(local_path, "md5")
        if actual.lower() != expected_item["md5"]:
            raise BootstrapError(f"Downloaded tree checksum mismatch: {relative}")


def _download_tree(
    artifact: Artifact,
    *,
    remote: str,
    remote_manifest: list[dict[str, Any]],
    staging: Path,
    rclone_bin: str,
    runner: Run,
) -> Path:
    total_size = sum(item["size"] for item in remote_manifest)
    staging.mkdir(parents=True, exist_ok=True)
    _require_space(staging, total_size)
    staged_target = staging / f"{artifact.artifact_id}.tree.part"
    _run_rclone(
        [rclone_bin, "copy", "--partial", "--inplace", remote, str(staged_target)], runner=runner,
    )
    _verify_downloaded_tree(staged_target, remote_manifest)
    return staged_target


def _promote_tree(staged_target: Path, target: Path) -> None:
    if target.exists():
        raise BootstrapError(f"Refusing to overwrite existing tree target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staged_target, target)
    except OSError as exc:
        raise BootstrapError(f"Cannot atomically promote staged tree to {target}: {exc}") from exc


def _remote_checksum(artifact: Artifact, metadata: dict[str, Any]) -> tuple[str, str]:
    algorithm = artifact.checksum_algorithm
    if not algorithm:
        raise BootstrapError(f"Artifact {artifact.artifact_id} has no checksum algorithm")
    expected = artifact.checksum_value
    if expected:
        return algorithm, str(expected).lower()
    hashes = metadata.get("Hashes") or {}
    exact = next((value for key, value in hashes.items() if str(key).lower() == algorithm.lower()), None)
    if artifact.checksum_remote_required and not exact:
        raise BootstrapError(
            f"Remote checksum {algorithm} is unavailable for {artifact.artifact_id}; refusing to download/extract"
        )
    if not exact:
        raise BootstrapError(f"No expected checksum for {artifact.artifact_id}; refusing fail-open download")
    return algorithm, str(exact).lower()


def _safe_tar_parts(name: str, strip_components: int) -> tuple[str, ...] | None:
    raw = Path(name)
    if raw.is_absolute() or ".." in raw.parts:
        raise BootstrapError(f"Unsafe archive member path: {name!r}")
    parts = tuple(part for part in raw.parts if part not in {"", "."})
    if len(parts) <= strip_components:
        return None
    return parts[strip_components:]


def _verify_tar(archive: Path, target: Path, strip_components: int) -> list[tarfile.TarInfo]:
    try:
        handle = tarfile.open(archive, mode="r:")
    except (tarfile.TarError, OSError) as exc:
        raise BootstrapError(f"Cannot read verified tar archive {archive}: {exc}") from exc
    with handle:
        members = handle.getmembers()
        if not members:
            raise BootstrapError(f"Archive contains no members: {archive}")
        target_root = target.resolve()
        for member in members:
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise BootstrapError(f"Archive contains prohibited link/device member: {member.name!r}")
            parts = _safe_tar_parts(member.name, strip_components)
            if parts is None:
                continue
            destination = target_root.joinpath(*parts)
            try:
                destination.resolve().relative_to(target_root)
            except ValueError as exc:
                raise BootstrapError(f"Archive member escapes extraction target: {member.name!r}") from exc
            if destination.exists():
                raise BootstrapError(
                    f"Refusing to overwrite existing target path for {archive.name}: {destination}. "
                    "Inspect it or restore the receipt; bootstrap never overwrites existing data."
                )
        return members


def _extract_tar(archive: Path, target: Path, strip_components: int, staging: Path) -> None:
    members = _verify_tar(archive, target, strip_components)
    extract_root = Path(tempfile.mkdtemp(prefix="extract-", dir=staging))
    try:
        with tarfile.open(archive, mode="r:") as handle:
            for member in members:
                parts = _safe_tar_parts(member.name, strip_components)
                if parts is None:
                    continue
                destination = extract_root.joinpath(*parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise BootstrapError(f"Archive member is not a regular file: {member.name!r}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise BootstrapError(f"Cannot read archive member: {member.name!r}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(destination, member.mode & 0o777)
        target.mkdir(parents=True, exist_ok=True)
        for child in extract_root.iterdir():
            destination = target / child.name
            if destination.exists():
                raise BootstrapError(f"Refusing to overwrite existing extraction root: {destination}")
            os.replace(child, destination)
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)


def _free_bytes(path: Path) -> int:
    probe = path if path.exists() else path.parent
    return shutil.disk_usage(probe).free


def _require_space(staging: Path, remote_size: int) -> None:
    # tar archives are uncompressed in the current Drive manifest. Allocate a
    # full second copy plus a small filesystem margin while extracting one at a time.
    required = remote_size * 2 + 128 * 1024 * 1024
    free = _free_bytes(staging)
    if free < required:
        raise BootstrapError(
            f"Insufficient free space: need at least {required} bytes for one archive, have {free} bytes"
        )


@contextlib.contextmanager
def bootstrap_lock(project_root: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux server contract
        raise BootstrapError("A POSIX filesystem is required for bootstrap locking") from exc
    lock_dir = project_root / STATE_DIR_NAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "bootstrap.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BootstrapError("Another runtime bootstrap process is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_receipt(
    artifact: Artifact,
    *,
    remote: str,
    size: int,
    algorithm: str,
    digest: str,
    files: list[dict[str, Any]] | None = None,
) -> None:
    path = _state_path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "artifact_id": artifact.artifact_id,
        "remote": remote,
        "size_bytes": size,
        "checksum": {"algorithm": algorithm, "value": digest},
        "installed_at_epoch": int(time.time()),
    }
    if files is not None:
        payload["files"] = files
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _download_file(
    artifact: Artifact,
    *,
    remote: str,
    remote_size: int,
    algorithm: str,
    expected_digest: str,
    staging: Path,
    rclone_bin: str,
    runner: Run,
) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    _require_space(staging, remote_size)
    partial = staging / f"{artifact.artifact_id}.part"
    _run_rclone([rclone_bin, "copyto", "--partial", "--inplace", remote, str(partial)], runner=runner)
    if not partial.is_file() or partial.stat().st_size != remote_size:
        raise BootstrapError(f"Downloaded artifact size mismatch for {artifact.artifact_id}")
    actual_digest = _checksum(partial, algorithm)
    if actual_digest.lower() != expected_digest.lower():
        raise BootstrapError(
            f"Downloaded checksum mismatch for {artifact.artifact_id}: expected {expected_digest}, got {actual_digest}"
        )
    return partial


def fetch(
    manifest: Manifest,
    project_root: Path,
    *,
    selected: set[str] | None,
    rclone_bin: str,
    runner: Run = subprocess.run,
) -> dict[str, Any]:
    initial = plan(manifest, selected)
    if initial["blocked"]:
        raise BootstrapError(
            "Target contains unmanaged paths for: " + ", ".join(initial["blocked"]) +
            ". Refusing to overwrite or claim existing data; restore its receipt or use a clean target."
        )
    targets = [a for a in manifest.artifacts if a.artifact_id in set(initial["missing"])]
    staging = project_root / STATE_DIR_NAME / DOWNLOAD_DIR_NAME
    installed: list[str] = []
    skipped = [item["id"] for item in initial["artifacts"] if item["status"] == "present"]
    with bootstrap_lock(project_root):
        for artifact in targets:
            # Recheck after obtaining the lock, so two callers never race.
            current_status = artifact_status(artifact)["status"]
            if current_status == "present":
                skipped.append(artifact.artifact_id)
                continue
            if current_status != "missing":
                raise BootstrapError(
                    f"Target became unmanaged while waiting for the bootstrap lock: {artifact.artifact_id}"
                )
            remote = _remote_file(manifest.remote_root, artifact.remote_path)
            if artifact.kind == "tree":
                remote_tree = _remote_tree_metadata(rclone_bin, remote, runner=runner)
                staged_tree = _download_tree(
                    artifact,
                    remote=remote,
                    remote_manifest=remote_tree,
                    staging=staging,
                    rclone_bin=rclone_bin,
                    runner=runner,
                )
                try:
                    _promote_tree(staged_tree, artifact.target)
                    _write_receipt(
                        artifact,
                        remote=remote,
                        size=sum(item["size"] for item in remote_tree),
                        algorithm="sha256",
                        digest=hashlib.sha256(
                            json.dumps(remote_tree, sort_keys=True, separators=(",", ":")).encode("utf-8")
                        ).hexdigest(),
                        files=remote_tree,
                    )
                except Exception:
                    # A staging tree is intentionally retained after failure so
                    # rclone can resume it. A promoted target is never deleted.
                    raise
                installed.append(artifact.artifact_id)
                continue
            metadata = _remote_metadata(rclone_bin, remote, runner=runner)
            remote_size = int(metadata["Size"])
            if artifact.expected_size_bytes is not None and remote_size != artifact.expected_size_bytes:
                raise BootstrapError(
                    f"Remote size mismatch for {artifact.artifact_id}: expected {artifact.expected_size_bytes}, got {remote_size}"
                )
            algorithm, expected_digest = _remote_checksum(artifact, metadata)
            downloaded = _download_file(
                artifact, remote=remote, remote_size=remote_size, algorithm=algorithm,
                expected_digest=expected_digest, staging=staging, rclone_bin=rclone_bin, runner=runner,
            )
            try:
                if artifact.kind == "tar":
                    _extract_tar(downloaded, artifact.target, artifact.strip_components, staging)
                    _write_receipt(
                        artifact, remote=remote, size=remote_size, algorithm=algorithm, digest=expected_digest,
                    )
                else:
                    artifact.target.parent.mkdir(parents=True, exist_ok=True)
                    if artifact.target.exists():
                        raise BootstrapError(f"Refusing to overwrite existing file: {artifact.target}")
                    os.replace(downloaded, artifact.target)
                installed.append(artifact.artifact_id)
            finally:
                if downloaded.exists():
                    downloaded.unlink()
    return {
        "schema": "hcmai.runtime_bootstrap_fetch.v1",
        "remote_root": manifest.remote_root,
        "network_used": True,
        "installed": installed,
        "skipped": sorted(set(skipped)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fetch"))
    parser.add_argument("--project-root", type=Path, default=project_root_from_script())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--remote-root", help="Override manifest remote root (or HCMAI_RUNTIME_REMOTE)")
    parser.add_argument("--rclone-bin", default=os.environ.get("HCMAI_RCLONE_BIN", "rclone"))
    parser.add_argument("--artifact", action="append", dest="artifacts", help="Artifact id to process; repeatable")
    parser.add_argument("--yes", action="store_true", help="Required acknowledgement before fetch can download/extract")
    parser.add_argument("--json", action="store_true", help="Emit JSON (the default output format)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    try:
        manifest = load_manifest(manifest_path.resolve(), project_root, args.remote_root)
        selected = set(args.artifacts) if args.artifacts else None
        if args.command == "plan":
            report = plan(manifest, selected)
        else:
            if not args.yes:
                raise BootstrapError("Refusing download/extraction without --yes. Run `plan` first to inspect changes.")
            report = fetch(manifest, project_root, selected=selected, rclone_bin=args.rclone_bin)
    except BootstrapError as exc:
        report = {"schema": "hcmai.runtime_bootstrap_error.v1", "ok": False, "error": str(exc)}
        print(json.dumps(report, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
