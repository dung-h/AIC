"""Offline, fail-closed competition preflight for the HCMAI production assets.

The command performs no network requests and never includes credential values
in its JSON report.  It validates persisted artifacts cheaply (Parquet
metadata and memory-mapped NumPy headers) before a competition process starts.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence
import zipfile

from src.core.providers import provider_for
from src.utils.paths import activate_runtime_env, load_runtime_env


ROOT = Path(__file__).resolve().parents[2]
# Preflight has environment-derived parser defaults, so activate the shared
# dotenv contract before those defaults are constructed.  Explicit exports
# remain authoritative.
activate_runtime_env()
DEFAULT_CANONICAL = ROOT / "data/index/global_keyframes.parquet"
DEFAULT_KEYFRAMES = ROOT / "data/keyframes"
DEFAULT_VISUAL_INDEXES = (
    ROOT / "data/index/global_siglip_vitl.npy",
    ROOT / "data/index/global_so400m384.npy",
)
DEFAULT_VISUAL_MAPS = (
    ROOT / "data/index/global_keyframes_vitl.parquet",
    ROOT / "data/index/global_keyframes_so400m384.parquet",
)
DEFAULT_HF_HUB = Path(os.getenv(
    "HF_HUB_CACHE",
    str(Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"),
))
DEFAULT_VISUAL_BACKBONES = (
    DEFAULT_HF_HUB / "models--timm--ViT-L-16-SigLIP2-256",
    DEFAULT_HF_HUB / "models--timm--ViT-SO400M-16-SigLIP2-384",
)
DEFAULT_ASR = ROOT / "data/index/modality_global_v2/asr_global_merged_v2"
DEFAULT_OCR = ROOT / "data/index/modality_global_v2/ocr_global_merged_v2"
DEFAULT_MODEL = ROOT / "models/Qwen2.5-VL-7B-Instruct"
DEFAULT_OUTPUT = ROOT / "results/submissions"
EXPECTED_PACKS = tuple(
    [f"K{i:02d}" for i in range(1, 21)]
    + [f"L{i:02d}" for i in range(21, 31)]
)
REQUIRED_MODULES = (
    "numpy", "pandas", "pyarrow", "torch", "transformers", "PIL",
)
MODALITY_RUNTIME_MODULES = ("sentence_transformers",)
SENSITIVE_MARKERS = ("secret", "password", "token", "api_key", "apikey")
# A WSL virtualenv stored on /mnt/c performs thousands of small metadata reads
# during a cold sentence-transformers import. Profiling on the competition
# workstation shows this can exceed 180 seconds even though individual modules
# are healthy. Native Linux/ext4 remains much faster; 300s keeps the probe
# fail-closed without turning cold NTFS I/O into a false dependency blocker.
IMPORT_PROBE_TIMEOUT_SECONDS = 300
_IMPORT_PROBE_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def _normalise_active_video_prefixes(value: object | None) -> tuple[str, ...]:
    """Return the installed-corpus selector used by every retrieval lane.

    The global metadata/index bundle intentionally contains both K and L
    series.  A preselection deployment may install only the L keyframe
    archives, so preflight must validate that *active* corpus rather than
    reject the valid full metadata bundle or demand unavailable K images.
    """
    if value is None:
        value = os.getenv("HCMAI_ACTIVE_VIDEO_PREFIXES", "")
    if isinstance(value, str):
        values = value.split(",")
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("active video prefixes must be a comma-separated string or sequence") from exc
    prefixes = tuple(dict.fromkeys(str(item).strip().upper() for item in values if str(item).strip()))
    if any(not prefix.replace("_", "").isalnum() for prefix in prefixes):
        raise ValueError("active video prefixes must contain only alphanumeric characters or underscores")
    return prefixes


DEFAULT_ACTIVE_VIDEO_PREFIXES = _normalise_active_video_prefixes(None)


def _default_expected_video_count(prefixes: Sequence[str]) -> int | None:
    """Known corpus sizes; leave custom partial selections unconstrained."""
    selected = frozenset(prefixes)
    if not selected or selected == frozenset(("L", "K")):
        return 1478
    if selected == frozenset(("L",)):
        return 873
    return None


def _is_active_video(video_id: str, prefixes: Sequence[str]) -> bool:
    return not prefixes or str(video_id).strip().upper().startswith(tuple(prefixes))


def _active_expected_packs(prefixes: Sequence[str]) -> set[str]:
    return {
        pack for pack in EXPECTED_PACKS
        if not prefixes or pack.startswith(tuple(prefixes))
    }


@dataclass(frozen=True)
class PreflightConfig:
    project_root: Path = ROOT
    canonical_map: Path = DEFAULT_CANONICAL
    keyframes_dir: Path = DEFAULT_KEYFRAMES
    visual_indexes: tuple[Path, ...] = DEFAULT_VISUAL_INDEXES
    visual_maps: tuple[Path, ...] = DEFAULT_VISUAL_MAPS
    visual_backbone_dirs: tuple[Path, ...] = DEFAULT_VISUAL_BACKBONES
    asr_dir: Path = DEFAULT_ASR
    ocr_dir: Path = DEFAULT_OCR
    provider: str = "local"
    local_model: Path = DEFAULT_MODEL
    output_dir: Path = DEFAULT_OUTPUT
    query_path: Path | None = None
    output_package: Path | None = None
    active_video_prefixes: tuple[str, ...] = DEFAULT_ACTIVE_VIDEO_PREFIXES
    expected_video_count: int | None = field(
        default_factory=lambda: _default_expected_video_count(DEFAULT_ACTIVE_VIDEO_PREFIXES)
    )
    min_free_gb: float = 5.0
    require_gpu: bool = False
    require_modality_runtime: bool = False
    dotenv_path: Path | None = None


@dataclass
class ReportBuilder:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        name: str,
        ok: bool,
        message: str,
        *,
        blocking: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.checks.append({
            "name": name,
            "status": "pass" if ok else ("blocker" if blocking else "warning"),
            "blocking": bool(blocking and not ok),
            "message": message,
            "details": details or {},
        })


def _safe_path(path: Path) -> str:
    return str(path.resolve(strict=False))


def _parquet_metadata(path: Path, required: Iterable[str]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"missing parquet: {path}")
    metadata = pq.ParquetFile(path).metadata
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(required) - names)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if metadata.num_rows <= 0:
        raise ValueError("parquet has no rows")
    return {"rows": int(metadata.num_rows), "columns": sorted(names)}


def _canonical_summary(
    path: Path,
    *,
    active_video_prefixes: Sequence[str] = (),
    collect_pairs: bool = False,
) -> tuple[dict[str, Any], set[tuple[str, int]] | None]:
    import pyarrow.parquet as pq

    meta = _parquet_metadata(path, ("video_id", "kf_n", "frame_idx", "pts_time"))
    table = pq.read_table(path, columns=["video_id", "kf_n", "frame_idx"])
    videos = table.column("video_id").to_pylist()
    keyframes = table.column("kf_n").to_pylist()
    frame_ids = table.column("frame_idx").to_pylist()
    identities: set[tuple[str, int]] = set()
    pairs: set[tuple[str, int]] | None = set() if collect_pairs else None
    video_ids: set[str] = set()
    for video, keyframe, frame_id in zip(videos, keyframes, frame_ids):
        video_id = str(video).strip().upper()
        if not video_id:
            raise ValueError("canonical map contains an empty video_id")
        if not _is_active_video(video_id, active_video_prefixes):
            continue
        kf_n, frame_idx = int(keyframe), int(frame_id)
        if kf_n < 0 or frame_idx < 0:
            raise ValueError("canonical map contains a negative frame identity")
        identity = (video_id, kf_n)
        if identity in identities:
            raise ValueError(f"duplicate canonical identity: {identity}")
        identities.add(identity)
        video_ids.add(video_id)
        if pairs is not None:
            pairs.add((video_id, frame_idx))
    meta.update({
        "videos": len(video_ids),
        "unique_identities": len(identities),
        "active_video_prefixes": list(active_video_prefixes),
    })
    return meta, pairs


def _check_python(builder: ReportBuilder, config: PreflightConfig) -> None:
    executable = str(Path(sys.executable).resolve())
    linux = sys.platform.startswith("linux") and not executable.lower().endswith(".exe")
    isolated = sys.prefix != sys.base_prefix
    forbidden = [item for item in sys.path if "\\" in item or item.lower().endswith((".dll", ".pyd"))]
    # WSL's /mnt mounts are Linux paths syntactically, but they retain slow
    # cross-filesystem metadata semantics. A virtualenv there is forbidden by
    # the runtime contract and must fail before any expensive package probe.
    # ``Path(sys.executable).resolve()`` follows a venv's Python symlink to
    # /usr/bin, so inspect sys.prefix as well; it names the actual venv root.
    windows_mounted = any(
        str(path).startswith("/mnt/")
        for path in (executable, sys.prefix, sys.exec_prefix)
    )
    environment_safe = linux and isolated and not forbidden and not windows_mounted
    required_modules = list(REQUIRED_MODULES)
    if config.require_modality_runtime:
        required_modules.extend(MODALITY_RUNTIME_MODULES)
    missing = []
    if environment_safe:
        missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
        if config.provider == "local" and importlib.util.find_spec("bitsandbytes") is None:
            missing.append("bitsandbytes")
    probe_modules = required_modules + (["bitsandbytes"] if config.provider == "local" else [])
    probe_key = tuple(probe_modules)
    # A Windows-mounted or non-venv interpreter is already rejected by the
    # deployment contract.  Do not make an invalid runtime spend minutes
    # importing Torch/Transformers from a slow mount merely to report that
    # same failure. Native Linux venvs retain the full isolated import probe.
    if probe_key not in _IMPORT_PROBE_CACHE and not missing and environment_safe:
        code = "import importlib; " + "; ".join(
            f"importlib.import_module({name!r})" for name in probe_modules
        )
        clean_env = {
            key: value for key, value in os.environ.items()
            if not any(marker in key.casefold() for marker in SENSITIVE_MARKERS)
        }
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"):
            clean_env.pop(key, None)
        clean_env.update({"PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code], cwd=config.project_root,
                env=clean_env, capture_output=True, text=True,
                timeout=IMPORT_PROBE_TIMEOUT_SECONDS, check=False,
            )
            _IMPORT_PROBE_CACHE[probe_key] = {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "error": completed.stderr.strip()[-500:] if completed.returncode else "",
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            _IMPORT_PROBE_CACHE[probe_key] = {
                "ok": False, "returncode": None, "error": type(exc).__name__,
            }
    import_probe = _IMPORT_PROBE_CACHE.get(probe_key) or {
        "ok": False,
        "returncode": None,
        "error": (
            "invalid_linux_virtualenv" if not environment_safe
            else "modules missing"
        ),
    }
    ok = environment_safe and not missing and sys.version_info >= (3, 10) and bool(import_probe["ok"])
    builder.add(
        "python_environment", ok,
        "Linux virtual environment and required modules are available" if ok else "Python runtime is not competition-safe",
        details={
            "executable": executable,
            "version": ".".join(map(str, sys.version_info[:3])),
            "linux": linux,
            "virtual_environment": isolated,
            "windows_mounted": windows_mounted,
            "missing_modules": sorted(set(missing)),
            "required_modules": probe_modules if not missing else required_modules,
            "modality_runtime_required": config.require_modality_runtime,
            "forbidden_path_count": len(forbidden),
            "import_probe": import_probe,
        },
    )


def _check_gpu(builder: ReportBuilder, require_gpu: bool) -> None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        builder.add("gpu", False, "nvidia-smi is unavailable", blocking=require_gpu)
        return
    try:
        result = subprocess.run(
            [binary, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
            env={**os.environ, "CUDA_MODULE_LOADING": "LAZY"},
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        ok = result.returncode == 0 and bool(lines)
    except (OSError, subprocess.TimeoutExpired) as exc:
        lines, ok = [], False
        error = type(exc).__name__
    else:
        error = result.stderr.strip()[:300] if not ok else ""
    builder.add(
        "gpu", ok,
        "NVIDIA GPU is visible" if ok else "GPU probe failed",
        blocking=require_gpu,
        details={"devices": lines, "error": error},
    )


def _check_canonical(builder: ReportBuilder, config: PreflightConfig, *, collect_pairs: bool) -> set[tuple[str, int]] | None:
    try:
        summary, pairs = _canonical_summary(
            config.canonical_map,
            active_video_prefixes=config.active_video_prefixes,
            collect_pairs=collect_pairs,
        )
        expected = config.expected_video_count
        if expected is not None and summary["videos"] != expected:
            raise ValueError(f"video coverage {summary['videos']} != expected {expected}")
    except Exception as exc:
        builder.add("canonical_map", False, f"Canonical map is invalid: {exc}", details={"path": _safe_path(config.canonical_map)})
        return None
    builder.add("canonical_map", True, "Canonical frame map is valid", details={"path": _safe_path(config.canonical_map), **summary})
    return pairs


def _check_keyframes(builder: ReportBuilder, config: PreflightConfig) -> None:
    """Verify the portable flat image layout and one canonical image/video."""
    import pyarrow.parquet as pq

    try:
        root = config.keyframes_dir
        if not root.is_dir():
            raise FileNotFoundError(f"missing keyframe root: {root}")
        nested = root / "keyframes"
        if nested.exists():
            raise ValueError(f"redundant nested keyframe directory remains: {nested}")
        table = pq.read_table(config.canonical_map, columns=["video_id", "kf_n"])
        first_keyframe: dict[str, int] = {}
        for video, keyframe in zip(
            table.column("video_id").to_pylist(),
            table.column("kf_n").to_pylist(),
        ):
            video_id = str(video).strip().upper()
            if not _is_active_video(video_id, config.active_video_prefixes):
                continue
            kf_n = int(keyframe)
            current = first_keyframe.get(video_id)
            if current is None or kf_n < current:
                first_keyframe[video_id] = kf_n
        actual_videos = {path.name.upper() for path in root.iterdir() if path.is_dir()}
        expected_videos = set(first_keyframe)
        missing_dirs = sorted(expected_videos - actual_videos)
        # A developer workstation may contain an installed superset.  It is
        # harmless because retrieval is already bounded to active prefixes;
        # only the selected corpus must be complete on a portable server.
        inactive_extra_dirs = sorted(actual_videos - expected_videos)
        missing_probes = sorted(
            video_id for video_id, kf_n in first_keyframe.items()
            if not (root / video_id / f"{kf_n:03d}.jpg").is_file()
        )
        if missing_dirs or missing_probes:
            raise ValueError(
                "keyframe layout mismatch: "
                f"missing_dirs={missing_dirs[:10]}, "
                f"missing_probe_images={missing_probes[:10]}"
            )
    except Exception as exc:
        builder.add(
            "keyframe_images", False, f"Keyframe image layout is invalid: {exc}",
            details={"path": _safe_path(config.keyframes_dir)},
        )
        return
    builder.add(
        "keyframe_images", True,
        "Flat keyframe layout matches every active canonical video",
        details={
            "path": _safe_path(config.keyframes_dir),
            "videos": len(expected_videos),
            "probe_images": len(first_keyframe),
            "active_video_prefixes": list(config.active_video_prefixes),
            "inactive_extra_videos": len(inactive_extra_dirs),
            "layout": "data/keyframes/<video_id>/<kf_n>.jpg",
        },
    )


def _check_visual(builder: ReportBuilder, config: PreflightConfig) -> None:
    import numpy as np

    if not config.visual_indexes or len(config.visual_indexes) != len(config.visual_maps):
        builder.add("visual_indexes", False, "Visual index/map lists must be non-empty and have equal length")
        return
    items: list[dict[str, Any]] = []
    try:
        for index_path, map_path in zip(config.visual_indexes, config.visual_maps):
            if not index_path.is_file():
                raise FileNotFoundError(f"missing visual index: {index_path}")
            mapping = _parquet_metadata(map_path, ("video_id", "kf_n", "frame_idx", "pts_time"))
            if index_path.suffix.casefold() == ".npy":
                matrix = np.load(index_path, mmap_mode="r", allow_pickle=False)
                if matrix.ndim != 2 or matrix.shape[0] != mapping["rows"] or matrix.shape[1] <= 0:
                    raise ValueError(f"visual shape/map mismatch: {matrix.shape} vs {mapping['rows']} rows")
                shape = [int(value) for value in matrix.shape]
            else:
                manifest_path = Path(str(index_path) + ".manifest.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                shape = list(manifest.get("shape", []))
                if len(shape) != 2 or int(shape[0]) != mapping["rows"] or index_path.stat().st_size <= 0:
                    raise ValueError("visual binary manifest/map mismatch")
            items.append({"index": _safe_path(index_path), "map": _safe_path(map_path), "shape": shape})
    except Exception as exc:
        builder.add("visual_indexes", False, f"Visual retrieval artifacts are invalid: {exc}", details={"validated": items})
        return
    builder.add("visual_indexes", True, "Visual retrieval matrices align with canonical maps", details={"artifacts": items})


def _check_visual_backbones(builder: ReportBuilder, config: PreflightConfig) -> None:
    items: list[dict[str, Any]] = []
    try:
        if not config.visual_backbone_dirs:
            raise ValueError("no visual backbone cache directories configured")
        for directory in config.visual_backbone_dirs:
            if not directory.is_dir():
                raise FileNotFoundError(f"missing visual backbone cache: {directory}")
            snapshots = directory / "snapshots"
            weights = [path for path in snapshots.glob("*/*.safetensors") if path.is_file() and path.stat().st_size > 0]
            tokenizers = [path for path in snapshots.glob("*/tokenizer.json") if path.is_file() and path.stat().st_size > 0]
            tokenizer_configs = [
                path for path in snapshots.glob("*/tokenizer_config.json")
                if path.is_file() and path.stat().st_size > 0
            ]
            if not weights or not tokenizers or not tokenizer_configs:
                raise ValueError(
                    f"incomplete visual backbone snapshot: {directory}; "
                    "weights, tokenizer.json, and tokenizer_config.json are required"
                )
            items.append({
                "directory": _safe_path(directory),
                "weight_bytes": max(path.stat().st_size for path in weights),
                "snapshots": len({path.parent.name for path in weights}),
            })
    except Exception as exc:
        builder.add(
            "visual_backbones", False,
            f"Offline visual backbone cache is invalid: {exc}",
            details={"validated": items},
        )
        return
    builder.add(
        "visual_backbones", True,
        "Offline visual backbone weights and tokenizers are complete",
        details={"artifacts": items},
    )


def _resolve_artifact(directory: Path, manifest_path: Path, value: Any, fallback: str) -> Path:
    if value:
        candidate = Path(str(value))
        for path in (candidate, manifest_path.parent / candidate, ROOT / candidate):
            if path.is_file():
                return path
    return directory / fallback


def _active_modality_video_count(
    manifest: dict[str, Any],
    *,
    prefixes: Sequence[str],
    required_packs: set[str],
) -> int:
    """Read coverage from source metadata without mistaking text rows for videos.

    Silent/no-text videos do not appear in retrieval parquet rows.  ASR v2 has
    an explicit video-id list; OCR v2 records canonical counts per pack.
    """
    scope = manifest.get("scope") or {}
    video_ids = scope.get("video_ids")
    if isinstance(video_ids, list):
        return len({str(video).strip().upper() for video in video_ids if _is_active_video(str(video), prefixes)})
    if not prefixes:
        return int(scope.get("video_count", 0))
    records = manifest.get("packs") or {}
    if not isinstance(records, dict):
        raise ValueError("manifest pack coverage is not a mapping")
    total = 0
    for pack in required_packs:
        item = records.get(pack) or records.get(pack.lower())
        if not isinstance(item, dict):
            raise ValueError(f"manifest is missing coverage record for {pack}")
        value = item.get("canonical_videos", item.get("videos"))
        if value is None:
            raise ValueError(f"manifest coverage record for {pack} has no canonical video count")
        total += int(value)
    return total


def _check_modality(
    builder: ReportBuilder,
    name: str,
    directory: Path,
    expected_videos: int | None,
    active_video_prefixes: Sequence[str] = (),
) -> None:
    import numpy as np

    manifest_name = "asr_global_merge_v2_manifest.json" if name == "asr" else "manifest.json"
    manifest_path = directory / manifest_name
    required = (
        ("video_id", "chunk_index", "text", "start", "end", "frame_idx", "kf_n", "pts_time", "embedding_row", "source_pack", "source_provenance")
        if name == "asr" else
        ("video_id", "ocr_text", "frame_idx", "kf_n", "pts_time", "embedding_row", "source_pack")
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "ready":
            raise ValueError(f"manifest status is {manifest.get('status')!r}")
        if not bool((manifest.get("canonical") or {}).get("validated")):
            raise ValueError("manifest has no validated canonical mapping")
        packs = {str(item).upper() for item in (manifest.get("scope") or {}).get("packs", [])}
        required_packs = _active_expected_packs(active_video_prefixes)
        missing_packs = sorted(required_packs - packs)
        if missing_packs:
            raise ValueError(f"missing packs: {missing_packs}")
        video_count = _active_modality_video_count(
            manifest,
            prefixes=active_video_prefixes,
            required_packs=required_packs,
        )
        if expected_videos is not None and video_count != expected_videos:
            raise ValueError(f"video coverage {video_count} != expected {expected_videos}")
        artifacts = manifest.get("artifacts") or {}
        metadata_path = _resolve_artifact(directory, manifest_path, artifacts.get("retrieval"), "retrieval.parquet")
        embedding_path = _resolve_artifact(directory, manifest_path, artifacts.get("embeddings"), "embeddings.npy")
        metadata = _parquet_metadata(metadata_path, required)
        import pyarrow.parquet as pq
        embedding_rows = [int(value) for value in pq.read_table(metadata_path, columns=["embedding_row"]).column(0).to_pylist()]
        if len(embedding_rows) != len(set(embedding_rows)) or set(embedding_rows) != set(range(metadata["rows"])):
            raise ValueError("embedding_row is not a unique contiguous 0..N-1 mapping")
        embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != metadata["rows"]:
            raise ValueError(f"embedding/metadata mismatch: {embeddings.shape} vs {metadata['rows']}")
        declared = (manifest.get("embedding") or {}).get("shape")
        if declared and [int(value) for value in declared] != [int(value) for value in embeddings.shape]:
            raise ValueError(f"manifest shape {declared} != actual {list(embeddings.shape)}")
        details = {
            "directory": _safe_path(directory),
            "rows": metadata["rows"],
            "shape": [int(value) for value in embeddings.shape],
            "packs": len(packs),
            "videos": video_count,
            "active_packs": sorted(required_packs),
            "active_video_prefixes": list(active_video_prefixes),
        }
        if name == "ocr":
            coverage = manifest.get("coverage") or {}
            details["canonical_frame_coverage"] = coverage.get("canonical_frame_coverage")
            details["frame_complete"] = bool(coverage.get("frame_complete"))
        else:
            details["no_speech_videos"] = sum(
                len((item or {}).get("no_speech_videos", []))
                for item in (manifest.get("packs") or {}).values()
            )
    except Exception as exc:
        builder.add(f"{name}_index", False, f"{name.upper()} index is invalid: {exc}", details={"directory": _safe_path(directory)})
        return
    builder.add(f"{name}_index", True, f"{name.upper()} index is ready", details=details)
    if name == "ocr" and not details.get("frame_complete", False):
        builder.add(
            "ocr_frame_coverage", False,
            "OCR is global at video level but intentionally sampled, not frame-complete",
            blocking=False,
            details={"canonical_frame_coverage": details.get("canonical_frame_coverage")},
        )


def _check_provider(builder: ReportBuilder, config: PreflightConfig) -> None:
    if config.provider == "local":
        path = config.local_model
        try:
            model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            if model_config.get("model_type") != "qwen2_5_vl":
                raise ValueError(
                    "unsupported local model_type; the built-in adapter supports "
                    "the Qwen2.5-VL family (model_type=qwen2_5_vl)"
                )
            index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
            shards = sorted(set((index.get("weight_map") or {}).values()))
            if not shards:
                raise ValueError("weight map contains no shards")
            missing = [name for name in shards if not (path / name).is_file() or (path / name).stat().st_size <= 0]
            if missing:
                raise ValueError(f"missing/empty weight shards: {missing}")
            for name in ("tokenizer.json", "preprocessor_config.json"):
                if not (path / name).is_file():
                    raise FileNotFoundError(f"missing model asset: {name}")
        except Exception as exc:
            builder.add("answer_provider", False, f"Local VLM is invalid: {exc}", details={"provider": "local", "model_path": _safe_path(path)})
            return
        builder.add(
            "answer_provider", True, "Local Qwen2.5-VL assets are complete",
            details={
                "provider": "local",
                "model_path": _safe_path(path),
                "model_type": model_config.get("model_type"),
                "hidden_size": model_config.get("hidden_size"),
                "weight_shards": len(shards),
                "load_in_4bit_required_for_16gb": True,
            },
        )
        return

    dotenv_path = config.dotenv_path or (config.project_root / ".env")
    provider = provider_for("vision", load_runtime_env(dotenv_path))
    builder.add(
        "answer_provider", provider.configured,
        "Explicit OpenAI-compatible vision provider is configured" if provider.configured else "API provider is missing VLM_BASE_URL, VLM_API_KEY, or VLM_MODEL",
        details={
            "provider": "openai",
            "configuration_variables": ["VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL"],
            "configured": provider.configured,
        },
    )


def _check_output_dir(builder: ReportBuilder, config: PreflightConfig) -> None:
    path = config.output_dir
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < config.min_free_gb:
            raise OSError(f"only {free_gb:.2f} GiB free; require {config.min_free_gb:.2f} GiB")
        with tempfile.NamedTemporaryFile(prefix=".hcmai-preflight-", dir=path, delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
    except Exception as exc:
        builder.add("output_storage", False, f"Output storage is not ready: {exc}", details={"path": _safe_path(path)})
        return
    builder.add("output_storage", True, "Output directory is writable", details={"path": _safe_path(path), "free_gb": round(free_gb, 2), "minimum_free_gb": config.min_free_gb})


def _read_csv(data: bytes, label: str) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"{label}: CSV has no header")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{label}: CSV has no rows")
    return rows


def _validate_query_bytes(data: bytes, suffix: str, label: str) -> int:
    if suffix == ".csv":
        rows: Sequence[dict[str, Any]] = _read_csv(data, label)
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(data.decode("utf-8-sig"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("queries"), list):
            rows = payload["queries"]
        else:
            rows = [payload]
    else:
        raise ValueError(f"{label}: unsupported query format {suffix}")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label}: query records must be non-empty objects")
    ids: list[str] = []
    for number, row in enumerate(rows, 1):
        query_id = str(row.get("query_id", row.get("id", ""))).strip()
        if not query_id:
            raise ValueError(f"{label}: row {number} has no query_id/id")
        if not any(str(row.get(name, "")).strip() for name in ("query", "events", "event_1")):
            raise ValueError(f"{label}: row {number} has no query/events")
        ids.append(query_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate query_id")
    return len(rows)


def _iter_files(path: Path, suffixes: set[str]) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.casefold() in suffixes else []
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.casefold() in suffixes)


def _check_queries(builder: ReportBuilder, path: Path | None) -> None:
    if path is None:
        return
    try:
        text_files = _iter_files(path, {".txt"})
        if text_files:
            from src.cli.competition_run import load_query_specs
            count = len(load_query_specs(path))
            files = text_files
        else:
            files = _iter_files(path, {".csv", ".json", ".jsonl"})
            if not files:
                raise ValueError("query path contains no TXT/CSV/JSON/JSONL files")
            count = sum(_validate_query_bytes(item.read_bytes(), item.suffix.casefold(), str(item)) for item in files)
    except Exception as exc:
        builder.add("query_input", False, f"Query input is invalid: {exc}", details={"path": _safe_path(path)})
        return
    builder.add("query_input", True, "Query input is structurally valid", details={"path": _safe_path(path), "files": len(files), "queries": count})


def _canonical_answer(video: Any, frame: Any, pairs: set[tuple[str, int]] | None, label: str) -> None:
    video_id = str(video).strip().upper()
    frame_id = int(frame)
    if not video_id or frame_id < 0:
        raise ValueError(f"{label}: invalid video/frame")
    if pairs is not None and (video_id, frame_id) not in pairs:
        raise ValueError(f"{label}: non-canonical frame {video_id}/{frame_id}")


def _validate_output_json(data: bytes, label: str, pairs: set[tuple[str, int]] | None) -> int:
    payload = json.loads(data.decode("utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), dict) or not payload["queries"]:
        raise ValueError(f"{label}: JSON must contain a non-empty queries mapping")
    task = str(payload.get("task", "")).casefold()
    answers_seen = 0
    for query_id, answers in payload["queries"].items():
        if not str(query_id).strip() or not isinstance(answers, list) or not 1 <= len(answers) <= 100:
            raise ValueError(f"{label}: query {query_id!r} must have 1..100 answers")
        identities: set[tuple[Any, ...]] = set()
        event_count: int | None = None
        for answer in answers:
            if not isinstance(answer, dict):
                raise ValueError(f"{label}: answer is not an object")
            is_trake = task == "trake" or "frame_ids" in answer
            if is_trake:
                frames = answer.get("frame_ids")
                if not isinstance(frames, list) or not frames:
                    raise ValueError(f"{label}: TRAKE answer has no frame_ids")
                parsed = [int(value) for value in frames]
                if any(a >= b for a, b in zip(parsed, parsed[1:])):
                    raise ValueError(f"{label}: TRAKE frame_ids are not strictly increasing")
                if event_count is None:
                    event_count = len(parsed)
                elif len(parsed) != event_count:
                    raise ValueError(f"{label}: TRAKE answers disagree on event count")
                for frame in parsed:
                    _canonical_answer(answer.get("video_id"), frame, pairs, label)
                identity = (str(answer.get("video_id")).strip().upper(), *parsed)
            else:
                response = str(answer.get("answer", "")).strip()
                if (
                    not response
                    or response.casefold() in {"null", "unknown", "evidence-only", "placeholder"}
                    or response.casefold().startswith(("không tìm thấy", "khong tim thay"))
                ):
                    raise ValueError(f"{label}: Q&A answer is empty/placeholder")
                _canonical_answer(answer.get("video_id"), answer.get("frame_id"), pairs, label)
                identity = (str(answer.get("video_id")).strip().upper(), int(answer.get("frame_id")))
            if identity in identities:
                raise ValueError(f"{label}: duplicate ranked answer for query {query_id!r}")
            identities.add(identity)
            answers_seen += 1
    return answers_seen


def _validate_output_csv(data: bytes, label: str, pairs: set[tuple[str, int]] | None) -> int:
    rows = _read_csv(data, label)
    columns = set(rows[0])
    if {"query_id", "video_name", "frame_idx"} <= columns:
        for row in rows:
            _canonical_answer(row["video_name"], row["frame_idx"], pairs, label)
    elif {"query_id", "video_id", "frame_id", "answer"} <= columns:
        for row in rows:
            if not str(row["answer"]).strip():
                raise ValueError(f"{label}: empty Q&A answer")
            _canonical_answer(row["video_id"], row["frame_id"], pairs, label)
    elif {"query_id", "video_id", "frame_ids"} <= columns:
        for row in rows:
            frames = json.loads(row["frame_ids"])
            if not isinstance(frames, list) or not frames or any(int(a) >= int(b) for a, b in zip(frames, frames[1:])):
                raise ValueError(f"{label}: invalid TRAKE frame_ids")
            for frame in frames:
                _canonical_answer(row["video_id"], frame, pairs, label)
    else:
        raise ValueError(f"{label}: unknown submission CSV schema")
    counts: dict[str, int] = {}
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"{label}: empty query_id")
        counts[query_id] = counts.get(query_id, 0) + 1
        video = str(row.get("video_id", row.get("video_name", ""))).strip().upper()
        frames = str(row.get("frame_ids", row.get("frame_id", row.get("frame_idx", ""))))
        identity = (query_id, video, frames)
        if identity in identities:
            raise ValueError(f"{label}: duplicate ranked answer for query {query_id!r}")
        identities.add(identity)
    if any(count > 100 for count in counts.values()):
        raise ValueError(f"{label}: more than 100 answers for one query")
    return len(rows)


def _validate_output_bytes(data: bytes, suffix: str, label: str, pairs: set[tuple[str, int]] | None) -> int:
    if suffix == ".json":
        return _validate_output_json(data, label, pairs)
    if suffix == ".csv":
        return _validate_output_csv(data, label, pairs)
    raise ValueError(f"{label}: unsupported submission format {suffix}")


_OFFICIAL_MEMBER = re.compile(
    r"^query-[A-Za-z0-9._-]+-(kis|qa|trake)\.csv$",
    flags=re.IGNORECASE,
)


def _validate_official_member(data: bytes, label: str, task: str,
                              pairs: set[tuple[str, int]] | None) -> int:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: official CSV is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text)))
    if not 1 <= len(rows) <= 100:
        raise ValueError(f"{label}: official query CSV requires 1..100 rows")
    expected_width: int | None = None
    identities: set[tuple[Any, ...]] = set()
    for rank, row in enumerate(rows, 1):
        if task == "kis":
            if len(row) != 2:
                raise ValueError(f"{label}: KIS row {rank} must have two columns")
            video_id, frame = row
            _canonical_answer(video_id, frame, pairs, label)
            identity = (video_id.strip().upper(), int(frame))
        elif task == "qa":
            if len(row) != 3:
                raise ValueError(f"{label}: Q&A row {rank} must have three columns")
            video_id, frame, answer = row
            answer = answer.strip()
            if (
                not answer or len(answer) > 100
                or answer.casefold() in {"null", "unknown", "evidence-only", "placeholder"}
                or answer.casefold().startswith(("không tìm thấy", "khong tim thay"))
            ):
                raise ValueError(f"{label}: Q&A row {rank} has an invalid answer")
            _canonical_answer(video_id, frame, pairs, label)
            identity = (video_id.strip().upper(), int(frame), answer)
        else:
            if len(row) < 3:
                raise ValueError(f"{label}: TRAKE row {rank} requires video plus at least two frames")
            if expected_width is None:
                expected_width = len(row)
            elif len(row) != expected_width:
                raise ValueError(f"{label}: TRAKE rows disagree on event count")
            video_id, raw_frames = row[0], row[1:]
            try:
                frames = [int(value) for value in raw_frames]
            except ValueError as exc:
                raise ValueError(f"{label}: TRAKE row {rank} has a non-integer frame") from exc
            if any(left >= right for left, right in zip(frames, frames[1:])):
                raise ValueError(f"{label}: TRAKE row {rank} is not strictly increasing")
            for frame in frames:
                _canonical_answer(video_id, frame, pairs, label)
            identity = (video_id.strip().upper(), *frames)
        if identity in identities:
            raise ValueError(f"{label}: duplicate ranked row")
        identities.add(identity)
    return len(rows)


def _validate_package_file(path: Path, pairs: set[tuple[str, int]] | None) -> tuple[int, int]:
    if path.suffix.casefold() != ".zip":
        return 1, _validate_output_bytes(path.read_bytes(), path.suffix.casefold(), str(path), pairs)
    files = answers = 0
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if not members:
            raise ValueError(f"{path}: empty ZIP")
        official = any(Path(item.filename).parts[:1] == ("submission",) for item in members)
        seen_names: set[str] = set()
        for member in members:
            pure = Path(member.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"{path}: unsafe ZIP member {member.filename!r}")
            normalized_name = member.filename.casefold()
            if normalized_name in seen_names:
                raise ValueError(f"{path}: duplicate ZIP member {member.filename!r}")
            seen_names.add(normalized_name)
            suffix = pure.suffix.casefold()
            if member.file_size > 100 * 1024 * 1024:
                raise ValueError(f"{path}: oversized member {member.filename!r}")
            if official:
                if len(pure.parts) != 2 or pure.parts[0] != "submission":
                    raise ValueError(
                        f"{path}: official ZIP member must be directly under submission/: "
                        f"{member.filename!r}"
                    )
                match = _OFFICIAL_MEMBER.fullmatch(pure.name)
                if not match:
                    raise ValueError(f"{path}: invalid official member name {member.filename!r}")
                answers += _validate_official_member(
                    archive.read(member), f"{path}:{member.filename}",
                    match.group(1).casefold(), pairs,
                )
            else:
                if suffix not in {".json", ".csv"}:
                    continue
                answers += _validate_output_bytes(
                    archive.read(member), suffix, f"{path}:{member.filename}", pairs,
                )
            files += 1
    if files == 0:
        raise ValueError(f"{path}: ZIP contains no submission JSON/CSV")
    return files, answers


def _check_output_package(builder: ReportBuilder, path: Path | None, pairs: set[tuple[str, int]] | None) -> None:
    if path is None:
        return
    try:
        files = _iter_files(path, {".json", ".csv", ".zip"})
        if not files:
            raise ValueError("output package contains no JSON/CSV/ZIP")
        validated_files = answers = 0
        for item in files:
            count, rows = _validate_package_file(item, pairs)
            validated_files += count
            answers += rows
    except Exception as exc:
        builder.add("output_package", False, f"Output package is invalid: {exc}", details={"path": _safe_path(path)})
        return
    builder.add("output_package", True, "Output package contract is valid", details={"path": _safe_path(path), "files": validated_files, "answers": answers, "canonical_validated": pairs is not None})


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            output[key] = "<redacted>" if any(marker in str(key).casefold() for marker in SENSITIVE_MARKERS) else _redact(item)
        return output
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def run_preflight(config: PreflightConfig) -> dict[str, Any]:
    builder = ReportBuilder()
    _check_python(builder, config)
    _check_gpu(builder, config.require_gpu)
    pairs = _check_canonical(builder, config, collect_pairs=config.output_package is not None)
    _check_keyframes(builder, config)
    _check_visual(builder, config)
    _check_visual_backbones(builder, config)
    _check_modality(builder, "asr", config.asr_dir, config.expected_video_count, config.active_video_prefixes)
    _check_modality(builder, "ocr", config.ocr_dir, config.expected_video_count, config.active_video_prefixes)
    _check_provider(builder, config)
    _check_output_dir(builder, config)
    _check_queries(builder, config.query_path)
    _check_output_package(builder, config.output_package, pairs)
    blockers = [item["name"] for item in builder.checks if item["blocking"]]
    warnings = [item["name"] for item in builder.checks if item["status"] == "warning"]
    report = {
        "schema": "hcmai.competition_preflight.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready": not blockers,
        "exit_code": 0 if not blockers else 1,
        "network_calls": 0,
        "provider": config.provider,
        "active_video_prefixes": list(config.active_video_prefixes),
        "project_root": _safe_path(config.project_root),
        "blockers": blockers,
        "warnings": warnings,
        "checks": builder.checks,
    }
    return _redact(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--canonical-map", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--keyframes-dir", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--visual-index", action="append", type=Path, dest="visual_indexes")
    parser.add_argument("--visual-map", action="append", type=Path, dest="visual_maps")
    parser.add_argument("--visual-backbone-dir", action="append", type=Path, dest="visual_backbone_dirs")
    parser.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--provider", choices=("local", "openai"), default=os.getenv("VQA_ANSWER_PROVIDER", "local").casefold())
    parser.add_argument("--local-model", type=Path, default=Path(os.getenv("HCMAI_LOCAL_VLM_PATH", str(DEFAULT_MODEL))))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-path", type=Path)
    parser.add_argument("--output-package", type=Path)
    parser.add_argument(
        "--active-video-prefix", action="append", dest="active_video_prefixes",
        help="Restrict preflight to installed video-id prefix(es); defaults to HCMAI_ACTIVE_VIDEO_PREFIXES",
    )
    parser.add_argument("--expected-video-count", type=int, default=None, help="Use 0 to disable the corpus-size assertion")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--require-modality-runtime", action="store_true",
        help="Also import-probe sentence-transformers for live ASR/OCR query routing",
    )
    parser.add_argument("--dotenv", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.min_free_gb < 0:
        raise SystemExit("--min-free-gb must be non-negative")
    if args.expected_video_count is not None and args.expected_video_count < 0:
        raise SystemExit("--expected-video-count must be non-negative")
    active_video_prefixes = _normalise_active_video_prefixes(
        args.active_video_prefixes if args.active_video_prefixes is not None else DEFAULT_ACTIVE_VIDEO_PREFIXES
    )
    expected_video_count = (
        _default_expected_video_count(active_video_prefixes)
        if args.expected_video_count is None else args.expected_video_count or None
    )
    config = PreflightConfig(
        project_root=args.project_root,
        canonical_map=args.canonical_map,
        keyframes_dir=args.keyframes_dir,
        visual_indexes=tuple(args.visual_indexes or DEFAULT_VISUAL_INDEXES),
        visual_maps=tuple(args.visual_maps or DEFAULT_VISUAL_MAPS),
        visual_backbone_dirs=tuple(args.visual_backbone_dirs or DEFAULT_VISUAL_BACKBONES),
        asr_dir=args.asr_dir,
        ocr_dir=args.ocr_dir,
        provider=args.provider,
        local_model=args.local_model,
        output_dir=args.output_dir,
        query_path=args.query_path,
        output_package=args.output_package,
        active_video_prefixes=active_video_prefixes,
        expected_video_count=expected_video_count,
        min_free_gb=args.min_free_gb,
        require_gpu=args.require_gpu,
        require_modality_runtime=args.require_modality_runtime,
        dotenv_path=args.dotenv,
    )
    report = run_preflight(config)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
