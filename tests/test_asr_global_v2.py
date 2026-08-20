from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
import pytest

from src.eval.asr_global_v2 import (
    ASRGlobalV2Error,
    ASRGlobalV2Runner,
    ASRNetworkApprovalError,
    RunnerConfig,
    discover_pack_archives,
    extract_audio_default,
    inspect_archive_videos,
    iter_timestamped_chunks,
    load_canonical_map,
    main,
)


def _write_video_zip(root: Path, pack: str, video_ids: list[str], suffix: str = "a") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"Videos_{pack}_{suffix}.zip"
    with ZipFile(path, "w", compression=ZIP_STORED) as zipped:
        for video_id in video_ids:
            zipped.writestr(f"video/{video_id}.mp4", f"fake-{video_id}".encode())
    return path


def _write_canonical(path: Path, video_ids: list[str]) -> None:
    rows = []
    for video_id in video_ids:
        rows.extend(
            [
                {"video_id": video_id, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0},
                {"video_id": video_id, "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
                {"video_id": video_id, "kf_n": 3, "frame_idx": 30, "pts_time": 4.0},
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _config(tmp_path: Path, *, execute: bool = False, **kwargs) -> RunnerConfig:
    return RunnerConfig(
        archive_root=tmp_path / "archives",
        canonical_path=tmp_path / "canonical.parquet",
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        raw_dir=tmp_path / "raw",
        execute=execute,
        allow_network=kwargs.pop("allow_network", execute),
        confirm_api=kwargs.pop("confirm_api", execute),
        **kwargs,
    )


def test_discovery_includes_all_split_l26_archives(tmp_path: Path) -> None:
    root = tmp_path / "archives"
    _write_video_zip(root, "L21", ["L21_V001"])
    _write_video_zip(root, "L26", ["L26_V001"], "a")
    _write_video_zip(root, "L26", ["L26_V002"], "b")
    discovered = discover_pack_archives(root, ["L21", "L26"])
    assert [path.name for path in discovered["L26"].paths] == ["Videos_L26_a.zip", "Videos_L26_b.zip"]
    assert len(inspect_archive_videos(discovered["L26"])) == 2


def test_discovery_accepts_k_and_l_official_archives(tmp_path: Path) -> None:
    root = tmp_path / "archives"
    _write_video_zip(root, "K01", ["K01_V001"])
    _write_video_zip(root, "L21", ["L21_V001"])
    discovered = discover_pack_archives(root, ["K01", "L21"])
    assert sorted(discovered) == ["K01", "L21"]


def test_timestamped_chunks_are_deduplicated_and_sorted() -> None:
    chunks = list(
        iter_timestamped_chunks(
            {
                "results": {
                    "utterances": [
                        {"transcript": "B", "start": 2, "end": 3},
                        {"transcript": "A", "start": 1, "end": 2},
                        {"transcript": "A", "start": 1, "end": 2},
                    ]
                }
            }
        )
    )
    assert chunks == [
        {"text": "A", "start": 1.0, "end": 2.0},
        {"text": "B", "start": 2.0, "end": 3.0},
    ]


def test_default_run_is_dry_run_and_does_not_extract_or_call_provider(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    _write_video_zip(archive_root, "L21", ["L21_V001"])
    canonical = tmp_path / "canonical.parquet"
    _write_canonical(canonical, ["L21_V001"])
    config = _config(tmp_path)
    runner = ASRGlobalV2Runner(config)
    report = runner.run(["L21"])
    assert report["status"] == "dry_run"
    assert isinstance(report["preflight"]["deepgram_api_key_configured"], bool)
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "raw").exists()
    assert not list((tmp_path / "output").glob("*.parquet"))
    assert (tmp_path / "output" / "asr_global_v2_manifest.json").is_file()


def test_real_execution_requires_two_explicit_network_approvals() -> None:
    with pytest.raises(ASRNetworkApprovalError, match="--allow-network --confirm-api"):
        RunnerConfig(
            archive_root=Path("archives"),
            canonical_path=Path("canonical.parquet"),
            output_dir=Path("output"),
            work_dir=Path("work"),
            raw_dir=Path("raw"),
            execute=True,
            allow_network=True,
            confirm_api=False,
        ).validate()


def test_extract_audio_temp_path_keeps_wav_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"fake")
    wav = tmp_path / "clip.wav"
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"fake")
    seen: list[list[str]] = []

    class _Completed:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(command, **kwargs):
        seen.append(list(command))
        Path(command[-1]).write_bytes(b"RIFF" + b"0" * 100)
        return _Completed()

    monkeypatch.setattr("src.eval.asr_global_v2.subprocess.run", fake_run)
    extract_audio_default(mp4, wav, ffmpeg)

    assert wav.is_file()
    assert seen
    assert seen[0][-1].endswith(".part.wav")


class _FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(self, wav_path: Path) -> dict:
        self.calls.append(wav_path)
        return {
            "results": {
                "utterances": [
                    {"transcript": "Nha Trang", "start": 0.5, "end": 1.5},
                    {"transcript": "hai mươi lăm độ", "start": 3.0, "end": 3.5},
                ]
            }
        }


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts, batch_size=32, normalize=True):
        self.calls.append(list(texts))
        return np.arange(len(texts) * 4, dtype=np.float32).reshape(len(texts), 4) + 1


def test_execute_is_resumable_and_materializes_canonical_frame_idx(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    zip_path = _write_video_zip(archive_root, "L21", ["L21_V001"])
    canonical = tmp_path / "canonical.parquet"
    _write_canonical(canonical, ["L21_V001"])
    transcriber = _FakeTranscriber()
    embedder = _FakeEmbedder()
    ffmpeg_calls: list[tuple[Path, Path]] = []

    def fake_ffmpeg(mp4: Path, wav: Path) -> None:
        ffmpeg_calls.append((mp4, wav))
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF" + b"0" * 100)

    config = _config(tmp_path, execute=True)
    runner = ASRGlobalV2Runner(
        config,
        transcriber=transcriber,
        embedder=embedder,
        ffmpeg_runner=fake_ffmpeg,
    )
    first = runner.run(["L21"])
    assert first["status"] == "partial"  # only one selected pack; global scope is incomplete
    assert first["packs"]["L21"]["status"] == "complete"
    assert first["packs"]["L21"]["completed_videos"] == 1
    assert first["packs"]["L21"]["embeddings_materialized"] is True
    assert transcriber.calls == [tmp_path / "work" / "l21" / "audio" / "L21_V001.wav"]
    assert len(embedder.calls) == 1
    chunks = pd.read_parquet(tmp_path / "output" / "asr_chunks_l21_ts.parquet")
    assert chunks["frame_idx"].tolist() == [10, 30]
    embeddings = np.load(tmp_path / "output" / "emb_cache_asr_l21_chunks.npy")
    assert embeddings.shape == (2, 4)
    assert zip_path.is_file()  # source ZIP remains untouched

    second = runner.run(["L21"])
    assert second["packs"]["L21"]["status"] == "complete"
    assert len(transcriber.calls) == 1  # raw JSON was reused; no second API/provider call
    assert len(ffmpeg_calls) == 1  # resumed raw JSON also avoids re-extracting audio


def test_embedder_is_reused_across_pack_materialization(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    _write_video_zip(archive_root, "L21", ["L21_V001"])
    _write_video_zip(archive_root, "L22", ["L22_V001"])
    canonical = tmp_path / "canonical.parquet"
    rows = []
    for video_id in ("L21_V001", "L22_V001"):
        rows.extend(
            [
                {"video_id": video_id, "kf_n": 1, "frame_idx": 10, "pts_time": 0.0},
                {"video_id": video_id, "kf_n": 2, "frame_idx": 20, "pts_time": 2.0},
            ]
        )
    pd.DataFrame(rows).to_parquet(canonical, index=False)
    embedder = _FakeEmbedder()
    config = _config(tmp_path, execute=True)
    runner = ASRGlobalV2Runner(
        config,
        transcriber=_FakeTranscriber(),
        embedder=embedder,
        ffmpeg_runner=lambda _mp4, wav: wav.write_bytes(b"RIFF" + b"0" * 100),
    )

    runner.run(["L21", "L22"])

    assert runner._embedder_instance is embedder
    assert len(embedder.calls) == 2


def test_resume_skips_complete_packs_without_provider_calls(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    _write_video_zip(archive_root, "L21", ["L21_V001"])
    _write_video_zip(archive_root, "L22", ["L22_V001"])
    canonical = tmp_path / "canonical.parquet"
    _write_canonical(canonical, ["L21_V001", "L22_V001"])

    def fake_ffmpeg(_mp4: Path, wav: Path) -> None:
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF" + b"0" * 100)

    first_transcriber = _FakeTranscriber()
    first = ASRGlobalV2Runner(
        _config(tmp_path, execute=True),
        transcriber=first_transcriber,
        embedder=_FakeEmbedder(),
        ffmpeg_runner=fake_ffmpeg,
    )
    first.run(["L21", "L22"])
    assert len(first_transcriber.calls) == 2

    second_transcriber = _FakeTranscriber()
    resumed = ASRGlobalV2Runner(
        _config(tmp_path, execute=True, resume=True),
        transcriber=second_transcriber,
        embedder=_FakeEmbedder(),
        ffmpeg_runner=fake_ffmpeg,
    )
    report = resumed.run(["L21", "L22"])
    assert report["packs"]["L21"]["status"] == "complete"
    assert report["packs"]["L22"]["status"] == "complete"
    assert second_transcriber.calls == []


def test_invalid_existing_raw_is_quarantined_and_retried(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    _write_video_zip(archive_root, "L21", ["L21_V001"])
    canonical = tmp_path / "canonical.parquet"
    _write_canonical(canonical, ["L21_V001"])
    raw_path = tmp_path / "raw" / "l21" / "L21_V001.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"error": "provider timeout"}), encoding="utf-8")

    transcriber = _FakeTranscriber()
    runner = ASRGlobalV2Runner(
        _config(tmp_path, execute=True),
        transcriber=transcriber,
        embedder=_FakeEmbedder(),
        ffmpeg_runner=lambda _mp4, wav: wav.write_bytes(b"RIFF" + b"0" * 100),
    )
    runner.run(["L21"])

    assert transcriber.calls == [tmp_path / "work" / "l21" / "audio" / "L21_V001.wav"]
    assert raw_path.with_name("L21_V001.json.invalid").is_file()
    assert list(iter_timestamped_chunks(json.loads(raw_path.read_text(encoding="utf-8"))))


def test_valid_no_speech_response_completes_without_chunk_rows(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    _write_video_zip(archive_root, "L21", ["L21_V001"])
    canonical = tmp_path / "canonical.parquet"
    _write_canonical(canonical, ["L21_V001"])

    class _NoSpeech:
        calls = 0

        def transcribe(self, wav_path: Path) -> dict:
            self.calls += 1
            return {"results": {"channels": [{"alternatives": []}], "utterances": []}}

    transcriber = _NoSpeech()
    report = ASRGlobalV2Runner(
        _config(tmp_path, execute=True),
        transcriber=transcriber,
        embedder=_FakeEmbedder(),
        ffmpeg_runner=lambda _mp4, wav: wav.write_bytes(b"RIFF" + b"0" * 100),
    ).run(["L21"])
    assert report["packs"]["L21"]["status"] == "complete"
    assert report["packs"]["L21"]["no_speech_videos"] == ["L21_V001"]
    assert report["packs"]["L21"]["rows"] == 0


def test_archive_canonical_mismatch_fails_closed(tmp_path: Path) -> None:
    _write_video_zip(tmp_path / "archives", "L21", ["L21_V001"])
    _write_canonical(tmp_path / "canonical.parquet", ["L21_V002"])
    with pytest.raises(ASRGlobalV2Error, match="canonical/archive mismatch"):
        ASRGlobalV2Runner(_config(tmp_path)).preflight(["L21"])


def test_cli_default_is_dry_run(tmp_path: Path) -> None:
    _write_video_zip(tmp_path / "archives", "L21", ["L21_V001"])
    _write_canonical(tmp_path / "canonical.parquet", ["L21_V001"])
    code = main(
        [
            "--packs",
            "L21",
            "--archive-root",
            str(tmp_path / "archives"),
            "--canonical",
            str(tmp_path / "canonical.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--work-dir",
            str(tmp_path / "work"),
            "--raw-dir",
            str(tmp_path / "raw"),
        ]
    )
    assert code == 0
    manifest = json.loads((tmp_path / "output" / "asr_global_v2_manifest.json").read_text())
    assert manifest["mode"] == "dry_run"
