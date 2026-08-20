# ASR global v2 materializer

`src/eval/asr_global_v2.py` materializes K01--K20 and L21--L30 ASR into a new
index. It accepts official `Videos_<pack>*.zip` archives in one archive root;
therefore a fresh server can rebuild K and L rather than depending on
historical K artifacts. It does not edit `data/asr_*`, delete ZIPs or MP4s,
change the router, or silently fall back to another provider.

## Dry-run (default)

Run from the repository root in WSL:

```bash
cd /path/to/hcmai
.venv/bin/python -m src.eval.asr_global_v2 \
  --packs all \
  --archive-root data/raw/video_archives \
  --canonical data/index/global_keyframes.parquet \
  --output-dir data/index/asr_global_v2 \
  --work-dir data/work/asr_global_v2 \
  --raw-dir data/asr_global_v2_raw
```

The command only inspects ZIP members and the canonical parquet. It writes
`data/index/asr_global_v2/asr_global_v2_manifest.json` with the archive/video
scope and does not extract media, load bge-m3, or call Deepgram.

## Explicit execution

Execution requires all three flags. This is intentionally noisy because a
full run creates API cost and a large derived corpus:

```bash
.venv/bin/python -m src.eval.asr_global_v2 \
  --packs L21 \
  --execute --allow-network --confirm-api \
  --archive-root data/raw/video_archives \
  --canonical data/index/global_keyframes.parquet \
  --output-dir data/index/asr_global_v2 \
  --work-dir data/work/asr_global_v2 \
  --raw-dir data/asr_global_v2_raw \
  --model models/bge-m3 \
  --ffmpeg .venv/bin/ffmpeg
```

Start with one pack. `L26` may be split across five ZIPs and is discovered as
one logical pack. A run is resumable: existing valid per-video raw JSON is
reused, and successful outputs are not transcribed again. The runner never
removes source ZIPs, extracted MP4s, WAVs, or historical ASR artifacts.

For a bounded smoke run, use `--max-videos 1`; its manifest remains
`scope_limited` and cannot become `ready_for_global`.

## Outputs and contract

For each completed pack:

```text
asr_chunks_<pack>_ts.parquet
emb_cache_asr_<pack>_chunks.npy
asr_global_v2_<pack>_manifest.json
```

The parquet uses `video_id`, `chunk_index`, `text`, `start`, `end`, `kf_n`,
canonical `frame_idx`, `pts_time`, and `distance_seconds`. The numpy row order
is exactly the parquet row order. The global manifest records the canonical
map hash, archive provenance, API approval, call count, embedding model/dim,
per-pack status, and `ready_for_global`/`ready_for_production` gates.

An API key can be supplied through `DEEPGRAM_API_KEY` or `.env`; it is never
printed. `--execute` without a configured key fails before extraction. A
missing archive, canonical mismatch, malformed existing transcript,
timestamp-less transcript, invalid frame mapping, or embedding row/dimension
mismatch is a hard error for that video/pack and leaves the manifest partial.

## Fake-only tests

The test suite injects fake ZIPs, ffmpeg, Deepgram, and embedder objects:

```bash
.venv/bin/python -m pytest -q tests/test_asr_global_v2.py -rxX
```

These tests do not call Deepgram and are the required verification for this
materializer change.
