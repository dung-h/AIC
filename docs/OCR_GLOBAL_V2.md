# OCR global v2

`src.eval.ocr_global_v2` is the isolated materializer for a future global OCR
index. It does not modify historical `ocr_*.parquet` files, ASR, the shared
runtime, or the router.

## Contract

Input is the canonical keyframe table with:

```text
video_id, kf_n, frame_idx, pts_time
```

The runner processes every selected canonical keyframe in `full` mode. Each
attempt is appended to `attempt_manifest.jsonl` and to its pack-local
`packs/<PACK>/attempts.jsonl`. A row has `status=text`, `status=no_text`, or
`status=error`; no-text rows are retained for coverage accounting but never
enter retrieval.

Successful packs contain:

```text
packs/<PACK>/retrieval.parquet
packs/<PACK>/embeddings.npy
packs/<PACK>/checkpoint.json
```

`retrieval.parquet` carries the canonical `frame_idx` and the cleaned OCR
text. `embeddings.npy` has exactly one normalized bge-m3 vector per parquet
row. A pack with no readable text or any unresolved frame/inference error is
`blocked` and does not produce a retrieval index.

## Safe execution

```bash
# Inspect the exact full-corpus scope; no model is loaded.
.venv/bin/python -m src.eval.ocr_global_v2 --mode full --dry-run

# Small local pilot.
.venv/bin/python -m src.eval.ocr_global_v2 --mode pilot --execute \
  --pack K01,L21 --video-limit 2

# Resume the same versioned directory after an interruption.
.venv/bin/python -m src.eval.ocr_global_v2 --mode full --execute --resume \
  --output-dir data/index/modality_global_v2/ocr
```

Full mode refuses to run without `--execute`. A dry run never checks model
files, opens images, calls Qwen, or embeds text. The default backends set
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; there is no API or network
fallback. Use a new output directory/version when changing canonical metadata,
model, embedding dimension, or scope.

## Remaining runtime requirements

Before a real full run, verify in the WSL environment:

1. local Qwen2.5-VL-3B and its transformers/qwen vision dependencies load;
2. local bge-m3 and sentence-transformers produce 1024-dimensional vectors;
3. every canonical keyframe resolves under `data/keyframes/<video_id>/<kf_n:03d>.jpg`;
4. disk space is sufficient for the append-only attempts, parquet text index,
   and embedding matrix;
5. the completed manifest passes the existing modality preflight before any
   router feature flag is changed.

The full 379,765-keyframe job is deliberately not run by the unit tests or by
this implementation task.
