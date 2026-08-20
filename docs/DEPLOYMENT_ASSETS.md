# Deployment and asset transfer

The public Git repository contains the deployable source, its offline contract
tests, configuration and operational documentation. Corpora, vector indexes,
model weights, keyframe images, caches, private annotations/holdout evidence,
submissions and experiment outputs stay outside Git. Data-backed benchmark
generators and their tests remain in the private workspace too: they cannot be
run honestly from a source-only clone without their holdout evidence.

Create a source-only handoff archive from the same Git ignore contract:

```bash
scripts/build_source_bundle.sh
```

The command writes a tarball and SHA-256 sidecar under `dist/`; runtime assets
listed below are deliberately absent.

## Required runtime assets

For the full KIS + Q&A + TRAKE offline path, copy these paths while preserving
their repository-relative layout:

- `data/index/global_keyframes.parquet`
- `data/index/global_siglip_vitl.npy`
- `data/index/global_keyframes_vitl.parquet`
- `data/index/global_so400m384.npy`
- `data/index/global_keyframes_so400m384.parquet`
- `data/index/modality_global_v2/asr_global_merged_v2/`
- `data/index/modality_global_v2/ocr_global_merged_v2/`
- `data/keyframes/` (required for Q&A/VLM image input)
- `models/bge-m3/`
- `models/Qwen2.5-VL-7B-Instruct/` when using the local Q&A provider
- Hugging Face cache snapshots for
  `timm/ViT-L-16-SigLIP2-256` and
  `timm/ViT-SO400M-16-SigLIP2-384`, or an equivalent configured `HF_HOME`

Copy `.env` separately as a secret. Never commit it or store API keys in a
shared Drive folder. Recreate `.venv` from `requirements.txt`; do not transfer
the Windows/WSL virtual environment. On WSL, place the environment in the Linux
filesystem (for example under `$HOME/.venvs`) rather than `/mnt/c`: cold
`sentence_transformers` imports on the NTFS mount can take more than 180s.
Use `.env.example` as the single configuration schema: explicit OS variables
override `.env`, and `.env` overrides code defaults for both Python and Bash
entrypoints.

Optional/rebuild-only assets include `tmp/deepgram_audio/`, old per-pack ASR/OCR
intermediates, the 3B Qwen model, Jina experimental weights, diagnostic crops,
and old experiment outputs. Deepgram credentials and raw audio are not needed
for query-time retrieval when the merged ASR index is already present.

## Google Drive bootstrap

Use an object/file transfer client such as `rclone`; do not upload hundreds of
thousands of keyframes through a browser. Configure a personal/team Drive
remote named `gdrive`, then use the repository bootstrapper rather than a
destructive `sync`:

```bash
export HCMAI_RUNTIME_REMOTE='gdrive:HCMAI-2026/runtime'

# Local-only: inspect receipts and target state; this never calls rclone.
./scripts/competition.sh bootstrap plan --artifact runtime-index

# Networked and explicit: checksum-validate then fetch without overwriting
# unmanaged data. Repeat --artifact for the desired keyframe archive groups.
./scripts/competition.sh bootstrap fetch --yes --artifact runtime-index
```

The manifest at `configs/runtime_artifacts.v1.json` defines the published
runtime index and six keyframe archive groups. A fetch writes a receipt under
`.runtime_state/`, validates remote MD5 metadata and refuses to claim or
overwrite a pre-existing unmanaged target. Models and `.env` are intentionally
not part of this manifest; provision them securely and separately.

After download on a new server, run the fail-closed structural preflight:

```bash
.venv/bin/python -m src.cli.competition_ready --provider local --require-gpu

# Use this stricter form when live ASR/OCR query routing is enabled:
.venv/bin/python -m src.cli.competition_ready --provider local --require-gpu \
  --require-modality-runtime
```
