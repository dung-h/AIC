# HCMAI 2026 — Multimedia Retrieval

Mã nguồn cho hệ thống truy xuất video AI Challenge HCMC. Git chỉ giữ code,
cấu hình, test và tài liệu vận hành. Corpus, keyframes, vector index, model
weights, raw audio/video, cache, annotations/holdout nội bộ, submission và kết
quả thí nghiệm **không** được public.

> Trạng thái trung thực: contract và deployment flow KIS/Q&A/TRAKE đã chạy được
> và có preflight fail-closed. Điều đó không phải cam kết accuracy thi đấu của
> Q&A/TRAKE; mọi promotion model/reranker phải đo trên benchmark độc lập.

## Architecture

```text
official map CSV ──> canonical keyframe map ──────────────────────────────┐
keyframe images ──> visual embeddings ───────────────────────────────────┼─> global retrieval
audio/video ──> ASR chunks + bge-m3 embeddings ───────────────────────────┤       │
keyframe images ──> OCR text + bge-m3 embeddings ─────────────────────────┘       │
                                                                                     ▼
                                    KIS: visual fusion | Q&A: routing → allocator → VLM
                                                      | TRAKE: event scores → DANTE
                                                                                     ▼
                                                   canonical validator → competition serializer
```

Every stage preserves canonical `video_id`, `kf_n`, `frame_idx`, and
`pts_time`. Submission uses `frame_idx`, never a keyframe ordinal.

| Component | Owner | Output invariant |
|---|---|---|
| Canonical identity | `src/indexing/` | `(video_id,kf_n)` maps to exactly one `frame_idx/pts_time` |
| Visual retrieval | `src/pipelines/kis_*` | ranked canonical frame candidates |
| Q&A | `src/pipelines/vqa_pipeline_v3.py` | `{video_id, frame_id, answer}`; no blank/evidence-only answer |
| TRAKE | `src/pipelines/trake_*`, `src/trake/` | one strictly increasing canonical frame per event |
| Runtime boundary | `src/cli/`, `scripts/competition.sh` | fail-closed assets and valid serialized payload |

## Clone versus runtime data

A source clone is deliberately small and cannot answer queries by itself. A
full offline runtime needs these assets in the repository layout:

```text
data/index/
  global_keyframes.parquet
  global_siglip_vitl.npy + global_keyframes_vitl.parquet
  global_so400m384.npy + global_keyframes_so400m384.parquet
  modality_global_v2/asr_global_merged_v2/
  modality_global_v2/ocr_global_merged_v2/
data/keyframes/<video_id>/<kf_n>.jpg
models/bge-m3/
models/Qwen2.5-VL-7B-Instruct/          # local Q&A answer provider
Hugging Face cache for the visual backbones
```

The published Drive manifest bootstraps `data/index` and six keyframe archive
groups. It intentionally does not include VLM/bge weights or secrets. A
partial model directory is invalid; preflight reports it rather than falling
back silently.

## Fresh Linux/WSL server

Create the virtualenv on a native Linux filesystem, never under `/mnt/c`.

```bash
git clone https://github.com/dung-h/AIC.git hcmai
cd hcmai

python3 -m venv /opt/hcmai-venv
source /opt/hcmai-venv/bin/activate
python -m pip install --upgrade pip

# Select the appropriate CUDA wheel for the server before requirements.txt.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

export HCMAI_PYTHON=/opt/hcmai-venv/bin/python
export HCMAI_LOCAL_VLM_PATH=/opt/hcmai-models/Qwen2.5-VL-7B-Instruct
export VQA_MODALITY_MODEL_DIR=/opt/hcmai-models/bge-m3
export HF_HOME=/opt/hcmai-models/huggingface
export HF_HUB_CACHE=/opt/hcmai-models/huggingface/hub
export HCMAI_RUNTIME_REMOTE='gdrive:HCMAI-2026/runtime'
```

## Configuration contract

Copy `.env.example` to a private `.env` when this server needs local path
overrides or an explicit remote provider. Never commit the resulting file.
Every supported Python entrypoint and Bash wrapper reads the same safe
`KEY=VALUE` format; exported environment variables override `.env`, which in
turn overrides code defaults. `.env` is never sourced as shell code.

```bash
cp .env.example .env
# Edit only the fields this server needs; keep API keys private.
```

For a local-only server, the only required `.env` fields are the four
machine-specific paths above (or equivalent exported variables). Do **not**
copy an old `.env` by Git or Drive. Recreate it from `.env.example`, then move
only needed API keys via a secret manager or a protected SSH copy. The rclone
configuration also contains a refresh token: configure a new `gdrive` remote
on the server or transfer it through the same protected channel, never Git.

Remote capabilities are role-separated: `TEXT_*` for rewriting, `VLM_*` for
image-aware answering/reranking, and `EMBEDDING_*` for remote text embeddings.
The offline competition default needs no API key. Provider configuration
accepts only these role-specific keys; unsupported names are never translated
or used as provider fallbacks.

### Optional external fact grounding for Q&A

`VQA_EXTERNAL_GROUNDING` is an integrated **query-planning** capability, not a
source of submitted answers. When explicitly enabled for an online/API run,
the allow-listed SearXNG source produces aliases, quotation variants and entity
hypotheses. The pipeline then retrieves raw evidence from the local ASR/OCR
indexes, joins it to the video timeline, and validates the final canonical
`frame_idx`. A web snippet can never fill `answer` or `frame_id` by itself.

It is disabled by default and rejected in offline or benchmark-strict runs.
To exercise it, configure `VQA_EXTERNAL_ALLOWED_DOMAINS` in private `.env`.
The default `VQA_EXTERNAL_SEARCH_BACKEND=searxng` additionally needs
`VQA_EXTERNAL_SEARCH_URL`; use the explicit `ddg` backend when a self-hosted
SearXNG service is unavailable (it uses the pinned `ddgs` package). Then opt
in per run:

```bash
./scripts/competition.sh run ... --external-grounding
```

Use that option only if outbound lookup is permitted for the target deployment;
the normal submission command omits it and remains corpus-local.

For rare visual entities (logos, brands, mascots, toys), the independent
`VQA_EXTERNAL_IMAGE_GROUNDING` branch implements the AIC-2025 QUEST pattern:
SearXNG returns bounded web-image references, each reference is encoded by the
local VKIS index, and only the resulting in-corpus video/frame candidates are
allowed into Q&A. Configure `VQA_EXTERNAL_IMAGE_ALLOWED_DOMAINS`, or make the
broader download decision explicitly with `VQA_EXTERNAL_IMAGE_ALLOW_ANY_HOST=1`.
Then opt in independently:

```bash
./scripts/competition.sh run ... --external-image-grounding
```

The image branch automatically skips factual/ASR/OCR-contracted questions;
it is reserved for visual OOK entities and records every source page in the
runtime trace.

## Provision runtime assets

Choose exactly one transfer path. Both paths preserve the same local runtime
layout and end with the same `preflight` command.

### Private Drive with rclone

Configure a personal/team `rclone` remote named `gdrive`, then inspect the
local state first. `plan` makes no network call. `fetch --yes` is explicit,
uses checksum validation and refuses to overwrite unmanaged data.

```bash
./scripts/competition.sh bootstrap plan --artifact runtime-index
./scripts/competition.sh bootstrap fetch --yes --artifact runtime-index

./scripts/competition.sh bootstrap fetch --yes \
  --artifact keyframes-k01-k05 --artifact keyframes-k06-k10 \
  --artifact keyframes-k11-k15 --artifact keyframes-k16-k20 \
  --artifact keyframes-l21-l25 --artifact keyframes-l26-l30

# Model files are separate from the data manifest so they can live outside the
# source checkout. The current Drive remote contains these exact directories.
for model in bge-m3 Qwen2.5-VL-7B-Instruct; do
  rclone copy --checksum --progress \
    "$HCMAI_RUNTIME_REMOTE/models/$model" "/opt/hcmai-models/$model"
done

# Hydrate the two visual backbones once. This is setup-time only; switch back
# to offline mode before any benchmark or competition run.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  --cache-dir "$HF_HUB_CACHE" timm/ViT-L-16-SigLIP2-256
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  --cache-dir "$HF_HUB_CACHE" timm/ViT-SO400M-16-SigLIP2-384
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

./scripts/competition.sh preflight --provider local --require-gpu \
  --require-modality-runtime --json-output results/preflight.json
```

### Public Google Drive, no credential on the server

Do not publish the top-level `runtime` folder: it contains a legacy raw
`data/keyframes` tree. Instead, publish exactly these three child folders and
copy their three Google Drive *folder* links:

```text
data/index
data/keyframe_archives_v2
models
```

Set the non-secret links and runtime paths in the server's `.env`:

```text
HCMAI_PYTHON=/opt/hcmai-venv/bin/python
HCMAI_LOCAL_VLM_PATH=/opt/hcmai-models/Qwen2.5-VL-7B-Instruct
VQA_MODALITY_MODEL_DIR=/opt/hcmai-models/bge-m3
HF_HOME=/opt/hcmai-models/huggingface
HF_HUB_CACHE=/opt/hcmai-models/huggingface/hub
HCMAI_PUBLIC_INDEX_URL=https://drive.google.com/drive/folders/<index-id>
HCMAI_PUBLIC_KEYFRAMES_URL=https://drive.google.com/drive/folders/<archive-id>
HCMAI_PUBLIC_MODELS_URL=https://drive.google.com/drive/folders/<models-id>
HCMAI_PUBLIC_MODEL_ROOT=/opt/hcmai-models
HCMAI_PUBLIC_DOWNLOAD_ROOT=/opt/hcmai-downloads
# Optional; this is already the default for the preselection runtime.
HCMAI_PUBLIC_KEYFRAME_ARCHIVES=keyframes-L21-L25.tar,keyframes-L26-L30.tar
# Match retrieval to the installed archive packs. Remove this only after all
# K-series archives have also been installed.
HCMAI_ACTIVE_VIDEO_PREFIXES=L
```

If the server user is not root, grant it ownership of the two external asset
directories once:

```bash
sudo install -d -o "$USER" -g "$USER" /opt/hcmai-models /opt/hcmai-downloads
```

Then the public bootstrap has no OAuth token, rclone config, or API key:

```bash
./scripts/competition.sh public-bootstrap plan
./scripts/competition.sh public-bootstrap fetch --yes
```

The default is deliberately minimal for the preselection corpus: it downloads
only `keyframes-L21-L25.tar` and `keyframes-L26-L30.tar`, plus `bge-m3` and
`Qwen2.5-VL-7B-Instruct`. It does **not** download the raw `data/keyframes`
Drive tree, the four K-series archives, or Qwen 3B. It validates archive paths,
extracts only the selected packs to `data/keyframes`, and refuses to overwrite
an already present pack. `HCMAI_ACTIVE_VIDEO_PREFIXES=L` also filters the
global visual/ASR/OCR retrieval lanes before ranking, so K-series rows cannot
consume a candidate slot without an installed image. The tool stores only local SHA-256 receipts because a
public Drive folder does not expose the authenticated rclone MD5 manifest. Run
preflight afterward; this validates the real index/model/frame contracts. The
public bootstrap uses [gdown](https://github.com/wkentaro/gdown), which
supports public Drive folders and resumes partial downloads.

To install only additional K-series packs later, without redownloading the
index or models, select the keyframe asset explicitly. Repeat `--archive` for
each needed pack:

```bash
./scripts/competition.sh public-bootstrap fetch --yes --asset keyframes \
  --archive keyframes-K01-K05.tar
```

After all four K archives are installed, remove `HCMAI_ACTIVE_VIDEO_PREFIXES`
from `.env` (or set `HCMAI_ACTIVE_VIDEO_PREFIXES=L,K`) before `preflight`.

Finally hydrate the two visual backbones once, then return to offline mode:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$HCMAI_PYTHON" -c \
  'import os; from huggingface_hub import snapshot_download; snapshot_download("timm/ViT-L-16-SigLIP2-256", cache_dir=os.environ["HF_HUB_CACHE"])'
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$HCMAI_PYTHON" -c \
  'import os; from huggingface_hub import snapshot_download; snapshot_download("timm/ViT-SO400M-16-SigLIP2-384", cache_dir=os.environ["HF_HUB_CACHE"])'

./scripts/competition.sh preflight --provider local --require-gpu \
  --require-modality-runtime --json-output results/preflight.json
```

The Drive bootstrap is intentionally not called from `run` or `preflight`:
automatically downloading a large runtime during a competition request is
unsafe. On a new server the operator performs one explicit, selective fetch;
thereafter `run` is offline-first.

### Private API profile for a remote server

The public bootstrap does not need a secret. To use the already configured
Lightning VLM for Q&A on a remote server, generate a minimal **private** dotenv
locally; the file is written under ignored `dist/` and is mode `0600`:

```bash
.venv/bin/python scripts/export_remote_env.py --answer-provider openai
```

It contains only the Lightning VLM configuration required by the API answer
provider, plus the public L-series bootstrap configuration. Transfer it only
over an authenticated channel, place it at `<remote-clone>/.env`, and keep its
mode at `0600`. Do not commit it. Deepgram is not required to query the
materialized ASR index; add it only for an authorized ASR rebuild:

```bash
.venv/bin/python scripts/export_remote_env.py --answer-provider openai \
  --include-deepgram
```

## Competition commands

```bash
./scripts/competition.sh preflight [options]
./scripts/competition.sh bootstrap plan|fetch [options]
./scripts/competition.sh run --queries QUERY.json --output OUT.json \
  --answer-provider local --topk 20 --max-vlm-candidates 20
```

An all-task smoke input is included, but needs the real runtime assets:

```bash
./scripts/competition.sh run \
  --queries tests/fixtures/competition_all_tasks_smoke.json \
  --output results/submissions/smoke.json \
  --answer-provider local --topk 2 --max-vlm-candidates 1
```

`--answer-provider openai` is explicit opt-in after a VLM provider is
configured in `.env`; API usage is not a dependency of the local path. The
remote adapter is OpenAI-compatible, so Lightning Model API can be configured
with `VLM_BASE_URL=https://lightning.ai/api/v1`, a private Lightning key in
`VLM_API_KEY`, and a vision-capable `VLM_MODEL` such as `openai/gpt-4o`.

## Rebuild indexes on a stronger server

Yes. The source contains portable materializers. Keep a fixed canonical map,
write each new encoder to a distinct variant, validate it, benchmark against
the baseline, and only then change a production default.

### 1. Build canonical mapping from official map CSVs

The official CSV map is necessary: keyframe images alone cannot prove the
submission `frame_idx`.

```bash
python -m src.indexing.build_canonical_map \
  --map-root data/raw/map-keyframes-aic25-b1 \
  --map-root data/raw/map-keyframes-b2 \
  --require-keyframes-root data/keyframes \
  --output data/index/global_keyframes.parquet
```

This writes a checksumed `*.manifest.json` beside the parquet and refuses
duplicate/non-monotonic mappings.

### 2. Rebuild visual embeddings

The encoder checkpoints by pack and is resumable. Its `vitl` and `so400m384`
variants write the exact artifact names consumed by the runtime.

```bash
# Build the ViT-L production-shaped artifacts from the canonical map.
python src/encode/encode_full_index.py \
  --model ViT-L-16-SigLIP2-256 --pretrained webli --out-name vitl \
  --canonical data/index/global_keyframes.parquet --batch 128 --workers 8
python src/encode/encode_full_index.py --merge --out-name vitl \
  --model ViT-L-16-SigLIP2-256 --pretrained webli \
  --canonical data/index/global_keyframes.parquet

# A second model uses an independent, versioned output directory first.
python src/encode/encode_full_index.py \
  --model ViT-SO400M-16-SigLIP2-384 --pretrained webli --out-name so400m384_v2 \
  --canonical data/index/global_keyframes.parquet --batch 128 --workers 8
python src/encode/encode_full_index.py --merge --out-name so400m384_v2
```

The merge emits a model/dimension/checksum manifest. Missing keyframe images
are a hard error, not silently replaced by zero vectors. First-time weight
download is allowed only while building; competition runtime defaults offline.

### 3. Rebuild ASR and text embeddings

`asr_global_v2` accepts both official K01–K20 and L21–L30 `Videos_<pack>*.zip`
archives in one directory. It maps timestamped chunks to canonical frames and
embeds them with local bge-m3. A live Deepgram transcription requires explicit
acknowledgement and a private `DEEPGRAM_API_KEY`.

```bash
# Safe scope validation only; no audio/model/API operation.
python -m src.eval.asr_global_v2 --packs all \
  --archive-root data/raw/video_archives \
  --canonical data/index/global_keyframes.parquet \
  --output-dir data/index/asr_global_v3

# Execute an intentionally approved, resumable pack run.
python -m src.eval.asr_global_v2 --packs K01,L21 --execute --allow-network --confirm-api \
  --archive-root data/raw/video_archives \
  --canonical data/index/global_keyframes.parquet \
  --output-dir data/index/asr_global_v3 --work-dir data/work/asr_global_v3 \
  --raw-dir data/asr_global_v3_raw --model /opt/models/bge-m3 --resume

# Rebuild retrieval-resolution windows from existing Deepgram JSON only.
# This never extracts media and never calls Deepgram; it prefers word
# timestamps over coarse provider paragraphs.  Write a staged artifact first.
python -m src.eval.asr_global_v2 --packs L21,L22,L23,L24,L25,L26,L27,L28,L29,L30 \
  --raw-only --execute --canonical data/index/global_keyframes.parquet \
  --raw-dir data/asr_global_v2_raw \
  --output-dir data/index/modality_global_v3/asr_global_v2 \
  --model /opt/models/bge-m3

# Merge all completed K/L shards. Legacy K shards remain compatible, but are
# no longer a mandatory dependency for a complete rebuild.
python -m src.eval.asr_global_merge_v2 \
  --canonical data/index/global_keyframes.parquet \
  --legacy-dir data/index --l-dir data/index/modality_global_v3/asr_global_v2 \
  --output-dir data/index/modality_global_v3/asr_global_merged_v2

# After index preflight + benchmark pass, select the staged merged ASR source
# for Q&A without overwriting the previous production artifact.
VQA_ASR_GLOBAL_DIR=/opt/hcmai/data/index/modality_global_v3/asr_global_merged_v2
```

### 4. Rebuild OCR and text embeddings

OCR uses local Qwen2.5-VL-3B and bge-m3; no remote API fallback exists.

```bash
python -m src.eval.ocr_global_v2 --mode full --dry-run
python -m src.eval.ocr_global_v2 --mode full --execute --resume \
  --output-dir data/index/modality_global_v3/ocr \
  --model /opt/models/Qwen2.5-VL-3B-Instruct --embed-model /opt/models/bge-m3 \
  --device cuda --load-in-4bit
```

Read [docs/ASR_GLOBAL_V2.md](docs/ASR_GLOBAL_V2.md),
[docs/OCR_GLOBAL_V2.md](docs/OCR_GLOBAL_V2.md), and
[docs/DEPLOYMENT_ASSETS.md](docs/DEPLOYMENT_ASSETS.md) before a full rebuild.

## Verify and hand off source

```bash
python -m pytest -q \
  tests/test_build_canonical_map.py \
  tests/test_encode_full_index_contract.py \
  tests/test_asr_global_v2.py tests/test_asr_global_merge_v2.py \
  tests/test_runtime_data_bootstrap.py \
  tests/test_competition_ready_preflight.py tests/test_competition_run.py \
  tests/test_competition_submission_contract.py tests/test_qna_trake_contracts.py

scripts/build_source_bundle.sh
```

The source bundle follows `.gitignore`: it contains no corpus, embedding,
model, secret, private annotation or experiment output.

## Source layout

```text
src/indexing/      canonical mapping and index validation
src/encode/        visual embedding rebuilders
src/pipelines/     KIS, Q&A, TRAKE and output adapters
src/reranking/     ASR/OCR index runtime and routing
src/trake/         DANTE temporal alignment
src/cli/           competition preflight and runner
src/eval/          guarded ASR/OCR materializers only
scripts/           wrapper, Drive bootstrap and source handoff
configs/           published runtime artifact manifest
docs/              architecture, contracts and deployment guide
tests/             offline contract/regression tests
```
