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
export HCMAI_LOCAL_VLM_PATH=/opt/models/Qwen2.5-VL-7B-Instruct
export VQA_MODALITY_MODEL_DIR=/opt/models/bge-m3
export HF_HUB_CACHE=/opt/models/huggingface/hub
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

Remote capabilities are role-separated: `TEXT_*` for rewriting, `VLM_*` for
image-aware answering/reranking, and `EMBEDDING_*` for remote text embeddings.
The offline competition default needs no API key. Legacy `DO_*` vision names
are read only as a migration alias at the configuration boundary; new
deployments must use the role-specific keys in `.env.example`.

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

./scripts/competition.sh preflight --provider local --require-gpu \
  --require-modality-runtime --json-output results/preflight.json
```

The Drive bootstrap is intentionally not called from `run` or `preflight`:
automatically downloading 90+ GB during a competition request is unsafe. On a
new server the operator performs one explicit fetch; thereafter `run` is
offline-first.

## Competition commands

```bash
./scripts/competition.sh preflight [options]
./scripts/competition.sh bootstrap plan|fetch [options]
./scripts/competition.sh run --queries QUERY.json --output OUT.json \
  --answer-provider local --topk 20 --max-vlm-candidates 12
```

An all-task smoke input is included, but needs the real runtime assets:

```bash
./scripts/competition.sh run \
  --queries tests/fixtures/competition_all_tasks_smoke.json \
  --output results/submissions/smoke.json \
  --answer-provider local --topk 2 --max-vlm-candidates 1
```

`--answer-provider openai` is explicit opt-in after a VLM provider is
configured in `.env`; API usage is not a dependency of the local path.

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

# Merge all completed K/L shards. Legacy K shards remain compatible, but are
# no longer a mandatory dependency for a complete rebuild.
python -m src.eval.asr_global_merge_v2 \
  --canonical data/index/global_keyframes.parquet \
  --legacy-dir data/index --l-dir data/index/asr_global_v3 \
  --output-dir data/index/modality_global_v3/asr_global_merged_v3
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
