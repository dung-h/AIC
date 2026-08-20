# AIC2026 model-agnostic architecture

The competition system is split into stable contracts and replaceable
implementations. The model or API provider can change without changing the
submission contract or the canonical catalog.

```text
query
  -> Encoder adapters (local or remote)
  -> one FAISS/HNSW index per modality
  -> stable ranked hits
  -> video-level weighted RRF
  -> canonical SQLite frame resolution
  -> frame allocator / temporal neighbors
  -> Reranker or Answerer adapter (local or remote)
  -> Q&A / TRAKE serializer
```

## Compatibility rules

- A query encoder must have the same `model_id`, dimension, metric,
  normalization and version as the index that it searches.
- Changing an answerer/reranker does not require re-embedding the corpus.
- Changing an ASR/OCR/visual encoder requires rebuilding only that modality's
  vector shard and its manifest/id map.
- Fusion never combines raw vectors or raw scores from unrelated spaces. It
  combines ranked hits, so visual/ASR/OCR dimensions may differ.
- A missing provider dependency, API key, index or manifest fails closed. No
  silent lexical or remote fallback is allowed on the production path.

## Artifact boundaries

- `data/catalog/aic2026_catalog.sqlite`: canonical metadata, ASR/OCR evidence,
  provenance and shard registry.
- `data/index/*.faiss`: semantic retrieval artifacts.
- `*.manifest.json` and `*.idmap.json`: encoder/index/row identity contract.
- `src/architecture/runtime.py`: integration facade used by a future production
  pipeline; legacy pipelines remain unchanged until an A/B benchmark promotes
  this path.
