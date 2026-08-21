# Production Pipelines

Production submission ownership is intentionally singular:

```text
scripts/competition.sh run -> src.cli.competition_run -> HCMAIPipeline -> submission adapter
```

`codabench_submit.py` remains a serializer plus a compatibility-only,
single-task CLI for regression or migration. It is never the default
production orchestrator; invoking its CLI requires `--compatibility-only`.
Use `./scripts/competition.sh run` for competition submissions.

Task-level pipelines live here:

- `kis_fusion_retriever.py`: KIS production visual candidate generation.
- `vkis_pipeline.py`: VKIS clip retrieval.
- `vqa_pipeline_v3.py`: VQA pipeline.
- `trake_pipeline.py`: explicit ASR TRAKE diagnostic/alternate path; it is not an automatic fallback.
- `trake_visual.py`: visual TRAKE production candidate used by the shared
  `HCMAIPipeline` and service runtime by default.

## Optional VQA precision route

The default VQA path remains unchanged. An explicit online A/B route can add:

```text
query/question → answer-free hypothesis planner → allow-listed external search
→ local ASR/OCR/visual retrieval → canonical evidence packet → answer model
→ independent semantic-evidence judge → submission adapter
```

`VQA_HYPOTHESIS_GENERATION=1` requires `VQA_EXTERNAL_GROUNDING=1`; its output
is only a bounded external-search view and never an answer or direct RRF
input. `VQA_SEMANTIC_EVIDENCE_VERIFIER=1` invokes a separate VLM/API pass that
can only accept or reject an already retrieved candidate with a canonical
frame. Both are off by default, forbidden by offline/benchmark-strict policy,
and recorded in the Q&A trace for A/B evaluation.

Pipelines should not import from `src/eval/`. Evaluation code may import stable
pipeline contracts or shared primitives, but production code must remain usable
without experiment-only modules.
