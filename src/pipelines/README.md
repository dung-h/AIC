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
- `trake_pipeline.py`: explicit ASR TRAKE fallback/diagnostic path.
- `trake_visual.py`: visual TRAKE production candidate used by the shared
  `HCMAIPipeline` and service runtime by default.

Pipelines should not import from `src/eval/`. Evaluation code may import stable
pipeline contracts or shared primitives, but production code must remain usable
without experiment-only modules.
