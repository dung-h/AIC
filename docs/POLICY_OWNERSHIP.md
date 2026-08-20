# Runtime policy ownership

## Finding

The code previously had overlapping policy owners. A default could be chosen
by the shared pipeline, duplicated in the service/API layer, or overridden by a
CLI/Web UI endpoint. That is safe only when the precedence is explicit.

## Current ownership and precedence

| Layer | Owns | May override | Must not own |
|---|---|---|---|
| Shared `HCMAIPipeline` | production defaults for Codabench/Web UI | explicit constructor or request mode | answer ranking internals |
| `RetrievalRuntime` | resident service loading and service-only defaults | explicit HTTP request fields | benchmark policy claims |
| HTTP/Web UI request | one-request mode/selector choice | shared default for that request only | process-wide state |
| Codabench CLI | one submission run, e.g. `--trake-mode` | shared default for that run only | silent fallback/provider substitution |
| task pipeline | retrieval, allocation, alignment algorithm | explicit function parameters | which path is production |
| `src/eval` | experiment configuration and metrics | nothing in production | production defaults |

Precedence is:

```text
research config (diagnostic only)
        < shared production default
        < explicit process/constructor setting
        < explicit request/CLI setting
```

An absent request/CLI field must resolve to the shared production default. It
must not introduce a second task default.

## Production choices currently locked

- KIS: visual ViT-L + SO400M fusion; remote translation is opt-in.
- VKIS: `hybrid0.5` selector unless explicitly requested otherwise.
- VQA ranked path: local/offline provider, `balanced` selector; modality
  routing is opt-in and strict when enabled.
- TRAKE: visual-DANTE; ASR-DANTE is explicit fallback/diagnostic only.
- Final serializer: canonical `(video_id, frame_idx)` validation is mandatory;
  missing map entries fail closed.

## Known non-production paths

Interactive Web UI VQA has `mode=interactive|ranked`; `interactive` is a
provider/API path and must not be used for offline benchmark claims. KIS OOK,
remote translation, ASR TRAKE, selector sweeps, and legacy pipelines are
explicit research or fallback paths.

## Remaining architectural debt

The production policy fields and defaults are now parsed by
`RuntimePolicy.from_env()` and injected into both `HCMAIPipeline` and
`RetrievalRuntime`. The remaining `os.getenv` calls are for unrelated concerns
(provider credentials, annotation-pack paths, or research scripts), not task
selection policy. Any new production policy must still be added to
`RuntimePolicy` first and covered by an entrypoint-default test; it must not be
added only to a task module or HTTP route.
