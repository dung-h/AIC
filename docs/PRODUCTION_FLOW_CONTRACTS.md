# Production flow contracts

This document describes the behavior that all production entrypoints must
share. Research scripts may experiment with other providers, but they must not
silently become a production path.

## Common lifecycle

```text
request
  -> boundary validation
  -> immutable RuntimePolicy snapshot
  -> exactly one task path
  -> retrieval/candidate generation
  -> task-specific ranking or alignment
  -> canonical frame + output-contract validation
  -> response/submission
```

An error before the task path is selected is a 4xx/configuration error. An
unavailable model or index is a 503/configuration failure. A malformed model
result is rejected; it is never replaced with frame 0, an empty answer, or a
different provider.

## Q&A

```text
query + question
  -> validate question_type/modalities
  -> visual KIS candidates
  -> optional global ASR/OCR retrieval
  -> video-level RRF
  -> bounded frame allocator
  -> local VLM answer + grounding metadata
  -> reject abstain/empty/evidence-only
  -> rank answers
  -> canonical (video_id, frame_idx)
```

Rules:

- `question_type` must be one of the eight benchmark types.
- `required_modalities` may contain only `visual`, `asr`, and `ocr`.
- A routing typo fails before loading retrieval/VLM models.
- `max_answers` is applied after candidate scoring, not before it.
- Remote/API VLM output belongs to interactive mode and cannot enter the
  ranked offline submission path.

## TRAKE

```text
event list
  -> validate and normalize event descriptions
  -> policy selects visual-DANTE (default) or explicit ASR diagnostic
  -> candidate video retrieval
  -> event x frame alignment
  -> strict temporal-order check
  -> canonical frame validation at serializer boundary
```

Rules:

- The UI does not force ASR; omission means the shared visual production
  policy applies.
- ASR requires a timestamped chunk index and embedding cache. Missing data
  produces a clear failure instead of a partially initialized object.
- Every answer must contain exactly one video and one strictly increasing frame
  per event.

## UI/API separation

- Web UI VQA sends `mode=interactive` explicitly.
- Ranked VQA is explicit through `mode=ranked` and uses the local VLM path.
- Web UI TRAKE omits `mode` so `RuntimePolicy` selects visual-DANTE.
- KIS/VKIS/TRAKE request limits are rejected, not clamped silently.

## Ownership rule

`RuntimePolicy` owns defaults. `HCMAIPipeline` and `RetrievalRuntime` consume
the snapshot. A request or CLI may create a local override, but no task module,
UI route, or serializer may introduce a second default. The serializer owns
only final contract validation; it must not repair retrieval semantics.
