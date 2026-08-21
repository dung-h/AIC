"""Run a mixed AIC26 preselection query set and build one official ZIP.

Accepted inputs:

* a directory containing ``query-<id>-kis|qa|trake.txt`` files;
* a JSON manifest containing a list (or ``{"queries": [...]}``) of records.

JSON is the unambiguous emergency format.  Plain-text Q&A accepts labelled
``Description:/Question:`` blocks (or exactly two paragraphs); TRAKE accepts
labelled/numbered event lines.  Ambiguous files fail closed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.codabench_submit import (  # noqa: E402
    _aic26_task,
    load_production_canonical_registry,
    validate_aic26_submission_zip,
    write_aic26_mixed_submission_zip,
)
from src.runtime_policy import RuntimePolicy  # noqa: E402


QUERY_FILENAME = re.compile(
    r"^query-(?P<query_id>[A-Za-z0-9._-]+)-(?P<task>kis|qa|trake)\.txt$",
    flags=re.IGNORECASE,
)

# The production package is accuracy-first.  A model-free holdout replay on
# the frozen visual lattice showed that 20 balanced candidates retains more
# correct videos and frames than the historical 12-frame adaptive budget.
# The value remains overridable from the CLI for a constrained deployment.
COMPETITION_QA_VLM_CANDIDATES = 20
COMPETITION_QA_SELECTOR_POLICY = "balanced"
LABEL = re.compile(r"^\s*([^:：]+)\s*[:：]\s*(.*)$")
NUMBERED_EVENT = re.compile(
    r"^\s*(?:[-*•]|(?:e|event|sự\s*kiện)?\s*\d+[.)-])\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    task: str
    query: str = ""
    question: str = ""
    events: tuple[str, ...] = ()
    question_type: str | None = None
    required_modalities: Any = None
    source: str = ""


def _infer_qa_contract(query: str, question: str) -> tuple[str | None, tuple[str, ...] | None]:
    """Return the neutral contract for unlabelled JSON Q&A.

    A modality label is an evidence *contract*, not a query-classification
    hint.  Earlier keyword routing made words such as ``xã``, ``câu lạc bộ``
    or ``câu thơ`` silently require ASR, and recipe words silently require
    OCR.  That could suppress a valid visual/OCR/ASR combination before the
    parallel retrieval lanes had produced any evidence.  Only benchmark or
    query-manifest metadata may declare a required modality; otherwise the
    VQA entrypoint receives ``unknown`` and runs its bounded parallel support
    plan without letting any one lane override the answer contract.
    """
    del query, question
    return "unknown", None


def _plain_qa_contract() -> tuple[str, None]:
    """Return the neutral contract for an unannotated plain-text QA file.

    ``visual`` is implicit in the VQA entrypoint.  The parent routing policy
    is responsible for adding ASR/OCR as optional parallel support lanes;
    they must not be declared here as required evidence.
    """
    return "unknown", None


def _nonempty(value: Any, field: str, source: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{source}: {field} must not be empty")
    return text


def _record_to_spec(record: Any, *, source: str, default_id: str | None = None,
                    default_task: str | None = None) -> QuerySpec:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: query record must be a JSON object")
    query_id = _nonempty(record.get("query_id", record.get("id", default_id)), "query_id", source)
    task = _aic26_task(record.get("task", record.get("task_type", default_task)))
    if task not in {"kis", "qa", "trake"}:
        raise ValueError(f"{source}: unsupported task {task!r}")
    if task == "kis":
        query = _nonempty(
            record.get("query", record.get("description", record.get("text"))),
            "query", source,
        )
        return QuerySpec(query_id=query_id, task=task, query=query, source=source)
    if task == "qa":
        query = _nonempty(record.get("query", record.get("description")), "query", source)
        question = _nonempty(record.get("question"), "question", source)
        inferred_type, inferred_modalities = _infer_qa_contract(query, question)
        return QuerySpec(
            query_id=query_id, task=task, query=query, question=question,
            question_type=record.get("question_type") or inferred_type,
            required_modalities=(
                record.get("required_modalities")
                if record.get("required_modalities") is not None
                else inferred_modalities
            ),
            source=source,
        )
    raw_events = record.get("events")
    if not isinstance(raw_events, list) or len(raw_events) < 2:
        raise ValueError(f"{source}: TRAKE requires an events list with at least two items")
    events = tuple(_nonempty(value, f"event {index}", source)
                   for index, value in enumerate(raw_events, 1))
    return QuerySpec(query_id=query_id, task=task, events=events, source=source)


def _plain_blocks(text: str) -> list[str]:
    return [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _label_key(value: str) -> str | None:
    key = " ".join(value.casefold().split())
    if key in {"query", "description", "visual description", "context", "mô tả", "mo ta"}:
        return "query"
    if key in {"question", "q", "câu hỏi", "cau hoi"}:
        return "question"
    if re.fullmatch(r"(?:e|event|sự kiện|su kien)\s*\d+", key):
        return "event"
    return None


def _parse_plain_text(path: Path, query_id: str, task: str, text: str) -> QuerySpec:
    source = str(path)
    if task == "kis":
        return QuerySpec(
            query_id=query_id, task=task,
            query=_nonempty(text, "query", source), source=source,
        )

    fields: dict[str, list[str]] = {"query": [], "question": []}
    events: list[str] = []
    active: str | None = None
    saw_label = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LABEL.match(line)
        key = _label_key(match.group(1)) if match else None
        if key is not None:
            saw_label = True
            value = match.group(2).strip()
            if key == "event":
                if value:
                    events.append(value)
                active = None
            else:
                active = key
                if value:
                    fields[key].append(value)
            continue
        if active is not None:
            fields[active].append(line)

    if task == "qa":
        if saw_label and fields["query"] and fields["question"]:
            question_type, required_modalities = _plain_qa_contract()
            return QuerySpec(
                query_id=query_id, task=task,
                query=" ".join(fields["query"]), question=" ".join(fields["question"]),
                question_type=question_type, required_modalities=required_modalities,
                source=source,
            )
        blocks = _plain_blocks(text)
        if len(blocks) == 2:
            question_type, required_modalities = _plain_qa_contract()
            return QuerySpec(
                query_id=query_id, task=task, query=blocks[0], question=blocks[1],
                question_type=question_type,
                required_modalities=required_modalities,
                source=source,
            )
        if not saw_label:
            # The official Round 1 files use a legacy one-paragraph format:
            # the visual description is followed by either ``Hỏi ...`` or a
            # final question sentence on the next line.  Preserve the strict
            # fail-closed behavior for arbitrary prose, but parse this
            # unambiguous delimiter instead of rejecting the real query pack.
            compact_lines = [" ".join(line.split()) for line in text.splitlines()
                             if line.strip()]
            compact = " ".join(compact_lines)
            ask_match = re.search(r"\bHỏi\b", compact, flags=re.IGNORECASE)
            if ask_match:
                query = compact[:ask_match.start()].strip(" .:-")
                question = compact[ask_match.start():].strip()
                if query and question:
                    question_type, required_modalities = _plain_qa_contract()
                    return QuerySpec(
                        query_id=query_id, task=task, query=query,
                        question=question, question_type=question_type,
                        required_modalities=required_modalities, source=source,
                    )
            question_line = next(
                (index for index, line in enumerate(compact_lines)
                 if "?" in line),
                None,
            )
            if question_line is not None and question_line > 0:
                query = " ".join(compact_lines[:question_line]).strip(" .:-")
                question = " ".join(compact_lines[question_line:]).strip()
                if query and question:
                    question_type, required_modalities = _plain_qa_contract()
                    return QuerySpec(
                        query_id=query_id, task=task, query=query,
                        question=question, question_type=question_type,
                        required_modalities=required_modalities, source=source,
                    )
            if len(compact_lines) == 1 and "?" in compact:
                question_end = compact.rfind("?") + 1
                boundary = compact.rfind(". ", 0, question_end)
                if boundary > 0:
                    query = compact[:boundary].strip(" .:-")
                    question = compact[boundary + 2:question_end].strip()
                    if query and question:
                        question_type, required_modalities = _plain_qa_contract()
                        return QuerySpec(
                            query_id=query_id, task=task, query=query,
                            question=question, question_type=question_type,
                            required_modalities=required_modalities, source=source,
                        )
        raise ValueError(
            f"{source}: ambiguous Q&A text; use JSON or Description:/Question: labels"
        )

    if not events:
        numbered = []
        for line in text.splitlines():
            match = NUMBERED_EVENT.match(line)
            if match:
                numbered.append(match.group(1).strip())
        events = numbered
    if len(events) < 2:
        raise ValueError(
            f"{source}: ambiguous TRAKE text; use JSON events or Event 1:/Event 2: labels"
        )
    return QuerySpec(query_id=query_id, task=task, events=tuple(events), source=source)


def parse_query_file(path: Path) -> QuerySpec:
    match = QUERY_FILENAME.fullmatch(path.name)
    if not match:
        raise ValueError(
            f"{path}: expected filename query-<id>-kis|qa|trake.txt"
        )
    query_id = match.group("query_id")
    task = _aic26_task(match.group("task"))
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{path}: query file is empty")
    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        if isinstance(payload, list):
            payload = {"events": payload}
        return _record_to_spec(
            payload, source=str(path), default_id=query_id, default_task=task,
        )
    return _parse_plain_text(path, query_id, task, text)


def load_query_specs(path: Path) -> list[QuerySpec]:
    if path.is_dir():
        files = sorted(path.glob("query-*.txt"), key=lambda item: item.name.casefold())
        if not files:
            raise ValueError(f"{path}: no query-*.txt files found")
        specs = [parse_query_file(item) for item in files]
    elif path.suffix.casefold() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = payload.get("queries") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}: JSON manifest requires a non-empty queries list")
        specs = [
            _record_to_spec(record, source=f"{path}#{index}")
            for index, record in enumerate(records, 1)
        ]
    else:
        raise ValueError("--queries must be a query directory or JSON manifest")

    identities = set()
    for spec in specs:
        identity = (spec.query_id.casefold(), spec.task)
        if identity in identities:
            raise ValueError(f"duplicate query/task identity: {spec.query_id}/{spec.task}")
        identities.add(identity)
    return specs


def run_competition(specs: list[QuerySpec], *, output: Path, answer_provider: str,
                    modality_routing: bool, topk: int, max_vlm_candidates: int | None,
                    external_grounding: bool = False,
                    external_image_grounding: bool = False,
                    audit_output: Path | None = None) -> dict[str, Any]:
    # All retrieval/model artifacts are local even in the API answer profile.
    # This prevents sentence-transformers/transformers from probing the Hub;
    # the explicit answer provider and optional, allow-listed grounding remain
    # the only allowed network boundaries.
    # These are assignments, rather than defaults: a caller must not be able
    # to re-enable model/download traffic by inheriting a permissive shell.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from src.pipelines.hcmai_pipeline import HCMAIPipeline

    base = RuntimePolicy.from_env()
    if answer_provider == "openai":
        policy = base.override(
            execution_mode="production", network_mode="online",
            vqa_answer_provider="openai", kis_remote_translation=False,
            trake_remote_embeddings=False,
            vqa_modality_routing=modality_routing,
            vqa_visual_selector_policy=COMPETITION_QA_SELECTOR_POLICY,
            vqa_external_grounding=external_grounding,
            vqa_external_image_grounding=external_image_grounding,
        )
    else:
        policy = base.override(
            execution_mode="production",
            network_mode=(
                "online" if external_grounding or external_image_grounding else "offline"
            ),
            vqa_answer_provider="local", kis_remote_translation=False,
            trake_remote_embeddings=False, vqa_modality_routing=modality_routing,
            vqa_visual_selector_policy=COMPETITION_QA_SELECTOR_POLICY,
            vqa_external_grounding=external_grounding,
            vqa_external_image_grounding=external_image_grounding,
        )
    pipe = HCMAIPipeline(policy=policy)
    entries = []
    task_seen = set()
    timings = []
    for position, spec in enumerate(specs, 1):
        started = time.perf_counter()
        if spec.task == "kis":
            result = pipe.kis(spec.query, topk=topk)
            answers = [
                {"video_id": video_id, "frame_idx": frame_idx}
                for video_id, frame_idx, _pts_time, _score in result["results"][:topk]
            ]
            entry = {"task": "kis", "query_id": spec.query_id, "answers": answers}
        elif spec.task == "qa":
            budget = max_vlm_candidates or COMPETITION_QA_VLM_CANDIDATES
            result = pipe.vqa_ranked(
                spec.query, spec.question, max_answers=min(topk, 20),
                max_vlm_candidates=budget, question_type=spec.question_type,
                required_modalities=spec.required_modalities,
                modality_routing=modality_routing,
                offline=(
                    answer_provider == "local"
                    and not external_grounding
                    and not external_image_grounding
                ),
                external_grounding=external_grounding,
                external_image_grounding=external_image_grounding,
            )
            answers = result.get("answers", [])
            entry = {"task": "qa", "query_id": spec.query_id, "answers": answers}
        else:
            result = pipe.trake(spec.events, topk=topk, mode=policy.trake_mode)
            answers = [{
                "video_id": item.get("video_id"),
                "frame_ids": [int(step["frame_idx"]) for step in item.get("path", [])],
            } for item in result.get("results", [])[:topk]]
            entry = {
                "task": "trake", "query_id": spec.query_id,
                "event_count": len(spec.events), "answers": answers,
            }
        if not answers:
            raise RuntimeError(f"{spec.source}: pipeline returned no valid answers")
        entries.append(entry)
        task_seen.add(spec.task)
        elapsed = time.perf_counter() - started
        timings.append({"query_id": spec.query_id, "task": spec.task, "seconds": elapsed})
        print(f"[{position}/{len(specs)}] {spec.task.upper()} {spec.query_id}: "
              f"{len(answers)} answers in {elapsed:.2f}s", flush=True)

    # Do not let a retriever validate its own persisted map.  Every task is
    # checked against one immutable global registry shared by the official
    # dataset, even when an internal visual/ASR index happens to be stale.
    canonical_pairs, canonical_registry = load_production_canonical_registry()
    canonical = {task: canonical_pairs for task in sorted(task_seen)}
    package = write_aic26_mixed_submission_zip(
        entries, output, canonical_frames_by_task=canonical,
    )
    package_validation = validate_aic26_submission_zip(
        output, canonical_frames=canonical_pairs,
    )
    report = {
        "ready": True,
        "package": str(Path(package["path"]).resolve()),
        "query_count": package["query_count"],
        "row_count": package["row_count"],
        "task_counts": package["task_counts"],
        "answer_provider": answer_provider,
        "modality_routing": modality_routing,
        "external_grounding": external_grounding,
        "external_image_grounding": external_image_grounding,
        "vqa_selector_policy": policy.vqa_visual_selector_policy,
        "max_vlm_candidates": (
            max_vlm_candidates or COMPETITION_QA_VLM_CANDIDATES
        ),
        "topk": topk,
        "timings": timings,
        "members": package["members"],
        "canonical_registry": canonical_registry,
        "package_validation": package_validation,
    }
    if audit_output is not None:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    # Offline/local is the portable production default.  Remote VLM usage is
    # still supported, but must be an explicit choice so a migrated server
    # cannot silently depend on credentials or network availability.
    parser.add_argument("--answer-provider", choices=("openai", "local"), default="local")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--max-vlm-candidates", type=int, default=None,
        help=("Q&A answer-frame budget; default 20 for the accuracy-first "
              "competition profile"),
    )
    parser.add_argument(
        "--modality-routing", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--external-grounding", action=argparse.BooleanOptionalAction, default=False,
        help=("allow-listed external knowledge may generate retrieval hypotheses; "
              "requires VQA_EXTERNAL_* configuration and enables network only "
              "for that resolver"),
    )
    parser.add_argument(
        "--external-image-grounding", action=argparse.BooleanOptionalAction,
        default=False,
        help=("web images may seed local image-to-image retrieval for visual OOK "
              "entities; requires VQA_EXTERNAL_IMAGE_* configuration"),
    )
    parser.add_argument("--audit-output", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.topk <= 100:
        raise ValueError("--topk must be between 1 and 100")
    if args.max_vlm_candidates is not None and not 1 <= args.max_vlm_candidates <= 100:
        raise ValueError("--max-vlm-candidates must be between 1 and 100")
    specs = load_query_specs(args.queries)
    report = run_competition(
        specs, output=args.output, answer_provider=args.answer_provider,
        modality_routing=args.modality_routing, topk=args.topk,
        max_vlm_candidates=args.max_vlm_candidates,
        external_grounding=args.external_grounding,
        external_image_grounding=args.external_image_grounding,
        audit_output=args.audit_output,
    )
    print(json.dumps({key: report[key] for key in (
        "ready", "package", "query_count", "row_count", "task_counts",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
