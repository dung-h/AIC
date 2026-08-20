"""Submission generator for AIC HCMC.

The official AIC26 transport is a ZIP whose root contains ``submission/``.
Each query is stored as one UTF-8, headerless CSV with at most 100 rows:

* KIS: ``video_id,frame_idx``
* QA: ``video_id,frame_idx,answer``
* TRAKE: ``video_id,frame_1,...,frame_N``

The historical 2025 and ranked JSON/CSV adapters remain available explicitly
for compatibility.  They are not the official AIC26 transport.

Hệ thống: dùng HCMAIPipeline (4 task: KIS, VKIS, VQA, TRAKE)
KIS dùng KISFusionRetriever (ViT-L + SO400M-384 zscore fusion).

Cách dùng AIC26 chính thức:
  python codabench_submit.py --input queries.csv --output answer.zip --task KIS

``.zip`` tự chọn ``aic2026_official``; các format cũ vẫn có thể chọn rõ
qua ``--format`` để phục vụ regression/compatibility.

Input CSV expected columns:
  query_id, query, [task_type] (KIS/VKIS/VQA/TRAKE)
  Nếu không có task_type → mặc định KIS.

Official output: answer.zip chứa submission/query-*-kis|qa|trake.csv
"""
import csv
import io
import os, sys, json, time
import re
import tempfile
import zipfile
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from collections.abc import Mapping, Sequence
import pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "..")))
sys.path.insert(0, os.path.join(ROOT, "..", "router"))
sys.path.insert(0, os.path.join(ROOT, "..", "core"))


CODABENCH_2025_COLUMNS = ["query_id", "video_name", "frame_idx"]
RANKED_CSV_COLUMNS = {
    "qa": ["query_id", "video_id", "frame_id", "answer", "rank"],
    "trake": ["query_id", "video_id", "frame_ids", "rank"],
}
AIC26_OFFICIAL_FORMAT = "aic2026_official"
AIC26_MAX_ROWS = 100
AIC26_MAX_ANSWER_CHARS = 100
_AIC26_QUERY_TOKEN = re.compile(r"^[A-Za-z0-9._-]+$")


def _aic26_task(task):
    value = str(task).strip().casefold()
    return {"vqa": "qa", "q&a": "qa", "textual_kis": "kis"}.get(value, value)


def _aic26_filename(query_id, task):
    """Return a traversal-safe official filename for one query."""
    task = _aic26_task(task)
    if task not in {"kis", "qa", "trake"}:
        raise ValueError(f"unsupported AIC26 task: {task}")
    token = str(query_id).strip()
    if not token or not _AIC26_QUERY_TOKEN.fullmatch(token):
        raise ValueError(f"invalid AIC26 query_id for filename: {query_id!r}")
    if token in {".", ".."}:
        raise ValueError(f"invalid AIC26 query_id for filename: {query_id!r}")
    prefix = token if token.startswith("query-") else f"query-{token}"
    return f"{prefix}-{task}.csv"


def _aic26_video_id(value, *, query_id, rank):
    if value is None:
        raise ValueError(f"query {query_id} rank {rank}: video_id must not be null")
    video_id = str(value).strip()
    if not video_id or "\x00" in video_id or "\n" in video_id or "\r" in video_id:
        raise ValueError(f"query {query_id} rank {rank}: invalid video_id")
    return video_id


def _aic26_frame_idx(value, *, query_id, rank):
    """Normalize an integer frame without silently truncating floats/bools."""
    if isinstance(value, bool):
        raise ValueError(f"query {query_id} rank {rank}: frame_idx must be an integer")
    if isinstance(value, Integral):
        frame_idx = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        frame_idx = int(value.strip())
    else:
        raise ValueError(f"query {query_id} rank {rank}: frame_idx must be an integer")
    if frame_idx < 0:
        raise ValueError(f"query {query_id} rank {rank}: frame_idx must be non-negative")
    return frame_idx


def _aic26_canonical_contains(canonical_frames, video_id, frame_idx):
    if canonical_frames is None:
        raise ValueError("canonical frame index is required for official AIC26 submission")
    contains = getattr(canonical_frames, "contains", None)
    if callable(contains):
        return bool(contains(video_id, frame_idx))
    if callable(canonical_frames):
        return bool(canonical_frames(video_id, frame_idx))
    if isinstance(canonical_frames, Mapping):
        try:
            return frame_idx in {int(item) for item in canonical_frames.get(video_id, ())}
        except (TypeError, ValueError) as exc:
            raise TypeError("invalid canonical frame mapping") from exc
    try:
        return (video_id, frame_idx) in canonical_frames
    except TypeError as exc:
        raise TypeError("canonical_frames must support membership or contains()") from exc


def format_aic26_query_csv(task, query_id, answers, *, canonical_frames, event_count=None):
    """Validate and serialize one official AIC26 query CSV as UTF-8 bytes."""
    task = _aic26_task(task)
    if task not in {"kis", "qa", "trake"}:
        raise ValueError(f"unsupported AIC26 task: {task}")
    if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
        raise TypeError(f"query {query_id}: ranked answers must be a sequence")
    if not answers:
        raise ValueError(f"query {query_id}: ranked answers must not be empty")
    if len(answers) > AIC26_MAX_ROWS:
        raise ValueError(f"query {query_id}: at most 100 rows are allowed")
    if task == "trake" and (
        isinstance(event_count, bool) or not isinstance(event_count, Integral) or event_count <= 0
    ):
        raise ValueError(f"query {query_id}: TRAKE event_count must be a positive integer")

    csv_rows = []
    for rank, answer in enumerate(answers, 1):
        if not isinstance(answer, Mapping):
            raise TypeError(f"query {query_id} rank {rank}: answer must be an object")
        video_id = _aic26_video_id(answer.get("video_id", answer.get("video_name")),
                                   query_id=query_id, rank=rank)
        if task == "trake":
            frame_values = answer.get("frame_ids")
            if not isinstance(frame_values, Sequence) or isinstance(frame_values, (str, bytes)):
                raise ValueError(f"query {query_id} rank {rank}: frame_ids must be a sequence")
            if len(frame_values) != int(event_count):
                raise ValueError(
                    f"query {query_id} rank {rank}: expected {int(event_count)} frames, "
                    f"got {len(frame_values)}"
                )
            frames = [
                _aic26_frame_idx(value, query_id=query_id, rank=rank)
                for value in frame_values
            ]
            if any(left >= right for left, right in zip(frames, frames[1:])):
                raise ValueError(
                    f"query {query_id} rank {rank}: TRAKE frames must be strictly increasing"
                )
            for frame_idx in frames:
                if not _aic26_canonical_contains(canonical_frames, video_id, frame_idx):
                    raise ValueError(
                        f"query {query_id} rank {rank}: non-canonical frame "
                        f"{video_id}/{frame_idx}"
                    )
            csv_rows.append([video_id, *frames])
            continue

        frame_value = answer.get("frame_idx", answer.get("frame_id"))
        frame_idx = _aic26_frame_idx(frame_value, query_id=query_id, rank=rank)
        if not _aic26_canonical_contains(canonical_frames, video_id, frame_idx):
            raise ValueError(
                f"query {query_id} rank {rank}: non-canonical frame {video_id}/{frame_idx}"
            )
        if task == "kis":
            csv_rows.append([video_id, frame_idx])
            continue

        raw_answer = answer.get("answer")
        if raw_answer is None:
            raise ValueError(f"query {query_id} rank {rank}: QA answer must not be null")
        text_answer = str(raw_answer).strip()
        if (
            not text_answer
            or "\x00" in text_answer
            or "\n" in text_answer
            or "\r" in text_answer
        ):
            raise ValueError(f"query {query_id} rank {rank}: QA answer must not be empty")
        if len(text_answer) > AIC26_MAX_ANSWER_CHARS:
            raise ValueError(
                f"query {query_id} rank {rank}: QA answer exceeds 100 characters"
            )
        csv_rows.append([video_id, frame_idx, text_answer])

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(csv_rows)
    return stream.getvalue().encode("utf-8")


def write_aic26_submission_zip(
    task,
    queries,
    output,
    *,
    canonical_frames,
    event_counts=None,
):
    """Write an official fail-closed AIC26 ZIP with ``submission/`` at root."""
    task = _aic26_task(task)
    if not isinstance(queries, Mapping) or not queries:
        raise ValueError("official AIC26 submission must contain at least one query")
    output_path = Path(output)
    if output_path.suffix.casefold() != ".zip":
        raise ValueError("official AIC26 submission output must use the .zip extension")

    members = {}
    for query_id, answers in queries.items():
        filename = _aic26_filename(query_id, task)
        archive_name = f"submission/{filename}"
        if archive_name.casefold() in {name.casefold() for name in members}:
            raise ValueError(f"duplicate AIC26 submission filename: {archive_name}")
        event_count = None
        if task == "trake":
            key = str(query_id)
            if not isinstance(event_counts, Mapping) or key not in event_counts:
                raise ValueError(f"missing TRAKE event_count for query {key}")
            event_count = event_counts[key]
        members[archive_name] = format_aic26_query_csv(
            task,
            query_id,
            answers,
            canonical_frames=canonical_frames,
            event_count=event_count,
        )

    # Build the complete archive in memory.  Validation failures therefore
    # cannot leave a partially valid production ZIP on disk.
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("submission/", b"")
        for archive_name, content in members.items():
            bundle.writestr(archive_name, content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive.getvalue())
    return {
        "path": output_path,
        "query_count": len(members),
        "row_count": sum(content.count(b"\n") for content in members.values()),
        "members": list(members),
    }


def write_aic26_mixed_submission_zip(
    entries,
    output,
    *,
    canonical_frames_by_task,
):
    """Write one official ZIP containing KIS, Q&A and/or TRAKE queries.

    ``entries`` is a sequence of records with ``task``, ``query_id`` and
    ``answers``.  TRAKE records additionally require ``event_count``.  Every
    record is validated before any output file is replaced, so a malformed
    query cannot leave a partially usable competition archive behind.
    """
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("mixed AIC26 entries must be a sequence")
    if not entries:
        raise ValueError("official AIC26 submission must contain at least one query")
    if not isinstance(canonical_frames_by_task, Mapping):
        raise TypeError("canonical_frames_by_task must be a mapping")

    output_path = Path(output)
    if output_path.suffix.casefold() != ".zip":
        raise ValueError("official AIC26 submission output must use the .zip extension")

    members = {}
    row_count = 0
    task_counts = {"kis": 0, "qa": 0, "trake": 0}
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, Mapping):
            raise TypeError(f"mixed entry {position}: expected an object")
        task = _aic26_task(entry.get("task"))
        if task not in task_counts:
            raise ValueError(f"mixed entry {position}: unsupported task {task!r}")
        query_id = entry.get("query_id")
        filename = _aic26_filename(query_id, task)
        archive_name = f"submission/{filename}"
        archive_key = archive_name.casefold()
        if archive_key in members:
            raise ValueError(f"duplicate AIC26 submission filename: {archive_name}")
        canonical_frames = canonical_frames_by_task.get(task)
        if canonical_frames is None:
            raise ValueError(f"canonical frame index is unavailable for task {task}")
        content = format_aic26_query_csv(
            task,
            query_id,
            entry.get("answers"),
            canonical_frames=canonical_frames,
            event_count=entry.get("event_count") if task == "trake" else None,
        )
        members[archive_key] = (archive_name, content)
        row_count += content.count(b"\n")
        task_counts[task] += 1

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("submission/", b"")
        for archive_name, content in (value for value in members.values()):
            bundle.writestr(archive_name, content)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".zip.tmp", prefix=f".{output_path.stem}-",
            dir=output_path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(archive.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    return {
        "path": output_path,
        "query_count": len(members),
        "row_count": row_count,
        "task_counts": task_counts,
        "members": [value[0] for value in members.values()],
    }


def format_submission_rows(rows, fmt="codabench_2025"):
    """
    Format rows for Codabench submission.

    Current default follows observed/expected 2025 format:
      query_id, video_name, frame_idx

    Keep this function as the single place to update once 2026 Terms publish
    the official CSV schema.
    """
    if fmt != "codabench_2025":
        raise ValueError(f"Unsupported submission format: {fmt}")
    if not rows:
        raise ValueError("KIS submission must contain at least one ranked row")
    df = pd.DataFrame(rows)
    missing = [col for col in CODABENCH_2025_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"KIS submission is missing required columns: {missing}")
    if df["query_id"].isna().any() or df["video_name"].isna().any():
        raise ValueError("KIS submission contains a null query_id/video_name")
    if df["query_id"].astype(str).str.strip().eq("").any() or df["video_name"].astype(str).str.strip().eq("").any():
        raise ValueError("KIS submission contains an empty query_id/video_name")
    try:
        frame_values = pd.to_numeric(df["frame_idx"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("KIS submission contains a non-numeric frame_idx") from exc
    if frame_values.isna().any() or (frame_values < 0).any() or (frame_values % 1 != 0).any():
        raise ValueError("KIS submission contains an invalid frame_idx")
    df["frame_idx"] = frame_values.astype("int64")
    return df[CODABENCH_2025_COLUMNS]


def write_submission_csv(rows, output, fmt="codabench_2025"):
    out = format_submission_rows(rows, fmt=fmt)
    out.to_csv(output, index=False)
    return out


def format_ranked_submission_rows(task, payload, *, columns=None):
    """Convert a validated ranked JSON payload to an explicit CSV schema."""
    task = {"vqa": "qa", "q&a": "qa"}.get(str(task).strip().lower(), str(task).strip().lower())
    if task not in RANKED_CSV_COLUMNS:
        raise ValueError(f"ranked CSV is unsupported for task: {task}")
    queries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(queries, dict) or not queries:
        raise ValueError("ranked submission must contain non-empty queries")
    expected = list(columns or RANKED_CSV_COLUMNS[task])
    if not expected:
        raise ValueError("ranked CSV schema must contain at least one column")
    rows = []
    for query_id, answers in queries.items():
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"query {query_id} has no ranked answers")
        for rank, answer in enumerate(answers, 1):
            if not isinstance(answer, dict):
                raise ValueError(f"query {query_id} rank {rank} is not an answer object")
            row = {"query_id": str(query_id), "rank": rank}
            if task == "qa":
                required = ("video_id", "frame_id", "answer")
                if any(field not in answer for field in required):
                    raise ValueError(f"query {query_id} rank {rank} is missing Q&A fields")
                row.update({field: answer[field] for field in required})
            else:
                if any(field not in answer for field in ("video_id", "frame_ids")):
                    raise ValueError(f"query {query_id} rank {rank} is missing TRAKE fields")
                row["video_id"] = answer["video_id"]
                row["frame_ids"] = json.dumps(answer["frame_ids"], ensure_ascii=False)
            rows.append(row)
    frame = pd.DataFrame(rows)
    missing = [field for field in expected if field not in frame.columns]
    if missing:
        raise ValueError(f"ranked CSV schema mismatch; missing={missing}")
    return frame[expected]


def write_ranked_submission_csv(task, payload, output, *, columns=None):
    out = format_ranked_submission_rows(task, payload, columns=columns)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def _parse_events(row):
    """Parse a TRAKE row from JSON/list text or event_1, event_2... columns."""
    raw = getattr(row, "events", None)
    missing = raw is None
    if not missing and not isinstance(raw, (list, tuple)):
        try:
            missing = bool(pd.isna(raw))
        except (TypeError, ValueError):
            missing = False
    if not missing:
        if isinstance(raw, (list, tuple)):
            events = list(raw)
        else:
            text = str(raw).strip()
            try:
                value = json.loads(text)
                events = value if isinstance(value, list) else [text]
            except json.JSONDecodeError:
                events = [part.strip() for part in text.replace("\n", "|").split("|")]
        events = [str(event).strip() for event in events if str(event).strip()]
        if events:
            return events
    names = sorted(
        name for name in row._fields
        if name.lower().startswith("event_") and str(getattr(row, name)).strip()
    )
    return [str(getattr(row, name)).strip() for name in names]


def _write_ranked_json(payload, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _canonical_pairs(pipe, task, trake_mode=None):
    """Return the canonical ``(video_id, frame_idx)`` set used by this run."""
    if trake_mode is None:
        policy = getattr(pipe, "policy", None)
        trake_mode = getattr(policy, "trake_mode", "visual")
    def _pairs_from_map(frame_map, *, video_attr, frame_attr, label):
        if frame_map is None:
            raise RuntimeError(f"{label} canonical frame index is unavailable")
        videos = getattr(frame_map, video_attr, None)
        frames = getattr(frame_map, frame_attr, None)
        if videos is None or frames is None:
            raise RuntimeError(f"{label} canonical frame index is missing video_id/frame_idx")
        try:
            videos = list(videos)
            frames = list(frames)
        except TypeError as exc:
            raise RuntimeError(f"{label} canonical frame index is not iterable") from exc
        if len(videos) != len(frames):
            raise RuntimeError(f"{label} canonical frame index has mismatched columns")
        pairs = set()
        for video_id, frame_idx in zip(videos, frames):
            if not str(video_id).strip():
                raise RuntimeError(f"{label} canonical frame index contains an empty video_id")
            try:
                frame = int(frame_idx)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{label} canonical frame index contains an invalid frame_idx") from exc
            if frame < 0:
                raise RuntimeError(f"{label} canonical frame index contains a negative frame_idx")
            pairs.add((str(video_id), frame))
        if not pairs:
            raise RuntimeError(f"{label} canonical frame index is empty")
        return pairs

    if task == "kis":
        try:
            frame_map = pipe._ensure_kis().km
        except Exception as exc:
            raise RuntimeError("KIS canonical frame index/provider is unavailable") from exc
        return _pairs_from_map(frame_map, video_attr="video_id", frame_attr="frame_idx", label="KIS")
    if task == "qa":
        provider = getattr(pipe, "_vqa_ranked", None) or getattr(pipe, "_vqa", None)
        if provider is None:
            raise RuntimeError("Q&A provider was not initialized; canonical frame index unavailable")
        frame_map = getattr(provider, "km", None)
        return _pairs_from_map(frame_map, video_attr="video_id", frame_attr="frame_idx", label="Q&A")
    if str(trake_mode).strip().lower() == "asr":
        try:
            chunks = pipe._ensure_trake().ac
        except Exception as exc:
            raise RuntimeError("TRAKE ASR provider/index is unavailable") from exc
        return _pairs_from_map(chunks, video_attr="vid", frame_attr="frame_idx", label="TRAKE ASR")
    try:
        frame_map = pipe._ensure_trake_visual().km
    except Exception as exc:
        raise RuntimeError("TRAKE visual provider/index is unavailable") from exc
    return _pairs_from_map(frame_map, video_attr="video_id", frame_attr="frame_idx", label="TRAKE visual")


def _submission_policy(policy, task: str, offline: bool):
    """Apply submission-only hard constraints without mutating base policy."""
    if offline and task in {"qa", "trake"}:
        return replace(
            policy,
            kis_remote_translation=False,
            trake_remote_embeddings=False,
            # Submission is the network boundary for the ranked tasks.  Do
            # not let an environment-selected API provider leak into an
            # explicitly offline payload.
            vqa_answer_provider="local",
            network_mode="offline",
        )
    return policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV (query_id, query, [task_type])")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--task", default=None, help="Force task type: KIS/VKIS/VQA/TRAKE")
    parser.add_argument("--topk", type=int, default=20, help="Top-K ranked answers per query (max 100)")
    parser.add_argument("--offline", action="store_true", help="Use offline-first Q&A/TRAKE paths")
    parser.add_argument(
        "--answer-provider", choices=["local", "openai"], default=None,
        help="Q&A answer provider. openai is explicit online opt-in; local remains the default",
    )
    parser.add_argument(
        "--max-vlm-candidates", type=int, default=None,
        help="Q&A answer budget (default: 12 local, 3 API)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["codabench_2025", "aic2026_json", "aic2026_csv", AIC26_OFFICIAL_FORMAT],
        help=(
            "Submission schema. aic2026_official writes the official ZIP with "
            "one headerless CSV per query; other values are legacy compatibility formats"
        ),
    )
    parser.add_argument("--audit-output", default=None,
                        help="Optional JSON sidecar for submission validation/diagnostics")
    parser.add_argument("--trake-mode", choices=["visual", "asr"], default=None,
                        help="Local override; otherwise use RuntimePolicy.from_env()")
    args = parser.parse_args()

    from src.runtime_policy import RuntimePolicy
    policy = RuntimePolicy.from_env()
    if args.answer_provider is not None:
        if args.offline and args.answer_provider == "openai":
            raise ValueError("--offline cannot be combined with --answer-provider openai")
        policy = policy.override(
            vqa_answer_provider=args.answer_provider,
            network_mode=("online" if args.answer_provider == "openai" else policy.network_mode),
        )
    trake_mode = policy.trake_mode if args.trake_mode is None else args.trake_mode

    df = pd.read_csv(args.input)
    print(f"Input: {len(df)} queries from {args.input}")

    if not 1 <= args.topk <= 100:
        raise ValueError("--topk must be between 1 and 100")
    tasks = {str(args.task or getattr(df.iloc[0], "task_type", "KIS")).lower()}
    if "task_type" in df.columns and args.task is None:
        tasks = {str(value).lower() for value in df["task_type"].dropna().unique()}
    if len(tasks) != 1:
        raise ValueError("mixed task_type input is unsupported; submit one task per payload")
    task = next(iter(tasks))
    task = {"textual_kis": "kis", "vqa": "qa", "q&a": "qa"}.get(task, task)
    if task not in {"kis", "qa", "trake"}:
        raise ValueError(f"unsupported task: {task}")
    if task == "trake" and not args.offline:
        raise ValueError(
            "TRAKE submission requires explicit --offline"
        )
    if task == "qa" and not args.offline and policy.vqa_answer_provider != "openai":
        raise ValueError(
            "Q&A submission requires --offline for local inference or explicit "
            "--answer-provider openai for the configured online provider"
        )
    if args.max_vlm_candidates is not None and not 1 <= args.max_vlm_candidates <= 100:
        raise ValueError("--max-vlm-candidates must be between 1 and 100")
    # Offline is a hard submission invariant, not a hint.  Do not allow
    # process environment flags to re-enable network-backed translation or
    # ASR embeddings for a supposedly offline run.
    policy = _submission_policy(policy, task, args.offline)
    from hcmai_pipeline import HCMAIPipeline
    pipe = HCMAIPipeline(policy=policy)
    # A .zip output opts into the official transport automatically.  Historical
    # .csv/.json defaults remain unchanged for backward compatibility.
    output_suffix = Path(args.output).suffix.casefold()
    output_format = args.format or (
        AIC26_OFFICIAL_FORMAT
        if output_suffix == ".zip"
        else ("codabench_2025" if task == "kis" else "aic2026_json")
    )
    if task == "kis" and output_format not in {"codabench_2025", AIC26_OFFICIAL_FORMAT}:
        raise ValueError("KIS requires codabench_2025 or aic2026_official")
    if task != "kis" and output_format not in {
        "aic2026_json", "aic2026_csv", AIC26_OFFICIAL_FORMAT,
    }:
        raise ValueError("Q&A/TRAKE require aic2026_json, aic2026_csv or aic2026_official")
    if output_format == AIC26_OFFICIAL_FORMAT and Path(args.output).suffix.casefold() != ".zip":
        raise ValueError("official AIC26 submission output must use the .zip extension")
    if output_suffix == ".zip" and output_format != AIC26_OFFICIAL_FORMAT:
        raise ValueError("a .zip output requires the aic2026_official format")

    rows = []
    ranked_queries = {}
    trake_event_counts = {}
    seen_query_ids = set()
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        qid = getattr(r, "query_id", getattr(r, "id", r.Index))
        query_key = str(qid).strip()
        if not query_key:
            raise ValueError(f"row {i + 1}: query_id must not be empty")
        if query_key in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_key}")
        seen_query_ids.add(query_key)
        query = r.query
        if task == "kis":
            out = pipe.kis(query, topk=args.topk)
            ranked_queries[query_key] = []
            for rank, (vid, fidx, t, sc) in enumerate(out["results"][:args.topk]):
                rows.append({
                    "query_id": qid,
                    "video_name": vid,
                    "frame_idx": fidx,
                })
                ranked_queries[query_key].append({"video_id": vid, "frame_idx": fidx})
        elif task == "qa":
            question = getattr(r, "question", None)
            if question is None or not str(question).strip():
                raise ValueError(f"query {qid}: Q&A input requires a non-empty question column")
            out = pipe.vqa_ranked(
                query, str(question),
                # The answer core intentionally returns at most 20 trusted
                # candidates even though the transport permits up to 100 rows.
                max_answers=min(args.topk, 20),
                max_vlm_candidates=(
                    args.max_vlm_candidates
                    if args.max_vlm_candidates is not None
                    else (3 if policy.vqa_answer_provider == "openai" else 12)
                ),
                question_type=getattr(r, "question_type", None),
                required_modalities=getattr(r, "required_modalities", None),
            )
            ranked_queries[query_key] = out.get("answers", [])
            if not ranked_queries[query_key]:
                raise ValueError(f"query {query_key}: local Q&A returned no valid answer")
        else:
            events = _parse_events(r)
            if not events:
                raise ValueError(f"query {qid}: TRAKE input requires events or event_1... columns")
            trake_event_counts[query_key] = len(events)
            out = pipe.trake(events, topk=args.topk, mode=trake_mode)
            answers = []
            for item in out.get("results", [])[:args.topk]:
                if not isinstance(item, dict):
                    raise ValueError(f"query {query_key}: TRAKE returned a non-object ranked answer")
                path = item.get("path")
                if not isinstance(path, list):
                    raise ValueError(f"query {query_key}: TRAKE returned an invalid ranked path")
                frame_ids = []
                for step in path:
                    if not isinstance(step, dict) or "frame_idx" not in step:
                        raise ValueError(f"query {query_key}: TRAKE path has no canonical frame_idx")
                    try:
                        frame_ids.append(int(step["frame_idx"]))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"query {query_key}: TRAKE path has a non-integer frame_idx") from exc
                answers.append({
                    "video_id": item.get("video_id"),
                    "frame_ids": frame_ids,
                    "score": item.get("score", 0.0),
                    "provider": item.get("provider"),
                    "model_id": item.get("model_id"),
                    "metadata": item.get("metadata", {}),
                })
            if not answers:
                raise ValueError(f"query {query_key}: TRAKE returned no valid answer")
            ranked_queries[query_key] = answers

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(df)} ({elapsed/(i+1):.1f}s/q)")

    if task == "kis":
        canonical = _canonical_pairs(pipe, task, trake_mode)
        invalid = [row for row in rows
                   if (str(row.get("video_name")), int(row.get("frame_idx", -1))) not in canonical]
        if invalid:
            raise ValueError(f"KIS submission contains {len(invalid)} non-canonical frame(s)")
        if output_format == AIC26_OFFICIAL_FORMAT:
            package = write_aic26_submission_zip(
                task,
                ranked_queries,
                args.output,
                canonical_frames=canonical,
            )
            count = package["row_count"]
        else:
            out = write_submission_csv(rows, args.output, fmt=output_format)
            count = len(out)
    else:
        # The stable submission adapter is the only production serializer.
        # The eval adapter remains available for historical reports only.
        from src.submission.adapters import audit_submission, serialize_submission
        canonical = _canonical_pairs(pipe, task, trake_mode)
        payload = serialize_submission(
            task,
            ranked_queries,
            canonical_frames=canonical,
            event_counts=trake_event_counts if task == "trake" else None,
        )
        audit = audit_submission(
            task,
            ranked_queries,
            canonical_frames=canonical,
            event_counts=trake_event_counts if task == "trake" else None,
            metadata={
                "entrypoint": "codabench_submit",
                "offline": bool(args.offline),
                "network_mode": getattr(policy, "network_mode", None),
                "vqa_answer_provider": getattr(policy, "vqa_answer_provider", None),
                "max_vlm_candidates": (
                    args.max_vlm_candidates
                    if args.max_vlm_candidates is not None
                    else (3 if policy.vqa_answer_provider == "openai" else 12)
                ) if task == "qa" else None,
                "trake_mode": trake_mode if task == "trake" else None,
                "output_format": output_format,
            },
        )
        if output_format == AIC26_OFFICIAL_FORMAT:
            package = write_aic26_submission_zip(
                task,
                payload["queries"],
                args.output,
                canonical_frames=canonical,
                event_counts=trake_event_counts if task == "trake" else None,
            )
        elif output_format == "aic2026_json":
            _write_ranked_json(payload, args.output)
        else:
            write_ranked_submission_csv(task, payload, args.output)
        if args.audit_output:
            _write_ranked_json(audit, args.audit_output)
        count = sum(len(items) for items in payload["queries"].values())
    elapsed = time.time() - t0
    print(f"\nDone: {count} ranked answers → {args.output} ({elapsed:.1f}s total, "
          f"{elapsed/len(df):.1f}s/query)")
    if task != "kis":
        print(
            f"Audit: {audit['answer_count']} answers, "
            f"max {audit['max_answers_per_query']}/query, "
            f"canonical={audit['canonical_frames_validated']}, "
            f"rank_order={audit['ranked_order_preserved']}"
        )


if __name__ == "__main__":
    main()
