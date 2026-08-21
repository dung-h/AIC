#!/usr/bin/env python3
"""Ask a reproducible visual question about one known local keyframe.

This is deliberately separate from retrieval: it tests the VQA answerer on a
known canonical frame.  It is useful for distinguishing a retrieval failure
from an answer-generation failure and for checking that greedy local VLM
decoding is repeatable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.core.local_vlm import LocalVLM


ROOT = Path(__file__).resolve().parents[1]


def _canonical_identity(image: Path, canonical_index: Path) -> dict[str, int | float | str] | None:
    """Resolve a standard ``<video>/<kf_n>.jpg`` path without guessing IDs."""
    try:
        video_id = image.parent.name
        kf_n = int(image.stem)
    except ValueError:
        return None
    if not canonical_index.is_file():
        return None

    import pandas as pd

    rows = pd.read_parquet(
        canonical_index,
        filters=[("video_id", "==", video_id), ("kf_n", "==", kf_n)],
        columns=["video_id", "kf_n", "frame_idx", "pts_time"],
    )
    if len(rows) != 1:
        raise SystemExit(
            f"canonical mapping is not unique for {video_id} keyframe {kf_n}: {len(rows)} rows"
        )
    row = rows.iloc[0]
    return {
        "video_id": str(row["video_id"]),
        "kf_n": int(row["kf_n"]),
        "frame_id": int(row["frame_idx"]),
        "pts_time": float(row["pts_time"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--canonical-index",
        type=Path,
        default=ROOT / "data" / "index" / "global_keyframes_vitl.parquet",
        help="optional canonical map used to verify a normal keyframe path",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"image does not exist: {args.image}")
    if not args.model.is_dir():
        raise SystemExit(f"local VLM model does not exist: {args.model}")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    canonical = _canonical_identity(args.image, args.canonical_index)

    prompt = (
        "Answer using only the supplied frame. Do not infer facts that are not "
        "visible. Give a short direct answer in Vietnamese.\nQuestion: "
        f"{args.question}"
    )
    vlm = LocalVLM(args.model)
    results: list[dict[str, object]] = []
    for number in range(1, args.runs + 1):
        started = time.perf_counter()
        record = vlm.answer_with_metadata(
            str(args.image), prompt, max_new_tokens=args.max_new_tokens
        )
        record["run"] = number
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        results.append(record)

    answers = [str(row["answer"]).strip().casefold() for row in results]
    valid = [
        bool(answer) and not bool(row["abstain"])
        for answer, row in zip(answers, results, strict=True)
    ]
    print(json.dumps({
        "image": str(args.image),
        "canonical": canonical,
        "question": args.question,
        "runs": results,
        "all_runs_answered": all(valid),
        "answer_repeatable": len(set(answers)) == 1,
    }, ensure_ascii=False, indent=2))
    return 0 if all(valid) else 2


if __name__ == "__main__":
    raise SystemExit(main())
