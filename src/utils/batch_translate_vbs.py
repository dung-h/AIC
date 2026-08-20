"""
Batch translate VBS-325 Vietnamese descriptions → English.

Uses 10 parallel threads with llama-4-maverick (fast, low throttle).
Saves result to data/index/vbs_queryset_300_en.parquet (adds 'desc_en' column).
Resume-safe: skips already-translated rows if output file exists.

Run:
    python src/utils/batch_translate_vbs.py
    python src/utils/batch_translate_vbs.py --workers 20  # faster if API allows
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
IDX = os.path.join(REPO, "data", "index")
VBS_IN  = os.path.join(IDX, "vbs_queryset_300.parquet")
VBS_OUT = os.path.join(IDX, "vbs_queryset_300_en.parquet")


def load_env():
    p = os.path.join(REPO, ".env")
    e = {}
    try:
        for line in open(p):
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                e[k.strip()] = v.strip()
    except Exception:
        pass
    return e


ENV = load_env()


def translate_one(idx: int, text: str, retries: int = 3) -> tuple[int, str]:
    key = ENV.get("DO_INFERENCE_KEY")
    base = ENV.get("DO_INFERENCE_BASE", "")
    if not key:
        return idx, text  # fallback: return original

    payload = {
        "model": "llama-4-maverick",
        "messages": [{"role": "user", "content":
            "Translate this Vietnamese visual scene description to a concise English "
            "image caption (keep all visual details). Output ONLY the caption:\n\n" + text}],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return idx, json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] translate failed idx={idx}: {e}", flush=True)
                return idx, text  # fallback
    return idx, text


def main(workers: int = 10):
    qs = pd.read_parquet(VBS_IN)
    print(f"Loaded {len(qs)} queries from {VBS_IN}")

    # Resume: if output exists, skip already-translated rows
    if os.path.exists(VBS_OUT):
        existing = pd.read_parquet(VBS_OUT)
        done_ids = set(existing.video_id.tolist()) if "desc_en" in existing.columns else set()
        print(f"  Resuming: {len(done_ids)} already translated")
    else:
        existing = None
        done_ids = set()

    todo = qs[~qs.video_id.isin(done_ids)].reset_index(drop=True)
    print(f"  To translate: {len(todo)} queries (workers={workers})\n")

    if len(todo) == 0:
        print("All done already.")
        return

    results = {}  # idx -> desc_en

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(translate_one, i, row["desc"]): i
                   for i, row in todo.iterrows()}
        done = 0
        for fut in as_completed(futures):
            idx, en = fut.result()
            results[idx] = en
            done += 1
            if done % 20 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(f"  [{done:3d}/{len(todo)}] {rate:.1f} q/s  ETA {eta:.0f}s", flush=True)

    # Write desc_en back to todo rows
    todo = todo.copy()
    todo["desc_en"] = [results.get(i, todo.loc[i, "desc"]) for i in todo.index]

    # Merge with existing if resuming
    if existing is not None and "desc_en" in existing.columns:
        merged = pd.concat([existing, todo], ignore_index=True).drop_duplicates("video_id")
    else:
        merged = todo

    merged.to_parquet(VBS_OUT, index=False)
    elapsed = time.time() - t0
    print(f"\nSaved {len(merged)} rows → {VBS_OUT}  ({elapsed:.1f}s)")
    print(f"Sample:\n  VN: {merged.iloc[0]['desc'][:80]}")
    print(f"  EN: {merged.iloc[0]['desc_en'][:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    main(workers=args.workers)
