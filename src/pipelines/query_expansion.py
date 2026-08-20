"""
Query Expansion for AIC HCMC 2026 KIS.

Strategy (RAPID-style, LSC 2024 measured +15-20% Top-10 recall):
1. LLM generates N visual description variants of the original query
2. Each variant is searched in parallel
3. Results fused via Reciprocal Rank Fusion (RRF, k=60)
4. Original query anchor is always included (prevents hallucination drift)

API: llama-4-maverick (vision-capable, 1.6s/call, avoids llama3.3-70b rate limit)
Fallback: return [original_query] when API unavailable
"""
import json
import os
import time
import urllib.error
import urllib.request
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from paths import load_env


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
# Result tuple: (video_id, frame_idx, pts_time, score)
Result = Tuple[str, int, float, float]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _load_env(path: str | None = None) -> dict:
    return load_env(path)


def _llm_call(
    prompt: str,
    env: dict,
    model: str = "llama-4-maverick",
    max_tokens: int = 300,
    temperature: float = 0.7,
    retries: int = 2,
) -> Optional[str]:
    """
    Call DigitalOcean inference API (OpenAI-compatible).

    Returns raw text content or None on failure.
    """
    key = env.get("DO_INFERENCE_KEY")
    base = env.get("DO_INFERENCE_BASE", "https://inference.do-ai.run/v1")
    if not key:
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < retries:
                time.sleep(1)
            continue
    return None


# ---------------------------------------------------------------------------
# Core: variant generation
# ---------------------------------------------------------------------------
EXPANSION_PROMPT = """\
You are helping a video retrieval system find a specific Vietnamese TV broadcast moment.

Original query (Vietnamese visual description):
"{query}"

Generate exactly {n} alternative visual descriptions of the SAME scene.
Rules:
- Each variant must describe the same visual moment, NOT a different event
- Use different vocabulary, sentence structure, or level of detail
- Keep Vietnamese language and visual focus (what is SEEN, not heard)
- Do NOT add fictional details not implied by the original
- Output ONLY the {n} variants, one per line, no numbering or prefix

Variants:"""


def generate_variants(
    query: str,
    n: int = 3,
    env: Optional[dict] = None,
    model: str = "llama-4-maverick",
) -> List[str]:
    """
    Generate N visual description variants using LLM.

    Always returns at least [query] (original as fallback).

    Args:
        query: Original Vietnamese visual description
        n: Number of variants to generate (not counting original)
        env: API credentials dict (loaded from .env if None)
        model: LLM model name

    Returns:
        List of [original] + variants (length 1–n+1)
    """
    if env is None:
        env = _load_env()

    prompt = EXPANSION_PROMPT.format(query=query, n=n)
    raw = _llm_call(prompt, env, model=model, max_tokens=n * 60, temperature=0.7)

    if not raw:
        return [query]  # graceful fallback

    variants = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Strip leading numbering artifacts (e.g. "1. ", "- ")
        if line and len(line) > 3:
            import re
            line = re.sub(r"^[\d\-\*\.]+\s*", "", line).strip()
            if line:
                variants.append(line)

    # Deduplicate while preserving order
    seen = {query}
    unique_variants = []
    for v in variants[:n]:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)

    return [query] + unique_variants


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    result_lists: List[List[Result]],
    k: int = 60,
) -> List[Result]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score: sum over lists of 1 / (rank + k)
    Key identity: (video_id, frame_idx) — merges same frame across queries.

    Args:
        result_lists: Each inner list is [(video_id, frame_idx, pts_time, score), ...]
        k: RRF constant (default 60, standard in literature)

    Returns:
        Merged list sorted by descending RRF score, duplicates removed.
    """
    rrf_scores: dict = {}
    metadata: dict = {}  # store pts_time for each (vid, fidx)

    for results in result_lists:
        for rank, (vid, fidx, pts, sc) in enumerate(results):
            key = (vid, fidx)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
            if key not in metadata:
                metadata[key] = pts

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [
        (vid, fidx, metadata[(vid, fidx)], score)
        for (vid, fidx), score in ranked
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def expand_and_search(
    query: str,
    search_fn: Callable[[str, int], List[Result]],
    n_variants: int = 3,
    topk: int = 20,
    env: Optional[dict] = None,
    model: str = "llama-4-maverick",
    max_workers: int = 4,
) -> dict:
    """
    Full query expansion pipeline:
    1. Generate n_variants alternatives via LLM
    2. Search each variant in parallel
    3. Fuse results with RRF

    Args:
        query: Original Vietnamese query
        search_fn: Callable(query_text, topk) → List[Result]
        n_variants: Number of LLM variants to generate
        topk: Top-K results per variant search
        env: API credentials (loaded from .env if None)
        model: LLM model for variant generation
        max_workers: Thread pool size for parallel search

    Returns:
        {
            "variants": [str],          # original + generated variants
            "results": List[Result],    # RRF-fused, sorted
            "variant_counts": [int],    # how many results each variant returned
        }
    """
    if env is None:
        env = _load_env()

    # Step 1: Generate variants (includes original as index 0)
    variants = generate_variants(query, n=n_variants, env=env, model=model)

    # Step 2: Parallel search
    result_lists: List[List[Result]] = [[] for _ in variants]

    def _search(idx: int, q: str) -> Tuple[int, List[Result]]:
        return idx, search_fn(q, topk)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_search, i, v): i for i, v in enumerate(variants)}
        for future in as_completed(futures):
            try:
                idx, res = future.result()
                result_lists[idx] = res
            except Exception:
                pass  # keep empty list on failure

    # Step 3: RRF fusion
    fused = reciprocal_rank_fusion(result_lists, k=60)

    return {
        "variants": variants,
        "results": fused[:topk],
        "variant_counts": [len(r) for r in result_lists],
    }
