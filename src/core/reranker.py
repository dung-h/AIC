"""
Module 4 (L10): Cross-encoder Reranker.

3 reranker types:
1. VLM Visual Reranker (gemma-4-31B): query + frame_image → score 0-10
2. Text Cross-Encoder (BGE-m3 dense interaction): query + ASR_chunk → similarity
3. LLM Judge: query + frame_caption → relevance Y/N + score

Common interface: rerank(query, candidates, type="visual"|"text"|"judge") → reordered candidates.

Cache mỗi rerank call (query+candidate_id) → < 1ms repeat.
"""
import os, sys, json, base64, urllib.request, urllib.error, time
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
from cache import get_cache
from paths import KEYFRAMES_DIR
from src.core.providers import provider_for

RERANK_CACHE = get_cache("reranker", version="v1")
KF_DIR = str(KEYFRAMES_DIR)


def _vlm_visual_score(image_b64, query, model=None):
    provider = provider_for("vision")
    if not provider.configured:
        raise RuntimeError(
            "visual reranking requires VLM_BASE_URL, VLM_API_KEY and VLM_MODEL"
        )
    prompt = (f"On a scale 0-10, how well does this image match: \"{query}\"? "
              f"Output ONLY a single integer 0-10.")
    pl = {"model": model or provider.model, "messages": [{"role": "user", "content": [
              {"type": "text", "text": prompt},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]}],
          "max_tokens": 5, "temperature": 0.0}
    req = urllib.request.Request(provider.base_url + "/chat/completions",
        data=json.dumps(pl).encode(),
        headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            txt = d["choices"][0]["message"]["content"].strip()
            import re
            m = re.search(r"\d+", txt)
            if m: return float(m.group(0))
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(8 * (a + 1)); continue
            raise
        except Exception:
            time.sleep(3 * (a + 1))
    return 5.0


@RERANK_CACHE.cached
def vlm_visual_score(image_path, query):
    """Cached visual rerank score 0-10."""
    if not os.path.exists(image_path): return 5.0
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    return _vlm_visual_score(b64, query)


@RERANK_CACHE.cached
def llm_judge(query, frame_caption):
    """LLM judge: relevance (frame caption vs query) → score 0-10."""
    provider = provider_for("text")
    if not provider.configured:
        raise RuntimeError(
            "caption judging requires TEXT_BASE_URL, TEXT_API_KEY and TEXT_MODEL"
        )
    prompt = (f"Query: \"{query}\"\nCaption: \"{frame_caption}\"\n"
              f"How well does the caption match the query (0-10)? Just one integer.")
    pl = {"model": provider.model,
          "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 5, "temperature": 0.0}
    req = urllib.request.Request(provider.base_url + "/chat/completions",
        data=json.dumps(pl).encode(),
        headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            txt = d["choices"][0]["message"]["content"].strip()
            import re
            m = re.search(r"\d+", txt)
            if m: return float(m.group(0))
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(5 * (a + 1)); continue
            raise
        except Exception:
            time.sleep(2)
    return 5.0


class Reranker:
    """High-level reranker interface."""

    def visual_rerank(self, query, candidates, kmap, alpha=0.4, beta=0.6, top_k=5):
        """
        Rerank candidates list[(video_id, frame_idx, pts_time, base_score)] bằng VLM.
        Combine alpha * normalized_base + beta * vlm_score/10.
        Cache aggressive.
        """
        if not candidates: return []
        scored = []
        for vid, fidx, t, base in candidates[:top_k]:
            m = kmap[(kmap.video_id == vid) & (kmap.frame_idx == fidx)]
            if len(m) == 0:
                scored.append((vid, fidx, t, base, 5.0)); continue
            kf_n = int(m.iloc[0].kf_n)
            fp = os.path.join(KF_DIR, vid, f"{kf_n:03d}.jpg")
            vlm = vlm_visual_score(fp, query)
            scored.append((vid, fidx, t, base, vlm))

        # Normalize base scores
        bases = np.array([s[3] for s in scored])
        if bases.max() != bases.min():
            bn = (bases - bases.min()) / (bases.max() - bases.min())
        else:
            bn = np.zeros_like(bases)
        vlms = np.array([s[4] for s in scored])
        vn = vlms / 10.0
        combined = alpha * bn + beta * vn

        # Sort by combined desc
        order = np.argsort(-combined)
        out = []
        for i in order:
            s = scored[i]
            out.append((s[0], s[1], s[2], float(combined[i])))
        # Append remaining unseen
        seen_keys = {(s[0], s[1]) for s in scored}
        for c in candidates:
            if (c[0], c[1]) not in seen_keys:
                out.append(c)
        return out
