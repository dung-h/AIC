"""
Hybrid Search for AIC HCMC 2026.

Late-fusion strategy combining:
  - Visual (ViT-L SigLIP2) — dominant for scene/action queries
  - Object (OpenImages V4 inverted index) — precise for entity queries
  - OCR (BM25 over text frames) — dominant for named-entity / text-overlay queries

Signal selection follows AIC data reality (FINDINGS.md):
  - Visual confusion 47% on news → object/OCR supplement when query mentions specific things
  - Object pack-signature lift 8–22× → object filter is high-precision for specific entities
  - Routing > fusion for complementary signals → weights are query-adaptive, not fixed

Fusion method: weighted Reciprocal Rank Fusion (wRRF)
  score(d) = Σ  w_i / (rank_i(d) + k)
This preserves rank order and avoids score scale mismatch.
"""
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

# Result type shared with query_expansion.py
Result = Tuple[str, int, float, float]  # (video_id, frame_idx, pts_time, score)

# RRF constant (standard, k=60 from Cormack et al.)
_RRF_K = 60

# ---------------------------------------------------------------------------
# Object keyword vocabulary (AIC-relevant, from EDA pack signatures)
# These trigger the object channel when detected in query text.
# ---------------------------------------------------------------------------
_OBJECT_KEYWORDS = {
    # cycling / sports
    "xe đạp", "đua xe", "xe đạp leo núi", "bicycle", "cycling",
    # food / cooking
    "chảo", "nấu", "ăn", "thức ăn", "đồ ăn", "pizza", "cà chua",
    "rau", "thịt", "bánh", "nướng", "xào", "chiên", "frying pan",
    # festival / culture
    "lân", "sư", "rồng", "đèn lồng", "lantern", "dragon",
    # nature / weather
    "bão", "lũ", "núi", "biển", "cây", "rừng",
    # animals
    "chó", "mèo", "cá", "gà",
    # architecture
    "toà nhà", "cầu", "nhà thờ",
}

# OCR keyword triggers: named entities, numbers, titles.
# These are CASE-SENSITIVE on purpose — capitalization is the signal that a
# proper noun / on-screen text is present. Using re.IGNORECASE here would make
# the proper-noun pattern match any two lowercase words (defeating its purpose).
#
# Note: a bare "3+ accented Latin chars" pattern is deliberately NOT used: almost
# every Vietnamese word carries diacritics, so it would fire on virtually all
# queries and route everything to OCR.
_OCR_PATTERNS_CASE_SENSITIVE = [
    r"\b[A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)+",  # Multi-word proper nouns (Shigeru Ishiba, Biển Đông)
    r"['\"][^'\"]{2,}['\"]",                          # Quoted on-screen text
    r"\b[A-Z]{2,}\b",                                  # Acronyms / channel logos (HTV, NASA)
]
# Case-insensitive lexical triggers (numbers, titles, admin units)
_OCR_PATTERNS_LEXICAL = [
    r"\d{4}",                                # Years
    r"cấp\s*\d+",                            # Level numbers (bão cấp 16)
    r"(thủ tướng|chủ tịch|bộ trưởng|tổng thống|tổng bí thư)",  # Political titles
    r"(quận|huyện|tỉnh|thành phố)\s+[A-ZĐ]",  # Administrative unit + proper name
]


def _has_object_mention(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _OBJECT_KEYWORDS)


def _has_text_mention(query: str) -> bool:
    if any(re.search(p, query) for p in _OCR_PATTERNS_CASE_SENSITIVE):
        return True
    return any(re.search(p, query, re.IGNORECASE) for p in _OCR_PATTERNS_LEXICAL)


def _wrrf_fusion(
    signal_results: Dict[str, List[Result]],
    weights: Dict[str, float],
    k: int = _RRF_K,
    topk: int = 20,
) -> List[Result]:
    """
    Weighted Reciprocal Rank Fusion over multiple signal result lists.

    Args:
        signal_results: {"visual": [...], "object": [...], "ocr": [...]}
        weights: per-signal weight multiplier (0.0 = exclude)
        k: RRF constant
        topk: max results to return

    Returns:
        Fused result list sorted by wRRF score descending.
    """
    rrf_scores: Dict[Tuple[str, int], float] = {}
    metadata: Dict[Tuple[str, int], float] = {}  # pts_time

    for signal, results in signal_results.items():
        w = weights.get(signal, 0.0)
        if w <= 0 or not results:
            continue
        for rank, (vid, fidx, pts, _sc) in enumerate(results):
            key = (vid, fidx)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + w / (rank + k)
            if key not in metadata:
                metadata[key] = pts

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:topk]
    return [
        (vid, fidx, metadata[(vid, fidx)], score)
        for (vid, fidx), score in ranked
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class HybridSearcher:
    """
    Combines visual, object, and OCR signals with adaptive late fusion.

    Designed as a thin wrapper: callers inject their own search functions,
    so this module has no hard dependency on specific pipeline internals.
    """

    def __init__(
        self,
        visual_fn: Callable[[str, int], List[Result]],
        object_fn: Optional[Callable[[str, int], List[Result]]] = None,
        ocr_fn: Optional[Callable[[str, int], List[Result]]] = None,
    ):
        """
        Args:
            visual_fn: visual search function (query, topk) → List[Result]
            object_fn: object search function (query, topk) → List[Result] (optional)
            ocr_fn: OCR/BM25 search function (query, topk) → List[Result] (optional)
        """
        self._visual = visual_fn
        self._object = object_fn
        self._ocr = ocr_fn

    def search(
        self,
        query: str,
        topk: int = 20,
        object_boost: float = 1.2,
        ocr_boost: float = 1.0,
    ) -> dict:
        """
        Adaptive hybrid search.

        Query analysis determines which signals to activate:
        - Object mentions → activate object channel (boosted weight)
        - Named entities / numbers → activate OCR channel
        - Default → visual only (safest for generic visual queries)

        Args:
            query: Vietnamese/English query text
            topk: Max results
            object_boost: Weight multiplier for object channel when active
            ocr_boost: Weight multiplier for OCR channel when active

        Returns:
            {
                "results": List[Result],
                "signals_used": List[str],
                "weights": Dict[str, float],
            }
        """
        use_object = self._object is not None and _has_object_mention(query)
        use_ocr = self._ocr is not None and _has_text_mention(query)

        signals_used = ["visual"]
        weights: Dict[str, float] = {"visual": 1.0}

        # Fetch results per active signal
        fetch_topk = topk * 3  # fetch more, fuse down
        signal_results: Dict[str, List[Result]] = {
            "visual": self._visual(query, fetch_topk),
        }

        if use_object:
            signals_used.append("object")
            weights["object"] = object_boost
            try:
                signal_results["object"] = self._object(query, fetch_topk)
            except Exception:
                signal_results["object"] = []

        if use_ocr:
            signals_used.append("ocr")
            weights["ocr"] = ocr_boost
            try:
                signal_results["ocr"] = self._ocr(query, fetch_topk)
            except Exception:
                signal_results["ocr"] = []

        # If only visual active (no additional signals fired), skip fusion overhead
        if len(signals_used) == 1:
            return {
                "results": signal_results["visual"][:topk],
                "signals_used": signals_used,
                "weights": weights,
            }

        fused = _wrrf_fusion(signal_results, weights, topk=topk)
        return {
            "results": fused,
            "signals_used": signals_used,
            "weights": weights,
        }


# ---------------------------------------------------------------------------
# Standalone helpers (used directly without HybridSearcher class)
# ---------------------------------------------------------------------------
def detect_signals(query: str) -> Dict[str, bool]:
    """
    Analyse query and return which signals should be activated.

    Useful for UI display (show which channels were used).

    Returns:
        {"visual": True, "object": bool, "ocr": bool}
    """
    return {
        "visual": True,
        "object": _has_object_mention(query),
        "ocr": _has_text_mention(query),
    }
